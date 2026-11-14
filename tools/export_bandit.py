"""Converts BandIt Plus to the ONNX file HalalifyVid's dialogue separator loads.

BandIt Plus splits a soundtrack into speech, music and effects. Its published
checkpoint is a PyTorch file, and its reference code keeps the STFT, complex
tensors and in-place writes inside the model, none of which export cleanly.
This script rebuilds the network with real arithmetic only, loads the
published weights into it unchanged, and writes one ONNX graph:

    spec  [batch, 2, 1025, 517]  real and imaginary parts of the plain STFT
                                 (n_fft 2048, hop 512, periodic Hann, centred
                                 with zero padding) of one audio channel
    gains [3]                    how much of speech, music and effects to keep
    out   [batch, 2, 1025, 517]  the STFT of the requested mixture of stems

The STFT and its inverse stay outside the graph, in Rust. A masked STFT is
linear in the masks, so one output serves every combination of stems.

    python tools/export_bandit.py <checkpoint.chpt> <out.onnx> [--check <reference.npz>] [--full-precision]

By default the graph is stored in half precision except for its GRU layers:
ONNX Runtime 1.21's half-precision GRU on CUDA gives wrong results for this
model (15 dB agreement with PyTorch instead of 60 dB), while the rest runs
correctly and twice as fast in half precision. That step needs the
onnxconverter-common package.

The checkpoint is model_bandit_plus_dnr_sdr_11.47.chpt from
https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v.1.0.3
Needs: torch, numpy (and onnx or onnxscript if your torch asks for them).
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

SAMPLE_RATE = 44_100
N_FFT = 2048
HOP = 512
BINS = N_FFT // 2 + 1
N_BANDS = 64
EMB = 128
RNN = 256
MLP = 512
BLOCKS = 16  # eight passes along time alternating with eight along the bands
STEMS = ("speech", "music", "effects")
# 516 hops = 264,192 samples: the 6 s the model was trained on, rounded down
# to a whole number of hops so a chunk is exactly FRAMES centred frames.
FRAMES = 517
# torchaudio's "normalized" spectrogram divides by the window's energy.
WINDOW_NORM = math.sqrt(3 * N_FFT / 8)


def musical_bands(n_bands: int = N_BANDS, fs: int = SAMPLE_RATE, n_fft: int = N_FFT):
    """The model's overlapping bands: [start, end) bins, evenly spaced in pitch.

    Mirrors `musical_filterbank` in the reference code, with librosa's two
    pitch conversions written out.
    """
    n_freqs = n_fft // 2 + 1
    df = fs / n_fft
    f_max, f_min = fs / 2, fs / n_fft
    bandwidth_mult = 2.0 ** (np.log2(f_max / f_min) / n_bands)
    hz_to_midi = lambda hz: 12.0 * (np.log2(hz) - np.log2(440.0)) + 69.0
    midi_to_hz = lambda midi: 440.0 * 2.0 ** ((np.asarray(midi) - 69.0) / 12.0)
    centres = midi_to_hz(np.linspace(max(0.0, hz_to_midi(f_min)), hz_to_midi(f_max), n_bands))
    low = np.floor(centres / bandwidth_mult / df).astype(int)
    high = np.ceil(centres * bandwidth_mult / df).astype(int)
    bank = np.zeros((n_bands, n_freqs))
    for i in range(n_bands):
        bank[i, low[i] : high[i] + 1] = 1.0
    bank[0, : low[0]] = 1.0
    bank[-1, high[-1] + 1 :] = 1.0
    bands = []
    for row in bank:
        active = np.nonzero(row)[0]
        bands.append((int(active[0]), int(active[-1]) + 1))
    # Where bands overlap, each contributes an equal share of the bin.
    weights = bank / bank.sum(axis=0, keepdims=True)
    return bands, [weights[i, s:e] for i, (s, e) in enumerate(bands)]


class NormFC(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(2 * width)
        self.fc = nn.Linear(2 * width, EMB)


class ResidualRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(EMB)
        self.rnn = nn.GRU(EMB, RNN, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * RNN, EMB)


class NormMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(EMB)
        # Sequential only so that the parameter names match the checkpoint;
        # the activations are applied in `Bandit.forward`.
        self.hidden = nn.Sequential(nn.Linear(EMB, MLP))
        self.output = nn.Sequential(nn.Linear(MLP, 4 * width))


def module_list(items):
    return nn.ModuleList(list(items))


class Bandit(nn.Module):
    """BandIt Plus with real arithmetic. Module names mirror the checkpoint."""

    def __init__(self, bands, weights):
        super().__init__()
        self.bands = bands
        widths = [e - s for s, e in bands]
        bsrnn = nn.Module()
        bsrnn.band_split = nn.Module()
        bsrnn.band_split.norm_fc_modules = module_list(NormFC(w) for w in widths)
        bsrnn.tf_model = nn.Module()
        bsrnn.tf_model.seqband = module_list(ResidualRNN() for _ in range(BLOCKS))
        bsrnn.mask_estim = nn.ModuleDict()
        for stem in STEMS:
            head = nn.Module()
            head.norm_mlp = module_list(NormMLP(w) for w in widths)
            bsrnn.mask_estim[stem] = head
        self.bsrnn = bsrnn
        # One matrix places every band's mask on its bins with its share,
        # replacing 64 in-place slice additions.
        place = torch.zeros(sum(widths), BINS)
        row = 0
        for (start, _), w in zip(bands, weights):
            for k, share in enumerate(w):
                place[row, start + k] = float(share)
                row += 1
        self.register_buffer("place", place, persistent=False)

    def forward(self, spec: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        # spec: [B, 2, BINS, FRAMES]
        x = (spec * (1.0 / WINDOW_NORM)).permute(0, 3, 1, 2)  # [B, T, 2, F]
        z = []
        for (start, end), m in zip(self.bands, self.bsrnn.band_split.norm_fc_modules):
            band = x[:, :, :, start:end].reshape(-1, FRAMES, 2 * (end - start))
            z.append(m.fc(m.norm(band)))
        z = torch.stack(z, dim=1)  # [B, bands, T, EMB]

        for k, block in enumerate(self.bsrnn.tf_model.seqband):
            # Even blocks run along time for every band, odd ones along the
            # bands for every frame; the transpose swaps the roles each time.
            kept, along = (N_BANDS, FRAMES) if k % 2 == 0 else (FRAMES, N_BANDS)
            y = block.norm(z).reshape(-1, along, EMB)
            y = block.rnn(y)[0].reshape(-1, kept, along, 2 * RNN)
            z = (block.fc(y) + z).transpose(1, 2)

        mixed = None
        for s, stem in enumerate(STEMS):
            parts = []
            for b, m in enumerate(self.bsrnn.mask_estim[stem].norm_mlp):
                hidden = torch.tanh(m.hidden[0](m.norm(z[:, b])))
                gated = m.output[0](hidden)
                half = gated.shape[-1] // 2
                parts.append(gated[..., :half] * torch.sigmoid(gated[..., half:]))  # GLU
            stem_masks = torch.cat(parts, dim=-1) * gains[s]  # [B, T, 2 * total width]
            mixed = stem_masks if mixed is None else mixed + stem_masks
        mixed = mixed.reshape(-1, FRAMES, self.place.shape[0], 2)
        mask_re = torch.matmul(mixed[..., 0], self.place).transpose(1, 2)  # [B, F, T]
        mask_im = torch.matmul(mixed[..., 1], self.place).transpose(1, 2)

        re, im = spec[:, 0], spec[:, 1]
        return torch.stack([re * mask_re - im * mask_im, re * mask_im + im * mask_re], dim=1)


def load(checkpoint: Path) -> Bandit:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    bands, weights = musical_bands()

    # The checkpoint itself says how wide each band is and how overlapping
    # bands share a bin; the bands computed here must agree with it exactly.
    for i, ((start, end), w) in enumerate(zip(bands, weights)):
        stored = state[f"bsrnn.mask_estim.speech.freq_weights/{i}"].double().numpy()
        assert stored.shape == (end - start,), f"band {i}: width {end - start} but checkpoint has {stored.shape}"
        assert np.allclose(stored, w, atol=1e-9), f"band {i}: bin shares differ from the checkpoint"

    model = Bandit(bands, weights)
    wanted = {k: v for k, v in state.items() if "freq_weights/" not in k and not k.endswith(".window")}
    missing, unexpected = model.load_state_dict(wanted, strict=False)
    assert not missing and not unexpected, f"missing {missing}, unexpected {unexpected}"
    return model.eval()


def stft(wave: torch.Tensor) -> torch.Tensor:
    """[channels, samples] -> [channels, 2, BINS, frames], as the Rust side computes it."""
    spec = torch.stft(wave, N_FFT, HOP, window=torch.hann_window(N_FFT), center=True, pad_mode="constant", return_complex=True)
    return torch.view_as_real(spec).permute(0, 3, 1, 2).contiguous()


def istft(spec: torch.Tensor, length: int) -> torch.Tensor:
    complex_spec = torch.view_as_complex(spec.permute(0, 2, 3, 1).contiguous())
    return torch.istft(complex_spec, N_FFT, HOP, window=torch.hann_window(N_FFT), center=True, length=length)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--check", type=Path, help="npz with 'wave' [2, samples] and 'stems' [3, 2, samples] from the reference model")
    ap.add_argument("--full-precision", action="store_true", help="keep every weight in fp32 (twice the size, slower)")
    args = ap.parse_args()

    model = load(args.checkpoint)
    print(f"loaded {sum(p.numel() for p in model.parameters()):,} parameters into {N_BANDS} bands")

    if args.check:
        ref = np.load(args.check)
        wave = torch.from_numpy(ref["wave"]).float()
        spec = stft(wave)
        assert spec.shape[-1] == FRAMES, f"reference clip must give {FRAMES} frames, got {spec.shape[-1]}"
        with torch.no_grad():
            for s, stem in enumerate(STEMS):
                gains = torch.zeros(3)
                gains[s] = 1.0
                mine = istft(model(spec, gains), wave.shape[-1]).numpy()
                theirs = ref["stems"][s]
                error = np.abs(mine - theirs).max()
                snr = 10 * np.log10((theirs**2).sum() / ((mine - theirs) ** 2).sum())
                print(f"{stem:8s} max difference {error:.2e}, agreement {snr:.1f} dB")
                assert snr > 80, f"{stem}: this rebuild does not reproduce the reference model"

    example = (torch.zeros(2, 2, BINS, FRAMES), torch.tensor([1.0, 0.0, 0.0]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(args.output),
        input_names=["spec", "gains"],
        output_names=["out"],
        dynamic_axes={"spec": {0: "batch"}, "out": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    if not args.full_precision:
        import warnings

        import onnx
        from onnxconverter_common import float16

        warnings.filterwarnings("ignore")
        graph = float16.convert_float_to_float16(
            onnx.load(str(args.output)), keep_io_types=True, op_block_list=float16.DEFAULT_OP_BLOCK_LIST + ["GRU"]
        )
        onnx.save(graph, str(args.output))
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
