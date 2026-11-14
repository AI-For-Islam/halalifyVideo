//! Short-time Fourier transform matching `torch.stft` / `torch.istft` with
//! `center=True` and a periodic Hann window, which is what the separation
//! models were trained with.

use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex, num_complex::Complex};

pub type Bin = Complex<f32>;

/// What a centred transform assumes beyond the ends of the signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Padding {
    /// The signal mirrored about its end points (`pad_mode="reflect"`).
    Reflect,
    /// Silence (`pad_mode="constant"`).
    Zero,
}

pub struct Stft {
    pub n_fft: usize,
    pub hop: usize,
    padding: Padding,
    window: Vec<f32>,
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
}

/// Per-thread work buffers, so transforms allocate nothing in the hot loop.
pub struct Scratch {
    padded: Vec<f32>,
    frame: Vec<f32>,
    spectrum: Vec<Bin>,
    fft_scratch: Vec<Bin>,
    envelope: Vec<f32>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize, padding: Padding) -> Stft {
        let mut planner = RealFftPlanner::<f32>::new();
        // Periodic Hann, as torch.hann_window(n_fft) returns by default.
        let window = (0..n_fft)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos())
            .collect();
        Stft { n_fft, hop, padding, window, forward: planner.plan_fft_forward(n_fft), inverse: planner.plan_fft_inverse(n_fft) }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Number of frames a centred transform of `len` samples has.
    pub fn frames(&self, len: usize) -> usize {
        len / self.hop + 1
    }

    pub fn scratch(&self) -> Scratch {
        Scratch {
            padded: Vec::new(),
            frame: vec![0.0; self.n_fft],
            spectrum: vec![Bin::default(); self.bins()],
            fft_scratch: vec![Bin::default(); self.forward.get_scratch_len().max(self.inverse.get_scratch_len())],
            envelope: Vec::new(),
        }
    }

    /// Transforms `signal`, handing each frame's spectrum to `sink(frame, bins)`.
    pub fn analyze(&self, signal: &[f32], s: &mut Scratch, mut sink: impl FnMut(usize, &[Bin])) {
        let pad = self.n_fft / 2;
        let n = signal.len();
        s.padded.clear();
        match self.padding {
            Padding::Reflect => {
                assert!(n > pad, "signal shorter than the reflect padding");
                s.padded.extend((1..=pad).rev().map(|i| signal[i]));
                s.padded.extend_from_slice(signal);
                s.padded.extend((1..=pad).map(|i| signal[n - 1 - i]));
            }
            Padding::Zero => {
                s.padded.resize(pad, 0.0);
                s.padded.extend_from_slice(signal);
                s.padded.resize(n + 2 * pad, 0.0);
            }
        }

        for t in 0..self.frames(n) {
            let start = t * self.hop;
            for ((dst, src), w) in s.frame.iter_mut().zip(&s.padded[start..start + self.n_fft]).zip(&self.window) {
                *dst = src * w;
            }
            self.forward
                .process_with_scratch(&mut s.frame, &mut s.spectrum, &mut s.fft_scratch)
                .expect("buffer sizes match the plan");
            sink(t, &s.spectrum);
        }
    }

    /// Rebuilds a signal of `out.len()` samples from spectra supplied by
    /// `source(frame, bins)`, which must fill every bin.
    pub fn synthesize(&self, out: &mut [f32], s: &mut Scratch, mut source: impl FnMut(usize, &mut [Bin])) {
        let pad = self.n_fft / 2;
        let n = out.len();
        let frames = self.frames(n);
        let total = n + 2 * pad;
        s.padded.clear();
        s.padded.resize(total, 0.0);
        if s.envelope.len() != total {
            // Sum of squared windows at every sample; depends only on the length.
            s.envelope.clear();
            s.envelope.resize(total, 0.0);
            for t in 0..frames {
                for (e, w) in s.envelope[t * self.hop..t * self.hop + self.n_fft].iter_mut().zip(&self.window) {
                    *e += w * w;
                }
            }
        }
        let scale = 1.0 / self.n_fft as f32;
        let last = self.bins() - 1;
        for t in 0..frames {
            source(t, &mut s.spectrum);
            // A real signal has real DC and Nyquist bins; the inverse transform insists on it.
            s.spectrum[0].im = 0.0;
            s.spectrum[last].im = 0.0;
            self.inverse
                .process_with_scratch(&mut s.spectrum, &mut s.frame, &mut s.fft_scratch)
                .expect("buffer sizes match the plan");
            for ((acc, x), w) in s.padded[t * self.hop..t * self.hop + self.n_fft].iter_mut().zip(&s.frame).zip(&self.window) {
                *acc += x * w * scale;
            }
        }
        for ((o, x), e) in out.iter_mut().zip(&s.padded[pad..pad + n]).zip(&s.envelope[pad..pad + n]) {
            *o = if *e > 1e-10 { x / e } else { 0.0 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / 44_100.0;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin() + 0.3 * (2.0 * std::f32::consts::PI * 3_000.0 * t).cos()
            })
            .collect()
    }

    #[test]
    fn analysis_then_synthesis_is_identity() {
        for padding in [Padding::Reflect, Padding::Zero] {
            for (n_fft, hop, len) in [(2048, 512, 512 * 40), (7680, 1024, 1024 * 31), (1024, 441, 441 * 30)] {
                let stft = Stft::new(n_fft, hop, padding);
                let signal = test_signal(len);
                let mut s = stft.scratch();
                let mut spectra: Vec<Vec<Bin>> = Vec::new();
                stft.analyze(&signal, &mut s, |_, bins| spectra.push(bins.to_vec()));
                assert_eq!(spectra.len(), len / hop + 1);
                let mut out = vec![0.0; len];
                stft.synthesize(&mut out, &mut s, |t, bins| bins.copy_from_slice(&spectra[t]));
                let err = signal.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                assert!(err < 1e-4, "{padding:?} n_fft {n_fft}: max error {err}");
            }
        }
    }

    #[test]
    fn a_pure_tone_lands_in_its_bin() {
        // 1 kHz at 44.1 kHz with n_fft 4410 is exactly bin 100; a Hann window
        // gives that bin a magnitude of amplitude * n_fft / 4.
        let stft = Stft::new(4410, 441, Padding::Reflect);
        let signal: Vec<f32> = (0..44_100).map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44_100.0).sin()).collect();
        let mut s = stft.scratch();
        let mut mid = Vec::new();
        stft.analyze(&signal, &mut s, |t, bins| {
            if t == 50 {
                mid = bins.iter().map(|b| b.norm()).collect();
            }
        });
        let peak = mid.iter().cloned().fold(0.0, f32::max);
        assert_eq!(mid.iter().position(|&m| m == peak), Some(100));
        assert!((peak - 4410.0 / 4.0).abs() < 1.0, "peak {peak}");
    }

    #[test]
    fn padding_modes_differ_only_near_the_ends() {
        let signal = test_signal(512 * 40);
        let frames_of = |padding| {
            let stft = Stft::new(2048, 512, padding);
            let mut s = stft.scratch();
            let mut frames: Vec<Vec<Bin>> = Vec::new();
            stft.analyze(&signal, &mut s, |_, bins| frames.push(bins.to_vec()));
            frames
        };
        let (reflect, zero) = (frames_of(Padding::Reflect), frames_of(Padding::Zero));
        let differs = |t: usize| reflect[t].iter().zip(&zero[t]).any(|(a, b)| (a - b).norm() > 1e-3);
        // A frame reaches 1024 samples to each side, i.e. two hops into the padding.
        assert!(differs(0) && differs(1) && differs(40));
        assert!(!differs(2) && !differs(20) && !differs(38));
    }
}
