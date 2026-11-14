//! BandIt Plus: splits a soundtrack into speech, music and effects.
//!
//! Unlike a vocal model, which keeps any voice, BandIt was trained on film
//! and broadcast audio and knows speech from song: a song playing in the
//! background goes to "music" whole, voice included.
//!
//! The network is run from an ONNX file made by `tools/export_bandit.py`,
//! which takes the plain STFT of one audio channel and three gains, one per
//! stem, and returns the STFT of that mixture of stems. Channels are
//! independent, so a stereo chunk is two items of a batch.

use anyhow::{Context, Result, ensure};
use ort::{session::Session, value::Tensor};

use super::{
    ola::Stereo,
    spectral::{self, Layout},
    stft::{Padding, Stft},
};
use crate::{models, runtime::Runtime};

const N_FFT: usize = 2048;
const HOP: usize = 512;
/// Frames per chunk; fixed when the model was exported.
const FRAMES: usize = 517;

/// What to keep of the soundtrack, as gains for speech, music and effects.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keep(pub [f32; 3]);

impl Keep {
    pub const SPEECH: Keep = Keep([1.0, 0.0, 0.0]);
    pub const SPEECH_AND_EFFECTS: Keep = Keep([1.0, 0.0, 1.0]);
}

pub struct Bandit {
    session: Session,
    stft: Stft,
    keep: Keep,
}

impl Bandit {
    pub fn load(rt: &Runtime, keep: Keep) -> Result<Bandit> {
        let session = rt.session(&models::BANDIT_PLUS.fetch()?)?;
        let names: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        ensure!(names == ["spec", "gains"], "{} is not a HalalifyVid BandIt export (inputs {names:?})", models::BANDIT_PLUS.file);
        // torchaudio's Spectrogram pads with silence (pad_mode "constant").
        let stft = Stft::new(N_FFT, HOP, Padding::Zero);
        Ok(Bandit { session, stft, keep })
    }

    /// Samples per chunk: exactly `FRAMES` centred STFT frames, just under
    /// the six seconds the model was trained on.
    pub fn chunk_len(&self) -> usize {
        HOP * (FRAMES - 1)
    }

    /// Silence placed before the audio so it never starts at a chunk edge.
    pub fn edge(&self) -> usize {
        N_FFT
    }

    /// Returns the kept stems of each chunk.
    pub fn separate(&mut self, chunks: &[Stereo]) -> Result<Vec<Stereo>> {
        let chunk = self.chunk_len();
        ensure!(chunks.iter().all(|c| c[0].len() == chunk && c[1].len() == chunk), "chunk has the wrong length");
        let layout = Layout { bins: self.stft.bins(), frames: FRAMES, first_bin: 0 };

        // Much speech is recorded in mono and carried as two identical
        // channels; those are run once and the result used for both.
        let mut signals: Vec<&[f32]> = Vec::with_capacity(chunks.len() * 2);
        let mut slots: Vec<[usize; 2]> = Vec::with_capacity(chunks.len());
        for c in chunks {
            let left = signals.len();
            signals.push(&c[0]);
            let right = if c[0] == c[1] {
                left
            } else {
                signals.push(&c[1]);
                left + 1
            };
            slots.push([left, right]);
        }

        let input = spectral::to_planes(&self.stft, &signals, &layout);
        let shape = [signals.len() as i64, 2, layout.bins as i64, FRAMES as i64];
        let spec = Tensor::from_array((shape, input))?;
        let gains = Tensor::from_array(([3i64], self.keep.0.to_vec()))?;
        let outputs = self.session.run(ort::inputs!["spec" => spec, "gains" => gains])?;
        let (_, out) = outputs.get("out").context("BandIt export has no `out`")?.try_extract_tensor::<f32>()?;
        ensure!(out.len() == signals.len() * 2 * layout.plane(), "unexpected model output size");

        let waves = spectral::from_planes(&self.stft, out, signals.len(), &layout, chunk);
        Ok(slots.iter().map(|&[l, r]| [waves[l].clone(), waves[r].clone()]).collect())
    }
}
