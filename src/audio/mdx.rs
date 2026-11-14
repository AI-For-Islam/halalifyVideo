//! MDX-Net: a spectrogram-in, spectrogram-out U-Net (the model family behind
//! Ultimate Vocal Remover). Given a mix it returns the voice.
//!
//! The network sees a fixed window of `dim_t` STFT frames with the lowest
//! `dim_f` frequency bins, as a `[batch, 4, dim_f, dim_t]` tensor whose four
//! channels are left-real, left-imaginary, right-real, right-imaginary.

use anyhow::{Context, Result, ensure};
use ort::{session::Session, value::Tensor};

use super::{
    ola::Stereo,
    spectral::{self, Layout},
    stft::{Padding, Stft},
};
use crate::{models::Model, runtime::Runtime};

/// Parameters that belong to a particular MDX-Net checkpoint.
#[derive(Clone, Copy, Debug)]
pub struct MdxSpec {
    pub n_fft: usize,
    pub hop: usize,
    pub dim_f: usize,
    pub dim_t: usize,
    /// Gain that undoes the model's slight level loss.
    pub compensate: f32,
}

/// `Kim_Vocal_2.onnx`, per UVR's model data table.
pub const KIM_VOCAL_2: MdxSpec = MdxSpec { n_fft: 7680, hop: 1024, dim_f: 3072, dim_t: 256, compensate: 1.009 };

pub struct MdxNet {
    session: Session,
    input_name: String,
    spec: MdxSpec,
    stft: Stft,
}

impl MdxNet {
    pub fn load(rt: &Runtime, model: &Model, spec: MdxSpec) -> Result<MdxNet> {
        let session = rt.session(&model.fetch()?)?;
        let input_name = session.inputs().first().context("model has no input")?.name().to_string();
        let stft = Stft::new(spec.n_fft, spec.hop, Padding::Reflect);
        ensure!(spec.dim_f <= stft.bins(), "dim_f exceeds the number of STFT bins");
        Ok(MdxNet { session, input_name, spec, stft })
    }

    /// Samples per chunk: exactly `dim_t` centred STFT frames.
    pub fn chunk_len(&self) -> usize {
        self.spec.hop * (self.spec.dim_t - 1)
    }

    /// Samples at each chunk edge that the model reconstructs poorly.
    pub fn edge(&self) -> usize {
        self.spec.n_fft / 2
    }

    /// Extracts the voice from each chunk.
    pub fn separate(&mut self, chunks: &[Stereo]) -> Result<Vec<Stereo>> {
        let MdxSpec { dim_f, dim_t, compensate, .. } = self.spec;
        let chunk = self.chunk_len();
        ensure!(chunks.iter().all(|c| c[0].len() == chunk && c[1].len() == chunk), "chunk has the wrong length");
        // The model was trained with the three lowest bins zeroed.
        let layout = Layout { bins: dim_f, frames: dim_t, first_bin: 3 };

        // Left and right of a chunk are adjacent, so the planes of one chunk
        // are exactly the model's four channels.
        let signals: Vec<&[f32]> = chunks.iter().flat_map(|c| [c[0].as_slice(), c[1].as_slice()]).collect();
        let input = spectral::to_planes(&self.stft, &signals, &layout);
        let shape = [chunks.len() as i64, 4, dim_f as i64, dim_t as i64];
        let tensor = Tensor::from_array((shape, input))?;
        let outputs = self.session.run(ort::inputs![self.input_name.as_str() => tensor])?;
        let (_, out) = outputs[0].try_extract_tensor::<f32>()?;
        ensure!(out.len() == signals.len() * 2 * layout.plane(), "unexpected model output size");

        let mut waves = spectral::from_planes(&self.stft, out, signals.len(), &layout, chunk).into_iter();
        Ok((0..chunks.len())
            .map(|_| {
                let mut pair = [waves.next().unwrap(), waves.next().unwrap()];
                pair.iter_mut().flatten().for_each(|x| *x *= compensate);
                pair
            })
            .collect())
    }
}
