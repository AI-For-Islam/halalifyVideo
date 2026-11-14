//! MDX-Net: a spectrogram-in, spectrogram-out U-Net (the model family behind
//! Ultimate Vocal Remover). Given a mix it returns the voice.
//!
//! The network sees a fixed window of `dim_t` STFT frames with the lowest
//! `dim_f` frequency bins, as a `[batch, 4, dim_f, dim_t]` tensor whose four
//! channels are left-real, left-imaginary, right-real, right-imaginary.

use anyhow::{Context, Result, ensure};
use ort::{session::Session, value::Tensor};
use rayon::prelude::*;

use super::{
    ola::Stereo,
    stft::{Bin, Stft},
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
        let stft = Stft::new(spec.n_fft, spec.hop);
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
        let plane = dim_f * dim_t; // one real or imaginary plane
        let batch = chunks.len();
        ensure!(chunks.iter().all(|c| c[0].len() == chunk && c[1].len() == chunk), "chunk has the wrong length");

        // Forward STFT: each (chunk, channel) pair owns two adjacent planes.
        let mut input = vec![0.0f32; batch * 4 * plane];
        let stft = &self.stft;
        input.par_chunks_mut(2 * plane).enumerate().for_each_init(
            || stft.scratch(),
            |scratch, (i, planes)| {
                let (re, im) = planes.split_at_mut(plane);
                stft.analyze(&chunks[i / 2][i % 2], scratch, |t, bins| {
                    // The model was trained with the three lowest bins zeroed.
                    for f in 3..dim_f {
                        re[f * dim_t + t] = bins[f].re;
                        im[f * dim_t + t] = bins[f].im;
                    }
                });
            },
        );

        let shape = [batch as i64, 4, dim_f as i64, dim_t as i64];
        let tensor = Tensor::from_array((shape, input))?;
        let outputs = self.session.run(ort::inputs![self.input_name.as_str() => tensor])?;
        let (out_shape, out) = outputs[0].try_extract_tensor::<f32>()?;
        ensure!(out.len() == batch * 4 * plane, "unexpected model output shape {out_shape:?}");

        // Inverse STFT; bins above dim_f were never modelled and stay silent.
        let mut planes_out: Vec<Vec<f32>> = (0..batch * 2).map(|_| vec![0.0f32; chunk]).collect();
        planes_out.par_iter_mut().enumerate().for_each_init(
            || stft.scratch(),
            |scratch, (i, signal)| {
                let (re, im) = out[i * 2 * plane..(i + 1) * 2 * plane].split_at(plane);
                stft.synthesize(signal, scratch, |t, bins| {
                    for (f, bin) in bins.iter_mut().enumerate() {
                        *bin = if f < dim_f { Bin::new(re[f * dim_t + t], im[f * dim_t + t]) } else { Bin::default() };
                    }
                });
                signal.iter_mut().for_each(|x| *x *= compensate);
            },
        );

        let mut it = planes_out.into_iter();
        Ok((0..batch).map(|_| [it.next().unwrap(), it.next().unwrap()]).collect())
    }
}
