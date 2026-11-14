//! Telling who a detected person is, zero-shot.
//!
//! A SigLIP 2 image embedding of the person is compared with two fixed text
//! embeddings ("a woman ..." and "a man ..." descriptions, built by
//! `tools/embed_prompts.py`). The difference of the two similarities, on the
//! model's own logit scale, is the evidence that the person is a woman. No
//! model was trained for this; the image encoder is used as published.

use anyhow::{Context, Result, ensure};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::ImageRef};
use ort::{session::Session, value::Tensor};
use rayon::prelude::*;
use serde::Deserialize;

use crate::{models, runtime::Runtime};

/// Side of the square image the encoder takes.
const SIDE: usize = 256;
/// SigLIP multiplies cosine similarity by about this much to get logits.
const LOGIT_SCALE: f32 = 100.0;
const MAX_BATCH: usize = 32;

const PROMPTS_JSON: &str = include_str!("../../assets/person_prompts.json");

#[derive(Deserialize)]
struct Prompts {
    dim: usize,
    groups: Groups,
}

#[derive(Deserialize)]
struct Groups {
    female: Group,
    male: Group,
}

#[derive(Deserialize)]
struct Group {
    embedding: Vec<f32>,
}

/// A person to classify: the analysis frame they are in and their box.
pub struct Crop<'a> {
    pub frame: &'a [u8],
    /// `[x1, y1, x2, y2]` as fractions of the frame.
    pub bbox: [f32; 4],
}

pub struct PersonClassifier {
    session: Session,
    input_name: String,
    /// `female - male` text direction; one dot product gives the evidence.
    direction: Vec<f32>,
}

impl PersonClassifier {
    pub fn load(rt: &Runtime) -> Result<PersonClassifier> {
        let prompts: Prompts = serde_json::from_str(PROMPTS_JSON).context("assets/person_prompts.json is malformed")?;
        let (f, m) = (&prompts.groups.female.embedding, &prompts.groups.male.embedding);
        ensure!(f.len() == prompts.dim && m.len() == prompts.dim, "prompt embeddings have the wrong size");
        let direction = f.iter().zip(m).map(|(a, b)| a - b).collect();
        let session = rt.session(&models::SIGLIP2_VISION.fetch()?)?;
        let input_name = session.inputs().first().context("classifier has no input")?.name().to_string();
        Ok(PersonClassifier { session, input_name, direction })
    }

    /// Evidence, in log-odds, that each crop shows a woman (positive) rather
    /// than a man (negative). `frame_size` is the analysis frame size.
    pub fn female_logits(&mut self, crops: &[Crop], frame_size: (usize, usize)) -> Result<Vec<f32>> {
        let mut logits = Vec::with_capacity(crops.len());
        for batch in crops.chunks(MAX_BATCH) {
            let plane = SIDE * SIDE;
            let mut input = vec![0.0f32; batch.len() * 3 * plane];
            input.par_chunks_mut(3 * plane).zip(batch.par_iter()).for_each_init(
                || (Resizer::new(), vec![0u8; plane * 3]),
                |(resizer, rgb), (tensor, crop)| fill_tensor(tensor, crop, frame_size, resizer, rgb),
            );
            let shape = [batch.len() as i64, 3, SIDE as i64, SIDE as i64];
            let tensor = Tensor::from_array((shape, input))?;
            let outputs = self.session.run(ort::inputs![self.input_name.as_str() => tensor])?;
            let pooled = outputs.get("pooler_output").context("classifier has no pooler_output")?;
            let (_, embeddings) = pooled.try_extract_tensor::<f32>()?;
            let dim = self.direction.len();
            ensure!(embeddings.len() == batch.len() * dim, "unexpected embedding size");
            for e in embeddings.chunks_exact(dim) {
                let norm = e.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                let dot: f32 = e.iter().zip(&self.direction).map(|(a, b)| a * b).sum();
                logits.push(LOGIT_SCALE * dot / norm);
            }
        }
        Ok(logits)
    }
}

/// The part of the frame, in pixels `(left, top, width, height)`, shown to the
/// classifier for a person box.
///
/// For a standing person this is the head and upper body rather than the whole
/// figure: it is where sex is most visible (face, hair, head covering,
/// shoulders), and it fills the square input instead of squashing a tall thin
/// figure into it. Seated people and close-ups are used whole.
fn crop_rect(bbox: [f32; 4], (fw, fh): (usize, usize)) -> (f64, f64, f64, f64) {
    let (fw, fh) = (fw as f64, fh as f64);
    let (x1, y1, x2, y2) = (bbox[0] as f64 * fw, bbox[1] as f64 * fh, bbox[2] as f64 * fw, bbox[3] as f64 * fh);
    let margin = 0.06 * (x2 - x1);
    let left = (x1 - margin).max(0.0);
    let right = (x2 + margin).min(fw);
    let top = (y1 - margin).max(0.0);
    let width = (right - left).max(1.0);
    let bottom = (y2 + margin).min(fh).min(top + 1.3 * width);
    (left, top, width, (bottom - top).max(1.0))
}

/// Writes one crop into its `[3, SIDE, SIDE]` tensor slice, normalised to -1..1.
fn fill_tensor(tensor: &mut [f32], crop: &Crop, frame_size: (usize, usize), resizer: &mut Resizer, rgb: &mut [u8]) {
    let (left, top, width, height) = crop_rect(crop.bbox, frame_size);
    let src = ImageRef::new(frame_size.0 as u32, frame_size.1 as u32, crop.frame, PixelType::U8x3).expect("frame size matches");
    let mut dst = fast_image_resize::images::Image::from_slice_u8(SIDE as u32, SIDE as u32, rgb, PixelType::U8x3).expect("buffer size matches");
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)).crop(left, top, width, height);
    resizer.resize(&src, &mut dst, &options).expect("same pixel type");

    let plane = SIDE * SIDE;
    let (r, rest) = tensor.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    for (i, p) in rgb.chunks_exact(3).enumerate() {
        r[i] = p[0] as f32 * (1.0 / 127.5) - 1.0;
        g[i] = p[1] as f32 * (1.0 / 127.5) - 1.0;
        b[i] = p[2] as f32 * (1.0 / 127.5) - 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_prompt_embeddings_are_well_formed() {
        let p: Prompts = serde_json::from_str(PROMPTS_JSON).unwrap();
        for g in [&p.groups.female, &p.groups.male] {
            assert_eq!(g.embedding.len(), p.dim);
            let norm: f32 = g.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "embedding is not unit length: {norm}");
        }
        assert_ne!(p.groups.female.embedding, p.groups.male.embedding);
    }

    #[test]
    fn standing_person_is_cropped_to_head_and_torso() {
        // A 100 x 300 px figure in a 1000 x 500 frame.
        let (l, t, w, h) = crop_rect([0.4, 0.2, 0.5, 0.8], (1000, 500));
        assert!((l - 394.0).abs() < 1e-3 && (t - 94.0).abs() < 1e-3);
        assert!((w - 112.0).abs() < 1e-3);
        assert!((h - 1.3 * 112.0).abs() < 1e-3, "height {h}");
    }

    #[test]
    fn close_up_is_used_whole_and_stays_inside_the_frame() {
        let (l, t, w, h) = crop_rect([0.0, 0.0, 1.0, 1.0], (640, 360));
        assert_eq!((l, t, w, h), (0.0, 0.0, 640.0, 360.0));
    }

    #[test]
    fn crop_pixels_land_in_the_tensor() {
        // Left half red, right half blue; crop the left half.
        let (fw, fh) = (64, 32);
        let mut frame = vec![0u8; fw * fh * 3];
        for (i, p) in frame.chunks_exact_mut(3).enumerate() {
            if i % fw < fw / 2 { p[0] = 255 } else { p[2] = 255 }
        }
        let crop = Crop { frame: &frame, bbox: [0.0, 0.0, 0.4, 1.0] };
        let mut tensor = vec![0.0; 3 * SIDE * SIDE];
        fill_tensor(&mut tensor, &crop, (fw, fh), &mut Resizer::new(), &mut vec![0u8; SIDE * SIDE * 3]);
        let centre = SIDE * SIDE / 2 + SIDE / 2;
        assert!((tensor[centre] - 1.0).abs() < 1e-3, "red channel {}", tensor[centre]);
        assert!((tensor[2 * SIDE * SIDE + centre] + 1.0).abs() < 1e-3);
    }
}
