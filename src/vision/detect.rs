//! Person detection with outlines: YOLO26-seg in ONNX Runtime.
//!
//! YOLO26 is end-to-end (no non-maximum suppression): for each image it
//! returns up to 300 rows of `[x1, y1, x2, y2, score, class, 32 mask
//! coefficients]` in input pixels, plus 32 prototype masks at a quarter of the
//! input resolution. A person's outline is the sign of the coefficient-weighted
//! sum of the prototypes inside the person's box.

use anyhow::{Context, Result, ensure};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::ImageRef};
use ort::{session::Session, value::Tensor};
use rayon::prelude::*;

use super::{frames::Geometry, mask::MaskPatch};
use crate::{models, runtime::Runtime};

/// COCO class index of "person".
const PERSON: f32 = 0.0;
/// Grey that pads the image up to a multiple of 32, as in training.
const PAD: f32 = 114.0 / 255.0;
const COEFS: usize = 32;

#[derive(Clone, Debug)]
pub struct Detection {
    /// `[x1, y1, x2, y2]` as fractions of the frame.
    pub bbox: [f32; 4],
    pub score: f32,
    pub mask: MaskPatch,
}

pub struct Detector {
    session: Session,
    input_name: String,
    geometry: Geometry,
}

impl Detector {
    pub fn load(rt: &Runtime, geometry: Geometry) -> Result<Detector> {
        let session = rt.session(&models::YOLO26S_SEG.fetch()?)?;
        let input_name = session.inputs().first().context("detector has no input")?.name().to_string();
        ensure!(session.outputs().len() >= 2, "detector must output boxes and mask prototypes");
        Ok(Detector { session, input_name, geometry })
    }

    /// Detects people in a batch of analysis frames (packed RGB), keeping
    /// those scoring at least `min_score`.
    pub fn detect(&mut self, frames: &[&[u8]], min_score: f32) -> Result<Vec<Vec<Detection>>> {
        let g = self.geometry;
        let (in_w, in_h) = g.input;
        let plane = in_w * in_h;
        let mut input = vec![PAD; frames.len() * 3 * plane];
        input.par_chunks_mut(3 * plane).zip(frames.par_iter()).for_each_init(
            || (Resizer::new(), vec![0u8; g.content.0 * g.content.1 * 3]),
            |(resizer, small), (tensor, frame)| fill_tensor(tensor, frame, &g, resizer, small),
        );

        let shape = [frames.len() as i64, 3, in_h as i64, in_w as i64];
        let tensor = Tensor::from_array((shape, input))?;
        let outputs = self.session.run(ort::inputs![self.input_name.as_str() => tensor])?;
        let (rows_shape, rows) = outputs[0].try_extract_tensor::<f32>()?;
        let (protos_shape, protos) = outputs[1].try_extract_tensor::<f32>()?;
        ensure!(rows_shape.len() == 3 && rows_shape[2] as usize == 6 + COEFS, "unexpected detector output {rows_shape:?}");
        let per_image = rows_shape[1] as usize;
        let (pw, ph) = (in_w / 4, in_h / 4);
        ensure!(
            protos_shape.len() == 4 && protos_shape[1] as usize == COEFS && protos_shape[2] as usize == ph && protos_shape[3] as usize == pw,
            "unexpected prototype shape {protos_shape:?}"
        );

        let row_len = 6 + COEFS;
        let proto_len = COEFS * ph * pw;
        Ok((0..frames.len())
            .into_par_iter()
            .map(|i| {
                let rows = &rows[i * per_image * row_len..(i + 1) * per_image * row_len];
                let protos = &protos[i * proto_len..(i + 1) * proto_len];
                decode(rows, protos, pw, &g, min_score)
            })
            .collect())
    }
}

/// Writes one frame into its `[3, H, W]` slice of the batch tensor: scaled to
/// the content size, placed top-left, 0..1 range, planar RGB.
fn fill_tensor(tensor: &mut [f32], frame: &[u8], g: &Geometry, resizer: &mut Resizer, small: &mut Vec<u8>) {
    let (cw, ch) = g.content;
    let (in_w, in_h) = g.input;
    let pixels: &[u8] = if g.frame == g.content {
        frame
    } else {
        let src = ImageRef::new(g.frame.0 as u32, g.frame.1 as u32, frame, PixelType::U8x3).expect("frame size matches geometry");
        let mut dst = fast_image_resize::images::Image::from_slice_u8(cw as u32, ch as u32, small, PixelType::U8x3)
            .expect("buffer size matches geometry");
        let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
        resizer.resize(&src, &mut dst, &options).expect("same pixel type");
        small
    };
    let plane = in_w * in_h;
    let (r, rest) = tensor.split_at_mut(plane);
    let (gch, b) = rest.split_at_mut(plane);
    for y in 0..ch {
        let line = &pixels[y * cw * 3..(y + 1) * cw * 3];
        let o = y * in_w;
        for (x, p) in line.chunks_exact(3).enumerate() {
            r[o + x] = p[0] as f32 * (1.0 / 255.0);
            gch[o + x] = p[1] as f32 * (1.0 / 255.0);
            b[o + x] = p[2] as f32 * (1.0 / 255.0);
        }
    }
}

/// Turns one image's raw rows and prototypes into detections.
fn decode(rows: &[f32], protos: &[f32], proto_w: usize, g: &Geometry, min_score: f32) -> Vec<Detection> {
    let (cw, ch) = (g.content.0 as f32, g.content.1 as f32);
    let (gw, gh) = g.grid;
    let proto_plane = protos.len() / COEFS;
    let mut found: Vec<Detection> = Vec::new();
    for row in rows.chunks_exact(6 + COEFS) {
        let (score, class) = (row[4], row[5]);
        if class != PERSON || score < min_score {
            continue;
        }
        let (x1, y1) = (row[0].clamp(0.0, cw), row[1].clamp(0.0, ch));
        let (x2, y2) = (row[2].clamp(0.0, cw), row[3].clamp(0.0, ch));
        if x2 - x1 < 2.0 || y2 - y1 < 2.0 {
            continue;
        }
        // Grid cells the box touches.
        let (gx1, gy1) = ((x1 / 4.0).floor() as usize, (y1 / 4.0).floor() as usize);
        let (gx2, gy2) = (((x2 / 4.0).ceil() as usize).min(gw), ((y2 / 4.0).ceil() as usize).min(gh));
        if gx2 <= gx1 || gy2 <= gy1 {
            continue;
        }
        let coefs = &row[6..];
        let (pw, ph) = (gx2 - gx1, gy2 - gy1);
        let mut logits = vec![0.0f32; pw * ph];
        for (k, &c) in coefs.iter().enumerate() {
            let proto = &protos[k * proto_plane..(k + 1) * proto_plane];
            for py in 0..ph {
                let src = &proto[(gy1 + py) * proto_w + gx1..(gy1 + py) * proto_w + gx2];
                for (acc, v) in logits[py * pw..(py + 1) * pw].iter_mut().zip(src) {
                    *acc += c * v;
                }
            }
        }
        let mask = MaskPatch::from_fn(gx1, gy1, pw, ph, |col, row| logits[row * pw + col] > 0.0);
        found.push(Detection { bbox: [x1 / cw, y1 / ch, x2 / cw, y2 / ch], score, mask });
    }
    drop_duplicates(found)
}

/// Keeps the higher-scoring of any two detections that are nearly the same box.
fn drop_duplicates(mut found: Vec<Detection>) -> Vec<Detection> {
    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::with_capacity(found.len());
    for d in found {
        if kept.iter().all(|k| iou(&k.bbox, &d.bbox) < 0.8) {
            kept.push(d);
        }
    }
    kept
}

/// Intersection over union of two `[x1, y1, x2, y2]` boxes.
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = w * h;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> Geometry {
        Geometry::new(1280, 720, 640, 1.0)
    }

    /// Prototype 0 is +1 on the left half of the image and -1 on the right.
    fn half_protos(g: &Geometry) -> (Vec<f32>, usize) {
        let (pw, ph) = (g.input.0 / 4, g.input.1 / 4);
        let mut protos = vec![0.0; COEFS * pw * ph];
        for y in 0..ph {
            for x in 0..pw {
                protos[y * pw + x] = if x < pw / 2 { 1.0 } else { -1.0 };
            }
        }
        (protos, pw)
    }

    fn row(b: [f32; 4], score: f32, class: f32) -> Vec<f32> {
        let mut r = vec![b[0], b[1], b[2], b[3], score, class];
        r.extend(std::iter::once(1.0).chain(std::iter::repeat_n(0.0, COEFS - 1)));
        r
    }

    #[test]
    fn decodes_people_and_their_outlines() {
        let g = geometry();
        let (protos, pw) = half_protos(&g);
        let mut rows = row([240.0, 40.0, 400.0, 200.0], 0.9, 0.0); // straddles the middle
        rows.extend(row([0.0, 0.0, 100.0, 100.0], 0.9, 16.0)); // a dog
        rows.extend(row([500.0, 100.0, 600.0, 300.0], 0.1, 0.0)); // too uncertain
        let found = decode(&rows, &protos, pw, &g, 0.25);
        assert_eq!(found.len(), 1);
        let d = &found[0];
        assert_eq!(d.bbox, [240.0 / 640.0, 40.0 / 360.0, 400.0 / 640.0, 200.0 / 360.0]);
        assert_eq!((d.mask.x, d.mask.y, d.mask.w, d.mask.h), (60, 10, 40, 40));
        // Only the half of the box left of the image centre (grid column 80) is "person".
        assert!(d.mask.get(0, 0) && d.mask.get(19, 39));
        assert!(!d.mask.get(20, 0) && !d.mask.get(39, 39));
        assert_eq!(d.mask.area(), 20 * 40);
    }

    #[test]
    fn near_identical_boxes_collapse_to_the_best() {
        let g = geometry();
        let (protos, pw) = half_protos(&g);
        let mut rows = row([100.0, 100.0, 200.0, 300.0], 0.6, 0.0);
        rows.extend(row([102.0, 100.0, 201.0, 300.0], 0.8, 0.0));
        let found = decode(&rows, &protos, pw, &g, 0.25);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].score, 0.8);
    }

    #[test]
    fn iou_of_known_boxes() {
        assert_eq!(iou(&[0.0, 0.0, 1.0, 1.0], &[0.0, 0.0, 1.0, 1.0]), 1.0);
        assert_eq!(iou(&[0.0, 0.0, 1.0, 1.0], &[2.0, 2.0, 3.0, 3.0]), 0.0);
        assert!((iou(&[0.0, 0.0, 2.0, 2.0], &[1.0, 0.0, 3.0, 2.0]) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn tensor_is_scaled_and_padded() {
        let g = Geometry::new(1280, 720, 640, 1.0); // content 640x360 in a 640x384 input
        let frame = vec![255u8; g.frame_bytes()];
        let mut tensor = vec![PAD; 3 * g.input.0 * g.input.1];
        fill_tensor(&mut tensor, &frame, &g, &mut Resizer::new(), &mut Vec::new());
        assert_eq!(tensor[0], 1.0);
        assert_eq!(tensor[359 * 640 + 639], 1.0);
        assert_eq!(tensor[360 * 640], PAD); // first padded row
    }
}
