//! Streaming overlap-add: runs a fixed-length chunk model over audio of any
//! length while holding only a few chunks in memory.
//!
//! Chunks start every `step` samples and are cross-faded with a window. Once a
//! chunk has been added, the first `step` samples of the accumulator can never
//! be touched again, so they are emitted and dropped. A two-hour film therefore
//! needs the same memory as a two-minute clip.

use anyhow::Result;

/// Stereo audio as two planes.
pub type Stereo = [Vec<f32>; 2];

pub struct OlaPlan {
    pub chunk: usize,
    pub step: usize,
    /// Silence placed before the first sample, so that real audio never sits
    /// at the very edge of a chunk where models are least reliable.
    pub lead: usize,
    /// Chunks handed to the model per call.
    pub batch: usize,
}

/// Symmetric Hann window (`numpy.hanning`), the cross-fade weight per sample.
pub fn hann(len: usize) -> Vec<f32> {
    let denom = (len.max(2) - 1) as f32;
    (0..len).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / denom).cos()).collect()
}

/// Pulls audio from `read`, runs `model` over overlapping chunks and pushes the
/// cross-faded result to `write`. Returns the number of samples written, which
/// always equals the number read.
///
/// * `read(left, right)` appends more samples to both planes and returns how
///   many it appended; `0` means end of stream.
/// * `model(chunks)` returns one output per input chunk, same length.
/// * `write(left, right)` receives finished samples in order.
pub fn run(
    plan: &OlaPlan,
    mut read: impl FnMut(&mut Vec<f32>, &mut Vec<f32>) -> Result<usize>,
    mut model: impl FnMut(&[Stereo]) -> Result<Vec<Stereo>>,
    mut write: impl FnMut(&[f32], &[f32]) -> Result<()>,
) -> Result<u64> {
    let OlaPlan { chunk, step, lead, batch } = *plan;
    assert!(step > 0 && step <= chunk && batch > 0);
    let window = hann(chunk);

    // Input not yet consumed; index 0 is stream position `pos` (in padded samples).
    let mut input: Stereo = [vec![0.0; lead], vec![0.0; lead]];
    let mut acc: Stereo = [vec![0.0; chunk], vec![0.0; chunk]];
    let mut weight = vec![0.0f32; chunk];
    let mut pos: u64 = 0;
    let mut read_total: u64 = 0;
    let mut written: u64 = 0;
    let mut eof = false;

    loop {
        let wanted = (batch - 1) * step + chunk;
        while !eof && input[0].len() < wanted {
            let [l, r] = &mut input;
            match read(l, r)? {
                0 => eof = true,
                n => read_total += n as u64,
            }
        }
        let end = lead as u64 + read_total; // padded position one past the last real sample
        let count = if !eof {
            ((input[0].len() - chunk) / step + 1).min(batch)
        } else if pos >= end {
            0
        } else {
            (((end - pos) as usize).div_ceil(step)).min(batch)
        };
        if count == 0 {
            break;
        }

        let chunks: Vec<Stereo> = (0..count)
            .map(|k| {
                let cut = |plane: &Vec<f32>| {
                    let from = (k * step).min(plane.len());
                    let to = (k * step + chunk).min(plane.len());
                    let mut c = plane[from..to].to_vec();
                    c.resize(chunk, 0.0);
                    c
                };
                [cut(&input[0]), cut(&input[1])]
            })
            .collect();
        let outputs = model(&chunks)?;
        assert_eq!(outputs.len(), count, "model must return one output per chunk");

        for out in &outputs {
            for ch in 0..2 {
                for ((a, o), w) in acc[ch].iter_mut().zip(&out[ch]).zip(&window) {
                    *a += o * w;
                }
            }
            for (ws, w) in weight.iter_mut().zip(&window) {
                *ws += w;
            }
            // The first `step` samples are final. Emit the part that is real audio.
            let from = lead as u64;
            let lo = pos.max(from).min(pos + step as u64);
            let hi = (pos + step as u64).min(end).max(lo);
            if hi > lo {
                let (a, b) = ((lo - pos) as usize, (hi - pos) as usize);
                let finish = |plane: &[f32]| -> Vec<f32> {
                    plane[a..b].iter().zip(&weight[a..b]).map(|(x, w)| x / w.max(1e-8)).collect()
                };
                write(&finish(&acc[0]), &finish(&acc[1]))?;
                written += (b - a) as u64;
            }
            for plane in acc.iter_mut().chain(std::iter::once(&mut weight)) {
                plane.copy_within(step.., 0);
                plane[chunk - step..].fill(0.0);
            }
            pos += step as u64;
        }
        for plane in &mut input {
            let n = (count * step).min(plane.len());
            plane.drain(..n);
        }
    }
    debug_assert_eq!(written, read_total);
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds `total` samples in uneven blocks through an identity model.
    fn round_trip(total: usize, plan: OlaPlan) -> (Vec<f32>, Vec<f32>) {
        let source: Vec<f32> = (0..total).map(|i| ((i * 37) % 201) as f32 / 100.0 - 1.0).collect();
        let mut cursor = 0;
        let mut out = Vec::new();
        let mut block = 1;
        let written = run(
            &plan,
            |l, r| {
                block = block % 977 + 113;
                let n = block.min(total - cursor);
                l.extend_from_slice(&source[cursor..cursor + n]);
                r.extend(source[cursor..cursor + n].iter().map(|x| -x));
                cursor += n;
                Ok(n)
            },
            |chunks| Ok(chunks.to_vec()),
            |l, r| {
                assert!(l.iter().zip(r).all(|(a, b)| (a + b).abs() < 1e-5));
                out.extend_from_slice(l);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(written as usize, total);
        (source, out)
    }

    #[test]
    fn identity_model_reproduces_the_stream() {
        for total in [1, 999, 4096, 10_000, 50_001] {
            for (step, batch) in [(3072, 1), (2048, 3), (1024, 2)] {
                let (src, out) = round_trip(total, OlaPlan { chunk: 4096, step, lead: 256, batch });
                assert_eq!(out.len(), src.len());
                let err = src.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                assert!(err < 1e-3, "total {total} step {step} batch {batch}: max error {err}");
            }
        }
    }

    #[test]
    fn empty_stream_produces_nothing() {
        let (_, out) = round_trip(0, OlaPlan { chunk: 4096, step: 3072, lead: 256, batch: 2 });
        assert!(out.is_empty());
    }
}
