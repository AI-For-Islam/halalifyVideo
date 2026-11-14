//! Scene-cut detection on the analysis frames.
//!
//! A cut ends every track: people on either side of a cut are unrelated, and
//! letting a track continue across one would mix two people's evidence.

use std::collections::VecDeque;

const COLS: usize = 16;
const ROWS: usize = 9;
/// A frame-to-frame change this large (mean absolute brightness difference,
/// 0..1) is needed before a cut is even considered.
const MIN_JUMP: f32 = 0.05;
/// ...and it must also be this many times the recent typical change, so that
/// fast action or a shaky camera is not mistaken for a cut.
const JUMP_RATIO: f32 = 4.0;
const HISTORY: usize = 12;

#[derive(Default)]
pub struct CutDetector {
    previous: Option<[f32; COLS * ROWS]>,
    recent: VecDeque<f32>,
}

impl CutDetector {
    /// Feeds the next frame (packed RGB) and says whether it starts a new shot.
    pub fn is_cut(&mut self, rgb: &[u8], width: usize, height: usize) -> bool {
        let signature = signature(rgb, width, height);
        let Some(previous) = self.previous.replace(signature) else { return false };
        let change = signature.iter().zip(&previous).map(|(a, b)| (a - b).abs()).sum::<f32>() / (COLS * ROWS) as f32;

        let mut sorted: Vec<f32> = self.recent.iter().copied().collect();
        sorted.sort_by(f32::total_cmp);
        let typical = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
        let cut = change > MIN_JUMP.max(JUMP_RATIO * typical);

        if cut {
            self.recent.clear(); // the new shot has its own motion statistics
        } else {
            if self.recent.len() == HISTORY {
                self.recent.pop_front();
            }
            self.recent.push_back(change);
        }
        cut
    }
}

/// Mean brightness of each cell of a coarse grid, sampling every 4th pixel.
fn signature(rgb: &[u8], width: usize, height: usize) -> [f32; COLS * ROWS] {
    let mut sums = [0u32; COLS * ROWS];
    let mut counts = [0u32; COLS * ROWS];
    for y in (0..height).step_by(4) {
        let row = y * ROWS / height;
        let line = &rgb[y * width * 3..(y + 1) * width * 3];
        for x in (0..width).step_by(4) {
            let p = &line[x * 3..x * 3 + 3];
            let cell = row * COLS + x * COLS / width;
            // Integer luma approximation (BT.601 weights scaled to 256).
            sums[cell] += (77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32) >> 8;
            counts[cell] += 1;
        }
    }
    let mut out = [0.0; COLS * ROWS];
    for i in 0..COLS * ROWS {
        out[i] = sums[i] as f32 / (counts[i].max(1) as f32 * 255.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 160;
    const H: usize = 90;

    /// A frame with a bright vertical bar at `bar_x` on a background of `level`.
    fn frame(level: u8, bar_x: usize) -> Vec<u8> {
        let mut f = vec![level; W * H * 3];
        for y in 0..H {
            for x in bar_x..(bar_x + 12).min(W) {
                f[(y * W + x) * 3..(y * W + x) * 3 + 3].fill(255);
            }
        }
        f
    }

    #[test]
    fn steady_motion_is_not_a_cut() {
        let mut d = CutDetector::default();
        for i in 0..30 {
            assert!(!d.is_cut(&frame(40, i * 4), W, H), "frame {i}");
        }
    }

    #[test]
    fn a_hard_change_is_a_cut_once() {
        let mut d = CutDetector::default();
        for i in 0..10 {
            assert!(!d.is_cut(&frame(40, i), W, H));
        }
        assert!(d.is_cut(&frame(200, 30), W, H));
        assert!(!d.is_cut(&frame(200, 31), W, H));
    }

    #[test]
    fn the_first_frame_is_never_a_cut() {
        assert!(!CutDetector::default().is_cut(&frame(255, 0), W, H));
    }
}
