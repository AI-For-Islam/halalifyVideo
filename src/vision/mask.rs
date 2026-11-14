//! Blur masks on the mask grid (a quarter of the detector's resolution).
//!
//! A [`MaskPatch`] is one person's outline in one frame, stored as bits so
//! that a whole film's worth of outlines fits comfortably in memory. A
//! [`Canvas`] is one frame of the mask stream sent to ffmpeg: 0 keeps the
//! pixel, 255 blurs it, and ffmpeg's bilinear upscale softens the edge.

/// A rectangle of bits positioned on the mask grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskPatch {
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
    bits: Box<[u64]>,
}

impl MaskPatch {
    /// Builds a patch by asking `inside(col, row)` for each local cell.
    pub fn from_fn(x: usize, y: usize, w: usize, h: usize, mut inside: impl FnMut(usize, usize) -> bool) -> MaskPatch {
        let mut bits = vec![0u64; (w * h).div_ceil(64)].into_boxed_slice();
        for row in 0..h {
            for col in 0..w {
                if inside(col, row) {
                    let i = row * w + col;
                    bits[i / 64] |= 1 << (i % 64);
                }
            }
        }
        MaskPatch { x: x as u16, y: y as u16, w: w as u16, h: h as u16, bits }
    }

    #[inline]
    pub fn get(&self, col: usize, row: usize) -> bool {
        let i = row * self.w as usize + col;
        self.bits[i / 64] >> (i % 64) & 1 == 1
    }

    /// Number of set cells.
    pub fn area(&self) -> usize {
        self.bits.iter().map(|b| b.count_ones() as usize).sum()
    }
}

/// One frame of the mask stream.
pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
    scratch: Vec<u8>,
}

impl Canvas {
    pub fn new(w: usize, h: usize) -> Canvas {
        Canvas { w, h, data: vec![0; w * h], scratch: vec![0; w * h] }
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|&v| v == 0)
    }

    pub fn paint_patch(&mut self, p: &MaskPatch) {
        for row in 0..p.h as usize {
            let y = p.y as usize + row;
            if y >= self.h {
                break;
            }
            for col in 0..p.w as usize {
                let x = p.x as usize + col;
                if x < self.w && p.get(col, row) {
                    self.data[y * self.w + x] = 255;
                }
            }
        }
    }

    /// Paints a rounded rectangle given as `[x1, y1, x2, y2]` in 0..1 frame
    /// coordinates. Corners are rounded by 30% of the shorter side, which
    /// reads as a soft "pill" over a standing person rather than a hard box.
    pub fn paint_box(&mut self, b: [f32; 4]) {
        let (w, h) = (self.w as f32, self.h as f32);
        let (x1, y1, x2, y2) = (b[0] * w, b[1] * h, b[2] * w, b[3] * h);
        if x2 <= x1 || y2 <= y1 {
            return;
        }
        let radius = 0.3 * (x2 - x1).min(y2 - y1);
        let (cols, rows) = (clamp_range(x1, x2, self.w), clamp_range(y1, y2, self.h));
        for y in rows {
            let cy = y as f32 + 0.5;
            // Distance into the corner zone, vertically.
            let dy = (y1 + radius - cy).max(cy - (y2 - radius)).max(0.0);
            for x in cols.clone() {
                let cx = x as f32 + 0.5;
                let dx = (x1 + radius - cx).max(cx - (x2 - radius)).max(0.0);
                if dx * dx + dy * dy <= radius * radius {
                    self.data[y * self.w + x] = 255;
                }
            }
        }
    }

    /// Grows every painted area by `radius` cells in all directions.
    pub fn dilate(&mut self, radius: usize) {
        if radius == 0 {
            return;
        }
        let (w, h) = (self.w, self.h);
        // Separable maximum filter: rows into scratch, columns back into data.
        for y in 0..h {
            let row = &self.data[y * w..(y + 1) * w];
            for x in 0..w {
                let (a, b) = (x.saturating_sub(radius), (x + radius + 1).min(w));
                self.scratch[y * w + x] = row[a..b].iter().copied().max().unwrap_or(0);
            }
        }
        for y in 0..h {
            let (a, b) = (y.saturating_sub(radius), (y + radius + 1).min(h));
            for x in 0..w {
                self.data[y * w + x] = (a..b).map(|yy| self.scratch[yy * w + x]).max().unwrap_or(0);
            }
        }
    }
}

/// Cells whose centre lies within `[lo, hi)`, clipped to `0..limit`.
fn clamp_range(lo: f32, hi: f32, limit: usize) -> std::ops::Range<usize> {
    let a = (lo.floor().max(0.0) as usize).min(limit);
    let b = (hi.ceil().max(0.0) as usize).min(limit);
    a..b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_round_trips_its_cells() {
        let p = MaskPatch::from_fn(3, 2, 9, 9, |c, r| (c + r) % 3 == 0);
        for r in 0..9 {
            for c in 0..9 {
                assert_eq!(p.get(c, r), (c + r) % 3 == 0);
            }
        }
        assert_eq!(p.area(), 27);
    }

    #[test]
    fn patch_is_painted_at_its_offset_and_clipped() {
        let mut c = Canvas::new(8, 6);
        c.paint_patch(&MaskPatch::from_fn(6, 4, 4, 4, |_, _| true));
        let painted: Vec<usize> = c.data.iter().enumerate().filter(|(_, v)| **v == 255).map(|(i, _)| i).collect();
        assert_eq!(painted, vec![4 * 8 + 6, 4 * 8 + 7, 5 * 8 + 6, 5 * 8 + 7]);
    }

    #[test]
    fn dilation_grows_by_the_radius() {
        let mut c = Canvas::new(9, 9);
        c.data[4 * 9 + 4] = 255;
        c.dilate(2);
        for y in 0..9 {
            for x in 0..9 {
                let inside = (2..=6).contains(&x) && (2..=6).contains(&y);
                assert_eq!(c.data[y * 9 + x] == 255, inside, "({x},{y})");
            }
        }
    }

    #[test]
    fn box_is_filled_with_rounded_corners() {
        let mut c = Canvas::new(40, 40);
        c.paint_box([0.25, 0.25, 0.75, 0.75]);
        assert_eq!(c.data[20 * 40 + 20], 255); // centre
        assert_eq!(c.data[10 * 40 + 20], 255); // middle of the top edge
        assert_eq!(c.data[10 * 40 + 10], 0); // the very corner is rounded off
        assert_eq!(c.data[5 * 40 + 5], 0); // outside
        assert!(!c.is_empty());
        c.clear();
        assert!(c.is_empty());
    }
}
