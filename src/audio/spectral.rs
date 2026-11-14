//! Shared plumbing for spectrogram-in, spectrogram-out models.
//!
//! Such a model takes the STFT of an audio channel as two planes, the real
//! parts and the imaginary parts, each `bins x frames` with frames along the
//! last axis, and returns planes of the same layout. This module turns
//! signals into those planes and back, in parallel across channels.

use rayon::prelude::*;

use super::stft::{Bin, Stft};

#[derive(Clone, Copy, Debug)]
pub struct Layout {
    /// Frequency bins the model sees, starting from bin 0. Bins above them
    /// are not modelled and come back silent.
    pub bins: usize,
    pub frames: usize,
    /// Bins below this one are given to the model as zero.
    pub first_bin: usize,
}

impl Layout {
    /// Values in one plane.
    pub fn plane(&self) -> usize {
        self.bins * self.frames
    }
}

/// STFTs every signal into its real and imaginary planes, concatenated in
/// order: `[signal 0 re, signal 0 im, signal 1 re, ...]`.
pub fn to_planes(stft: &Stft, signals: &[&[f32]], layout: &Layout) -> Vec<f32> {
    let Layout { bins, frames, first_bin } = *layout;
    let plane = layout.plane();
    let mut planes = vec![0.0f32; signals.len() * 2 * plane];
    planes.par_chunks_mut(2 * plane).zip(signals.par_iter()).for_each_init(
        || stft.scratch(),
        |scratch, (pair, signal)| {
            debug_assert_eq!(stft.frames(signal.len()), frames);
            let (re, im) = pair.split_at_mut(plane);
            stft.analyze(signal, scratch, |t, spectrum| {
                for f in first_bin..bins {
                    re[f * frames + t] = spectrum[f].re;
                    im[f * frames + t] = spectrum[f].im;
                }
            });
        },
    );
    planes
}

/// Inverse of [`to_planes`]: rebuilds `count` signals of `len` samples.
pub fn from_planes(stft: &Stft, planes: &[f32], count: usize, layout: &Layout, len: usize) -> Vec<Vec<f32>> {
    let Layout { bins, frames, .. } = *layout;
    let plane = layout.plane();
    assert_eq!(planes.len(), count * 2 * plane, "model output has an unexpected size");
    let mut signals: Vec<Vec<f32>> = (0..count).map(|_| vec![0.0f32; len]).collect();
    signals.par_iter_mut().zip(planes.par_chunks(2 * plane)).for_each_init(
        || stft.scratch(),
        |scratch, (signal, pair)| {
            let (re, im) = pair.split_at(plane);
            stft.synthesize(signal, scratch, |t, spectrum| {
                for (f, bin) in spectrum.iter_mut().enumerate() {
                    *bin = if f < bins { Bin::new(re[f * frames + t], im[f * frames + t]) } else { Bin::default() };
                }
            });
        },
    );
    signals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::stft::Padding;

    fn tone(len: usize, hz: f32) -> Vec<f32> {
        (0..len).map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / 44_100.0).sin()).collect()
    }

    #[test]
    fn planes_round_trip_every_signal() {
        let stft = Stft::new(1024, 256, Padding::Zero);
        let len = 256 * 20;
        let layout = Layout { bins: stft.bins(), frames: 21, first_bin: 0 };
        let (a, b) = (tone(len, 440.0), tone(len, 2_000.0));
        let planes = to_planes(&stft, &[&a, &b], &layout);
        assert_eq!(planes.len(), 2 * 2 * layout.plane());
        let back = from_planes(&stft, &planes, 2, &layout, len);
        for (original, rebuilt) in [(&a, &back[0]), (&b, &back[1])] {
            let err = original.iter().zip(rebuilt).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
            assert!(err < 1e-4, "max error {err}");
        }
    }

    #[test]
    fn bins_outside_the_layout_are_dropped() {
        // 44.1 kHz / 1024 = 43 Hz per bin: 440 Hz is bin 10, 8 kHz is bin 186.
        let stft = Stft::new(1024, 256, Padding::Reflect);
        let len = 256 * 20;
        let (low, high) = (tone(len, 440.0), tone(len, 8_000.0));
        let energy = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();

        // The model never sees bins from 100 up, so the 8 kHz tone vanishes.
        let narrow = Layout { bins: 100, frames: 21, first_bin: 0 };
        let back = from_planes(&stft, &to_planes(&stft, &[&low, &high], &narrow), 2, &narrow, len);
        assert!(energy(&back[0]) > 0.9 * energy(&low));
        assert!(energy(&back[1]) < 1e-3 * energy(&high));

        // Zeroing the bins below 50 removes the 440 Hz tone instead.
        let no_bass = Layout { bins: stft.bins(), frames: 21, first_bin: 50 };
        let back = from_planes(&stft, &to_planes(&stft, &[&low, &high], &no_bass), 2, &no_bass, len);
        assert!(energy(&back[0]) < 1e-3 * energy(&low));
        assert!(energy(&back[1]) > 0.9 * energy(&high));
    }
}
