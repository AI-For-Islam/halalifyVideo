//! From tracks to a blur plan: who gets blurred, and where in every frame.
//!
//! The whole video has been analysed before anything is decided, so each
//! person is judged on all the evidence gathered while they were on screen,
//! and the blur covers them from their first frame to their last (plus a
//! little on both sides). Nothing flickers on or off as confidence changes.

use super::{
    mask::{Canvas, MaskPatch},
    track::Track,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Target {
    Women,
    Men,
    Everyone,
    Nobody,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Shape {
    /// The person's outline, grown a little.
    Silhouette,
    /// A rounded box around the person: coarser, but nothing can peek out.
    Box,
}

#[derive(Clone, Copy, Debug)]
pub struct DecideConfig {
    pub target: Target,
    pub shape: Shape,
    /// Probability of "woman" from which a person is treated as one. Below
    /// 0.5 on purpose: missing a woman is worse than blurring a man.
    pub threshold: f32,
    /// Frames of blur added before a person appears and after they leave,
    /// covering the frames in which the detector had not yet caught them.
    pub pad_frames: u32,
    /// Mask-grid cells the blurred area is grown by.
    pub grow: usize,
    /// Half-width, in frames, of the window whose samples count as "nearby"
    /// evidence for a moment of a track.
    pub window: u32,
}

impl Default for DecideConfig {
    fn default() -> Self {
        DecideConfig { target: Target::Women, shape: Shape::Silhouette, threshold: 0.35, pad_frames: 3, grow: 2, window: 36 }
    }
}

/// A single classifier sample counts for at most this much evidence, so one
/// confident mistake cannot outvote several ordinary samples.
const MAX_SAMPLE_LOGIT: f32 = 6.0;
/// Fraction of the frame height from which a sample counts in full; smaller
/// (more distant, blurrier) people count proportionally less.
const FULL_WEIGHT_HEIGHT: f32 = 0.25;
/// A box is grown by this fraction of its size on every side before painting.
const BOX_MARGIN: f32 = 0.04;

#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    /// Probability that the person is a woman, from all samples of the track;
    /// `None` if never classified.
    pub female: Option<f32>,
    pub samples: usize,
    /// Whether any part of the track is blurred.
    pub blur: bool,
    /// The decision for each observation of the track.
    pub blur_at: Vec<bool>,
}

/// Weighs the classifier samples of a track and applies the blur policy to
/// each of its observations.
///
/// Every moment is judged half on the whole track and half on the samples
/// near it in time. For a track that really is one person the two agree. When
/// the tracker has confused two people who crossed paths, so that one track
/// is a man for a while and then a woman, the nearby evidence pulls each
/// stretch towards the right answer instead of averaging the two away.
pub fn judge(track: &Track, cfg: &DecideConfig) -> Verdict {
    // (frame, weight, clipped logit) of every classified observation
    let samples: Vec<(u32, f32, f32)> = track
        .observations
        .iter()
        .filter_map(|o| {
            let weight = ((o.bbox[3] - o.bbox[1]) / FULL_WEIGHT_HEIGHT).clamp(0.2, 1.0);
            Some((o.frame, weight, o.female_logit?.clamp(-MAX_SAMPLE_LOGIT, MAX_SAMPLE_LOGIT)))
        })
        .collect();
    let mean = |items: &mut dyn Iterator<Item = &(u32, f32, f32)>| -> Option<f32> {
        let (sum, weight) = items.fold((0.0, 0.0), |(s, w), &(_, wi, l)| (s + wi * l, w + wi));
        (weight > 0.0).then(|| sum / weight)
    };
    let overall = mean(&mut samples.iter());
    let probability = |logit: f32| 1.0 / (1.0 + (-logit).exp());
    let wanted = |female: Option<f32>| match cfg.target {
        Target::Everyone => true,
        Target::Nobody => false,
        // Someone who could not be classified is blurred: the cautious choice.
        Target::Women => female.is_none_or(|p| p >= cfg.threshold),
        Target::Men => female.is_none_or(|p| p <= 1.0 - cfg.threshold),
    };
    let blur_at: Vec<bool> = track
        .observations
        .iter()
        .map(|o| {
            let nearby = mean(&mut samples.iter().filter(|s| s.0.abs_diff(o.frame) <= cfg.window));
            wanted(overall.map(|all| probability(0.5 * all + 0.5 * nearby.unwrap_or(all))))
        })
        .collect();
    Verdict { female: overall.map(probability), samples: samples.len(), blur: blur_at.contains(&true), blur_at }
}

enum Region<'a> {
    Outline(&'a MaskPatch),
    Rounded([f32; 4]),
}

/// What to blur in every frame.
pub struct BlurPlan<'a> {
    frames: Vec<Vec<Region<'a>>>,
    grow: usize,
}

impl<'a> BlurPlan<'a> {
    /// `cuts` lists the frames that start a new shot; blur padding never
    /// crosses one, since the neighbouring shot shows something else there.
    pub fn build(tracks: &'a [Track], verdicts: &[Verdict], total_frames: u32, cuts: &[u32], cfg: &DecideConfig) -> BlurPlan<'a> {
        let mut frames: Vec<Vec<Region<'a>>> = (0..total_frames).map(|_| Vec::new()).collect();
        for (track, verdict) in tracks.iter().zip(verdicts) {
            let obs = &track.observations;
            let (Some(first), Some(last)) = (obs.first(), obs.last()) else { continue };
            if !verdict.blur {
                continue;
            }
            let chosen = |i: usize| verdict.blur_at.get(i).copied().unwrap_or(true);
            // The shot this track lives in: [shot_start, shot_end).
            let shot_start = cuts.iter().copied().filter(|&c| c <= first.frame).max().unwrap_or(0);
            let shot_end = cuts.iter().copied().filter(|&c| c > last.frame).min().unwrap_or(total_frames).min(total_frames);
            let mut put = |frame: i64, region: Region<'a>| {
                if frame >= shot_start as i64 && frame < shot_end as i64 {
                    frames[frame as usize].push(region);
                }
            };
            let region_of = |o: &'a super::track::Observation| match cfg.shape {
                // A degenerate outline (the model found almost no pixels) falls back to the box.
                Shape::Silhouette if o.mask.area() * 8 >= o.mask.w as usize * o.mask.h as usize => Region::Outline(&o.mask),
                _ => Region::Rounded(grow_box(o.bbox)),
            };

            for (i, o) in obs.iter().enumerate() {
                // Each outline also covers the neighbouring frames, which
                // hides one-frame lapses of the outline and fast movement.
                if chosen(i) {
                    for f in o.frame as i64 - 1..=o.frame as i64 + 1 {
                        put(f, region_of(o));
                    }
                }
                // Frames between two sightings: move a box smoothly from one to the other.
                if let Some(next) = obs.get(i + 1).filter(|_| chosen(i) || chosen(i + 1)) {
                    let span = (next.frame - o.frame) as f32;
                    for f in o.frame + 1..next.frame {
                        let t = (f - o.frame) as f32 / span;
                        let lerp = |a: f32, b: f32| a + (b - a) * t;
                        let b = [0, 1, 2, 3].map(|k| lerp(o.bbox[k], next.bbox[k]));
                        put(f as i64, Region::Rounded(grow_box(b)));
                    }
                }
            }
            for k in 2..=cfg.pad_frames as i64 {
                if chosen(0) {
                    put(first.frame as i64 - k, region_of(first));
                }
                if chosen(obs.len() - 1) {
                    put(last.frame as i64 + k, region_of(last));
                }
            }
        }
        BlurPlan { frames, grow: cfg.grow }
    }

    pub fn frame_count(&self) -> u32 {
        self.frames.len() as u32
    }

    /// Number of frames in which something is blurred.
    pub fn blurred_frames(&self) -> usize {
        self.frames.iter().filter(|f| !f.is_empty()).count()
    }

    /// Paints the mask of `frame` onto `canvas` (cleared first).
    pub fn paint(&self, frame: u32, canvas: &mut Canvas) {
        canvas.clear();
        let Some(regions) = self.frames.get(frame as usize) else { return };
        if regions.is_empty() {
            return;
        }
        for region in regions {
            match region {
                Region::Outline(patch) => canvas.paint_patch(patch),
                Region::Rounded(b) => canvas.paint_box(*b),
            }
        }
        canvas.dilate(self.grow);
    }
}

fn grow_box(b: [f32; 4]) -> [f32; 4] {
    let (mx, my) = (BOX_MARGIN * (b[2] - b[0]), BOX_MARGIN * (b[3] - b[1]));
    [b[0] - mx, b[1] - my, b[2] + mx, b[3] + my]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::track::Observation;

    fn obs(frame: u32, x: f32, logit: Option<f32>) -> Observation {
        Observation {
            frame,
            bbox: [x, 0.25, x + 0.25, 0.75],
            score: 0.9,
            // A full 10x20 outline at the box position on a 40x40 grid.
            mask: MaskPatch::from_fn((x * 40.0) as usize, 10, 10, 20, |_, _| true),
            female_logit: logit,
        }
    }

    fn track(samples: &[(u32, f32, Option<f32>)]) -> Track {
        Track { observations: samples.iter().map(|&(f, x, l)| obs(f, x, l)).collect() }
    }

    #[test]
    fn evidence_is_pooled_over_the_whole_track() {
        let cfg = DecideConfig::default();
        // One doubtful sample among clear ones does not flip the verdict.
        let woman = track(&[(0, 0.1, Some(3.0)), (6, 0.1, Some(-0.5)), (12, 0.1, Some(4.0))]);
        let v = judge(&woman, &cfg);
        assert!(v.blur && v.samples == 3 && v.female.unwrap() > 0.85);
        let man = track(&[(0, 0.1, Some(-3.0)), (6, 0.1, Some(0.4)), (12, 0.1, Some(-4.0))]);
        assert!(!judge(&man, &cfg).blur);
    }

    #[test]
    fn one_extreme_sample_cannot_outvote_the_rest() {
        let t = track(&[(0, 0.1, Some(40.0)), (6, 0.1, Some(-3.0)), (12, 0.1, Some(-3.0)), (18, 0.1, Some(-3.0))]);
        assert!(!judge(&t, &DecideConfig::default()).blur);
    }

    #[test]
    fn the_threshold_leans_towards_blurring() {
        let cfg = DecideConfig::default();
        // 40% "woman" is below even odds but still blurred.
        let unsure = track(&[(0, 0.1, Some(-0.4))]);
        assert!(judge(&unsure, &cfg).blur);
        let unknown = track(&[(0, 0.1, None)]);
        assert_eq!(judge(&unknown, &cfg), Verdict { female: None, samples: 0, blur: true, blur_at: vec![true] });
    }

    #[test]
    fn targets_select_the_right_people() {
        let woman = track(&[(0, 0.1, Some(4.0))]);
        let man = track(&[(0, 0.1, Some(-4.0))]);
        let with = |target| DecideConfig { target, ..Default::default() };
        assert!(judge(&woman, &with(Target::Women)).blur && !judge(&man, &with(Target::Women)).blur);
        assert!(!judge(&woman, &with(Target::Men)).blur && judge(&man, &with(Target::Men)).blur);
        assert!(judge(&man, &with(Target::Everyone)).blur && !judge(&woman, &with(Target::Nobody)).blur);
    }

    #[test]
    fn a_track_that_switches_person_is_judged_stretch_by_stretch() {
        // The tracker followed a man for 100 frames, then a woman who crossed him.
        let mut samples: Vec<(u32, f32, Option<f32>)> = (0..100).map(|f| (f, 0.1, (f % 6 == 0).then_some(-3.5))).collect();
        samples.extend((100..200).map(|f| (f, 0.1, (f % 6 == 0).then_some(3.5))));
        let t = track(&samples);
        let v = judge(&t, &DecideConfig::default());
        assert!(v.blur);
        assert!(v.blur_at[..60].iter().all(|b| !b), "the man is left alone");
        assert!(v.blur_at[140..].iter().all(|b| *b), "the woman is blurred");
        // Overall the evidence cancels out, which is exactly why it is not used alone.
        assert!((v.female.unwrap() - 0.5).abs() < 0.1);
    }

    fn painted(plan: &BlurPlan, frame: u32) -> usize {
        let mut c = Canvas::new(40, 40);
        plan.paint(frame, &mut c);
        c.data.iter().filter(|&&v| v == 255).count()
    }

    #[test]
    fn blur_covers_the_track_its_gaps_and_a_margin_in_time() {
        let cfg = DecideConfig { grow: 0, ..Default::default() };
        let tracks = [track(&[(10, 0.1, Some(4.0)), (11, 0.1, None), (15, 0.5, None)])];
        let verdicts = [judge(&tracks[0], &cfg)];
        let plan = BlurPlan::build(&tracks, &verdicts, 30, &[], &cfg);
        assert_eq!(painted(&plan, 6), 0);
        for f in 7..=18 {
            assert!(painted(&plan, f) > 0, "frame {f} should be blurred");
        }
        assert_eq!(painted(&plan, 19), 0);
        assert_eq!(plan.blurred_frames(), 12);

        // Frame 13 lies in the gap: its box sits between the two sightings.
        let mut c = Canvas::new(40, 40);
        plan.paint(13, &mut c);
        let cols: Vec<usize> = (0..40).filter(|&x| c.data[20 * 40 + x] == 255).collect();
        assert!(cols[0] > 4 && *cols.last().unwrap() < 30, "gap box spans {cols:?}");
    }

    #[test]
    fn people_who_are_not_targets_are_left_alone() {
        let cfg = DecideConfig::default();
        let tracks = [track(&[(5, 0.1, Some(-4.0))])];
        let verdicts = [judge(&tracks[0], &cfg)];
        let plan = BlurPlan::build(&tracks, &verdicts, 20, &[], &cfg);
        assert_eq!(plan.blurred_frames(), 0);
    }

    #[test]
    fn padding_stops_at_scene_cuts() {
        let cfg = DecideConfig { grow: 0, ..Default::default() };
        let tracks = [track(&[(10, 0.1, Some(4.0)), (11, 0.1, None)])];
        let verdicts = [judge(&tracks[0], &cfg)];
        let plan = BlurPlan::build(&tracks, &verdicts, 30, &[10, 12], &cfg);
        assert_eq!(painted(&plan, 9), 0);
        assert!(painted(&plan, 10) > 0 && painted(&plan, 11) > 0);
        assert_eq!(painted(&plan, 12), 0);
    }

    #[test]
    fn growing_enlarges_the_painted_area() {
        let tracks = [track(&[(5, 0.3, Some(4.0))])];
        let tight = DecideConfig { grow: 0, ..Default::default() };
        let loose = DecideConfig { grow: 2, ..Default::default() };
        let v = [judge(&tracks[0], &tight)];
        let a = painted(&BlurPlan::build(&tracks, &v, 10, &[], &tight), 5);
        let b = painted(&BlurPlan::build(&tracks, &v, 10, &[], &loose), 5);
        assert_eq!(a, 200);
        assert_eq!(b, 14 * 24);
    }
}
