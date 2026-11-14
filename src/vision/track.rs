//! Following each person from frame to frame.
//!
//! A simplified ByteTrack: detections are matched to live tracks by overlap
//! with the track's motion-predicted box, in two rounds. Confident detections
//! go first and may start new tracks; weak ones (a person half hidden, motion
//! blurred) may only extend an existing track, which keeps people covered
//! through hard frames without letting noise start tracks of its own.

use super::{
    detect::{Detection, iou},
    mask::MaskPatch,
};

#[derive(Clone, Debug)]
pub struct Observation {
    pub frame: u32,
    /// `[x1, y1, x2, y2]` as fractions of the frame.
    pub bbox: [f32; 4],
    pub score: f32,
    pub mask: MaskPatch,
    /// Classifier evidence that this is a woman, in log-odds, when this
    /// observation was sampled for classification.
    pub female_logit: Option<f32>,
}

/// One person over time. Observations are in frame order.
#[derive(Clone, Debug, Default)]
pub struct Track {
    pub observations: Vec<Observation>,
}

#[derive(Clone, Copy, Debug)]
pub struct TrackerConfig {
    /// Score at which a detection may start a new track.
    pub confident: f32,
    /// Minimum overlap for a confident detection to join a track.
    pub match_iou: f32,
    /// Minimum overlap for a weak detection to join a track.
    pub weak_match_iou: f32,
    /// Frames a track survives without a detection.
    pub max_gap: u32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        TrackerConfig { confident: 0.4, match_iou: 0.2, weak_match_iou: 0.4, max_gap: 20 }
    }
}

struct Live {
    id: usize,
    bbox: [f32; 4],
    /// Centre movement per frame, smoothed.
    velocity: [f32; 2],
    last_frame: u32,
}

impl Live {
    fn predicted(&self, frame: u32) -> [f32; 4] {
        let dt = frame.saturating_sub(self.last_frame) as f32;
        let (dx, dy) = (self.velocity[0] * dt, self.velocity[1] * dt);
        [self.bbox[0] + dx, self.bbox[1] + dy, self.bbox[2] + dx, self.bbox[3] + dy]
    }
}

pub struct Tracker {
    cfg: TrackerConfig,
    tracks: Vec<Track>,
    live: Vec<Live>,
}

impl Tracker {
    pub fn new(cfg: TrackerConfig) -> Tracker {
        Tracker { cfg, tracks: Vec::new(), live: Vec::new() }
    }

    /// Assigns each detection of `frame` to a track. Returns, per detection,
    /// the track id it joined or started; `None` for weak detections that
    /// matched nothing and were dropped.
    pub fn update(&mut self, frame: u32, detections: Vec<Detection>) -> Vec<Option<usize>> {
        let cfg = self.cfg;
        self.live.retain(|l| frame.saturating_sub(l.last_frame) <= cfg.max_gap);
        let predicted: Vec<[f32; 4]> = self.live.iter().map(|l| l.predicted(frame)).collect();

        let mut det_track: Vec<Option<usize>> = vec![None; detections.len()]; // index into self.live
        let mut live_taken = vec![false; self.live.len()];
        for (is_confident, threshold) in [(true, cfg.match_iou), (false, cfg.weak_match_iou)] {
            let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
            for (d, det) in detections.iter().enumerate() {
                if det_track[d].is_some() || (det.score >= cfg.confident) != is_confident {
                    continue;
                }
                for (l, pred) in predicted.iter().enumerate() {
                    if !live_taken[l] {
                        let overlap = iou(&det.bbox, pred);
                        if overlap >= threshold {
                            pairs.push((overlap, d, l));
                        }
                    }
                }
            }
            // Greedy assignment, best overlap first. With a handful of people
            // per frame this matches the optimal assignment in practice.
            pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (_, d, l) in pairs {
                if det_track[d].is_none() && !live_taken[l] {
                    det_track[d] = Some(l);
                    live_taken[l] = true;
                }
            }
        }

        let mut ids = Vec::with_capacity(detections.len());
        for (d, det) in detections.into_iter().enumerate() {
            let id = match det_track[d] {
                Some(l) => {
                    let live = &mut self.live[l];
                    let dt = frame.saturating_sub(live.last_frame).max(1) as f32;
                    let centre = |b: &[f32; 4]| [(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5];
                    let (old, new) = (centre(&live.bbox), centre(&det.bbox));
                    for a in 0..2 {
                        live.velocity[a] = 0.5 * live.velocity[a] + 0.5 * (new[a] - old[a]) / dt;
                    }
                    live.bbox = det.bbox;
                    live.last_frame = frame;
                    Some(live.id)
                }
                None if det.score >= cfg.confident => {
                    let id = self.tracks.len();
                    self.tracks.push(Track::default());
                    self.live.push(Live { id, bbox: det.bbox, velocity: [0.0; 2], last_frame: frame });
                    Some(id)
                }
                None => None,
            };
            if let Some(id) = id {
                self.tracks[id].observations.push(Observation {
                    frame,
                    bbox: det.bbox,
                    score: det.score,
                    mask: det.mask,
                    female_logit: None,
                });
            }
            ids.push(id);
        }
        ids
    }

    /// Ends every live track; called at a scene cut.
    pub fn cut(&mut self) {
        self.live.clear();
    }

    /// True when nobody is being followed.
    pub fn is_idle(&self) -> bool {
        self.live.is_empty()
    }

    pub fn track(&self, id: usize) -> &Track {
        &self.tracks[id]
    }

    pub fn track_mut(&mut self, id: usize) -> &mut Track {
        &mut self.tracks[id]
    }

    pub fn into_tracks(self) -> Vec<Track> {
        self.tracks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, score: f32) -> Detection {
        Detection { bbox: [x, 0.2, x + 0.1, 0.8], score, mask: MaskPatch::from_fn(0, 0, 1, 1, |_, _| true) }
    }

    #[test]
    fn a_moving_person_keeps_one_track() {
        let mut t = Tracker::new(TrackerConfig::default());
        for f in 0..20 {
            assert_eq!(t.update(f, vec![det(0.1 + f as f32 * 0.03, 0.9)]), vec![Some(0)]);
        }
        assert_eq!(t.into_tracks()[0].observations.len(), 20);
    }

    #[test]
    fn two_people_keep_their_own_tracks() {
        let mut t = Tracker::new(TrackerConfig::default());
        assert_eq!(t.update(0, vec![det(0.1, 0.9), det(0.6, 0.9)]), vec![Some(0), Some(1)]);
        // Reported in the other order on the next frame.
        assert_eq!(t.update(1, vec![det(0.61, 0.9), det(0.11, 0.9)]), vec![Some(1), Some(0)]);
    }

    #[test]
    fn a_short_gap_is_bridged_and_a_long_one_is_not() {
        let cfg = TrackerConfig { max_gap: 5, ..Default::default() };
        let mut t = Tracker::new(cfg);
        t.update(0, vec![det(0.3, 0.9)]);
        assert_eq!(t.update(4, vec![det(0.3, 0.9)]), vec![Some(0)]);
        assert_eq!(t.update(12, vec![det(0.3, 0.9)]), vec![Some(1)]);
    }

    #[test]
    fn weak_detections_extend_tracks_but_never_start_them() {
        let mut t = Tracker::new(TrackerConfig::default());
        assert_eq!(t.update(0, vec![det(0.3, 0.2)]), vec![None]);
        t.update(1, vec![det(0.3, 0.9)]);
        assert_eq!(t.update(2, vec![det(0.3, 0.2)]), vec![Some(0)]);
        assert_eq!(t.track(0).observations.len(), 2);
    }

    #[test]
    fn motion_prediction_follows_fast_movement() {
        // The person accelerates until each step exceeds the box width (0.1),
        // so the last boxes do not overlap at all; only the predicted
        // position keeps the track together.
        let mut t = Tracker::new(TrackerConfig::default());
        for (frame, x) in [0.0, 0.05, 0.12, 0.21, 0.32].into_iter().enumerate() {
            assert_eq!(t.update(frame as u32, vec![det(x, 0.9)]), vec![Some(0)], "frame {frame}");
        }
    }

    #[test]
    fn a_cut_ends_all_tracks() {
        let mut t = Tracker::new(TrackerConfig::default());
        t.update(0, vec![det(0.3, 0.9)]);
        t.cut();
        assert_eq!(t.update(1, vec![det(0.3, 0.9)]), vec![Some(1)]);
    }
}
