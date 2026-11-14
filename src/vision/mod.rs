//! Video analysis: find every person, follow them, and gather evidence about
//! who they are. Produces tracks; `decide` turns them into a blur plan.

pub mod classify;
pub mod cuts;
pub mod decide;
pub mod detect;
pub mod frames;
pub mod mask;
pub mod track;

use std::{
    path::Path,
    sync::mpsc::{channel, sync_channel},
    time::Instant,
};

use anyhow::{Result, anyhow};

use self::{
    classify::{Crop, PersonClassifier},
    cuts::CutDetector,
    detect::Detector,
    frames::{FrameReader, Geometry},
    track::{Track, Tracker, TrackerConfig},
};
use crate::{audio::io::Span, probe::VideoStream, runtime::Runtime, tools::Tools};

#[derive(Clone, Copy, Debug)]
pub struct VisionConfig {
    /// Long side of the detector's input, in pixels.
    pub detect_side: usize,
    /// Analysis frames are this much larger than the detector input, giving
    /// the person classifier sharper crops.
    pub detail: f32,
    /// Frames per detector call.
    pub batch: usize,
    /// Weakest detection passed to the tracker.
    pub min_score: f32,
    /// Detect on every n-th frame; frames in between are covered by interpolation.
    pub stride: u32,
    /// A person is re-classified every this many frames while on screen.
    pub classify_every: u32,
    /// Whether the classifier is needed at all (not when blurring everyone).
    pub classify: bool,
    /// While nobody is on screen, look only at every this-many-th frame.
    pub idle_factor: u32,
    pub tracker: TrackerConfig,
}

impl Default for VisionConfig {
    fn default() -> Self {
        VisionConfig {
            detect_side: 640,
            detail: 1.5,
            batch: 16,
            min_score: 0.15,
            stride: 1,
            classify_every: 6,
            classify: true,
            idle_factor: 3,
            tracker: TrackerConfig::default(),
        }
    }
}

pub struct Analysis {
    pub geometry: Geometry,
    /// Frames in the normalised video; the blur pass must supply this many masks.
    pub frames: u32,
    pub tracks: Vec<Track>,
    /// Frames that start a new shot.
    pub cuts: Vec<u32>,
    /// Person crops sent to the classifier.
    pub classified: usize,
    pub elapsed: f64,
    /// Seconds spent in the detector and in the classifier (including their
    /// image preparation); the rest of `elapsed` is decoding and tracking.
    pub detect_seconds: f64,
    pub classify_seconds: f64,
}

/// Decides which frames the detector looks at.
///
/// While people are on screen that is every `stride`-th frame. While nobody
/// is, it relaxes to every `stride * idle_factor`-th, which makes footage
/// without people (slides, landscapes, screen recordings) several times
/// cheaper. The first frame of every shot is always looked at. Someone who
/// walks in during a relaxed stretch is found at most `idle_factor - 1`
/// frames late, which the blur plan's padding in time covers; `idle_factor`
/// must therefore not exceed that padding.
struct Pacer {
    stride: u32,
    idle_factor: u32,
    next: u32,
    idle: bool,
}

impl Pacer {
    fn wants(&mut self, frame: u32, starts_shot: bool) -> bool {
        if starts_shot {
            self.next = frame;
        }
        if frame % self.stride != 0 || frame < self.next {
            return false;
        }
        self.next = frame + self.stride * if self.idle { self.idle_factor } else { 1 };
        true
    }
}

/// Three samples agreeing this strongly (in log-odds) settle who someone is.
const SETTLED_LOGIT: f32 = 2.5;
/// A settled track is re-checked this many times less often.
const RELAXED: u32 = 4;
/// Below this fraction of the frame height a person is too small for one
/// more sample to tell much; they are sampled this many times less often.
const SMALL_HEIGHT: f32 = 0.12;
const SMALL_RELAXED: u32 = 3;

/// Running classifier evidence for one track, used to decide when to look again.
#[derive(Clone, Copy, Default)]
struct Evidence {
    last: Option<u32>,
    sum: f32,
    count: u32,
}

impl Evidence {
    /// Whether the track should be classified again at `frame`.
    ///
    /// A person is sampled every `every` frames until the samples agree
    /// strongly; after that only occasionally, to notice if the tracker has
    /// moved on to someone else. Distant people, whose crops stay
    /// inconclusive however often they are sampled, are also sampled less.
    fn due(&self, frame: u32, every: u32, height: f32) -> bool {
        let Some(last) = self.last else { return true };
        let settled = self.count >= 3 && (self.sum / self.count as f32).abs() >= SETTLED_LOGIT;
        let relax = if settled {
            RELAXED
        } else if height < SMALL_HEIGHT {
            SMALL_RELAXED
        } else {
            1
        };
        frame - last >= every * relax
    }
}

/// A classification to run once the batch's detections are tracked.
struct Job {
    slot: usize,
    track: usize,
    observation: usize,
    bbox: [f32; 4],
}

/// Runs the detector, tracker and classifier over the whole video.
///
/// `progress` receives the number of frames finished.
pub fn analyze(
    tools: &Tools,
    rt: &Runtime,
    input: &Path,
    video: &VideoStream,
    span: Span,
    cfg: &VisionConfig,
    mut progress: impl FnMut(u32),
) -> Result<Analysis> {
    let started = Instant::now();
    let geometry = Geometry::new(video.width, video.height, cfg.detect_side, cfg.detail);
    let mut detector = Detector::load(rt, geometry)?;
    let mut classifier = if cfg.classify { Some(PersonClassifier::load(rt)?) } else { None };
    let mut tracker = Tracker::new(cfg.tracker);
    let mut cut_detector = CutDetector::default();
    let mut cuts = Vec::new();
    let mut evidence: Vec<Evidence> = Vec::new();
    let mut pacer = Pacer { stride: cfg.stride.max(1), idle_factor: cfg.idle_factor.max(1), next: 0, idle: false };
    let mut classified = 0usize;
    let (mut detect_seconds, mut classify_seconds) = (0.0f64, 0.0f64);
    let mut frame_index = 0u32;

    let mut reader = FrameReader::open(tools, input, video, &geometry, span)?;
    // Decoding runs on its own thread so that it overlaps with inference;
    // frame buffers travel back through `spare` to be reused.
    let (frame_tx, frame_rx) = sync_channel::<Vec<u8>>(cfg.batch * 2);
    let (spare_tx, spare_rx) = channel::<Vec<u8>>();

    std::thread::scope(|scope| -> Result<()> {
        let decoder = scope.spawn(move || -> Result<()> {
            loop {
                let mut buffer = spare_rx.try_recv().unwrap_or_default();
                if !reader.read(&mut buffer)? || frame_tx.send(buffer).is_err() {
                    break;
                }
            }
            drop(frame_tx);
            reader.finish()
        });

        // The loop is a closure so that an error inside it still reaches the
        // clean-up below; returning early would leave the decoder blocked.
        let outcome = (|| -> Result<()> {
            let mut batch: Vec<Vec<u8>> = Vec::with_capacity(cfg.batch);
            loop {
                // A batch holds `cfg.batch` frames to detect on, plus skipped ones.
                let mut to_detect: Vec<usize> = Vec::with_capacity(cfg.batch);
                let mut cut_before: Vec<bool> = Vec::with_capacity(cfg.batch);
                let mut pending_cut = false;
                while to_detect.len() < cfg.batch {
                    let Ok(frame) = frame_rx.recv() else { break };
                    let index = frame_index + batch.len() as u32;
                    let starts_shot = cut_detector.is_cut(&frame, geometry.frame.0, geometry.frame.1);
                    if starts_shot {
                        cuts.push(index);
                        pending_cut = true;
                    }
                    if pacer.wants(index, starts_shot) {
                        to_detect.push(batch.len());
                        cut_before.push(std::mem::take(&mut pending_cut));
                    }
                    batch.push(frame);
                }
                if batch.is_empty() {
                    break;
                }

                let views: Vec<&[u8]> = to_detect.iter().map(|&slot| batch[slot].as_slice()).collect();
                let clock = Instant::now();
                let detections = if views.is_empty() { Vec::new() } else { detector.detect(&views, cfg.min_score)? };
                detect_seconds += clock.elapsed().as_secs_f64();

                let mut jobs: Vec<Job> = Vec::new();
                for ((&slot, found), cut) in to_detect.iter().zip(detections).zip(cut_before) {
                    let frame = frame_index + slot as u32;
                    if cut {
                        tracker.cut();
                    }
                    for id in tracker.update(frame, found).into_iter().flatten() {
                        if evidence.len() <= id {
                            evidence.resize(id + 1, Evidence::default());
                        }
                        let observations = &tracker.track(id).observations;
                        let observation = observations.len() - 1;
                        let bbox = observations[observation].bbox;
                        if classifier.is_some() && evidence[id].due(frame, cfg.classify_every, bbox[3] - bbox[1]) {
                            evidence[id].last = Some(frame);
                            jobs.push(Job { slot, track: id, observation, bbox });
                        }
                    }
                }
                // A cut on a skipped frame at the end of the batch still ends the tracks.
                if pending_cut {
                    tracker.cut();
                }
                pacer.idle = tracker.is_idle();

                if let Some(classifier) = classifier.as_mut().filter(|_| !jobs.is_empty()) {
                    let clock = Instant::now();
                    let crops: Vec<Crop> = jobs.iter().map(|j| Crop { frame: &batch[j.slot], bbox: j.bbox }).collect();
                    let logits = classifier.female_logits(&crops, geometry.frame)?;
                    classify_seconds += clock.elapsed().as_secs_f64();
                    for (job, logit) in jobs.iter().zip(logits) {
                        tracker.track_mut(job.track).observations[job.observation].female_logit = Some(logit);
                        evidence[job.track].sum += logit;
                        evidence[job.track].count += 1;
                    }
                    classified += jobs.len();
                }

                frame_index += batch.len() as u32;
                progress(frame_index);
                for buffer in batch.drain(..) {
                    let _ = spare_tx.send(buffer);
                }
            }
            Ok(())
        })();
        // Dropping the receiver unblocks the decoder if it is waiting to send.
        drop(frame_rx);
        let decoded = decoder.join().map_err(|_| anyhow!("the video decoding thread panicked"))?;
        outcome.and(decoded)
    })?;

    Ok(Analysis {
        geometry,
        frames: frame_index,
        tracks: tracker.into_tracks(),
        cuts,
        classified,
        elapsed: started.elapsed().as_secs_f64(),
        detect_seconds,
        classify_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn looked_at(pacer: &mut Pacer, frames: std::ops::Range<u32>, cut_at: Option<u32>) -> Vec<u32> {
        frames.filter(|&f| pacer.wants(f, cut_at == Some(f))).collect()
    }

    #[test]
    fn every_frame_is_looked_at_while_people_are_on_screen() {
        let mut p = Pacer { stride: 1, idle_factor: 3, next: 0, idle: false };
        assert_eq!(looked_at(&mut p, 0..5, None), [0, 1, 2, 3, 4]);
        let mut p = Pacer { stride: 2, idle_factor: 3, next: 0, idle: false };
        assert_eq!(looked_at(&mut p, 0..7, None), [0, 2, 4, 6]);
    }

    #[test]
    fn an_empty_scene_is_looked_at_less_often_but_a_new_shot_at_once() {
        let mut p = Pacer { stride: 1, idle_factor: 3, next: 0, idle: true };
        assert_eq!(looked_at(&mut p, 0..10, None), [0, 3, 6, 9]);
        let mut p = Pacer { stride: 1, idle_factor: 3, next: 0, idle: true };
        assert_eq!(looked_at(&mut p, 0..10, Some(4)), [0, 3, 4, 7]);
        // With a stride, a cut on a skipped frame is picked up on the next stride frame.
        let mut p = Pacer { stride: 2, idle_factor: 3, next: 0, idle: true };
        assert_eq!(looked_at(&mut p, 0..14, Some(3)), [0, 4, 10]);
    }

    #[test]
    fn idle_gaps_fit_inside_the_blur_padding() {
        // Someone entering an empty scene is seen at most idle_factor - 1
        // frames late; the plan pads that many frames before a first sighting.
        let pad = decide::DecideConfig::default().pad_frames;
        assert!(VisionConfig::default().idle_factor <= pad);
    }

    #[test]
    fn settled_tracks_are_sampled_less_often() {
        let fresh = Evidence::default();
        assert!(fresh.due(0, 6, 0.5));
        let unsure = Evidence { last: Some(10), sum: 0.9, count: 3 };
        assert!(!unsure.due(15, 6, 0.5) && unsure.due(16, 6, 0.5));
        let settled = Evidence { last: Some(10), sum: -10.5, count: 3 };
        assert!(!settled.due(16, 6, 0.5) && !settled.due(33, 6, 0.5) && settled.due(34, 6, 0.5));
        // Two samples are not enough to relax, however strong.
        let early = Evidence { last: Some(10), sum: 9.0, count: 2 };
        assert!(early.due(16, 6, 0.5));
        // A distant person is sampled less often, but always at least once.
        assert!(fresh.due(0, 6, 0.05));
        assert!(!unsure.due(16, 6, 0.05) && unsure.due(28, 6, 0.05));
    }
}
