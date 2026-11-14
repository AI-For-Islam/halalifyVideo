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
    pub tracker: TrackerConfig,
}

impl Default for VisionConfig {
    fn default() -> Self {
        VisionConfig {
            detect_side: 640,
            detail: 1.5,
            batch: 8,
            min_score: 0.15,
            stride: 1,
            classify_every: 6,
            classify: true,
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
    let mut last_classified: Vec<Option<u32>> = Vec::new();
    let mut classified = 0usize;
    let mut frame_index = 0u32;
    let stride = cfg.stride.max(1);

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
                    if cut_detector.is_cut(&frame, geometry.frame.0, geometry.frame.1) {
                        cuts.push(index);
                        pending_cut = true;
                    }
                    if index % stride == 0 {
                        to_detect.push(batch.len());
                        cut_before.push(std::mem::take(&mut pending_cut));
                    }
                    batch.push(frame);
                }
                if batch.is_empty() {
                    break;
                }

                let views: Vec<&[u8]> = to_detect.iter().map(|&slot| batch[slot].as_slice()).collect();
                let detections = if views.is_empty() { Vec::new() } else { detector.detect(&views, cfg.min_score)? };

                let mut jobs: Vec<Job> = Vec::new();
                for ((&slot, found), cut) in to_detect.iter().zip(detections).zip(cut_before) {
                    let frame = frame_index + slot as u32;
                    if cut {
                        tracker.cut();
                    }
                    for id in tracker.update(frame, found).into_iter().flatten() {
                        if last_classified.len() <= id {
                            last_classified.resize(id + 1, None);
                        }
                        let due = last_classified[id].is_none_or(|at| frame - at >= cfg.classify_every);
                        if due && classifier.is_some() {
                            last_classified[id] = Some(frame);
                            let observations = &tracker.track(id).observations;
                            let observation = observations.len() - 1;
                            jobs.push(Job { slot, track: id, observation, bbox: observations[observation].bbox });
                        }
                    }
                }
                // A cut on a skipped frame at the end of the batch still ends the tracks.
                if pending_cut {
                    tracker.cut();
                }

                if let Some(classifier) = classifier.as_mut().filter(|_| !jobs.is_empty()) {
                    let crops: Vec<Crop> = jobs.iter().map(|j| Crop { frame: &batch[j.slot], bbox: j.bbox }).collect();
                    let logits = classifier.female_logits(&crops, geometry.frame)?;
                    for (job, logit) in jobs.iter().zip(logits) {
                        tracker.track_mut(job.track).observations[job.observation].female_logit = Some(logit);
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
    })
}
