//! The job: one input in, one cleaned output out.

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::{
    audio::{self, io::Span},
    cli::{AudioMode, RunArgs},
    probe::{self, MediaInfo, VideoStream},
    progress,
    render::{Assembly, AudioTrack},
    runtime::Runtime,
    tools::{ToolOverrides, Tools},
    vision::{
        self, Analysis, VisionConfig,
        decide::{self, BlurPlan, DecideConfig, Target, Verdict},
    },
    workdir::WorkDir,
};

pub fn run(args: &RunArgs) -> Result<()> {
    let started = Instant::now();
    let over = ToolOverrides { ffmpeg: args.tools.ffmpeg.clone(), encoder: args.tools.encoder.clone(), ytdlp: None };
    let tools = Tools::discover(args.tools.codec, &over)?;

    let source = PathBuf::from(&args.input);
    if !source.is_file() {
        bail!("input file not found: {}", source.display());
    }
    let info = probe::probe(&tools, &source)?;
    let span = Span { start: args.start, duration: args.duration };
    let length = span.length(info.duration);
    let output = args.output.clone().unwrap_or_else(|| default_output(&source, info.video.is_some()));
    if same_file(&source, &output) {
        bail!("the output would overwrite the input; choose another name with -o");
    }

    let work = WorkDir::create(args.keep_temp)?;
    let rt = Runtime::new(args.device)?;

    let audio = clean_audio(args, &tools, &rt, &source, &info, span, length, &work)?;
    let assembly = Assembly {
        tools: &tools,
        source: &source,
        has_video: info.video.is_some(),
        audio: &audio,
        span,
        quality: args.quality,
        output: &output,
    };

    let analysis = match &info.video {
        Some(video) if args.blur != Target::Nobody => Some(analyze(args, &tools, &rt, &source, video, span, length)?),
        _ => None,
    };
    let decide_cfg = DecideConfig {
        target: args.blur,
        shape: args.shape,
        threshold: args.threshold,
        ..DecideConfig::default()
    };
    let mut blurred = false;
    if let (Some(analysis), Some(video)) = (&analysis, &info.video) {
        let verdicts: Vec<Verdict> = analysis.tracks.iter().map(|t| decide::judge(t, &decide_cfg)).collect();
        let plan = BlurPlan::build(&analysis.tracks, &verdicts, analysis.frames, &analysis.cuts, &decide_cfg);
        summarise(analysis, &verdicts, &plan, video);
        if let Some(path) = &args.report {
            write_report(path, analysis, &verdicts, video)?;
        }
        if plan.blurred_frames() > 0 {
            let bar = progress::media_bar("blurring", length);
            let fps = video.fps.as_f64();
            assembly.with_blur(
                video,
                analysis.geometry.grid,
                plan.frame_count(),
                args.style,
                |frame, canvas| plan.paint(frame, canvas),
                |frame| progress::set_media(&bar, frame as f64 / fps),
            )?;
            bar.finish_and_clear();
            blurred = true;
        }
    }
    if !blurred {
        assembly.without_blur()?;
    }

    eprintln!("done in {:.1}s -> {}", started.elapsed().as_secs_f64(), output.display());
    if args.keep_temp {
        eprintln!("work files kept in {}", work.path().display());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn clean_audio(
    args: &RunArgs,
    tools: &Tools,
    rt: &Runtime,
    source: &Path,
    info: &MediaInfo,
    span: Span,
    length: f64,
    work: &WorkDir,
) -> Result<AudioTrack> {
    Ok(match (args.audio, &info.audio) {
        (AudioMode::Clean, Some(_)) => {
            let wav = work.file("voice.wav");
            let bar = progress::media_bar("removing music", length);
            let report = audio::remove_music(tools, rt, source, &wav, span, |s| progress::set_media(&bar, s))?;
            bar.finish_and_clear();
            eprintln!(
                "music removed: {:.0}s of audio in {:.1}s ({:.0}x realtime, {})",
                report.seconds,
                report.elapsed,
                report.seconds / report.elapsed.max(1e-3),
                device_name(rt)
            );
            AudioTrack::Clean(wav)
        }
        (AudioMode::Keep, Some(_)) => AudioTrack::Original,
        _ => AudioTrack::None,
    })
}

fn analyze(
    args: &RunArgs,
    tools: &Tools,
    rt: &Runtime,
    source: &Path,
    video: &VideoStream,
    span: Span,
    length: f64,
) -> Result<Analysis> {
    let cfg = VisionConfig {
        stride: args.stride.unwrap_or_else(|| default_stride(video.fps.as_f64())),
        // Blurring everyone needs no classification at all.
        classify: args.blur != Target::Everyone,
        ..VisionConfig::default()
    };
    let bar = progress::media_bar("finding people", length);
    let fps = video.fps.as_f64();
    let analysis = vision::analyze(tools, rt, source, video, span, &cfg, |frames| progress::set_media(&bar, frames as f64 / fps))?;
    bar.finish_and_clear();
    Ok(analysis)
}

/// Detect on every frame up to 30 fps; above that every other frame is plenty
/// (a person moves little in 1/60 s, and the gaps are interpolated).
fn default_stride(fps: f64) -> u32 {
    ((fps / 30.0).round() as u32).max(1)
}

fn device_name(rt: &Runtime) -> String {
    rt.device().map_or_else(|| "?".to_string(), |d| d.to_string())
}

fn summarise(analysis: &Analysis, verdicts: &[Verdict], plan: &BlurPlan, video: &VideoStream) {
    let seconds = analysis.frames as f64 / video.fps.as_f64();
    let blurred = verdicts.iter().filter(|v| v.blur).count();
    eprintln!(
        "people found: {} tracks in {} frames ({:.1}s, {:.0} fps, {:.1}x realtime); {} blurred across {} frames",
        analysis.tracks.len(),
        analysis.frames,
        analysis.elapsed,
        analysis.frames as f64 / analysis.elapsed.max(1e-3),
        seconds / analysis.elapsed.max(1e-3),
        blurred,
        plan.blurred_frames(),
    );
}

#[derive(Serialize)]
struct TrackReport {
    id: usize,
    first_frame: u32,
    last_frame: u32,
    start_seconds: f64,
    end_seconds: f64,
    sightings: usize,
    samples: usize,
    /// Probability that the person is a woman.
    female: Option<f32>,
    blurred: bool,
    /// Box at the middle sighting, `[x1, y1, x2, y2]` as fractions of the frame.
    typical_box: [f32; 4],
}

#[derive(Serialize)]
struct Report<'a> {
    frames: u32,
    fps: f64,
    cuts: &'a [u32],
    classified_crops: usize,
    analysis_seconds: f64,
    tracks: Vec<TrackReport>,
}

/// Writes a JSON account of every person found, for checking the decisions.
fn write_report(path: &Path, analysis: &Analysis, verdicts: &[Verdict], video: &VideoStream) -> Result<()> {
    let fps = video.fps.as_f64();
    let tracks = analysis
        .tracks
        .iter()
        .zip(verdicts)
        .enumerate()
        .filter_map(|(id, (track, verdict))| {
            let (first, last) = (track.observations.first()?, track.observations.last()?);
            Some(TrackReport {
                id,
                first_frame: first.frame,
                last_frame: last.frame,
                start_seconds: first.frame as f64 / fps,
                end_seconds: (last.frame + 1) as f64 / fps,
                sightings: track.observations.len(),
                samples: verdict.samples,
                female: verdict.female,
                blurred: verdict.blur,
                typical_box: track.observations[track.observations.len() / 2].bbox,
            })
        })
        .collect();
    let report = Report {
        frames: analysis.frames,
        fps,
        cuts: &analysis.cuts,
        classified_crops: analysis.classified,
        analysis_seconds: analysis.elapsed,
        tracks,
    };
    std::fs::write(path, serde_json::to_vec_pretty(&report)?).with_context(|| format!("cannot write {}", path.display()))
}

/// `<name>.halal.mp4` next to the input (`.m4a` when there is no picture).
fn default_output(source: &Path, has_video: bool) -> PathBuf {
    let stem = source.file_stem().unwrap_or_default().to_string_lossy();
    let ext = if has_video { "mp4" } else { "m4a" };
    source.with_file_name(format!("{stem}.halal.{ext}"))
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_output_sits_next_to_the_input() {
        assert_eq!(default_output(Path::new("clips/My Talk.mkv"), true), Path::new("clips/My Talk.halal.mp4"));
        assert_eq!(default_output(Path::new("song.mp3"), false), Path::new("song.halal.m4a"));
    }

    #[test]
    fn high_frame_rates_are_analysed_at_about_thirty() {
        assert_eq!(default_stride(24.0), 1);
        assert_eq!(default_stride(29.97), 1);
        assert_eq!(default_stride(50.0), 2);
        assert_eq!(default_stride(59.94), 2);
        assert_eq!(default_stride(120.0), 4);
    }
}
