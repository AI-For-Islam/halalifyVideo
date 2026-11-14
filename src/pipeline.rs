//! The job: one input in, one cleaned output out.

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Result, bail};

use crate::{
    audio::{self, io::Span},
    cli::{AudioMode, RunArgs},
    probe, progress,
    render::{Assembly, AudioTrack},
    runtime::Runtime,
    tools::{ToolOverrides, Tools},
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

    let audio = match (args.audio, &info.audio) {
        (AudioMode::Clean, Some(_)) => {
            let wav = work.file("voice.wav");
            let bar = progress::media_bar("removing music", length);
            let report = audio::remove_music(&tools, &rt, &source, &wav, span, |s| progress::set_media(&bar, s))?;
            bar.finish_and_clear();
            eprintln!(
                "music removed: {:.0}s of audio in {:.1}s ({:.0}x realtime, {})",
                report.seconds,
                report.elapsed,
                report.seconds / report.elapsed.max(1e-3),
                rt.device().map_or("?".into(), |d| d.to_string())
            );
            AudioTrack::Clean(wav)
        }
        (AudioMode::Keep, Some(_)) => AudioTrack::Original,
        _ => AudioTrack::None,
    };

    Assembly {
        tools: &tools,
        source: &source,
        has_video: info.video.is_some(),
        audio: &audio,
        span,
        quality: args.quality,
        output: &output,
    }
    .without_blur()?;

    eprintln!("done in {:.1}s -> {}", started.elapsed().as_secs_f64(), output.display());
    if args.keep_temp {
        eprintln!("work files kept in {}", work.path().display());
    }
    Ok(())
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
}
