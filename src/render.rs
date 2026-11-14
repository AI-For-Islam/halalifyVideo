//! Final assembly: putting the processed video and audio into the output file.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, bail};

use crate::{
    audio::io::Span,
    tools::{self, Tools},
};

/// What the output's soundtrack is made from.
#[derive(Clone, Debug)]
pub enum AudioTrack {
    /// A cleaned WAV produced by this run, already cut to the requested span.
    Clean(PathBuf),
    /// The input's own audio, untouched.
    Original,
    None,
}

/// Bitrate for the re-encoded soundtrack; transparent for voice-only audio.
const AAC_BITRATE: &str = "192k";

pub struct Assembly<'a> {
    pub tools: &'a Tools,
    pub source: &'a Path,
    pub has_video: bool,
    pub audio: &'a AudioTrack,
    pub span: Span,
    /// CRF-like quality for re-encoded video.
    pub quality: u8,
    pub output: &'a Path,
}

impl Assembly<'_> {
    /// Writes the output without changing any frame.
    ///
    /// The video stream is copied bit-for-bit unless a span was requested:
    /// a copied stream can only be cut at keyframes, which would leave the
    /// picture out of step with the precisely cut audio.
    pub fn without_blur(&self) -> Result<()> {
        let mut cmd = self.tools.ffmpeg();
        cmd.args(self.span.input_args()).arg("-i").arg(self.source);
        let wav_index = self.add_audio_input(&mut cmd, 1);
        if self.has_video {
            cmd.args(["-map", "0:v:0"]);
            if self.span.is_whole() {
                cmd.args(["-c:v", "copy"]);
            } else {
                cmd.args(tools::encoder_args(&self.tools.encoder, self.quality)).args(["-pix_fmt", "yuv420p"]);
            }
        }
        self.finish(&mut cmd, wav_index);
        run(cmd, "writing the output file")
    }

    /// Adds the cleaned WAV as ffmpeg input number `index` when there is one.
    /// All inputs must be declared before any output option.
    fn add_audio_input(&self, cmd: &mut Command, index: usize) -> usize {
        if let AudioTrack::Clean(wav) = self.audio {
            cmd.arg("-i").arg(wav);
        }
        index
    }

    /// Adds the soundtrack mapping, container options and the output file.
    fn finish(&self, cmd: &mut Command, wav_index: usize) {
        match self.audio {
            AudioTrack::Clean(_) => {
                cmd.args(["-map", &format!("{wav_index}:a:0"), "-c:a", "aac", "-b:a", AAC_BITRATE]);
            }
            AudioTrack::Original if self.span.is_whole() => {
                cmd.args(["-map", "0:a:0", "-c:a", "copy"]);
            }
            AudioTrack::Original => {
                cmd.args(["-map", "0:a:0", "-c:a", "aac", "-b:a", AAC_BITRATE]);
            }
            AudioTrack::None => {
                cmd.arg("-an");
            }
        }
        cmd.args(["-map_metadata", "0", "-movflags", "+faststart"]).arg(self.output);
    }
}

fn run(mut cmd: Command, what: &str) -> Result<()> {
    let out = cmd.output().with_context(|| format!("cannot start ffmpeg for {what}"))?;
    if !out.status.success() {
        bail!("ffmpeg failed while {what}: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(())
}
