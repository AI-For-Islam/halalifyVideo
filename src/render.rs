//! Final assembly: putting the processed video and audio into the output file.
//!
//! When something must be blurred, ffmpeg does all the pixel work. This
//! program only streams it a tiny greyscale mask per frame (the mask grid,
//! about 160x90); ffmpeg scales the mask up, blurs a copy of the picture and
//! merges the two. Full-resolution frames never pass through this process,
//! and the picture stays in its native YUV format from decoder to encoder.

use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::{Context, Result, bail};

use crate::{
    audio::io::Span,
    probe::VideoStream,
    tools::{self, Tools},
    vision::{frames::normalise_filter, mask::Canvas},
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

/// How a blurred area looks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Style {
    /// A heavy, smooth blur.
    Blur,
    /// Large mosaic blocks.
    Pixelate,
    /// Solid black.
    Black,
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
    /// Upper limit for re-encoded video, in bits per second.
    pub max_bitrate: Option<u64>,
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
                cmd.args(tools::encoder_args(&self.tools.encoder, self.quality, self.max_bitrate)).args(["-pix_fmt", "yuv420p"]);
            }
        }
        self.finish(&mut cmd, wav_index);
        let out = cmd.output().context("cannot start ffmpeg to write the output file")?;
        if !out.status.success() {
            bail!("ffmpeg failed while writing the output file: {}", String::from_utf8_lossy(&out.stderr).trim());
        }
        Ok(())
    }

    /// Re-encodes the video with the masked areas obscured.
    ///
    /// `paint(frame, canvas)` must draw the mask of each frame, `frames` in
    /// total: exactly the frames the analysis pass saw, in the same order.
    /// `progress` receives the number of masks consumed so far.
    pub fn with_blur(
        &self,
        video: &VideoStream,
        grid: (usize, usize),
        frames: u32,
        style: Style,
        mut paint: impl FnMut(u32, &mut Canvas),
        mut progress: impl FnMut(u32),
    ) -> Result<()> {
        let mut cmd = self.tools.ffmpeg();
        cmd.args(self.span.input_args()).arg("-i").arg(self.source);
        cmd.args(["-f", "rawvideo", "-pix_fmt", "gray", "-video_size", &format!("{}x{}", grid.0, grid.1)])
            .args(["-framerate", &video.fps.to_string(), "-i", "pipe:0"]);
        let wav_index = self.add_audio_input(&mut cmd, 2);
        cmd.args(["-filter_complex", &filter_graph(video, style), "-map", "[out]"])
            .args(tools::encoder_args(&self.tools.encoder, self.quality, self.max_bitrate));
        self.finish(&mut cmd, wav_index);

        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .context("cannot start ffmpeg to render the video")?;
        let mut stdin = child.stdin.take().expect("stdin is piped");
        let mut stderr = child.stderr.take().expect("stderr is piped");
        // Drain stderr on the side so a chatty ffmpeg can never block on it.
        let errors = std::thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        });

        let mut canvas = Canvas::new(grid.0, grid.1);
        for frame in 0..frames {
            paint(frame, &mut canvas);
            // A failed write means ffmpeg has stopped reading: either it has
            // every frame it needs or it failed. Its exit status tells which.
            if stdin.write_all(&canvas.data).is_err() {
                break;
            }
            if frame % 16 == 0 {
                progress(frame);
            }
        }
        drop(stdin);
        let status = child.wait()?;
        let errors = errors.join().unwrap_or_default();
        if !status.success() {
            bail!("ffmpeg failed while rendering the video: {}", errors.trim());
        }
        progress(frames);
        Ok(())
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

/// The ffmpeg filter graph for the blur pass.
///
/// Input 0 is the source video and input 1 the mask stream. Both are stamped
/// with their frame number as timestamp, so mask `n` always meets frame `n`
/// no matter how the source's own timestamps look.
fn filter_graph(video: &VideoStream, style: Style) -> String {
    // 4:2:0 needs even sides; an odd source loses its last row or column.
    let (w, h) = (video.width & !1, video.height & !1);
    // The picture is obscured at a resolution tied to its height, so the
    // effect looks the same on a 480p clip and a 4K film.
    let obscured = match style {
        Style::Blur => {
            // Shrink, average, enlarge: a strong smooth blur for the cost of
            // two scalings. (ffmpeg's gblur corrupts chroma on frames this
            // small, so a box average is used instead.)
            let sh = (h / 20).max(8) & !1;
            let sw = ((w as u64 * sh as u64 / h as u64) as u32).max(8) & !1;
            format!("scale={sw}:{sh}:flags=area,avgblur=2,scale={w}:{h}:flags=bilinear")
        }
        Style::Pixelate => {
            let sh = (h / 40).max(4);
            let sw = ((w as u64 * sh as u64 / h as u64) as u32).max(4);
            format!("scale={sw}:{sh}:flags=area,scale={w}:{h}:flags=neighbor")
        }
        Style::Black => "lutyuv=y=16:u=128:v=128".to_string(),
    };
    // maskedmerge wants the mask in the picture's own format, plane by plane:
    // full size for luma, half size for the two chroma planes.
    format!(
        "[0:v]{norm},scale={w}:{h}:flags=bicubic,format=yuv420p,setpts=N,split[base][copy];\
         [copy]{obscured}[hidden];\
         [1:v]setpts=N,format=gray,split[m1][m2];\
         [m1]scale={w}:{h}:flags=bilinear[my];\
         [m2]scale={cw}:{ch}:flags=bilinear,split[mu][mv];\
         [my][mu][mv]mergeplanes=0x001020:yuv420p[mask];\
         [base][hidden][mask]maskedmerge[out]",
        norm = normalise_filter(video.fps),
        cw = w / 2,
        ch = h / 2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::Rational;

    fn video(width: u32, height: u32) -> VideoStream {
        VideoStream {
            width,
            height,
            fps: Rational { num: 30000, den: 1001 },
            codec: "h264".into(),
            pix_fmt: "yuv420p".into(),
            variable_rate: false,
            bit_rate: None,
        }
    }

    #[test]
    fn graph_normalises_time_and_merges_through_the_mask() {
        let g = filter_graph(&video(1920, 1080), Style::Blur);
        assert!(g.starts_with("[0:v]fps=30000/1001:start_time=0,scale=1920:1080"));
        assert!(g.contains("scale=96:54:flags=area,avgblur=2,scale=1920:1080:flags=bilinear"));
        assert!(g.contains("[m2]scale=960:540"));
        assert!(g.ends_with("[base][hidden][mask]maskedmerge[out]"));
        assert_eq!(g.matches("setpts=N").count(), 2);
    }

    #[test]
    fn odd_sizes_are_made_even() {
        let g = filter_graph(&video(853, 481), Style::Pixelate);
        assert!(g.contains("scale=852:480:flags=bicubic"));
        assert!(g.contains("flags=neighbor"));
        assert!(g.contains("[m2]scale=426:240"));
    }

    #[test]
    fn black_style_needs_no_scaling() {
        let g = filter_graph(&video(1280, 720), Style::Black);
        assert!(g.contains("[copy]lutyuv=y=16:u=128:v=128[hidden]"));
    }
}
