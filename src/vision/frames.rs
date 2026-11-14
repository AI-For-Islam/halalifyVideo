//! Reading video frames: ffmpeg decodes, normalises the frame rate and scales;
//! this side only receives small RGB frames through a pipe.

use std::{
    io::Read,
    path::Path,
    process::{Child, ChildStdout, Stdio},
};

use anyhow::{Context, Result, bail};

use crate::{
    audio::io::Span,
    probe::{Rational, VideoStream},
    tools::Tools,
};

/// The sizes one video is handled at.
///
/// * `frame`: what ffmpeg sends for analysis (aspect preserved, even sides).
/// * `content`: the frame as the detector sees it, before padding.
/// * `input`: `content` padded up to multiples of 32, the detector's tensor.
/// * `grid`: the mask grid, a quarter of `content`; blur masks live here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub frame: (usize, usize),
    pub content: (usize, usize),
    pub input: (usize, usize),
    pub grid: (usize, usize),
}

impl Geometry {
    /// `detect_side` is the detector's long side; `detail` (>= 1.0) is how
    /// much larger than that the analysis frames are, which gives the person
    /// classifier sharper crops.
    pub fn new(width: u32, height: u32, detect_side: usize, detail: f32) -> Geometry {
        let (w, h) = (width.max(2) as f64, height.max(2) as f64);
        let scale = detect_side as f64 / w.max(h);
        // The detector works on whole 4-pixel cells (the mask grid).
        let cells = |v: f64| ((v * scale / 4.0).round() as usize).max(8);
        let grid = (cells(w), cells(h));
        let content = (grid.0 * 4, grid.1 * 4);
        let input = (content.0.div_ceil(32) * 32, content.1.div_ceil(32) * 32);
        let even = |v: usize| ((v as f32 * detail.max(1.0) / 2.0).round() as usize) * 2;
        let frame = (even(content.0), even(content.1));
        Geometry { frame, content, input, grid }
    }

    pub fn frame_bytes(&self) -> usize {
        self.frame.0 * self.frame.1 * 3
    }
}

/// The ffmpeg filter that both passes apply first, so that they see exactly
/// the same sequence of frames: a constant frame rate starting at time zero.
pub fn normalise_filter(fps: Rational) -> String {
    format!("fps={fps}:start_time=0")
}

/// Streams the video as packed RGB frames of `geometry.frame` size.
pub struct FrameReader {
    child: Child,
    stdout: ChildStdout,
    frame_bytes: usize,
}

impl FrameReader {
    pub fn open(tools: &Tools, input: &Path, video: &VideoStream, geometry: &Geometry, span: Span) -> Result<FrameReader> {
        let (w, h) = geometry.frame;
        let filter = format!("{},scale={w}:{h}:flags=bilinear", normalise_filter(video.fps));
        let mut cmd = tools.ffmpeg();
        cmd.args(span.input_args())
            .arg("-i")
            .arg(input)
            .args(["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", &filter])
            .args(["-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = cmd.spawn().context("cannot start ffmpeg to decode video")?;
        let stdout = child.stdout.take().expect("stdout is piped");
        Ok(FrameReader { child, stdout, frame_bytes: geometry.frame_bytes() })
    }

    /// Fills `frame` with the next frame. Returns `false` at the end of the video.
    pub fn read(&mut self, frame: &mut Vec<u8>) -> Result<bool> {
        frame.resize(self.frame_bytes, 0);
        let mut filled = 0;
        while filled < self.frame_bytes {
            match self.stdout.read(&mut frame[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e).context("reading decoded video"),
            }
        }
        match filled {
            0 => Ok(false),
            n if n == self.frame_bytes => Ok(true),
            n => bail!("video stream ended inside a frame ({n} of {} bytes)", self.frame_bytes),
        }
    }

    pub fn finish(mut self) -> Result<()> {
        drop(self.stdout);
        let status = self.child.wait()?;
        if !status.success() {
            bail!("ffmpeg failed while decoding video ({status})");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widescreen_geometry() {
        let g = Geometry::new(1920, 1080, 640, 1.5);
        assert_eq!(g.grid, (160, 90));
        assert_eq!(g.content, (640, 360));
        assert_eq!(g.input, (640, 384));
        assert_eq!(g.frame, (960, 540));
    }

    #[test]
    fn portrait_and_scope_geometry() {
        let g = Geometry::new(1080, 1920, 640, 1.0);
        assert_eq!((g.content, g.input, g.frame), ((360, 640), (384, 640), (360, 640)));
        // 2.40:1 cinema frame
        let g = Geometry::new(1920, 800, 640, 2.0);
        assert_eq!(g.content, (640, 268));
        assert_eq!(g.input, (640, 288));
        assert_eq!(g.frame, (1280, 536));
        assert_eq!(g.grid, (160, 67));
    }

    #[test]
    fn tiny_video_still_gets_a_usable_grid() {
        let g = Geometry::new(64, 16, 640, 1.0);
        assert!(g.grid.1 >= 8 && g.input.1 % 32 == 0);
    }
}
