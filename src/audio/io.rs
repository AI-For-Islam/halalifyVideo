//! Audio in and out: decoding any input to stereo float PCM through an ffmpeg
//! pipe, and writing a float WAV file.

use std::{
    fs::File,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
    process::{Child, ChildStdout, Stdio},
};

use anyhow::{Context, Result, bail};

use crate::tools::Tools;

/// A section of the input, in seconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct Span {
    pub start: Option<f64>,
    pub duration: Option<f64>,
}

impl Span {
    pub fn is_whole(&self) -> bool {
        self.start.is_none() && self.duration.is_none()
    }

    /// Seconds of media this span covers in a file `total` seconds long.
    pub fn length(&self, total: f64) -> f64 {
        let rest = (total - self.start.unwrap_or(0.0)).max(0.0);
        self.duration.map_or(rest, |d| d.min(rest))
    }

    /// Input-side ffmpeg arguments (`-ss` / `-t` before `-i`).
    pub fn input_args(&self) -> Vec<String> {
        let mut a = Vec::new();
        if let Some(s) = self.start {
            a.extend(["-ss".into(), format!("{s:.3}")]);
        }
        if let Some(d) = self.duration {
            a.extend(["-t".into(), format!("{d:.3}")]);
        }
        a
    }
}

/// Decodes the first audio stream of a file to interleaved stereo `f32`.
pub struct PcmReader {
    child: Child,
    stdout: ChildStdout,
    bytes: Vec<u8>,
}

impl PcmReader {
    pub fn open(tools: &Tools, input: &Path, sample_rate: u32, span: Span) -> Result<PcmReader> {
        let mut cmd = tools.ffmpeg();
        cmd.args(span.input_args())
            .arg("-i")
            .arg(input)
            .args(["-map", "0:a:0", "-vn", "-sn", "-dn"])
            // first_pts=0 pads or trims the start so sample 0 is time 0, which
            // keeps the audio aligned with video that also starts at time 0.
            .args(["-af", &format!("aresample={sample_rate}:async=1:first_pts=0")])
            .args(["-ac", "2", "-f", "f32le", "-acodec", "pcm_f32le", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = cmd.spawn().context("cannot start ffmpeg to decode audio")?;
        let stdout = child.stdout.take().expect("stdout is piped");
        Ok(PcmReader { child, stdout, bytes: Vec::new() })
    }

    /// Appends up to `max_frames` sample frames to the two planes and returns
    /// how many were appended; `0` means the stream has ended.
    pub fn read(&mut self, left: &mut Vec<f32>, right: &mut Vec<f32>, max_frames: usize) -> Result<usize> {
        const FRAME: usize = 8; // two f32 samples
        self.bytes.resize(max_frames * FRAME, 0);
        let mut filled = 0;
        while filled < self.bytes.len() {
            match self.stdout.read(&mut self.bytes[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e).context("reading decoded audio"),
            }
        }
        let frames = filled / FRAME;
        left.reserve(frames);
        right.reserve(frames);
        for f in self.bytes[..frames * FRAME].chunks_exact(FRAME) {
            left.push(f32::from_le_bytes([f[0], f[1], f[2], f[3]]));
            right.push(f32::from_le_bytes([f[4], f[5], f[6], f[7]]));
        }
        Ok(frames)
    }

    pub fn finish(mut self) -> Result<()> {
        drop(self.stdout);
        let status = self.child.wait()?;
        if !status.success() {
            bail!("ffmpeg failed while decoding audio ({status})");
        }
        Ok(())
    }
}

/// Writes stereo 32-bit float PCM as a WAV file. The header is patched with
/// the real length on `finish`, so the length need not be known up front.
pub struct WavWriter {
    file: BufWriter<File>,
    data_bytes: u64,
}

impl WavWriter {
    pub fn create(path: &Path, sample_rate: u32) -> Result<WavWriter> {
        let file = File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
        let mut file = BufWriter::with_capacity(1 << 20, file);
        const CHANNELS: u16 = 2;
        const BYTES: u16 = 4;
        let mut h = Vec::with_capacity(44);
        h.extend_from_slice(b"RIFF\0\0\0\0WAVEfmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        h.extend_from_slice(&CHANNELS.to_le_bytes());
        h.extend_from_slice(&sample_rate.to_le_bytes());
        h.extend_from_slice(&(sample_rate * (CHANNELS * BYTES) as u32).to_le_bytes());
        h.extend_from_slice(&(CHANNELS * BYTES).to_le_bytes());
        h.extend_from_slice(&(BYTES * 8).to_le_bytes());
        h.extend_from_slice(b"data\0\0\0\0");
        file.write_all(&h)?;
        Ok(WavWriter { file, data_bytes: 0 })
    }

    pub fn write(&mut self, left: &[f32], right: &[f32]) -> Result<()> {
        for (l, r) in left.iter().zip(right) {
            self.file.write_all(&l.clamp(-1.0, 1.0).to_le_bytes())?;
            self.file.write_all(&r.clamp(-1.0, 1.0).to_le_bytes())?;
        }
        self.data_bytes += 8 * left.len().min(right.len()) as u64;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        // WAV sizes are 32-bit; ffmpeg ignores them when they are saturated.
        let riff = (self.data_bytes + 36).min(u32::MAX as u64) as u32;
        let data = self.data_bytes.min(u32::MAX as u64) as u32;
        self.file.seek(SeekFrom::Start(4))?;
        self.file.write_all(&riff.to_le_bytes())?;
        self.file.seek(SeekFrom::Start(40))?;
        self.file.write_all(&data.to_le_bytes())?;
        self.file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_describes_the_data() {
        let path = std::env::temp_dir().join(format!("halalify-wav-test-{}.wav", std::process::id()));
        let mut w = WavWriter::create(&path, 44_100).unwrap();
        w.write(&[0.25, -0.5, 2.0], &[0.0, 0.5, -2.0]).unwrap();
        w.finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(bytes.len(), 44 + 3 * 8);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 36 + 24);
        assert_eq!(u16::from_le_bytes(bytes[20..22].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 44_100);
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 24);
        // Out-of-range samples are clamped rather than wrapped.
        assert_eq!(f32::from_le_bytes(bytes[60..64].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(bytes[64..68].try_into().unwrap()), -1.0);
    }

    #[test]
    fn span_becomes_input_seek_arguments() {
        assert!(Span::default().input_args().is_empty());
        let s = Span { start: Some(12.5), duration: Some(3.0) };
        assert_eq!(s.input_args(), ["-ss", "12.500", "-t", "3.000"]);
        assert!(!s.is_whole());
        assert_eq!(s.length(60.0), 3.0);
        assert_eq!(s.length(14.0), 1.5);
        assert_eq!(Span::default().length(60.0), 60.0);
    }
}
