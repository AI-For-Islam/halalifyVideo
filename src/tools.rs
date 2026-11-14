//! Locating and probing the external tools: ffmpeg, ffprobe and yt-dlp.
//!
//! ffmpeg does all demuxing, decoding, filtering and encoding, so picking a
//! build that works on this machine matters. Listing `h264_nvenc` among a
//! build's encoders says nothing about whether the installed GPU driver accepts
//! it (a new ffmpeg with an older driver fails at open time), so every encoder
//! is test-driven with a few frames before it is chosen. Results are cached.

use std::{
    collections::BTreeMap,
    ffi::OsStr,
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    time::{Duration, Instant, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::paths;

const EXE: &str = if cfg!(windows) { ".exe" } else { "" };

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Codec {
    H264,
    Hevc,
    Av1,
}

impl Codec {
    /// Encoders to try, fastest first. The last entry of each list is software.
    pub fn encoders(self) -> &'static [&'static str] {
        match self {
            Codec::H264 => &["h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox", "libx264"],
            Codec::Hevc => &["hevc_nvenc", "hevc_qsv", "hevc_amf", "hevc_videotoolbox", "libx265"],
            Codec::Av1 => &["av1_nvenc", "av1_qsv", "av1_amf", "libsvtav1"],
        }
    }
}

/// Output-side ffmpeg arguments for `encoder` at a CRF-like `quality`
/// (lower is better; 18 is near transparent, 23 is small).
pub fn encoder_args(encoder: &str, quality: u8) -> Vec<String> {
    let q = quality.to_string();
    let mut a: Vec<&str> = vec!["-c:v", encoder];
    let owned: Vec<String>;
    if encoder.ends_with("_nvenc") {
        a.extend(["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", &q, "-b:v", "0"]);
    } else if encoder.ends_with("_qsv") {
        a.extend(["-preset", "faster", "-global_quality", &q]);
    } else if encoder.ends_with("_amf") {
        a.extend(["-quality", "balanced", "-rc", "cqp", "-qp_i", &q, "-qp_p", &q, "-qp_b", &q]);
    } else if encoder.ends_with("_videotoolbox") {
        owned = vec![(100u8.saturating_sub(quality.saturating_mul(2))).to_string()];
        a.extend(["-q:v", &owned[0]]);
    } else if encoder == "libx264" {
        a.extend(["-preset", "veryfast", "-crf", &q]);
    } else if encoder == "libx265" {
        owned = vec![(quality + 3).to_string()];
        a.extend(["-preset", "fast", "-crf", &owned[0], "-x265-params", "log-level=error"]);
    } else if encoder == "libsvtav1" {
        owned = vec![(quality + 9).to_string()];
        a.extend(["-preset", "8", "-crf", &owned[0]]);
    }
    let mut out: Vec<String> = a.into_iter().map(String::from).collect();
    if encoder.starts_with("hevc") || encoder == "libx265" {
        out.extend(["-tag:v".into(), "hvc1".into()]);
    }
    out
}

#[derive(Clone, Debug)]
pub struct Tools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    /// Working encoder of the requested codec, chosen by test-driving.
    pub encoder: String,
    pub ytdlp: Option<PathBuf>,
    /// A JavaScript runtime for yt-dlp (YouTube needs one for the full format list).
    pub js_runtime: Option<(String, PathBuf)>,
}

#[derive(Default, Clone, Debug)]
pub struct ToolOverrides {
    pub ffmpeg: Option<PathBuf>,
    pub encoder: Option<String>,
    pub ytdlp: Option<PathBuf>,
}

impl Tools {
    pub fn discover(codec: Codec, over: &ToolOverrides) -> Result<Tools> {
        let ffmpegs = match &over.ffmpeg {
            Some(p) => vec![p.clone()],
            None => find_all("ffmpeg", "HALALIFY_FFMPEG"),
        };
        if ffmpegs.is_empty() {
            bail!(
                "ffmpeg was not found. Install it (for example `winget install Gyan.FFmpeg`), \
                 put ffmpeg{EXE} in {} or set HALALIFY_FFMPEG to its full path.",
                paths::bin_dir().display()
            );
        }

        let mut cache = ProbeCache::load();
        let wanted: Vec<&str> = match &over.encoder {
            Some(e) => vec![e.as_str()],
            None => codec.encoders().to_vec(),
        };
        // Best (ffmpeg, encoder) pair: lowest encoder rank wins, earlier ffmpeg breaks ties.
        let mut best: Option<(usize, PathBuf, String)> = None;
        for ff in &ffmpegs {
            for (rank, enc) in wanted.iter().enumerate() {
                if best.as_ref().is_some_and(|b| b.0 <= rank) {
                    break;
                }
                if cache.encoder_works(ff, enc) {
                    best = Some((rank, ff.clone(), enc.to_string()));
                    break;
                }
            }
            if best.as_ref().is_some_and(|b| b.0 == 0) {
                break;
            }
        }
        cache.save();
        let Some((_, ffmpeg, encoder)) = best else {
            bail!(
                "no working video encoder among {:?} in {}",
                wanted,
                ffmpegs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
            );
        };

        let ffprobe = sibling(&ffmpeg, "ffprobe")
            .or_else(|| find_all("ffprobe", "HALALIFY_FFPROBE").into_iter().next())
            .context("ffprobe was not found next to ffmpeg or on PATH")?;

        let ytdlp = over.ytdlp.clone().or_else(|| find_all("yt-dlp", "HALALIFY_YTDLP").into_iter().next());
        let js_runtime = ["deno", "node", "bun"]
            .iter()
            .find_map(|n| find_all(n, "").into_iter().next().map(|p| (n.to_string(), p)));

        Ok(Tools { ffmpeg, ffprobe, encoder, ytdlp, js_runtime })
    }

    /// A fresh ffmpeg command with the quiet, non-interactive defaults every call wants.
    pub fn ffmpeg(&self) -> Command {
        let mut c = Command::new(&self.ffmpeg);
        c.args(["-hide_banner", "-nostdin", "-loglevel", "error", "-y"]);
        c
    }
}

/// Every `name` executable found, in priority order, without duplicates.
pub fn find_all(name: &str, env_var: &str) -> Vec<PathBuf> {
    let file = format!("{name}{EXE}");
    let mut found: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if p.is_file() {
            let p = p.canonicalize().map(strip_verbatim).unwrap_or(p);
            if !found.contains(&p) {
                found.push(p);
            }
        }
    };
    if !env_var.is_empty() {
        if let Some(p) = std::env::var_os(env_var).filter(|p| !p.is_empty()) {
            push(PathBuf::from(p));
        }
    }
    for dir in search_dirs() {
        push(dir.join(&file));
    }
    found
}

fn sibling(tool: &Path, name: &str) -> Option<PathBuf> {
    let p = tool.parent()?.join(format!("{name}{EXE}"));
    p.is_file().then_some(p)
}

/// `canonicalize` on Windows returns `\\?\C:\...`, which ffmpeg and yt-dlp print
/// and sometimes mishandle. Plain drive paths are safe to un-prefix.
fn strip_verbatim(p: PathBuf) -> PathBuf {
    match p.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(s) if s.as_bytes().get(1) == Some(&b':') => PathBuf::from(s),
        _ => p,
    }
}

fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![paths::bin_dir()];
    if let Some(d) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        dirs.push(d.join("bin"));
        dirs.push(d);
    }
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    // Where the common installers and package managers put these tools.
    if cfg!(windows) {
        let env = |k: &str| std::env::var_os(k).map(PathBuf::from);
        if let Some(drive) = env("SystemDrive") {
            dirs.push(drive.join(r"\ffmpeg\bin"));
        }
        if let Some(pf) = env("ProgramFiles") {
            dirs.push(pf.join("ffmpeg").join("bin"));
        }
        if let Some(l) = env("LOCALAPPDATA") {
            dirs.push(l.join("Microsoft").join("WinGet").join("Links"));
            dirs.extend(subdirs(&l.join("Programs").join("Python")).into_iter().map(|d| d.join("Scripts")));
        }
        if let Some(a) = env("APPDATA") {
            dirs.extend(subdirs(&a.join("Python")).into_iter().map(|d| d.join("Scripts")));
        }
        if let Some(u) = env("USERPROFILE") {
            dirs.push(u.join("scoop").join("shims"));
        }
        if let Some(p) = env("ProgramData") {
            dirs.push(p.join("chocolatey").join("bin"));
        }
    }
    dirs
}

fn subdirs(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut v: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    v.sort();
    v.reverse(); // newest version directory first
    v
}

/// Runs `cmd` to completion, killing it if it outlives `timeout`.
/// Returns `None` on timeout.
pub fn run_with_timeout(cmd: &mut Command, timeout: Duration) -> Result<Option<ExitStatus>> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("cannot start {:?}", cmd.get_program()))?;
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(15));
    }
}

/// Encodes a handful of frames to see whether `encoder` really works here.
fn test_drive(ffmpeg: &Path, encoder: &str) -> bool {
    let mut c = Command::new(ffmpeg);
    c.args(["-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg("color=c=black:s=640x360:r=30")
        .args(["-frames:v", "5", "-pix_fmt", "yuv420p", "-c:v", encoder, "-f", "null", "-"]);
    matches!(run_with_timeout(&mut c, Duration::from_secs(20)), Ok(Some(s)) if s.success())
}

/// Remembers encoder test-drives per ffmpeg binary (keyed by path, size and
/// modification time, so replacing the binary invalidates its entries).
#[derive(Default, Serialize, Deserialize)]
struct ProbeCache {
    encoders: BTreeMap<String, bool>,
    #[serde(skip)]
    dirty: bool,
}

impl ProbeCache {
    fn file() -> PathBuf {
        paths::cache_dir().join("tools.json")
    }

    fn load() -> ProbeCache {
        std::fs::read(Self::file())
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    fn save(&self) {
        if !self.dirty {
            return;
        }
        if paths::ensure(paths::cache_dir()).is_ok() {
            if let Ok(json) = serde_json::to_vec_pretty(self) {
                let _ = std::fs::write(Self::file(), json);
            }
        }
    }

    fn encoder_works(&mut self, ffmpeg: &Path, encoder: &str) -> bool {
        let meta = std::fs::metadata(ffmpeg).ok();
        let stamp = meta
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        let key = format!("{}|{}|{}|{}", ffmpeg.display(), meta.map_or(0, |m| m.len()), stamp, encoder);
        if let Some(&ok) = self.encoders.get(&key) {
            return ok;
        }
        let ok = test_drive(ffmpeg, encoder);
        self.encoders.insert(key, ok);
        self.dirty = true;
        ok
    }
}

/// Runs a command and returns its stdout, failing with stderr in the message.
pub fn capture<I, S>(program: &Path, args: I) -> Result<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let out = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("cannot start {}", program.display()))?;
    if !out.status.success() {
        bail!(
            "{} failed ({}): {}",
            program.file_name().unwrap_or_default().to_string_lossy(),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvenc_args_use_constant_quality() {
        let a = encoder_args("h264_nvenc", 21);
        assert!(a.windows(2).any(|w| w == ["-cq", "21"]));
        assert!(!a.contains(&"-tag:v".to_string()));
    }

    #[test]
    fn hevc_is_tagged_for_apple_players() {
        for enc in ["hevc_nvenc", "libx265", "hevc_qsv"] {
            let a = encoder_args(enc, 21);
            assert!(a.windows(2).any(|w| w == ["-tag:v", "hvc1"]), "{enc}");
        }
    }

    #[test]
    fn every_codec_ends_with_a_software_encoder() {
        for c in [Codec::H264, Codec::Hevc, Codec::Av1] {
            assert!(c.encoders().last().unwrap().starts_with("lib"));
        }
    }

    #[test]
    fn verbatim_prefix_is_removed_only_for_drive_paths() {
        assert_eq!(strip_verbatim(PathBuf::from(r"\\?\C:\a\b.exe")), PathBuf::from(r"C:\a\b.exe"));
        assert_eq!(strip_verbatim(PathBuf::from(r"\\?\UNC\srv\a")), PathBuf::from(r"\\?\UNC\srv\a"));
    }
}
