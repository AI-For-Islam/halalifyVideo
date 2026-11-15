//! Getting a video from a link, with yt-dlp.

use std::{
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};

use crate::{progress, tools::Tools};

/// yt-dlp prints the final file path on a line starting with this marker.
const MARKER: &str = "HALALIFY_SAVED:";

pub fn is_url(input: &str) -> bool {
    let lower = input.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// Downloads the video at `url` into `dir` and returns the file.
///
/// The best stream no taller than `max_height` is taken (for a vertical video
/// the limit applies to its short side), preferring H.264 with AAC so that
/// the result decodes fast and plays everywhere.
pub fn download(tools: &Tools, url: &str, dir: &Path, max_height: u32) -> Result<PathBuf> {
    let ytdlp = tools.ytdlp.as_ref().context(
        "yt-dlp is needed to download from a link but was not found. Install it (for example \
         `winget install yt-dlp.yt-dlp`) or set HALALIFY_YTDLP to its full path.",
    )?;
    let mut cmd = std::process::Command::new(ytdlp);
    cmd.args(["--no-playlist", "--no-simulate", "--newline", "--progress", "--no-warnings"])
        .args(["-S", &format!("res:{max_height},vcodec:h264,acodec:aac")])
        .args(["-f", "bv*+ba/b", "--merge-output-format", "mp4"])
        .args(["--print", &format!("after_move:{MARKER}%(filepath)s")])
        // Without this, yt-dlp writes to a pipe in the console's legacy
        // encoding and silently drops what that cannot carry: the path it
        // prints for a title with a colon (saved as a full-width one), or in
        // Arabic, would then not be the path of the file.
        .args(["--encoding", "utf-8"])
        // A short, byte-limited title keeps the path valid on every file system.
        .arg("-o")
        .arg(dir.join("%(title).80B [%(id)s].%(ext)s"));
    if let Some(folder) = tools.ffmpeg.parent() {
        cmd.arg("--ffmpeg-location").arg(folder);
    }
    if let Some((name, path)) = &tools.js_runtime {
        // YouTube only lists all of its formats to a client that can run JavaScript.
        cmd.arg("--js-runtimes").arg(format!("{name}:{}", path.display()));
    }
    cmd.arg(url).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());

    let spinner = ProgressBar::new_spinner().with_style(ProgressStyle::with_template("{prefix:>14} {spinner} {msg}").expect("valid template"));
    spinner.set_prefix("downloading");
    spinner.enable_steady_tick(Duration::from_millis(120));

    let mut child = cmd.spawn().with_context(|| format!("cannot start {}", ytdlp.display()))?;
    let stderr = child.stderr.take().expect("stderr is piped");
    let errors = std::thread::spawn(move || {
        BufReader::new(stderr).lines().map_while(Result::ok).filter(|l| l.contains("ERROR")).collect::<Vec<_>>().join("\n")
    });
    let mut saved = None;
    for line in BufReader::new(child.stdout.take().expect("stdout is piped")).lines().map_while(Result::ok) {
        match parse_line(&line) {
            Line::Saved(path) => saved = Some(path),
            Line::Progress(text) => {
                progress::fetching(&text);
                spinner.set_message(text);
            }
            Line::Other => {}
        }
    }
    let status = child.wait()?;
    let errors = errors.join().unwrap_or_default();
    spinner.finish_and_clear();
    if !status.success() {
        bail!("yt-dlp could not download {url}: {}", if errors.is_empty() { status.to_string() } else { errors });
    }
    // Should the printed path still not match the file, the job folder holds
    // nothing but the download.
    saved.filter(|p| p.is_file()).or_else(|| only_file(dir)).context("yt-dlp finished, but the file it downloaded was not found")
}

/// The one finished file in `dir`, if there is exactly one.
fn only_file(dir: &Path) -> Option<PathBuf> {
    let unfinished = |p: &Path| p.extension().and_then(|e| e.to_str()).is_some_and(|e| matches!(e, "part" | "ytdl" | "tmp"));
    let mut files = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.is_file() && !unfinished(p));
    let first = files.next()?;
    files.next().is_none().then_some(first)
}

enum Line {
    Saved(PathBuf),
    Progress(String),
    Other,
}

fn parse_line(line: &str) -> Line {
    let line = line.trim();
    if let Some(path) = line.strip_prefix(MARKER) {
        Line::Saved(PathBuf::from(path))
    } else if let Some(progress) = line.strip_prefix("[download]") {
        Line::Progress(progress.trim().to_string())
    } else {
        Line::Other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_are_recognised() {
        assert!(is_url("https://youtu.be/abc"));
        assert!(is_url("HTTP://example.com/v.mp4"));
        assert!(!is_url("C:\\videos\\https.mp4"));
        assert!(!is_url("video.mp4"));
    }

    #[test]
    fn the_download_is_found_without_its_printed_name() {
        let dir = std::env::temp_dir().join(format!("halalify-fetch-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(only_file(&dir), None);
        let video = dir.join("Engineer\u{ff1a} tired [abc].mp4");
        std::fs::write(&video, b"x").unwrap();
        std::fs::write(dir.join("leftover.part"), b"x").unwrap();
        assert_eq!(only_file(&dir), Some(video));
        std::fs::write(dir.join("second.mp4"), b"x").unwrap();
        assert_eq!(only_file(&dir), None);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn yt_dlp_output_is_understood() {
        assert!(matches!(parse_line("HALALIFY_SAVED:C:\\tmp\\My talk [abc].mp4"), Line::Saved(p) if p == Path::new("C:\\tmp\\My talk [abc].mp4")));
        assert!(matches!(parse_line("[download]  42.0% of 10.00MiB at 2.00MiB/s ETA 00:03"), Line::Progress(t) if t.starts_with("42.0%")));
        assert!(matches!(parse_line("[Merger] Merging formats"), Line::Other));
    }
}
