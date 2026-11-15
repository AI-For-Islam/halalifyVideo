//! Running the `halalify` program and passing on what it reports.
//!
//! One job runs at a time. Its standard output is read line by line: a line
//! holding a JSON object with an `event` field is forwarded to the screens as
//! it is, and every other line (and all of standard error) as a `log` event.
//! When the process ends, an `exit` event closes the job.

use std::{
    io::{BufRead, BufReader, Read},
    path::PathBuf,
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use serde_json::{Value, json};
use tauri::{AppHandle, Emitter};

/// Name of the event every job message is sent under.
const CHANNEL: &str = "engine";

struct Job {
    pid: u32,
    cancelled: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct Engine {
    job: Arc<Mutex<Option<Job>>>,
}

impl Engine {
    /// The `halalify doctor` report.
    pub fn status(&self) -> Result<Value, String> {
        let exe = locate()?;
        let mut cmd = Command::new(&exe);
        cmd.args(["doctor", "--progress", "json"]).stdin(Stdio::null());
        hide_window(&mut cmd);
        let output = cmd.output().map_err(|e| format!("cannot start {}: {e}", exe.display()))?;
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .rev()
            .find_map(event)
            .ok_or_else(|| format!("{} gave no status: {}", exe.display(), String::from_utf8_lossy(&output.stderr).trim()))
    }

    /// Starts `halalify` with `args`, in `folder` when given.
    pub fn start(&self, app: AppHandle, args: Vec<String>, folder: Option<PathBuf>) -> Result<(), String> {
        let mut cmd = Command::new(locate()?);
        cmd.args(args);
        if let Some(folder) = folder {
            cmd.current_dir(folder);
        }
        self.spawn(app, cmd)
    }

    /// Starts `winget install` for `packages`.
    pub fn start_winget(&self, app: AppHandle, packages: &[&str]) -> Result<(), String> {
        let mut cmd = Command::new("winget");
        cmd.args(["install", "--accept-source-agreements", "--accept-package-agreements", "--disable-interactivity"]).args(packages);
        self.spawn(app, cmd).map_err(|e| format!("winget, Windows' package manager, could not be started ({e}). Install ffmpeg and yt-dlp by hand; the README shows how."))
    }

    fn spawn(&self, app: AppHandle, mut cmd: Command) -> Result<(), String> {
        let mut slot = self.job.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_some() {
            return Err("something is already running".into());
        }
        cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        hide_window(&mut cmd);
        own_process_group(&mut cmd);
        let mut child = cmd.spawn().map_err(|e| format!("cannot start {}: {e}", cmd.get_program().to_string_lossy()))?;
        let cancelled = Arc::new(AtomicBool::new(false));
        *slot = Some(Job { pid: child.id(), cancelled: cancelled.clone() });
        drop(slot);

        let stdout = child.stdout.take().expect("stdout is piped");
        let stderr = child.stderr.take().expect("stderr is piped");
        let log_app = app.clone();
        let errors = std::thread::spawn(move || {
            for line in lines(stderr) {
                let _ = log_app.emit(CHANNEL, json!({ "event": "log", "text": line }));
            }
        });
        let job = self.job.clone();
        std::thread::spawn(move || {
            let mut scratch = None;
            for line in lines(stdout) {
                let message = event(&line).unwrap_or_else(|| json!({ "event": "log", "text": line }));
                if message["event"] == "work" {
                    scratch = message["dir"].as_str().map(PathBuf::from);
                }
                let _ = app.emit(CHANNEL, message);
            }
            let code = child.wait().ok().and_then(|s| s.code());
            let _ = errors.join();
            // A job that was stopped had no chance to remove its scratch folder.
            if let Some(dir) = scratch.filter(|d| is_scratch(d)) {
                let _ = std::fs::remove_dir_all(dir);
            }
            *job.lock().unwrap_or_else(|e| e.into_inner()) = None;
            let _ = app.emit(CHANNEL, json!({ "event": "exit", "code": code, "cancelled": cancelled.load(Ordering::Relaxed) }));
        });
        Ok(())
    }

    /// Ends the running job, with the ffmpeg and yt-dlp processes it started.
    pub fn cancel(&self) {
        if let Some(job) = self.job.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            job.cancelled.store(true, Ordering::Relaxed);
            kill_tree(job.pid);
        }
    }
}

/// Whether `dir` is a job's scratch folder, as `halalify` names them
/// (`.../work/job-<time>-<process>`): nothing else is ever deleted.
fn is_scratch(dir: &std::path::Path) -> bool {
    let named = |p: &std::path::Path, test: fn(&str) -> bool| p.file_name().and_then(|n| n.to_str()).is_some_and(test);
    dir.is_dir() && named(dir, |n| n.starts_with("job-")) && dir.parent().is_some_and(|p| named(p, |n| n == "work"))
}

/// The JSON object on `line`, if it is an event.
fn event(line: &str) -> Option<Value> {
    let line = line.trim();
    if !line.starts_with('{') {
        return None;
    }
    serde_json::from_str::<Value>(line).ok().filter(|v| v.get("event").is_some_and(Value::is_string))
}

/// Non-empty lines of a stream, whatever its encoding or line endings: tools
/// redraw their progress with carriage returns, which also end a line here.
fn lines(stream: impl Read) -> impl Iterator<Item = String> {
    let mut reader = BufReader::new(stream);
    std::iter::from_fn(move || {
        loop {
            let mut raw = Vec::new();
            if reader.read_until(b'\n', &mut raw).ok()? == 0 {
                return None;
            }
            let text = String::from_utf8_lossy(&raw);
            let last = text.rsplit('\r').map(str::trim).find(|part| !part.is_empty());
            if let Some(last) = last {
                return Some(last.to_string());
            }
        }
    })
}

/// Where the `halalify` program is: named by `HALALIFY_EXE`, next to this
/// program (a release), or in the repository's `target/release` (development).
fn locate() -> Result<PathBuf, String> {
    let name = if cfg!(windows) { "halalify.exe" } else { "halalify" };
    if let Some(path) = std::env::var_os("HALALIFY_EXE").map(PathBuf::from).filter(|p| p.is_file()) {
        return Ok(path);
    }
    let exe = std::env::current_exe().map_err(|e| format!("cannot find this program's own folder: {e}"))?;
    let dir = exe.parent().ok_or("this program has no folder")?;
    let beside = dir.join(name);
    if beside.is_file() {
        return Ok(beside);
    }
    dir.ancestors()
        .skip(1)
        .take(5)
        .map(|a| a.join("target").join("release").join(name))
        .find(|p| p.is_file())
        .ok_or_else(|| format!("{name} was not found in {}. Keep HalalifyVid and {name} in the same folder.", dir.display()))
}

/// Keeps a console window from flashing up behind the app. The child still
/// gets a (hidden) console, which the tools it starts inherit.
fn hide_window(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = cmd;
}

/// On Unix the job leads its own process group, so the whole group can be ended.
fn own_process_group(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(not(unix))]
    let _ = cmd;
}

fn kill_tree(pid: u32) {
    #[cfg(windows)]
    let mut cmd = {
        let mut cmd = Command::new("taskkill");
        cmd.args(["/PID", &pid.to_string(), "/T", "/F"]);
        cmd
    };
    #[cfg(not(windows))]
    let mut cmd = {
        let mut cmd = Command::new("kill");
        cmd.args(["-TERM", "--", &format!("-{pid}")]);
        cmd
    };
    cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    hide_window(&mut cmd);
    let _ = cmd.status();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_json_objects_with_an_event_are_events() {
        assert_eq!(event(r#"{"event":"stage","stage":"blurring","done":1.0,"total":2.0}"#).unwrap()["stage"], "blurring");
        assert!(event("installing into C:\\data").is_none());
        assert!(event(r#"{"frames": 12}"#).is_none());
        assert!(event("{not json").is_none());
    }

    #[test]
    fn only_a_job_folder_counts_as_scratch() {
        let root = std::env::temp_dir().join(format!("halalify-test-{}", std::process::id()));
        let job = root.join("work").join("job-1-2");
        let other = root.join("work").join("videos");
        let stray = root.join("job-3-4");
        for dir in [&job, &other, &stray] {
            std::fs::create_dir_all(dir).unwrap();
        }
        assert!(is_scratch(&job));
        assert!(!is_scratch(&other) && !is_scratch(&stray) && !is_scratch(&root.join("work").join("job-missing")));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn redrawn_progress_keeps_its_last_state() {
        let stream = b"first\r\n  10%\r  55%\r 100%\n\n\r\nlast" as &[u8];
        let got: Vec<String> = lines(stream).collect();
        assert_eq!(got, ["first", "100%", "last"]);
    }
}
