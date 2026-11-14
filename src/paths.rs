//! Where HalalifyVid keeps its files.
//!
//! Everything lives under one home directory so that models, cached probes and
//! temporary work files are easy to find and easy to delete:
//!
//! ```text
//! <home>/models   downloaded / converted ONNX models
//! <home>/bin      optional private copies of ffmpeg, yt-dlp, CUDA DLLs
//! <home>/cache    small JSON caches (tool probes)
//! <home>/work     per-job scratch space, removed when a job finishes
//! ```
//!
//! The home is, in order of preference:
//!
//! 1. `HALALIFY_HOME`, when set;
//! 2. a folder named `data` next to the executable, when it exists: release
//!    archives ship one, so an unpacked release is self-contained and portable;
//! 3. the per-user local data directory (`%LOCALAPPDATA%\HalalifyVid` on
//!    Windows, `~/.local/share/HalalifyVid` on Linux).

use std::path::PathBuf;

use anyhow::{Context, Result};

pub fn home() -> PathBuf {
    if let Some(h) = std::env::var_os("HALALIFY_HOME").filter(|h| !h.is_empty()) {
        return PathBuf::from(h);
    }
    if let Some(data) = portable_dir() {
        return data;
    }
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("HalalifyVid")
}

/// The `data` folder next to the executable, if there is one.
fn portable_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let data = exe.parent()?.join("data");
    data.is_dir().then_some(data)
}

pub fn models_dir() -> PathBuf {
    home().join("models")
}

pub fn bin_dir() -> PathBuf {
    home().join("bin")
}

pub fn cache_dir() -> PathBuf {
    home().join("cache")
}

pub fn work_root() -> PathBuf {
    home().join("work")
}

/// Creates the directory (and parents) if needed and returns it.
pub fn ensure(dir: PathBuf) -> Result<PathBuf> {
    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    Ok(dir)
}
