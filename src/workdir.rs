//! A private scratch directory for one job, deleted when the job ends.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::paths;

pub struct WorkDir {
    dir: PathBuf,
    keep: bool,
}

impl WorkDir {
    pub fn create(keep: bool) -> Result<WorkDir> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let dir = paths::ensure(paths::work_root().join(format!("job-{stamp}-{}", std::process::id())))?;
        Ok(WorkDir { dir, keep })
    }

    pub fn file(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    pub fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}
