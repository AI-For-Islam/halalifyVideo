//! ONNX Runtime setup: one environment per process and a GPU provider chosen
//! by trying them in order.
//!
//! * **CUDA** is the fastest on NVIDIA cards but needs the CUDA 13 and cuDNN 9
//!   runtime DLLs, which are found in `HALALIFY_CUDA_DIR`, `<home>/bin/cuda`,
//!   `CUDA_PATH` or on `PATH`.
//! * **DirectML** needs nothing beyond a DirectX 12 GPU of any vendor.
//! * **CPU** always works.
//!
//! `Device::Auto` walks that list and keeps the first provider that can
//! actually create a session, so a missing DLL downgrades speed, not function.

use std::{
    cell::Cell,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use ort::{
    ep,
    environment::Environment,
    session::{Session, builder::GraphOptimizationLevel},
};

use crate::paths;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Device {
    /// CUDA, then DirectML, then CPU: the first that works.
    Auto,
    Cuda,
    /// DirectML (any DirectX 12 GPU).
    Dml,
    Cpu,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Device::Auto => "auto",
            Device::Cuda => "CUDA",
            Device::Dml => "DirectML",
            Device::Cpu => "CPU",
        })
    }
}

pub struct Runtime {
    env: Environment,
    requested: Device,
    /// The provider that worked for the first session; later sessions reuse it.
    settled: Cell<Option<Device>>,
    cuda_dir: Option<PathBuf>,
}

impl Runtime {
    pub fn new(requested: Device) -> Result<Runtime> {
        let env = ort::init().with_name("halalify").build().context("cannot start ONNX Runtime")?;
        let cuda_dir = if matches!(requested, Device::Auto | Device::Cuda) { preload_cuda() } else { None };
        Ok(Runtime { env, requested, settled: Cell::new(None), cuda_dir })
    }

    /// The provider sessions run on, once at least one session exists.
    pub fn device(&self) -> Option<Device> {
        self.settled.get()
    }

    pub fn cuda_dir(&self) -> Option<&Path> {
        self.cuda_dir.as_deref()
    }

    /// Loads a model on the best available provider.
    pub fn session(&self, model: &Path) -> Result<Session> {
        let order: &[Device] = match self.settled.get().unwrap_or(self.requested) {
            Device::Auto => &[Device::Cuda, Device::Dml, Device::Cpu],
            Device::Cuda => &[Device::Cuda],
            Device::Dml => &[Device::Dml],
            Device::Cpu => &[Device::Cpu],
        };
        let mut failures = Vec::new();
        for &device in order {
            match self.try_session(model, device) {
                Ok(session) => {
                    self.settled.set(Some(device));
                    return Ok(session);
                }
                Err(e) => failures.push(format!("{device}: {e}")),
            }
        }
        bail!("cannot load {}:\n  {}", model.display(), failures.join("\n  "))
    }

    /// Returns `ort::Result` so that `?` can turn the builder's recoverable
    /// error type into a plain error.
    fn try_session(&self, model: &Path, device: Device) -> ort::Result<Session> {
        let builder = Session::builder(&self.env)?.with_optimization_level(GraphOptimizationLevel::All)?;
        let mut builder = match device {
            Device::Cuda => builder.with_execution_providers([ep::CUDA::default().build().error_on_failure()])?,
            // DirectML requires sequential execution and no memory-pattern planning.
            Device::Dml => builder
                .with_parallel_execution(false)?
                .with_memory_pattern(false)?
                .with_execution_providers([ep::DirectML::default().build().error_on_failure()])?,
            Device::Cpu | Device::Auto => builder.with_intra_threads(physical_cores())?,
        };
        builder.commit_from_file(model)
    }
}

fn physical_cores() -> usize {
    // Hyper-threads do not help dense linear algebra; half the logical count
    // is the usual physical-core estimate.
    std::thread::available_parallelism().map_or(4, |n| (n.get() / 2).max(1))
}

/// Finds the CUDA 13 / cuDNN 9 runtime DLLs and loads them into the process so
/// the CUDA provider resolves them regardless of `PATH`.
fn preload_cuda() -> Option<PathBuf> {
    if !cfg!(windows) {
        return None; // elsewhere the loader finds them through the usual library path
    }
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = std::env::var_os("HALALIFY_CUDA_DIR") {
        dirs.push(d.into());
    }
    dirs.push(paths::bin_dir().join("cuda"));
    if let Some(d) = std::env::var_os("CUDA_PATH") {
        let d = PathBuf::from(d);
        dirs.push(d.join("bin").join("x64"));
        dirs.push(d.join("bin"));
    }
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    let has = |d: &Path, dll: &str| d.join(dll).is_file();
    let cuda = dirs.iter().find(|d| has(d, ep::cuda::CUDA_DYLIBS[0]))?;
    let cudnn = dirs.iter().find(|d| has(d, ep::cuda::CUDNN_DYLIBS[0]))?;
    ep::cuda::preload_dylibs(Some(cuda), Some(cudnn)).ok()?;
    Some(cuda.clone())
}
