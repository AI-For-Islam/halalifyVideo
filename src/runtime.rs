//! ONNX Runtime setup: one environment per process and a GPU provider chosen
//! by trying them in order.
//!
//! ONNX Runtime is loaded from a DLL at run time rather than linked in, and
//! the version is pinned to 1.21. From 1.22 on, creating the environment runs
//! a hardware "device discovery" that takes about 11 seconds on some Windows
//! machines (measured here on a hybrid-graphics laptop) before any model can
//! load; 1.21 starts in 30 ms and runs these models just as fast.
//!
//! Providers:
//! * **CUDA** needs the CUDA 12 and cuDNN 9 runtime DLLs, looked up in
//!   `HALALIFY_CUDA_DIR`, `<home>/bin/cuda`, `CUDA_PATH` and `PATH`.
//! * **DirectML** works when the loaded onnxruntime.dll is a DirectML build.
//! * **CPU** always works.
//!
//! `Device::Auto` keeps the first provider that can really create a session,
//! so a missing DLL costs speed rather than function.

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

/// Shared-library file name of ONNX Runtime on this platform.
pub const ORT_DYLIB: &str = if cfg!(windows) {
    "onnxruntime.dll"
} else if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else {
    "libonnxruntime.so"
};

/// CUDA 12 and cuDNN 9 libraries the CUDA provider needs, dependencies first.
const CUDA_DYLIBS: &[&str] =
    &["cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cufft64_11.dll", "cudnn64_9.dll"];

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

/// Where the ONNX Runtime library is, if it is installed.
pub fn find_onnxruntime() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(p) = std::env::var_os("HALALIFY_ORT_DLL").filter(|p| !p.is_empty()) {
        candidates.push(PathBuf::from(p));
    }
    candidates.push(onnxruntime_dir().join(ORT_DYLIB));
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        candidates.push(dir.join(ORT_DYLIB));
    }
    candidates.into_iter().find(|p| p.is_file())
}

/// Private install location of ONNX Runtime.
pub fn onnxruntime_dir() -> PathBuf {
    paths::bin_dir().join("onnxruntime")
}

/// Private install location of the CUDA and cuDNN runtime libraries.
pub fn cuda_dir() -> PathBuf {
    paths::bin_dir().join("cuda")
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
        let dylib = find_onnxruntime().with_context(|| {
            format!("ONNX Runtime is not installed in {}; run `halalify setup`", onnxruntime_dir().display())
        })?;
        // The CUDA libraries must be findable before the provider DLL loads.
        let cuda_dir = if matches!(requested, Device::Auto | Device::Cuda) { expose_cuda() } else { None };
        let env = ort::init_from(&dylib)
            .map_err(|e| anyhow::anyhow!("cannot load {}: {e}", dylib.display()))?
            .with_name("halalify")
            .build()
            .context("cannot start ONNX Runtime")?;
        Ok(Runtime { env, requested, settled: Cell::new(None), cuda_dir })
    }

    /// The provider sessions run on, once at least one session exists.
    pub fn device(&self) -> Option<Device> {
        self.settled.get()
    }

    /// Directory the CUDA libraries were found in, if any.
    pub fn cuda_dir(&self) -> Option<&Path> {
        self.cuda_dir.as_deref()
    }

    /// Loads a model on the best available provider.
    pub fn session(&self, model: &Path) -> Result<Session> {
        let order: &[Device] = match self.settled.get().unwrap_or(self.requested) {
            Device::Auto if cfg!(windows) => &[Device::Cuda, Device::Dml, Device::Cpu],
            Device::Auto => &[Device::Cuda, Device::Cpu],
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
                Err(e) => failures.push(format!("{device}: {}", first_line(&e.to_string()))),
            }
        }
        bail!("cannot load {}:\n  {}", model.display(), failures.join("\n  "))
    }

    /// Returns `ort::Result` so that `?` can turn the builder's recoverable
    /// error type into a plain error.
    fn try_session(&self, model: &Path, device: Device) -> ort::Result<Session> {
        let builder = Session::builder(&self.env)?.with_optimization_level(GraphOptimizationLevel::All)?;
        let mut builder = match device {
            Device::Cuda => builder.with_execution_providers([ep::CUDA::default()
                .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
                .build()
                .error_on_failure()])?,
            // DirectML requires sequential execution and no memory-pattern planning.
            #[cfg(windows)]
            Device::Dml => builder
                .with_parallel_execution(false)?
                .with_memory_pattern(false)?
                .with_execution_providers([ep::DirectML::default().build().error_on_failure()])?,
            #[cfg(not(windows))]
            Device::Dml => return Err(ort::Error::new("DirectML exists only on Windows")),
            Device::Cpu | Device::Auto => builder.with_intra_threads(physical_cores())?,
        };
        builder.commit_from_file(model)
    }
}

/// The first line of an error that says something (ONNX Runtime's messages
/// sometimes start with a blank line).
fn first_line(s: &str) -> &str {
    s.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or(s)
}

fn physical_cores() -> usize {
    // Hyper-threads do not help dense linear algebra; half the logical count
    // is the usual physical-core estimate.
    std::thread::available_parallelism().map_or(4, |n| (n.get() / 2).max(1))
}

/// Finds the CUDA 12 / cuDNN 9 runtime libraries and makes them loadable.
///
/// The directories are put on the DLL search path (cuDNN loads its engine
/// libraries by bare name at run time, and newer releases ship more of them
/// than any fixed list knows) and the core libraries are loaded up front.
fn expose_cuda() -> Option<PathBuf> {
    if !cfg!(windows) {
        // On Linux the library path was arranged when the program started.
        let dir = cuda_dir();
        return dir.join("libcudart.so.12").exists().then_some(dir);
    }
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = std::env::var_os("HALALIFY_CUDA_DIR") {
        dirs.push(d.into());
    }
    dirs.push(cuda_dir());
    if let Some(d) = std::env::var_os("CUDA_PATH") {
        dirs.push(PathBuf::from(d).join("bin"));
    }
    let path_dirs: Vec<PathBuf> = std::env::var_os("PATH").iter().flat_map(std::env::split_paths).collect();
    dirs.extend(path_dirs.iter().cloned());

    let locate = |dll: &str| dirs.iter().find(|d| d.join(dll).is_file()).cloned();
    let cuda = locate(CUDA_DYLIBS[0])?;
    let cudnn = locate(CUDA_DYLIBS[CUDA_DYLIBS.len() - 1])?;

    let mut search = vec![cuda.clone(), cudnn.clone()];
    search.extend(path_dirs);
    if let Ok(joined) = std::env::join_paths(search) {
        // SAFETY: called once from `Runtime::new` on the main thread before any
        // worker thread exists, so nothing reads the environment concurrently.
        unsafe { std::env::set_var("PATH", joined) };
    }
    for dll in CUDA_DYLIBS {
        let dir = if dll.starts_with("cudnn") { &cudnn } else { &cuda };
        ort::util::preload_dylib(dir.join(dll)).ok()?;
    }
    Some(cuda)
}

/// Variable that marks a process started by [`prepare_linux_library_path`].
#[cfg(target_os = "linux")]
const REEXEC_MARK: &str = "HALALIFY_LIBRARY_PATH_SET";

/// Directories with the privately installed shared libraries, if any.
fn private_library_dirs() -> Vec<PathBuf> {
    [onnxruntime_dir(), cuda_dir()].into_iter().filter(|d| d.is_dir()).collect()
}

/// True when the private library folders are already on `LD_LIBRARY_PATH`
/// (or there are none).
pub fn linux_library_path_ready() -> bool {
    let current: Vec<PathBuf> = std::env::var_os("LD_LIBRARY_PATH").iter().flat_map(std::env::split_paths).collect();
    private_library_dirs().iter().all(|d| current.contains(d))
}

/// On Linux, restarts the program with the private ONNX Runtime and CUDA
/// folders on `LD_LIBRARY_PATH`.
///
/// The CUDA provider and cuDNN load their libraries by name, and the Linux
/// loader reads the search path only when a process starts, so setting it
/// later has no effect. Called first thing in `main`; does nothing when the
/// folders are already on the path, when there are none, or on other systems.
pub fn prepare_linux_library_path() {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        if std::env::var_os(REEXEC_MARK).is_some() || linux_library_path_ready() {
            return;
        }
        let mut dirs = private_library_dirs();
        dirs.extend(std::env::var_os("LD_LIBRARY_PATH").iter().flat_map(std::env::split_paths));
        let (Ok(exe), Ok(joined)) = (std::env::current_exe(), std::env::join_paths(dirs)) else { return };
        let mut args = std::env::args_os();
        let argv0 = args.next().unwrap_or_else(|| exe.clone().into_os_string());
        // `exec` only returns on failure, in which case the program simply
        // carries on without the GPU libraries.
        let error = std::process::Command::new(&exe)
            .arg0(argv0)
            .args(args)
            .env("LD_LIBRARY_PATH", joined)
            .env(REEXEC_MARK, "1")
            .exec();
        eprintln!("note: could not restart with the GPU libraries on the library path: {error}");
    }
}
