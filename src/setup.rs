//! Installing what the program needs but does not ship: ONNX Runtime, the GPU
//! libraries behind it, and the models.
//!
//! Everything is fetched from its publisher (Microsoft's GitHub releases and
//! NuGet feed, NVIDIA's packages on PyPI), pinned to an exact version and
//! verified against a SHA-256 before a single file is unpacked. Only the
//! needed libraries are kept; the archives are deleted.

use std::{
    collections::BTreeMap,
    fs::File,
    io,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::{
    models, paths,
    progress::{self, Event},
    runtime::{self, Device, Runtime},
};

/// Which GPU route to install.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    /// NVIDIA GPUs (Windows and Linux): fastest, about 1.5 GB of downloads.
    Cuda,
    /// Any DirectX 12 GPU (Windows only): about 220 MB of downloads.
    Directml,
    /// No GPU: works everywhere, but is many times slower.
    Cpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    /// Zip archives, which NuGet packages and Python wheels also are.
    Zip,
    TarGz,
}

#[derive(Clone, Copy)]
struct Archive {
    /// Key in the install manifest.
    name: &'static str,
    url: &'static str,
    sha256: &'static str,
    bytes: u64,
    format: Format,
    /// Whether the archive entry with this path is kept.
    wanted: fn(&str) -> bool,
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// A Python wheel's library folder entry with this file name prefix.
fn wheel_lib(entry: &str, prefixes: &[&str]) -> bool {
    let name = file_name(entry);
    let library = name.ends_with(".dll") || name.contains(".so");
    library && prefixes.iter().any(|p| name.starts_with(p))
}

// ------------------------------------------------------------------ Windows

const WIN_ORT_CUDA: Archive = Archive {
    name: "onnxruntime-gpu-1.21.0",
    url: "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-win-x64-gpu-1.21.0.zip",
    sha256: "ae703748eafd13bd9ed48e605c2b1a8936b90ae2c84fb10840804c8a2c38f397",
    bytes: 307_588_400,
    format: Format::Zip,
    wanted: |entry| entry.contains("/lib/") && entry.ends_with(".dll"),
};

const WIN_ORT_DIRECTML: Archive = Archive {
    name: "onnxruntime-directml-1.21.0",
    url: "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/1.21.0/microsoft.ml.onnxruntime.directml.1.21.0.nupkg",
    sha256: "12bed6e1e870f30e0150e039ebb6853da32eba381eda1c373faeeb17168785e6",
    bytes: 17_410_861,
    format: Format::Zip,
    wanted: |entry| entry == "runtimes/win-x64/native/onnxruntime.dll",
};

const WIN_DIRECTML: Archive = Archive {
    name: "directml-1.15.4",
    url: "https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg",
    sha256: "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9",
    bytes: 202_292_617,
    format: Format::Zip,
    wanted: |entry| entry == "bin/x64-win/DirectML.dll",
};

/// CUDA 12.6 and cuDNN 9.6 runtime libraries, as NVIDIA publishes them on
/// PyPI. CUDA 12.6 runs on drivers from 560 (August 2024) onwards.
const WIN_CUDA: [Archive; 4] = [
    Archive {
        name: "cuda-runtime-12.6.77",
        url: "https://files.pythonhosted.org/packages/fa/76/4c80fa138333cc975743fd0687a745fccb30d167f906f13c1c7f9a85e5ea/nvidia_cuda_runtime_cu12-12.6.77-py3-none-win_amd64.whl",
        sha256: "86c58044c824bf3c173c49a2dbc7a6c8b53cb4e4dca50068be0bf64e9dab3f7f",
        bytes: 891_773,
        format: Format::Zip,
        wanted: |entry| entry.ends_with("cudart64_12.dll"),
    },
    Archive {
        name: "cublas-12.6.4.1",
        url: "https://files.pythonhosted.org/packages/84/f7/985e9bdbe3e0ac9298fcc8cfa51a392862a46a0ffaccbbd56939b62a9c83/nvidia_cublas_cu12-12.6.4.1-py3-none-win_amd64.whl",
        sha256: "9e4fa264f4d8a4eb0cdbd34beadc029f453b3bafae02401e999cf3d5a5af75f8",
        bytes: 434_535_301,
        format: Format::Zip,
        wanted: |entry| entry.ends_with("cublas64_12.dll") || entry.ends_with("cublasLt64_12.dll"),
    },
    Archive {
        name: "cufft-11.3.0.4",
        url: "https://files.pythonhosted.org/packages/b4/38/36fd800cec8f6e89b7c1576edaaf8076e69ec631644cdbc1b5f2e2b5a9df/nvidia_cufft_cu12-11.3.0.4-py3-none-win_amd64.whl",
        sha256: "6048ebddfb90d09d2707efb1fd78d4e3a77cb3ae4dc60e19aab6be0ece2ae464",
        bytes: 199_356_881,
        format: Format::Zip,
        wanted: |entry| entry.ends_with("cufft64_11.dll"),
    },
    Archive {
        name: "cudnn-9.6.0.74",
        url: "https://files.pythonhosted.org/packages/07/1d/3c7055e1d189f55ed25183f50e1d82f477024289f925df22ac89c0f4d387/nvidia_cudnn_cu12-9.6.0.74-py3-none-win_amd64.whl",
        sha256: "6a6466301b5ab1f0d8e62b14cdd02a714158f32f70ee99892218bec661e76595",
        bytes: 502_799_547,
        format: Format::Zip,
        wanted: |entry| entry.ends_with(".dll") && file_name(entry).starts_with("cudnn"),
    },
];

// ------------------------------------------------------------------ Linux

const LINUX_ORT_CPU: Archive = Archive {
    name: "onnxruntime-linux-1.21.0",
    url: "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz",
    sha256: "7485c7e7aac6501b27e353dcbe068e45c61ab51fbaf598d13970dfae669d20bf",
    bytes: 7_585_530,
    format: Format::TarGz,
    wanted: |entry| entry.contains("/lib/") && entry.contains(".so"),
};

const LINUX_ORT_CUDA: Archive = Archive {
    name: "onnxruntime-linux-gpu-1.21.0",
    url: "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-gpu-1.21.0.tgz",
    sha256: "ef37a33ba75e457aebfd0d7b342ab20424aa6126bc5f565d247f1201b66996cf",
    bytes: 225_318_955,
    format: Format::TarGz,
    wanted: |entry| entry.contains("/lib/") && entry.contains(".so"),
};

/// The same CUDA 12.6 / cuDNN 9.6 releases as on Windows, plus NVRTC and
/// cuRAND, which the Linux build of the CUDA provider also links against.
const LINUX_CUDA: [Archive; 6] = [
    Archive {
        name: "linux-cuda-runtime-12.6.77",
        url: "https://files.pythonhosted.org/packages/e1/23/e717c5ac26d26cf39a27fbc076240fad2e3b817e5889d671b67f4f9f49c5/nvidia_cuda_runtime_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256: "ba3b56a4f896141e25e19ab287cd71e52a6a0f4b29d0d31609f60e3b4d5219b7",
        bytes: 897_690,
        format: Format::Zip,
        wanted: |entry| wheel_lib(entry, &["libcudart.so"]),
    },
    Archive {
        name: "linux-cuda-nvrtc-12.6.77",
        url: "https://files.pythonhosted.org/packages/75/2e/46030320b5a80661e88039f59060d1790298b4718944a65a7f2aeda3d9e9/nvidia_cuda_nvrtc_cu12-12.6.77-py3-none-manylinux2014_x86_64.whl",
        sha256: "35b0cc6ee3a9636d5409133e79273ce1f3fd087abb0532d2d2e8fff1fe9efc53",
        bytes: 23_650_380,
        format: Format::Zip,
        wanted: |entry| wheel_lib(entry, &["libnvrtc"]),
    },
    Archive {
        name: "linux-curand-10.3.7.77",
        url: "https://files.pythonhosted.org/packages/73/1b/44a01c4e70933637c93e6e1a8063d1e998b50213a6b65ac5a9169c47e98e/nvidia_curand_cu12-10.3.7.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256: "a42cd1344297f70b9e39a1e4f467a4e1c10f1da54ff7a85c12197f6c652c8bdf",
        bytes: 56_279_010,
        format: Format::Zip,
        wanted: |entry| wheel_lib(entry, &["libcurand.so"]),
    },
    Archive {
        name: "linux-cublas-12.6.4.1",
        url: "https://files.pythonhosted.org/packages/af/eb/ff4b8c503fa1f1796679dce648854d58751982426e4e4b37d6fce49d259c/nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256: "08ed2686e9875d01b58e3cb379c6896df8e76c75e0d4a7f7dace3d7b6d9ef8eb",
        bytes: 393_138_322,
        format: Format::Zip,
        wanted: |entry| wheel_lib(entry, &["libcublas.so", "libcublasLt.so"]),
    },
    Archive {
        name: "linux-cufft-11.3.0.4",
        url: "https://files.pythonhosted.org/packages/8f/16/73727675941ab8e6ffd86ca3a4b7b47065edcca7a997920b831f8147c99d/nvidia_cufft_cu12-11.3.0.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256: "ccba62eb9cef5559abd5e0d54ceed2d9934030f51163df018532142a8ec533e5",
        bytes: 200_221_632,
        format: Format::Zip,
        wanted: |entry| wheel_lib(entry, &["libcufft.so"]),
    },
    Archive {
        name: "linux-cudnn-9.6.0.74",
        url: "https://files.pythonhosted.org/packages/67/4c/0f18ddce75f2c17cf028d16e067c07f11c421c981b8c2a45c99bc6169af6/nvidia_cudnn_cu12-9.6.0.74-py3-none-manylinux_2_27_x86_64.whl",
        sha256: "cb700ea574ba8ffdbf9381fed2cdfa0b1847852c3811fb5f0d998d8d47a1cf1a",
        bytes: 508_134_668,
        format: Format::Zip,
        wanted: |entry| wheel_lib(entry, &["libcudnn"]),
    },
];

/// What to install for a provider on this platform: ONNX Runtime archives,
/// GPU library archives, and ONNX Runtime archives of other routes that must
/// be removed first (they share file names).
fn plan(provider: Provider) -> Result<(Vec<Archive>, Vec<Archive>, Vec<Archive>)> {
    if cfg!(all(windows, target_arch = "x86_64")) {
        Ok(match provider {
            Provider::Cuda => (vec![WIN_ORT_CUDA], WIN_CUDA.to_vec(), vec![WIN_ORT_DIRECTML, WIN_DIRECTML]),
            // The DirectML build also runs on the CPU when no GPU is usable.
            Provider::Directml | Provider::Cpu => (vec![WIN_ORT_DIRECTML, WIN_DIRECTML], vec![], vec![WIN_ORT_CUDA]),
        })
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Ok(match provider {
            Provider::Cuda => (vec![LINUX_ORT_CUDA], LINUX_CUDA.to_vec(), vec![LINUX_ORT_CPU]),
            Provider::Cpu => (vec![LINUX_ORT_CPU], vec![], vec![LINUX_ORT_CUDA]),
            Provider::Directml => bail!("DirectML exists only on Windows; use --provider cuda or --provider cpu"),
        })
    } else {
        bail!(
            "setup knows the Windows and Linux x64 packages only. On this platform install ONNX Runtime 1.21 \
             yourself and point HALALIFY_ORT_DLL at its shared library."
        )
    }
}

/// What has been installed, so that a second `setup` does nothing.
#[derive(Default, Serialize, Deserialize)]
struct Manifest {
    /// Archive name -> the files it provided.
    installed: BTreeMap<String, Vec<String>>,
}

impl Manifest {
    fn path() -> PathBuf {
        paths::bin_dir().join("installed.json")
    }

    fn load() -> Manifest {
        std::fs::read(Self::path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    fn save(&self) -> Result<()> {
        paths::ensure(paths::bin_dir())?;
        std::fs::write(Self::path(), serde_json::to_vec_pretty(self)?).context("cannot write the install manifest")
    }

    /// True when `archive` was installed into `dir` and its files are still there.
    fn has(&self, archive: &Archive, dir: &Path) -> bool {
        self.installed.get(archive.name).is_some_and(|files| !files.is_empty() && files.iter().all(|f| dir.join(f).exists()))
    }
}

/// Installs the runtime for `provider` (detected when `None`) and the models.
pub fn run(provider: Option<Provider>, skip_models: bool) -> Result<()> {
    let provider = provider.unwrap_or_else(detect_provider);
    let (ort, gpu, conflicting) = plan(provider)?;
    println!("installing into {}", paths::home().display());
    println!(
        "GPU route: {}",
        match provider {
            Provider::Cuda => "CUDA (NVIDIA)",
            Provider::Directml => "DirectML",
            Provider::Cpu if cfg!(windows) => "DirectML build, CPU fallback",
            Provider::Cpu => "CPU only",
        }
    );

    let mut manifest = Manifest::load();
    let ort_dir = runtime::onnxruntime_dir();
    if conflicting.iter().any(|a| manifest.installed.contains_key(a.name)) {
        let _ = std::fs::remove_dir_all(&ort_dir);
        for a in &conflicting {
            manifest.installed.remove(a.name);
        }
        manifest.save()?;
    }
    for archive in &ort {
        install(archive, &ort_dir, &mut manifest)?;
    }
    for archive in &gpu {
        install(archive, &runtime::cuda_dir(), &mut manifest)?;
    }

    if !skip_models {
        for model in models::ALL {
            let present = model.is_present();
            model.fetch()?;
            println!("{:<28} {}", if present { "already installed" } else { "installed" }, model.file);
            progress::emit(&Event::Installed { name: model.file });
        }
    }

    // Prove the installation works by loading a model.
    if cfg!(target_os = "linux") && provider == Provider::Cuda && !runtime::linux_library_path_ready() {
        println!("ready: installed. The CUDA libraries take effect the next time halalify starts.");
        progress::emit(&Event::Ready { device: None });
        return Ok(());
    }
    let rt = Runtime::new(Device::Auto)?;
    if let Some(model) = models::ALL.iter().find(|m| m.is_present()) {
        rt.session(&model.path())?;
        let device = rt.device().map(|d| d.to_string());
        println!("ready: models run on {}", device.as_deref().unwrap_or("?"));
        progress::emit(&Event::Ready { device: device.as_deref() });
    } else {
        println!("ready: ONNX Runtime loads; models will download on first use");
        progress::emit(&Event::Ready { device: None });
    }
    Ok(())
}

/// CUDA when an NVIDIA driver is installed; otherwise DirectML on Windows and
/// the CPU elsewhere.
pub fn detect_provider() -> Provider {
    if cfg!(windows) {
        let system = std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
        if system.join("System32").join("nvcuda.dll").is_file() { Provider::Cuda } else { Provider::Directml }
    } else {
        // The driver's user-space library is the one sign present everywhere:
        // /proc/driver/nvidia is missing inside containers and under WSL.
        let nvidia = Path::new("/proc/driver/nvidia/version").exists()
            || ["/usr/lib/wsl/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"]
                .iter()
                .any(|dir| Path::new(dir).join("libcuda.so.1").exists());
        if nvidia { Provider::Cuda } else { Provider::Cpu }
    }
}

/// Where yt-dlp publishes its newest release.
const YTDLP_RELEASE: &str = "https://github.com/yt-dlp/yt-dlp/releases/latest/download";

/// Installs the newest yt-dlp into the program's own `bin` folder, which is
/// searched before anything else, so an older copy elsewhere stops mattering.
///
/// Unlike the runtime, yt-dlp cannot be pinned to a version: the sites it
/// reads change, and only a recent release keeps working. The download is
/// checked against the checksum list published with the same release, which
/// catches a damaged download but, having the same origin, nothing more.
pub fn update_ytdlp() -> Result<()> {
    let (asset, name) = if cfg!(all(windows, target_arch = "x86_64")) {
        ("yt-dlp.exe", "yt-dlp.exe")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        ("yt-dlp_linux", "yt-dlp")
    } else {
        bail!("yt-dlp is only fetched for Windows and Linux x64; install it with your package manager")
    };
    let bin = paths::ensure(paths::bin_dir())?;
    let downloads = paths::ensure(paths::cache_dir().join("downloads"))?;

    let sums = downloads.join("yt-dlp.sums");
    models::download(&format!("{YTDLP_RELEASE}/SHA2-256SUMS"), &sums, 2048, "yt-dlp checksums")
        .context("downloading yt-dlp's checksum list")?;
    let listed = std::fs::read_to_string(&sums).context("cannot read yt-dlp's checksum list")?;
    let _ = std::fs::remove_file(&sums);
    let expected = listed_checksum(&listed, asset).with_context(|| format!("yt-dlp's checksum list does not name {asset}"))?;

    let file = downloads.join("yt-dlp.download");
    models::download(&format!("{YTDLP_RELEASE}/{asset}"), &file, 24_000_000, "yt-dlp").context("downloading yt-dlp")?;
    let got = models::sha256_file(&file)?;
    if got != expected {
        let _ = std::fs::remove_file(&file);
        bail!("yt-dlp: checksum mismatch (expected {expected}, got {got}); the download was discarded");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o755))?;
    }
    let dest = bin.join(name);
    std::fs::rename(&file, &dest).with_context(|| format!("cannot move yt-dlp to {}", dest.display()))?;
    println!("{:<28} yt-dlp ({})", "installed", dest.display());
    progress::emit(&Event::Installed { name: "yt-dlp" });
    Ok(())
}

/// The SHA-256 that a `SHA2-256SUMS` list (`<hash>  <file>` per line) gives for `file`.
fn listed_checksum(list: &str, file: &str) -> Option<String> {
    list.lines().find_map(|line| {
        let (hash, name) = line.split_once(char::is_whitespace)?;
        (name.trim().trim_start_matches('*') == file && hash.len() == 64).then(|| hash.to_ascii_lowercase())
    })
}

fn install(archive: &Archive, dir: &Path, manifest: &mut Manifest) -> Result<()> {
    if manifest.has(archive, dir) {
        println!("{:<28} {}", "already installed", archive.name);
        progress::emit(&Event::Installed { name: archive.name });
        return Ok(());
    }
    paths::ensure(dir.to_path_buf())?;
    let downloads = paths::ensure(paths::cache_dir().join("downloads"))?;
    let file = downloads.join(format!("{}.download", archive.name));
    models::download(archive.url, &file, archive.bytes, archive.name)
        .with_context(|| format!("downloading {} from {}", archive.name, archive.url))?;
    let got = models::sha256_file(&file)?;
    if got != archive.sha256 {
        let _ = std::fs::remove_file(&file);
        bail!("{}: checksum mismatch (expected {}, got {got}); the download was discarded", archive.name, archive.sha256);
    }
    let files = match archive.format {
        Format::Zip => unpack_zip(&file, dir, archive.wanted),
        Format::TarGz => unpack_tar_gz(&file, dir, archive.wanted),
    }
    .with_context(|| format!("unpacking {}", archive.name))?;
    let _ = std::fs::remove_file(&file);
    if files.is_empty() {
        bail!("{} did not contain the expected files", archive.name);
    }
    println!("{:<28} {} ({})", "installed", archive.name, files.join(", "));
    progress::emit(&Event::Installed { name: archive.name });
    manifest.installed.insert(archive.name.to_string(), files);
    manifest.save()
}

/// Writes `reader` to `dir/name` through a temporary name, so a failed run
/// never leaves half a library behind.
fn write_file(dir: &Path, name: &str, reader: &mut impl io::Read) -> Result<()> {
    let partial = dir.join(format!("{name}.part"));
    io::copy(reader, &mut File::create(&partial)?)?;
    std::fs::rename(&partial, dir.join(name))?;
    Ok(())
}

/// Extracts the wanted entries of a zip archive into `dir`, flattening paths.
/// Returns the names of the files written.
fn unpack_zip(archive: &Path, dir: &Path, wanted: fn(&str) -> bool) -> Result<Vec<String>> {
    let mut zip = zip::ZipArchive::new(File::open(archive)?)?;
    let mut written = Vec::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let path = entry.name()?.replace('\\', "/");
        if entry.is_dir() || !wanted(&path) {
            continue;
        }
        let name = file_name(&path).to_string();
        write_file(dir, &name, &mut entry)?;
        written.push(name);
    }
    Ok(written)
}

/// Extracts the wanted entries of a .tar.gz archive into `dir`, flattening
/// paths. Symbolic links (`libonnxruntime.so -> libonnxruntime.so.1.21.0`)
/// are recreated as links on Unix and as copies elsewhere.
fn unpack_tar_gz(archive: &Path, dir: &Path, wanted: fn(&str) -> bool) -> Result<Vec<String>> {
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(File::open(archive)?));
    let mut written = Vec::new();
    let mut links: Vec<(String, String)> = Vec::new();
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_string_lossy().replace('\\', "/");
        if !wanted(&path) {
            continue;
        }
        let name = file_name(&path).to_string();
        let kind = entry.header().entry_type();
        if kind.is_symlink() {
            if let Some(target) = entry.link_name()? {
                links.push((name, file_name(&target.to_string_lossy()).to_string()));
            }
        } else if kind.is_file() {
            write_file(dir, &name, &mut entry)?;
            written.push(name);
        }
    }
    // Links last, once their targets exist.
    for (name, target) in links {
        let link = dir.join(&name);
        let _ = std::fs::remove_file(&link);
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &link)?;
        #[cfg(not(unix))]
        std::fs::copy(dir.join(&target), &link)?;
        written.push(name);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_checksum_is_read_from_the_published_list() {
        let a = "a".repeat(64);
        let b = "B".repeat(64);
        let list = format!("{a}  yt-dlp\n{b}  yt-dlp.exe\n{a} *yt-dlp_linux\nnot-a-hash  yt-dlp.tar.gz\n");
        assert_eq!(listed_checksum(&list, "yt-dlp.exe"), Some("b".repeat(64)));
        assert_eq!(listed_checksum(&list, "yt-dlp_linux"), Some(a.clone()));
        assert_eq!(listed_checksum(&list, "yt-dlp.tar.gz"), None);
        assert_eq!(listed_checksum(&list, "yt-dlp_macos"), None);
    }

    #[test]
    fn archive_filters_keep_only_the_needed_libraries() {
        assert!((WIN_ORT_CUDA.wanted)("onnxruntime-win-x64-gpu-1.21.0/lib/onnxruntime_providers_cuda.dll"));
        assert!(!(WIN_ORT_CUDA.wanted)("onnxruntime-win-x64-gpu-1.21.0/lib/onnxruntime.lib"));
        assert!(!(WIN_ORT_CUDA.wanted)("onnxruntime-win-x64-gpu-1.21.0/include/onnxruntime_c_api.h"));
        assert!((WIN_ORT_DIRECTML.wanted)("runtimes/win-x64/native/onnxruntime.dll"));
        assert!(!(WIN_ORT_DIRECTML.wanted)("runtimes/win-arm64/native/onnxruntime.dll"));
        assert!((WIN_DIRECTML.wanted)("bin/x64-win/DirectML.dll"));
        assert!(!(WIN_DIRECTML.wanted)("bin/x64-win/DirectML.Debug.dll"));
        assert!((WIN_CUDA[3].wanted)("nvidia/cudnn/bin/cudnn_ops64_9.dll"));
        assert!(!(WIN_CUDA[3].wanted)("nvidia/cudnn/include/cudnn.h"));
        assert!(!(WIN_CUDA[1].wanted)("nvidia/cublas/bin/nvblas64_12.dll"));

        assert!((LINUX_ORT_CUDA.wanted)("onnxruntime-linux-x64-gpu-1.21.0/lib/libonnxruntime.so.1.21.0"));
        assert!((LINUX_ORT_CUDA.wanted)("onnxruntime-linux-x64-gpu-1.21.0/lib/libonnxruntime_providers_cuda.so"));
        assert!(!(LINUX_ORT_CUDA.wanted)("onnxruntime-linux-x64-gpu-1.21.0/lib/cmake/onnxruntimeConfig.cmake"));
        assert!((LINUX_CUDA[5].wanted)("nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9"));
        assert!(!(LINUX_CUDA[5].wanted)("nvidia/cudnn/include/cudnn_ops.h"));
        assert!((LINUX_CUDA[2].wanted)("nvidia/curand/lib/libcurand.so.10"));
        assert!((LINUX_CUDA[3].wanted)("nvidia/cublas/lib/libcublasLt.so.12"));
        assert!(!(LINUX_CUDA[3].wanted)("nvidia/cublas/lib/libnvblas.so.12"));
        assert!((LINUX_CUDA[1].wanted)("nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.12.6"));
    }

    #[test]
    fn every_archive_is_pinned() {
        let all = [WIN_ORT_CUDA, WIN_ORT_DIRECTML, WIN_DIRECTML, LINUX_ORT_CPU, LINUX_ORT_CUDA];
        for a in all.iter().chain(&WIN_CUDA).chain(&LINUX_CUDA) {
            assert_eq!(a.sha256.len(), 64, "{}", a.name);
            assert!(a.url.starts_with("https://"), "{}", a.name);
            assert!(a.bytes > 0);
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("halalify-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn zip_unpack_flattens_and_filters() {
        use std::io::Write;
        let dir = temp_dir("unzip");
        let zip_path = dir.join("t.zip");
        let mut w = zip::ZipWriter::new(File::create(&zip_path).unwrap());
        let options = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        w.start_file("pkg/lib/keep.dll", options).unwrap();
        w.write_all(b"library").unwrap();
        w.start_file("pkg/include/skip.h", options).unwrap();
        w.write_all(b"header").unwrap();
        w.finish().unwrap();

        let files = unpack_zip(&zip_path, &dir, |e| e.ends_with(".dll")).unwrap();
        assert_eq!(files, ["keep.dll"]);
        assert_eq!(std::fs::read(dir.join("keep.dll")).unwrap(), b"library");
        assert!(!dir.join("skip.h").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn tar_unpack_recreates_library_links() {
        let dir = temp_dir("untar");
        let tgz = dir.join("t.tgz");
        {
            let gz = flate2::write::GzEncoder::new(File::create(&tgz).unwrap(), flate2::Compression::fast());
            let mut b = tar::Builder::new(gz);
            let mut h = tar::Header::new_gnu();
            h.set_size(7);
            h.set_mode(0o644);
            h.set_entry_type(tar::EntryType::Regular);
            b.append_data(&mut h, "pkg/lib/libx.so.1.2", &b"library"[..]).unwrap();
            let mut l = tar::Header::new_gnu();
            l.set_size(0);
            l.set_entry_type(tar::EntryType::Symlink);
            b.append_link(&mut l, "pkg/lib/libx.so", "libx.so.1.2").unwrap();
            let mut s = tar::Header::new_gnu();
            s.set_size(1);
            s.set_entry_type(tar::EntryType::Regular);
            b.append_data(&mut s, "pkg/include/x.h", &b"h"[..]).unwrap();
            b.into_inner().unwrap().finish().unwrap();
        }
        let files = unpack_tar_gz(&tgz, &dir, |e| e.contains("/lib/")).unwrap();
        assert_eq!(files, ["libx.so.1.2", "libx.so"]);
        assert_eq!(std::fs::read(dir.join("libx.so")).unwrap(), b"library");
        assert!(!dir.join("x.h").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
