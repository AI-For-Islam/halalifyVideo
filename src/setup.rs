//! Installing what the program needs but does not ship: ONNX Runtime, the GPU
//! libraries behind it, and the models.
//!
//! Everything is fetched from its publisher (Microsoft's GitHub releases and
//! NuGet feed, NVIDIA's packages on PyPI), pinned to an exact version and
//! verified against a SHA-256 before a single file is unpacked. Only the
//! needed DLLs are kept; the archives are deleted.

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
    runtime::{self, Device, Runtime},
};

/// Which GPU route to install.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    /// NVIDIA GPUs: fastest, about 1.5 GB of downloads.
    Cuda,
    /// Any DirectX 12 GPU: about 220 MB of downloads.
    Directml,
}

#[derive(Clone, Copy)]
struct Archive {
    /// Key in the install manifest.
    name: &'static str,
    url: &'static str,
    sha256: &'static str,
    bytes: u64,
    /// Whether the archive entry with this path is kept.
    wanted: fn(&str) -> bool,
}

const ORT_CUDA: Archive = Archive {
    name: "onnxruntime-gpu-1.21.0",
    url: "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-win-x64-gpu-1.21.0.zip",
    sha256: "ae703748eafd13bd9ed48e605c2b1a8936b90ae2c84fb10840804c8a2c38f397",
    bytes: 307_588_400,
    wanted: |entry| entry.contains("/lib/") && entry.ends_with(".dll"),
};

const ORT_DIRECTML: Archive = Archive {
    name: "onnxruntime-directml-1.21.0",
    url: "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/1.21.0/microsoft.ml.onnxruntime.directml.1.21.0.nupkg",
    sha256: "12bed6e1e870f30e0150e039ebb6853da32eba381eda1c373faeeb17168785e6",
    bytes: 17_410_861,
    wanted: |entry| entry == "runtimes/win-x64/native/onnxruntime.dll",
};

const DIRECTML: Archive = Archive {
    name: "directml-1.15.4",
    url: "https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg",
    sha256: "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9",
    bytes: 202_292_617,
    wanted: |entry| entry == "bin/x64-win/DirectML.dll",
};

/// CUDA 12.6 and cuDNN 9.6 runtime libraries, as NVIDIA publishes them on
/// PyPI. CUDA 12.6 runs on drivers from 560 (August 2024) onwards.
const CUDA: [Archive; 4] = [
    Archive {
        name: "cuda-runtime-12.6.77",
        url: "https://files.pythonhosted.org/packages/fa/76/4c80fa138333cc975743fd0687a745fccb30d167f906f13c1c7f9a85e5ea/nvidia_cuda_runtime_cu12-12.6.77-py3-none-win_amd64.whl",
        sha256: "86c58044c824bf3c173c49a2dbc7a6c8b53cb4e4dca50068be0bf64e9dab3f7f",
        bytes: 891_773,
        wanted: |entry| entry.ends_with("cudart64_12.dll"),
    },
    Archive {
        name: "cublas-12.6.4.1",
        url: "https://files.pythonhosted.org/packages/84/f7/985e9bdbe3e0ac9298fcc8cfa51a392862a46a0ffaccbbd56939b62a9c83/nvidia_cublas_cu12-12.6.4.1-py3-none-win_amd64.whl",
        sha256: "9e4fa264f4d8a4eb0cdbd34beadc029f453b3bafae02401e999cf3d5a5af75f8",
        bytes: 434_535_301,
        wanted: |entry| entry.ends_with("cublas64_12.dll") || entry.ends_with("cublasLt64_12.dll"),
    },
    Archive {
        name: "cufft-11.3.0.4",
        url: "https://files.pythonhosted.org/packages/b4/38/36fd800cec8f6e89b7c1576edaaf8076e69ec631644cdbc1b5f2e2b5a9df/nvidia_cufft_cu12-11.3.0.4-py3-none-win_amd64.whl",
        sha256: "6048ebddfb90d09d2707efb1fd78d4e3a77cb3ae4dc60e19aab6be0ece2ae464",
        bytes: 199_356_881,
        wanted: |entry| entry.ends_with("cufft64_11.dll"),
    },
    Archive {
        name: "cudnn-9.6.0.74",
        url: "https://files.pythonhosted.org/packages/07/1d/3c7055e1d189f55ed25183f50e1d82f477024289f925df22ac89c0f4d387/nvidia_cudnn_cu12-9.6.0.74-py3-none-win_amd64.whl",
        sha256: "6a6466301b5ab1f0d8e62b14cdd02a714158f32f70ee99892218bec661e76595",
        bytes: 502_799_547,
        wanted: |entry| entry.ends_with(".dll") && file_name(entry).starts_with("cudnn"),
    },
];

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
        self.installed.get(archive.name).is_some_and(|files| !files.is_empty() && files.iter().all(|f| dir.join(f).is_file()))
    }
}

/// Installs the runtime for `provider` (detected when `None`) and the models.
pub fn run(provider: Option<Provider>, skip_models: bool) -> Result<()> {
    if !cfg!(all(windows, target_arch = "x86_64")) {
        bail!(
            "setup only knows the Windows x64 packages. On this platform install ONNX Runtime 1.21 yourself and \
             point HALALIFY_ORT_DLL at its shared library."
        );
    }
    let provider = provider.unwrap_or_else(detect_provider);
    println!("installing into {}", paths::home().display());
    println!("GPU route: {}", if provider == Provider::Cuda { "CUDA (NVIDIA)" } else { "DirectML" });

    let mut manifest = Manifest::load();
    let ort_dir = runtime::onnxruntime_dir();
    let (wanted, unwanted): (&[Archive], &[Archive]) = match provider {
        Provider::Cuda => (&[ORT_CUDA], &[ORT_DIRECTML, DIRECTML]),
        Provider::Directml => (&[ORT_DIRECTML, DIRECTML], &[ORT_CUDA]),
    };
    // The two builds share file names, so switching route starts from an empty folder.
    if unwanted.iter().any(|a| manifest.installed.contains_key(a.name)) {
        let _ = std::fs::remove_dir_all(&ort_dir);
        for a in unwanted {
            manifest.installed.remove(a.name);
        }
        manifest.save()?;
    }
    for archive in wanted {
        install(archive, &ort_dir, &mut manifest)?;
    }
    if provider == Provider::Cuda {
        for archive in &CUDA {
            install(archive, &runtime::cuda_dir(), &mut manifest)?;
        }
    }

    if !skip_models {
        for model in models::ALL {
            let present = model.is_present();
            model.fetch()?;
            println!("{:<28} {}", if present { "already installed" } else { "installed" }, model.file);
        }
    }

    // Prove the installation works by loading a model on the GPU.
    let rt = Runtime::new(Device::Auto)?;
    if let Some(model) = models::ALL.iter().find(|m| m.is_present()) {
        rt.session(&model.path())?;
        println!("ready: models run on {}", rt.device().map_or("?".into(), |d| d.to_string()));
    } else {
        println!("ready: ONNX Runtime loads; models will download on first use");
    }
    Ok(())
}

/// NVIDIA's driver installs `nvcuda.dll`; its presence means a CUDA-capable GPU.
fn detect_provider() -> Provider {
    let system = std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    if system.join("System32").join("nvcuda.dll").is_file() { Provider::Cuda } else { Provider::Directml }
}

fn install(archive: &Archive, dir: &Path, manifest: &mut Manifest) -> Result<()> {
    if manifest.has(archive, dir) {
        println!("{:<28} {}", "already installed", archive.name);
        return Ok(());
    }
    paths::ensure(dir.to_path_buf())?;
    let downloads = paths::ensure(paths::cache_dir().join("downloads"))?;
    let file = downloads.join(format!("{}.zip", archive.name));
    models::download(archive.url, &file, archive.bytes, archive.name)
        .with_context(|| format!("downloading {} from {}", archive.name, archive.url))?;
    let got = models::sha256_file(&file)?;
    if got != archive.sha256 {
        let _ = std::fs::remove_file(&file);
        bail!("{}: checksum mismatch (expected {}, got {got}); the download was discarded", archive.name, archive.sha256);
    }
    let files = unpack(&file, dir, archive.wanted).with_context(|| format!("unpacking {}", archive.name))?;
    let _ = std::fs::remove_file(&file);
    if files.is_empty() {
        bail!("{} did not contain the expected files", archive.name);
    }
    println!("{:<28} {} ({})", "installed", archive.name, files.join(", "));
    manifest.installed.insert(archive.name.to_string(), files);
    manifest.save()
}

/// Extracts the wanted entries of a zip archive into `dir`, flattening paths.
/// Returns the names of the files written.
fn unpack(archive: &Path, dir: &Path, wanted: fn(&str) -> bool) -> Result<Vec<String>> {
    let mut zip = zip::ZipArchive::new(File::open(archive)?)?;
    let mut written = Vec::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let path = entry.name()?.replace('\\', "/");
        if entry.is_dir() || !wanted(&path) {
            continue;
        }
        let name = file_name(&path).to_string();
        // Written under a temporary name so a failed run never leaves half a DLL.
        let partial = dir.join(format!("{name}.part"));
        io::copy(&mut entry, &mut File::create(&partial)?)?;
        std::fs::rename(&partial, dir.join(&name))?;
        written.push(name);
    }
    Ok(written)
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_filters_keep_only_the_needed_libraries() {
        assert!((ORT_CUDA.wanted)("onnxruntime-win-x64-gpu-1.21.0/lib/onnxruntime_providers_cuda.dll"));
        assert!(!(ORT_CUDA.wanted)("onnxruntime-win-x64-gpu-1.21.0/lib/onnxruntime.lib"));
        assert!(!(ORT_CUDA.wanted)("onnxruntime-win-x64-gpu-1.21.0/include/onnxruntime_c_api.h"));
        assert!((ORT_DIRECTML.wanted)("runtimes/win-x64/native/onnxruntime.dll"));
        assert!(!(ORT_DIRECTML.wanted)("runtimes/win-arm64/native/onnxruntime.dll"));
        assert!((DIRECTML.wanted)("bin/x64-win/DirectML.dll"));
        assert!(!(DIRECTML.wanted)("bin/x64-win/DirectML.Debug.dll"));
        let cudnn = CUDA[3].wanted;
        assert!(cudnn("nvidia/cudnn/bin/cudnn_ops64_9.dll"));
        assert!(!cudnn("nvidia/cudnn/include/cudnn.h"));
        assert!((CUDA[1].wanted)("nvidia/cublas/bin/cublasLt64_12.dll"));
        assert!(!(CUDA[1].wanted)("nvidia/cublas/bin/nvblas64_12.dll"));
    }

    #[test]
    fn every_archive_is_pinned() {
        for a in [ORT_CUDA, ORT_DIRECTML, DIRECTML].iter().chain(&CUDA) {
            assert_eq!(a.sha256.len(), 64, "{}", a.name);
            assert!(a.url.starts_with("https://"), "{}", a.name);
            assert!(a.bytes > 0);
        }
    }

    #[test]
    fn unpack_flattens_and_filters() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("halalify-unpack-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A stored (uncompressed) zip with two entries, written by hand.
        let zip_path = dir.join("t.zip");
        let mut w = zip::ZipWriter::new(File::create(&zip_path).unwrap());
        let options = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        w.start_file("pkg/lib/keep.dll", options).unwrap();
        w.write_all(b"library").unwrap();
        w.start_file("pkg/include/skip.h", options).unwrap();
        w.write_all(b"header").unwrap();
        w.finish().unwrap();

        let files = unpack(&zip_path, &dir, |e| e.ends_with(".dll")).unwrap();
        assert_eq!(files, ["keep.dll"]);
        assert_eq!(std::fs::read(dir.join("keep.dll")).unwrap(), b"library");
        assert!(!dir.join("skip.h").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
