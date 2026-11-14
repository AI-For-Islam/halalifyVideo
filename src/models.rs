//! The model registry: which ONNX files HalalifyVid uses, where they come
//! from, and fetching them on first use.
//!
//! Models are plain data files kept in `<home>/models`. A file that is already
//! there is used as it is, so models exported locally (see `tools/`) or copied
//! in by hand need no network access.

use std::{
    fs::File,
    io::{Read, Write},
    path::PathBuf,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

use crate::paths;

#[derive(Debug, Clone, Copy)]
pub struct Model {
    /// File name inside the models directory.
    pub file: &'static str,
    /// Direct download link, or empty when the file must be exported locally.
    pub url: &'static str,
    /// Lower-case hex SHA-256 of the file; downloads are rejected on mismatch.
    pub sha256: &'static str,
    pub bytes: u64,
    pub license: &'static str,
    pub about: &'static str,
}

impl Model {
    pub fn path(&self) -> PathBuf {
        paths::models_dir().join(self.file)
    }

    pub fn is_present(&self) -> bool {
        self.path().is_file()
    }

    /// Returns the local path, downloading the file first if it is missing.
    pub fn fetch(&self) -> Result<PathBuf> {
        let dest = self.path();
        if dest.is_file() {
            return Ok(dest);
        }
        if self.url.is_empty() {
            bail!(
                "model {} is not installed and has no download link; create it with the export \
                 script in tools/ and place it in {}",
                self.file,
                paths::models_dir().display()
            );
        }
        paths::ensure(paths::models_dir())?;
        let part = dest.with_extension("part");
        download(self.url, &part, self.bytes, self.file)
            .with_context(|| format!("downloading {} from {}", self.file, self.url))?;
        let got = sha256_file(&part)?;
        if !self.sha256.is_empty() && got != self.sha256 {
            let _ = std::fs::remove_file(&part);
            bail!("{}: checksum mismatch (expected {}, got {got}); the download was discarded", self.file, self.sha256);
        }
        std::fs::rename(&part, &dest).with_context(|| format!("cannot move {} into place", dest.display()))?;
        Ok(dest)
    }
}

fn download(url: &str, dest: &std::path::Path, expected: u64, label: &str) -> Result<()> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .build()
        .into();
    let mut response = agent.get(url).header("User-Agent", "halalify").call()?;
    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(expected);

    let bar = ProgressBar::new(total).with_style(
        ProgressStyle::with_template("{msg:>24} {bar:32.cyan/blue} {bytes:>10}/{total_bytes} {bytes_per_sec} eta {eta}")
            .expect("valid template")
            .progress_chars("=> "),
    );
    bar.set_message(label.to_string());

    let mut reader = response.body_mut().as_reader();
    let mut file = File::create(dest)?;
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        bar.inc(n as u64);
    }
    file.flush()?;
    bar.finish_and_clear();
    Ok(())
}

pub fn sha256_file(path: &std::path::Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// MDX-Net vocal model by KimberleyJensen, as distributed with Ultimate Vocal
/// Remover. Keeps any human voice (speech or singing) and drops instruments.
pub const MDX_KIM_VOCAL_2: Model = Model {
    file: "Kim_Vocal_2.onnx",
    url: "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx",
    sha256: "ce74ef3b6a6024ce44211a07be9cf8bc6d87728cc852a68ab34eb8e58cde9c8b",
    bytes: 66_759_214,
    license: "MIT (model by KimberleyJensen, UVR model repo)",
    about: "MDX-Net vocals: keeps voices, removes instruments",
};

/// RF-DETR-Seg Small (Roboflow): finds people and their outlines in one pass.
/// ONNX export hosted by the `usls` project.
pub const RFDETR_SEG_S: Model = Model {
    file: "rfdetr-seg-small-fp16.onnx",
    url: "https://github.com/jamjamjon/assets/releases/download/rfdetr/seg-small-fp16.onnx",
    sha256: "c7d3bc18bec5d886557515afd543de32f016b3720c2bca7458402529a69001b1",
    bytes: 61_897_886,
    license: "Apache-2.0 (Roboflow RF-DETR)",
    about: "RF-DETR-Seg S: person detection with instance masks",
};

/// SigLIP 2 base image encoder (Google), used zero-shot to tell who a detected
/// person is. ONNX export by the onnx-community organisation.
pub const SIGLIP2_VISION: Model = Model {
    file: "siglip2-base-patch16-256-vision-fp16.onnx",
    url: "https://huggingface.co/onnx-community/siglip2-base-patch16-256-ONNX/resolve/main/onnx/vision_model_fp16.onnx",
    sha256: "fe9ad8020a6d3d98d394c9be8f07064066135fc2f87ec11692de0b677c0ac4db",
    bytes: 186_131_676,
    license: "Apache-2.0 (Google SigLIP 2)",
    about: "SigLIP 2 B/16-256 image encoder: zero-shot person classification",
};

pub const ALL: &[Model] = &[MDX_KIM_VOCAL_2, RFDETR_SEG_S, SIGLIP2_VISION];
