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
use sha2::{Digest, Sha256};

use crate::{paths, progress};

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

/// Downloads `url` to `dest` with a progress bar. `expected` is the size
/// shown when the server does not report one.
pub fn download(url: &str, dest: &std::path::Path, expected: u64, label: &str) -> Result<()> {
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

    let bar = progress::download_bar(label, total);

    let mut reader = response.body_mut().as_reader();
    let mut file = File::create(dest)?;
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        progress::add_download(&bar, n as u64);
    }
    file.flush()?;
    progress::finish_download(&bar);
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

/// YOLO26s-seg (Ultralytics): finds people and their outlines in one pass,
/// with no non-maximum suppression step. ONNX export (fp16 weights, float32
/// inputs and outputs, dynamic size) hosted by the `usls` project.
pub const YOLO26S_SEG: Model = Model {
    file: "yolo26s-seg-fp16.onnx",
    url: "https://github.com/jamjamjon/assets/releases/download/yolo/v26-s-seg-fp16.onnx",
    sha256: "dba035e75fa52c53f20a9fd44d49122bb32fe1d278ac87bcdab0dfafff187a12",
    bytes: 21_014_540,
    license: "AGPL-3.0 (Ultralytics YOLO26)",
    about: "YOLO26s-seg: person detection with instance masks",
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

/// BandIt Plus (speech / music / effects), the better music remover for
/// film and broadcast audio. It is published as a PyTorch checkpoint only, so
/// there is nothing to download here: `tools/export_bandit.py` converts the
/// checkpoint once, and the result is used whenever it is present.
pub const BANDIT_PLUS: Model = Model {
    file: "bandit-plus-dnr.onnx",
    url: "",
    sha256: "",
    bytes: 98_000_000,
    license: "MIT repository (ZFTurbo Music-Source-Separation-Training)",
    about: "BandIt Plus: speech / music / effects",
};

/// Models every installation needs; `setup` downloads them.
pub const ALL: &[Model] = &[MDX_KIM_VOCAL_2, YOLO26S_SEG, SIGLIP2_VISION];

/// Models that improve results when present but are not downloadable.
pub const OPTIONAL: &[Model] = &[BANDIT_PLUS];
