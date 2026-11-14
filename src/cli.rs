//! Command-line interface definitions.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::{runtime::Device, tools::Codec};

#[derive(Parser, Debug)]
#[command(
    name = "halalify",
    version,
    about = "Removes music from a video and blurs the people you choose - locally, on your GPU.",
    after_help = "Examples:\n  halalify lecture.mp4\n  halalify https://youtu.be/VIDEO_ID -o clean.mp4\n  halalify doctor"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Show which tools, encoders, GPU providers and models this machine has.
    Doctor(ToolArgs),
    /// Print what ffprobe sees in a file, as HalalifyVid interprets it.
    Probe {
        input: PathBuf,
        #[command(flatten)]
        tools: ToolArgs,
    },
    /// Download every model now instead of on first use.
    Models,
    /// Time a model on random input (developer aid).
    #[command(hide = true)]
    Bench {
        model: PathBuf,
        /// Input shape, e.g. 1,3,384,384
        #[arg(long, value_delimiter = ',', required = true)]
        shape: Vec<usize>,
        #[arg(long, value_enum, default_value = "auto")]
        device: Device,
        #[arg(long, default_value_t = 30)]
        runs: usize,
    },
}

/// Options for locating external tools; shared by every command.
#[derive(Args, Debug, Clone)]
pub struct ToolArgs {
    /// Path to ffmpeg (default: search PATH and common install folders).
    #[arg(long, value_name = "PATH")]
    pub ffmpeg: Option<PathBuf>,
    /// Force a video encoder such as h264_nvenc or libx264 (default: fastest that works).
    #[arg(long, value_name = "NAME")]
    pub encoder: Option<String>,
    /// Output video codec.
    #[arg(long, value_enum, default_value = "h264")]
    pub codec: Codec,
}
