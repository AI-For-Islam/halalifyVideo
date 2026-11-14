//! Command-line interface definitions.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::{
    audio::MusicModel,
    render::Style,
    runtime::Device,
    setup::Provider,
    tools::Codec,
    vision::decide::{Shape, Target},
};

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
    /// Clean a video (the default command: `halalify <INPUT>` works too).
    Run(RunArgs),
    /// Show which tools, encoders, GPU providers and models this machine has.
    Doctor(ToolArgs),
    /// Print what ffprobe sees in a file, as HalalifyVid interprets it.
    Probe {
        input: PathBuf,
        #[command(flatten)]
        tools: ToolArgs,
    },
    /// Install ONNX Runtime, the GPU libraries and the models (run once).
    Setup {
        /// GPU route (default: CUDA when an NVIDIA driver is present, else DirectML).
        #[arg(long, value_enum)]
        provider: Option<Provider>,
        /// Leave the models to be downloaded on first use.
        #[arg(long)]
        skip_models: bool,
    },
    /// Download every model now instead of on first use.
    Models,
    /// Time a model on constant input (developer aid).
    #[command(hide = true)]
    Bench {
        model: PathBuf,
        /// Input shape, e.g. 1,3,384,640
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum AudioMode {
    /// Keep voices, remove music.
    Clean,
    /// Leave the soundtrack as it is.
    Keep,
    /// Drop the soundtrack.
    Mute,
}

#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    /// A video file, or a link to a video (YouTube and whatever else yt-dlp supports).
    pub input: String,
    /// Where to write the result (default: <name>.halal.mp4 next to the input,
    /// or in the current folder for a link).
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,
    /// Who to blur.
    #[arg(long, value_enum, default_value = "women")]
    pub blur: Target,
    /// Area covered for each blurred person.
    #[arg(long, value_enum, default_value = "silhouette")]
    pub shape: Shape,
    /// How a blurred area looks.
    #[arg(long, value_enum, default_value = "blur")]
    pub style: Style,
    /// Probability of "woman" from which a person is blurred (lower blurs more readily).
    #[arg(long, default_value_t = 0.35, value_name = "0..1")]
    pub threshold: f32,
    /// What to do with the soundtrack.
    #[arg(long, value_enum, default_value = "clean")]
    pub audio: AudioMode,
    /// Which model removes the music.
    #[arg(long, value_enum, default_value = "auto")]
    pub music_model: MusicModel,
    /// Keep sound effects as well as speech (dialogue model only).
    #[arg(long)]
    pub keep_effects: bool,
    /// Process only from this time, in seconds.
    #[arg(long, value_name = "SECONDS")]
    pub start: Option<f64>,
    /// Process only this many seconds.
    #[arg(long, value_name = "SECONDS")]
    pub duration: Option<f64>,
    /// Where to run the models.
    #[arg(long, value_enum, default_value = "auto")]
    pub device: Device,
    /// Video quality when re-encoding: lower is better and bigger (18 near-lossless, 28 small).
    #[arg(long, default_value_t = 21, value_parser = clap::value_parser!(u8).range(10..=40))]
    pub quality: u8,
    /// For links: the tallest picture to download, in pixels.
    #[arg(long, default_value_t = 1080, value_name = "PIXELS")]
    pub max_height: u32,
    /// Look for people on every n-th frame only (default: every frame up to 30 fps).
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..=16))]
    pub stride: Option<u32>,
    /// Write a JSON account of every person found and the decision taken.
    #[arg(long, value_name = "FILE")]
    pub report: Option<PathBuf>,
    /// Keep the temporary work files for inspection.
    #[arg(long)]
    pub keep_temp: bool,
    #[command(flatten)]
    pub tools: ToolArgs,
}
