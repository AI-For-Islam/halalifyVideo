mod audio;
mod cli;
mod fetch;
mod models;
mod paths;
mod pipeline;
mod probe;
mod progress;
mod render;
mod runtime;
mod setup;
mod tools;
mod vision;
mod workdir;

use std::{ffi::OsString, process::ExitCode};

use anyhow::Result;
use clap::{CommandFactory, Parser};

use cli::{Cli, Command, ToolArgs};
use tools::{ToolOverrides, Tools};

fn main() -> ExitCode {
    match run(Cli::parse_from(args_with_default_command())) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// `halalify video.mp4` means `halalify run video.mp4`: when the first
/// argument is neither a command nor a flag, the `run` command is implied.
fn args_with_default_command() -> Vec<OsString> {
    let mut args: Vec<OsString> = std::env::args_os().collect();
    let cli = Cli::command();
    let is_command = |a: &str| a == "help" || cli.get_subcommands().any(|c| c.get_name() == a);
    if let Some(first) = args.get(1).and_then(|a| a.to_str()) {
        if !first.starts_with('-') && !is_command(first) {
            args.insert(1, "run".into());
        }
    }
    args
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Run(args) => pipeline::run(&args),
        Command::Doctor(t) => doctor(&t),
        Command::Setup { provider, skip_models } => setup::run(provider, skip_models),
        Command::Probe { input, tools } => {
            let tools = discover(&tools)?;
            let info = probe::probe(&tools, &input)?;
            println!("{}", serde_json::to_string_pretty(&info)?);
            Ok(())
        }
        Command::Models => {
            for m in models::ALL {
                let path = m.fetch()?;
                println!("{:<44} {}", m.file, models::sha256_file(&path)?);
            }
            Ok(())
        }
        Command::Bench { model, shape, device, runs } => bench(&model, &shape, device, runs),
    }
}

/// Runs `model` on a constant input `runs` times and prints the mean latency.
fn bench(model: &std::path::Path, shape: &[usize], device: runtime::Device, runs: usize) -> Result<()> {
    use ort::value::Tensor;
    let rt = runtime::Runtime::new(device)?;
    let t0 = std::time::Instant::now();
    let mut session = rt.session(model)?;
    println!("loaded on {} in {:.1}s", rt.device().unwrap(), t0.elapsed().as_secs_f32());
    for i in session.inputs() {
        println!("  in  {:<24} {:?}", i.name(), i.dtype());
    }
    for o in session.outputs() {
        println!("  out {:<24} {:?}", o.name(), o.dtype());
    }
    let name = session.inputs()[0].name().to_string();
    let len: usize = shape.iter().product();
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let mut times = Vec::new();
    for _ in 0..runs + 3 {
        let input = Tensor::from_array((dims.clone(), vec![0.5f32; len]))?;
        let t = std::time::Instant::now();
        let out = session.run(ort::inputs![name.as_str() => input])?;
        std::hint::black_box(&out);
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let steady = &times[3..];
    let mean = steady.iter().sum::<f64>() / steady.len() as f64;
    let batch = shape[0] as f64;
    println!("first {:.0} ms, steady {:.2} ms/run = {:.0} items/s", times[0], mean, batch * 1e3 / mean);
    Ok(())
}

fn discover(args: &ToolArgs) -> Result<Tools> {
    let over = ToolOverrides { ffmpeg: args.ffmpeg.clone(), encoder: args.encoder.clone(), ytdlp: None };
    Tools::discover(args.codec, &over)
}

fn doctor(args: &ToolArgs) -> Result<()> {
    println!("home         {}", paths::home().display());
    match discover(args) {
        Ok(t) => {
            println!("ffmpeg       {}", t.ffmpeg.display());
            println!("ffprobe      {}", t.ffprobe.display());
            println!("encoder      {}", t.encoder);
            match &t.ytdlp {
                Some(p) => println!("yt-dlp       {}", p.display()),
                None => println!("yt-dlp       not found (only needed for links)"),
            }
            match &t.js_runtime {
                Some((name, p)) => println!("js runtime   {name} ({})", p.display()),
                None => println!("js runtime   none (YouTube may offer fewer formats)"),
            }
        }
        Err(e) => println!("tools        PROBLEM: {e:#}"),
    }
    match runtime::find_onnxruntime() {
        Some(p) => println!("onnxruntime  {}", p.display()),
        None => println!("onnxruntime  not installed: run `halalify setup`"),
    }
    for m in models::ALL {
        let state = if m.is_present() { "installed" } else { "missing (downloads on first use)" };
        println!("model        {:<44} {state}", m.file);
        println!("             {} [{}]", m.about, m.license);
    }
    if runtime::find_onnxruntime().is_some() {
        match runtime::Runtime::new(runtime::Device::Auto) {
            Ok(rt) => {
                if let Some(dir) = rt.cuda_dir() {
                    println!("cuda libs    {}", dir.display());
                }
                // Loading a model is the only real proof that a provider works.
                match models::ALL.iter().find(|m| m.is_present()).map(|m| rt.session(&m.path())) {
                    Some(Ok(_)) => println!("models run   on {}", rt.device().map_or("?".into(), |d| d.to_string())),
                    Some(Err(e)) => println!("models run   PROBLEM: {e:#}"),
                    None => println!("models run   unknown until a model is installed"),
                }
            }
            Err(e) => println!("onnxruntime  PROBLEM: {e:#}"),
        }
    }
    Ok(())
}
