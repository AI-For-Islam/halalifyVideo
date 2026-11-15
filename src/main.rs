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
    runtime::prepare_linux_library_path();
    let cli = Cli::parse_from(args_with_default_command());
    progress::set_mode(cli.progress);
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            progress::emit(&progress::Event::Error { message: &format!("{e:#}") });
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
        Command::Doctor(t) if progress::json() => status(&t),
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

/// What `doctor` reports to a driving program: one JSON object.
#[derive(serde::Serialize)]
struct Status {
    event: &'static str,
    version: &'static str,
    home: String,
    ffmpeg: Option<String>,
    ffprobe: Option<String>,
    encoder: Option<String>,
    ytdlp: Option<String>,
    js_runtime: Option<String>,
    /// Why ffmpeg cannot be used, when it cannot.
    tools_problem: Option<String>,
    onnxruntime: Option<String>,
    /// The GPU route `setup` would choose on this machine.
    provider: setup::Provider,
    models: Vec<ModelStatus>,
    /// Whether a video can be cleaned right now.
    ready: bool,
}

#[derive(serde::Serialize)]
struct ModelStatus {
    file: &'static str,
    installed: bool,
    optional: bool,
    bytes: u64,
    about: &'static str,
    license: &'static str,
}

/// `doctor` for a program: what is installed, without loading a model (which
/// takes seconds; `setup` proves that the models run).
fn status(args: &ToolArgs) -> Result<()> {
    let show = |p: &std::path::Path| p.display().to_string();
    let tools = discover(args);
    let onnxruntime = runtime::find_onnxruntime();
    let model = |m: &models::Model, optional| ModelStatus {
        file: m.file,
        installed: m.is_present(),
        optional,
        bytes: m.bytes,
        about: m.about,
        license: m.license,
    };
    let models: Vec<ModelStatus> =
        models::ALL.iter().map(|m| model(m, false)).chain(models::OPTIONAL.iter().map(|m| model(m, true))).collect();
    let ready = tools.is_ok() && onnxruntime.is_some() && models.iter().all(|m| m.installed || m.optional);
    let t = tools.as_ref().ok();
    progress::write_line(&Status {
        event: "status",
        version: env!("CARGO_PKG_VERSION"),
        home: show(&paths::home()),
        ffmpeg: t.map(|t| show(&t.ffmpeg)),
        ffprobe: t.map(|t| show(&t.ffprobe)),
        encoder: t.map(|t| t.encoder.to_string()),
        ytdlp: t.and_then(|t| t.ytdlp.as_deref().map(show)),
        js_runtime: t.and_then(|t| t.js_runtime.as_ref().map(|(name, _)| name.to_string())),
        tools_problem: tools.as_ref().err().map(|e| format!("{e:#}")),
        onnxruntime: onnxruntime.as_deref().map(show),
        provider: setup::detect_provider(),
        models,
        ready,
    });
    Ok(())
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
    for m in models::OPTIONAL {
        let state = if m.is_present() { "installed" } else { "not installed (optional; see the README)" };
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
