mod audio;
mod cli;
mod models;
mod paths;
mod probe;
mod runtime;
mod tools;

use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Command, ToolArgs};
use tools::{ToolOverrides, Tools};

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Doctor(t) => doctor(&t),
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
    println!("home      {}", paths::home().display());
    match discover(args) {
        Ok(t) => {
            println!("ffmpeg    {}", t.ffmpeg.display());
            println!("ffprobe   {}", t.ffprobe.display());
            println!("encoder   {}", t.encoder);
            match &t.ytdlp {
                Some(p) => println!("yt-dlp    {}", p.display()),
                None => println!("yt-dlp    not found (only needed for links)"),
            }
            match &t.js_runtime {
                Some((name, p)) => println!("js        {name} ({})", p.display()),
                None => println!("js        none (YouTube may offer fewer formats)"),
            }
        }
        Err(e) => println!("tools     PROBLEM: {e:#}"),
    }
    Ok(())
}
