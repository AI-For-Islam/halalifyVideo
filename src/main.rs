mod cli;
mod models;
mod paths;
mod probe;
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
    }
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
