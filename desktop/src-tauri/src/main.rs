//! The HalalifyVid window.
//!
//! This program draws the screens and nothing else. All the work is done by
//! the `halalify` command-line program, started as a child process with
//! `--progress json`; its events are passed on to the screens as they arrive.
//! Stopping a job is therefore just ending that process and its children.

// No console window behind the app in a release build.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod engine;

use std::path::{Path, PathBuf};

use serde::Deserialize;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;

use engine::Engine;

type Reply<T> = Result<T, String>;

/// What the home screen sends to start a job. Every choice is checked against
/// the words the `halalify` program accepts before it becomes an argument.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunOptions {
    /// A video file or a link.
    input: String,
    /// Folder for the result; empty means next to the original (for a link,
    /// the default folder).
    output_dir: String,
    blur: String,
    shape: String,
    style: String,
    threshold: f32,
    audio: String,
    music_model: String,
    /// Look for people on every other frame.
    faster: bool,
}

fn one_of<'a>(value: &'a str, allowed: &[&str], what: &str) -> Reply<&'a str> {
    if allowed.contains(&value) { Ok(value) } else { Err(format!("unknown {what}: {value}")) }
}

fn is_link(input: &str) -> bool {
    let lower = input.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// `Videos\HalalifyVid`, where results of links go unless another folder is chosen.
fn default_folder(app: &AppHandle) -> PathBuf {
    let videos = app.path().video_dir().or_else(|_| app.path().home_dir()).unwrap_or_else(|_| PathBuf::from("."));
    videos.join("HalalifyVid")
}

/// The arguments for `halalify run`, and the folder to run it in.
fn run_arguments(options: &RunOptions, default_folder: &Path) -> Reply<(Vec<String>, Option<PathBuf>)> {
    let input = options.input.trim().trim_matches('"').to_string();
    if input.is_empty() {
        return Err("no video was given".into());
    }
    let mut args: Vec<String> = vec!["run".into(), "--progress".into(), "json".into()];
    let mut choice = |flag: &str, value: &str| args.extend([flag.to_string(), value.to_string()]);
    choice("--blur", one_of(&options.blur, &["women", "men", "everyone", "nobody"], "blur target")?);
    choice("--shape", one_of(&options.shape, &["silhouette", "box"], "blur shape")?);
    choice("--style", one_of(&options.style, &["blur", "pixelate", "black"], "blur style")?);
    choice("--audio", one_of(&options.audio, &["clean", "keep", "mute"], "soundtrack choice")?);
    choice("--music-model", one_of(&options.music_model, &["auto", "dialogue", "vocals"], "music model")?);
    if !(0.0..=1.0).contains(&options.threshold) {
        return Err("the threshold must lie between 0 and 1".into());
    }
    choice("--threshold", &format!("{:.2}", options.threshold));
    if options.faster {
        choice("--stride", "2");
    }

    let chosen = options.output_dir.trim();
    let link = is_link(&input);
    let mut folder = None;
    if link {
        // The program names a link's result after its title, in the folder it runs in.
        folder = Some(if chosen.is_empty() { default_folder.to_path_buf() } else { PathBuf::from(chosen) });
    } else if !chosen.is_empty() {
        let stem = Path::new(&input).file_stem().unwrap_or_default().to_string_lossy().to_string();
        let output = Path::new(chosen).join(format!("{stem}.halal.mp4"));
        choice("--output", &output.to_string_lossy());
        folder = Some(PathBuf::from(chosen));
    }
    // After `--`, so that no file name can be mistaken for an option.
    args.extend(["--".to_string(), input]);
    Ok((args, folder))
}

/// What is installed: the `halalify doctor` report, as JSON.
#[tauri::command]
async fn status(engine: State<'_, Engine>) -> Reply<serde_json::Value> {
    engine.status()
}

/// Starts cleaning a video. Progress arrives as `engine` events.
#[tauri::command]
fn start_run(app: AppHandle, engine: State<'_, Engine>, options: RunOptions) -> Reply<()> {
    let (args, folder) = run_arguments(&options, &default_folder(&app))?;
    if let Some(folder) = &folder {
        std::fs::create_dir_all(folder).map_err(|e| format!("cannot create {}: {e}", folder.display()))?;
    }
    engine.start(app.clone(), args, folder)
}

/// Starts installing the runtime and the models. Progress arrives as `engine` events.
#[tauri::command]
fn start_setup(app: AppHandle, engine: State<'_, Engine>, provider: String) -> Reply<()> {
    let mut args: Vec<String> = vec!["setup".into(), "--progress".into(), "json".into()];
    if provider != "auto" {
        args.extend(["--provider".to_string(), one_of(&provider, &["cuda", "directml", "cpu"], "GPU route")?.to_string()]);
    }
    engine.start(app.clone(), args, None)
}

/// Installs ffmpeg and yt-dlp with winget, the way the README describes.
/// Progress arrives as `engine` events.
#[tauri::command]
fn install_tools(app: AppHandle, engine: State<'_, Engine>) -> Reply<()> {
    engine.start_winget(app.clone(), &["Gyan.FFmpeg", "yt-dlp.yt-dlp"])
}

/// Stops whatever is running.
#[tauri::command]
fn cancel(engine: State<'_, Engine>) {
    engine.cancel();
}

#[tauri::command]
async fn pick_video(app: AppHandle) -> Option<String> {
    app.dialog()
        .file()
        .add_filter("Video", &["mp4", "mkv", "mov", "webm", "avi", "m4v", "ts", "flv", "wmv"])
        .add_filter("Audio", &["mp3", "m4a", "wav", "aac", "ogg", "opus", "flac"])
        .add_filter("All files", &["*"])
        .blocking_pick_file()
        .and_then(|f| f.into_path().ok())
        .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
async fn pick_folder(app: AppHandle) -> Option<String> {
    app.dialog().file().blocking_pick_folder().and_then(|f| f.into_path().ok()).map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
fn default_output_folder(app: AppHandle) -> String {
    default_folder(&app).to_string_lossy().to_string()
}

/// Plays a result in the system's video player.
#[tauri::command]
fn open_file(app: AppHandle, path: String) -> Reply<()> {
    if !Path::new(&path).is_file() {
        return Err(format!("{path} is not there any more"));
    }
    app.opener().open_path(path, None::<&str>).map_err(|e| e.to_string())
}

/// Shows a result in the file manager.
#[tauri::command]
fn show_in_folder(app: AppHandle, path: String) -> Reply<()> {
    app.opener().reveal_item_in_dir(path).map_err(|e| e.to_string())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(Engine::default())
        .invoke_handler(tauri::generate_handler![
            status,
            start_run,
            start_setup,
            install_tools,
            cancel,
            pick_video,
            pick_folder,
            default_output_folder,
            open_file,
            show_in_folder,
        ])
        .on_window_event(|window, event| {
            // Closing the window must not leave a job running unseen.
            if let tauri::WindowEvent::Destroyed = event {
                window.state::<Engine>().cancel();
            }
        })
        .run(tauri::generate_context!())
        .expect("the HalalifyVid window could not start");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(input: &str) -> RunOptions {
        RunOptions {
            input: input.into(),
            output_dir: String::new(),
            blur: "women".into(),
            shape: "silhouette".into(),
            style: "blur".into(),
            threshold: 0.35,
            audio: "clean".into(),
            music_model: "auto".into(),
            faster: false,
        }
    }

    #[test]
    fn a_file_is_cleaned_next_to_itself_by_default() {
        let (args, folder) = run_arguments(&options(r"C:\v\talk.mp4"), Path::new(r"C:\out")).unwrap();
        assert_eq!(&args[..3], ["run", "--progress", "json"]);
        assert_eq!(&args[args.len() - 2..], ["--", r"C:\v\talk.mp4"]);
        assert!(!args.contains(&"--output".to_string()) && !args.contains(&"--stride".to_string()));
        assert_eq!(folder, None);
    }

    #[test]
    fn a_chosen_folder_receives_the_result_of_a_file() {
        let mut o = options(r#""C:\v\My Talk.mkv""#);
        o.output_dir = r"D:\clean".into();
        o.faster = true;
        let (args, folder) = run_arguments(&o, Path::new(r"C:\out")).unwrap();
        let at = args.iter().position(|a| a == "--output").unwrap();
        assert_eq!(Path::new(&args[at + 1]), Path::new(r"D:\clean").join("My Talk.halal.mp4"));
        assert!(args.windows(2).any(|w| w == ["--stride", "2"]));
        assert_eq!(args.last().unwrap(), r"C:\v\My Talk.mkv");
        assert_eq!(folder, Some(PathBuf::from(r"D:\clean")));
    }

    #[test]
    fn a_link_runs_in_the_default_folder() {
        let (args, folder) = run_arguments(&options("https://youtu.be/abc"), Path::new(r"C:\out")).unwrap();
        assert!(!args.contains(&"--output".to_string()));
        assert_eq!(folder, Some(PathBuf::from(r"C:\out")));
    }

    #[test]
    fn only_known_choices_become_arguments() {
        let mut o = options("a.mp4");
        o.blur = "--output".into();
        assert!(run_arguments(&o, Path::new(".")).is_err());
        let mut o = options("a.mp4");
        o.threshold = 1.5;
        assert!(run_arguments(&o, Path::new(".")).is_err());
        assert!(run_arguments(&options("   "), Path::new(".")).is_err());
    }

    #[test]
    fn an_input_that_looks_like_an_option_stays_an_input() {
        let (args, _) = run_arguments(&options("--setup.mp4"), Path::new(".")).unwrap();
        assert_eq!(&args[args.len() - 2..], ["--", "--setup.mp4"]);
    }
}
