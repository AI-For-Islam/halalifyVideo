//! Progress reporting: bars for a person at a terminal, or one JSON object
//! per line on standard output for a program that drives this one (the
//! desktop app does).

use std::{
    io::Write,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use serde::Serialize;

/// How progress is reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    /// Progress bars, for a terminal.
    Bars,
    /// One JSON object per line on standard output, for other programs.
    Json,
}

static JSON: AtomicBool = AtomicBool::new(false);
/// When the last frequent event (a bar position) was written.
static LAST: Mutex<Option<Instant>> = Mutex::new(None);
/// Frequent events are written at most this often.
const EVERY: Duration = Duration::from_millis(120);

pub fn set_mode(mode: Mode) {
    JSON.store(mode == Mode::Json, Ordering::Relaxed);
}

pub fn json() -> bool {
    JSON.load(Ordering::Relaxed)
}

/// What a driving program is told. Each is written as one line such as
/// `{"event":"stage","stage":"finding people","done":12.5,"total":60.0}`.
#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event<'a> {
    /// A stage measured in seconds of media: `done` of `total` are finished.
    Stage { stage: &'a str, done: f64, total: f64 },
    /// A file being downloaded, in bytes.
    Download { name: &'a str, done: u64, total: u64 },
    /// A line of progress from yt-dlp while a link is fetched.
    Fetch { text: &'a str },
    /// `setup` has a runtime package or model in place.
    Installed { name: &'a str },
    /// `setup` finished; the models run on `device` (absent when not yet known).
    Ready { device: Option<&'a str> },
    /// The job's scratch folder. It is deleted when the job ends; a driving
    /// program that ends this one early should delete it itself.
    Work { dir: &'a str },
    /// The job finished and wrote `output`.
    Done { output: &'a str, seconds: f64 },
    /// The command failed.
    Error { message: &'a str },
}

/// Writes `event` when in JSON mode; does nothing otherwise.
pub fn emit(event: &Event) {
    if json() {
        write_line(event);
    }
}

/// Like `emit`, for events that arrive many times a second: all but one per
/// `EVERY` are dropped, unless `last` marks the one that must get through.
fn emit_frequent(event: &Event, last: bool) {
    if !json() {
        return;
    }
    let mut written = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if last || written.is_none_or(|t| t.elapsed() >= EVERY) {
        *written = Some(Instant::now());
        write_line(event);
    }
}

/// Serialises any value as one line of standard output.
pub fn write_line(value: &impl Serialize) {
    let mut out = std::io::stdout().lock();
    if serde_json::to_writer(&mut out, value).is_ok() {
        let _ = out.write_all(b"\n");
        let _ = out.flush();
    }
}

fn bar(length: u64, template: &str) -> ProgressBar {
    let target = if json() { ProgressDrawTarget::hidden() } else { ProgressDrawTarget::stderr() };
    ProgressBar::with_draw_target(Some(length), target)
        .with_style(ProgressStyle::with_template(template).expect("valid template").progress_chars("=> "))
}

/// A bar for a stage measured in seconds of media processed.
pub fn media_bar(label: &str, total_seconds: f64) -> ProgressBar {
    let bar = bar((total_seconds.max(0.0) * 1000.0) as u64, "{prefix:>14} {bar:32.cyan/blue} {percent:>3}%  {msg}  eta {eta}");
    bar.set_prefix(label.to_string());
    emit(&Event::Stage { stage: label, done: 0.0, total: total_seconds.max(0.0) });
    bar
}

/// Moves a `media_bar` to `seconds` and shows the speed relative to real time.
pub fn set_media(bar: &ProgressBar, seconds: f64) {
    bar.set_position((seconds * 1000.0) as u64);
    let elapsed = bar.elapsed().as_secs_f64();
    if elapsed > 0.5 {
        bar.set_message(format!("{:.1}x realtime", seconds / elapsed));
    }
    emit_frequent(&Event::Stage { stage: &bar.prefix(), done: seconds, total: media_total(bar) }, false);
}

/// Ends a `media_bar`: the stage is complete.
pub fn finish_media(bar: &ProgressBar) {
    let total = media_total(bar);
    emit_frequent(&Event::Stage { stage: &bar.prefix(), done: total, total }, true);
    bar.finish_and_clear();
}

fn media_total(bar: &ProgressBar) -> f64 {
    bar.length().unwrap_or(0) as f64 / 1000.0
}

/// A bar for a download of `total` bytes.
pub fn download_bar(label: &str, total: u64) -> ProgressBar {
    let bar = bar(total, "{msg:>24} {bar:32.cyan/blue} {bytes:>10}/{total_bytes} {bytes_per_sec} eta {eta}");
    bar.set_message(label.to_string());
    bar
}

/// Adds `bytes` to a `download_bar`.
pub fn add_download(bar: &ProgressBar, bytes: u64) {
    bar.inc(bytes);
    emit_frequent(&download_event(bar, &bar.message()), false);
}

/// Ends a `download_bar`: the file is complete.
pub fn finish_download(bar: &ProgressBar) {
    emit_frequent(&download_event(bar, &bar.message()), true);
    bar.finish_and_clear();
}

fn download_event<'a>(bar: &ProgressBar, name: &'a str) -> Event<'a> {
    let done = bar.position();
    Event::Download { name, done, total: bar.length().unwrap_or(0).max(done) }
}

/// Passes on a line of yt-dlp's progress.
pub fn fetching(text: &str) {
    emit_frequent(&Event::Fetch { text }, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(event: &Event) -> String {
        serde_json::to_string(event).unwrap()
    }

    #[test]
    fn events_are_tagged_objects() {
        assert_eq!(
            line(&Event::Stage { stage: "blurring", done: 1.5, total: 60.0 }),
            r#"{"event":"stage","stage":"blurring","done":1.5,"total":60.0}"#
        );
        assert_eq!(
            line(&Event::Download { name: "model.onnx", done: 10, total: 20 }),
            r#"{"event":"download","name":"model.onnx","done":10,"total":20}"#
        );
        assert_eq!(line(&Event::Ready { device: None }), r#"{"event":"ready","device":null}"#);
        assert_eq!(
            line(&Event::Done { output: r"C:\v\a.halal.mp4", seconds: 2.0 }),
            r#"{"event":"done","output":"C:\\v\\a.halal.mp4","seconds":2.0}"#
        );
    }

    #[test]
    fn a_hidden_bar_still_knows_its_stage_and_length() {
        let bar = ProgressBar::with_draw_target(Some(60_000), ProgressDrawTarget::hidden());
        bar.set_prefix("finding people");
        assert_eq!(bar.prefix(), "finding people");
        assert_eq!(media_total(&bar), 60.0);
    }
}
