//! Progress bars with one consistent look.

use indicatif::{ProgressBar, ProgressStyle};

/// A bar for a stage measured in seconds of media processed.
pub fn media_bar(label: &str, total_seconds: f64) -> ProgressBar {
    let bar = ProgressBar::new((total_seconds.max(0.0) * 1000.0) as u64).with_style(
        ProgressStyle::with_template("{prefix:>14} {bar:32.cyan/blue} {percent:>3}%  {msg}  eta {eta}")
            .expect("valid template")
            .progress_chars("=> "),
    );
    bar.set_prefix(label.to_string());
    bar
}

/// Moves a `media_bar` to `seconds` and shows the speed relative to real time.
pub fn set_media(bar: &ProgressBar, seconds: f64) {
    bar.set_position((seconds * 1000.0) as u64);
    let elapsed = bar.elapsed().as_secs_f64();
    if elapsed > 0.5 {
        bar.set_message(format!("{:.1}x realtime", seconds / elapsed));
    }
}
