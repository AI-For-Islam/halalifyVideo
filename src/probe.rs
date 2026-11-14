//! Reading what is inside a media file with ffprobe.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::tools::{self, Tools};

/// A frame rate kept as an exact fraction: `30000/1001` must not become `29.97`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rational {
    pub num: u32,
    pub den: u32,
}

impl Rational {
    pub fn parse(s: &str) -> Option<Rational> {
        let (n, d) = s.split_once('/').unwrap_or((s, "1"));
        let (num, den) = (n.trim().parse().ok()?, d.trim().parse().ok()?);
        (num > 0 && den > 0).then_some(Rational { num, den })
    }

    pub fn as_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.num, self.den)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct VideoStream {
    /// Size as displayed, i.e. after applying any rotation flag.
    pub width: u32,
    pub height: u32,
    /// Constant rate every pass normalises the video to.
    pub fps: Rational,
    pub codec: String,
    pub pix_fmt: String,
    /// True when the container timestamps are not evenly spaced.
    pub variable_rate: bool,
    /// Bits per second of the video stream, when the file says.
    pub bit_rate: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AudioStream {
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct MediaInfo {
    pub duration: f64,
    pub video: Option<VideoStream>,
    pub audio: Option<AudioStream>,
}

#[derive(Deserialize)]
struct RawProbe {
    #[serde(default)]
    streams: Vec<RawStream>,
    format: Option<RawFormat>,
}

#[derive(Deserialize)]
struct RawFormat {
    duration: Option<String>,
    bit_rate: Option<String>,
}

#[derive(Deserialize, Default)]
struct RawStream {
    codec_type: Option<String>,
    codec_name: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    pix_fmt: Option<String>,
    r_frame_rate: Option<String>,
    avg_frame_rate: Option<String>,
    duration: Option<String>,
    bit_rate: Option<String>,
    sample_rate: Option<String>,
    channels: Option<u32>,
    #[serde(default)]
    disposition: RawDisposition,
    #[serde(default)]
    tags: RawTags,
    #[serde(default)]
    side_data_list: Vec<RawSideData>,
}

#[derive(Deserialize, Default)]
struct RawDisposition {
    #[serde(default)]
    attached_pic: u8,
}

#[derive(Deserialize, Default)]
struct RawTags {
    rotate: Option<String>,
}

#[derive(Deserialize)]
struct RawSideData {
    rotation: Option<f64>,
}

pub fn probe(tools: &Tools, input: &Path) -> Result<MediaInfo> {
    let json = tools::capture(
        &tools.ffprobe,
        [
            "-v".as_ref(),
            "error".as_ref(),
            "-print_format".as_ref(),
            "json".as_ref(),
            "-show_format".as_ref(),
            "-show_streams".as_ref(),
            input.as_os_str(),
        ],
    )
    .with_context(|| format!("cannot read {}", input.display()))?;
    parse(&json)
}

fn parse(json: &str) -> Result<MediaInfo> {
    let raw: RawProbe = serde_json::from_str(json).context("unexpected ffprobe output")?;
    let secs = |s: &Option<String>| s.as_deref().and_then(|d| d.parse::<f64>().ok()).filter(|d| *d > 0.0);

    // Cover art is exposed as a one-frame "video" stream; it is not the video.
    let v = raw
        .streams
        .iter()
        .find(|s| s.codec_type.as_deref() == Some("video") && s.disposition.attached_pic == 0);
    let a = raw.streams.iter().find(|s| s.codec_type.as_deref() == Some("audio"));

    let duration = raw
        .format
        .as_ref()
        .and_then(|f| secs(&f.duration))
        .or_else(|| v.and_then(|s| secs(&s.duration)))
        .or_else(|| a.and_then(|s| secs(&s.duration)))
        .unwrap_or(0.0);

    let video = match v {
        None => None,
        Some(s) => {
            let (Some(w), Some(h)) = (s.width, s.height) else { bail!("video stream has no size") };
            let rotation = s
                .side_data_list
                .iter()
                .find_map(|d| d.rotation)
                .or_else(|| s.tags.rotate.as_deref().and_then(|r| r.parse().ok()))
                .unwrap_or(0.0);
            let sideways = (rotation.abs() as i64 % 180) == 90;
            let (width, height) = if sideways { (h, w) } else { (w, h) };
            let (fps, variable_rate) = pick_rate(
                s.r_frame_rate.as_deref().and_then(Rational::parse),
                s.avg_frame_rate.as_deref().and_then(Rational::parse),
            );
            let bits = |b: &Option<String>| b.as_deref().and_then(|b| b.parse::<u64>().ok()).filter(|b| *b > 0);
            // Some containers only give a total; the soundtrack's share is taken off it.
            let bit_rate = bits(&s.bit_rate).or_else(|| {
                let total = bits(&raw.format.as_ref()?.bit_rate)?;
                let audio = a.map_or(0, |a| bits(&a.bit_rate).unwrap_or(128_000));
                Some(total.saturating_sub(audio)).filter(|b| *b > 0)
            });
            Some(VideoStream {
                width,
                height,
                fps,
                codec: s.codec_name.clone().unwrap_or_default(),
                pix_fmt: s.pix_fmt.clone().unwrap_or_default(),
                variable_rate,
                bit_rate,
            })
        }
    };
    let audio = a.map(|s| AudioStream {
        codec: s.codec_name.clone().unwrap_or_default(),
        sample_rate: s.sample_rate.as_deref().and_then(|r| r.parse().ok()).unwrap_or(44_100),
        channels: s.channels.unwrap_or(2),
    });
    if video.is_none() && audio.is_none() {
        bail!("no video or audio stream found");
    }
    Ok(MediaInfo { duration, video, audio })
}

/// Chooses the constant rate to resample the video to.
///
/// For constant-rate files `r_frame_rate` is exact. For variable-rate files
/// (phone recordings, screen captures) it is the smallest frame interval and
/// can be absurd (`90000/1`), so the average rate is used instead, capped at
/// 60 fps so a stray timestamp cannot explode the frame count.
fn pick_rate(real: Option<Rational>, avg: Option<Rational>) -> (Rational, bool) {
    const FALLBACK: Rational = Rational { num: 30, den: 1 };
    match (real, avg) {
        (Some(r), Some(a)) => {
            let ratio = r.as_f64() / a.as_f64();
            if (0.99..=1.01).contains(&ratio) {
                (r, false)
            } else {
                (round_rate(a.as_f64().min(60.0)), true)
            }
        }
        (Some(r), None) if r.as_f64() <= 240.0 => (r, false),
        (_, Some(a)) => (round_rate(a.as_f64().min(60.0)), true),
        _ => (FALLBACK, true),
    }
}

fn round_rate(fps: f64) -> Rational {
    const COMMON: [(u32, u32); 9] =
        [(24000, 1001), (24, 1), (25, 1), (30000, 1001), (30, 1), (48, 1), (50, 1), (60000, 1001), (60, 1)];
    for (num, den) in COMMON {
        if (fps - num as f64 / den as f64).abs() < 0.02 {
            return Rational { num, den };
        }
    }
    Rational { num: (fps * 1000.0).round().max(1000.0) as u32, den: 1000 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_rate_keeps_the_exact_fraction() {
        let (r, vfr) = pick_rate(Rational::parse("30000/1001"), Rational::parse("30000/1001"));
        assert_eq!((r, vfr), (Rational { num: 30000, den: 1001 }, false));
    }

    #[test]
    fn variable_rate_uses_the_rounded_average() {
        let (r, vfr) = pick_rate(Rational::parse("90000/1"), Rational::parse("2997/100"));
        assert_eq!((r, vfr), (Rational { num: 30000, den: 1001 }, true));
        let (r, _) = pick_rate(Rational::parse("120/1"), Rational::parse("3412/113"));
        assert!((r.as_f64() - 30.195).abs() < 0.01);
    }

    #[test]
    fn zero_rates_are_rejected() {
        assert_eq!(Rational::parse("0/0"), None);
        assert_eq!(Rational::parse("25"), Some(Rational { num: 25, den: 1 }));
    }

    #[test]
    fn rotated_phone_video_reports_display_size() {
        let json = r#"{"streams":[{"codec_type":"video","codec_name":"hevc","width":1920,"height":1080,
            "pix_fmt":"yuv420p","r_frame_rate":"30/1","avg_frame_rate":"30/1",
            "side_data_list":[{"side_data_type":"Display Matrix","rotation":-90}]},
            {"codec_type":"audio","codec_name":"aac","sample_rate":"48000","channels":2}],
            "format":{"duration":"12.5"}}"#;
        let info = parse(json).unwrap();
        let v = info.video.clone().unwrap();
        assert_eq!((v.width, v.height), (1080, 1920));
        assert_eq!(info.audio.unwrap().sample_rate, 48000);
        assert_eq!(info.duration, 12.5);
        assert_eq!(v.bit_rate, None);
    }

    #[test]
    fn video_bit_rate_comes_from_the_stream_or_the_container() {
        let own = r#"{"streams":[{"codec_type":"video","width":640,"height":360,"r_frame_rate":"25/1",
            "avg_frame_rate":"25/1","bit_rate":"900000"}],"format":{"duration":"10","bit_rate":"1100000"}}"#;
        assert_eq!(parse(own).unwrap().video.unwrap().bit_rate, Some(900_000));
        let total_only = r#"{"streams":[{"codec_type":"video","width":640,"height":360,"r_frame_rate":"25/1",
            "avg_frame_rate":"25/1"},{"codec_type":"audio","bit_rate":"100000","sample_rate":"44100","channels":2}],
            "format":{"duration":"10","bit_rate":"1100000"}}"#;
        assert_eq!(parse(total_only).unwrap().video.unwrap().bit_rate, Some(1_000_000));
    }

    #[test]
    fn cover_art_is_not_treated_as_video() {
        let json = r#"{"streams":[{"codec_type":"video","codec_name":"mjpeg","width":500,"height":500,
            "disposition":{"attached_pic":1}},
            {"codec_type":"audio","codec_name":"mp3","sample_rate":"44100","channels":2}],
            "format":{"duration":"200.0"}}"#;
        let info = parse(json).unwrap();
        assert!(info.video.is_none());
        assert!(info.audio.is_some());
    }
}
