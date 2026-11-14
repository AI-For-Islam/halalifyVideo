//! Music removal: decode the soundtrack, keep the voice, write a WAV.

pub mod io;
mod mdx;
mod ola;
mod stft;

use std::{path::Path, time::Instant};

use anyhow::Result;

use self::{
    io::{PcmReader, Span, WavWriter},
    mdx::MdxNet,
    ola::OlaPlan,
};
use crate::{models, runtime::Runtime, tools::Tools};

/// Sample rate the separation model works at; the cleaned track is written at this rate.
pub const SAMPLE_RATE: u32 = 44_100;

pub struct AudioReport {
    /// Length of the cleaned audio in seconds.
    pub seconds: f64,
    /// Wall-clock seconds spent.
    pub elapsed: f64,
}

/// Writes the voice-only version of `input`'s first audio stream to `out_wav`.
///
/// `progress` is called with the seconds of audio finished so far.
pub fn remove_music(
    tools: &Tools,
    rt: &Runtime,
    input: &Path,
    out_wav: &Path,
    span: Span,
    mut progress: impl FnMut(f64),
) -> Result<AudioReport> {
    let started = Instant::now();
    let mut net = MdxNet::load(rt, &models::MDX_KIM_VOCAL_2, mdx::KIM_VOCAL_2)?;
    let chunk = net.chunk_len();
    // A quarter of each chunk overlaps the next; the Hann cross-fade gives the
    // unreliable chunk edges almost no weight.
    let plan = OlaPlan { chunk, step: chunk * 3 / 4, lead: net.edge(), batch: 1 };

    let mut reader = PcmReader::open(tools, input, SAMPLE_RATE, span)?;
    let mut writer = WavWriter::create(out_wav, SAMPLE_RATE)?;
    let mut done = 0u64;
    let written = ola::run(
        &plan,
        |l, r| reader.read(l, r, SAMPLE_RATE as usize),
        |chunks| net.separate(chunks),
        |l, r| {
            writer.write(l, r)?;
            done += l.len() as u64;
            progress(done as f64 / SAMPLE_RATE as f64);
            Ok(())
        },
    )?;
    reader.finish()?;
    writer.finish()?;
    Ok(AudioReport { seconds: written as f64 / SAMPLE_RATE as f64, elapsed: started.elapsed().as_secs_f64() })
}
