//! Music removal: decode the soundtrack, keep the voice, write a WAV.

mod bandit;
pub mod io;
mod mdx;
mod ola;
mod spectral;
mod stft;

use std::{path::Path, sync::mpsc::sync_channel, time::Instant};

use anyhow::{Result, anyhow};

use self::{
    bandit::{Bandit, Keep},
    io::{PcmReader, Span, WavWriter},
    mdx::MdxNet,
    ola::{OlaPlan, Stereo},
};
use crate::{models, runtime::Runtime, tools::Tools};

/// Sample rate the separation models work at; the cleaned track is written at this rate.
pub const SAMPLE_RATE: u32 = 44_100;

/// Which model takes the music out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum MusicModel {
    /// The dialogue models when BandIt is installed, otherwise the vocal model.
    Auto,
    /// MDX-Net followed by BandIt Plus: keeps speech only. Stretches without
    /// speech become silent, and a song goes as a whole, its singer included.
    Dialogue,
    /// MDX-Net alone: keeps every human voice, sung or spoken.
    Vocals,
}

#[derive(Clone, Copy, Debug)]
pub struct AudioOptions {
    pub model: MusicModel,
    /// Keep sound effects as well as speech (BandIt Plus alone).
    pub keep_effects: bool,
}

pub struct AudioReport {
    /// Length of the cleaned audio in seconds.
    pub seconds: f64,
    /// Wall-clock seconds spent.
    pub elapsed: f64,
    /// The models that did the work.
    pub method: &'static str,
}

/// One separation model with its way of cutting the stream into chunks.
enum Stage {
    Dialogue(Bandit),
    Vocals(MdxNet),
}

impl Stage {
    fn plan(&self) -> OlaPlan {
        match self {
            // Chunks overlap by half, so every sample is estimated twice and
            // weighted towards the chunk it sits nearer the middle of, where
            // the recurrent model has the most context. Two stereo chunks per
            // call use about 5 GB of GPU memory.
            Stage::Dialogue(m) => OlaPlan { chunk: m.chunk_len(), step: m.chunk_len() / 2, lead: m.edge(), batch: 2 },
            // A quarter of each chunk overlaps the next; the Hann cross-fade
            // gives the unreliable chunk edges almost no weight.
            Stage::Vocals(m) => OlaPlan { chunk: m.chunk_len(), step: m.chunk_len() * 3 / 4, lead: m.edge(), batch: 1 },
        }
    }

    fn separate(&mut self, chunks: &[Stereo]) -> Result<Vec<Stereo>> {
        match self {
            Stage::Dialogue(m) => m.separate(chunks),
            Stage::Vocals(m) => m.separate(chunks),
        }
    }
}

/// The models to run, in order, for the requested kind of cleaning.
///
/// The dialogue method chains two models because each covers the other's
/// weakness (measured on speech mixed with film music, see docs/DESIGN.md):
/// MDX-Net suppresses music under speech best but leaves music audible where
/// nobody speaks; BandIt is silent where nobody speaks but lets a little more
/// music through under speech. MDX-Net first, BandIt on its output, gets both.
fn stages(rt: &Runtime, options: &AudioOptions) -> Result<(Vec<Stage>, &'static str)> {
    let dialogue = match options.model {
        MusicModel::Dialogue => true,
        MusicModel::Vocals => false,
        MusicModel::Auto => models::BANDIT_PLUS.is_present(),
    };
    let vocals = || MdxNet::load(rt, &models::MDX_KIM_VOCAL_2, mdx::KIM_VOCAL_2).map(Stage::Vocals);
    Ok(match (dialogue, options.keep_effects) {
        (true, false) => (vec![vocals()?, Stage::Dialogue(Bandit::load(rt, Keep::SPEECH)?)], "MDX-Net then BandIt Plus"),
        // Effects must come from BandIt on the untouched soundtrack: the
        // vocal model would already have removed them.
        (true, true) => (vec![Stage::Dialogue(Bandit::load(rt, Keep::SPEECH_AND_EFFECTS)?)], "BandIt Plus, speech and effects"),
        (false, keep_effects) => {
            if keep_effects {
                eprintln!("note: --keep-effects needs the dialogue model; the vocal model keeps voices only");
            }
            (vec![vocals()?], "MDX-Net Kim Vocal 2")
        }
    })
}

/// Something that appends more samples to two planes; `0` means the end.
type Source<'a> = Box<dyn FnMut(&mut Vec<f32>, &mut Vec<f32>) -> Result<usize> + Send + 'a>;

/// Writes the music-free version of `input`'s first audio stream to `out_wav`.
///
/// `progress` is called with the seconds of audio finished so far.
pub fn remove_music(
    tools: &Tools,
    rt: &Runtime,
    input: &Path,
    out_wav: &Path,
    span: Span,
    options: &AudioOptions,
    mut progress: impl FnMut(f64),
) -> Result<AudioReport> {
    let started = Instant::now();
    let (mut stages, method) = stages(rt, options)?;
    let mut last = stages.pop().expect("at least one stage");

    let mut reader = Some(PcmReader::open(tools, input, SAMPLE_RATE, span)?);
    let mut writer = WavWriter::create(out_wav, SAMPLE_RATE)?;
    let mut done = 0u64;

    let written = std::thread::scope(|scope| -> Result<u64> {
        let mut source: Source = Box::new(move |l, r| {
            let Some(pcm) = reader.as_mut() else { return Ok(0) };
            let n = pcm.read(l, r, SAMPLE_RATE as usize)?;
            if n == 0 {
                // The decoder's exit status is part of the result.
                reader.take().expect("checked above").finish()?;
            }
            Ok(n)
        });
        // Every stage but the last runs on its own thread and hands its
        // output to the next through a short queue, so the stages overlap and
        // nothing but a few blocks of audio is ever held in memory.
        let mut upstream = Vec::new();
        for mut stage in stages.drain(..) {
            let (tx, rx) = sync_channel::<Stereo>(8);
            let mut read = source;
            upstream.push(scope.spawn(move || -> Result<u64> {
                ola::run(
                    &stage.plan(),
                    |l, r| read(l, r),
                    |chunks| stage.separate(chunks),
                    |l, r| tx.send([l.to_vec(), r.to_vec()]).map_err(|_| anyhow!("the next audio stage stopped")),
                )
            }));
            source = Box::new(move |l, r| match rx.recv() {
                Ok([a, b]) => {
                    l.extend_from_slice(&a);
                    r.extend_from_slice(&b);
                    Ok(a.len())
                }
                Err(_) => Ok(0),
            });
        }

        let outcome = ola::run(
            &last.plan(),
            |l, r| source(l, r),
            |chunks| last.separate(chunks),
            |l, r| {
                writer.write(l, r)?;
                done += l.len() as u64;
                progress(done as f64 / SAMPLE_RATE as f64);
                Ok(())
            },
        );
        // Dropping the source closes the queue, which stops an earlier stage
        // that is still running after a failure here.
        drop(source);
        let mut earlier = Ok(());
        for handle in upstream {
            let result = handle.join().map_err(|_| anyhow!("an audio stage panicked"))?;
            if earlier.is_ok() {
                earlier = result.map(|_| ());
            }
        }
        // A failure here is the root cause when both failed; a failure only
        // upstream would otherwise pass as a short but successful run.
        let written = outcome?;
        earlier?;
        Ok(written)
    })?;
    writer.finish()?;
    Ok(AudioReport { seconds: written as f64 / SAMPLE_RATE as f64, elapsed: started.elapsed().as_secs_f64(), method })
}

