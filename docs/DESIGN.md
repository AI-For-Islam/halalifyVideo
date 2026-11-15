# HalalifyVid: design

This document explains how HalalifyVid works and why it is built the way it is.
The numbers were measured on the development machine: an RTX 2000 Ada laptop
GPU (8 GB), a 32-thread CPU, Windows 11.

## What it does

Given a video file or a link, it writes a new video in which:

1. the music is removed from the soundtrack and voices are kept, and
2. every woman (or, by option, every man or everyone) is blurred for as long
   as she is on screen.

Nothing is trained. Published models are used as they are, run through ONNX
Runtime from a single Rust program, with ffmpeg doing all media work.

## The pipeline

```text
input (file or link)
  |  yt-dlp (links only)
  v
probe ............ ffprobe: size, exact frame rate, rotation, streams
  |
  +--> audio ----- ffmpeg decodes to 44.1 kHz stereo
  |                 -> STFT -> MDX-Net -> inverse STFT, in overlapping chunks
  |                 -> voice-only WAV
  |
  +--> analysis -- ffmpeg decodes, resamples to a constant frame rate, scales
  |                 -> YOLO26-seg: every person, with an outline
  |                 -> tracker: who is who from frame to frame
  |                 -> SigLIP 2: evidence that a tracked person is a woman
  |                 -> tracks (boxes, outlines, evidence) for the whole video
  |
  +--> decision -- each track judged on all its evidence -> blur plan
  |
  v
render ........... ffmpeg again: this program streams one tiny mask per frame;
                   ffmpeg blurs, merges, encodes and muxes the cleaned audio
```

If nobody needs blurring, the render step is skipped and the original video
stream is copied bit-for-bit next to the cleaned soundtrack.

## Decisions and their reasons

### Two passes, decide at the end

A browser extension such as HaramBlur must decide frame by frame, so its blur
lags behind the picture and flickers as confidence changes. A file can be read
twice. The first pass only gathers evidence; decisions are made once the whole
video has been seen:

- a person is judged on every sample taken while they were on screen, not on
  the first glimpse;
- the blur covers a person from their first frame to their last, plus three
  frames either side, never crossing a scene cut;
- frames in which the detector lost a person are filled in by interpolation.

### ffmpeg does the pixels

The obvious design pipes full-resolution frames into the program, blurs them
and pipes them back. A 1080p frame is 3 MB each way.

Instead the program sends ffmpeg a greyscale mask per frame on the mask grid
(a quarter of the detector resolution, about 160x90, 14 kB). ffmpeg scales the
mask up, blurs a copy of the picture, and merges the two with `maskedmerge`.
Full-resolution frames never enter this process and stay in native YUV from
decoder to encoder. The bilinear upscale of the mask gives the blurred area a
soft edge for free.

Both passes start with the same filter, `fps=<rate>:start_time=0`, so they see
exactly the same frames, and in the render pass both streams are re-stamped
with their frame number. Mask `n` therefore always meets frame `n`, whatever
the source timestamps look like (variable frame rate, late start).

The blur itself is "shrink to 1/20, box-average, enlarge". `gblur` is avoided:
on frames that small it corrupts chroma at the top and bottom edges.

### Models

| Job | Model | Why |
|---|---|---|
| People and outlines | YOLO26s-seg (fp16) | 298 frames/s at 640x384, batch 8. RF-DETR-Seg S, the permissively licensed alternative, reached 89 frames/s and takes a fixed 384x384 input. YOLO26 needs no non-maximum suppression. |
| Who is a woman | SigLIP 2 B/16-256 image encoder, zero-shot | Works on whole people, so it does not need a visible face: backs, hijab and niqab are handled. 269 crops/s. |
| Music removal | MDX-Net Kim Vocal 2 | Already ONNX, 40x real time. Keeps any human voice. |

**Classification is zero-shot.** The image embedding of a person's head and
upper body is compared with two text embeddings, each the mean of sixteen
mirrored descriptions ("a woman wearing a hijab" / "a man wearing a turban",
"a woman seen from behind" / "a man seen from behind", and so on). The
difference of the similarities, on the model's logit scale, is the evidence.
The text embeddings are computed once by `tools/embed_prompts.py` and shipped
in `assets/person_prompts.json`; only the image encoder runs at run time.

**What the evidence looks like.** Four clips were labelled by eye from contact
sheets (`tools/eval/montage.py`): a short film and street interviews from
London, Kuala Lumpur and Cairo. Clear views of women scored 0.73 or more
(most above 0.9, including women in hijab, in face masks, seen from behind and
drawn on advertising posters). Clear views of men scored 0.30 or less. In
between sit people who are far away, blurred or partly hidden, of either sex.

**So the threshold leans towards blurring.** A person is blurred from a
probability of 0.35, not 0.5, and a person who could not be classified at all
is blurred. Missing a woman is the worse mistake. The price is that some
distant or half-hidden men are blurred too; `--threshold` moves the line.

**A track is judged stretch by stretch.** When two people cross, the tracker
sometimes continues one person's track on the other. Each observation is
therefore judged half on the whole track and half on the samples within 1.5 s
of it. A track that is one person gets one answer; a track that is a man and
then a woman gets two.

### Music removal

`Kim_Vocal_2` is a spectrogram-to-spectrogram U-Net. The STFT, the overlapping
chunks and the inverse STFT are implemented in Rust (`src/audio`) to match
`torch.stft` exactly, and the stream is processed as it is decoded, so memory
use does not grow with the length of the video.

Checked against BS PolarFormer (the strongest open vocal model, see
`music-removal-models.md` in the parent project) on a film excerpt: passages
without voices are attenuated by 25 to 55 dB, and voices match. It is weaker
than PolarFormer on loud sound effects, which it sometimes lets through.

### ONNX Runtime 1.21, loaded at run time

ONNX Runtime 1.22 and later run a hardware "device discovery" when the
environment is created. On the development machine that takes 11 seconds
before any model can load, in every build flavour. Version 1.21 starts in
30 ms and runs these models at the same speed, so the program loads
`onnxruntime.dll` 1.21 at run time (the `ort` crate's `load-dynamic` mode with
API level 21) instead of linking a newer one in.

The provider is chosen by trying CUDA, then DirectML, then CPU, and keeping the
first that can really create a session. Both GPU routes were run end to end on
the development machine: with CUDA the film analyses at 233 frames/s; with the
DirectML build a 30 s excerpt, model warm-up included, ran at 96 frames/s.

### Tools are test-driven, not trusted

ffmpeg lists `h264_nvenc` even when the installed GPU driver is too old for
that ffmpeg build; the encoder only fails when opened. (Seen here: ffmpeg 9.0
needs driver 610, the machine has 596.) So every candidate encoder is
test-driven with a few frames, across every ffmpeg found on the machine, and
the fastest pair that really works is used. Results are cached.

## Source map

| Path | Contents |
|---|---|
| `src/main.rs`, `src/cli.rs` | Commands and options |
| `src/pipeline.rs` | One job from input to output |
| `src/tools.rs`, `src/probe.rs`, `src/fetch.rs` | ffmpeg, ffprobe, yt-dlp |
| `src/runtime.rs`, `src/models.rs`, `src/setup.rs` | ONNX Runtime, model registry, installer |
| `src/audio/` | STFT, streaming overlap-add, MDX-Net, PCM and WAV I/O |
| `src/vision/frames.rs` | Frame reader and the sizes a video is handled at |
| `src/vision/detect.rs`, `track.rs`, `cuts.rs` | Detector, tracker, scene cuts |
| `src/vision/classify.rs` | Zero-shot person classifier |
| `src/vision/decide.rs`, `mask.rs` | Verdicts, blur plan, mask painting |
| `src/render.rs` | The ffmpeg render pass |
| `src/progress.rs` | Progress bars, and the JSON events for a driving program |
| `desktop/` | The desktop app: a window that runs this program |
| `tools/` | One-time scripts: prompt embeddings, evaluation contact sheets |

### The dialogue method: MDX-Net, then BandIt Plus

BandIt Plus separates speech, music and effects. `tools/export_bandit.py`
rebuilds it with real arithmetic only (no complex tensors, STFT outside the
graph, the 64 overlapping band masks merged by one matrix) and loads the
published weights unchanged; on a 6 s chunk the rebuild agrees with the
original code to 140 dB. The Rust side (`src/audio/bandit.rs`) reproduces the
STFT and overlap-add, and agrees with PyTorch on the same 40 s to 57 dB.

Scored on English and Egyptian Arabic speech mixed with the film's music at
two levels (higher is better for the first two columns, lower for the last):

| Method | Music alone | Speech kept (SDR) | Music left under speech |
|---|---|---|---|
| MDX-Net alone | -19 to -28 dB | 12 to 16 dB | -9 to -15 dB |
| BandIt alone | -98 dB | 11 to 13 dB | -4 to -12 dB |
| MDX-Net then BandIt | -66 to -96 dB | 11 to 13 dB | -7 to -14 dB |

MDX-Net suppresses music best under speech but leaves it audible where nobody
speaks; BandIt silences those stretches completely. Chained, each covers the
other's weakness, at the cost of slightly duller speech. The two models run
on separate threads so they overlap.

ONNX Runtime 1.21's half-precision GRU on CUDA gives wrong results for this
model (15 dB agreement), so the export keeps the GRU layers in fp32 and the
rest in fp16: 60 dB agreement and twice the speed of fp32. The CUDA memory
arena grows only as requested, which keeps BandIt at about 5 GB for two
stereo chunks per call.

## The desktop app

The app in `desktop/` is a window and nothing more. It starts `halalify` as a
child process and draws what the program reports; no model, no ffmpeg call and
no decision lives in the app. Three things follow from that:

- **One engine.** The command line and the app cannot drift apart, and
  everything the app does can be reproduced in a terminal.
- **A crash stays a message.** If ONNX Runtime or a GPU driver brings the
  process down, the window is still there to say so.
- **Stopping is ending a process.** The app ends `halalify` together with the
  ffmpeg and yt-dlp processes it started, then removes the job's scratch
  folder, which the program reported when it created it.

The program's side of this is `--progress json`: instead of drawing bars it
writes one JSON object per line to standard output.

| Event | Meaning |
|---|---|
| `stage` | `done` of `total` seconds of media are through `removing music`, `finding people` or `blurring` |
| `download` | bytes of a model or runtime package fetched so far |
| `fetch` | a line of yt-dlp's progress while a link is downloaded |
| `work` | the job's scratch folder |
| `installed`, `ready` | `setup` has a package in place; `setup` is finished and the models run on `device` |
| `done` | the job wrote `output` in `seconds` |
| `error` | the command failed, with the message |

`halalify doctor --progress json` prints a single `status` object: the tools
found, the models present, and whether a video can be cleaned right now.
Position events are limited to one per 120 ms, and every other line the
program prints stays plain text, which the app keeps as its details log.

**yt-dlp is the one thing that is not pinned.** Sites change, and a yt-dlp a
few months old is refused by YouTube. `halalify update-ytdlp` therefore
fetches the newest release into the program's own `bin` folder (searched
first), checked against the checksum list published with that release. The
app offers it as a button when a link could not be downloaded, and then
tries the video again.

The window is built with Tauri: the screens are plain HTML, CSS and
JavaScript shown by the system's WebView2, so the app adds about 6 MB. Each
choice made on screen is checked against the words `halalify` accepts before
it becomes an argument, and the input is passed after `--`, so nothing typed
or dropped can turn into an option.

## Known limits

- **It can miss a woman.** The detector must find her first; people smaller
  than about 3% of the frame height, heavily occluded or in very dark scenes
  can be missed. Review the result before relying on it.
- **It blurs some men.** By design, when the evidence is unclear.
- **Outline versus box.** The default blur follows the person's outline, grown
  slightly. `--shape box` covers a rounded box instead and cannot leave
  anything out.
- **The vocal model alone keeps all voices**, sung ones included, and can let
  loud sound effects through. The dialogue method keeps speech only, which
  also removes non-musical sound such as applause.
- **Variable-frame-rate sources** are converted to a constant rate.
- **HDR** sources are written as 8-bit SDR without tone mapping.
- **Windows and Linux x64** are the packaged platforms; macOS is untested.
- **The desktop app is built for Windows only** so far.

## Ideas not yet built

- Showing the classifier only the pixels inside a person's outline, so a
  neighbour in the crop cannot sway the verdict.
- A TensorRT provider, roughly three times faster for the detector.
- Re-encoding only the stretches that contain blur.
