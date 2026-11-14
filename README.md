# HalalifyVid

Give it a video file or a link. It writes a new video with

- **the music removed** (voices are kept), and
- **every woman blurred** for as long as she is on screen (or every man, or
  everyone: your choice).

Everything runs on your own machine. It is one Rust program driving ONNX
Runtime on the GPU, with ffmpeg doing the media work. No model was trained for
it; published models are used as they are.

```text
halalify lecture.mp4
halalify https://youtu.be/VIDEO_ID -o clean.mp4
```

## How fast

Measured on an RTX 2000 Ada laptop GPU (8 GB) with a 12-minute 1920x800 film
(*Tears of Steel*, 24 fps):

| Stage | Time | Speed |
|---|---|---|
| Remove music (dialogue models) | 125 s | 6x real time |
| Find, follow and classify people | 92 s | 8x real time (191 frames/s) |
| Blur and encode | 59 s | 12x real time |
| **Whole job** | **277 s** | **2.7x real time** |

With `--music-model vocals` (the fast model alone) music removal takes 32 s
and the whole job about 150 s.

With `--stride 2` the analysis takes 49 s instead of 76 s. A video in which
nobody is blurred skips the last stage: its picture is copied untouched.

## How accurate

Checked by eye on four clips (the film above and street interviews from London,
Kuala Lumpur and Cairo), labelling the people with a clear view:

- **29 of 29 women were blurred**, including women in hijab, in face masks,
  seen from behind, and drawn on advertising posters.
- **37 of 38 men were left alone.** The exception was a man seen from behind
  with a woman's face inside his crop.

People who are far away, blurred or half hidden cannot be told apart reliably
by any model. For those the program is deliberately cautious and blurs when
unsure, so some distant men and long-haired men seen from behind are blurred
too. `--threshold` moves that line. **It can still miss someone, so review the
result before relying on it.** See [docs/DESIGN.md](docs/DESIGN.md) for the
details and the known limits.

## Download

Ready-to-run archives for **Windows** and **Linux** (64-bit) are built by
`scripts/package.sh` and published on the releases page once the repository
is on GitHub. Unpack one, then:

- **Windows:** drag a video onto `HalalifyVid.cmd`, or double-click it and
  paste a file path or a YouTube link.
- **Linux:** `./halalifyvid.sh video.mp4`

The first run installs the runtime and models into the folder's `data`
directory. You also need ffmpeg (and yt-dlp for links).
[docs/RELEASES.md](docs/RELEASES.md) covers requirements, what gets
installed, and how to build the archives yourself.

## Build from source (Windows)

You need [Rust](https://rustup.rs), ffmpeg and, for links, yt-dlp:

```text
winget install Gyan.FFmpeg yt-dlp.yt-dlp
```

Then build and install the runtime and models:

```text
cargo build --release
target\release\halalify setup
```

`setup` downloads ONNX Runtime 1.21 and its GPU libraries, each pinned to a
version and verified by SHA-256: about 1.5 GB for an NVIDIA GPU (CUDA), about
220 MB for any other DirectX 12 GPU (DirectML). The three models (275 MB)
follow. Everything goes into `%LOCALAPPDATA%\HalalifyVid` (`~/.local/share/HalalifyVid`
on Linux), the folder named by `HALALIFY_HOME`, or a `data` folder next to
the program when there is one.

Check what was found at any time:

```text
halalify doctor
```

`halalify.cmd` in this folder runs the release build with `HALALIFY_HOME` set
to `.halalify` here, which keeps the runtime and models beside the code.

## Use

```text
halalify <video file or link> [options]
```

| Option | Meaning | Default |
|---|---|---|
| `-o, --output FILE` | Where to write the result | `<name>.halal.mp4` |
| `--blur women\|men\|everyone\|nobody` | Who to blur | `women` |
| `--shape silhouette\|box` | Blur the person's outline, or a rounded box around them | `silhouette` |
| `--style blur\|pixelate\|black` | How a blurred area looks | `blur` |
| `--threshold 0..1` | Probability of "woman" from which a person is blurred; lower blurs more readily | `0.35` |
| `--audio clean\|keep\|mute` | Remove music, leave the soundtrack alone, or drop it | `clean` |
| `--music-model auto\|dialogue\|vocals` | `dialogue`: MDX-Net then BandIt, keeps speech only. `vocals`: MDX-Net alone, keeps every voice (sung too), 3x faster | `auto` (dialogue if installed) |
| `--keep-effects` | Keep sound effects as well as speech (BandIt alone) | off |
| `--start S`, `--duration S` | Process only part of the video | whole video |
| `--stride N` | Look for people on every N-th frame only (faster) | every frame up to 30 fps |
| `--max-height PIXELS` | For links: tallest picture to download | `1080` |
| `--quality N` | Encoding quality, lower is better and bigger | `21` |
| `--codec h264\|hevc\|av1` | Output video codec | `h264` |
| `--device auto\|cuda\|dml\|cpu` | Where the models run | `auto` |
| `--report FILE` | Write a JSON account of every person found and each decision | off |

Examples:

```text
halalify talk.mp4 --blur nobody              only remove the music
halalify film.mkv --audio keep --shape box   only blur, with full boxes
halalify clip.mp4 --blur everyone --style pixelate
halalify long.mp4 --stride 2                 faster analysis
```

## How it works

1. **Music.** MDX-Net keeps the voices; BandIt Plus then keeps only the
   speech, so music in stretches without dialogue becomes silence.
2. **First pass.** ffmpeg decodes small frames. YOLO26-seg finds every person
   and their outline, a tracker follows each person, and SigLIP 2 compares a
   crop of each person with text descriptions of women and of men.
3. **Decision.** Only after the whole video has been seen is each person
   judged, on all the evidence gathered while they were on screen. The blur
   therefore starts with their first frame and never flickers.
4. **Second pass.** The program sends ffmpeg one tiny mask per frame; ffmpeg
   blurs, encodes and adds the cleaned soundtrack.

[docs/DESIGN.md](docs/DESIGN.md) explains each choice and the measurements
behind it.

## Models

| Model | Job | Size | Licence |
|---|---|---|---|
| [YOLO26s-seg](https://github.com/ultralytics/ultralytics) | People and outlines | 21 MB | AGPL-3.0 |
| [SigLIP 2 B/16-256](https://huggingface.co/google/siglip2-base-patch16-256) image encoder | Who is a woman | 186 MB | Apache-2.0 |
| [Kim Vocal 2](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models) (MDX-Net) | Music removal | 67 MB | MIT |
| [BandIt Plus](https://github.com/ZFTurbo/Music-Source-Separation-Training) (optional, converted locally) | Speech / music / effects | 98 MB | MIT repository; the original BandIt weights on Zenodo are CC BY-NC 4.0 |

The models are downloaded by `setup`, not distributed with this code, and each
keeps its own licence. Note that YOLO26's licence is AGPL-3.0.

## Development

```text
cargo test --release
```

runs the unit tests (they need no GPU, models or ffmpeg).

`tools/embed_prompts.py` rebuilds the text embeddings after the lists of
descriptions are edited. `tools/eval/montage.py` turns a `--report` file into
a contact sheet of the people found, for checking decisions by eye.

## Licence

The code is MIT licensed; see [LICENSE](LICENSE).
