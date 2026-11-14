# Releases

HalalifyVid is released as one archive per operating system. Each unpacks to a
self-contained folder: the program, a launcher, this documentation and an empty
`data` folder. On the first run the program installs ONNX Runtime, the GPU
libraries and the models into `data`, so the folder can be moved anywhere or
deleted to uninstall.

| Archive | For | Built on |
|---|---|---|
| `halalify-<version>-windows-x86_64.zip` | Windows 10 and 11, 64-bit | Windows, static C runtime |
| `halalify-<version>-linux-x86_64.tar.gz` | Linux, 64-bit, glibc 2.30 or newer (Ubuntu 20.04+, Debian 11+, Fedora 32+ and their relatives) | Debian 11 |

Each archive comes with a `.sha256` file to check the download.

## What you need besides the archive

| | Windows | Linux |
|---|---|---|
| ffmpeg | `winget install Gyan.FFmpeg` | `sudo apt install ffmpeg` (or your distribution's package) |
| yt-dlp (only for links) | `winget install yt-dlp.yt-dlp` | `pipx install yt-dlp` or your package manager |
| Node.js (optional, YouTube's full format list) | `winget install OpenJS.NodeJS` | your package manager |
| NVIDIA GPU (fastest) | driver 560 or newer | driver 560 or newer (`nvidia-smi` works) |
| Other GPUs | any DirectX 12 GPU through DirectML | not supported: the CPU is used |

No CUDA toolkit, Python or Visual C++ Redistributable is needed; the program
fetches the exact runtime libraries it uses.

## Install and run

### Windows

1. Unpack the zip anywhere, for example to `C:\HalalifyVid`.
2. Drag a video onto **`HalalifyVid.cmd`**, or double-click it and paste a file
   path or a YouTube link. The first run installs the runtime and models
   (about 1.8 GB with an NVIDIA GPU, about 500 MB otherwise) and then cleans
   the video.
3. The result is written next to the original as `<name>.halal.mp4` (for a
   link, in the release folder).

From a terminal, `halalify.exe` takes every option:

```text
halalify.exe setup
halalify.exe "%USERPROFILE%\Videos\talk.mp4" --blur nobody
halalify.exe --help
```

### Linux

```bash
tar -xzf halalify-<version>-linux-x86_64.tar.gz
cd halalify-<version>-linux-x86_64
./halalifyvid.sh ~/Videos/talk.mp4
```

The first run of `halalifyvid.sh` installs the runtime and models, as on
Windows. Use `./halalify` directly for every option (`./halalify --help`). To
put it on your `PATH`, link the program rather than copying it, so it still
finds its `data` folder:

```bash
ln -s "$PWD/halalify" ~/.local/bin/halalify
```

## What `setup` installs

`setup` picks the route from the machine (`--provider cuda|directml|cpu`
overrides it). Every file comes from its publisher, pinned to a version and
checked against a SHA-256 before it is unpacked:

| Route | Windows | Linux |
|---|---|---|
| CUDA (NVIDIA driver present) | ONNX Runtime 1.21.0 GPU build; CUDA 12.6 runtime, cuBLAS, cuFFT; cuDNN 9.6 | ONNX Runtime 1.21.0 GPU build; CUDA 12.6 runtime, NVRTC, cuRAND, cuBLAS, cuFFT; cuDNN 9.6 |
| DirectML (Windows, no NVIDIA driver) | ONNX Runtime 1.21.0 DirectML build; DirectML 1.15.4 | n/a |
| CPU | the DirectML build, which falls back to the CPU | ONNX Runtime 1.21.0 CPU build |

Then the three models: MDX-Net Kim Vocal 2, YOLO26s-seg and the SigLIP 2
image encoder (275 MB). The optional BandIt Plus dialogue model is converted
locally; see the README.

Run `halalify doctor` at any time to see what was found and where models run.

## Where files go

The program looks for its home folder in this order:

1. `HALALIFY_HOME`, when set;
2. `data` next to the program, which every release has;
3. the per-user folder, `%LOCALAPPDATA%\HalalifyVid` or `~/.local/share/HalalifyVid`.

## Building a release yourself

The archives are made by [`scripts/package.sh`](../scripts/package.sh), which
builds for the system it runs on and writes `dist/<archive>` and its checksum.

**Windows** (in Git Bash, with Rust installed):

```bash
bash scripts/package.sh
```

**Linux**, on Linux:

```bash
bash scripts/package.sh
```

**Linux, from Windows or macOS with Docker.** Building in a Debian 11 image
keeps the glibc requirement low:

```bash
docker run --rm -v "$PWD:/work" -w /work -e CARGO_TARGET_DIR=/tmp/target rust:1-bullseye bash scripts/package.sh
```

(In Git Bash on Windows, prefix the command with `MSYS_NO_PATHCONV=1` so the
container paths are passed through unchanged.)

### Automatic releases

[`.github/workflows/release.yml`](../.github/workflows/release.yml) builds and
tests both archives on GitHub Actions. Pushing a tag publishes them as a
GitHub release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Running the workflow by hand ("Run workflow" on the Actions page) builds the
archives and keeps them as downloadable artifacts without publishing.

## Tested

Both archives were unpacked into an empty folder and run through their
launcher on an 8-second clip:

| Archive | Where | Result |
|---|---|---|
| Windows | Windows 11, RTX 2000 Ada laptop GPU | runs on CUDA; 8 s clip cleaned in 7.5 s. `setup` itself was tested from scratch on the same machine, for both CUDA and DirectML |
| Linux | Fresh Debian 12 container with the GPU passed through (Docker on WSL 2) | `setup` detected the NVIDIA driver and installed the CUDA route; clip cleaned in 8.2 s on CUDA |
| Linux | Same container, `--device cpu` | works; about 30x slower than CUDA |

The GitHub Actions workflow has not run yet: this repository has no GitHub
remote so far.

macOS is not packaged: ONNX Runtime's CUDA and DirectML routes do not exist
there, and it has not been tested.
