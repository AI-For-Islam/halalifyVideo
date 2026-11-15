# HalalifyVid explainer video

A narrated infographic video, about four and a half minutes long, that walks
through how HalalifyVid works: the pipeline, music removal (STFT, MDX-Net,
BandIt Plus), finding, following and classifying people (YOLO26-seg, tracker,
SigLIP 2), the two-pass decision, mask-based rendering, the runtime and setup,
and the measured speed and accuracy.

The visuals are one three.js scene with an HTML layer on top. The narration is
made with edge-tts, and its word timings drive both the animation cues and the
captions.

## Build it

You need Node.js, Python 3 with `edge-tts`, `playwright` and `Pillow`, and
ffmpeg. The scripts use `../.halalify/bin/ffmpeg.exe` when it exists and fall
back to `ffmpeg` on the PATH.

```text
npm install                       three.js
python -m playwright install chromium
python narration.py               build/: one clip per scene, timeline.json, narration.wav
python render.py video            out/HalalifyVid-explained.mp4
```

`render.py video` splits the frames between several headless Chromium
instances (`--workers`, default 4) and joins the parts. Each frame is a pure
function of time, so any part can be rendered alone: `--from 60 --to 90`.

## Work on it

```text
python -m http.server 8765        then open http://127.0.0.1:8765/?t=61.5
                                  or http://127.0.0.1:8765/?play  and click to play with sound
python render.py stills 61.5 130  full-size PNG stills in out/stills
python render.py sheet 0.8        one still per scene, at 80 % of each
python sheet.py out/sheets out/stills/*.png   2x2 review sheets
```

| File | Contents |
|---|---|
| `narration.py` | The narration text for each scene, edge-tts synthesis, word timings, captions, the narration track |
| `src/main.js` | Renderer, background, camera blending between scenes, captions, chapter bar, `seek(t)` |
| `src/kit.js` | Easing, seeded randomness, DOM and 3D helpers, the synthetic spectrogram and the drawn street scene |
| `src/scenes1.js`, `src/scenes2.js` | The fourteen scenes; each reacts to cue words in its narration |
| `render.py` | Stills, review sheets and the frame-by-frame video render |

To change what is said, edit `SCENES` in `narration.py` and run it again: the
scene lengths, cues and captions follow the new audio.
