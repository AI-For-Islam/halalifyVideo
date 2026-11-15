"""Narration for the explainer video: text, edge-tts audio and word timings.

Writes, under build/:
  audio/<scene>.mp3   one narration clip per scene
  timeline.json       scene start times, durations, word timings, captions
  narration.wav       the whole narration track, 48 kHz mono

Run:  python narration.py [--voice en-US-AndrewNeural] [--rate +4%]
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import wave
from array import array

import edge_tts

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
LEAD = 0.45  # silence before each scene's narration starts (s)
TAIL = 0.75  # silence after it, before the next scene (s)
RATE_HZ = 48000

# Spoken text. Some words are spelled the way the voice should say them; the
# captions turn them back with DISPLAY below.
SCENES = [
    ("title",
     "This is HalalifyVid. Give it a video file or a YouTube link, and it returns the same video "
     "with the music removed and women blurred, entirely on your own machine."),
    ("overview",
     "It is a single Rust program. No model was trained for it: published models run as they are, "
     "through Onyx Runtime on your GPU, while FFmpeg handles all the media work."),
    ("pipeline",
     "Links are downloaded with Y T D L P, and F F probe reads the frame size, the exact frame rate and the streams. "
     "Then the work splits. The audio branch removes the music. The analysis branch finds, follows and "
     "classifies every person. A decision step turns that evidence into a blur plan, and a render pass "
     "writes the result."),
    ("stft",
     "First, the sound. FFmpeg decodes it to 44.1 kilohertz stereo, and a short-time Fourier transform, "
     "written in Rust to match PyTorch exactly, turns it into a spectrogram. The audio flows through in "
     "overlapping chunks as it is decoded, so memory stays flat for any length of video."),
    ("mdx",
     "MDX-Net Kim Vocal 2, a U-Net, maps that spectrogram to the voices alone, at forty times real time, "
     "pushing music down by twenty-five to fifty-five decibels. But where nobody speaks, faint music can survive."),
    ("bandit",
     "So BandIt Plus follows. It splits sound into speech, music and effects across sixty-four frequency bands, "
     "rebuilt for Onyx to match the original within a hundred and forty decibels. Chained on its own thread, "
     "it drives the leftover music from about minus twenty to below minus sixty-six decibels. It runs in half "
     "precision, except its G R U layers, which stay in full precision for accuracy."),
    ("detect",
     "Now the picture. FFmpeg decodes small frames at a constant rate, and YOLO 26 segmentation finds every "
     "person, with a full outline, in batches of eight, at nearly three hundred frames per second."),
    ("track",
     "A tracker links the detections into one track per person. Scene cuts end every track, and frames where "
     "someone was briefly lost are filled in by interpolation."),
    ("siglip",
     "Then: who is a woman? A crop of each person's head and upper body goes through the Sig-Lip 2 image encoder, "
     "and is compared with two text embeddings, each the average of sixteen paired descriptions, such as "
     "a woman wearing a hijab, versus a man wearing a turban. The difference is the evidence. No face is needed, "
     "so backs, hijab and niqab all work. The text side is computed once, offline."),
    ("decide",
     "Here is the key idea. A browser extension must decide frame by frame, so its blur lags and flickers. "
     "A file can be read twice. The first pass only gathers evidence, and each person is judged after the whole "
     "video, on every sample. Missing a woman is the worse mistake, so the threshold sits at zero point three five, "
     "and people who cannot be classified are blurred too. The blur then spans a person's first frame to their "
     "last, plus three frames, never crossing a cut."),
    ("render",
     "In the second pass, instead of piping three-megabyte frames in and out, the program streams FFmpeg one tiny "
     "mask per frame, about fourteen kilobytes. FFmpeg scales it up, which softens the edge for free, blurs a copy "
     "of the picture, and merges the two. Both passes share one frame-rate filter, so mask N always meets frame N. "
     "If nobody needs blurring, the video is copied untouched."),
    ("runtime",
     "Under the hood, Onyx Runtime one point two one is loaded at run time: newer versions stall for eleven seconds "
     "discovering hardware, while one point two one starts in thirty milliseconds. It tries CUDA, then Direct M L, "
     "then the CPU, and keeps the first that really works. Encoders are test-driven the same way, and setup "
     "verifies every download by S H A 256."),
    ("results",
     "On a laptop RTX GPU, a twelve-minute film takes about four and a half minutes: two point seven times real "
     "time. On four test clips, twenty-nine of twenty-nine women were blurred, and thirty-seven of thirty-eight "
     "men were left alone."),
    ("outro",
     "Using it is one command: halalify, then a file or a link, with options for who to blur, how, and which "
     "music model. Releases are ready for Windows and Linux. It can still miss someone, so always review the "
     "result. That is HalalifyVid."),
]

DISPLAY = [
    ("Onyx", "ONNX"), ("Y T D L P", "yt-dlp"), ("F F probe", "ffprobe"), ("G R U", "GRU"),
    ("YOLO 26", "YOLO26"), ("Sig-Lip 2", "SigLIP 2"), ("one point two one", "1.21"),
    ("Direct M L", "DirectML"), ("S H A 256", "SHA-256"), ("zero point three five", "0.35"),
    ("two point seven", "2.7"), ("halalify,", "halalify,"),
]


def display(s: str) -> str:
    for a, b in DISPLAY:
        s = s.replace(a, b)
    return s


async def synth(text: str, voice: str, rate: str, mp3: str):
    """Writes the clip and returns its word boundaries; the service drops a
    request now and then, so each clip is tried a few times."""
    for attempt in range(6):
        try:
            return await synth_once(text, voice, rate, mp3)
        except edge_tts.exceptions.NoAudioReceived:
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"edge-tts gave no audio for: {text[:40]}...")


async def synth_once(text: str, voice: str, rate: str, mp3: str):
    com = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    words = []
    with open(mp3, "wb") as f:
        async for chunk in com.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append({
                    "w": chunk["text"],
                    "t0": chunk["offset"] / 1e7,
                    "t1": (chunk["offset"] + chunk["duration"]) / 1e7,
                })
    return words


def align(text: str, words):
    """Attaches the text between words (punctuation) to each boundary word."""
    low = text.lower()
    pos = 0
    for w in words:
        i = low.find(w["w"].lower(), pos)
        if i < 0:
            w["src"] = w["w"]
            continue
        end = i + len(w["w"])
        # keep trailing punctuation up to the next space
        m = re.match(r"[^\s]*", text[end:])
        tail = m.group(0) if m else ""
        w["src"] = text[i:end] + tail
        pos = end + len(tail)
    return words


def captions(words, scene_start):
    """Groups words into caption lines: break at sentence ends or ~7 words."""
    lines, cur = [], []
    for w in words:
        cur.append(w)
        src = w.get("src", w["w"])
        n = len(" ".join(x.get("src", x["w"]) for x in cur))
        if src.endswith((".", "?", "!")) or (src.endswith((",", ":", ";")) and n > 34) or n > 58:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        txt = display(" ".join(x.get("src", x["w"]) for x in ln))
        out.append({"t0": round(scene_start + ln[0]["t0"], 3), "t1": round(scene_start + ln[-1]["t1"], 3), "text": txt})
    # hold each line until the next one starts (no flicker between lines)
    for a, b in zip(out, out[1:]):
        if b["t0"] - a["t1"] < 0.6:
            a["t1"] = b["t0"]
    return out


def probe_seconds(ffprobe, path):
    r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def decode_pcm(ffmpeg, path):
    r = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(RATE_HZ), "-"],
                       capture_output=True, check=True)
    a = array("h")
    a.frombytes(r.stdout)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="en-US-AndrewNeural")
    ap.add_argument("--rate", default="+4%")
    ap.add_argument("--ffmpeg", default=os.path.join(HERE, "..", ".halalify", "bin", "ffmpeg.exe"))
    ap.add_argument("--ffprobe", default=os.path.join(HERE, "..", ".halalify", "bin", "ffprobe.exe"))
    args = ap.parse_args()
    if not os.path.exists(args.ffmpeg):
        args.ffmpeg, args.ffprobe = "ffmpeg", "ffprobe"

    os.makedirs(os.path.join(BUILD, "audio"), exist_ok=True)
    scenes, t = [], 0.0
    track = array("h")
    for sid, text in SCENES:
        mp3 = os.path.join(BUILD, "audio", f"{sid}.mp3")
        words = asyncio.run(synth(text, args.voice, args.rate, mp3))
        align(text, words)
        voice_len = probe_seconds(args.ffprobe, mp3)
        pcm = decode_pcm(args.ffmpeg, mp3)
        start = t
        voice_at = start + LEAD
        dur = LEAD + voice_len + TAIL
        # place the clip on the track
        need = int(round((start + dur) * RATE_HZ))
        at = int(round(voice_at * RATE_HZ))
        if len(track) < need:
            track.extend([0] * (need - len(track)))
        for i, v in enumerate(pcm):
            if at + i < len(track):
                track[at + i] = v
        for w in words:
            w["t0"] = round(voice_at + w["t0"], 3)
            w["t1"] = round(voice_at + w["t1"], 3)
        caps = captions([{**w, "t0": w["t0"] - voice_at, "t1": w["t1"] - voice_at} for w in words], voice_at)
        scenes.append({"id": sid, "start": round(start, 3), "dur": round(dur, 3), "voice": round(voice_at, 3),
                       "voiceLen": round(voice_len, 3), "words": [{"w": w["w"], "t0": w["t0"], "t1": w["t1"]} for w in words],
                       "captions": caps})
        print(f"{sid:9s} start {start:6.2f}  voice {voice_len:5.2f}s  words {len(words)}", flush=True)
        t = start + dur

    total = t + 0.8  # a short hold on the last frame
    track.extend([0] * (int(round(total * RATE_HZ)) - len(track)))
    with wave.open(os.path.join(BUILD, "narration.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE_HZ)
        wf.writeframes(track.tobytes())
    with open(os.path.join(BUILD, "timeline.json"), "w", encoding="utf-8") as f:
        json.dump({"total": round(total, 3), "scenes": scenes}, f, indent=1)
    with open(os.path.join(BUILD, "timeline.js"), "w", encoding="utf-8") as f:
        f.write("export default " + json.dumps({"total": round(total, 3), "scenes": scenes}) + ";\n")
    print(f"total {total:.2f}s ({total / 60:.2f} min)")


if __name__ == "__main__":
    sys.exit(main())
