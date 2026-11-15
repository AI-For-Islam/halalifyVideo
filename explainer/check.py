"""Checks the finished video: length, streams, black or frozen stretches, and
writes a contact sheet (one frame per scene, at its middle) to out/contact.png."""
import json
import os
import subprocess
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "..", ".halalify", "bin")
FF = os.path.join(BIN, "ffmpeg.exe") if os.path.exists(os.path.join(BIN, "ffmpeg.exe")) else "ffmpeg"
FP = os.path.join(BIN, "ffprobe.exe") if os.path.exists(os.path.join(BIN, "ffprobe.exe")) else "ffprobe"
video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "HalalifyVid-explained.mp4")
tl = json.load(open(os.path.join(HERE, "build", "timeline.json"), encoding="utf-8"))

info = json.loads(subprocess.run([FP, "-v", "error", "-show_streams", "-show_format", "-of", "json", video],
                                 capture_output=True, text=True, check=True).stdout)
dur = float(info["format"]["duration"])
print(f"file      {os.path.getsize(video) / 1e6:.1f} MB, {dur:.2f} s ({dur / 60:.2f} min), timeline {tl['total']:.2f} s")
for s in info["streams"]:
    extra = f"{s.get('width')}x{s.get('height')} {s.get('r_frame_rate')}" if s["codec_type"] == "video" else f"{s.get('sample_rate')} Hz"
    print(f"stream    {s['codec_type']:5s} {s['codec_name']:5s} {extra} {float(s.get('duration', 0)):.2f} s")
assert dur <= 300, "longer than 5 minutes"

r = subprocess.run([FF, "-v", "info", "-i", video, "-vf", "blackdetect=d=0.4:pix_th=0.06,freezedetect=n=0.0005:d=3",
                    "-an", "-f", "null", "-"], capture_output=True, text=True)
issues = [ln.split("] ", 1)[-1] for ln in r.stderr.splitlines() if "black_start" in ln or "freeze_start" in ln]
print("black/frozen:", issues if issues else "none")

r = subprocess.run([FF, "-v", "info", "-i", video, "-af", "volumedetect", "-vn", "-f", "null", "-"], capture_output=True, text=True)
print("audio     ", " ".join(ln.split("] ", 1)[-1] for ln in r.stderr.splitlines() if "mean_volume" in ln or "max_volume" in ln))

tiles = []
os.makedirs(os.path.join(HERE, "out", "contact"), exist_ok=True)
for i, s in enumerate(tl["scenes"]):
    t = s["start"] + s["dur"] * 0.62
    p = os.path.join(HERE, "out", "contact", f"{i:02d}.png")
    subprocess.run([FF, "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1", "-vf", "scale=480:270", p], check=True)
    tiles.append(p)
sheet = Image.new("RGB", (4 * 480 + 5 * 8, 4 * 270 + 5 * 8), (20, 20, 20))
for i, p in enumerate(tiles):
    sheet.paste(Image.open(p), (8 + (i % 4) * 488, 8 + (i // 4) * 278))
out = os.path.join(HERE, "out", "contact.png")
sheet.save(out)
print("contact  ", out)
