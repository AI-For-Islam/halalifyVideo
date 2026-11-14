"""Draws a contact sheet of the people a run found, for labelling by eye.

    python tools/eval/montage.py <video> <report.json> <out.jpg> [--start S] [--min-sightings N]

Each row is one track: its id, the probability of "woman" the run assigned,
and up to six of the crops the classifier judged. Also saves every crop's
source frame region to <out>.npz so that other prompts or crop shapes can be
tried offline (see tune.py). Needs: numpy, pillow, and ffmpeg on PATH or FFMPEG.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.path.join(os.path.dirname(FFMPEG), "ffprobe" + os.path.splitext(FFMPEG)[1]) if os.path.dirname(FFMPEG) else "ffprobe"
WIDTH = 960
CELL = 128


def read_frames(video, start, fps, wanted, height):
    """Yields (index, HxWx3 array) for the wanted frame indices, decoding once."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", video, "-map", "0:v:0", "-vf", f"fps={fps}:start_time=0,scale={WIDTH}:{height}",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = WIDTH * height * 3
    last = max(wanted)
    index = 0
    while index <= last:
        data = proc.stdout.read(size)
        if len(data) < size:
            break
        if index in wanted:
            yield index, np.frombuffer(data, np.uint8).reshape(height, WIDTH, 3)
        index += 1
    proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("report")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--min-sightings", type=int, default=6)
    ap.add_argument("--per-track", type=int, default=6)
    args = ap.parse_args()

    report = json.load(open(args.report))
    probe = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", args.video],
                           capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    height = int(round(WIDTH * stream["height"] / stream["width"] / 2)) * 2

    tracks = [t for t in report["tracks"] if t["sightings"] >= args.min_sightings and t["evidence"]]
    picks = {}  # frame -> [(track id, slot, box)]
    for t in tracks:
        ev = t["evidence"]
        idx = np.unique(np.linspace(0, len(ev) - 1, min(args.per_track, len(ev))).round().astype(int))
        for slot, i in enumerate(idx):
            picks.setdefault(ev[i]["frame"], []).append((t["id"], slot, ev[i]["box"], ev[i]["logit"]))
    # every classified sample is stored for offline tuning
    samples = {}
    for t in tracks:
        for e in t["evidence"]:
            samples.setdefault(e["frame"], []).append((t["id"], e["box"]))

    sheet = Image.new("RGB", (110 + args.per_track * CELL, len(tracks) * CELL), "white")
    draw = ImageDraw.Draw(sheet)
    rows = {t["id"]: r for r, t in enumerate(tracks)}
    for t in tracks:
        y = rows[t["id"]] * CELL
        draw.text((4, y + 6), f"#{t['id']}", fill="black")
        draw.text((4, y + 24), f"p={t['female']:.2f}", fill="red" if t["blurred"] else "green")
        draw.text((4, y + 42), f"{t['start_seconds']:.0f}-{t['end_seconds']:.0f}s", fill="black")
        draw.text((4, y + 60), f"n={t['samples']}", fill="black")

    stored_ids, stored_frames, stored_boxes, stored_regions = [], [], [], []
    for index, frame in read_frames(args.video, args.start, report["fps"], set(samples), height):
        for tid, box in samples[index]:
            x1, y1, x2, y2 = (box[0] * WIDTH, box[1] * height, box[2] * WIDTH, box[3] * height)
            m = 0.06 * (x2 - x1)
            l, r_, tp, bt = int(max(0, x1 - m)), int(min(WIDTH, x2 + m)), int(max(0, y1 - m)), int(min(height, y2 + m))
            region = Image.fromarray(frame[tp:bt, l:r_])
            # stored at a fixed small size, whole box, aspect kept by padding
            canvas = Image.new("RGB", (256, 384), (127, 127, 127))
            region.thumbnail((256, 384))
            canvas.paste(region, (0, 0))
            stored_regions.append(np.asarray(canvas))
            stored_ids.append(tid)
            stored_frames.append(index)
            stored_boxes.append([region.width, region.height])
        for tid, slot, box, logit in picks.get(index, []):
            x1, y1, x2, y2 = (box[0] * WIDTH, box[1] * height, box[2] * WIDTH, box[3] * height)
            m = 0.06 * (x2 - x1)
            l, r_, tp = max(0, x1 - m), min(WIDTH, x2 + m), max(0, y1 - m)
            bt = min(height, y2 + m, tp + 1.3 * (r_ - l))
            crop = Image.fromarray(frame[int(tp):int(bt), int(l):int(r_)]).resize((CELL - 4, CELL - 4))
            x = 110 + slot * CELL
            y = rows[tid] * CELL
            sheet.paste(crop, (x + 2, y + 2))
            draw.text((x + 4, y + 4), f"{logit:+.1f}", fill="yellow")
    sheet.save(args.out, quality=85)
    np.savez_compressed(os.path.splitext(args.out)[0] + ".npz", ids=np.array(stored_ids), frames=np.array(stored_frames),
                        sizes=np.array(stored_boxes), regions=np.array(stored_regions))
    print(f"{len(tracks)} tracks, {len(stored_ids)} samples -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
