"""Renders the explainer page to a video, frame by frame, with headless Chromium.

  python render.py stills 6 30 61.5        PNG stills at those times (out/stills)
  python render.py sheet                   one still per scene, middle of each
  python render.py video [--fps 30] [--workers 4] [--from S --to S]

The page exposes seek(t); every frame is a pure function of t, so the frames
are split between several browsers and each part is encoded on its own, then
joined and muxed with build/narration.wav.
"""

import argparse
import functools
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import sys
import threading
import time
from multiprocessing import Process

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FF = os.path.join(HERE, "..", ".halalify", "bin", "ffmpeg.exe")
if not os.path.exists(FF):
    FF = "ffmpeg"
ARGS = ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-gpu-rasterization", "--force-color-profile=srgb"]


def serve():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    handler = functools.partial(Quiet, directory=HERE)
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


def open_page(p, port):
    browser = p.chromium.launch(headless=True, channel="chromium", args=ARGS)
    page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    page.on("console", lambda m: m.type == "error" and print("console:", m.text, file=sys.stderr))
    page.on("pageerror", lambda e: print("pageerror:", e, file=sys.stderr))
    page.goto(f"http://127.0.0.1:{port}/index.html?render=1", wait_until="load")
    page.evaluate("window.ready")
    return browser, page


def stills(times, port, prefix="t"):
    os.makedirs(os.path.join(OUT, "stills"), exist_ok=True)
    with sync_playwright() as p:
        browser, page = open_page(p, port)
        misses = page.evaluate("window.__cueMisses")
        if misses:
            print("cue misses:", misses)
        paths = []
        for t in times:
            page.evaluate(f"window.seek({t})")
            path = os.path.join(OUT, "stills", f"{prefix}{t:07.2f}.png")
            page.screenshot(path=path)
            paths.append(path)
        browser.close()
    return paths


def worker(k, a, b, fps, port, seg, encoder):
    enc = (["-c:v", "h264_nvenc", "-preset", "p6", "-cq", "17", "-b:v", "0"] if encoder == "nvenc"
           else ["-c:v", "libx264", "-preset", "medium", "-crf", "16"])
    ff = subprocess.Popen([FF, "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg", "-i", "-",
                           *enc, "-pix_fmt", "yuv420p", "-r", str(fps), seg], stdin=subprocess.PIPE)
    t0 = time.time()
    with sync_playwright() as p:
        browser, page = open_page(p, port)
        for i in range(a, b):
            page.evaluate(f"window.seek({i / fps})")
            ff.stdin.write(page.screenshot(type="jpeg", quality=93))
            if (i - a) % 300 == 0:
                done = i - a
                rate = done / max(1e-6, time.time() - t0)
                print(f"[w{k}] {done}/{b - a} frames  {rate:.1f} fps", flush=True)
        browser.close()
    ff.stdin.close()
    ff.wait()
    print(f"[w{k}] done {b - a} frames in {time.time() - t0:.0f}s", flush=True)


def video(fps, workers, t_from, t_to, encoder):
    with open(os.path.join(HERE, "build", "timeline.json"), encoding="utf-8") as f:
        total = json.load(f)["total"]
    t_to = min(t_to or total, total)
    n0, n1 = int(round(t_from * fps)), int(round(t_to * fps))
    port = serve()
    tmp = os.path.join(OUT, "parts")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    step = (n1 - n0 + workers - 1) // workers
    procs, segs = [], []
    for k in range(workers):
        a, b = n0 + k * step, min(n1, n0 + (k + 1) * step)
        if a >= b:
            break
        seg = os.path.join(tmp, f"part{k:02d}.mp4")
        segs.append(seg)
        pr = Process(target=worker, args=(k, a, b, fps, port, seg, encoder))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()
        if pr.exitcode:
            sys.exit(f"a worker failed ({pr.exitcode})")
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{s}'\n" for s in segs)
    name = "HalalifyVid-explained.mp4" if t_from == 0 and t_to == total else f"part-{t_from:g}-{t_to:g}.mp4"
    out = os.path.join(OUT, name)
    subprocess.run([FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", str(t_from), "-t", str(t_to - t_from), "-i", os.path.join(HERE, "build", "narration.wav"),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-movflags", "+faststart", "-shortest", out], check=True)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stills", "sheet", "video"])
    ap.add_argument("times", nargs="*", type=float)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=0.0)
    ap.add_argument("--encoder", choices=["nvenc", "x264"], default="nvenc")
    a = ap.parse_args()
    if a.mode == "video":
        return video(a.fps, a.workers, a.t_from, a.t_to, a.encoder)
    port = serve()
    if a.mode == "sheet":
        with open(os.path.join(HERE, "build", "timeline.json"), encoding="utf-8") as f:
            tl = json.load(f)
        frac = a.times[0] if a.times else 0.8
        times = [round(s["start"] + s["dur"] * frac, 2) for s in tl["scenes"]]
        for p in stills(times, port, "s"):
            print(p)
    else:
        for p in stills(a.times, port):
            print(p)


if __name__ == "__main__":
    main()
