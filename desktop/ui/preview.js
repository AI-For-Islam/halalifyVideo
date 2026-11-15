// A stand-in for the backend, used only when the screens are opened in an
// ordinary browser (no window.__TAURI__): it makes working on the screens
// possible without building the app. Open index.html, optionally with
//   ?state=fresh   nothing installed yet (the setup screen)
//   ?fail=1        the next job ends with an error
// Inside the app this file does nothing.

(() => {
  "use strict";
  if (window.__TAURI__) return;

  const query = new URLSearchParams(location.search);
  let installed = query.get("state") !== "fresh";
  let handler = () => {};
  let timer = null;

  const status = () => ({
    event: "status",
    version: "0.1.0",
    home: "C:\\Users\\you\\AppData\\Local\\HalalifyVid",
    ffmpeg: "C:\\ffmpeg\\bin\\ffmpeg.exe",
    ffprobe: "C:\\ffmpeg\\bin\\ffprobe.exe",
    encoder: "h264_nvenc",
    ytdlp: query.get("ytdlp") === "0" ? null : "C:\\tools\\yt-dlp.exe",
    js_runtime: "node",
    tools_problem: null,
    onnxruntime: installed ? "C:\\...\\onnxruntime.dll" : null,
    provider: "directml",
    models: [
      { file: "Kim_Vocal_2.onnx", installed, optional: false, bytes: 66759214, about: "", license: "MIT" },
      { file: "yolo26s-seg-fp16.onnx", installed, optional: false, bytes: 21014540, about: "", license: "AGPL-3.0" },
      { file: "siglip2-base-patch16-256-vision-fp16.onnx", installed, optional: false, bytes: 186131676, about: "", license: "Apache-2.0" },
      { file: "bandit-plus-dnr.onnx", installed: false, optional: true, bytes: 98000000, about: "", license: "MIT" },
    ],
    ready: installed,
  });

  // Plays a list of [delay in ms, event] pairs, then an exit.
  function play(script, code) {
    let at = 0;
    const next = () => {
      if (at >= script.length) {
        timer = null;
        handler({ event: "exit", code, cancelled: false });
        return;
      }
      const [delay, event] = script[at++];
      timer = setTimeout(() => {
        handler(event);
        next();
      }, delay);
    };
    next();
  }

  function ramp(make, steps, delay) {
    return Array.from({ length: steps + 1 }, (_, i) => [delay, make(i / steps)]);
  }

  function runScript(options) {
    const link = /^https?:/i.test(options.input);
    const script = [[300, { event: "log", text: "encoder h264_nvenc" }]];
    if (link) script.push(...ramp((f) => ({ event: "fetch", text: `[download] ${(f * 100).toFixed(1)}% of 48.2MiB at 5.1MiB/s` }), 12, 120));
    if (options.audio === "clean") script.push(...ramp((f) => ({ event: "stage", stage: "removing music", done: 90 * f, total: 90 }), 20, 110));
    if (options.blur !== "nobody") {
      script.push(...ramp((f) => ({ event: "stage", stage: "finding people", done: 90 * f, total: 90 }), 20, 110));
      script.push(...ramp((f) => ({ event: "stage", stage: "blurring", done: 90 * f, total: 90 }), 16, 110));
    }
    if (query.get("fail")) {
      script.splice(8, script.length, [200, { event: "log", text: "error: ffmpeg could not read the file" }], [50, { event: "error", message: "ffmpeg could not read the file: moov atom not found" }]);
      return play(script, 1);
    }
    script.push([200, { event: "done", output: "C:\\Users\\you\\Videos\\HalalifyVid\\lecture.halal.mp4", seconds: 41.3 }]);
    play(script, 0);
  }

  function setupScript() {
    const parts = [
      ["onnxruntime-directml-1.21.0", 17410861],
      ["directml-1.15.4", 202292617],
      ["Kim_Vocal_2.onnx", 66759214],
      ["yolo26s-seg-fp16.onnx", 21014540],
      ["siglip2-base-patch16-256-vision-fp16.onnx", 186131676],
    ];
    const script = [[200, { event: "log", text: "installing into C:\\Users\\you\\AppData\\Local\\HalalifyVid" }]];
    for (const [name, total] of parts) {
      script.push(...ramp((f) => ({ event: "download", name, done: Math.round(total * f), total }), 10, 90));
      script.push([150, { event: "installed", name }]);
    }
    script.push([300, { event: "ready", device: "DirectML" }]);
    installed = true;
    play(script, 0);
  }

  window.PREVIEW_BACKEND = {
    onEvent: (h) => (handler = h),
    onDrop: () => {},
    call: async (name, args) => {
      switch (name) {
        case "status":
          return status();
        case "default_output_folder":
          return "C:\\Users\\you\\Videos\\HalalifyVid";
        case "pick_video":
          return "C:\\Users\\you\\Videos\\Friday lecture.mp4";
        case "pick_folder":
          return "D:\\Clean videos";
        case "start_run":
          return runScript(args.options);
        case "start_setup":
        case "install_tools":
          return setupScript();
        case "cancel":
          clearTimeout(timer);
          timer = null;
          handler({ event: "exit", code: null, cancelled: true });
          return;
        default:
          return;
      }
    },
  };
})();
