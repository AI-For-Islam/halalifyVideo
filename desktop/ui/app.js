// The screens of HalalifyVid. They hold no logic of their own about video:
// they ask the backend to start the `halalify` program and draw the events it
// sends ("stage", "download", "fetch", "installed", "ready", "done", "error",
// "log", and "exit" when the process has ended).

(() => {
  "use strict";

  const backend = window.__TAURI__
    ? {
        call: (name, args) => window.__TAURI__.core.invoke(name, args),
        onEvent: (handler) => window.__TAURI__.event.listen("engine", (e) => handler(e.payload)),
        onDrop: (handler) => {
          window.__TAURI__.event.listen("tauri://drag-enter", () => handler("over"));
          window.__TAURI__.event.listen("tauri://drag-leave", () => handler("leave"));
          window.__TAURI__.event.listen("tauri://drag-drop", (e) => handler("drop", e.payload.paths || []));
        },
      }
    : window.PREVIEW_BACKEND; // a browser, for working on the screens: see preview.js

  const $ = (id) => document.getElementById(id);
  const store = {
    get(key, fallback) {
      try {
        const raw = localStorage.getItem("halalify." + key);
        return raw === null ? fallback : JSON.parse(raw);
      } catch {
        return fallback;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem("halalify." + key, JSON.stringify(value));
      } catch {
        /* a full or blocked store only loses a preference */
      }
    },
  };

  // ------------------------------------------------------------ language

  let lang = store.get("lang", (navigator.language || "en").toLowerCase().startsWith("ar") ? "ar" : "en");

  function t(key, values) {
    let text = (window.TEXT[lang] && window.TEXT[lang][key]) || window.TEXT.en[key] || key;
    for (const [name, value] of Object.entries(values || {})) text = text.replace("{" + name + "}", value);
    return text;
  }

  function applyLanguage() {
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
    for (const el of document.querySelectorAll("[data-t]")) el.textContent = t(el.dataset.t);
    for (const el of document.querySelectorAll("[data-t-placeholder]")) el.placeholder = t(el.dataset.tPlaceholder);
    for (const b of document.querySelectorAll("#lang button")) b.setAttribute("aria-pressed", String(b.dataset.lang === lang));
    redraw();
  }

  for (const b of document.querySelectorAll("#lang button")) {
    b.addEventListener("click", () => {
      lang = b.dataset.lang;
      store.set("lang", lang);
      applyLanguage();
    });
  }

  // ------------------------------------------------------------ state

  const DEFAULTS = { blur: "women", audio: "clean", style: "blur", shape: "silhouette", musicModel: "auto", threshold: 0.35, faster: false, outputDir: "" };
  const options = { ...DEFAULTS, ...store.get("options", {}) };
  let input = ""; // the chosen file or link
  let status = null; // the last `halalify doctor` report
  let defaultFolder = "";
  let device = store.get("device", ""); // what setup last said the models run on
  let view = "loading";
  let job = null; // { kind: "run" | "setup" | "tools", ... } while something runs
  let last = { error: "", log: [] };

  const VIEWS = ["loading", "broken", "setup", "home", "work", "done", "error"];
  function show(name) {
    view = name;
    for (const v of VIEWS) $("view-" + v).hidden = v !== name;
    document.querySelector("main").scrollTop = 0;
    redraw();
  }

  function redraw() {
    drawTop();
    if (view === "setup") drawSetup();
    if (view === "home") drawHome();
    if (view === "work") drawWork();
    if (view === "done") drawDone();
  }

  function toast(text) {
    const el = $("toast");
    el.textContent = text;
    el.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => (el.hidden = true), 2600);
  }

  const fileName = (path) => path.split(/[\\/]/).pop() || path;
  const isLink = (text) => /^https?:\/\//i.test(text.trim());

  function clock(seconds) {
    const s = Math.max(0, Math.round(seconds));
    const pad = (n) => String(n).padStart(2, "0");
    return s >= 3600 ? `${Math.floor(s / 3600)}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}` : `${Math.floor(s / 60)}:${pad(s % 60)}`;
  }

  function megabytes(bytes) {
    return bytes >= 1e9 ? (bytes / 1e9).toFixed(2) + " GB" : Math.round(bytes / 1e6) + " MB";
  }

  function appendLog(target, line) {
    last.log.push(line);
    if (last.log.length > 400) last.log.shift();
    const pre = $(target);
    pre.textContent = last.log.join("\n");
    pre.scrollTop = pre.scrollHeight;
  }

  // ------------------------------------------------------------ top bar

  function drawTop() {
    const chip = $("device-chip");
    const usable = status && status.ready && device;
    chip.hidden = !usable;
    if (usable) {
      chip.textContent = t("deviceChip", { device });
      chip.classList.toggle("warn", /cpu/i.test(device));
    }
    $("setup-link").hidden = !(status && view === "home");
  }

  $("setup-link").addEventListener("click", () => openSetup(true));

  // ------------------------------------------------------------ start-up

  async function refresh() {
    try {
      status = await backend.call("status");
      defaultFolder = defaultFolder || (await backend.call("default_output_folder"));
    } catch (e) {
      status = null;
      $("broken-text").textContent = String(e);
      show("broken");
      return false;
    }
    return true;
  }

  async function boot() {
    show("loading");
    if (!(await refresh())) return;
    if (status.ready) show("home");
    else openSetup(false);
  }

  $("broken-retry").addEventListener("click", boot);

  // ------------------------------------------------------------ setup

  let setupNote = { text: "", bad: false };
  let cameFromHome = false;
  let routeChanged = false; // the user picked another graphics route on this visit

  function openSetup(fromHome) {
    cameFromHome = fromHome;
    routeChanged = false;
    setupNote = { text: "", bad: false };
    last = { error: "", log: [] };
    $("setup-log").textContent = "";
    const route = store.get("route", "") || (status && status.provider) || "directml";
    for (const r of document.querySelectorAll('input[name="route"]')) r.checked = r.value === route;
    show("setup");
  }

  function drawSetup() {
    if (!status) return;
    const running = job && (job.kind === "setup" || job.kind === "tools");
    const models = status.models.filter((m) => !m.optional);
    const modelsOk = models.every((m) => m.installed);
    const modelBytes = models.reduce((sum, m) => sum + m.bytes, 0);
    const rows = [
      { name: t("itemFfmpeg"), ok: !!status.ffmpeg },
      { name: t("itemYtdlp"), ok: !!status.ytdlp, optional: true },
      { name: t("itemRuntime"), ok: !!status.onnxruntime },
      { name: t("itemModels"), ok: modelsOk, size: megabytes(modelBytes) },
    ];
    const list = $("checklist");
    list.textContent = "";
    for (const row of rows) {
      const li = document.createElement("li");
      li.className = row.ok ? "ok" : row.optional ? "optional" : "missing";
      const mark = document.createElement("span");
      mark.className = "mark";
      mark.textContent = row.ok ? "✓" : row.optional ? "–" : "!";
      const what = document.createElement("span");
      what.className = "what";
      what.textContent = row.name;
      if (row.size && !row.ok) {
        const size = document.createElement("span");
        size.className = "size num";
        size.textContent = "  " + row.size;
        what.append(size);
      }
      const state = document.createElement("span");
      state.className = "state";
      state.textContent = row.ok ? t("stateOk") : row.optional ? t("stateOptional") : t("stateMissing");
      li.append(mark, what, state);
      list.append(li);
    }

    const needsTools = !status.ffmpeg || !status.ytdlp;
    $("tools-box").hidden = !needsTools;
    $("tools-text").textContent = status.ffmpeg ? t("toolsLinks") : t("toolsMissing");
    $("tools-install").disabled = !!running;

    const installed = !!status.onnxruntime && modelsOk;
    $("route-field").disabled = !!running;
    $("setup-start").hidden = !!running;
    $("setup-start").disabled = !status.ffmpeg;
    // With everything in place the button only looks again, until another route is picked.
    $("setup-start").textContent = installed && status.ffmpeg && !routeChanged ? t("retry") : t("setupStart");
    $("setup-cancel").hidden = !running;
    $("setup-back").hidden = !!running || !(cameFromHome && status.ready);
    $("setup-progress").hidden = !running;
    if (running) {
      const d = job.download;
      $("setup-now").textContent = d ? t("setupDownloading", { name: d.name }) : job.note || t("setupWorking");
      $("setup-bytes").textContent = d && d.total ? `${megabytes(d.done)} / ${megabytes(d.total)}` : "";
      $("setup-bar").style.width = d && d.total ? (100 * d.done) / d.total + "%" : "0%";
      $("setup-bar").parentElement.classList.toggle("busy", !d);
    }
    const note = $("setup-error");
    note.hidden = !setupNote.text;
    note.textContent = setupNote.text;
    note.style.color = setupNote.bad ? "" : "var(--green)";
  }

  for (const r of document.querySelectorAll('input[name="route"]')) {
    r.addEventListener("change", () => {
      routeChanged = true;
      drawSetup();
    });
  }

  $("setup-start").addEventListener("click", async () => {
    const chosen = document.querySelector('input[name="route"]:checked');
    const installed = status && status.onnxruntime && status.models.every((m) => m.installed || m.optional);
    if (installed && status.ffmpeg && !routeChanged) {
      if (await refresh()) status.ready ? show("home") : drawSetup();
      return;
    }
    await begin({ kind: "setup", route: chosen ? chosen.value : "auto" }, "start_setup", { provider: chosen ? chosen.value : "auto" });
  });

  $("tools-install").addEventListener("click", () => begin({ kind: "tools" }, "install_tools", {}));
  $("setup-cancel").addEventListener("click", () => backend.call("cancel"));
  $("setup-back").addEventListener("click", () => show("home"));

  async function begin(newJob, command, args) {
    last = { error: "", log: [] };
    setupNote = { text: "", bad: false };
    job = { ...newJob, started: Date.now() };
    redraw();
    try {
      await backend.call(command, args);
    } catch (e) {
      const wasRun = job.kind === "run";
      job = null;
      if (wasRun) fail(String(e));
      else {
        setupNote = { text: String(e), bad: true };
        redraw();
      }
    }
  }

  // ------------------------------------------------------------ home

  function setInput(value) {
    input = value.trim().replace(/^"|"$/g, "");
    $("link").value = "";
    drawHome();
  }

  function drawHome() {
    const has = !!input;
    $("drop").classList.toggle("has", has);
    $("drop-empty").hidden = has;
    $("drop-chosen").hidden = !has;
    if (has) {
      $("chosen-kind").textContent = isLink(input) ? t("kindLink") : t("kindFile");
      $("chosen-name").textContent = isLink(input) ? input : fileName(input);
      $("chosen-name").title = input;
    }
    for (const group of document.querySelectorAll(".segmented[data-option]")) {
      for (const b of group.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.value === options[group.dataset.option]));
    }
    const blurring = options.blur !== "nobody";
    $("look-option").classList.toggle("off", !blurring);
    $("shape-option").classList.toggle("off", !blurring);
    $("threshold-field").classList.toggle("off", !blurring || options.blur === "everyone");
    $("music-field").classList.toggle("off", options.audio !== "clean");
    $("threshold").value = options.threshold;
    $("threshold-out").textContent = Number(options.threshold).toFixed(2);
    $("faster").checked = !!options.faster;
    $("music-hint").textContent = t({ auto: "musicHintAuto", dialogue: "musicHintDialogue", vocals: "musicHintVocals" }[options.musicModel]);
    $("save-path").textContent = options.outputDir || t("saveBeside", { folder: defaultFolder });
    $("save-reset").hidden = !options.outputDir;

    const linkWithoutYtdlp = has && isLink(input) && status && !status.ytdlp;
    const nothingToDo = options.blur === "nobody" && options.audio === "keep";
    $("start").disabled = !has || linkWithoutYtdlp || nothingToDo;
    $("start").title = linkWithoutYtdlp ? t("startNeedsYtdlp") : "";
  }

  function setOption(name, value) {
    options[name] = value;
    store.set("options", options);
    drawHome();
  }

  for (const group of document.querySelectorAll(".segmented[data-option]")) {
    for (const b of group.querySelectorAll("button")) b.addEventListener("click", () => setOption(group.dataset.option, b.dataset.value));
  }
  $("threshold").addEventListener("input", (e) => setOption("threshold", Number(e.target.value)));
  $("faster").addEventListener("change", (e) => setOption("faster", e.target.checked));
  $("save-reset").addEventListener("click", () => setOption("outputDir", ""));
  $("save-choose").addEventListener("click", async () => {
    const folder = await backend.call("pick_folder");
    if (folder) setOption("outputDir", folder);
  });

  $("browse").addEventListener("click", async () => {
    const file = await backend.call("pick_video");
    if (file) setInput(file);
  });
  $("chosen-clear").addEventListener("click", () => {
    input = "";
    drawHome();
    $("link").focus();
  });
  $("link").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && $("link").value.trim()) setInput($("link").value);
  });
  $("link").addEventListener("paste", () => setTimeout(() => isLink($("link").value) && setInput($("link").value), 0));
  $("link").addEventListener("change", () => $("link").value.trim() && setInput($("link").value));

  backend.onDrop((what, paths) => {
    if (view !== "home") return;
    $("drop").classList.toggle("over", what === "over");
    if (what === "drop" && paths.length) setInput(paths[0]);
  });

  $("start").addEventListener("click", startRun);

  // ------------------------------------------------------------ working

  // The rows of the working screen, in the order the program goes through them.
  const STAGES = [
    { id: "fetch", label: "stageFetch" },
    { id: "removing music", label: "stageMusic" },
    { id: "finding people", label: "stagePeople" },
    { id: "blurring", label: "stageBlur" },
  ];

  async function startRun() {
    const wanted = STAGES.filter((s) => {
      if (s.id === "fetch") return isLink(input);
      if (s.id === "removing music") return options.audio === "clean";
      return options.blur !== "nobody";
    }).map((s) => ({ ...s, state: "waiting", done: 0, total: 0, since: 0, text: "" }));
    $("work-log").textContent = "";
    show("work");
    await begin(
      { kind: "run", input, stages: wanted, download: null, output: "", seconds: 0 },
      "start_run",
      { options: { input, outputDir: options.outputDir, blur: options.blur, shape: options.shape, style: options.style, threshold: options.threshold, audio: options.audio, musicModel: options.musicModel, faster: !!options.faster } },
    );
  }

  function stage(id) {
    return job && job.stages ? job.stages.find((s) => s.id === id) : null;
  }

  // A stage that reports progress is the one running; those before it are over.
  function advanceTo(id) {
    const at = job.stages.findIndex((s) => s.id === id);
    if (at < 0) return null;
    job.stages.forEach((s, i) => {
      if (i < at && s.state !== "done") s.state = s.state === "active" ? "done" : "skipped";
    });
    const s = job.stages[at];
    if (s.state === "waiting") {
      s.state = "active";
      s.since = Date.now();
    }
    return s;
  }

  function drawWork() {
    if (!job || job.kind !== "run") return;
    $("work-name").textContent = isLink(job.input) ? job.input : fileName(job.input);
    const list = $("stages");
    list.textContent = "";
    const anyActive = job.stages.some((s) => s.state === "active");
    const rows = [...job.stages];
    if (job.download) rows.unshift({ id: "model", label: "stageModel", state: "active", done: job.download.done, total: job.download.total, bytes: true, name: job.download.name });
    rows.forEach((s, i) => {
      const li = document.createElement("li");
      // Before the first event the program is starting up: show that on the first row.
      const preparing = !anyActive && !job.download && s.state === "waiting" && i === rows.findIndex((r) => r.state === "waiting") && !rows.some((r) => r.state === "done");
      li.className = "stage " + (preparing ? "active" : s.state);
      const dot = document.createElement("span");
      dot.className = "dot";
      const head = document.createElement("div");
      head.className = "head";
      const label = document.createElement("span");
      label.className = "label";
      label.textContent = t(s.label) + (s.name ? " · " + s.name : "");
      const info = document.createElement("span");
      info.className = "info num";
      const fraction = s.total > 0 ? Math.min(1, s.done / s.total) : 0;
      if (preparing) info.textContent = t("statePreparing");
      else if (s.state === "waiting") info.textContent = t("stateWaiting");
      else if (s.state === "done") info.textContent = t("stateDone");
      else if (s.state === "skipped") info.textContent = t("stateSkipped");
      else if (s.bytes) info.textContent = `${megabytes(s.done)} / ${megabytes(s.total)}`;
      else if (s.id === "fetch") info.textContent = s.text || "";
      else {
        const took = (Date.now() - s.since) / 1000;
        const speed = took > 1 && s.done > 0 ? "  ·  " + t("speed", { x: (s.done / took).toFixed(1) }) : "";
        info.textContent = Math.round(fraction * 100) + "%" + speed;
      }
      head.append(label, info);
      const bar = document.createElement("div");
      bar.className = "bar" + ((s.state === "active" && s.total <= 0) || preparing ? " busy" : "");
      const fill = document.createElement("i");
      fill.style.width = fraction * 100 + "%";
      bar.append(fill);
      li.append(dot, head, bar);
      list.append(li);
    });
    $("work-clock").textContent = t("elapsed", { time: clock((Date.now() - job.started) / 1000) });
  }

  setInterval(() => {
    if (view === "work" && job) $("work-clock").textContent = t("elapsed", { time: clock((Date.now() - job.started) / 1000) });
  }, 1000);

  $("work-cancel").addEventListener("click", () => backend.call("cancel"));

  // ------------------------------------------------------------ events

  backend.onEvent((e) => {
    if (!job) return;
    const logTarget = job.kind === "run" ? "work-log" : "setup-log";
    switch (e.event) {
      case "log":
        appendLog(logTarget, e.text);
        if (/^error:/i.test(e.text)) last.error = last.error || e.text.replace(/^error:\s*/i, "");
        return; // a log line changes nothing else on screen
      case "stage": {
        job.download = null;
        const s = advanceTo(e.stage);
        if (s) {
          s.done = e.done;
          s.total = e.total;
          if (e.total > 0 && e.done >= e.total) s.state = "done";
        }
        break;
      }
      case "fetch": {
        const s = advanceTo("fetch");
        if (s) {
          const percent = /(\d+(?:\.\d+)?)%/.exec(e.text);
          if (percent) {
            s.done = Number(percent[1]);
            s.total = 100;
          }
          s.text = percent ? Math.round(Number(percent[1])) + "%" : "";
        }
        break;
      }
      case "download":
        job.download = e.total > 0 && e.done >= e.total ? null : { name: e.name, done: e.done, total: e.total };
        break;
      case "installed":
        job.download = null;
        job.note = t("setupInstalled", { name: e.name });
        appendLog(logTarget, "installed " + e.name);
        break;
      case "ready":
        job.device = e.device || "";
        break;
      case "done":
        job.output = e.output;
        job.seconds = e.seconds;
        break;
      case "error":
        last.error = e.message;
        break;
      case "exit":
        finish(e);
        return;
    }
    redraw();
  });

  async function finish(exit) {
    const ended = job;
    job = null;
    if (ended.kind === "run") {
      if (exit.cancelled) {
        show("home");
        toast(t("stopped"));
      } else if (exit.code === 0 && ended.output) {
        last.done = ended;
        show("done");
      } else {
        fail(last.error);
      }
      return;
    }
    // Setup or the tools: look again at what is installed.
    await refresh();
    if (!status) return;
    if (exit.cancelled) setupNote = { text: t("stopped"), bad: false };
    else if (ended.kind === "tools") setupNote = status.ffmpeg && status.ytdlp ? { text: "", bad: false } : { text: t("setupToolsFailed"), bad: true };
    else if (exit.code !== 0) setupNote = { text: last.error || t("setupFailed"), bad: true };
    else {
      store.set("route", ended.route);
      device = ended.device || "";
      store.set("device", device);
      const slow = /cpu/i.test(device) && ended.route !== "cpu";
      setupNote = { text: slow ? t("setupReadyCpu") : device ? t("setupReady", { device }) : "", bad: slow };
      if (status.ready && !slow) {
        show("home");
        if (device) toast(t("setupReady", { device }));
        return;
      }
    }
    show("setup");
  }

  // ------------------------------------------------------------ done and error

  function drawDone() {
    if (!last.done) return;
    $("done-path").textContent = last.done.output;
    $("done-time").textContent = t("doneTime", { time: clock(last.done.seconds) });
  }

  const complain = (e) => toast(String(e));
  $("done-play").addEventListener("click", () => backend.call("open_file", { path: last.done.output }).catch(complain));
  $("done-show").addEventListener("click", () => backend.call("show_in_folder", { path: last.done.output }).catch(complain));
  $("done-again").addEventListener("click", () => {
    input = "";
    show("home");
  });

  function fail(message) {
    $("error-text").textContent = message || t("errorUnknown");
    // A link that yt-dlp could not fetch: the usual cure is a newer yt-dlp.
    const hint = $("error-hint");
    hint.hidden = !/yt-dlp could not download/i.test(message || "");
    hint.textContent = t("errorYtdlp");
    $("error-log").textContent = last.log.join("\n");
    show("error");
  }

  $("error-retry").addEventListener("click", startRun);
  $("error-back").addEventListener("click", () => show("home"));

  applyLanguage();
  boot();
})();
