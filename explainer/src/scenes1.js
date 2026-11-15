// Scenes 1-7: title, overview, pipeline, STFT, MDX-Net, BandIt, detection.
import * as THREE from 'three';
import {
  C, hex, ep, ease, prog, lerp, clamp01, smooth, noise1, rng, el, place, show, label3d, setOp,
  glow, std, rbox, node, lightNode, tube, growTube, flow, fatLine, setLine, canvasTex, plane, framed,
  speaking, cell, spectro, STREET, drawStreet, drawDetections, drawBlurred,
} from './kit.js';

const V = (x, y, z = 0) => new THREE.Vector3(x, y, z);

// ---------- 1. title ----------
export function title({ group, root, cue, camera }) {
  const fr = new THREE.Group();
  fr.position.set(3.7, 0.15, 0);
  group.add(fr);
  const street = canvasTex(960, 540, (c, w, h) => drawStreet(c, w, h));
  const blurred = canvasTex(960, 540, (c, w, h) => drawBlurred(c, w, h));
  const f = framed(5.2, 2.925, street, C.emerald);
  const fb = plane(5.2, 2.925, blurred, 0);
  fb.position.z = 0.005;
  fr.add(f, fb);
  const N = 220, R = 3.2;
  const ring = (color, w) => {
    const pts = [];
    for (let i = 0; i <= N; i++) pts.push(V(Math.cos((i / N) * Math.PI * 2) * R, Math.sin((i / N) * Math.PI * 2) * R, 0));
    const l = fatLine(pts, color, w, 1);
    fr.add(l);
    return l;
  };
  const musicRing = ring(C.amber, 3);
  const voiceRing = ring(C.emerald, 4);
  const tMusic = cue('music'), tBlur = cue('blurred');
  const arr = new Float32Array((N + 1) * 3);
  const ringPos = (t, kind, amt) => {
    for (let i = 0; i <= N; i++) {
      const a = (i / N) * Math.PI * 2;
      const tau = t * 1.2 + i * 0.03;
      let d;
      if (kind === 'm') d = 0.32 * Math.abs(Math.sin(a * 24 + t * 6)) * (0.5 + 0.5 * noise1(i * 0.31 + t * 3, 2)) + 0.15 * Math.exp(-((t * 2) % 1) * 6);
      else d = 0.22 * speaking(t * 0.8 + a) * Math.abs(Math.sin(a * 9 + t * 4)) * (0.6 + 0.4 * noise1(tau * 2, 5));
      const r = R + d * amt;
      arr[i * 3] = Math.cos(a) * r;
      arr[i * 3 + 1] = Math.sin(a) * r;
      arr[i * 3 + 2] = 0;
    }
    return arr;
  };

  const box = place(el('div', '', '', root), 130, 300, 820);
  const k = el('div', 'kicker', 'An explainer', box);
  const h = el('div', 'h1', 'Halalify<span class="em">Vid</span>', box);
  h.style.marginTop = '18px';
  const s = el('div', 'sub', 'Removes the music. Blurs who should not be seen.<br>Runs entirely on your own machine.', box);
  s.style.marginTop = '28px';
  const chips = el('div', 'row', '', box);
  chips.style.marginTop = '40px';
  const c1 = el('span', 'chip', '<i class="sk"></i>video file', chips);
  const c2 = el('span', 'chip', '<i class="ro"></i>YouTube link', chips);
  const c3 = el('span', 'chip', '<i class="em"></i>clean video out', chips);
  c1.querySelector('i').style.color = hex(C.sky);
  c2.querySelector('i').style.color = hex(C.rose);
  c3.querySelector('i').style.color = hex(C.emerald);
  const lab1 = label3d(root, '<div class="tag am">music</div>');
  const lab2 = label3d(root, '<div class="tag em">voice only</div>');

  return {
    cam: (lt) => ({ p: [0.6 - lt * 0.03, 0.4, 15.5 - lt * 0.12], l: [0.9, 0.1, 0] }),
    update(lt) {
      show(k, ep(lt, 0.2, 0.8));
      show(h, ep(lt, 0.4, 0.9), 30);
      show(s, ep(lt, 0.9, 0.9));
      show(c1, ep(lt, 2.2, 0.5));
      show(c2, ep(lt, 2.5, 0.5));
      show(c3, ep(lt, tMusic, 0.5));
      fr.rotation.y = -0.32 + Math.sin(lt * 0.4) * 0.05;
      fr.rotation.x = 0.04;
      const mOff = ep(lt, tMusic, 1.2);
      setLine(musicRing, ringPos(lt, 'm', 1 - mOff));
      setOp(musicRing, 1 - mOff * 0.85);
      setLine(voiceRing, ringPos(lt, 'v', 1));
      setOp(voiceRing, 0.3 + 0.7 * mOff);
      fb.material.userData.o = ep(lt, tBlur, 1.0);
      fr.updateMatrixWorld(true);
      lab1.at(camera, fr.localToWorld(V(-2.9, 2.75, 0)), ep(lt, 1.6, 0.5) * (1 - mOff));
      lab2.at(camera, fr.localToWorld(V(2.9, 2.75, 0)), mOff);
    },
  };
}

// ---------- 2. overview ----------
export function overview({ group, root, cue, camera }) {
  const core = node(3.2, 1.7, C.emerald);
  const ort = node(2.6, 1.3, C.sky);
  ort.position.set(-5.4, 0.3, 0);
  const ff = node(2.6, 1.3, C.amber);
  ff.position.set(5.4, 0.3, 0);
  group.add(core, ort, ff);
  const l1 = tube([V(-4.1, 0.3, 0), V(-2.6, 0.15, 0.2), V(-1.6, 0, 0)], 0.04, C.sky, 0.9);
  const l2 = tube([V(1.6, 0, 0), V(2.6, 0.15, 0.2), V(4.1, 0.3, 0)], 0.04, C.amber, 0.9);
  group.add(l1, l2);
  const f1 = flow(l1.userData.curve, 6, C.sky, 0.07);
  const f2 = flow(l2.userData.curve, 6, C.amber, 0.07);
  group.add(f1, f2);
  const models = ['MDX-Net', 'BandIt Plus', 'YOLO26-seg', 'SigLIP 2'];
  const colors = [C.amber, C.gold, C.sky, C.rose];
  const cubes = models.map((m, i) => {
    const c = rbox(0.55, 0.55, 0.55, 0.08, std(0x0f172a, colors[i], 0.9));
    group.add(c);
    return c;
  });
  const gpu = rbox(1.8, 0.5, 0.9, 0.06, std(0x111827, C.emerald, 0.5));
  gpu.position.set(-5.4, -1.6, 0);
  group.add(gpu);
  const reels = [];
  for (let i = 0; i < 5; i++) {
    const r = rbox(0.55, 0.32, 0.05, 0.03, glow(C.gold, 0.85));
    group.add(r);
    reels.push(r);
  }

  const L = {
    core: label3d(root, '<div class="t" style="font-size:34px">halalify</div><div class="d">one Rust program</div>'),
    ort: label3d(root, '<div class="t">ONNX Runtime</div><div class="d">runs the models</div>'),
    ff: label3d(root, '<div class="t">ffmpeg</div><div class="d">all media work</div>'),
    gpu: label3d(root, '<div class="d" style="color:#e2e8f0">GPU</div>'),
    m: models.map((m) => label3d(root, `<div class="d" style="color:#e2e8f0;font-size:17px">${m}</div>`)),
    mods: label3d(root, `<div class="d" style="font-size:18px">${models.join(' · ')}</div>`),
  };
  const chips = place(el('div', 'row', '', root), 0, 800, 1920);
  chips.style.justifyContent = 'center';
  const cs = [
    ['No training: published models as they are', C.emerald],
    ['100% local', C.sky],
    ['Single binary', C.amber],
  ].map(([t, c]) => {
    const e = el('span', 'chip', `<i></i>${t}`, chips);
    e.querySelector('i').style.color = hex(c);
    return e;
  });
  const tCore = cue('Rust'), tPub = cue('published'), tOrt = cue('Onyx'), tGpu = cue('GPU'), tFf = cue('FFmpeg');

  return {
    cam: (lt) => ({ p: [Math.sin(lt * 0.12) * 1.2, 1.2, 15 - lt * 0.1], l: [0, -0.1, 0] }),
    update(lt) {
      const pc = ep(lt, tCore - 0.4, 0.7, ease.back);
      core.scale.setScalar(Math.max(0.001, pc));
      lightNode(core, 0.6 + 0.4 * Math.sin(lt * 2) ** 2);
      core.scale.multiplyScalar(Math.max(0.001, pc));
      core.rotation.y = Math.sin(lt * 0.5) * 0.15;
      const po = ep(lt, tOrt - 0.2, 0.6, ease.back);
      ort.scale.setScalar(Math.max(0.001, po));
      const pf = ep(lt, tFf - 0.2, 0.6, ease.back);
      ff.scale.setScalar(Math.max(0.001, pf));
      setOp(l1, po * 0.9);
      setOp(l2, pf * 0.9);
      f1.userData.tick(lt, 0.35, po);
      f2.userData.tick(lt, 0.35, pf);
      const pm = ep(lt, tPub - 0.2, 0.8);
      const orbit = ep(lt, tOrt - 0.1, 1.1, ease.inOut);
      cubes.forEach((c, i) => {
        const a = lt * 0.7 + (i / 4) * Math.PI * 2;
        const row = V(-1.5 + i * 1.0, 1.95, 0.4);
        const orb = V(-5.4 + Math.cos(a) * 1.95, 0.3 + Math.sin(a) * 1.2, 0.9);
        c.position.lerpVectors(row, orb, orbit);
        c.rotation.set(lt * 0.7 + i, lt * 0.9, 0);
        c.scale.setScalar(Math.max(0.001, ep(lt, tPub - 0.2 + i * 0.15, 0.5, ease.back)));
        L.m[i].at(camera, c.position.clone().add(V(0, 0.6, 0)), pm * (1 - orbit));
      });
      L.mods.at(camera, V(-5.4, -2.35, 0.5), orbit);
      gpu.scale.setScalar(Math.max(0.001, ep(lt, tGpu - 0.2, 0.5, ease.back)));
      reels.forEach((r, i) => {
        const u = (lt * 0.25 + i / 5) % 1;
        r.position.set(5.4 + lerp(-1.6, 1.6, u), -1.2, 0.3);
        r.rotation.y = 0.2;
        r.scale.setScalar(Math.max(0.001, pf * Math.sin(Math.PI * u)));
      });
      L.core.at(camera, V(0, 0, 0.3), pc);
      L.ort.at(camera, V(-5.4, 0.3, 0.3), po);
      L.ff.at(camera, V(5.4, 0.3, 0.3), pf);
      L.gpu.at(camera, V(-5.4, -1.6, 0.5), ep(lt, tGpu, 0.4));
      show(cs[0], ep(lt, tPub, 0.5));
      show(cs[1], ep(lt, tGpu, 0.5));
      show(cs[2], ep(lt, tFf + 0.6, 0.5));
    },
  };
}

// ---------- 3. pipeline ----------
export function pipeline({ group, root, cue, camera }) {
  const N = {
    input: [-8.6, 0, C.white, 'Input', 'file or link'],
    ytdlp: [-5.8, 0, C.white, 'yt-dlp', 'links only'],
    probe: [-3.0, 0, C.white, 'ffprobe', 'size · fps · rotation'],
    audio: [0.6, 2.4, C.amber, 'Audio', 'remove the music'],
    analysis: [0.6, -2.4, C.sky, 'Analysis', 'find · follow · classify'],
    decision: [3.6, -2.4, C.violet, 'Decision', 'blur plan'],
    render: [6.4, 0, C.emerald, 'Render', 'blur · encode · mux'],
    output: [9.2, 0, C.emerald, 'Output', 'video.halal.mp4'],
  };
  const nodes = {}, labels = {};
  for (const [k, [x, y, c, t, d]] of Object.entries(N)) {
    const n = node(2.1, 1.0, c);
    n.position.set(x, y, 0);
    group.add(n);
    nodes[k] = n;
    labels[k] = {
      t: label3d(root, `<div class="t">${t}</div>`),
      d: label3d(root, `<div class="d">${d}</div>`),
    };
  }
  const E = [
    ['input', 'ytdlp'], ['ytdlp', 'probe'], ['probe', 'audio'], ['probe', 'analysis'],
    ['analysis', 'decision'], ['decision', 'render'], ['audio', 'render'], ['render', 'output'],
  ];
  const edges = E.map(([a, b]) => {
    const A = nodes[a].position, B = nodes[b].position;
    const pts = [V(A.x + 1.05, A.y, 0), V((A.x + B.x) / 2 + 0.3, lerp(A.y, B.y, 0.15), 0.15), V((A.x + B.x) / 2 - 0.3, lerp(A.y, B.y, 0.85), 0.15), V(B.x - 1.05, B.y, 0)];
    if (A.y === B.y) pts.splice(1, 2, V((A.x + B.x) / 2, A.y, 0.1));
    const col = N[b][2] === C.white ? C.slate : N[b][2];
    const tb = growTube(pts, 0.035, col === C.slate ? 0x64748b : col, 0.95);
    const fl = flow(tb.userData.curve, 4, col === C.slate ? 0x94a3b8 : col, 0.065);
    group.add(tb, fl);
    return { a, b, tb, fl };
  });
  const T = {
    input: cue('Links'), ytdlp: cue('downloaded'), probe: cue('probe'), audio: cue('audio'),
    analysis: cue('analysis'), decision: cue('decision'), render: cue('render'), output: cue('result'),
  };
  const tSplit = cue('splits');
  const startOf = (b) => (b === 'audio' || b === 'analysis' ? Math.min(T[b], tSplit) : T[b]);

  return {
    cam: (lt) => {
      const u = ease.inOut(lt / 20);
      return { p: [lerp(-3.5, 0.4, u), lerp(1.2, 0.4, u), lerp(16, 20.5, u)], l: [lerp(-3.5, 0.3, u), lerp(0, -0.2, u), 0] };
    },
    update(lt) {
      for (const k of Object.keys(N)) {
        const n = nodes[k];
        const p = ep(lt, T[k] - 0.25, 0.55, ease.back);
        n.scale.setScalar(Math.max(0.001, p));
        const active = ep(lt, T[k] - 0.25, 0.4) * (1 - 0.55 * ep(lt, T[k] + 2.2, 1.2));
        lightNode(n, active);
        n.scale.multiplyScalar(Math.max(0.001, p));
        const pos = n.position;
        labels[k].t.at(camera, pos.clone().add(V(0, 0, 0.3)), p);
        labels[k].d.at(camera, pos.clone().add(V(0, -0.85, 0)), ep(lt, T[k], 0.5));
      }
      for (const e of edges) {
        const s = startOf(e.b) - 0.5;
        e.tb.userData.grow(ep(lt, s, 0.6, ease.inOut));
        e.fl.userData.tick(lt, 0.3, ep(lt, s + 0.5, 0.5));
      }
    },
  };
}

// ---------- 4. STFT ----------
export function stft({ group, root, cue, camera }) {
  const NW = 260;
  const waves = [0, 1].map((ch) => {
    const pts = [];
    for (let i = 0; i < NW; i++) pts.push(V(0, 0, 0));
    const l = fatLine(pts, ch ? C.gold : C.amber, 2.2, 1);
    l.position.set(-8.6, 1.0 - ch * 1.6, 0);
    group.add(l);
    return l;
  });
  const warr = new Float32Array(NW * 3);
  const wave = (t, ch) => {
    for (let i = 0; i < NW; i++) {
      const x = (i / (NW - 1)) * 4.6;
      const tau = t * 1.4 + i * 0.022;
      const env = 0.25 + 0.5 * speaking(tau) + 0.35 * Math.exp(-((tau % 0.5) * 12));
      const y = env * 0.62 * (Math.sin(i * 1.7 + t * 30 + ch) * 0.6 + Math.sin(i * 0.53 + t * 11) * 0.4) * (0.6 + 0.4 * noise1(i * 0.2 + t * 4, ch + 1));
      warr[i * 3] = x;
      warr[i * 3 + 1] = y;
      warr[i * 3 + 2] = 0;
    }
    return warr;
  };
  const stn = node(1.7, 1.0, C.violet);
  stn.position.set(-2.4, 0.2, 0);
  group.add(stn);
  const a1 = growTube([V(-3.9, 0.2, 0), V(-3.25, 0.2, 0)], 0.03, C.violet);
  const a2 = growTube([V(-1.5, 0.2, 0), V(0.0, 0.2, 0)], 0.03, C.violet);
  group.add(a1, a2);
  const COLS = 64, ROWS = 36;
  const sp = spectro(COLS, ROWS, 0.12, 0.11, 1.5);
  const spg = new THREE.Group();
  spg.position.set(3.5, -0.9, 0);
  spg.rotation.set(0.0, -0.18, 0);
  spg.add(sp);
  group.add(spg);
  // overlapping chunk windows over the time axis
  const chunks = [0, 1, 2].map((i) => {
    const m = rbox(2.0, 1.7, ROWS * 0.11 + 0.2, 0.1, glow(C.violet, 0.07));
    spg.add(m);
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2.0, 1.7, ROWS * 0.11 + 0.2)), new THREE.LineBasicMaterial({ color: C.violet, transparent: true, opacity: 0.8 }));
    m.add(edge);
    return m;
  });
  const L = {
    st: label3d(root, '<div class="t">STFT</div>'),
    wave: label3d(root, '<div class="d" style="color:#fde68a">44.1 kHz · stereo</div>'),
    sp: label3d(root, '<div class="d">frequency ↑ · time →</div>'),
  };
  const panel = place(el('div', 'panel col', '', root), 110, 640, 560);
  const head = el('div', 'kicker', 'Audio · step 0', panel);
  head.style.color = hex(C.amber);
  const items = [
    ['Decode', 'ffmpeg → 44.1 kHz stereo PCM'],
    ['Transform', 'short-time Fourier, in Rust, equal to <span class="mono">torch.stft</span>'],
    ['Stream', 'overlapping chunks, overlap-add back'],
  ].map(([a, b]) => el('div', 'sub', `<b>${a}</b> <span class="muted" style="font-size:24px">${b}</span>`, panel));
  items.forEach((e) => (e.style.fontSize = '26px'));
  const mem = place(el('div', 'panel', '', root), 1300, 700, 500);
  mem.innerHTML = `<div class="kicker" style="color:${hex(C.emerald)}">Memory vs. video length</div>
  <svg width="440" height="170" viewBox="0 0 440 170" style="margin-top:12px">
    <line x1="30" y1="150" x2="430" y2="150" stroke="#475569" stroke-width="2"/>
    <line x1="30" y1="150" x2="30" y2="10" stroke="#475569" stroke-width="2"/>
    <path class="naive" d="M30 140 L420 18" stroke="${hex(C.rose)}" stroke-width="4" stroke-dasharray="10 8" fill="none"/>
    <text x="300" y="34" fill="${hex(C.rose)}" font-size="19">load it all</text>
    <path class="flat" d="M30 112 L420 112" stroke="${hex(C.emerald)}" stroke-width="6" fill="none"/>
    <text x="300" y="100" fill="${hex(C.emerald)}" font-size="19">streaming</text>
    <text x="300" y="168" fill="#94a3b8" font-size="16">length →</text>
  </svg>`;
  const naive = mem.querySelector('.naive'), flat = mem.querySelector('.flat');
  naive.setAttribute('pathLength', '1');
  flat.setAttribute('pathLength', '1');
  const tSound = cue('sound'), tKhz = cue('44.1', 1, cue('kilohertz')), tF = cue('Fourier'), tSpec = cue('spectrogram');
  const tChunk = cue('overlapping'), tMem = cue('memory');

  return {
    cam: (lt) => ({ p: [lerp(-0.8, 0.6, ease.inOut(lt / 20)), 4.4, 17 - lt * 0.05], l: [-0.3, -0.2, 0] }),
    update(lt, t) {
      const pw = ep(lt, tSound - 0.3, 0.8);
      waves.forEach((w, ch) => {
        setLine(w, wave(lt, ch));
        setOp(w, pw);
      });
      stn.scale.setScalar(Math.max(0.001, ep(lt, tF - 0.4, 0.5, ease.back)));
      lightNode(stn, ep(lt, tF - 0.4, 0.5) * (0.6 + 0.4 * Math.sin(lt * 5) ** 2));
      a1.userData.grow(ep(lt, tF - 0.4, 0.5));
      a2.userData.grow(ep(lt, tF, 0.5));
      const reveal = ep(lt, tF + 0.2, 2.5, ease.inOut);
      sp.userData.update((c, r) => {
        if (c / COLS > reveal) return { v: 0, m: 0 };
        return cell(lt * 1.4 + c * 0.09, r, ROWS);
      });
      const pc = ep(lt, tChunk - 0.2, 0.6);
      chunks.forEach((m, i) => {
        const u = ((lt - tChunk) * 0.35 + i / 3) % 1;
        m.position.set(lerp(-3.6, 3.6, (u + 1) % 1), 0.85, 0);
        const fade = Math.sin(Math.PI * ((u + 1) % 1));
        m.material.userData.o = 0.08 * pc * fade;
        m.children[0].material.userData.o = 0.7 * pc * fade;
      });
      L.st.at(camera, V(-2.4, 0.2, 0.3), ep(lt, tF - 0.4, 0.5));
      L.wave.at(camera, V(-6.3, 2.0, 0), ep(lt, tKhz - 0.2, 0.5));
      L.sp.at(camera, V(3.5, 1.9, -1.2), ep(lt, tSpec, 0.5));
      show(panel, ep(lt, tKhz, 0.6));
      items.forEach((e, i) => show(e, ep(lt, [tKhz, tF, tChunk][i], 0.5), 10));
      show(mem, ep(lt, tMem - 0.3, 0.6));
      naive.style.strokeDashoffset = '0';
      naive.style.strokeDasharray = `${ep(lt, tMem, 1.2)} 1`;
      flat.style.strokeDasharray = `${ep(lt, tMem + 0.6, 1.2)} 1`;
    },
  };
}

// ---------- 5. MDX-Net ----------
export function mdx({ group, root, cue, camera }) {
  const lv = [
    { x: -3.0, y: 1.3, h: 2.2, w: 0.28 }, { x: -2.0, y: 0.3, h: 1.6, w: 0.42 },
    { x: -1.0, y: -0.7, h: 1.15, w: 0.58 }, { x: 0, y: -1.6, h: 0.8, w: 0.8 },
    { x: 1.0, y: -0.7, h: 1.15, w: 0.58 }, { x: 2.0, y: 0.3, h: 1.6, w: 0.42 }, { x: 3.0, y: 1.3, h: 2.2, w: 0.28 },
  ];
  const blocks = lv.map((b, i) => {
    const m = rbox(b.w, b.h, 1.3, 0.05, std(0x1e1b4b, C.violet, 0.5));
    m.position.set(b.x, b.y, 0);
    group.add(m);
    return m;
  });
  const skips = [0, 1, 2].map((i) => {
    const a = lv[i], b = lv[6 - i];
    const tb = growTube([V(a.x, a.y + a.h / 2, 0), V(lerp(a.x, b.x, 0.5), a.y + a.h / 2 + 0.7 - i * 0.1, 0), V(b.x, b.y + b.h / 2, 0)], 0.025, C.violet, 0.7);
    group.add(tb);
    return tb;
  });
  const path = new THREE.CatmullRomCurve3(lv.map((b) => V(b.x, b.y, 0.75)));
  const pulse = flow(path, 5, C.emerald, 0.09);
  group.add(pulse);
  const CO = 34, RO = 24;
  const inSp = spectro(CO, RO, 0.1, 0.1, 1.1);
  inSp.position.set(-7.2, -0.8, 0);
  inSp.rotation.y = 0.35;
  const outSp = spectro(CO, RO, 0.1, 0.1, 1.1);
  outSp.position.set(7.2, -0.8, 0);
  outSp.rotation.y = -0.35;
  group.add(inSp, outSp);
  const L = {
    net: label3d(root, '<div class="t">U-Net</div><div class="d">spectrogram → spectrogram</div>'),
    inp: label3d(root, '<div class="d" style="color:#fde68a">mix: voice + music</div>'),
    out: label3d(root, '<div class="d" style="color:#6ee7b7">voices</div>'),
    warn: label3d(root, `<div class="tag" style="color:${hex(C.gold)};border-color:${hex(C.gold)}">faint music where nobody speaks</div>`),
  };
  const head = place(el('div', '', '', root), 110, 120, 900);
  head.innerHTML = `<div class="kicker" style="color:${hex(C.amber)}">Audio · step 1</div><div class="h2" style="margin-top:10px">MDX-Net <span class="muted">Kim Vocal 2</span></div>`;
  const stats = place(el('div', 'row', '', root), 0, 790, 1920);
  stats.style.justifyContent = 'center';
  stats.style.gap = '26px';
  const st = [
    ['40×', 'real time', C.emerald],
    ['−25 to −55 dB', 'music, where voices are', C.amber],
    ['every voice', 'kept, sung ones too', C.sky],
  ].map(([a, b, c]) => {
    const e = el('div', 'panel', `<div class="stat" style="color:${hex(c)};font-size:50px">${a}</div><div class="statlbl">${b}</div>`, stats);
    e.style.padding = '18px 26px';
    return e;
  });
  const tNet = cue('U-Net', 1, cue('U')), tVoice = cue('voices'), tForty = cue('forty'), tDb = cue('twenty-five', 1, cue('twenty')), tNobody = cue('nobody');

  return {
    cam: (lt) => ({ p: [Math.sin(lt * 0.15) * 1.5, 2.0, 18 - lt * 0.1], l: [0, -0.2, 0] }),
    update(lt) {
      blocks.forEach((b, i) => {
        const p = ep(lt, tNet - 0.6 + i * 0.12, 0.5, ease.back);
        b.scale.set(Math.max(0.001, p), Math.max(0.001, p), Math.max(0.001, p));
        b.material.emissiveIntensity = 0.35 + 0.35 * Math.max(0, Math.sin(lt * 3 - i * 0.7));
      });
      skips.forEach((s, i) => s.userData.grow(ep(lt, tNet + 0.4 + i * 0.15, 0.6)));
      pulse.userData.tick(lt, 0.4, ep(lt, tNet + 0.5, 0.5));
      const tt = lt * 1.3;
      inSp.userData.update((c, r) => cell(tt + c * 0.09, r, RO));
      const sep = ep(lt, tVoice - 0.3, 1.2);
      const warn = ep(lt, tNobody - 0.2, 0.6);
      outSp.userData.update((c, r) => {
        const tau = tt + c * 0.09 - 1.2;
        const { v, m } = cell(tau, r, RO);
        const sp = speaking(tau);
        const resid = m * (sp > 0.5 ? 0.03 : 0.13 * (1 + warn * 0.9 * (0.6 + 0.4 * Math.sin(lt * 6))));
        return { v: v * sep, m: lerp(m, resid, sep) * sep };
      });
      L.net.at(camera, V(0, 3.45, 0), ep(lt, tNet, 0.5));
      L.inp.at(camera, V(-7.2, 1.2, 0), ep(lt, 0.6, 0.5));
      L.out.at(camera, V(7.2, 1.2, 0), sep);
      L.warn.at(camera, V(7.2, -2.3, 0.5), warn);
      show(head, ep(lt, 0.2, 0.6));
      show(st[0], ep(lt, tForty, 0.5));
      show(st[1], ep(lt, tDb, 0.5));
      show(st[2], ep(lt, tNobody - 1.6, 0.5));
    },
  };
}

// ---------- 6. BandIt Plus ----------
export function bandit({ group, root, cue, camera }) {
  const NB = 64;
  const bandG = new THREE.Group();
  bandG.position.set(-6.0, 0.2, 0);
  bandG.rotation.y = 0.45;
  group.add(bandG);
  const bands = [];
  const cA = new THREE.Color(C.amber), cE = new THREE.Color(C.emerald), cV = new THREE.Color(C.violet);
  for (let i = 0; i < NB; i++) {
    const u = i / (NB - 1);
    const col = u < 0.5 ? cE.clone().lerp(cA, u * 2) : cA.clone().lerp(cV, (u - 0.5) * 2);
    const m = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.035, 1.4), glow(col, 0.85));
    bandG.add(m);
    bands.push(m);
  }
  const outs = [
    { name: 'speech', c: C.emerald, y: 2.0 },
    { name: 'music', c: C.amber, y: 0.0 },
    { name: 'effects', c: C.violet, y: -2.0 },
  ].map((o) => {
    const s = spectro(30, 12, 0.1, 0.1, 0.8);
    s.position.set(0.0, o.y - 0.3, 0);
    s.rotation.set(0.25, -0.3, 0);
    group.add(s);
    const t = growTube([V(-4.2, 0.2, 0), V(-2.9, lerp(0.2, o.y, 0.6), 0.2), V(-1.7, o.y, 0)], 0.03, o.c, 0.85);
    group.add(t);
    const f = flow(t.userData.curve, 4, o.c, 0.06);
    group.add(f);
    return { ...o, s, t, f, lab: label3d(root, `<div class="t" style="color:${hex(o.c)}">${o.name}</div>`) };
  });
  const lBands = label3d(root, '<div class="t">64 bands</div><div class="d">overlapping, merged by one matrix</div>');
  const head = place(el('div', '', '', root), 110, 120, 1000);
  head.innerHTML = `<div class="kicker" style="color:${hex(C.amber)}">Audio · step 2 · the dialogue method</div><div class="h2" style="margin-top:10px">BandIt Plus</div>`;
  const chart = place(el('div', 'panel', '', root), 1250, 250, 600);
  chart.innerHTML = `<div class="kicker" style="color:${hex(C.gold)}">Music left where nobody speaks</div>
   <div class="col" style="margin-top:20px;gap:22px">
    <div><div style="font-size:23px;margin-bottom:8px">MDX-Net alone <span class="muted" style="float:right">−19 to −28 dB</span></div>
      <div style="height:26px;background:rgba(148,163,184,.12);border-radius:8px"><div class="b1" style="height:100%;width:0;background:${hex(C.amber)};border-radius:8px"></div></div></div>
    <div><div style="font-size:23px;margin-bottom:8px">MDX-Net → BandIt <span class="em" style="float:right">−66 to −96 dB</span></div>
      <div style="height:26px;background:rgba(148,163,184,.12);border-radius:8px"><div class="b2" style="height:100%;width:0;background:${hex(C.emerald)};border-radius:8px;box-shadow:0 0 18px ${hex(C.emerald)}"></div></div></div>
    <div class="muted" style="font-size:19px">bar = how far the music is pushed down</div>
   </div>`;
  const b1 = chart.querySelector('.b1'), b2 = chart.querySelector('.b2');
  const chips = place(el('div', 'col', '', root), 1250, 600, 600);
  chips.style.gap = '12px';
  const cs = [
    ['Rebuilt for ONNX: matches the original to 140 dB', C.sky],
    ['Own thread: overlaps with MDX-Net', C.emerald],
    ['fp16 weights · GRU layers kept fp32', C.violet],
  ].map(([t, c]) => {
    const e = el('span', 'chip', `<i></i>${t}`, chips);
    e.querySelector('i').style.color = hex(c);
    e.style.alignSelf = 'flex-start';
    return e;
  });
  const tSplit = cue('splits'), tBands = cue('sixty-four', 1, cue('sixty')), tRebuilt = cue('rebuilt'), tChain = cue('Chained');
  const tLeft = cue('leftover'), tHalf = cue('half'), tSpeech = cue('speech');

  return {
    cam: (lt) => ({ p: [Math.sin(lt * 0.12) * 0.6, 1.4, 15.5], l: [0, 0, 0] }),
    update(lt) {
      const spread = ep(lt, tBands - 0.3, 1.2, ease.inOut);
      bands.forEach((b, i) => {
        const u = i / (NB - 1);
        const gap = lerp(0.04, 0.075, spread);
        b.position.set(Math.sin(lt * 1.5 + i * 0.3) * 0.04 * spread, (i - NB / 2) * gap, 0);
        const k = cell(lt * 1.3, i * 0.6, 40);
        b.scale.x = 0.25 + 0.75 * clamp01((k.v + k.m) * 1.2);
        b.material.userData.o = 0.85 * ep(lt, 0.3 + u * 0.8, 0.4);
      });
      const pOut = ep(lt, tSpeech - 0.3, 0.8);
      const dim = ep(lt, tLeft - 0.3, 1.0);
      outs.forEach((o, i) => {
        o.t.userData.grow(ep(lt, tSpeech - 0.3 + i * 0.25, 0.6));
        o.f.userData.tick(lt, 0.35, ep(lt, tSpeech + i * 0.25, 0.5) * (i === 0 ? 1 : 1 - dim * 0.8));
        const g = (i === 0 ? 1 : 1 - dim * 0.88) * ep(lt, tSpeech + i * 0.25, 0.6);
        o.s.userData.update((c, r) => {
          const tau = lt * 1.3 + c * 0.09;
          const k = cell(tau, r * 2, 24);
          if (i === 0) return { v: k.v, m: 0 };
          if (i === 1) return { v: 0, m: k.m * g };
          const click = ((tau * 0.7) % 1.6 < 0.07 ? 0.9 : 0) * Math.exp(-r * 0.08);
          return { v: 0, m: click * g + 0.02 };
        }, i === 0 ? ep(lt, tSpeech, 0.6) : 1);
        o.lab.at(camera, V(-1.15, o.y + 0.62, 0), ep(lt, tSpeech + i * 0.25, 0.5) * (i === 0 ? 1 : 1 - dim * 0.5));
      });
      lBands.at(camera, V(-6.0, -3.0, 0), ep(lt, tBands, 0.6));
      show(head, ep(lt, 0.2, 0.6));
      show(chart, ep(lt, tChain - 0.2, 0.6));
      b1.style.width = (29 * ep(lt, tChain + 0.4, 0.8)).toFixed(1) + '%';
      b2.style.width = (100 * ep(lt, tLeft + 1.2, 1.4)).toFixed(1) + '%';
      show(cs[0], ep(lt, tRebuilt, 0.5));
      show(cs[1], ep(lt, tChain + 0.3, 0.5));
      show(cs[2], ep(lt, tHalf, 0.5));
    },
  };
}

// ---------- 7. detection ----------
export function detect({ group, root, cue, camera }) {
  const deck = new THREE.Group();
  deck.position.set(-3.6, 0.3, 0);
  deck.rotation.y = 0.32;
  group.add(deck);
  const frames = [];
  for (let i = 0; i < 8; i++) {
    const ppl = STREET.map((p, j) => ({ ...p, x: p.x + (j % 2 ? -1 : 1) * i * 0.012 }));
    const tex = canvasTex(640, 360, (c, w, h) => drawStreet(c, w, h, ppl, 7));
    const det = canvasTex(640, 360, (c, w, h) => drawDetections(c, w, h, ppl));
    const f = framed(3.2, 1.8, tex, C.sky);
    const d = plane(3.2, 1.8, det, 0);
    d.position.z = 0.01;
    f.add(d);
    deck.add(f);
    frames.push({ f, d });
  }
  const bracket = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(3.6 + 7 * 0.5, 2.3, 7 * 0.6 + 0.4)),
    new THREE.LineBasicMaterial({ color: C.emerald, transparent: true, opacity: 0.9 }),
  );
  bracket.position.set(1.75, 0, -2.1);
  deck.add(bracket);
  const lB = label3d(root, `<div class="tag" style="color:${hex(C.emerald)};border-color:${hex(C.emerald)}">batch of 8 frames</div>`);
  const head = place(el('div', '', '', root), 110, 120, 1000);
  head.innerHTML = `<div class="kicker" style="color:${hex(C.sky)}">Picture · pass 1 · find people</div><div class="h2" style="margin-top:10px">YOLO26s-seg <span class="muted">fp16</span></div>`;
  const stats = place(el('div', 'col', '', root), 1330, 240, 480);
  stats.style.gap = '18px';
  const st = [
    ['640×384', 'small frames, constant rate', C.white],
    ['outline', 'per person, not just a box', C.sky],
    ['8', 'frames per batch', C.emerald],
    ['298 fps', 'no NMS step needed', C.gold],
  ].map(([a, b, c]) => el('div', 'panel', `<div class="stat" style="color:${hex(c)};font-size:52px">${a}</div><div class="statlbl">${b}</div>`, stats));
  st.forEach((e) => (e.style.padding = '16px 24px'));
  const tDec = cue('decodes'), tSeg = cue('segmentation'), tOut = cue('outline'), tBatch = cue('batches'), tFps = cue('hundred');

  return {
    cam: (lt) => ({ p: [-0.8 + lt * 0.05, 1.5, 13.5 - lt * 0.06], l: [-1.0, 0, -2] }),
    update(lt) {
      frames.forEach(({ f, d }, i) => {
        const p = ep(lt, tDec - 0.4 + i * 0.12, 0.7, ease.out);
        f.position.set(lerp(9, 0, p) + i * 0.5, 0, -i * 0.6);
        f.rotation.y = lerp(-0.8, 0, p);
        setOp(f, p);
        d.material.userData.o = ep(lt, tSeg + i * 0.1, 0.5);
      });
      bracket.material.userData.o = 0.9 * ep(lt, tBatch - 0.2, 0.5);
      deck.updateMatrixWorld(true);
      lB.at(camera, deck.localToWorld(V(1.75, 1.55, 0.1)), ep(lt, tBatch - 0.2, 0.5));
      show(head, ep(lt, 0.2, 0.6));
      show(st[0], ep(lt, tDec, 0.5), 0, 30);
      show(st[1], ep(lt, tOut, 0.5), 0, 30);
      show(st[2], ep(lt, tBatch, 0.5), 0, 30);
      show(st[3], ep(lt, tFps, 0.5), 0, 30);
    },
  };
}
