// Shared helpers for the explainer scenes: easing, seeded randomness, DOM
// builders, 3D builders and the synthetic signals the visuals are drawn from.
// Everything is a pure function of time so any frame can be rendered alone.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

export const W = 1920;
export const H = 1080;

export const C = {
  bg: 0x070b14,
  emerald: 0x34d399,
  amber: 0xf59e0b,
  gold: 0xfbbf24,
  sky: 0x38bdf8,
  rose: 0xfb7185,
  violet: 0xa78bfa,
  slate: 0x334155,
  white: 0xe2e8f0,
  red: 0xef4444,
};
export const hex = (c) => '#' + c.toString(16).padStart(6, '0');

// ---------- maths ----------
export const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
export const lerp = (a, b, t) => a + (b - a) * t;
export const prog = (t, t0, d) => clamp01((t - t0) / d);
export const ease = {
  out: (t) => 1 - Math.pow(1 - clamp01(t), 3),
  inOut: (t) => {
    t = clamp01(t);
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  },
  back: (t) => {
    t = clamp01(t);
    const c1 = 1.70158, c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
};
/** Eased 0..1 progress of an animation starting at t0 lasting d seconds. */
export const ep = (t, t0, d = 0.6, f = ease.out) => f(prog(t, t0, d));
export const smooth = (a, b, x) => {
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** Smooth value noise in 1D, deterministic. */
export function noise1(x, seed = 1) {
  const i = Math.floor(x), f = x - i;
  const h = (n) => {
    const s = Math.sin((n + seed * 101.3) * 127.1) * 43758.5453;
    return s - Math.floor(s);
  };
  const u = f * f * (3 - 2 * f);
  return lerp(h(i), h(i + 1), u);
}

// ---------- DOM ----------
export function el(tag, cls, html, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined && html !== null) e.innerHTML = html;
  if (parent) parent.appendChild(e);
  return e;
}
export function place(e, x, y, w) {
  e.style.left = x + 'px';
  e.style.top = y + 'px';
  if (w) e.style.width = w + 'px';
  return e;
}
/** Shows an element by progress p: fade and slide up. */
export function show(e, p, dy = 18, dx = 0) {
  e.style.opacity = p.toFixed(3);
  e.style.transform = `translate(${((1 - p) * dx).toFixed(1)}px, ${((1 - p) * dy).toFixed(1)}px)`;
  e.style.visibility = p <= 0.001 ? 'hidden' : 'visible';
}
const _v = new THREE.Vector3();
export function project(cam, v) {
  _v.copy(v).project(cam);
  return { x: (_v.x + 1) * 0.5 * W, y: (1 - _v.y) * 0.5 * H, z: _v.z };
}
/** A label that follows a 3D point; anchored at its centre (or bottom with 'up'). */
export function label3d(parent, html, cls = '') {
  const e = el('div', 'l3d ' + cls, html, parent);
  return {
    el: e,
    at(cam, v, p = 1, dy = 0) {
      const s = project(cam, v);
      e.style.left = s.x.toFixed(1) + 'px';
      e.style.top = (s.y + dy).toFixed(1) + 'px';
      e.style.opacity = p.toFixed(3);
      e.style.visibility = p <= 0.001 || s.z > 1 ? 'hidden' : 'visible';
    },
  };
}

// ---------- 3D ----------
export function setOp(obj, o) {
  obj.traverse((m) => {
    if (!m.material) return;
    for (const mat of Array.isArray(m.material) ? m.material : [m.material]) mat.userData.o = o;
  });
}
export function glow(color, o = 1) {
  const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: o, depthWrite: o >= 1 });
  return m;
}
export function std(color, emissive = 0x000000, ei = 0.6) {
  return new THREE.MeshStandardMaterial({ color, emissive, emissiveIntensity: ei, roughness: 0.38, metalness: 0.25, transparent: true });
}
export function rbox(w, h, d, r, mat) {
  return new THREE.Mesh(new RoundedBoxGeometry(w, h, d, 4, r), mat);
}
/** A glowing node: dark rounded slab with a coloured rim. */
export function node(w, h, color) {
  const g = new THREE.Group();
  const body = rbox(w, h, 0.34, 0.12, std(0x0b1222, color, 0.1));
  body.material.roughness = 0.6;
  body.material.metalness = 0.1;
  const rim = rbox(w + 0.08, h + 0.08, 0.2, 0.15, glow(color, 0.9));
  rim.position.z = -0.1;
  g.add(rim, body);
  g.userData = { body, rim, color };
  return g;
}
/** Lights a node: k in 0..1. */
export function lightNode(n, k) {
  n.userData.body.material.emissiveIntensity = 0.06 + 0.16 * k;
  n.userData.rim.material.userData.o = 0.25 + 0.75 * k;
  const s = 1 + 0.04 * k;
  n.scale.set(s, s, s);
}
export function tube(points, radius, color, o = 1, closed = false) {
  const curve = new THREE.CatmullRomCurve3(points, closed, 'centripetal');
  const m = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(24, points.length * 16), radius, 10, closed), glow(color, o));
  m.userData.curve = curve;
  return m;
}
/** A tube that grows along its length: call grow(p). */
export function growTube(points, radius, color, o = 1) {
  const m = tube(points, radius, color, o);
  const total = m.geometry.index.count;
  m.userData.grow = (p) => m.geometry.setDrawRange(0, Math.floor((total * clamp01(p)) / 6) * 6);
  m.userData.grow(0);
  return m;
}
/** Dots flowing along a curve. */
export function flow(curve, n, color, size = 0.06) {
  const g = new THREE.Group();
  const geo = new THREE.SphereGeometry(size, 12, 8);
  const mat = glow(color, 1);
  const dots = [];
  for (let i = 0; i < n; i++) {
    const d = new THREE.Mesh(geo, mat);
    g.add(d);
    dots.push(d);
  }
  g.userData.tick = (t, speed = 0.25, on = 1) => {
    dots.forEach((d, i) => {
      const u = (t * speed + i / n) % 1;
      curve.getPointAt(u, d.position);
      const s = on * Math.sin(Math.PI * u);
      d.scale.setScalar(Math.max(0.001, s));
    });
  };
  return g;
}
export function fatLine(points, color, width = 3, o = 1) {
  const geo = new LineGeometry();
  geo.setPositions(points.flatMap((p) => [p.x, p.y, p.z]));
  const mat = new LineMaterial({ color, linewidth: width, transparent: true, opacity: o, worldUnits: false });
  mat.resolution.set(W, H);
  const l = new Line2(geo, mat);
  l.computeLineDistances();
  return l;
}
export function setLine(line, arr) {
  line.geometry.setPositions(arr);
}
export function canvasTex(w, h, draw, nearest = false) {
  const cv = document.createElement('canvas');
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext('2d');
  draw(ctx, w, h);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  if (nearest) {
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.NearestFilter;
    tex.generateMipmaps = false;
  } else tex.anisotropy = 8;
  tex.userData = { canvas: cv, ctx };
  return tex;
}
export function plane(w, h, tex, o = 1) {
  return new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: o, toneMapped: false }));
}
/** A 16:9 frame: picture plane with a thin glowing border. */
export function framed(w, h, tex, color = C.sky) {
  const g = new THREE.Group();
  const pic = plane(w, h, tex);
  const border = new THREE.Mesh(new THREE.PlaneGeometry(w + 0.07, h + 0.07), glow(color, 0.85));
  border.position.z = -0.01;
  g.add(border, pic);
  g.userData = { pic, border };
  return g;
}

// ---------- synthetic audio ----------
/** Is someone speaking at time tau (seconds)? 0..1, with pauses. */
export function speaking(tau) {
  const s = Math.sin(tau * 0.55) + 0.55 * Math.sin(tau * 1.31 + 1.7);
  return smooth(-0.35, 0.05, s);
}
/** Voice and music energy for a spectrogram cell (bin r of R) at time tau. */
export function cell(tau, r, R) {
  const fr = r / R; // 0 low .. 1 high
  // voice: harmonics of a moving pitch, in syllables
  const f0 = 2.6 + 0.7 * Math.sin(tau * 0.8) + 0.3 * Math.sin(tau * 2.1);
  const syl = 0.35 + 0.65 * Math.pow(Math.abs(Math.sin(tau * 4.2 + Math.sin(tau * 1.3))), 0.7);
  let v = 0;
  for (let k = 1; k <= 9; k++) {
    const b = k * f0;
    const d = r - b;
    v += Math.exp(-k * 0.22) * Math.exp(-(d * d) / 0.5);
  }
  v *= speaking(tau) * syl * 0.95;
  // music: held chord tones that change every 2 s, and a kick/hat pattern
  const bar = Math.floor(tau / 2);
  const roots = [4.0, 5.3, 3.4, 4.6];
  const root = roots[((bar % 4) + 4) % 4];
  let m = 0;
  for (const mul of [1, 1.26, 1.5, 2, 2.52, 3, 4, 5]) {
    const d = r - root * mul * 1.6;
    m += 0.45 * Math.exp(-(d * d) / 0.35) * (0.7 + 0.3 * Math.sin(tau * 3 + mul));
  }
  const beat = tau % 0.5;
  const kick = Math.exp(-beat * 18) * Math.exp(-fr * 7) * 0.9;
  const hat = (tau + 0.25) % 0.5 < 0.06 ? 0.35 * smooth(0.55, 0.9, fr) : 0;
  m += kick + hat + 0.05 * noise1(tau * 9 + r * 0.7, 3);
  return { v: Math.min(1.2, v), m: Math.min(1.2, m) };
}

const _c = new THREE.Color();
const _a = new THREE.Color(C.amber);
const _e = new THREE.Color(C.emerald);
const _dark = new THREE.Color(0x0b1220);
/**
 * An instanced spectrogram: x = time, z = frequency, y = energy.
 * update(fn) where fn(col,row) returns {v, m} energies to show.
 */
export function spectro(cols, rows, cw, cd, maxH) {
  const geo = new THREE.BoxGeometry(1, 1, 1);
  geo.translate(0, 0.5, 0);
  const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true });
  const mesh = new THREE.InstancedMesh(geo, mat, cols * rows);
  const m4 = new THREE.Matrix4();
  for (let i = 0; i < cols * rows; i++) mesh.setColorAt(i, _dark);
  mesh.userData.update = (fn, gain = 1) => {
    let i = 0;
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++, i++) {
        const { v, m } = fn(c, r);
        const e = Math.max(0, v + m);
        const h = Math.max(0.006, Math.min(1.25, e) * maxH * gain);
        m4.makeScale(cw * 0.82, h, cd * 0.82);
        m4.setPosition((c - (cols - 1) / 2) * cw, 0, ((rows - 1) / 2 - r) * cd);
        mesh.setMatrixAt(i, m4);
        const mix = e > 1e-4 ? v / e : 0.5;
        _c.copy(_a).lerp(_e, mix);
        const b = clamp01(0.08 + e * 1.15);
        _c.lerp(_dark, 1 - b);
        mesh.setColorAt(i, _c);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
  };
  mesh.frustumCulled = false;
  return mesh;
}

// ---------- drawn pictures ----------
/** People in a stylised street; kind 'w' (woman) or 'm' (man). */
export const STREET = [
  { x: 0.2, y: 0.9, s: 1.0, kind: 'm', col: '#3b6e8f' },
  { x: 0.38, y: 0.86, s: 0.85, kind: 'w', col: '#7c4a6e' },
  { x: 0.6, y: 0.92, s: 1.05, kind: 'w', col: '#4f6b4a' },
  { x: 0.8, y: 0.84, s: 0.75, kind: 'm', col: '#8a6a3b' },
];
function personPath(ctx, p, w, h, grow = 0) {
  const u = h * 0.34 * p.s; // person height unit
  const cx = p.x * w, by = p.y * h;
  const g = grow * u;
  ctx.beginPath();
  if (p.kind === 'w') {
    // head with a draped scarf and a long dress
    ctx.ellipse(cx, by - u * 0.93, u * 0.15 + g, u * 0.17 + g, 0, 0, Math.PI * 2);
    ctx.moveTo(cx - u * 0.2 - g, by - u * 0.8);
    ctx.quadraticCurveTo(cx - u * 0.36 - g, by - u * 0.45, cx - u * 0.33 - g, by + g * 0.2);
    ctx.lineTo(cx + u * 0.33 + g, by + g * 0.2);
    ctx.quadraticCurveTo(cx + u * 0.36 + g, by - u * 0.45, cx + u * 0.2 + g, by - u * 0.8);
    ctx.closePath();
  } else {
    ctx.ellipse(cx, by - u * 0.93, u * 0.13 + g, u * 0.14 + g, 0, 0, Math.PI * 2);
    ctx.moveTo(cx - u * 0.22 - g, by - u * 0.74);
    ctx.lineTo(cx + u * 0.22 + g, by - u * 0.74);
    ctx.lineTo(cx + u * 0.19 + g, by - u * 0.36);
    ctx.lineTo(cx + u * 0.14 + g, by + g * 0.2);
    ctx.lineTo(cx - u * 0.14 - g, by + g * 0.2);
    ctx.lineTo(cx - u * 0.19 - g, by - u * 0.36);
    ctx.closePath();
  }
}
export function drawStreet(ctx, w, h, people = STREET, seed = 7) {
  const r = rng(seed);
  const sky = ctx.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, '#1b2a4a');
  sky.addColorStop(0.6, '#2c3f63');
  sky.addColorStop(1, '#3a4c6b');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);
  // buildings
  let x = 0;
  while (x < w) {
    const bw = w * (0.08 + r() * 0.12), bh = h * (0.3 + r() * 0.4);
    ctx.fillStyle = ['#18233b', '#1e2b45', '#222f4c'][Math.floor(r() * 3)];
    ctx.fillRect(x, h * 0.8 - bh, bw, bh);
    for (let wy = h * 0.8 - bh + 10; wy < h * 0.76; wy += h * 0.06)
      for (let wx = x + 8; wx < x + bw - 12; wx += w * 0.025)
        if (r() > 0.45) {
          ctx.fillStyle = r() > 0.5 ? 'rgba(251,191,36,0.55)' : 'rgba(148,163,184,0.25)';
          ctx.fillRect(wx, wy, w * 0.012, h * 0.025);
        }
    x += bw + 4;
  }
  ctx.fillStyle = '#273449';
  ctx.fillRect(0, h * 0.8, w, h * 0.2);
  ctx.fillStyle = 'rgba(148,163,184,0.18)';
  ctx.fillRect(0, h * 0.8, w, 3);
  for (const p of people) {
    personPath(ctx, p, w, h);
    ctx.fillStyle = p.col;
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    ctx.fill();
  }
}
/** Detection overlay: outlines (cyan) and boxes. */
export function drawDetections(ctx, w, h, people = STREET) {
  ctx.clearRect(0, 0, w, h);
  for (const p of people) {
    personPath(ctx, p, w, h, 0.02);
    ctx.fillStyle = 'rgba(56,189,248,0.22)';
    ctx.fill();
    ctx.lineWidth = Math.max(2, w / 300);
    ctx.strokeStyle = '#7dd3fc';
    ctx.shadowColor = '#38bdf8';
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;
    const u = h * 0.34 * p.s;
    ctx.setLineDash([8, 6]);
    ctx.strokeStyle = 'rgba(226,232,240,0.7)';
    ctx.lineWidth = Math.max(1.5, w / 500);
    ctx.strokeRect(p.x * w - u * 0.4, p.y * h - u * 1.15, u * 0.8, u * 1.18);
    ctx.setLineDash([]);
  }
}
/** Mask: white where the chosen people are. */
export function drawMask(ctx, w, h, people = STREET, kinds = ['w']) {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#fff';
  for (const p of people) {
    if (!kinds.includes(p.kind)) continue;
    personPath(ctx, p, w, h, 0.06);
    ctx.fill();
  }
}
/** The finished picture: the street with women blurred inside their outline. */
export function drawBlurred(ctx, w, h, people = STREET, px = 18) {
  const src = document.createElement('canvas');
  src.width = w;
  src.height = h;
  drawStreet(src.getContext('2d'), w, h, people);
  ctx.drawImage(src, 0, 0);
  ctx.save();
  ctx.beginPath();
  for (const p of people) if (p.kind === 'w') personPath(ctx, p, w, h, 0.06);
  ctx.clip();
  ctx.filter = `blur(${px}px)`;
  ctx.drawImage(src, 0, 0);
  ctx.filter = 'none';
  ctx.fillStyle = 'rgba(148,163,184,0.12)';
  ctx.fillRect(0, 0, w, h);
  ctx.restore();
}
export function drawPortrait(ctx, w, h, kind = 'w') {
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, '#24334f');
  g.addColorStop(1, '#3a4a68');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  const p = { x: 0.5, y: 1.35, s: 2.6, kind, col: kind === 'w' ? '#7c4a6e' : '#3b6e8f' };
  personPath(ctx, p, w, h);
  ctx.fillStyle = p.col;
  ctx.fill();
  ctx.fillStyle = 'rgba(255,255,255,0.08)';
  ctx.fill();
}
