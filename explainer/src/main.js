// The engine: one three.js scene, an HTML layer, and seek(t), which sets every
// object for time t and renders. Nothing depends on wall-clock time, so the
// renderer can ask for frames in any order.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import TL from '../build/timeline.js';
import { W, H, C, ease, clamp01, lerp, prog, rng, el, canvasTex } from './kit.js';
import { MAKERS, CHAPTERS } from './scenes.js';

const canvas = document.getElementById('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
scene.background = new THREE.Color(C.bg);
scene.fog = new THREE.Fog(C.bg, 22, 70);
const camera = new THREE.PerspectiveCamera(36, W / H, 0.1, 400);

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(4, 8, 10);
scene.add(key);
const rimL = new THREE.PointLight(C.emerald, 40, 40);
rimL.position.set(-8, 4, 6);
const rimR = new THREE.PointLight(C.sky, 40, 40);
rimR.position.set(8, -2, 6);
scene.add(rimL, rimR);

// ---------- persistent background ----------
const bg = new THREE.Group();
scene.add(bg);
{
  const r = rng(42);
  const n = 2400;
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  const a = new THREE.Color(C.emerald), b = new THREE.Color(C.sky), w = new THREE.Color(0x94a3b8);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = (r() - 0.5) * 120;
    pos[i * 3 + 1] = (r() - 0.5) * 60;
    pos[i * 3 + 2] = -10 - r() * 70;
    const c = r() < 0.5 ? w : r() < 0.5 ? a : b;
    col.set([c.r, c.g, c.b], i * 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const stars = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.14, vertexColors: true, transparent: true, opacity: 0.55, depthWrite: false }));
  stars.name = 'stars';
  bg.add(stars);

  const grid = new THREE.GridHelper(240, 120, 0x1e3a5f, 0x14213a);
  grid.material.transparent = true;
  grid.material.opacity = 0.45;
  grid.position.y = -6.5;
  grid.name = 'grid';
  bg.add(grid);

  const orb = (color, x, y, z, s) => {
    const tex = canvasTex(256, 256, (ctx) => {
      const gr = ctx.createRadialGradient(128, 128, 0, 128, 128, 128);
      gr.addColorStop(0, 'rgba(255,255,255,0.9)');
      gr.addColorStop(0.4, 'rgba(255,255,255,0.25)');
      gr.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = gr;
      ctx.fillRect(0, 0, 256, 256);
    });
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, color, transparent: true, opacity: 0.16, depthWrite: false, fog: false }));
    sp.position.set(x, y, z);
    sp.scale.set(s, s, 1);
    bg.add(sp);
    return sp;
  };
  bg.userData.orbs = [orb(C.emerald, -26, 8, -45, 50), orb(C.sky, 28, -6, -50, 56), orb(C.violet, 0, 18, -60, 40)];
}
function updateBackground(t) {
  const stars = bg.getObjectByName('stars');
  stars.rotation.y = t * 0.006;
  stars.rotation.x = Math.sin(t * 0.05) * 0.02;
  const grid = bg.getObjectByName('grid');
  grid.position.z = (t * 0.6) % 2;
  bg.userData.orbs.forEach((o, i) => {
    o.position.x += Math.sin(t * 0.07 + i * 2) * 0.0;
    o.material.opacity = 0.12 + 0.05 * Math.sin(t * 0.3 + i * 1.7);
  });
}

// ---------- post ----------
const target = new THREE.WebGLRenderTarget(W, H, { samples: 4, type: THREE.HalfFloatType });
const composer = new EffectComposer(renderer, target);
composer.setPixelRatio(1);
composer.setSize(W, H);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(W, H), 0.55, 0.45, 0.62);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// ---------- scenes ----------
const ui = document.getElementById('ui');
const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, '');
window.__cueMisses = [];
const SC = TL.scenes.map((s, index) => {
  const root = el('div', 'scene', '', ui);
  root.dataset.id = s.id;
  const group = new THREE.Group();
  scene.add(group);
  const cue = (word, n = 1, fallback = null) => {
    const k = norm(word);
    let seen = 0;
    for (const w of s.words) if (norm(w.w) === k && ++seen === n) return w.t0 - s.start;
    seen = 0;
    for (const w of s.words) if (norm(w.w).startsWith(k) && ++seen === n) return w.t0 - s.start;
    window.__cueMisses.push(`${s.id}:${word}`);
    return fallback ?? s.voice - s.start + 0.5;
  };
  const api = MAKERS[s.id]({ group, root, cue, data: s, camera, index, total: TL.scenes.length });
  group.traverse((m) => {
    if (!m.material) return;
    for (const mat of Array.isArray(m.material) ? m.material : [m.material]) {
      if (mat.userData.base === undefined) mat.userData.base = mat.opacity;
      mat.transparent = true;
    }
  });
  return { ...s, index, root, group, api };
});

// HUD
const bar = document.getElementById('bar');
const fills = SC.map((s) => {
  const i = el('i', '', '<b></b>', bar);
  i.style.flexGrow = s.dur;
  return i.firstChild;
});
const chn = document.getElementById('chn'), cht = document.getElementById('cht');
const capEl = document.querySelector('#cap span');
const CAPS = SC.flatMap((s) => s.captions);

function applyOpacity(group, f) {
  group.traverse((m) => {
    if (!m.material) return;
    for (const mat of Array.isArray(m.material) ? m.material : [m.material]) {
      if (mat.userData.base === undefined) {
        mat.userData.base = mat.opacity;
        mat.transparent = true;
      }
      const o = mat.userData.o ?? mat.userData.base;
      mat.opacity = o * f;
    }
  });
}

const BLEND = 0.75;
const _p = new THREE.Vector3(), _l = new THREE.Vector3(), _p2 = new THREE.Vector3(), _l2 = new THREE.Vector3();
function poseAt(s, lt, outP, outL) {
  const c = s.api.cam(Math.max(0, Math.min(lt, s.dur)));
  outP.set(...c.p);
  outL.set(...c.l);
  return c.fov ?? 36;
}
function setCamera(t, i) {
  const s = SC[i];
  const lt = t - s.start;
  let fov = poseAt(s, lt, _p, _l);
  let other = null, k = 0;
  if (i > 0 && lt < BLEND) {
    other = SC[i - 1];
    k = 1 - ease.inOut(0.5 + lt / (2 * BLEND));
  } else if (i < SC.length - 1 && lt > s.dur - BLEND) {
    other = SC[i + 1];
    k = ease.inOut((lt - (s.dur - BLEND)) / (2 * BLEND));
  }
  if (other && k > 0) {
    const fov2 = poseAt(other, t - other.start, _p2, _l2);
    _p.lerp(_p2, k);
    _l.lerp(_l2, k);
    fov = lerp(fov, fov2, k);
  }
  camera.position.copy(_p);
  camera.lookAt(_l);
  if (camera.fov !== fov) {
    camera.fov = fov;
    camera.updateProjectionMatrix();
  }
  camera.updateMatrixWorld();
}

function seek(t) {
  t = Math.max(0, Math.min(t, TL.total));
  let i = SC.findIndex((s) => t >= s.start && t < s.start + s.dur);
  if (i < 0) i = SC.length - 1;
  updateBackground(t);
  setCamera(t, i);
  SC.forEach((s, j) => {
    const on = j === i;
    s.group.visible = on;
    s.root.style.display = on ? 'block' : 'none';
    if (!on) return;
    const lt = t - s.start;
    const last = j === SC.length - 1;
    const fin = i === 0 ? ease.out(prog(lt, 0.0, 0.9)) : ease.out(prog(lt, 0, 0.6));
    const fout = last ? 1 : 1 - ease.inOut(prog(lt, s.dur - 0.5, 0.5));
    const f = Math.min(fin, fout);
    s.api.update(lt, t);
    applyOpacity(s.group, f);
    s.root.style.opacity = f.toFixed(3);
  });
  // HUD
  SC.forEach((s, j) => {
    fills[j].style.width = (clamp01((t - s.start) / s.dur) * 100).toFixed(2) + '%';
  });
  const ch = CHAPTERS[SC[i].id];
  chn.textContent = String(i + 1).padStart(2, '0');
  cht.textContent = ch;
  document.getElementById('chapter').style.opacity = i === 0 ? clamp01((t - 3) / 1) : 1;
  // captions
  const c = CAPS.find((c) => t >= c.t0 - 0.08 && t < c.t1 + 0.3);
  if (c) {
    if (capEl.textContent !== c.text) capEl.textContent = c.text;
    const a = Math.min(prog(t, c.t0 - 0.08, 0.12), 1 - prog(t, c.t1 + 0.15, 0.15));
    capEl.parentElement.style.opacity = a.toFixed(3);
  } else capEl.parentElement.style.opacity = 0;
  composer.render();
}

window.TOTAL = TL.total;
window.seek = seek;

// fit the 1920x1080 stage to the window for previewing
function fit() {
  const s = Math.min(innerWidth / W, innerHeight / H);
  const st = document.getElementById('stage');
  st.style.transform = `scale(${s})`;
  st.style.left = (innerWidth - W * s) / 2 + 'px';
  st.style.top = (innerHeight - H * s) / 2 + 'px';
}
const params = new URLSearchParams(location.search);
if (!params.has('render')) {
  addEventListener('resize', fit);
  fit();
}

window.ready = document.fonts.ready.then(() => {
  seek(Number(params.get('t') || 0));
  if (params.has('play')) {
    const audio = new Audio('./build/narration.wav');
    const t0 = Number(params.get('t') || 0);
    audio.currentTime = t0;
    const go = () => {
      audio.play();
      const loop = () => {
        seek(audio.currentTime);
        if (!audio.ended) requestAnimationFrame(loop);
      };
      loop();
    };
    addEventListener('click', go, { once: true });
  }
  return true;
});
