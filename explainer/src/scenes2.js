// Scenes 8-14: tracking, SigLIP 2, decision, render, runtime, results, outro.
import * as THREE from 'three';
import {
  C, hex, ep, ease, prog, lerp, clamp01, smooth, noise1, rng, el, place, show, label3d, setOp,
  glow, std, rbox, node, lightNode, tube, growTube, flow, fatLine, setLine, canvasTex, plane, framed,
  STREET, drawStreet, drawMask, drawBlurred, drawPortrait,
} from './kit.js';

const V = (x, y, z = 0) => new THREE.Vector3(x, y, z);

// ---------- 8. tracking ----------
export function track({ group, root, cue, camera }) {
  const X0 = -6.5, X1 = 6.5, CUT = 2.0;
  const nf = 16;
  for (let i = 0; i < nf; i++) {
    const x = lerp(X0, X1, i / (nf - 1));
    const p = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 3.2), glow(C.sky, 0.05));
    p.rotation.y = Math.PI / 2;
    p.position.x = x;
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(2.6, 3.2)), new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.8 }));
    e.rotation.y = Math.PI / 2;
    e.position.x = x;
    group.add(p, e);
  }
  const cutPlane = new THREE.Mesh(new THREE.PlaneGeometry(3.0, 3.6), glow(C.red, 0.3));
  cutPlane.rotation.y = Math.PI / 2;
  cutPlane.position.x = CUT;
  group.add(cutPlane);
  const path = (xa, xb, f) => {
    const pts = [];
    for (let i = 0; i <= 24; i++) {
      const x = lerp(xa, xb, i / 24);
      pts.push(V(x, ...f(x)));
    }
    return pts;
  };
  const T = [
    { id: 1, c: C.emerald, xa: X0, xb: CUT - 0.05, f: (x) => [0.8 + 0.25 * Math.sin(x * 0.7), 0.5 * Math.cos(x * 0.5)] },
    { id: 2, c: C.rose, xa: X0, xb: -1.8, f: (x) => [-0.6 + 0.3 * Math.sin(x * 0.6 + 1), -0.4 + 0.3 * Math.sin(x * 0.4)] },
    { id: 2, c: C.rose, xa: -0.4, xb: CUT - 0.05, f: (x) => [-0.6 + 0.3 * Math.sin(x * 0.6 + 1), -0.4 + 0.3 * Math.sin(x * 0.4)] },
    { id: 3, c: C.sky, xa: CUT + 0.05, xb: X1, f: (x) => [0.3 + 0.3 * Math.sin(x * 0.8), 0.6 * Math.sin(x * 0.3)] },
    { id: 4, c: C.gold, xa: CUT + 0.05, xb: X1, f: (x) => [-0.9 + 0.2 * Math.sin(x), -0.5] },
  ].map((tr) => {
    const m = growTube(path(tr.xa, tr.xb, tr.f), 0.07, tr.c, 1);
    group.add(m);
    return { ...tr, m };
  });
  const gap = [];
  const gf = T[1].f;
  for (let i = 1; i < 8; i++) {
    const x = lerp(-1.8, -0.4, i / 8);
    const d = new THREE.Mesh(new THREE.SphereGeometry(0.06, 10, 8), glow(C.rose, 1));
    d.position.set(x, ...gf(x));
    group.add(d);
    gap.push(d);
  }
  const labs = T.filter((_, i) => i !== 2).map((tr) => ({ tr, l: label3d(root, `<div class="tag" style="color:${hex(tr.c)};border-color:${hex(tr.c)}">#${tr.id}</div>`) }));
  const lCut = label3d(root, `<div class="t" style="color:${hex(C.red)}">scene cut</div><div class="d">every track ends here</div>`);
  const lGap = label3d(root, `<div class="d" style="color:#fda4af">lost for a moment → interpolated</div>`);
  const lTime = label3d(root, '<div class="d">time →</div>');
  const head = place(el('div', '', '', root), 110, 120, 1000);
  head.innerHTML = `<div class="kicker" style="color:${hex(C.sky)}">Picture · pass 1 · follow people</div><div class="h2" style="margin-top:10px">One track per person</div>`;
  const tTr = cue('tracker'), tCut = cue('cuts'), tInt = cue('interpolation');

  return {
    cam: (lt) => ({ p: [lerp(-3.5, 1.2, ease.inOut(lt / 11)), 3.4, 13.8], l: [0, 0.35, 0] }),
    update(lt) {
      T.forEach((tr) => {
        const g = ep(lt, tTr - 0.3 + (tr.xa - X0) * 0.25, 2.4, (x) => x);
        tr.m.userData.grow(g);
      });
      cutPlane.material.userData.o = 0.32 * ep(lt, tCut - 0.3, 0.5) * (0.75 + 0.25 * Math.sin(lt * 5));
      gap.forEach((d, i) => d.scale.setScalar(Math.max(0.001, ep(lt, tInt - 0.4 + i * 0.08, 0.3, ease.back))));
      labs.forEach(({ tr, l }) => {
        const x = tr.xa + 0.3;
        l.at(camera, V(x, tr.f(x)[0] + 0.45, tr.f(x)[1]), ep(lt, tTr + (tr.xa - X0) * 0.25, 0.5));
      });
      lCut.at(camera, V(CUT, 2.4, 0), ep(lt, tCut - 0.2, 0.5));
      lGap.at(camera, V(-1.1, -1.6, 0), ep(lt, tInt - 0.2, 0.5));
      lTime.at(camera, V(5.5, -2.0, 1.4), ep(lt, tTr, 0.5));
      show(head, ep(lt, 0.2, 0.6));
    },
  };
}

// ---------- 9. SigLIP 2 ----------
export function siglip({ group, root, cue, camera }) {
  const crop = framed(1.7, 2.1, canvasTex(340, 420, (c, w, h) => drawPortrait(c, w, h, 'w')), C.rose);
  crop.position.set(-6.6, 1.0, 0);
  group.add(crop);
  const prism = new THREE.Mesh(new THREE.IcosahedronGeometry(0.95, 0), std(0x1e1b4b, C.violet, 0.55));
  prism.material.flatShading = true;
  prism.position.set(-3.1, 1.0, 0);
  const cage = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.IcosahedronGeometry(1.0, 0)), new THREE.LineBasicMaterial({ color: 0xc4b5fd, transparent: true, opacity: 0.9 }));
  prism.add(cage);
  group.add(prism);
  const t1 = growTube([V(-5.6, 1.0, 0), V(-4.1, 1.0, 0)], 0.03, C.violet);
  const t2 = growTube([V(-2.1, 1.0, 0), V(1.0, 0.75, 0)], 0.03, C.violet);
  group.add(t1, t2);
  const f2 = flow(t2.userData.curve, 4, C.white, 0.06);
  group.add(f2);
  const S = V(3.6, 0.6, 0), R = 2.25;
  const sphere = new THREE.Mesh(new THREE.IcosahedronGeometry(R, 3), new THREE.MeshBasicMaterial({ color: 0x334155, wireframe: true, transparent: true, opacity: 0.35 }));
  sphere.position.copy(S);
  group.add(sphere);
  const arrow = (dir, color, len = R) => {
    const g = new THREE.Group();
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, len - 0.25, 10), glow(color, 1));
    shaft.position.y = (len - 0.25) / 2;
    const tip = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.25, 14), glow(color, 1));
    tip.position.y = len - 0.125;
    g.add(shaft, tip);
    g.position.copy(S);
    g.quaternion.setFromUnitVectors(V(0, 1, 0), dir.clone().normalize());
    group.add(g);
    return g;
  };
  const dW = V(-0.55, 0.75, 0.35).normalize(), dM = V(0.85, 0.2, 0.45).normalize(), dI = V(-0.35, 0.85, 0.4).normalize();
  const aW = arrow(dW, C.rose), aM = arrow(dM, C.sky), aI = arrow(dI, C.white);
  const r = rng(9);
  const cloud = (d, color) =>
    Array.from({ length: 16 }, () => {
      const j = V(r() - 0.5, r() - 0.5, r() - 0.5).multiplyScalar(0.9);
      const p = d.clone().add(j).normalize();
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.07, 10, 8), glow(color, 1));
      group.add(m);
      return { m, p };
    });
  const cW = cloud(dW, C.rose), cM = cloud(dM, C.sky);
  const arc = (a, b, color) => {
    const pts = [];
    for (let i = 0; i <= 20; i++) pts.push(S.clone().add(a.clone().lerp(b, i / 20).normalize().multiplyScalar(R * 0.55)));
    const m = growTube(pts, 0.025, color, 1);
    group.add(m);
    return m;
  };
  const arcW = arc(dI, dW, C.rose), arcM = arc(dI, dM, C.sky);
  const L = {
    crop: label3d(root, '<div class="d">head + upper body</div>'),
    enc: label3d(root, '<div class="t">SigLIP 2</div><div class="d">image encoder · 269 crops/s</div>'),
    w: label3d(root, `<div class="t" style="color:${hex(C.rose)}">"woman"</div>`),
    m: label3d(root, `<div class="t" style="color:${hex(C.sky)}">"man"</div>`),
    i: label3d(root, '<div class="d" style="color:#fff">this person</div>'),
  };
  const head = place(el('div', '', '', root), 110, 120, 1000);
  head.innerHTML = `<div class="kicker" style="color:${hex(C.rose)}">Picture · pass 1 · who is a woman?</div><div class="h2" style="margin-top:10px">Zero-shot, no training</div>`;
  const pairs = [
    ['a woman wearing a hijab', 'a man wearing a turban'],
    ['a woman seen from behind', 'a man seen from behind'],
    ['a woman in a long dress', 'a man in a long robe'],
    ['a photo of a woman', 'a photo of a man'],
  ];
  const pp = place(el('div', 'panel', '', root), 110, 600, 860);
  pp.innerHTML = `<div class="kicker" style="color:#cbd5e1">16 paired descriptions → one mean each</div>` +
    pairs.map(([a, b]) => `<div class="pr" style="display:flex;gap:18px;font-size:22px;margin-top:12px"><span class="ro" style="flex:1">${a}</span><span class="muted">vs</span><span class="sk" style="flex:1">${b}</span></div>`).join('') +
    `<div class="muted" style="font-size:19px;margin-top:12px">… computed once, offline; shipped as JSON</div>`;
  const rows = [...pp.querySelectorAll('.pr')];
  const gauge = place(el('div', 'panel', '', root), 1250, 600, 560);
  gauge.innerHTML = `<div class="kicker" style="color:${hex(C.rose)}">Evidence</div>
    <div style="display:flex;align-items:baseline;gap:14px;margin-top:10px"><div class="stat ro gv">0.00</div><div class="muted" style="font-size:22px">P(woman)</div></div>
    <div style="height:14px;background:rgba(148,163,184,.15);border-radius:7px;margin-top:14px"><div class="gb" style="height:100%;width:0;background:${hex(C.rose)};border-radius:7px"></div></div>`;
  const gv = gauge.querySelector('.gv'), gb = gauge.querySelector('.gb');
  const chips = place(el('div', 'row', '', root), 1250, 800, 600);
  const cs = ['backs', 'hijab', 'niqab', 'face masks'].map((t) => {
    const e = el('span', 'chip', `<i></i>${t}`, chips);
    e.querySelector('i').style.color = hex(C.emerald);
    return e;
  });
  const noFace = el('div', 'muted', 'no visible face needed', chips);
  noFace.style.fontSize = '20px';
  const tCrop = cue('crop'), tEnc = cue('encoder'), tText = cue('text'), tSix = cue('sixteen'), tAvg = cue('average');
  const tSuch = cue('such'), tDiff = cue('difference'), tEv = cue('evidence'), tFace = cue('face'), tOff = cue('offline');

  return {
    cam: (lt) => ({ p: [Math.sin(lt * 0.1) * 1.0 - 0.6, 1.3, 16.5 - lt * 0.05], l: [-0.6, 0.2, 0] }),
    update(lt) {
      const pc = ep(lt, tCrop - 0.3, 0.6, ease.back);
      crop.scale.setScalar(Math.max(0.001, pc));
      crop.rotation.y = 0.25 + Math.sin(lt * 0.6) * 0.06;
      const pe = ep(lt, tEnc - 0.6, 0.6, ease.back);
      prism.scale.setScalar(Math.max(0.001, pe));
      prism.rotation.set(lt * 0.5, lt * 0.7, 0);
      t1.userData.grow(ep(lt, tEnc - 0.6, 0.6));
      t2.userData.grow(ep(lt, tEnc, 0.7));
      f2.userData.tick(lt, 0.4, ep(lt, tEnc + 0.4, 0.5));
      const ps = ep(lt, tEnc + 0.2, 0.8);
      sphere.material.userData.o = 0.35 * ps;
      sphere.rotation.y = lt * 0.08;
      aI.scale.setScalar(Math.max(0.001, ep(lt, tEnc + 0.5, 0.6, ease.back)));
      const pcl = ep(lt, tSix - 0.3, 0.6);
      const conv = ep(lt, tAvg, 1.0, ease.inOut);
      [[cW, dW], [cM, dM]].forEach(([cl, d]) => {
        cl.forEach(({ m, p }, i) => {
          const q = p.clone().lerp(d, conv).normalize().multiplyScalar(R);
          m.position.copy(S).add(q);
          m.scale.setScalar(Math.max(0.001, ep(lt, tSix - 0.3 + i * 0.03, 0.4) * (1 - conv * 0.6)));
        });
      });
      aW.scale.setScalar(Math.max(0.001, ep(lt, tAvg + 0.6, 0.5, ease.back)));
      aM.scale.setScalar(Math.max(0.001, ep(lt, tAvg + 0.8, 0.5, ease.back)));
      arcW.userData.grow(ep(lt, tDiff - 0.2, 0.6));
      arcM.userData.grow(ep(lt, tDiff + 0.1, 0.8));
      L.crop.at(camera, V(-6.6, 2.4, 0), pc);
      L.enc.at(camera, V(-3.1, 2.55, 0), pe);
      L.w.at(camera, S.clone().add(dW.clone().multiplyScalar(R + 0.45)), ep(lt, tAvg + 0.6, 0.5));
      L.m.at(camera, S.clone().add(dM.clone().multiplyScalar(R + 0.5)), ep(lt, tAvg + 0.8, 0.5));
      L.i.at(camera, S.clone().add(dI.clone().multiplyScalar(R + 0.35)), ep(lt, tEnc + 0.6, 0.5) * (1 - ep(lt, tAvg + 0.5, 0.4)));
      show(head, ep(lt, 0.2, 0.6));
      show(pp, ep(lt, tText - 0.2, 0.6));
      rows.forEach((rw, i) => show(rw, ep(lt, tSuch - 0.3 + i * 0.35, 0.4), 8));
      show(gauge, ep(lt, tDiff - 0.2, 0.6));
      const g = 0.94 * ep(lt, tEv - 0.2, 1.4, ease.inOut);
      gv.textContent = g.toFixed(2);
      gb.style.width = (g * 100).toFixed(1) + '%';
      cs.forEach((c, i) => show(c, ep(lt, tFace + 0.4 + i * 0.25, 0.4)));
      show(noFace, ep(lt, tFace, 0.5));
      void tOff;
    },
  };
}

// ---------- 10. decision ----------
export function decide({ group, root, cue, camera }) {
  // 3D: the two passes as two loops and the decision between them
  const loop = (y, color) => {
    const m = new THREE.Mesh(new THREE.TorusGeometry(1.15, 0.05, 12, 90), glow(color, 0.9));
    m.position.set(6.3, y, 0);
    m.rotation.x = 1.15;
    group.add(m);
    const curve = new THREE.CatmullRomCurve3(Array.from({ length: 40 }, (_, i) => {
      const a = (i / 40) * Math.PI * 2;
      return V(Math.cos(a) * 1.15, Math.sin(a) * 1.15, 0);
    }), true);
    const f = flow(curve, 6, color, 0.09);
    f.position.copy(m.position);
    f.rotation.copy(m.rotation);
    group.add(f);
    return { m, f };
  };
  const p1 = loop(2.3, C.sky), p2 = loop(-2.3, C.emerald);
  const dia = new THREE.Mesh(new THREE.OctahedronGeometry(0.55), std(0x1e1b4b, C.violet, 0.9));
  dia.position.set(6.3, 0, 0);
  group.add(dia);
  const L = {
    p1: label3d(root, `<div class="t" style="color:${hex(C.sky)}">pass 1</div><div class="d">gather evidence</div>`),
    d: label3d(root, `<div class="t" style="color:${hex(C.violet)}">decide</div><div class="d">whole video seen</div>`),
    p2: label3d(root, `<div class="t" style="color:${hex(C.emerald)}">pass 2</div><div class="d">render the blur</div>`),
  };
  // strips: frame-by-frame vs two passes
  const NC = 44, ON = 7, OFF = 34, CUTAT = 39;
  const r = rng(5);
  const noisy = Array.from({ length: NC }, (_, i) => (i >= ON && i <= OFF ? 0.55 + 0.4 * (r() - 0.5) * 2 : 0));
  const strip = (title, cls) => {
    const w = el('div', '', `<div style="font-size:23px;margin-bottom:10px" class="${cls}">${title}</div>`);
    const s = el('div', '', '', w);
    s.style.cssText = 'display:flex;gap:3px;position:relative';
    const cells = Array.from({ length: NC }, (_, i) => {
      const c = el('i', '', '', s);
      c.style.cssText = 'display:block;width:22px;height:40px;border-radius:4px;background:rgba(148,163,184,.13)';
      if (i >= ON && i <= OFF) c.style.background = 'rgba(148,163,184,.3)';
      return c;
    });
    return { w, s, cells };
  };
  const top = place(el('div', 'panel', '', root), 110, 205, 1150);
  top.style.padding = '22px 28px';
  const sa = strip('Frame by frame (a browser extension): starts late, flickers', 'muted');
  const sb = strip('Two passes (HalalifyVid): first frame − 3 to last frame + 3, stops at the cut', 'em');
  top.appendChild(sa.w);
  sb.w.style.marginTop = '20px';
  top.appendChild(sb.w);
  const cutMark = el('div', '', '', sb.s);
  cutMark.style.cssText = `position:absolute;left:${CUTAT * 25 - 2}px;top:-8px;width:4px;height:56px;background:${hex(C.red)};box-shadow:0 0 12px ${hex(C.red)};border-radius:2px`;
  const cutLab = el('div', '', 'cut', sb.s);
  cutLab.style.cssText = `position:absolute;left:${CUTAT * 25 + 10}px;top:6px;font-size:20px;color:${hex(C.red)}`;
  const legend = el('div', 'row', `<span class="chip" style="font-size:18px;padding:6px 12px"><i style="color:#94a3b8"></i>person on screen</span><span class="chip" style="font-size:18px;padding:6px 12px"><i style="color:${hex(C.rose)}"></i>blurred</span>`, top);
  legend.style.marginTop = '16px';
  // evidence chart
  const chart = place(el('div', 'panel', '', root), 110, 545, 1150);
  chart.style.padding = '18px 24px';
  const CW = 1100, CH = 300;
  const samples = Array.from({ length: 46 }, (_, i) => {
    const far = i > 14 && i < 22;
    const p = far ? 0.25 + r() * 0.5 : 0.82 + r() * 0.17;
    return { x: 60 + (i / 45) * (CW - 120), p, q: i === 18 || i === 19 };
  });
  const yOf = (p) => 52 + (1 - p) * (CH - 92);
  chart.innerHTML = `<div class="kicker" style="color:${hex(C.violet)}">One track's evidence, judged as a whole</div>
  <svg width="${CW}" height="${CH}" viewBox="0 0 ${CW} ${CH}" style="margin-top:6px">
   <line x1="50" y1="${yOf(0)}" x2="${CW - 40}" y2="${yOf(0)}" stroke="#475569" stroke-width="2"/>
   <text x="10" y="${yOf(1) + 6}" fill="#94a3b8" font-size="18">1.0</text>
   <text x="10" y="${yOf(0) + 6}" fill="#94a3b8" font-size="18">0</text>
   <g class="thr"><line x1="50" y1="${yOf(0.35)}" x2="${CW - 40}" y2="${yOf(0.35)}" stroke="${hex(C.gold)}" stroke-width="3" stroke-dasharray="12 8"/>
     <text x="${CW - 300}" y="${yOf(0.35) + 30}" fill="${hex(C.gold)}" font-size="22" font-weight="600">blur from 0.35, not 0.5</text></g>
   <g class="mean"><line x1="50" y1="${yOf(0.86)}" x2="${CW - 40}" y2="${yOf(0.86)}" stroke="${hex(C.rose)}" stroke-width="3" opacity="0.7"/>
     <text x="60" y="26" fill="${hex(C.rose)}" font-size="22" font-weight="600">verdict on all samples: woman → blur the whole track</text></g>
   ${samples.map((s, i) => s.q
    ? `<text class="s" data-i="${i}" x="${s.x}" y="${yOf(0.5) + 8}" text-anchor="middle" fill="#cbd5e1" font-size="26" font-weight="700">?</text>`
    : `<circle class="s" data-i="${i}" cx="${s.x}" cy="${yOf(s.p)}" r="8" fill="${hex(C.rose)}"/>`).join('')}
   <text class="qn" x="${samples[23].x + 10}" y="${yOf(0.5) + 8}" fill="#cbd5e1" font-size="20">← not classifiable: blurred too</text>
  </svg>`;
  const dots = [...chart.querySelectorAll('.s')];
  const thr = chart.querySelector('.thr'), mean = chart.querySelector('.mean'), qn = chart.querySelector('.qn');
  const tBrowser = cue('browser'), tFlick = cue('flickers'), tTwice = cue('twice'), tGather = cue('gathers');
  const tJudged = cue('judged'), tThr = cue('threshold'), tClass = cue('classified'), tSpan = cue('spans');
  const tWorse = cue('worse');

  return {
    cam: (lt) => ({ p: [1.0, 0.4, 16], l: [1.0, 0.0, 0] }),
    update(lt) {
      const a1 = ep(lt, tTwice - 0.2, 0.6), ad = ep(lt, tJudged - 0.3, 0.6), a2 = ep(lt, tSpan - 0.4, 0.6);
      setOp(p1.m, 0.9 * a1);
      p1.f.userData.tick(lt, 0.25, a1);
      setOp(p2.m, 0.9 * a2);
      p2.f.userData.tick(lt, 0.25, a2);
      dia.scale.setScalar(Math.max(0.001, ad));
      dia.rotation.y = lt * 0.9;
      L.p1.at(camera, V(6.3, 3.75, 0), a1);
      L.d.at(camera, V(7.9, 0, 0), ad);
      L.p2.at(camera, V(6.3, -0.85, 0), a2);
      show(top, ep(lt, tBrowser - 0.3, 0.6));
      // frame-by-frame row: blur starts late and flickers
      const shownA = ep(lt, tBrowser, 0.3);
      sa.cells.forEach((c, i) => {
        const lag = i >= ON + 4 && i <= OFF + 1;
        const flick = noise1(i * 1.9 + Math.floor(lt * 6) * 0.77, 4) > 0.42;
        const on = shownA > 0 && lag && (lt < tFlick ? noisy[i] > 0.5 : flick);
        c.style.background = on ? hex(C.rose) : i >= ON && i <= OFF ? 'rgba(148,163,184,.3)' : 'rgba(148,163,184,.13)';
      });
      const pb = ep(lt, tTwice, 0.4);
      show(sb.w, pb, 10);
      const fill = ep(lt, tSpan, 1.2);
      sb.cells.forEach((c, i) => {
        const inBlur = i >= ON - 3 && i <= Math.min(OFF + 3, CUTAT - 1);
        const on = inBlur && (i - (ON - 3)) / (CUTAT - ON + 3) < fill;
        c.style.background = on ? hex(C.rose) : i >= ON && i <= OFF ? 'rgba(148,163,184,.3)' : 'rgba(148,163,184,.13)';
        c.style.boxShadow = on ? `0 0 10px ${hex(C.rose)}` : 'none';
      });
      cutMark.style.opacity = cutLab.style.opacity = ep(lt, tSpan + 1.2, 0.4);
      show(chart, ep(lt, tGather - 0.3, 0.6));
      dots.forEach((d) => {
        const i = Number(d.dataset.i);
        const s = samples[i];
        const vis = s.q ? ep(lt, tClass - 0.2, 0.4) : ep(lt, tGather + (i / 46) * 2.6, 0.25);
        d.style.opacity = vis.toFixed(3);
      });
      mean.style.opacity = ep(lt, tJudged, 0.6);
      thr.style.opacity = ep(lt, tThr - 0.2, 0.5);
      qn.style.opacity = ep(lt, tClass, 0.5);
      void tWorse;
    },
  };
}

// ---------- 11. render ----------
export function render({ group, root, cue, camera }) {
  const street = canvasTex(960, 540, (c, w, h) => drawStreet(c, w, h));
  const maskNear = canvasTex(160, 90, (c, w, h) => drawMask(c, w, h), true);
  const maskSoft = canvasTex(160, 90, (c, w, h) => drawMask(c, w, h));
  const blurCopy = canvasTex(960, 540, (c, w, h) => {
    const s = document.createElement('canvas');
    s.width = w;
    s.height = h;
    drawStreet(s.getContext('2d'), w, h);
    c.drawImage(s, 0, 0, w / 20, h / 20);
    c.imageSmoothingEnabled = true;
    c.filter = 'blur(1px)';
    c.drawImage(c.canvas, 0, 0, w / 20, h / 20, 0, 0, w, h);
  });
  const out = canvasTex(960, 540, (c, w, h) => drawBlurred(c, w, h));
  const pic = framed(3.4, 1.9125, street, C.sky);
  pic.position.set(-5.8, 1.35, 0);
  const mask = framed(1.2, 0.675, maskNear, C.white);
  mask.position.set(-5.8, -1.6, 0);
  const up = framed(3.0, 1.6875, maskSoft, C.white);
  up.position.set(0, 1.35, 0);
  const bl = framed(3.0, 1.6875, blurCopy, C.sky);
  bl.position.set(0, -1.35, 0);
  const res = framed(4.2, 2.3625, out, C.emerald);
  res.position.set(5.9, 0, 0);
  group.add(pic, mask, up, bl, res);
  const links = [
    [V(-4.0, 1.35, 0), V(-2.5, -0.2, 0), V(-1.6, -1.35, 0)],
    [V(-5.1, -1.6, 0), V(-3, -0.2, 0), V(-1.6, 1.35, 0)],
    [V(1.6, 1.35, 0), V(2.8, 0.7, 0), V(3.7, 0.1, 0)],
    [V(1.6, -1.35, 0), V(2.8, -0.7, 0), V(3.7, -0.1, 0)],
  ].map((pts, i) => {
    const t = growTube(pts, 0.028, [C.sky, C.white, C.white, C.sky][i], 0.8);
    const f = flow(t.userData.curve, 3, [C.sky, C.white, C.white, C.sky][i], 0.06);
    group.add(t, f);
    return { t, f };
  });
  const L = {
    pic: label3d(root, '<div class="t">frame n</div><div class="d">1080p · 3 MB · stays inside ffmpeg</div>'),
    mask: label3d(root, '<div class="t">mask n</div><div class="d">160×90 · ~14 kB · from this program</div>'),
    up: label3d(root, '<div class="d" style="color:#e2e8f0">scaled up → soft edge</div>'),
    bl: label3d(root, '<div class="d" style="color:#e2e8f0">shrink 1/20 → box blur → enlarge</div>'),
    res: label3d(root, `<div class="t" style="color:${hex(C.emerald)}">maskedmerge</div><div class="d">+ encode + clean audio</div>`),
  };
  const head = place(el('div', '', '', root), 110, 120, 1000);
  head.innerHTML = `<div class="kicker" style="color:${hex(C.emerald)}">Pass 2 · render</div><div class="h2" style="margin-top:10px">ffmpeg does the pixels</div>`;
  const sizes = place(el('div', 'panel', '', root), 1210, 120, 600);
  sizes.style.padding = '18px 24px';
  sizes.innerHTML = `<div class="kicker" style="color:#cbd5e1">Data through this program, per frame</div>
   <div style="font-size:21px;margin-top:14px">full frames in and out <span class="ro" style="float:right">2 × 3 MB</span></div>
   <div style="height:18px;background:rgba(148,163,184,.12);border-radius:6px;margin-top:6px"><div class="s1" style="height:100%;width:0;background:${hex(C.rose)};border-radius:6px"></div></div>
   <div style="font-size:21px;margin-top:14px">one mask <span class="em" style="float:right">14 kB</span></div>
   <div style="height:18px;background:rgba(148,163,184,.12);border-radius:6px;margin-top:6px"><div class="s2" style="height:100%;width:0;background:${hex(C.emerald)};border-radius:6px;min-width:0"></div></div>`;
  const s1 = sizes.querySelector('.s1'), s2 = sizes.querySelector('.s2');
  const sync = place(el('div', 'row', '', root), 0, 820, 1920);
  sync.style.justifyContent = 'center';
  const cnt = el('span', 'chip mono', '', sync);
  cnt.style.fontSize = '24px';
  const fps = el('span', 'chip', '<i></i>same <span class="mono">fps=…:start_time=0</span> filter in both passes', sync);
  fps.querySelector('i').style.color = hex(C.violet);
  const copy = el('span', 'chip', '<i></i>nobody to blur → video stream copied untouched', sync);
  copy.querySelector('i').style.color = hex(C.gold);
  const tMb = cue('three-megabyte', 1, cue('three')), tMask = cue('mask'), tKb = cue('kilobytes'), tScale = cue('scales');
  const tBlur = cue('blurs'), tMerge = cue('merges'), tShare = cue('share'), tN = cue('always'), tNobody = cue('nobody');

  return {
    cam: (lt) => ({ p: [Math.sin(lt * 0.1) * 0.8, 0.5, 16.5], l: [0, 0.0, 0] }),
    update(lt) {
      const pp = ep(lt, 0.3, 0.6);
      setOp(pic, pp);
      const pm = ep(lt, tMask - 0.2, 0.6, ease.back);
      mask.scale.setScalar(Math.max(0.001, pm));
      const pu = ep(lt, tScale - 0.2, 0.7, ease.back), pbl = ep(lt, tBlur - 0.2, 0.7, ease.back), pr = ep(lt, tMerge - 0.2, 0.8, ease.back);
      up.scale.setScalar(Math.max(0.001, pu));
      bl.scale.setScalar(Math.max(0.001, pbl));
      res.scale.setScalar(Math.max(0.001, pr));
      links[1].t.userData.grow(ep(lt, tScale - 0.5, 0.5));
      links[0].t.userData.grow(ep(lt, tBlur - 0.5, 0.5));
      links[2].t.userData.grow(ep(lt, tMerge - 0.5, 0.5));
      links[3].t.userData.grow(ep(lt, tMerge - 0.4, 0.5));
      links.forEach((l, i) => l.f.userData.tick(lt, 0.4, ep(lt, [tBlur, tScale, tMerge, tMerge][i], 0.5)));
      L.pic.at(camera, V(-5.8, 2.75, 0), pp);
      L.mask.at(camera, V(-5.8, -2.45, 0), pm);
      L.up.at(camera, V(0, 2.55, 0), pu);
      L.bl.at(camera, V(0, -2.5, 0), pbl);
      L.res.at(camera, V(5.9, -1.75, 0), pr);
      show(head, ep(lt, 0.2, 0.6));
      show(sizes, ep(lt, tMb - 0.2, 0.5));
      s1.style.width = (100 * ep(lt, tMb, 0.8)).toFixed(1) + '%';
      s2.style.width = Math.max(ep(lt, tKb, 0.4) * 0.6, 0).toFixed(2) + '%';
      s2.style.minWidth = ep(lt, tKb, 0.4) > 0.01 ? '4px' : '0';
      const pn = ep(lt, tShare - 0.2, 0.5);
      show(cnt, pn);
      const n = 1040 + Math.floor(Math.max(0, lt - tShare) * 24);
      cnt.innerHTML = `frame <b class="sk">${n}</b> ⟷ mask <b class="em">${n}</b>`;
      show(fps, ep(lt, tN, 0.5));
      show(copy, ep(lt, tNobody - 0.2, 0.5));
    },
  };
}

// ---------- 12. runtime ----------
export function runtime({ group, root, cue, camera }) {
  const tiles = [
    { n: 'CUDA', d: 'NVIDIA', c: C.emerald, x: 1.4 },
    { n: 'DirectML', d: 'any DirectX 12 GPU', c: C.sky, x: 4.4 },
    { n: 'CPU', d: 'always works', c: 0x94a3b8, x: 7.4 },
  ].map((t) => {
    const n = node(2.4, 1.4, t.c);
    n.position.set(t.x, 1.9, 0);
    group.add(n);
    return { ...t, n, l: label3d(root, `<div class="t">${t.n}</div><div class="d">${t.d}</div>`) };
  });
  const orb = new THREE.Mesh(new THREE.SphereGeometry(0.16, 16, 12), glow(C.white, 1));
  group.add(orb);
  const fb = [0, 1].map((i) => {
    const t = growTube([V(tiles[i].x + 1.25, 1.9, 0), V(tiles[i + 1].x - 1.25, 1.9, 0)], 0.022, 0x64748b, 0.7);
    group.add(t);
    return t;
  });
  const check = label3d(root, `<div class="tag" style="color:${hex(C.emerald)};border-color:${hex(C.emerald)}">✓ session created: keep it</div>`);
  const lIf = label3d(root, '<div class="d">if it fails, next →</div>');
  const head = place(el('div', '', '', root), 110, 120, 900);
  head.innerHTML = `<div class="kicker" style="color:${hex(C.sky)}">Under the hood</div><div class="h2" style="margin-top:10px">Start fast, verify everything</div>`;
  const start = place(el('div', 'panel', '', root), 110, 300, 800);
  start.innerHTML = `<div class="kicker" style="color:#cbd5e1">ONNX Runtime start-up on this laptop</div>
   <div style="font-size:23px;margin-top:16px">1.22 and later <span class="muted">(hardware discovery)</span><span class="ro" style="float:right">11 s</span></div>
   <div style="height:22px;background:rgba(148,163,184,.12);border-radius:7px;margin-top:6px"><div class="r1" style="height:100%;width:0;background:${hex(C.rose)};border-radius:7px"></div></div>
   <div style="font-size:23px;margin-top:16px">1.21, loaded at run time <span class="em" style="float:right">30 ms</span></div>
   <div style="height:22px;background:rgba(148,163,184,.12);border-radius:7px;margin-top:6px"><div class="r2" style="height:100%;width:0;background:${hex(C.emerald)};border-radius:7px"></div></div>`;
  const r1 = start.querySelector('.r1'), r2 = start.querySelector('.r2');
  const enc = place(el('div', 'panel', '', root), 110, 600, 800);
  enc.innerHTML = `<div class="kicker" style="color:#cbd5e1">Encoders: test-driven with a few frames</div><div class="row er" style="margin-top:16px"></div><div class="muted" style="font-size:20px;margin-top:14px">fastest pair that really works wins · result cached</div>`;
  const er = enc.querySelector('.er');
  const encs = [['h264_nvenc', 1], ['hevc_nvenc', 1], ['h264_qsv', 1], ['libx264', 1], ['h264_nvenc @ ffmpeg 9', 0]].map(([n, ok]) => {
    const e = el('span', 'chip mono', `${n} <b class="${ok ? 'em' : 'ro'}">${ok ? '✓' : '✗'}</b>`, er);
    e.style.fontSize = '19px';
    return e;
  });
  const sha = place(el('div', 'panel', '', root), 1010, 600, 800);
  sha.innerHTML = `<div class="kicker" style="color:#cbd5e1">setup: pinned packages, checked by SHA-256</div><div class="mono sh" style="font-size:20px;line-height:1.7;margin-top:14px;white-space:pre"></div>`;
  const sh = sha.querySelector('.sh');
  const lines = [
    ['onnxruntime 1.21.0', 'a3f1 9c07 … 5be2'],
    ['cuda runtime 12.6', '7d20 e4aa … 0c91'],
    ['cudnn 9.6', 'c55b 18f3 … 9e04'],
    ['models (3 + BandIt)', '0b6e 4d2c … a7f8'],
  ];
  const tNew = cue('newer'), tThirty = cue('thirty'), tCuda = cue('CUDA'), tDml = cue('Direct'), tCpu = cue('CPU');
  const tKeep = cue('keeps'), tEnc = cue('Encoders'), tVer = cue('verifies');

  return {
    cam: (lt) => ({ p: [0.5, 0.4, 16], l: [0.5, 0.2, 0] }),
    update(lt) {
      tiles.forEach((t, i) => {
        const p = ep(lt, [tCuda, tDml, tCpu][i] - 0.3, 0.5, ease.back);
        lightNode(t.n, i === 0 ? ep(lt, tKeep - 0.2, 0.4) * 0.9 + 0.1 : 0.15);
        t.n.scale.multiplyScalar(Math.max(0.001, p));
        t.l.at(camera, V(t.x, 1.9, 0.3), p);
      });
      fb.forEach((t, i) => t.userData.grow(ep(lt, [tDml, tCpu][i] - 0.3, 0.5)));
      // the probe tries CUDA and stays there
      const u = ep(lt, tCuda - 0.6, 0.8, ease.inOut);
      orb.position.set(lerp(-1.5, tiles[0].x, u), lerp(3.4, 2.75, u), 0.6);
      orb.scale.setScalar(Math.max(0.001, ep(lt, tCuda - 0.8, 0.3)) * (1 + 0.15 * Math.sin(lt * 8)));
      check.at(camera, V(tiles[0].x, 0.75, 0), ep(lt, tKeep - 0.2, 0.5));
      lIf.at(camera, V(4.4, 3.05, 0), ep(lt, tDml, 0.5));
      show(head, ep(lt, 0.2, 0.6));
      show(start, ep(lt, 0.8, 0.6));
      r1.style.width = (100 * ep(lt, tNew, 2.2, (x) => x)).toFixed(1) + '%';
      r2.style.width = (0.4 * ep(lt, tThirty, 0.3)).toFixed(2) + '%';
      r2.style.minWidth = ep(lt, tThirty, 0.3) > 0.01 ? '5px' : '0';
      show(enc, ep(lt, tEnc - 0.3, 0.5));
      encs.forEach((e, i) => show(e, ep(lt, tEnc + i * 0.3, 0.4), 8));
      show(sha, ep(lt, tVer - 0.6, 0.5));
      const k = Math.floor(ep(lt, tVer - 0.3, 2.4, (x) => x) * lines.length * 10);
      sh.innerHTML = lines.map(([a, b], i) => {
        const shown = Math.max(0, Math.min(10, k - i * 10));
        if (shown <= 0) return '';
        return `${a.padEnd(22)}<span class="muted">${b.slice(0, Math.ceil((b.length * shown) / 10))}</span>${shown >= 10 ? ' <b class="em">✓</b>' : ''}`;
      }).join('\n');
    },
  };
}

// ---------- 13. results ----------
export function results({ group, root, cue, camera }) {
  const S = 0.0062;
  const bars = [
    { n: 'remove music', s: 125, c: C.amber },
    { n: 'find + classify', s: 92, c: C.sky },
    { n: 'blur + encode', s: 59, c: C.emerald },
    { n: 'whole job', s: 277, c: C.violet },
    { n: 'film length', s: 720, c: 0x475569 },
  ].map((b, i) => {
    const m = rbox(0.9, 1, 0.9, 0.06, i === 4 ? glow(b.c, 0.35) : std(0x0f172a, b.c, 0.9));
    m.geometry.translate(0, 0.5, 0);
    m.position.set(-7.6 + i * 1.45, -2.6, 0);
    group.add(m);
    return { ...b, m, l: label3d(root, `<div class="t">${b.s >= 600 ? '12 min' : b.s + ' s'}</div>`), n2: label3d(root, `<div class="d" style="font-size:17px">${b.n}</div>`) };
  });
  const base = new THREE.Mesh(new THREE.BoxGeometry(7.6, 0.06, 1.6), glow(0x334155, 0.8));
  base.position.set(-4.7, -2.63, 0);
  group.add(base);
  const head = place(el('div', '', '', root), 110, 120, 1000);
  head.innerHTML = `<div class="kicker" style="color:${hex(C.emerald)}">Results</div><div class="h2" style="margin-top:10px">Fast, and careful</div>`;
  const speed = place(el('div', 'panel', '', root), 110, 250, 700);
  speed.innerHTML = `<div class="stat em" style="font-size:84px">2.7× <small>real time</small></div><div class="statlbl">12-minute 1920×800 film · laptop RTX 2000 Ada (8 GB)</div>`;
  const acc = place(el('div', 'col', '', root), 1060, 250, 760);
  acc.style.gap = '24px';
  const dotsHtml = (n, ok, cOk, cBad) => Array.from({ length: n }, (_, i) =>
    `<i class="dt" style="display:inline-block;width:16px;height:16px;border-radius:50%;margin:3px;background:${i < ok ? cOk : cBad};opacity:0"></i>`).join('');
  const wbox = el('div', 'panel', `<div class="stat ro wn">0/29</div><div class="statlbl">women blurred, incl. hijab, face masks, seen from behind, on posters</div><div style="margin-top:12px">${dotsHtml(29, 29, hex(C.rose), hex(C.rose))}</div>`, acc);
  const mbox = el('div', 'panel', `<div class="stat sk mn">0/38</div><div class="statlbl">men left alone</div><div style="margin-top:12px">${dotsHtml(38, 37, hex(C.sky), hex(C.gold))}</div>`, acc);
  const wn = wbox.querySelector('.wn'), mn = mbox.querySelector('.mn');
  const wd = [...wbox.querySelectorAll('.dt')], md = [...mbox.querySelectorAll('.dt')];
  const note = place(el('div', 'muted', 'four clips, people with a clear view, checked by eye', root), 1060, 770, 760);
  note.style.fontSize = '20px';
  const tFilm = cue('twelve-minute', 1, cue('twelve')), tTwo = cue('two'), tW = cue('twenty-nine'), tM = cue('thirty-seven', 1, cue('thirty'));

  return {
    cam: (lt) => ({ p: [-1.5 + Math.sin(lt * 0.2) * 0.6, 1.0, 16], l: [-1.5, 0.3, 0] }),
    update(lt) {
      bars.forEach((b, i) => {
        const p = ep(lt, tFilm - 0.2 + i * 0.25, 1.0, ease.out);
        b.m.scale.y = Math.max(0.001, b.s * S * p);
        b.l.at(camera, V(b.m.position.x, -2.6 + b.s * S * p + 0.35, 0), ep(lt, tFilm + i * 0.25 + 0.5, 0.4));
        b.n2.at(camera, V(b.m.position.x, -3.05, 0.5), ep(lt, tFilm + i * 0.25, 0.4));
      });
      show(head, ep(lt, 0.2, 0.6));
      show(speed, ep(lt, tTwo - 0.2, 0.5));
      show(wbox, ep(lt, tW - 0.4, 0.5));
      show(mbox, ep(lt, tM - 0.4, 0.5));
      show(note, ep(lt, tM + 1.5, 0.5));
      const kw = Math.round(29 * ep(lt, tW, 1.6, (x) => x));
      wn.textContent = `${kw}/29`;
      wd.forEach((d, i) => {
        d.style.opacity = i < kw ? '1' : '0.15';
        d.style.filter = i < kw ? 'blur(2px)' : 'none';
      });
      const km = Math.round(38 * ep(lt, tM, 1.6, (x) => x));
      mn.textContent = `${Math.min(km, 37)}/38`;
      md.forEach((d, i) => (d.style.opacity = i < km ? '1' : '0.15'));
    },
  };
}

// ---------- 14. outro ----------
export function outro({ group, root, cue, camera }) {
  const fr = new THREE.Group();
  fr.position.set(0, 1.85, 0);
  group.add(fr);
  const f = framed(3.4, 1.9125, canvasTex(960, 540, (c, w, h) => drawBlurred(c, w, h)), C.emerald);
  fr.add(f);
  const N = 200, R = 2.3;
  const pts = Array.from({ length: N + 1 }, (_, i) => V(Math.cos((i / N) * Math.PI * 2) * R, Math.sin((i / N) * Math.PI * 2) * R, 0));
  const ring = fatLine(pts, C.emerald, 4, 1);
  fr.add(ring);
  const arr = new Float32Array((N + 1) * 3);
  const term = place(el('div', 'term', '', root), 110, 150, 1020);
  const cmds = [
    ['halalify lecture.mp4', ''],
    ['halalify https://youtu.be/VIDEO_ID -o clean.mp4', ''],
    ['halalify film.mkv --blur everyone --shape box', ''],
    ['halalify talk.mp4 --music-model vocals --stride 2', ''],
  ];
  const opts = place(el('div', 'panel', '', root), 1200, 150, 620);
  opts.innerHTML = `<div class="kicker" style="color:#cbd5e1">Main options</div>
   <div class="mono" style="font-size:21px;line-height:1.9;margin-top:10px">
    <div><span class="em">--blur</span> women | men | everyone | nobody</div>
    <div><span class="em">--shape</span> silhouette | box</div>
    <div><span class="em">--style</span> blur | pixelate | black</div>
    <div><span class="em">--music-model</span> dialogue | vocals</div>
    <div><span class="em">--threshold</span> 0.35 · <span class="em">--device</span> cuda | dml | cpu</div>
   </div>`;
  const os = [...opts.querySelectorAll('.mono > div')];
  const plat = place(el('div', 'row', '', root), 110, 560, 1700);
  plat.style.gap = '20px';
  const win = el('div', 'panel', `<div class="h3">Windows</div><div class="muted" style="font-size:22px;margin-top:6px">drag a video onto <span class="mono em">HalalifyVid.cmd</span></div>`, plat);
  const lin = el('div', 'panel', `<div class="h3">Linux</div><div class="muted" style="font-size:22px;margin-top:6px"><span class="mono em">./halalifyvid.sh video.mp4</span></div>`, plat);
  const first = el('div', 'panel', `<div class="h3">First run</div><div class="muted" style="font-size:22px;margin-top:6px">installs runtime + models into <span class="mono">data/</span></div>`, plat);
  const warn = place(el('div', 'panel', '', root), 110, 760, 1700);
  warn.style.borderColor = hex(C.gold);
  warn.innerHTML = `<div class="h3" style="color:${hex(C.gold)}">It can still miss someone. Review the result before relying on it.</div>`;
  const end = place(el('div', '', '', root), 0, 690, 1920);
  end.style.textAlign = 'center';
  end.innerHTML = `<div class="h1" style="font-size:110px">Halalify<span class="em">Vid</span></div><div class="sub" style="margin-top:16px">Music removed. Modesty kept. On your own machine.</div>`;
  const tCmd = cue('command'), tOpt = cue('options'), tRel = cue('Releases'), tMiss = cue('miss'), tThat = cue('That');

  return {
    cam: (lt) => ({ p: [0, 0.6, lerp(15, 13, ease.inOut(prog(lt, tThat - 0.5, 2.5)))], l: [0, 0.3, 0] }),
    update(lt) {
      const out = ep(lt, tThat - 0.6, 0.6);
      show(term, ep(lt, 0.2, 0.6) * (1 - out));
      const chars = Math.max(0, (lt - tCmd + 0.3) * 38);
      let used = 0;
      term.innerHTML = cmds.map(([c]) => {
        const n = Math.max(0, Math.min(c.length, Math.floor(chars - used)));
        used += c.length + 6;
        if (n <= 0 && used - c.length - 6 > chars) return '<span class="p">$</span> ';
        return `<span class="p">$</span> ${c.slice(0, n)}`;
      }).join('\n');
      show(opts, ep(lt, tOpt - 0.3, 0.5) * (1 - out));
      os.forEach((o, i) => show(o, ep(lt, tOpt + i * 0.25, 0.4), 6));
      show(win, ep(lt, tRel - 0.1, 0.5) * (1 - out));
      show(lin, ep(lt, tRel + 0.3, 0.5) * (1 - out));
      show(first, ep(lt, tRel + 0.7, 0.5) * (1 - out));
      show(warn, ep(lt, tMiss - 0.8, 0.5) * (1 - out));
      const pe = ep(lt, tThat - 0.2, 0.9);
      show(end, pe, 26);
      setOp(fr, pe);
      fr.scale.setScalar(Math.max(0.001, 0.85 + 0.15 * pe));
      fr.rotation.y = Math.sin(lt * 0.4) * 0.12;
      for (let i = 0; i <= N; i++) {
        const a = (i / N) * Math.PI * 2;
        const d = 0.16 * Math.abs(Math.sin(a * 9 + lt * 4)) * (0.6 + 0.4 * noise1(i * 0.2 + lt * 2, 5));
        arr[i * 3] = Math.cos(a) * (R + d);
        arr[i * 3 + 1] = Math.sin(a) * (R + d);
        arr[i * 3 + 2] = 0;
      }
      setLine(ring, arr);
    },
  };
}
