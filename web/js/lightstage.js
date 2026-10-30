// OpenRelight light stage: a 3D sphere editor for the Light Rig node.
//
// Everything it edits is written back to the node's `rig_json` widget (and `preset` is set to "custom"),
// so the Python side reads exactly what the editor shows.
//
// The sphere is the space around the subject (centre). The camera is in front, marked Front. A light sits on
// the sphere surface: where it sits is the direction it comes from (rig azimuth / elevation; azimuth 0 =
// camera side, +90 = screen right, 180 = behind; elevation 0 = eye level, 90 = overhead).
//   drag a light            move it (at another time on the timeline this records a keyframe)
//   drag a keyframe ◆       move that keyframe; click it to jump to its time; right-click to delete it
//   drag empty space        turn the view;  double-click the sphere: add a light there
//   wheel over a light      its distance
// A light with one keyframe is static; with two or more it moves along the path drawn on the sphere.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE = "OR_LightRig";
const FPS = 24;
const EASES = [["smooth", "Smooth"], ["easeInOut", "Ease in and out"], ["easeIn", "Ease in"], ["easeOut", "Ease out"], ["linear", "Linear"], ["hold", "Hold, then jump"]];
// the fixture menu (Blender's four): picking one also sets its usual softness. Where it sits, how near, how big
// (panel) and how wide (spot beam) are set by dragging on the stage.
const TYPES = [["area", "Panel light", 0.75], ["spot", "Spotlight (beam)", 0.2], ["point", "Point light", 0.5], ["directional", "Sun", 0.1]];
const SHAPES = [["square", "Square"], ["disc", "Round"]];
// where a new light of each fixture starts. The name is plain on purpose: the LM reads it, and a red searchlight left
// named "Fill" told it the wrong thing (2026-10-30)
const NEW_LIGHT = {
  area: { name: "Panel", az: -40, el: 25, dist: 1.5, ev: 0, kelvin: 5600 },
  spot: { name: "Spot", az: -30, el: 30, dist: 1.5, ev: 0.5, kelvin: 5600 },
  point: { name: "Point", az: 60, el: -10, dist: 1.0, ev: 0, kelvin: 2700 },
  directional: { name: "Sun", az: 120, el: 35, dist: 1.0, ev: 0.5, kelvin: 5200 },
};
// presets and looks stay out of the editor until they are tested with the lighting director: a look never reached the
// prompts and a preset's strobe never reached the LM (2026-10-30)
const SHOW_UNTESTED = false;
const ANIM = ["azimuth", "elevation", "distance", "intensity_ev", "kelvin"];
const DEG = Math.PI / 180;

let PRESET_DATA = null;
async function presets() {
  if (PRESET_DATA) return PRESET_DATA;
  try { PRESET_DATA = await (await fetch("/openrelight/presets")).json(); } catch (e) { PRESET_DATA = { presets: {}, looks: {}, films: {}, grades: {} }; }
  return PRESET_DATA;
}

// --------------------------------------------------------------------------- data helpers
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const wrap180 = (a) => ((((a + 180) % 360) + 360) % 360) - 180;
const easeFn = {
  linear: (u) => u, easeIn: (u) => u * u, easeOut: (u) => 1 - (1 - u) * (1 - u),
  easeInOut: (u) => (u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2), smooth: (u) => u * u * (3 - 2 * u), hold: () => 0,
};

function kelvinRGB(k) {
  const t = clamp(k, 1000, 40000) / 100;
  let r, g, b;
  if (t <= 66) { r = 255; g = 99.47 * Math.log(t) - 161.12; } else { r = 329.7 * (t - 60) ** -0.1332; g = 288.1 * (t - 60) ** -0.0755; }
  if (t >= 66) b = 255; else if (t <= 19) b = 0; else b = 138.52 * Math.log(t - 10) - 305.04;
  return [clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255)].map(Math.round);
}
const lightCSS = (l, kelvin) => (l.color?.hex ? l.color.hex : `rgb(${kelvinRGB(kelvin ?? l.color?.kelvin ?? 5600).join(",")})`);
// distance (in subject distances) <-> radius on the stage: the default 1.5 sits on the sphere, nearer lights
// inside it, farther ones outside; the sun has no distance and sits just outside
const DIST_MIN = 0.5, DIST_MAX = 4, SUN_R = 1.18;
const radiusOf = (d) => 0.45 + 0.9 * Math.log(clamp(d, DIST_MIN, DIST_MAX) / DIST_MIN) / Math.log(DIST_MAX / DIST_MIN);
const distOf = (r) => DIST_MIN * Math.exp(((clamp(r, 0.45, 1.35) - 0.45) / 0.9) * Math.log(DIST_MAX / DIST_MIN));
// a panel's softness is its apparent size seen from the subject (pro.py: 2 + 43 x softness degrees), so a bigger
// or nearer panel is softer
const softnessOf = (size, d) => clamp((2 * Math.atan(size / (2 * d)) / DEG - 2) / 43, 0, 1);
const sizeOf = (soft, d) => clamp(2 * d * Math.tan(((2 + 43 * soft) / 2) * DEG), 0.1, 2.5);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (v) => { const n = Math.hypot(...v) || 1; return v.map((c) => c / n); };
// two unit vectors across a panel that faces the subject from direction d
function panelAxes(d) {
  const c = cross([0, 1, 0], d);
  if (Math.hypot(...c) < 1e-3) return [[1, 0, 0], unit(cross(d, [1, 0, 0]))];
  const u = unit(c);
  return [u, unit(cross(d, u))];
}
const rgbHex = (rgb) => "#" + rgb.map((c) => c.toString(16).padStart(2, "0")).join("");
// warm (-1, 2000K) .. neutral (0, n) .. cool (+1, 12000K): the UI shows the slider, the rig keeps kelvin
const toWarm = (k, n) => clamp(k < n ? -(n - k) / (n - 2000) : (k - n) / (12000 - n), -1, 1);
const fromWarm = (v, n) => Math.round(v < 0 ? n + v * (n - 2000) : n + v * (12000 - n));
const warmWord = (v) => {
  const a = Math.abs(+v);
  if (a < 0.08) return "Neutral white";
  return (a < 0.4 ? "Slightly " : a < 0.75 ? "" : "Very ") + (+v < 0 ? "warm" : "cool");
};

// on between its start_sec and end_sec (either empty = the clip's edge), as in openrelight/rig.py light_on
const lightOn = (l, t) => (l.start_sec == null || t >= l.start_sec - 1e-6) && (l.end_sec == null || t <= l.end_sec + 1e-6);
const getW = (node, name) => node.widgets?.find((w) => w.name === name);

function keySec(key, duration) {
  if (key.sec != null) return +key.sec;
  if (key.frame != null) return key.frame / FPS;
  if (key.t != null) return key.t * duration;
  return 0;
}

function normalizeRig(raw, duration) {
  const rig = JSON.parse(JSON.stringify(raw || {}));
  rig.version = 1;
  rig.ambient = rig.ambient || null;   // null = the original's own light stays as it is
  rig.look = Object.assign({ preset: "", film: "", grade: "" }, rig.look || {});
  rig.user_prompt = rig.user_prompt || "";
  rig.duration_sec = rig.duration_sec || duration || 5;
  rig.removed_originals = rig.removed_originals || [];
  rig.lights = (rig.lights || []).map((l, i) => {
    const out = Object.assign({ id: `light${i}`, name: `Light ${i + 1}`, type: "spot", mode: "generate", enabled: true,
      color: { kelvin: 5600 }, softness: 0.3, cone: { angle: 40, feather: 0.4 }, shadow: true, motion: null, prompt: "" }, l);
    out.cone = Object.assign({ angle: 40, feather: 0.4 }, l.cone || {});
    out.color = Object.assign({}, l.color || { kelvin: 5600 });
    let last = { azimuth: 45, elevation: 20, distance: 1.5, intensity_ev: 0, kelvin: out.color.kelvin ?? 5600 };
    out.keys = (l.keys && l.keys.length ? l.keys : [{}]).map((k) => {
      const nk = { sec: +keySec(k, rig.duration_sec).toFixed(3) };
      for (const f of ANIM) nk[f] = k[f] != null ? +k[f] : last[f];
      nk.ease = k.ease || "smooth";
      last = nk;
      return nk;
    }).sort((a, b) => a.sec - b.sec);
    return out;
  });
  return rig;
}

// direction on the unit sphere: x right, y up, z toward the camera
const dirOf = (az, el) => [Math.sin(az * DEG) * Math.cos(el * DEG), Math.sin(el * DEG), Math.cos(az * DEG) * Math.cos(el * DEG)];
const angOf = ([x, y, z]) => [Math.atan2(x, z) / DEG, Math.asin(clamp(y, -1, 1)) / DEG];

// every animated field at time t (seconds); directions interpolate along the sphere like the Python slerp
function evalLight(l, t) {
  const ks = l.keys;
  if (t <= ks[0].sec || ks.length === 1) return Object.assign({}, ks[0]);
  if (t >= ks[ks.length - 1].sec) return Object.assign({}, ks[ks.length - 1]);
  let i = 0;
  while (i < ks.length - 2 && ks[i + 1].sec <= t) i++;
  const a = ks[i], b = ks[i + 1];
  const u = (easeFn[a.ease] || easeFn.smooth)((t - a.sec) / Math.max(b.sec - a.sec, 1e-6));
  const out = {};
  for (const f of ANIM) out[f] = a[f] + (b[f] - a[f]) * u;
  const v0 = dirOf(a.azimuth, a.elevation), v1 = dirOf(b.azimuth, b.elevation);
  const dot = clamp(v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2], -1, 1);
  let v;
  if (dot > 0.9995) v = v0.map((c, j) => c + (v1[j] - c) * u);
  else {
    const th = Math.acos(dot) * u;
    let rel = v1.map((c, j) => c - v0[j] * dot);
    const n = Math.hypot(...rel) || 1;
    rel = rel.map((c) => c / n);
    v = v0.map((c, j) => c * Math.cos(th) + rel[j] * Math.sin(th));
  }
  [out.azimuth, out.elevation] = angOf(v.map((c) => c / (Math.hypot(...v) || 1)));
  return out;
}

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(kid));
  return e;
}

const CSS = `
.ols{font:12px/1.35 system-ui,sans-serif;color:#ddd;background:#1b1d22;border-radius:8px;padding:6px;box-sizing:border-box;
 display:flex;flex-direction:column;gap:6px;height:100%;overflow:auto}
.ols .row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.ols button,.ols select,.ols input[type=text],.ols input[type=number],.ols textarea{background:#2b2e36;color:#eee;border:1px solid #444;
 border-radius:5px;padding:3px 7px;font:inherit}
.ols button{cursor:pointer}.ols button:hover{background:#3a3e48}.ols button.on{background:#4a6bd8;border-color:#6d8cf0}
.ols .chip{padding:2px 9px;border-radius:11px;border:1px solid #555;cursor:pointer;display:flex;align-items:center;gap:5px}
.ols .chip.sel{border-color:#ffcc55;background:#3a3320}.ols .dot{width:10px;height:10px;border-radius:50%}
.ols .grid{display:grid;grid-template-columns:64px 1fr 52px;gap:3px 6px;align-items:center}
.ols .grid input[type=range]{width:100%}.ols .lab{color:#aaa}.ols .val{text-align:right;color:#ffd27a;font-variant-numeric:tabular-nums}
.ols .sec{border-top:1px solid #333;padding-top:5px}.ols .title{color:#ffcc55;font-weight:600}
.ols .muted{color:#888}.ols textarea{width:100%;box-sizing:border-box;resize:vertical}
.ols canvas{background:radial-gradient(circle at 50% 45%,#22262e,#14161a);border-radius:6px;display:block}
.ols details summary{cursor:pointer;color:#aaa;margin:3px 0}
`;

// --------------------------------------------------------------------------- editor
class LightStage {
  constructor(node, root) {
    this.node = node;
    this.root = root;
    this.t = 0;
    this.sel = 0;
    this.playing = false;
    this.yaw = 55;      // seen from the right of the camera
    this.pitch = 35;    // and from above, so lights in front of the subject do not sit on it
    this.rig = normalizeRig({}, 5);
    this.fps = FPS;     // the source clip's own rate once Source Clip has run
    this.clip = null;   // {info, img}: the clip's contact sheet, drawn as a card in the middle of the sphere
    this.build();
  }

  // Source Clip ran (or the workflow remembers its last run): show the clip, take its length and rate
  setClip(info, save = true) {
    if (!info?.filename) return;
    const img = new Image();
    const clip = { info, img, ready: false };
    img.onload = () => { clip.ready = true; this.draw(); };
    img.onerror = () => { if (this.clip === clip) this.clip = null; this.draw(); };   // temp files go on restart
    img.src = api.apiURL(`/view?filename=${encodeURIComponent(info.filename)}&type=${info.type || "temp"}` +
                         `&subfolder=${encodeURIComponent(info.subfolder || "")}`);
    this.clip = clip;
    this.fps = info.fps || FPS;
    if (save) {
      this.node.properties = this.node.properties || {};
      this.node.properties.openrelight_clip = info;
      const d = +(info.frames / this.fps).toFixed(3);
      if (Math.abs(d - this.rig.duration_sec) > 1e-3) { this.rig.duration_sec = d; this.commit(); }
    }
    this.refresh();
  }

  // ---- data in / out
  async loadFromNode() {
    // a node is created with its default preset and then configured from the workflow; the first load
    // waits for the presets fetch, so without this token it would land last and replace the workflow's rig
    const token = (this._loadToken = (this._loadToken || 0) + 1);
    const preset = getW(this.node, "preset")?.value;
    const text = getW(this.node, "rig_json")?.value || "";
    let raw = null;
    if (preset && preset !== "custom") raw = (await presets()).presets[preset];
    else if (text.trim()) { try { raw = JSON.parse(text); } catch (e) { raw = null; } }
    if (token !== this._loadToken) return;
    this.rig = normalizeRig(raw || {}, this.rig.duration_sec);
    this.sel = clamp(this.sel, 0, Math.max(0, this.rig.lights.length - 1));
    this.refresh();
  }

  commit() {
    const w = getW(this.node, "rig_json");
    const p = getW(this.node, "preset");
    if (p && p.value !== "custom") { this._settingPreset = true; p.value = "custom"; this._settingPreset = false; }
    if (w) w.value = JSON.stringify(this.rig, null, 2);
    this.node.setDirtyCanvas(true, true);
  }

  light() { return this.rig.lights[this.sel]; }
  keyAt(l, t) { return l.keys.find((x) => Math.abs(x.sec - t) < 0.5 / this.fps); }

  // the key the edit goes to: the only key of a still light; of a moving one, its start or its end, whichever is
  // nearer the playhead (the playhead jumps there). No key is added in between: a motion is a start and an end.
  editKey(l) {
    if (l.keys.length === 1) return l.keys[0];
    let k = this.keyAt(l, this.t);
    if (!k) {
      k = l.keys.reduce((a, b) => (Math.abs(b.sec - this.t) < Math.abs(a.sec - this.t) ? b : a));
      this.t = k.sec;
    }
    return k;
  }

  setAnim(field, value) {
    const l = this.light();
    if (!l) return;
    this.editKey(l)[field] = value;
    this.commit(); this.refresh();
  }

  delKey(k) {
    const l = this.light();
    if (!l || l.keys.length < 2) return;
    if (!k) { k = l.keys[0]; for (const x of l.keys) if (Math.abs(x.sec - this.t) < Math.abs(k.sec - this.t)) k = x; }
    l.keys.splice(l.keys.indexOf(k), 1);
    this.commit(); this.refresh();
  }

  addLight(type, az, elv) {
    const r = NEW_LIGHT[type];
    const n = this.rig.lights.length;
    const same = this.rig.lights.filter((l) => l.type === type && !l.original).length;
    this.rig.lights.push(normalizeRig({ lights: [{
      id: `${type}${n}`, name: same ? `${r.name} ${same + 1}` : r.name, type, color: { kelvin: r.kelvin },
      softness: TYPES.find((t) => t[0] === type)[2],
      cone: { angle: type === "spot" ? 20 : 40, feather: 0.4 },
      keys: [{ sec: 0, azimuth: az ?? r.az, elevation: elv ?? r.el, distance: r.dist, intensity_ev: r.ev, kelvin: r.kelvin }],
    }] }, this.rig.duration_sec).lights[0]);
    this.sel = this.rig.lights.length - 1;
    this.commit(); this.refresh();
  }

  async addPreset(name) {
    const p = (await presets()).presets[name];
    if (!p) return;
    const add = normalizeRig(p, this.rig.duration_sec);
    // presets are additive: their lights join the rig, their look fills only what is empty, and their
    // ambient is taken only when the rig had no lights yet
    const wasEmpty = this.rig.lights.length === 0;
    // a lighting setup replaces the clip's own light: the detected light is switched off, not deleted
    if (p.setup) this.rig.lights.forEach((l) => { if (l.original) l.enabled = false; });
    add.lights.forEach((l, i) => { l.id = `${l.id}_${this.rig.lights.length + i}`; this.rig.lights.push(l); });
    for (const f of ["preset", "film", "grade"]) if (!this.rig.look[f] && add.look[f]) this.rig.look[f] = add.look[f];
    if (wasEmpty && !p.setup) this.rig.ambient = add.ambient;
    this.sel = this.rig.lights.length - 1;
    this.commit(); this.refresh();
  }

  // ---- layout
  build() {
    if (!document.getElementById("ols-css")) document.head.append(el("style", { id: "ols-css" }, CSS));
    this.root.classList.add("ols");
    this.root.innerHTML = "";

    const roleSel = el("select", { onchange: (e) => { if (e.target.value) this.addLight(e.target.value); e.target.value = ""; } },
      el("option", { value: "" }, "+ Add light"), ...TYPES.map(([v, t]) => el("option", { value: v }, t)));
    this.presetSel = el("select", { onchange: (e) => { if (e.target.value) this.addPreset(e.target.value); e.target.value = ""; } },
      el("option", { value: "" }, "+ Add preset"));
    presets().then((d) => {
      // professional setups first (they replace the clip's light), then special effects
      const names = Object.keys(d.presets), lab = d.labels || {};
      const setups = names.filter((n) => d.presets[n].setup), rest = names.filter((n) => !d.presets[n].setup);
      const group = (title, list) => list.length && this.presetSel.append(
        el("optgroup", { label: title }, ...list.map((n) => el("option", { value: n }, lab[n] || n))));
      group("Lighting setups (switch the original lights off)", setups);
      group("Effects (added to the current lights)", rest);
    });
    this.root.append(el("div", { class: "row" }, roleSel, SHOW_UNTESTED ? this.presetSel : null,
      el("button", { onclick: () => { this.yaw = 55; this.pitch = 35; this.draw(); } }, "Reset view"),
      el("button", { onclick: () => { this.yaw = 0; this.pitch = 0; this.draw(); } }, "Camera view"),
      el("button", { onclick: () => { this.yaw = 0; this.pitch = 89; this.draw(); } }, "Top view"),
      el("button", { onclick: () => { if (confirm("Remove all lights?")) { this.rig.lights = []; this.sel = 0; this.commit(); this.refresh(); } } }, "Clear")));

    this.canvas = el("canvas", { width: 640, height: 400, style: { width: "100%", height: "400px", cursor: "grab" } });
    this.root.append(this.canvas);
    this.root.append(el("div", { class: "muted" },
      "Drag a light to change its direction | Drag the yellow dot, or Shift+drag or scroll, to move it nearer or farther | Drag the square handle on a panel to resize it | Drag the round handle on a beam to change its angle | Set how a light moves under Motion | Drag empty space to turn the view | Double-click the sphere to add a light"));
    this.chips = el("div", { class: "row" });
    this.root.append(this.chips);
    this.bindCanvas();
    // the node is laid out after the editor is built: redraw at the real size, or the sphere is stretched
    const ro = new ResizeObserver(() => { this.draw(); this.drawStrip(); });
    ro.observe(this.canvas);
    ro.observe(this.root);
    // the node can be resized without either element reporting it (DOM widgets are positioned by the
    // graph canvas); a cheap check keeps the drawing buffer at the displayed size
    this._sizeTimer = setInterval(() => {
      if (this._started && !this.node.graph) { clearInterval(this._sizeTimer); return; }
      if (this.node.graph) this._started = true;
      if (this.canvas.clientWidth && Math.abs(this.canvas.width - this.canvas.clientWidth) > 1) { this.draw(); this.drawStrip(); }
    }, 500);

    // timeline
    this.playBtn = el("button", { onclick: () => this.togglePlay() }, "▶");
    this.time = el("input", { type: "range", min: 0, max: 5, step: 1 / FPS, value: 0, style: { flex: "1" },
      oninput: (e) => { this.t = +e.target.value; this.refresh(false); } });
    this.timeLab = el("span", { class: "val", style: { minWidth: "92px" } });
    this.dur = el("input", { type: "number", min: 0.5, max: 60, step: 0.1, value: 5, style: { width: "56px" }, title: "Clip length in seconds, matches the video",
      onchange: (e) => { this.rig.duration_sec = clamp(+e.target.value || 5, 0.5, 60); this.commit(); this.refresh(); } });
    this.strip = el("canvas", { width: 600, height: 20, style: { width: "100%", height: "20px", cursor: "pointer", background: "#15171b" } });
    this.strip.addEventListener("pointerdown", (e) => {
      const r = this.strip.getBoundingClientRect();
      this.t = clamp((e.clientX - r.left) / r.width, 0, 1) * this.rig.duration_sec;
      const l = this.light();
      const near = l?.keys.find((k) => Math.abs(k.sec - this.t) < this.rig.duration_sec * 0.02);
      if (near) this.t = near.sec;
      this.refresh(false);
    });
    this.root.append(el("div", { class: "sec" },
      el("div", { class: "row" }, this.playBtn, el("button", { onclick: () => { this.t = 0; this.refresh(false); } }, "⏮"),
        this.time, this.timeLab, el("span", { class: "lab" }, "Length"), this.dur, el("span", { class: "lab" }, "s")),
      this.strip));

    this.panel = el("div", { class: "sec" });
    this.scene = el("div", { class: "sec" });
    this.root.append(this.panel, this.scene);
    this.refresh();
  }

  togglePlay() {
    this.playing = !this.playing;
    this.playBtn.textContent = this.playing ? "⏸" : "▶";
    let last = performance.now();
    const step = (now) => {
      if (!this.playing) return;
      this.t += (now - last) / 1000; last = now;
      if (this.t > this.rig.duration_sec) this.t = 0;
      this.refresh(false);
      requestAnimationFrame(step);
    };
    if (this.playing) requestAnimationFrame(step);
  }

  slider(label, value, min, max, step, onDone, fmt = (v) => (+v).toFixed(step < 1 ? 2 : 0)) {
    const val = el("span", { class: "val" }, fmt(value));
    const inp = el("input", { type: "range", min, max, step, value,
      oninput: (e) => { val.textContent = fmt(e.target.value); }, onchange: (e) => onDone(+e.target.value) });
    return { nodes: [el("span", { class: "lab" }, label), inp, val], inp, val, fmt };
  }

  select(label, value, options, onChange) {
    const s = el("select", { onchange: (e) => onChange(e.target.value) }, ...options.map(([v, t]) => el("option", { value: v }, t)));
    s.value = value ?? "";
    return [el("span", { class: "lab" }, label), s, el("span")];
  }

  refresh(rebuild = true) {
    const d = this.rig.duration_sec;
    this.t = clamp(this.t, 0, d);
    this.time.max = d; this.time.step = 1 / this.fps; this.time.value = this.t; this.dur.value = d;
    this.timeLab.textContent = `${this.t.toFixed(2)}s · frame ${Math.round(this.t * this.fps)}`;
    this.draw(); this.drawStrip();
    if (rebuild) { this.buildChips(); this.buildPanel(); this.buildScene(); } else this.updateLive();
  }

  buildChips() {
    this.chips.innerHTML = "";
    if (!this.rig.lights.length) { this.chips.append(el("span", { class: "muted" }, "No lights yet: use + Add light, or double-click the sphere")); return; }
    this.rig.lights.forEach((l, i) => {
      const s = evalLight(l, this.t);
      this.chips.append(el("div", { class: "chip" + (i === this.sel ? " sel" : ""), style: { opacity: l.enabled ? 1 : 0.45 },
        onclick: () => { this.sel = i; this.refresh(); } },
        el("span", { class: "dot", style: { background: lightCSS(l, s.kelvin) } }), l.name,
        l.original ? el("span", { class: "muted", title: "Detected from the clip" }, "Original") : null,
        l.reference ? el("span", { class: "muted", title: "Detected from the lighting reference" }, "Reference") : null,
        el("span", { class: "muted" }, l.keys.length > 1 ? `Moves ◆${l.keys.length}` : "")));
    });
  }

  updateLive() {
    const l = this.light();
    if (l && this.live) {
      const s = evalLight(l, this.t);
      for (const [f, sl] of this.live) {
        const v = sl.toUI ? sl.toUI(s[f]) : s[f];
        sl.inp.value = v; sl.val.textContent = sl.fmt(v);
      }
    }
    this.chips.querySelectorAll(".dot").forEach((d, i) => { const li = this.rig.lights[i]; if (li) d.style.background = lightCSS(li, evalLight(li, this.t).kelvin); });
  }

  buildPanel() {
    const p = this.panel;
    p.innerHTML = "";
    this.live = [];
    const l = this.light();
    if (!l) return;
    const s = evalLight(l, this.t);
    const set = (fn) => { fn(l); this.commit(); this.refresh(); };
    const anim = (g, label, f, min, max, step, fmt) => {
      const sl = this.slider(label, s[f], min, max, step, (v) => this.setAnim(f, v), fmt);
      this.live.push([f, sl]);
      g.append(...sl.nodes);
    };
    const pickType = (v) => set((x) => {
      x.type = v;
      x.softness = TYPES.find((t) => t[0] === v)?.[2] ?? x.softness;
      if (v === "spot") x.cone.angle = 20;   // a beam: its pool lands on part of the scene (lm.py _narrow_beam)
      if (v === "area") { x.size = +sizeOf(x.softness, s.distance).toFixed(2); x.shape = x.shape || "square"; }
      else delete x.size;
    });
    p.append(el("div", { class: "row" }, el("span", { class: "title" }, "This light"),
      el("input", { type: "text", value: l.name, style: { width: "100px" }, onchange: (e) => set((x) => (x.name = e.target.value)) }),
      el("select", { title: "Fixture: also sets its usual softness", onchange: (e) => pickType(e.target.value) },
        ...TYPES.map(([v, t]) => { const o = el("option", { value: v }, t); if (v === l.type) o.selected = true; return o; })),
      l.type === "area" ? el("select", { title: "Panel shape (drag the square handle on the stage to resize)", onchange: (e) => set((x) => (x.shape = e.target.value)) },
        ...SHAPES.map(([v, t]) => { const o = el("option", { value: v }, t); if (v === (l.shape || "square")) o.selected = true; return o; })) : null,
      el("label", {}, el("input", { type: "checkbox", checked: l.enabled, onchange: (e) => set((x) => (x.enabled = e.target.checked)) }), " On"),
      el("button", { onclick: () => {
        if (!confirm(l.original ? `Remove original light "${l.name}"? The clip's light from this direction will be switched off.` : `Remove "${l.name}"?`)) return;
        if (l.original && l.detected) this.rig.removed_originals.push(l.detected);
        this.rig.lights.splice(this.sel, 1); this.sel = Math.max(0, this.sel - 1); this.commit(); this.refresh(); } }, "Remove"),
      l.original ? el("button", { title: "Back to the detected position and settings", onclick: () => set((x) => {
        const d = JSON.parse(JSON.stringify(x.detected)); Object.assign(x, d, { detected: x.detected, original: true }); }) }, "Reset to original") : null));
    // four settings: colour (any colour, or white light from warm to cool), brightness, softness. Direction,
    // beam and distance are where the light sits on the stage; effects and motion details go in the prompt.
    const g = el("div", { class: "grid" });
    const hex = l.color.hex;
    g.append(el("span", { class: "lab" }, "Colour"),
      el("input", { type: "color", value: hex || rgbHex(kelvinRGB(s.kelvin)), title: "Pick any colour. The Warm/cool slider below goes back to white light",
        onchange: (e) => set((x) => (x.color = { hex: e.target.value })) }),
      el("span", { class: "val" }, hex ? "Colour" : ""));
    const warm = this.slider("Warm/cool", hex ? 0 : toWarm(s.kelvin, 5600), -1, 1, 0.05, (v) => {
      if (l.color.hex) l.color = { kelvin: fromWarm(v, 5600) };
      this.setAnim("kelvin", fromWarm(v, 5600));
    }, warmWord);
    warm.toUI = (k) => toWarm(k, 5600);
    if (hex) warm.val.textContent = "(using colour)";
    else this.live.push(["kelvin", warm]);
    g.append(...warm.nodes);
    anim(g, "Brightness", "intensity_ev", -6, 4, 0.1, (v) => `${(+v) >= 0 ? "+" : ""}${(+v).toFixed(1)} stops`);
    // a panel's softness and size go together (bigger = softer at the same distance)
    const soft = this.slider("Softness", l.softness, 0, 1, 0.05, (v) => set((x) => {
      x.softness = v;
      if (x.type === "area") x.size = +sizeOf(v, s.distance).toFixed(2);
    }),
      (v) => (+v < 0.3 ? "Hard" : +v > 0.6 ? "Soft" : "Medium") + ` ${Math.round(+v * 100)}%`);
    g.append(...soft.nodes);
    p.append(g);

    // motion as a few numbers: still, or from a start (time, position) to an end (time, position) with one way of
    // moving between them; plus when the light comes on and goes out. A recorded drag gave keys the keyframes and
    // the prompts could not keep to (a 2 s sweep came out as a jump, 2026-10-30). A position is set by dragging the
    // light with the playhead at that end ("Set start" and "Set end" put it there); brightness and colour can differ too.
    const d = this.rig.duration_sec, ks = l.keys, A = ks[0], B = ks[ks.length - 1];
    const at = (k) => Math.abs(this.t - k.sec) < 0.5 / this.fps;
    const num = (value, onSet, ph = "") => el("input", { type: "number", min: 0, max: d, step: 0.05, value: value ?? "",
      placeholder: ph, style: { width: "56px" }, onchange: (e) => onSet(e.target.value === "" ? null : clamp(+e.target.value, 0, d)) });
    const mode = el("select", { onchange: (e) => set((x) => {
      if (e.target.value === "move" && x.keys.length === 1) {
        const a = Object.assign(x.keys[0], { sec: 0, ease: x.keys[0].ease || "smooth" });
        x.keys = [a, Object.assign({}, a, { sec: +d.toFixed(3) })];
        this.t = d;                       // the end is set next: drag the light to where it goes
      } else if (e.target.value === "still") x.keys = [Object.assign(x.keys[0], { sec: 0 })];
    }) }, el("option", { value: "still" }, "Still"), el("option", { value: "move" }, "Move from start to end"));
    mode.value = ks.length > 1 ? "move" : "still";
    const motion = el("div", { class: "sec" }, el("div", { class: "row" }, el("span", { class: "title" }, "Motion"), mode,
      ks.length > 1 ? el("span", { class: "muted" }, at(A) ? "Dragging the light now sets its start" : at(B) ? "Dragging the light now sets its end" : "") : null));
    if (ks.length > 2)   // keys from an earlier recording
      motion.append(el("div", { class: "row" }, el("span", { class: "muted" }, `This light has ${ks.length} key points (older recording)`),
        el("button", { onclick: () => set((x) => { x.keys = [x.keys[0], x.keys[x.keys.length - 1]]; }) }, "Simplify to start and end")));
    if (ks.length > 1) {
      const ease = el("select", { onchange: (e) => set(() => { A.ease = e.target.value; }) },
        ...EASES.filter(([v]) => v !== "hold").map(([v, t]) => el("option", { value: v }, t)));
      ease.value = A.ease || "smooth";
      motion.append(el("div", { class: "row" },
        el("span", { class: "lab" }, "Start"), num(A.sec, (v) => set(() => { A.sec = +Math.min(v ?? 0, B.sec - 0.05).toFixed(3); })),
        el("span", { class: "lab" }, "s"),
        el("button", { class: at(A) ? "on" : "", onclick: () => { this.t = A.sec; this.refresh(); } }, "Set start"),
        el("span", { class: "lab" }, "→ End"), num(B.sec, (v) => set(() => { B.sec = +Math.max(v ?? d, A.sec + 0.05).toFixed(3); })),
        el("span", { class: "lab" }, "s"),
        el("button", { class: at(B) ? "on" : "", onclick: () => { this.t = B.sec; this.refresh(); } }, "Set end"),
        el("span", { class: "lab" }, "Easing"), ease));
    }
    motion.append(el("div", { class: "row" },
      el("span", { class: "lab" }, "On at"), num(l.start_sec, (v) => set((x) => { x.start_sec = v; }), "0"),
      el("span", { class: "lab" }, "s, off at"), num(l.end_sec, (v) => set((x) => { x.end_sec = v; }), d.toFixed(1)),
      el("span", { class: "lab" }, "s (blank = the whole clip)")));
    p.append(motion);

    // this light's own words: always shown, and kept as they are typed (a panel rebuilt when a run finishes
    // dropped text that had not been committed on blur yet)
    p.append(el("div", { class: "lab" }, "Describe this light (optional)"),
      el("textarea", { rows: 2, placeholder: "Example: headlights passing by, or stage lights on faces and shoulders",
        oninput: (e) => { l.prompt = e.target.value; this.commit(); } }, l.prompt || ""));
  }

  buildScene() {
    const sc = this.scene;
    sc.innerHTML = "";
    const set = (fn) => { fn(this.rig); this.commit(); };
    // ambient: everything no light on the stage reaches. Both sliders in the middle = the original's own
    // light stays as it is (ambient null)
    const amb = Object.assign({ kelvin: 6500, intensity_ev: 0 }, this.rig.ambient || {});
    const setAmb = (f, v) => set((r) => {
      const a = Object.assign({ kelvin: 6500, intensity_ev: 0 }, r.ambient || {}, { [f]: v });
      r.ambient = Math.abs(a.intensity_ev) < 0.05 && Math.abs(a.kelvin - 6500) < 100 ? null : a;
    });
    const g = el("div", { class: "grid" });
    g.append(...this.slider("Ambient level", amb.intensity_ev, -6, 3, 0.1, (v) => setAmb("intensity_ev", v),
      (v) => (Math.abs(+v) < 0.05 ? "As in clip" : `${(+v) >= 0 ? "+" : ""}${(+v).toFixed(1)} stops`)).nodes,
      ...this.slider("Ambient warm/cool", toWarm(amb.kelvin, 6500), -1, 1, 0.05, (v) => setAmb("kelvin", fromWarm(v, 6500)), warmWord).nodes);
    sc.append(el("div", { class: "title" }, "Overall"), g,
      el("div", { class: "lab", style: { marginTop: "4px" } }, "Prompt: the picture and mood you want, and anything else to change"),
      el("textarea", { rows: 3, placeholder: "Example: remove the green screen and its spill, and make the background a night sea under a black sky",
        oninput: (e) => set((r) => (r.user_prompt = e.target.value)) }, this.rig.user_prompt || ""));
    // looks (black-mist style filters and the like) come later; kept, folded
    const box = el("details", {}, el("summary", {}, "Look (filters, film, grade; not available yet)"));
    if (this._sceneOpen) box.open = true;
    box.addEventListener("toggle", () => (this._sceneOpen = box.open));
    const g2 = el("div", { class: "grid" });
    presets().then((d) => {
      const opts = (o) => [["", "(none)"], ...Object.keys(o).map((k) => [k, k])];
      g2.append(...this.select("Look", this.rig.look.preset, opts(d.looks), (v) => set((r) => (r.look.preset = v))),
        ...this.select("Film", this.rig.look.film, opts(d.films), (v) => set((r) => (r.look.film = v))),
        ...this.select("Grade", this.rig.look.grade, opts(d.grades), (v) => set((r) => (r.look.grade = v))));
    });
    box.append(g2);
    if (SHOW_UNTESTED) sc.append(box);
    if (this.rig.removed_originals.length) {
      sc.append(el("div", { class: "row" }, el("span", { class: "lab" }, "Original lights switched off:"),
        ...this.rig.removed_originals.map((d, i) => el("button", { title: "Restore this original light", onclick: () => {
          const back = JSON.parse(JSON.stringify(d)); back.detected = d; back.original = true;
          this.rig.removed_originals.splice(i, 1); this.rig.lights.push(back); this.sel = this.rig.lights.length - 1;
          this.commit(); this.refresh(); } }, `↺ ${d.name || "Original light"}`))));
    }
    if (this._refOffer) {
      const n = this._refOffer.lights.length;
      sc.append(el("div", { class: "row" }, el("span", { class: "muted" }, this._refReport || ""),
        el("button", { onclick: () => this.applyReference(this._refOffer, true) }, `Apply the reference's lights again (${n})`)));
    }
    if (this._newClipNote) {
      sc.append(el("div", { class: "row" }, el("span", { style: { color: "#ffcc55" } }, this._newClipNote),
        el("span", { class: "muted" }, this._offerReport || "")));
    }
    if (this._offer) {
      const n = this._offer.lights.length;
      sc.append(el("div", { class: "row" }, el("span", { class: "muted" }, this._offerReport || ""),
        el("button", { onclick: () => this.applyDetected(this._offer, true) }, n ? `Apply the detected lights (${n}, replacing the current ones)` : "No clear main light was detected")));
    }
  }

  // Reference Lights ran: put the reference's lights on the stage (original light off, earlier reference lights
  // replaced). Applied automatically when the reference changed; the same reference again is only offered, so
  // edits made to its lights are not undone by the next run.
  applyReference(det, force = false) {
    const lights = (det.lights || []).map((l) => Object.assign({}, l, { reference: true }));
    if (!lights.length) return;
    const sig = JSON.stringify(lights.map((l) => [l.color, l.keys?.[0]?.azimuth, l.keys?.[0]?.elevation, l.keys?.[0]?.intensity_ev]));
    if (!force && this.rig.reference_sig === sig) { this._refOffer = det; this.buildScene(); return; }
    this.rig.lights = this.rig.lights.filter((l) => !l.reference);
    this.rig.lights.forEach((l) => { if (l.original) l.enabled = false; });
    const add = normalizeRig({ lights, duration_sec: this.rig.duration_sec }, this.rig.duration_sec).lights;
    add.forEach((l) => { l.reference = true; });
    this.rig.lights.push(...add);
    if (det.ambient) this.rig.ambient = Object.assign({}, det.ambient);
    this.rig.reference_sig = sig;
    this._refOffer = det;
    this.sel = this.rig.lights.length - add.length;
    this.commit(); this.refresh();
  }

  // Detect Original Light ran: start from it when the stage is empty, otherwise offer it
  applyDetected(det, force = false) {
    const lights = (det.lights || []).map((l) => Object.assign({}, l));
    // only reference lights so far (they may arrive first): the original light still joins, switched off
    const onlyRef = this.rig.lights.length && this.rig.lights.every((l) => l.reference);
    // another clip (Detect Original Light's clip_id changed): its light replaces the old clip's original light at
    // once (lights the user added stay; a new clip kept the old clip's moved original light, 2026-10-29). The same
    // clip only offers the detection and keeps the user's edits: the LM reads one clip a little differently run to
    // run (2 lights, then 3), and comparing readings took that for a new clip and put the original lights back
    // over the user's edits on every run (2026-10-30). A stage saved before clip_id existed takes it as the same clip.
    const newClip = !!det.clip_id && !!this.rig.clip_id && det.clip_id !== this.rig.clip_id;
    if (det.clip_id && !this.rig.clip_id) { this.rig.clip_id = det.clip_id; this.commit(); }
    if (!force && !newClip && this.rig.lights.length && !onlyRef) { this._offer = det; this.buildScene(); return; }
    this._newClipNote = newClip && !force ? "New clip: the original lights now come from this clip. The lights you added are kept." : "";
    const kept = this.rig.lights.filter((l) => !l.original);
    const add = normalizeRig({ lights: lights, duration_sec: this.rig.duration_sec }, this.rig.duration_sec).lights;
    if (kept.some((l) => l.reference)) add.forEach((l) => { l.enabled = false; });
    this.rig.lights = [...add, ...kept];
    this.rig.removed_originals = [];
    if (det.clip_id) this.rig.clip_id = det.clip_id;
    this._offer = null;
    this.sel = 0;
    this.commit(); this.refresh();
  }

  // ---- 3D view (orthographic)
  geom() { const w = this.canvas.width, h = this.canvas.height; return { cx: w / 2, cy: h / 2 + 6, R: Math.min(w, h) * 0.36 }; }

  rot([x, y, z]) {
    const a = this.yaw * DEG, b = this.pitch * DEG;
    const x1 = x * Math.cos(a) + z * Math.sin(a), z1 = -x * Math.sin(a) + z * Math.cos(a);
    return [x1, y * Math.cos(b) - z1 * Math.sin(b), y * Math.sin(b) + z1 * Math.cos(b)];
  }

  unrot([x2, y2, z2]) {
    const a = this.yaw * DEG, b = this.pitch * DEG;
    const y1 = y2 * Math.cos(b) + z2 * Math.sin(b), z1 = -y2 * Math.sin(b) + z2 * Math.cos(b);
    return [x2 * Math.cos(a) - z1 * Math.sin(a), y1, x2 * Math.sin(a) + z1 * Math.cos(a)];
  }

  proj(v, r = 1) {
    const { cx, cy, R } = this.geom();
    const p = this.rot(v.map((c) => c * r));
    return [cx + p[0] * R, cy - p[1] * R, p[2]];
  }

  // mouse -> direction on the sphere; front (toward the viewer) or back hemisphere
  pick(mx, my, back = false, r = 1) {
    const { cx, cy, R } = this.geom();
    let px = (mx - cx) / (R * r), py = -(my - cy) / (R * r);
    const d = px * px + py * py;
    let pz;
    if (d <= 1) pz = Math.sqrt(1 - d) * (back ? -1 : 1);
    else { const n = Math.sqrt(d); px /= n; py /= n; pz = 0; }
    return { dir: this.unrot([px, py, pz]), onSphere: d <= 1 };
  }

  draw() {
    const c = this.canvas;
    const w = Math.max(320, Math.round(c.clientWidth || 640));
    if (c.width !== w) c.width = w;
    const ctx = c.getContext("2d");
    const { cx, cy, R } = this.geom();
    ctx.clearRect(0, 0, c.width, c.height);
    const line = (pts, style, width = 1, dash = []) => {
      ctx.lineWidth = width; ctx.setLineDash(dash);
      for (let i = 1; i < pts.length; i++) {
        const [x0, y0, z0] = pts[i - 1], [x1, y1, z1] = pts[i];
        ctx.strokeStyle = style; ctx.globalAlpha = z0 + z1 >= 0 ? 0.9 : 0.25;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      }
      ctx.globalAlpha = 1; ctx.setLineDash([]);
    };
    // sphere: outline, meridians every 30 deg, latitudes (eye level brighter)
    ctx.strokeStyle = "#3a4050"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
    for (let az = 0; az < 360; az += 30) line(Array.from({ length: 37 }, (_, i) => this.proj(dirOf(az, -90 + i * 5))), "#39404c");
    for (const e of [-30, 30, 60]) line(Array.from({ length: 73 }, (_, i) => this.proj(dirOf(i * 5, e))), "#39404c");
    line(Array.from({ length: 73 }, (_, i) => this.proj(dirOf(i * 5, 0))), "#5d6b85", 1.5);
    // direction labels
    ctx.font = "12px system-ui"; ctx.textAlign = "center";
    for (const [az, elv, t] of [[0, 0, "Front"], [90, 0, "Right"], [-90, 0, "Left"], [180, 0, "Back"], [0, 90, "Top"]]) {
      const [x, y, z] = this.proj(dirOf(az, elv), 1.13);
      ctx.fillStyle = z >= 0 ? "#9aa6bd" : "#55607a"; ctx.fillText(t, x, y + 4);
    }
    // camera, in front of the subject
    {
      const [x, y] = this.proj([0, 0, 1], 1.32), [x2, y2] = this.proj([0, 0, 1], 1.1);
      ctx.fillStyle = "#8fb3d9"; ctx.fillRect(x - 11, y - 7, 22, 14);
      ctx.beginPath(); ctx.moveTo(x2 + (x - x2) * 0.25, y2 + (y - y2) * 0.25); ctx.lineTo(x - 6, y); ctx.lineTo(x + 6, y); ctx.fill();
      ctx.fillStyle = "#8fb3d9"; ctx.fillText("Camera", x, y + 22);
    }
    // lights: back side first, so the front ones draw on top; the subject sits between the two halves
    const order = this.rig.lights.map((l, i) => {
      const s = evalLight(l, this.t);
      return [i, this.proj(dirOf(s.azimuth, s.elevation))[2]];
    }).sort((a, b) => a[1] - b[1]);
    this.hits = [];
    for (const [i, z] of order) if (z < 0) this.drawLight(ctx, i);
    if (this.clip?.ready) this.drawClipCard(ctx);
    else {
      // no clip yet: a head facing the camera
      ctx.fillStyle = "#666d7a"; ctx.beginPath(); ctx.arc(cx, cy, R * 0.11, 0, 7); ctx.fill();
      const [nx, ny] = this.proj([0, 0, 1], 0.2);
      ctx.strokeStyle = "#c8ccd4"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(nx, ny); ctx.stroke();
      ctx.fillStyle = "#aaa"; ctx.fillText("Subject", cx, cy + R * 0.11 + 14);
    }
    for (const [i, z] of order) if (z >= 0) this.drawLight(ctx, i);
    ctx.textAlign = "left";
  }

  // the clip's frame at the playhead, standing at the centre and facing the camera: its right is the
  // sphere's right, its top the sphere's top, so a light is placed against the picture itself
  drawClipCard(ctx) {
    const { info, img } = this.clip;
    const f = clamp(Math.round(this.t * this.fps), 0, info.frames - 1);
    const tile = clamp(Math.floor(f / info.step), 0, info.count - 1);
    const sx = (tile % info.cols) * info.tile_w, sy = Math.floor(tile / info.cols) * info.tile_h;
    const hw = 0.42, hh = hw * info.height / info.width;            // half size, in sphere radii
    const [ox, oy] = this.proj([0, 0, 0]);
    const [rx, ry] = this.proj([hw, 0, 0]), [ux, uy] = this.proj([0, hh, 0]);
    const X = [rx - ox, ry - oy], Y = [ux - ox, uy - oy];           // screen vectors of the card's right / up
    const facing = this.rot([0, 0, 1])[2] >= 0;
    const corners = [[-1, 1], [1, 1], [1, -1], [-1, -1]].map(([a, b]) => [ox + a * X[0] + b * Y[0], oy + a * X[1] + b * Y[1]]);
    ctx.save();
    // image pixel (u, v) -> card: u across the right vector, v down the up vector
    const tw = info.tile_w, th = info.tile_h;
    ctx.setTransform(2 * X[0] / tw, 2 * X[1] / tw, -2 * Y[0] / th, -2 * Y[1] / th, ox - X[0] + Y[0], oy - X[1] + Y[1]);
    ctx.globalAlpha = facing ? 1 : 0.35;                              // seen from behind: dim (and mirrored)
    ctx.drawImage(img, sx, sy, tw, th, 0, 0, tw, th);
    ctx.restore();
    ctx.strokeStyle = facing ? "#c8ccd4" : "#555c68"; ctx.lineWidth = 1.5;
    ctx.beginPath(); corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.stroke();
    const bottom = Math.max(...corners.map((c) => c[1]));
    ctx.fillStyle = "#aaa"; ctx.textAlign = "center";
    ctx.fillText(facing ? `Clip · frame ${f}` : "Clip (back)", ox, bottom + 14);
  }

  // a panel's size: set by its handle, else what its softness means at its distance
  areaSize(l, st) { return l.size ?? sizeOf(l.softness, st.distance); }

  drawLight(ctx, i) {
    const l = this.rig.lights[i];
    const sel = i === this.sel;
    const { cx, cy } = this.geom();
    const s = evalLight(l, this.t);
    const col = lightCSS(l, s.kelvin);
    const fade = l.enabled && lightOn(l, this.t) ? 1 : 0.35;   // switched off, or not on at this time
    const rad = (e) => (l.type === "directional" ? SUN_R : radiusOf(e.distance));
    // motion path and keyframes, each at its own distance
    if (l.keys.length > 1) {
      const d = this.rig.duration_sec;
      const pts = Array.from({ length: 61 }, (_, j) => { const e = evalLight(l, (d * j) / 60); return this.proj(dirOf(e.azimuth, e.elevation), rad(e)); });
      ctx.lineWidth = sel ? 2.5 : 1.5; ctx.setLineDash([6, 4]);
      for (let j = 1; j < pts.length; j++) {
        ctx.strokeStyle = sel ? "#ffcc55" : col; ctx.globalAlpha = (pts[j][2] >= 0 ? 0.95 : 0.3) * fade;
        ctx.beginPath(); ctx.moveTo(pts[j - 1][0], pts[j - 1][1]); ctx.lineTo(pts[j][0], pts[j][1]); ctx.stroke();
      }
      ctx.setLineDash([]);
      for (const k of l.keys) {
        const [x, y, z] = this.proj(dirOf(k.azimuth, k.elevation), rad(k));
        ctx.globalAlpha = (z >= 0 ? 1 : 0.4) * fade;
        ctx.fillStyle = sel ? "#ffcc55" : "#bbb"; ctx.beginPath();
        ctx.moveTo(x, y - 7); ctx.lineTo(x + 7, y); ctx.lineTo(x, y + 7); ctx.lineTo(x - 7, y); ctx.fill();
        if (sel) { ctx.fillStyle = "#ffe9a8"; ctx.font = "11px system-ui"; ctx.fillText(`${k.sec.toFixed(1)}s`, x, y - 10); }
        this.hits.push({ kind: "key", light: i, key: k, x, y, z });
      }
    }
    if (l.original && l.detected?.keys?.[0]) {
      const d0 = l.detected.keys[0];
      const [gx, gy, gz] = this.proj(dirOf(d0.azimuth, d0.elevation), radiusOf(d0.distance ?? 1.5));
      ctx.globalAlpha = (gz >= 0 ? 0.8 : 0.3) * fade;
      ctx.strokeStyle = "#9fd6a0"; ctx.setLineDash([3, 3]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(gx, gy, 9, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "#9fd6a0"; ctx.font = "10px system-ui"; ctx.fillText("Original position", gx, gy + 20);
    }
    // the light at the playhead and what it throws toward the subject
    const dir = dirOf(s.azimuth, s.elevation), r = rad(s);
    const c3 = dir.map((q) => q * r);
    const [x, y, z] = this.proj(dir, r);
    const front = z >= 0, vis = (front ? 1 : 0.45) * fade;
    const ang = Math.atan2(cy - y, cx - x), len = Math.hypot(cx - x, cy - y);
    const size = l.type === "area" ? this.areaSize(l, s) : 0;
    ctx.globalAlpha = (front ? 0.18 : 0.07) * fade; ctx.fillStyle = col; ctx.strokeStyle = col;
    if (l.type === "directional") {          // parallel rays
      ctx.lineWidth = 2;
      for (const o of [-10, 0, 10]) { const ox = -Math.sin(ang) * o, oy = Math.cos(ang) * o; ctx.beginPath(); ctx.moveTo(x + ox, y + oy); ctx.lineTo(cx + ox, cy + oy); ctx.stroke(); }
    } else {
      const half = l.type === "spot" ? (l.cone.angle / 2) * DEG
        : l.type === "area" ? Math.min(1.1, Math.atan(size / 2 / s.distance) + 0.1) : 0.5;
      ctx.beginPath(); ctx.moveTo(x, y);
      ctx.lineTo(x + len * Math.cos(ang - half), y + len * Math.sin(ang - half));
      ctx.lineTo(x + len * Math.cos(ang + half), y + len * Math.sin(ang + half)); ctx.fill();
    }
    // the fixture itself
    ctx.globalAlpha = vis;
    let reach = 10;                          // label offset from the light's centre
    if (l.type === "area") {
      const [u, v] = panelAxes(dir);
      const hs = (size / 2) * (r / s.distance);
      const at = (a, b) => this.proj(c3.map((q, m) => q + (u[m] * a + v[m] * b) * hs));
      const pts = l.shape === "disc"
        ? Array.from({ length: 28 }, (_, j) => { const t = (j / 28) * 2 * Math.PI; return at(Math.cos(t), Math.sin(t)); })
        : [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => at(a, b));
      ctx.fillStyle = col; ctx.globalAlpha = vis * 0.55;   // see-through, so a big near panel does not hide the subject
      ctx.beginPath(); pts.forEach(([px, py], j) => (j ? ctx.lineTo(px, py) : ctx.moveTo(px, py))); ctx.closePath(); ctx.fill();
      ctx.globalAlpha = vis; ctx.strokeStyle = sel ? "#ffcc55" : "#f2f2f2"; ctx.lineWidth = sel ? 2 : 1; ctx.stroke();
      reach = Math.max(10, ...pts.map(([px, py]) => Math.hypot(px - x, py - y)));
      if (sel) {                              // size handle on the panel's edge
        const [hx, hy] = at(1, 0);
        ctx.fillStyle = "#ffcc55"; ctx.fillRect(hx - 5, hy - 5, 10, 10);
        this.hits.push({ kind: "size", light: i, x: hx, y: hy, z, c3, u, r, d: s.distance });
      }
    } else if (l.type === "spot") {
      const g = ctx.createRadialGradient(x, y, 1, x, y, 13);
      g.addColorStop(0, "#fff"); g.addColorStop(0.4, col); g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 13, 0, 7); ctx.fill();
      ctx.strokeStyle = sel ? "#ffcc55" : "#ddd"; ctx.lineWidth = 2;   // the lamp's barrel, pointing at the subject
      ctx.beginPath(); ctx.moveTo(x - Math.cos(ang) * 9, y - Math.sin(ang) * 9); ctx.lineTo(x + Math.cos(ang) * 9, y + Math.sin(ang) * 9); ctx.stroke();
      if (sel) {                              // beam-angle handle where the cone's edge reaches the subject
        const half = (l.cone.angle / 2) * DEG, e = len * Math.tan(half);
        const hx = cx - Math.sin(ang) * e, hy = cy + Math.cos(ang) * e;
        ctx.strokeStyle = "#ffcc55"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(hx, hy); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = "#ffcc55"; ctx.beginPath(); ctx.arc(hx, hy, 6, 0, 7); ctx.fill();
        this.hits.push({ kind: "beam", light: i, x: hx, y: hy, z, lx: x, ly: y });
      }
    } else if (l.type === "point") {         // a bulb: bigger = softer
      const b = 6 + 10 * l.softness;
      const g = ctx.createRadialGradient(x, y, 1, x, y, b * 1.8);
      g.addColorStop(0, "#fff"); g.addColorStop(0.35, col); g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, b * 1.8, 0, 7); ctx.fill();
      reach = b + 4;
    } else {                                  // the sun
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, 8, 0, 7); ctx.fill();
      ctx.strokeStyle = col; ctx.lineWidth = 2;
      for (let j = 0; j < 8; j++) {
        const t = (j / 8) * 2 * Math.PI;
        ctx.beginPath(); ctx.moveTo(x + Math.cos(t) * 11, y + Math.sin(t) * 11); ctx.lineTo(x + Math.cos(t) * 16, y + Math.sin(t) * 16); ctx.stroke();
      }
      reach = 16;
    }
    if (sel && l.type !== "area") { ctx.strokeStyle = "#ffcc55"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, Math.min(reach, 18) + 3, 0, 7); ctx.stroke(); }
    ctx.fillStyle = "#e6e6e6"; ctx.font = "12px system-ui"; ctx.textAlign = "left";
    ctx.fillText(`${l.name}${l.original && !l.name.includes("Original") ? " (original)" : ""}${l.motion?.type ? " ⚡" : ""}`, x + reach + 6, y - 4);
    const detail = l.type === "area" ? ` · ${size.toFixed(1)}m` : l.type === "spot" ? ` · ${l.cone.angle.toFixed(0)}°` : "";
    const far = l.type === "directional" ? "" : ` · dist ${s.distance.toFixed(1)}`;
    ctx.fillStyle = "#9aa"; ctx.font = "10px system-ui";
    ctx.fillText(`${s.azimuth.toFixed(0)}° / ${s.elevation.toFixed(0)}° · ${s.intensity_ev.toFixed(1)}EV${detail}${far}`, x + reach + 6, y + 9);
    ctx.textAlign = "center"; ctx.globalAlpha = 1;
    this.hits.push({ kind: "light", light: i, x, y, z, r });
    // near / far handle on the line to the subject (the sun has no distance)
    if (sel && l.type !== "directional") {
      const [hx, hy] = this.proj(dir, r * 0.55);
      ctx.globalAlpha = vis; ctx.strokeStyle = "#ffd84a"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "#ffd84a"; ctx.beginPath(); ctx.arc(hx, hy, 6, 0, 7); ctx.fill();
      ctx.globalAlpha = 1;
      this.hits.push({ kind: "dist", light: i, x: hx, y: hy, z, dir });
    }
  }

  drawStrip() {
    const c = this.strip, ctx = c.getContext("2d");
    c.width = Math.max(200, Math.round(c.clientWidth || 600));
    ctx.clearRect(0, 0, c.width, c.height);
    const d = this.rig.duration_sec, X = (s) => (s / d) * (c.width - 10) + 5;
    ctx.fillStyle = "#555";
    for (let s = 0; s <= d + 1e-6; s += 0.5) ctx.fillRect(X(s), s % 1 ? 13 : 9, 1, 10);
    this.rig.lights.forEach((l, i) => {
      if (l.keys.length < 2 && i !== this.sel) return;
      for (const k of l.keys) {
        const x = X(k.sec);
        ctx.fillStyle = i === this.sel ? "#ffcc55" : "#777";
        ctx.beginPath(); ctx.moveTo(x, 3); ctx.lineTo(x + 6, 10); ctx.lineTo(x, 17); ctx.lineTo(x - 6, 10); ctx.fill();
      }
    });
    ctx.fillStyle = "#4a8bff"; ctx.fillRect(X(this.t) - 1, 0, 2, c.height);
  }

  bindCanvas() {
    const c = this.canvas;
    const pos = (e) => { const r = c.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * c.width, ((e.clientY - r.top) / r.height) * c.height]; };
    const hitAt = (x, y, only) => {
      let best = null, bd = 16;
      for (const h of [...(this.hits || [])].reverse()) {        // topmost first
        if (only && h.kind !== only) continue;
        const d = Math.hypot(h.x - x, h.y - y) - (h.kind === "light" ? 6 : 0);
        if (d < bd) { bd = d; best = h; }
      }
      return best;
    };
    let drag = null;
    c.addEventListener("contextmenu", (e) => {
      const h = hitAt(...pos(e), "key");                       // a key under its light is still the target
      if (h) { e.preventDefault(); e.stopPropagation(); this.sel = h.light; this.delKey(h.key); }
    });
    c.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.stopPropagation(); e.preventDefault();
      const [x, y] = pos(e);
      const h = hitAt(x, y);
      try { c.setPointerCapture(e.pointerId); } catch (err) { /* synthetic or already released pointer */ }
      if (h) {
        this.sel = h.light;
        if (h.kind === "key") { this.t = h.key.sec; }
        // Shift + drag on a light = nearer / farther, like its yellow handle
        const kind = h.kind === "light" && e.shiftKey && this.light().type !== "directional" ? "dist" : h.kind;
        drag = Object.assign({}, h, { kind, back: h.z < 0, moved: false, frac: h.kind === "light" ? 1 : 0.55 });
        if (kind === "dist" && !drag.dir) { const st = evalLight(this.light(), this.t); drag.dir = dirOf(st.azimuth, st.elevation); }
        this.refresh();
      } else drag = { kind: "view", x, y, yaw: this.yaw, pitch: this.pitch };
      c.style.cursor = "grabbing";
    });
    c.addEventListener("pointermove", (e) => {
      if (!drag) return;
      e.stopPropagation();
      const [x, y] = pos(e);
      if (drag.kind === "view") {
        this.yaw = drag.yaw + (x - drag.x) * 0.5;
        this.pitch = clamp(drag.pitch + (y - drag.y) * 0.4, -89, 89);
        this.draw();
        return;
      }
      const l = this.light();
      const { cx, cy, R } = this.geom();
      if (drag.kind === "dist") {
        // along the light's own line: where the mouse falls on the projected line from the subject to the light
        const [px, py] = this.proj(drag.dir, 1), vx = px - cx, vy = py - cy, n2 = vx * vx + vy * vy;
        if (n2 < (0.12 * R) ** 2) return;      // light pointing at the viewer: no line to drag along (use the wheel)
        const along = ((x - cx) * vx + (y - cy) * vy) / n2;
        const k = (drag.k = drag.k || this.editKey(l));
        if (l.type === "area" && l.size == null) l.size = this.areaSize(l, k);
        k.distance = +distOf(along / drag.frac).toFixed(2);
        if (l.type === "area") l.softness = +softnessOf(l.size, k.distance).toFixed(3);
      } else if (drag.kind === "size") {
        // the panel's half size: the mouse on the projected axis across it
        const [px, py] = this.proj(drag.c3), [qx, qy] = this.proj(drag.c3.map((q, m) => q + drag.u[m]));
        const vx = qx - px, vy = qy - py, n2 = vx * vx + vy * vy;
        if (n2 < 4) return;
        const hs = Math.max(0.01, ((x - px) * vx + (y - py) * vy) / n2);
        l.size = +clamp((2 * hs * drag.d) / drag.r, 0.1, 2.5).toFixed(2);
        l.softness = +softnessOf(l.size, drag.d).toFixed(3);
      } else if (drag.kind === "beam") {
        // half the beam = the angle at the lamp between the subject and the mouse
        const off = Math.abs((x - cx) * (drag.ly - cy) - (y - cy) * (drag.lx - cx)) / (Math.hypot(drag.lx - cx, drag.ly - cy) || 1);
        const len = Math.hypot(cx - drag.lx, cy - drag.ly) || 1;
        l.cone.angle = Math.round(clamp((2 * Math.atan(off / len)) / DEG, 5, 150));
      } else {
        const r = l.type === "directional" ? SUN_R : radiusOf((drag.kind === "key" ? drag.key : evalLight(l, this.t)).distance);
        const [az, elv] = angOf(this.pick(x, y, drag.back, r).dir).map((v) => Math.round(v));
        const k = drag.kind === "key" ? drag.key : (drag.k = drag.k || this.editKey(l));
        k.azimuth = az; k.elevation = clamp(elv, -60, 90);
      }
      drag.moved = true;
      this.draw(); this.updateLive();
    });
    const up = () => {
      if (!drag) return;
      if (drag.moved) { this.commit(); this.refresh(); }
      drag = null; c.style.cursor = "grab";
    };
    c.addEventListener("pointerup", up);
    c.addEventListener("pointercancel", up);
    c.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      const [x, y] = pos(e);
      const p = this.pick(x, y);
      if (!p.onSphere) { this.yaw = 55; this.pitch = 35; this.draw(); return; }
      const [az, elv] = angOf(p.dir);
      this.addLight("area", Math.round(az), clamp(Math.round(elv), -60, 90));
    });
    c.addEventListener("wheel", (e) => {
      const h = hitAt(...pos(e));
      if (h?.kind !== "light") return;
      e.preventDefault(); e.stopPropagation();
      this.sel = h.light;
      const l = this.light();
      const cur = evalLight(l, this.t).distance;
      if (l.type === "area" && l.size == null) l.size = this.areaSize(l, evalLight(l, this.t));
      const d = +clamp(cur + (e.deltaY < 0 ? 0.1 : -0.1), DIST_MIN, DIST_MAX).toFixed(2);
      if (l.type === "area") l.softness = +softnessOf(l.size, d).toFixed(3);
      this.setAnim("distance", d);
    }, { passive: false });
  }
}

// --------------------------------------------------------------------------- registration
// Reference Lights ran: the light stage it feeds takes the reference's lights
api.addEventListener("executed", ({ detail }) => {
  const text = detail?.output?.openrelight_reference?.[0];
  if (!text || !app.graph) return;
  let det;
  try { det = JSON.parse(text); } catch (e) { return; }
  for (const n of app.graph._nodes || []) {
    if (n.type !== NODE || !n._lightstage) continue;
    n._lightstage._refReport = detail.output.text?.[0] || "";
    n._lightstage.applyReference(det, false);
  }
});

// Source Clip ran: every light stage in the graph shows that clip (a workflow relights one clip)
api.addEventListener("executed", ({ detail }) => {
  const text = detail?.output?.openrelight_clip?.[0];
  if (!text || !app.graph) return;
  let info;
  try { info = JSON.parse(text); } catch (e) { return; }
  for (const n of app.graph._nodes || []) if (n.type === NODE && n._lightstage) n._lightstage.setClip(info);
});

api.addEventListener("executed", ({ detail }) => {
  const text = detail?.output?.openrelight_rig?.[0];
  if (!text || !app.graph) return;
  let det;
  try { det = JSON.parse(text); } catch (e) { return; }
  const src = String(detail.node).split(":").pop();
  for (const n of app.graph._nodes || []) {
    if (n.type !== NODE || !n._lightstage) continue;
    const inp = n.inputs?.find((i) => i.name === "detected");
    const link = inp?.link != null ? app.graph.links?.[inp.link] ?? app.graph.links?.get?.(inp.link) : null;
    if (!link || String(link.origin_id) !== src) continue;
    const ed = n._lightstage;
    ed._offerReport = detail.output.text?.[0] || "";
    // the timeline follows the analysed clip's length (keys are stored in seconds, so they stay put)
    if (!ed.clip && det.duration_sec > 0 && Math.abs(det.duration_sec - ed.rig.duration_sec) > 1e-3) {
      ed.rig.duration_sec = det.duration_sec;
      ed.t = Math.min(ed.t || 0, det.duration_sec);
      ed.commit(); ed.refresh();
    }
    ed.applyDetected(det, false);
  }
});

// The timeline follows the clip as soon as it is trimmed, before anything runs: Trim Video's length (0 = the whole
// clip from its start, read from the file's own metadata), capped by Source Clip's max_seconds. Source Clip's report
// after a run sets the exact frame count. Keys keep their times, so a recorded move stays on the same moment of the
// footage; keys past the new end are kept, not shown (the user trimmed a 4 s clip to 2 s and had to set the stage's
// length by hand, 2026-10-30).
const fileSeconds = {};
function videoSeconds(file) {
  if (!file) return 0;
  if (!(file in fileSeconds)) {
    fileSeconds[file] = 0;
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => { fileSeconds[file] = v.duration || 0; v.removeAttribute("src"); v.load(); };
    v.src = api.apiURL(`/view?type=input&filename=${encodeURIComponent(file)}&subfolder=`);
  }
  return fileSeconds[file];
}

function trimmedSeconds(graph) {
  const find = (t) => (graph?._nodes || []).find((n) => n.type === t && n.mode === 0);
  const trim = find("VideoTrim"), load = find("LoadVideo"), sc = find("OR_SourceClip");
  if (!load) return 0;
  const tv = (trim && getW(trim, "trim")?.value?.trim) || {};
  let d = +tv.duration > 0 ? +tv.duration : videoSeconds(getW(load, "file")?.value) - (+tv.start_time || 0);
  const cap = +((sc && getW(sc, "max_seconds")?.value) ?? 0);
  if (cap > 0) d = Math.min(d, cap);
  return d > 0 ? d : 0;
}

setInterval(() => {
  for (const n of app.graph?._nodes || []) {
    const ed = n.type === NODE && n._lightstage;
    if (!ed) continue;
    const d = trimmedSeconds(n.graph || app.graph);
    if (!d || Math.abs(d - (ed._trimmed ?? -1)) < 1e-3) continue;   // only when the trim itself changes
    ed._trimmed = d;
    const sec = Math.round(d * ed.fps) / ed.fps;
    if (Math.abs(sec - ed.rig.duration_sec) < 0.5 / ed.fps) continue;
    ed.rig.duration_sec = +sec.toFixed(3);
    ed.t = Math.min(ed.t || 0, ed.rig.duration_sec);
    ed.commit(); ed.refresh();
  }
}, 1000);

// After every run its OpenRelight results are read back from the history and handed to the listeners above as if
// they had just arrived. After a page reload that reuses the client id, ComfyUI can drop this page's socket and the
// page then gets no node results, so the stage missed a trimmed clip's new length and the detected light
// (2026-10-30); "status" goes to every page and still arrives. A result handled twice changes nothing.
let queueBusy = false;
api.addEventListener("status", async ({ detail }) => {
  const left = detail?.exec_info?.queue_remaining ?? 0;
  if (left > 0) { queueBusy = true; return; }
  if (!queueBusy) return;
  queueBusy = false;
  let h = {};
  try { h = await (await api.fetchApi("/history?max_items=1")).json(); } catch (e) { return; }
  for (const [promptId, run] of Object.entries(h || {})) {
    for (const [node, out] of Object.entries(run?.outputs || {})) {
      if (!Object.keys(out).some((k) => k.startsWith("openrelight_"))) continue;
      const output = Object.fromEntries(Object.entries(out).filter(([k]) => k.startsWith("openrelight_") || k === "text"));
      api.dispatchEvent(new CustomEvent("executed", { detail: { node, display_node: node, output, prompt_id: promptId } }));
    }
  }
});

app.registerExtension({
  name: "OpenRelight.LightStage",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const node = this;
      if (node.widgets?.some((w) => w.name === "lightstage")) return r;
      const root = document.createElement("div");
      root.style.width = "100%";
      const H = 760;
      const w = node.addDOMWidget("lightstage", "openrelightLightStage", root, {
        serialize: false, hideOnZoom: false, getMinHeight: () => H, getMaxHeight: () => H, getHeight: () => H,
      });
      if (w) w.serialize = false;
      node.size = [Math.max(node.size?.[0] || 0, 640), node.size?.[1] || 0];
      const ed = new LightStage(node, root);
      node._lightstage = ed;
      // the raw JSON stays reachable but small; the editor is the way to change it
      const jw = getW(node, "rig_json");
      if (jw?.inputEl) { jw.inputEl.style.maxHeight = "48px"; jw.inputEl.style.fontSize = "10px"; }
      // Nodes 2.0 reads options.hidden, the older canvas reads hidden
      if (jw) { jw.hidden = true; jw.options = Object.assign(jw.options || {}, { hidden: true }); }
      const pw = getW(node, "preset");
      if (pw) {
        pw.hidden = !SHOW_UNTESTED;
        pw.options = Object.assign(pw.options || {}, { hidden: !SHOW_UNTESTED });
        const cb = pw.callback;
        pw.callback = function () { const rr = cb?.apply(this, arguments); if (!ed._settingPreset) ed.loadFromNode(); return rr; };
      }
      ed.loadFromNode();
      return r;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      this._lightstage?.loadFromNode();
      if (this.properties?.openrelight_clip) this._lightstage?.setClip(this.properties.openrelight_clip, false);
      return r;
    };
  },
});
