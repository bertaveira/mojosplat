"""The viewer's web page (served at /). Plain HTML + JS, no dependencies.

The page keeps the camera (camera-to-world rotation with OpenCV axes:
right, down, forward; position; orbit distance; world up) and sends the
world-to-camera matrix over the WebSocket whenever it changes:
    cam W H fy r00 r01 r02 t0 r10 r11 r12 t1 r20 r21 r22 t2
plus `pause`, `resume` and `view K`. The server answers with JPEG frames
(binary messages) and JSON text messages: init, stats, view.
(No backslashes below: this is a Mojo string literal.)
"""

comptime PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MojoSplat viewer</title>
<style>
:root { color-scheme: dark; }
html, body { margin: 0; height: 100%; overflow: hidden; background: #0d0d10; color: #d8d8de;
  font: 13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
canvas { position: fixed; inset: 0; width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }
canvas.drag { cursor: grabbing; }
#panel { position: fixed; top: 12px; left: 12px; width: 250px; padding: 12px 14px; border-radius: 10px;
  background: rgba(22, 22, 28, 0.84); border: 1px solid #2c2c35; backdrop-filter: blur(8px); user-select: none; }
h1 { margin: 0 0 8px; font-size: 14px; font-weight: 650; letter-spacing: 0.2px; }
h1 span { color: #8a8a96; font-weight: 400; }
.row { display: flex; justify-content: space-between; gap: 12px; }
.row span { color: #9a9aa6; }
.row b { color: #fff; font-weight: 500; font-variant-numeric: tabular-nums; }
.bar { height: 4px; margin: 8px 0 6px; border-radius: 2px; background: #2c2c35; overflow: hidden; }
.bar div { height: 100%; width: 0; background: #4cc2ff; transition: width 0.2s; }
.btns { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
button, select { font: inherit; color: #eee; background: #2a2a32; border: 1px solid #3c3c46; border-radius: 6px;
  padding: 4px 9px; cursor: pointer; }
button:hover, select:hover { background: #34343e; }
button.on { background: #4cc2ff; color: #000; border-color: #4cc2ff; }
#status { margin-top: 8px; color: #8a8a96; font-size: 12px; }
#help { position: fixed; bottom: 10px; left: 12px; padding: 3px 8px; border-radius: 6px; color: #a0a0aa;
  background: rgba(22, 22, 28, 0.7); font-size: 12px; user-select: none; }
[hidden] { display: none !important; }
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<div id="panel">
  <h1>MojoSplat <span id="title"></span></h1>
  <div class="bar" id="bar" hidden><div></div></div>
  <div id="stats"></div>
  <div class="row"><span>Viewer</span><b id="fps">-</b></div>
  <div class="btns">
    <button id="pause" hidden>Pause</button>
    <button id="prev" hidden title="Previous dataset camera">&#9664; View</button>
    <button id="next" hidden title="Next dataset camera">View &#9654;</button>
    <button id="reset" title="Back to the initial camera">Reset</button>
    <button id="up" title="Use the current camera's up direction for orbiting">Set up</button>
    <select id="res" title="Render resolution">
      <option value="1">Full res</option><option value="0.5">1/2 res</option><option value="0.25">1/4 res</option>
    </select>
  </div>
  <div id="status">Connecting...</div>
</div>
<div id="help">Drag: orbit &middot; Right-drag / Shift-drag: pan &middot; Scroll: zoom &middot; WASD QE: move</div>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const canvas = $("canvas"), g = canvas.getContext("2d");

// --- small vector helpers ---------------------------------------------------
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => mul(a, 1 / Math.hypot(a[0], a[1], a[2]));
function rot(v, k, t) {  // Rodrigues: rotate v about unit axis k by t radians
  const c = Math.cos(t), s = Math.sin(t);
  return add(add(mul(v, c), mul(cross(k, v), s)), mul(k, dot(k, v) * (1 - c)));
}

// --- camera: OpenCV axes (right, down, forward) in world coordinates ---------
let right = [1, 0, 0], down = [0, 1, 0], fwd = [0, 0, 1], pos = [0, 0, -3];
let dist = 3, up = [0, -1, 0], fyRatio = 1, initial = null, maxSize = 1920;
let changed = true, nViews = 0, viewIndex = 0, training = false, paused = false;

function apply(v) {  // v: {rot: 9 floats (camera-to-world, row-major), pos, fy, dist?}
  right = [v.rot[0], v.rot[3], v.rot[6]];
  down = [v.rot[1], v.rot[4], v.rot[7]];
  fwd = [v.rot[2], v.rot[5], v.rot[8]];
  pos = v.pos.slice();
  fyRatio = v.fy;
  if (v.dist) dist = v.dist;
  changed = true;
}
function reorthonormalize() {
  fwd = unit(fwd); right = unit(cross(down, fwd)); down = cross(fwd, right);
}
function target() { return add(pos, mul(fwd, dist)); }

function orbit(dx, dy) {
  const t = target(), k = 0.005;
  let off = sub(pos, t);
  const yaw = -dx * k;
  right = rot(right, up, yaw); down = rot(down, up, yaw); fwd = rot(fwd, up, yaw); off = rot(off, up, yaw);
  const pitch = -dy * k, f2 = rot(fwd, right, pitch);
  if (Math.abs(dot(f2, up)) < 0.985) {
    down = rot(down, right, pitch); fwd = f2; off = rot(off, right, pitch);
  }
  reorthonormalize();
  pos = add(t, off);
  changed = true;
}
function pan(dx, dy) {
  const s = dist / (fyRatio * canvas.clientHeight);
  pos = sub(pos, add(mul(right, dx * s), mul(down, dy * s)));
  changed = true;
}
function zoom(amount) {
  const t = target();
  dist = Math.max(dist * Math.exp(amount), 1e-3);
  pos = sub(t, mul(fwd, dist));
  changed = true;
}

// --- input -------------------------------------------------------------------
let dragging = null, lastX = 0, lastY = 0;
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  dragging = (e.button === 2 || e.shiftKey) ? "pan" : "orbit";
  lastX = e.clientX; lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId); canvas.classList.add("drag");
});
canvas.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  if (dragging === "pan") pan(dx, dy); else orbit(dx, dy);
});
const endDrag = () => { dragging = null; canvas.classList.remove("drag"); };
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoom(e.deltaY * 0.0012); }, { passive: false });
const keys = new Set();
addEventListener("keydown", (e) => { if (!e.target.closest("select")) keys.add(e.key.toLowerCase()); });
addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
addEventListener("blur", () => keys.clear());
addEventListener("resize", () => { changed = true; });
$("res").addEventListener("change", () => { changed = true; });
$("reset").onclick = () => { if (initial) apply(initial); };
$("up").onclick = () => { up = mul(down, -1); };
$("pause").onclick = () => send(paused ? "resume" : "pause");
$("prev").onclick = () => { viewIndex = (viewIndex + nViews - 1) % nViews; send("view " + viewIndex); };
$("next").onclick = () => { viewIndex = (viewIndex + 1) % nViews; send("view " + viewIndex); };

// --- connection ----------------------------------------------------------------
let ws = null, frames = 0, lastFps = performance.now(), decoding = false, pendingBlob = null;
function send(msg) { if (ws && ws.readyState === 1) ws.send(msg); }
function connect() {
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.binaryType = "blob";
  ws.onopen = () => { $("status").textContent = "Connected"; changed = true; };
  ws.onclose = () => { $("status").textContent = "Disconnected, retrying..."; setTimeout(connect, 1000); };
  ws.onmessage = (e) => {
    if (typeof e.data === "string") return onMessage(JSON.parse(e.data));
    pendingBlob = e.data;
    if (!decoding) drawLatest();
  };
}
async function drawLatest() {
  decoding = true;
  while (pendingBlob) {
    const blob = pendingBlob; pendingBlob = null;
    try {
      const img = await createImageBitmap(blob);
      g.drawImage(img, 0, 0, canvas.width, canvas.height);
      img.close();
      frames++;
    } catch (err) { console.warn(err); }
  }
  decoding = false;
}
function onMessage(m) {
  if (m.type === "init") {
    $("title").textContent = m.title || "";
    training = m.training; nViews = m.views; maxSize = m.max_size;
    up = m.up; initial = m.view; apply(m.view);
    $("pause").hidden = !training;
    $("prev").hidden = $("next").hidden = nViews === 0;
  } else if (m.type === "view") {
    apply(m.view);
  } else if (m.type === "stats") {
    paused = m.paused;
    const pb = $("pause");
    pb.textContent = paused ? "Resume" : "Pause";
    pb.classList.toggle("on", paused);
    $("bar").hidden = !(m.progress >= 0);
    $("bar").firstElementChild.style.width = (100 * Math.max(0, m.progress)) + "%";
    $("stats").innerHTML = m.items.map(([k, v]) => '<div class="row"><span>' + k + "</span><b>" + v + "</b></div>").join("");
  }
}

// --- render loop: move with keys, resize, send the camera when it changes ---------
let lastT = performance.now();
function frame(t) {
  const dt = Math.min((t - lastT) / 1000, 0.1); lastT = t;
  const v = dist * 0.8 * dt;
  if (keys.has("w")) { pos = add(pos, mul(fwd, v)); changed = true; }
  if (keys.has("s")) { pos = sub(pos, mul(fwd, v)); changed = true; }
  if (keys.has("d")) { pos = add(pos, mul(right, v)); changed = true; }
  if (keys.has("a")) { pos = sub(pos, mul(right, v)); changed = true; }
  if (keys.has("e")) { pos = add(pos, mul(up, v)); changed = true; }
  if (keys.has("q")) { pos = sub(pos, mul(up, v)); changed = true; }
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const cw = Math.round(canvas.clientWidth * dpr), ch = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; changed = true; }
  if (changed && ws && ws.readyState === 1) {
    const s = Math.min(parseFloat($("res").value), maxSize / Math.max(cw, ch));
    const W = Math.max(16, Math.round(cw * s)), H = Math.max(16, Math.round(ch * s));
    const tr = [-dot(right, pos), -dot(down, pos), -dot(fwd, pos)];
    const m = [right[0], right[1], right[2], tr[0], down[0], down[1], down[2], tr[1], fwd[0], fwd[1], fwd[2], tr[2]];
    send("cam " + W + " " + H + " " + (fyRatio * H) + " " + m.map((x) => x.toPrecision(9)).join(" "));
    changed = false;
  }
  if (t - lastFps > 1000) {
    $("fps").textContent = frames ? Math.round(frames * 1000 / (t - lastFps)) + " fps" : "idle";
    frames = 0; lastFps = t;
  }
  requestAnimationFrame(frame);
}
connect();
requestAnimationFrame(frame);
</script>
</body>
</html>
"""
