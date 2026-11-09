// A small scene that every figure shares, and the renderer's steps run on it:
// the same culling, tile test, sorts and compositing as the GPU kernels.

export const W = 64;
export const H = 48;
export const TILE_W = 16;
export const TILE_H = 12;
export const COLS = W / TILE_W;
export const ROWS = H / TILE_H;
export const MIN_ALPHA = 1 / 255;
export const T_MIN = 1e-4;
export const NEAR = 0.1;

// x, y, sigma1, sigma2, angle, opacity, color, depth
const RAW = [
  [23, 18, 5.5, 3.2, 20, 0.96, [235, 94, 72], 2.0],
  [27, 16, 4.2, 2.8, -35, 0.95, [244, 179, 61], 2.6],
  [20, 21, 3.8, 3.0, 60, 0.93, [86, 180, 124], 3.1],
  [29, 20, 3.4, 3.0, 10, 0.97, [76, 142, 232], 3.6],
  [18, 15, 2.8, 2.2, 0, 0.7, [176, 112, 224], 4.2],
  [45, 33, 6.5, 3.5, 30, 0.8, [242, 128, 170], 2.3],
  [51, 13, 4, 2.5, -20, 0.85, [108, 200, 214], 3.0],
  [10, 38, 5, 3.2, 45, 0.75, [244, 200, 104], 2.8],
  [37, 40, 3.5, 2.4, -10, 0.65, [150, 196, 96], 3.9],
  [80, 10, 4, 3, 0, 0.9, [200, 200, 200], 2.5], // off the image
  [33, 8, 3.5, 3, 0, 0.002, [200, 200, 200], 2.4], // transparent
  [12, 9, 3.5, 3, 0, 0.9, [200, 200, 200], -0.5], // behind the camera
];

export const GAUSSIANS = RAW.map(([x, y, s1, s2, deg, op, color, depth], id) => {
  const r = (deg * Math.PI) / 180;
  const [c, s] = [Math.cos(r), Math.sin(r)];
  const xx = c * c * s1 * s1 + s * s * s2 * s2;
  const xy = c * s * (s1 * s1 - s2 * s2);
  const yy = s * s * s1 * s1 + c * c * s2 * s2;
  const det = xx * yy - xy * xy;
  return {id, x, y, s1, s2, deg, op, color, depth, xx, yy, ca: yy / det, cb: -xy / det, cc: xx / det};
});

export const css = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

export function alphaAt(g, px, py) {
  const dx = g.x - px;
  const dy = g.y - py;
  return g.op * Math.exp(-0.5 * (g.ca * dx * dx + 2 * g.cb * dx * dy + g.cc * dy * dy));
}

export function tileRect(t) {
  return {x: (t % COLS) * TILE_W, y: Math.floor(t / COLS) * TILE_H};
}

function preprocessOne(g) {
  if (g.depth < NEAR) return {g, status: 'behind', reason: 'behind the near plane'};
  if (g.op < MIN_ALPHA) return {g, status: 'transparent', reason: 'opacity below 1/255'};
  const cut = 2 * Math.log(g.op * 255);
  const ex = Math.max(Math.sqrt(g.xx * cut) - 0.5, 0);
  const ey = Math.max(Math.sqrt(g.yy * cut) - 0.5, 0);
  const clamp = (v, hi) => Math.min(hi, Math.max(0, v));
  const rect = {
    x0: clamp(Math.floor(g.x - ex), W),
    x1: clamp(Math.ceil(g.x + ex), W),
    y0: clamp(Math.floor(g.y - ey), H),
    y1: clamp(Math.ceil(g.y + ey), H),
  };
  const candidates = [];
  const tiles = [];
  for (let ty = Math.floor(rect.y0 / TILE_H); ty < Math.ceil(rect.y1 / TILE_H); ty++) {
    for (let tx = Math.floor(rect.x0 / TILE_W); tx < Math.ceil(rect.x1 / TILE_W); tx++) {
      const id = ty * COLS + tx;
      candidates.push(id);
      let hit = false;
      for (let py = 0; py < TILE_H && !hit; py++) {
        for (let px = 0; px < TILE_W && !hit; px++) {
          hit = alphaAt(g, tx * TILE_W + px + 0.5, ty * TILE_H + py + 0.5) >= MIN_ALPHA;
        }
      }
      if (hit) tiles.push(id);
    }
  }
  if (tiles.length === 0) return {g, status: 'offscreen', reason: 'touches no tile on the image'};
  return {g, status: 'visible', rect, candidates, tiles};
}

export const PREPROCESS = (() => {
  let slot = 0;
  return GAUSSIANS.map((g) => {
    const r = preprocessOne(g);
    if (r.status === 'visible') r.slot = slot++;
    return r;
  });
})();

export const VISIBLE = PREPROCESS.filter((r) => r.status === 'visible');
export const DEPTH_ORDER = [...VISIBLE].sort((a, b) => a.g.depth - b.g.depth);
// One (tile, Gaussian) pair per tile a Gaussian touches, in depth order.
export const INSTANCES = DEPTH_ORDER.flatMap((r) => r.tiles.map((tile) => ({key: `${r.slot}-${tile}`, r, tile})));
// The stable sort by tile keeps depth order inside each tile.
export const BY_TILE = [...INSTANCES].sort((a, b) => a.tile - b.tile);

export const FOCUS_TILE = 5; // the tile the blending figures zoom into
export const FOCUS_LIST = BY_TILE.filter((i) => i.tile === FOCUS_TILE).map((i) => i.r.g);

export function compositeFocus(upTo) {
  // Front-to-back compositing of the focus tile: { rgb, T } per pixel.
  const {x: tx, y: ty} = tileRect(FOCUS_TILE);
  const px = [];
  for (let y = 0; y < TILE_H; y++) {
    for (let x = 0; x < TILE_W; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      let T = 1;
      let done = false;
      for (let i = 0; i < upTo && !done; i++) {
        const gi = FOCUS_LIST[i];
        const a = alphaAt(gi, tx + x + 0.5, ty + y + 0.5);
        if (a < MIN_ALPHA) continue;
        r += T * a * gi.color[0];
        g += T * a * gi.color[1];
        b += T * a * gi.color[2];
        T *= 1 - a;
        if (T < T_MIN) done = true;
      }
      px.push({r, g, b, T, done});
    }
  }
  return px;
}

// Which of the six 8x4 sub-tiles a Gaussian's rectangle overlaps.
export function overlappingWarps(g) {
  const r = PREPROCESS[g.id].rect;
  const {x: tx, y: ty} = tileRect(FOCUS_TILE);
  const out = [];
  for (let w = 0; w < 6; w++) {
    const x0 = tx + (w % 2) * 8;
    const y0 = ty + Math.floor(w / 2) * 4;
    if (r.x0 < x0 + 8 && x0 < r.x1 && r.y0 < y0 + 4 && y0 < r.y1) out.push(w);
  }
  return out;
}
