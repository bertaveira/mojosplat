import React, {useEffect, useRef} from 'react';
import {Figure, useStepper} from './Figure';
import {FOCUS_LIST, FOCUS_TILE, TILE_H, TILE_W, compositeFocus, css, overlappingWarps} from './scene';
import styles from './styles.module.css';

function PixelCanvas({pixels, map, overlay}) {
  const ref = useRef(null);
  useEffect(() => {
    const ctx = ref.current.getContext('2d');
    const img = ctx.createImageData(TILE_W, TILE_H);
    pixels.forEach((p, i) => {
      const [r, g, b] = map(p);
      img.data.set([r, g, b, 255], i * 4);
    });
    ctx.putImageData(img, 0, 0);
  }, [pixels, map]);
  return (
    <div className={styles.mat}>
      <canvas ref={ref} width={TILE_W} height={TILE_H} className={styles.pixels} />
      {overlay}
    </div>
  );
}

function WarpGrid({active}) {
  return (
    <svg viewBox="0 0 16 12" style={{position: 'absolute', inset: 0, width: '100%', height: '100%'}} aria-hidden="true">
      {[0, 1, 2, 3, 4, 5].map((w) => {
        const x = (w % 2) * 8;
        const y = Math.floor(w / 2) * 4;
        const on = active.includes(w);
        return (
          <g key={w}>
            <rect x={x} y={y} width={8} height={4} fill={on ? 'rgba(251,191,36,0.13)' : 'none'} stroke={on ? '#fbbf24' : 'rgba(0,0,0,0.55)'} strokeWidth={on ? 0.14 : 0.06} />
            <text x={x + 0.35} y={y + 1} fontSize="0.85" fill="#fff" stroke="#000" strokeWidth="0.18" paintOrder="stroke">
              w{w}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

const color = (p) => [p.r, p.g, p.b].map((c) => Math.min(255, c + 16 * p.T));
const trans = (p) => {
  // Light left = bright blue, none left = near black; red marks stopped pixels.
  if (p.done) return [150, 45, 45];
  const v = Math.pow(p.T, 0.45);
  return [18 + 60 * v, 24 + 150 * v, 42 + 195 * v];
};

export default function Blend() {
  const count = FOCUS_LIST.length + 1;
  const stepper = useStepper(count, 1500);
  const n = stepper.step;
  const pixels = compositeFocus(n);
  const cur = n > 0 ? FOCUS_LIST[n - 1] : null;
  const warps = cur ? overlappingWarps(cur) : [];
  const done = pixels.filter((p) => p.done).length;
  const caption = cur
    ? `Gaussian ${cur.id}, number ${n} in depth order for this tile. Warps ${warps.join(', ')} see it overlap their sub-tile, and only they composite it. ${done} of 192 pixels have gone opaque and stopped.`
    : `Tile ${FOCUS_TILE}, about to blend ${FOCUS_LIST.length} Gaussians front to back. Each outlined sub-tile is one warp of 32 threads, one thread per pixel.`;
  return (
    <Figure title={`Blend: tile ${FOCUS_TILE}, 16×12 pixels, 6 warps`} stepper={stepper} count={count} caption={caption}>
      <div className={styles.twoCol}>
      <div className={styles.panel}>
        <div className={styles.panelTitle}>Color so far</div>
        <PixelCanvas pixels={pixels} map={color} overlay={<WarpGrid active={warps} />} />
      </div>
      <div className={styles.panel}>
        <div className={styles.panelTitle}>Light still getting through (blue = plenty, dark = almost none)</div>
        <PixelCanvas pixels={pixels} map={trans} overlay={<WarpGrid active={warps} />} />
      </div>
      </div>
      <div className={`${styles.panel} ${styles.listPanel}`}>
        <div className={styles.panelTitle}>This tile's list, nearest first</div>
        <svg viewBox={`0 0 ${Math.max(FOCUS_LIST.length, 1) * 30} 34`} style={{maxWidth: Math.max(FOCUS_LIST.length, 1) * 34}} role="img" aria-label="Gaussians of this tile in depth order">
          {FOCUS_LIST.map((g, i) => (
            <g key={g.id} opacity={i < n ? 1 : 0.3}>
              <rect x={i * 30 + 2} y="4" width="26" height="26" rx="4" fill={css(g.color)} stroke={i === n - 1 ? 'currentColor' : 'none'} strokeWidth="2" />
              <text x={i * 30 + 15} y="22" textAnchor="middle" fontSize="12" fill="#10131a">
                {g.id}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </Figure>
  );
}
