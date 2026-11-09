import React from 'react';
import {Figure, useStepper} from './Figure';
import {COLS, GAUSSIANS, H, PREPROCESS, ROWS, TILE_H, TILE_W, W, css, tileRect} from './scene';
import styles from './styles.module.css';

const describe = (r) => {
  if (r.status !== 'visible') return `Gaussian ${r.g.id} is culled: ${r.reason}. It gets no slot, and later stages never see it.`;
  const extra = r.candidates.length - r.tiles.length;
  const cut = extra > 0 ? ` Its rectangle overlaps ${r.candidates.length} tiles, but the ellipse only reaches ${r.tiles.length}.` : '';
  return `Gaussian ${r.g.id} is visible and takes slot ${r.slot}, touching ${r.tiles.length} tile${r.tiles.length === 1 ? '' : 's'}.${cut}`;
};

function Cell({x, y, color, label, dim, active}) {
  return (
    <g>
      <rect x={x} y={y} width={22} height={22} rx={3} fill={color ?? 'none'} fillOpacity={dim ? 0.18 : 1} stroke={active ? 'currentColor' : 'var(--ifm-color-emphasis-300)'} strokeWidth={active ? 2 : 1} />
      {label !== undefined && (
        <text x={x + 11} y={y + 15} textAnchor="middle" fontSize="11" fill={dim || !color ? 'currentColor' : '#10131a'}>
          {label}
        </text>
      )}
    </g>
  );
}

export default function Preprocess() {
  const stepper = useStepper(PREPROCESS.length, 1500);
  const cur = PREPROCESS[stepper.step];
  const seen = PREPROCESS.slice(0, stepper.step + 1);
  const slots = seen.filter((r) => r.status === 'visible');
  return (
    <Figure title="Preprocess: one thread per Gaussian" stepper={stepper} count={PREPROCESS.length} caption={describe(cur)}>
      <div className={styles.twoCol}>
      <svg viewBox={`-1 -1 ${W + 2} ${H + 2}`} className={styles.imagePlane} role="img" aria-label="Image with tiles and Gaussians">
        {PREPROCESS.map((r) => {
          if (r.status !== 'visible') return null;
          const isCur = r === cur;
          const done = PREPROCESS.indexOf(r) <= stepper.step;
          return (
            <ellipse
              key={r.g.id}
              cx={r.g.x}
              cy={r.g.y}
              rx={r.g.s1 * 1.6}
              ry={r.g.s2 * 1.6}
              transform={`rotate(${r.g.deg} ${r.g.x} ${r.g.y})`}
              fill={css(r.g.color, isCur ? 0.95 : done ? 0.45 : 0.1)}
            />
          );
        })}
        {cur.status === 'visible' &&
          cur.candidates.map((t) => {
            const {x, y} = tileRect(t);
            const hit = cur.tiles.includes(t);
            return <rect key={t} x={x} y={y} width={TILE_W} height={TILE_H} fill={hit ? 'rgba(255,255,255,0.14)' : 'none'} stroke={hit ? '#fff' : 'rgba(255,255,255,0.35)'} strokeWidth={hit ? 0.45 : 0.25} strokeDasharray={hit ? undefined : '1 1'} />;
          })}
        {Array.from({length: COLS + 1}, (_, i) => (
          <line key={`v${i}`} x1={i * TILE_W} x2={i * TILE_W} y1={0} y2={H} stroke="rgba(255,255,255,0.18)" strokeWidth="0.2" />
        ))}
        {Array.from({length: ROWS + 1}, (_, i) => (
          <line key={`h${i}`} y1={i * TILE_H} y2={i * TILE_H} x1={0} x2={W} stroke="rgba(255,255,255,0.18)" strokeWidth="0.2" />
        ))}
        {cur.status === 'visible' && (
          <rect x={cur.rect.x0} y={cur.rect.y0} width={cur.rect.x1 - cur.rect.x0} height={cur.rect.y1 - cur.rect.y0} fill="none" stroke="#fbbf24" strokeWidth="0.5" />
        )}
      </svg>
      <div className={styles.panel}>
        <svg viewBox="0 0 330 150" style={{maxWidth: 420}} role="img" aria-label="All Gaussians and the compacted slots">
          <text x="0" y="14" fontSize="12" fill="currentColor">
            All Gaussians (N)
          </text>
          {GAUSSIANS.map((g) => {
            const r = PREPROCESS[g.id];
            const done = g.id <= stepper.step;
            return <Cell key={g.id} x={g.id * 27} y={22} color={done && r.status === 'visible' ? css(g.color) : undefined} label={g.id} dim={done && r.status !== 'visible'} active={g.id === stepper.step} />;
          })}
          <text x="0" y="86" fontSize="12" fill="currentColor">
            Slots: only the visible ones, packed together
          </text>
          {Array.from({length: slots.length}, (_, i) => (
            <Cell key={i} x={i * 27} y={94} color={css(slots[i].g.color)} label={slots[i].slot} active={slots[i] === cur} />
          ))}
        </svg>
      </div>
      </div>
    </Figure>
  );
}
