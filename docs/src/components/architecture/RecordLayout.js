import React, {useState} from 'react';
import {Chips, Panel, cx, tone} from './ui';
import s from './arch.module.css';

const MODES = {
  record: {
    label: 'Record (32 bytes)',
    intro: <><code>records[slot]</code>, written by preprocess. Six floats belong to the Response; the kernel fills lanes 6 and 7 with the pixel rectangle.</>,
    grid: s.layout,
    cells: [
      {t: 'response', n: 'mx', d: 'Mean in image space (x).'},
      {t: 'response', n: 'my', d: 'Mean in image space (y).'},
      {t: 'response', n: 'a', d: 'Conic, entry a: the inverse of the dilated 2D covariance.'},
      {t: 'response', n: 'b', d: 'Conic, off-diagonal entry b.'},
      {t: 'response', n: 'c', d: 'Conic, entry c.'},
      {t: 'response', n: 'opacity', d: 'Activated opacity, after the sigmoid.'},
      {t: 'kernel', n: 'bounds', d: 'Pixel rectangle [x0, x1) × [y0, y1), packed as 4 × 16 bits across lanes 6 and 7 by store_record. Read by blend to skip sub-tiles.'},
      {t: 'kernel', n: 'bounds', d: 'Upper 32 bits of the packed pixel rectangle.'},
    ],
  },
  slab: {
    label: 'Blend slab (12 floats)',
    intro: <>One entry per lane of a warp, in shared memory. <code>store_slab</code> writes it for the Gaussians that overlap the warp’s sub-tile.</>,
    grid: cx(s.layout, s.layoutSlab),
    cells: [
      {t: 'response', n: 'mx', d: 'Mean x, copied from the record.'},
      {t: 'response', n: 'my', d: 'Mean y, copied from the record.'},
      {t: 'response', n: 'ea', d: 'Conic a times −0.5·log₂(e). Folding this constant in means alpha is one exp2 per pixel.'},
      {t: 'response', n: 'eb', d: 'Conic b times −log₂(e).'},
      {t: 'response', n: 'ec', d: 'Conic c times −0.5·log₂(e).'},
      {t: 'response', n: 'opacity', d: 'Copied from the record.'},
      {t: 'appearance', n: 'c0', d: 'Color channel 0, clamped at 0 by the kernel.'},
      {t: 'appearance', n: 'c1', d: 'Color channel 1, clamped at 0.'},
      {t: 'appearance', n: 'c2', d: 'Color channel 2, clamped at 0. Read through Response.channel.'},
      {t: 'none', n: '0', d: 'Unused by Gaussian2D.'},
      {t: 'none', n: '0', d: 'Unused by Gaussian2D.'},
      {t: 'none', n: '0', d: 'Unused by Gaussian2D.'},
    ],
  },
  grads: {
    label: 'Slot gradients',
    intro: <>In the rasterizer’s <code>grad_scratch</code>, per visible slot. Blend backward adds to it with atomics; <code>Response.grad_index</code> decides the layout.</>,
    bars: [
      [
        {t: 'response', n: 'mean2d', u: 2, d: 'dL/d(mean2d): grad_index 0 and 1, an array of [slot, 2].'},
        {t: 'response', n: 'conic', u: 3, d: 'dL/d(conic entries): grad_index 2..4, an array of [slot, 3]. The off-diagonal is per entry of the symmetric matrix.'},
        {t: 'response', n: 'opacity', u: 1, d: 'dL/d(opacity logit): grad_index 5, an array of [slot].'},
      ],
      [{t: 'appearance', n: 'color', u: 3, d: 'dL/d(color before clamping): [slot, 3], stored right after the Response’s gradients. Not per-Response: every pipeline has it.'}],
    ],
  },
};

export default function RecordLayout() {
  const [mode, setMode] = useState('record');
  const [cell, setCell] = useState(0);
  const m = MODES[mode];
  const all = m.cells || m.bars.flat();
  const sel = all[Math.min(cell, all.length - 1)];
  const choose = (i) => setCell(i);
  let next = 0;
  return (
    <Panel title="What one slot holds" caption={m.intro}>
      <Chips
        label="Layout"
        options={Object.entries(MODES).map(([k, v]) => ({id: k, label: v.label}))}
        value={mode}
        onChange={(v) => {
          setMode(v);
          setCell(0);
        }}
      />
      {m.cells && (
        <div className={m.grid}>
          {m.cells.map((c, i) => (
            <button key={i} type="button" aria-pressed={i === cell} className={cx(s.layoutCell, tone(c.t), i === cell && s.layoutCellOn)} onClick={() => choose(i)}>
              {c.n}
              <span className={s.layoutIdx}>{i}</span>
            </button>
          ))}
        </div>
      )}
      {m.bars && (
        <div className={s.gradBars}>
          {m.bars.map((row, r) => (
            <div key={r} className={s.gradBar}>
              {row.map((c) => {
                const i = next++;
                return (
                  <button key={c.n} type="button" aria-pressed={i === cell} className={cx(s.gradSeg, tone(c.t), i === cell && s.layoutCellOn)} style={{flexGrow: c.u}} onClick={() => choose(i)}>
                    {c.n}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      )}
      <div className={s.detail} aria-live="polite">
        <span className={s.mono}>{sel.n}</span>: {sel.d}
      </div>
    </Panel>
  );
}
