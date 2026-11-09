import React, {useState} from 'react';
import {Panel, cx, tone} from './ui';
import s from './arch.module.css';

const BASE = [
  {i: 0, name: 'means', w: '3', role: 'core'},
  {i: 1, name: 'log_scales', w: '3', role: 'core'},
  {i: 2, name: 'rotations', w: '4', role: 'core'},
  {i: 3, name: 'opacity_logits', w: '1', role: 'core'},
  {i: 4, name: 'sh0', w: '3', role: 'appearance', ex: 'c0'},
  {i: 5, name: 'sh_rest', w: 'K×3', role: 'appearance', ex: 'c1'},
];

export default function ColumnMap() {
  const [training, setTraining] = useState(false);
  const [resp, setResp] = useState(false);
  const rows = resp ? [...BASE, {i: 6, name: 'scale_of_alpha', w: '1', role: 'response', ex: 'c2'}] : BASE;
  const extras = [
    {c: 'c0', name: 'sh0', t: 'appearance'},
    {c: 'c1', name: 'sh_rest', t: 'appearance'},
    {c: 'c2', name: resp ? 'scale_of_alpha' : 'placeholder', t: resp ? 'response' : 'none'},
    {c: 'c3', name: 'placeholder', t: 'none'},
  ];
  return (
    <Panel
      title="Splats: one array per column"
      caption={
        <>
          <strong>K</strong> is <code>rest_stride</code>, the SH bases stored per Gaussian (0, 3, 8 or 15). The extra column shown is <code>scale_of_alpha</code>, from the test-only{' '}
          <code>ScaledGaussian</code>. The Trainer runs with <code>sh_rest_grad=False</code>, so <code>grads[5]</code> is a one-float placeholder there.
        </>
      }>
      <div className={s.toggles}>
        <label>
          <input type="checkbox" checked={training} onChange={(e) => setTraining(e.target.checked)} /> <code>enable_training()</code>
        </label>
        <label>
          <input type="checkbox" checked={resp} onChange={(e) => setResp(e.target.checked)} /> Response that owns a column
        </label>
      </div>
      <div className={cx(s.colRow, s.colHead)}>
        <span>#</span>
        <span>Column</span>
        <span>W</span>
        <span>Arrays</span>
      </div>
      {rows.map((r) => (
        <div key={r.i} className={s.colRow}>
          <span className={s.colIdx}>{r.i}</span>
          <span className={s.colName}>{r.name}</span>
          <span className={s.colW}>{r.w}</span>
          <span className={s.colTags}>
            <span className={cx(s.tag, tone(r.role === 'core' ? 'shape' : r.role))}>params</span>
            {training && <span className={cx(s.tag, tone('kernel'))}>grads</span>}
            {training && <span className={cx(s.tag, tone('data'))}>exp_avg</span>}
            {training && <span className={cx(s.tag, tone('data'))}>exp_avg_sq</span>}
            {r.ex && <span className={cx(s.tag, tone('pipeline'))}>extras.{r.ex}</span>}
          </span>
        </div>
      ))}
      {training && (
        <div className={s.colRow}>
          <span />
          <span className={s.colName}>densification</span>
          <span className={s.colW}>2</span>
          <span className={s.colTags}>
            <span className={cx(s.tag, tone('kernel'))}>not a column</span>
          </span>
        </div>
      )}
      <div className={s.extrasRow} aria-label="ExtraColumns passed to kernels">
        {extras.map((e) => (
          <div key={e.c} className={cx(s.extraBox, tone(e.t))}>
            <b>{e.c}</b>
            {e.name}
          </div>
        ))}
      </div>
    </Panel>
  );
}
