import React from 'react';
import {Figure, useStepper} from '../rendering/Figure';
import {Code, cx, tone} from './ui';
import s from './arch.module.css';

const BLOCKS = [
  {id: 'shape', name: 'QuatScale', sub: 'shape.mojo', tone: 'shape', fields: ['v0', 'v1', 'v2', 'q', 'norm_sq', 'R', 'S', 'cov'], read: ['v0', 'v1', 'v2', 'q', 'norm_sq', 'R', 'S']},
  {
    id: 'place',
    name: 'PinholeEWA',
    sub: 'pinhole_ewa.mojo',
    tone: 'placement',
    fields: ['cam', 'depth_', 'x', 'y', 'x_cl', 'y_cl', 'valid_x', 'valid_y', 'j11', 'j22', 'u', 'w', 'uc', 'wc', 'cov'],
    read: ['cam', 'depth_', 'x', 'y', 'x_cl', 'y_cl', 'valid_x', 'valid_y', 'j11', 'j22', 'u', 'w', 'uc', 'wc'],
  },
  {id: 'resp', name: 'Gaussian2D', sub: 'gaussian2d.mojo, ConicFromCov', tone: 'response', fields: ['mean', 'inverse.cov', 'inverse.det', 'inverse.conic'], read: ['inverse.cov', 'inverse.det']},
];

const LINES = [
  'var shape = QuatScale[dt](log_scales, rotations)',
  'var place = P(mean, shape, cam_v)',
  'var resp = R(place.mean2d().f32(), place.cov2d().f32())',
  'var screen = resp.backward(slot_grad)',
  'var grad = place.backward(shape, screen.mean2d, screen.cov2d)',
];

const STEPS = [
  {filled: ['shape'], line: 0, cap: 'Forward: the QuatScale constructor computes the 3D covariance and keeps what its backward needs in its fields.'},
  {filled: ['shape', 'place'], line: 1, cap: 'PinholeEWA reads shape.cov and keeps the Jacobian pieces. Nothing is written to memory: these are local variables of a GPU thread.'},
  {filled: ['shape', 'place', 'resp'], line: 2, cap: 'Gaussian2D inverts the dilated covariance. The forward is finished, and the three local variables are the tape.'},
  {filled: ['shape', 'place', 'resp'], line: 3, back: 'resp', cap: 'Backward, bottom to top: Gaussian2D uses the saved covariance and determinant to turn conic gradients into covariance gradients.'},
  {filled: ['shape', 'place', 'resp'], line: 4, back: 'place', cap: 'PinholeEWA pulls those gradients through its stored Jacobian.'},
  {filled: ['shape', 'place', 'resp'], line: 4, back: 'shape', cap: 'Inside place.backward, QuatScale turns the 3D covariance gradient into scale and quaternion gradients. Fields the kernel never reads are removed by the compiler.'},
];

export default function OpTape() {
  const stepper = useStepper(STEPS.length, 2200);
  const st = STEPS[stepper.step];
  return (
    <Figure title="One Gaussian: constructors are the forward, fields are the tape" stepper={stepper} count={STEPS.length} caption={st.cap}>
      <div className={s.tape}>
        {BLOCKS.map((b) => {
          const filled = st.filled.includes(b.id);
          const reading = st.back === b.id;
          return (
            <div key={b.id} className={cx(s.tapeBlock, tone(b.tone), reading && s.boxOn)}>
              <div className={s.tapeName}>{b.name}</div>
              <div className={s.tapeSub}>{b.sub}</div>
              <div className={s.fields}>
                {b.fields.map((f) => (
                  <span key={f} className={cx(s.field, filled && s.fieldW, reading && b.read.includes(f) && s.fieldR)}>
                    {f}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
      <div style={{marginTop: '0.75rem'}}>
        <Code>{LINES.map((l, i) => `${i === st.line ? '> ' : '  '}${l}`).join('\n')}</Code>
      </div>
      <div className={s.legend}>
        <span>
          <i className={cx(s.swatch, s.tShape)} />
          filled field
        </span>
        <span>
          <strong>bold</strong> read by this backward
        </span>
      </div>
    </Figure>
  );
}
