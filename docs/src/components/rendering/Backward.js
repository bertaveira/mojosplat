import React from 'react';
import {Figure, useStepper} from './Figure';
import {FOCUS_LIST, FOCUS_TILE, alphaAt, css, tileRect} from './scene';

const lanes = FOCUS_LIST;
const LANES = lanes.length;
const PIXELS = 12;
const STEPS = PIXELS + LANES;
const PITCH = 66;
const VIEW_W = LANES * PITCH;

const {x: tx, y: ty} = tileRect(FOCUS_TILE);
const pixelPos = (p) => [tx + (p % 16) + 0.5, ty + 5.5];

function total(g, count) {
  let s = 0;
  for (let p = 0; p < count; p++) s += alphaAt(g, ...pixelPos(p));
  return s;
}

export default function Backward() {
  const stepper = useStepper(STEPS, 700);
  const t = stepper.step;
  const totals = lanes.map((g, lane) => total(g, Math.max(0, Math.min(PIXELS, t - lane + 1))));
  const max = Math.max(...lanes.map((g) => total(g, PIXELS)));
  const finished = t === STEPS - 1;
  const caption = finished
    ? 'All pixels have passed every lane. Each lane now holds its Gaussian’s full gradient and issues one set of atomic additions, instead of one per pixel.'
    : 'Pixels stream through the warp, one lane per step. A lane evaluates its Gaussian at the pixel it holds, adds to that Gaussian’s gradient sum, and shuffles the pixel’s state to the next lane.';
  return (
    <Figure title={`Blend backward: one lane per Gaussian (${LANES} of 32 lanes shown)`} stepper={stepper} count={STEPS} caption={caption}>
      <svg viewBox={`0 0 ${VIEW_W} 262`} style={{maxWidth: 620}} role="img" aria-label="Pixels moving through lanes that each hold one Gaussian">
        <text x="0" y="16" fontSize="15" fill="currentColor">
          Lanes, each owning one Gaussian
        </text>
        {lanes.map((g, i) => (
          <g key={g.id} transform={`translate(${i * PITCH} 28)`}>
            <rect width="58" height="50" rx="6" fill={css(g.color)} />
            <text x="29" y="21" textAnchor="middle" fontSize="13" fill="#10131a">
              lane {i}
            </text>
            <text x="29" y="40" textAnchor="middle" fontSize="13" fontWeight="600" fill="#10131a">
              G{g.id}
            </text>
          </g>
        ))}
        {Array.from({length: PIXELS}, (_, p) => {
          const lane = t - p;
          if (lane < 0 || lane >= LANES) return null;
          return (
            <g key={p} style={{transform: `translate(${lane * PITCH + 12}px, 92px)`, transition: 'transform 500ms linear'}}>
              <rect width="34" height="26" rx="4" fill="var(--ifm-color-emphasis-200)" stroke="currentColor" strokeOpacity="0.5" />
              <text x="17" y="18" textAnchor="middle" fontSize="13" fill="currentColor">
                p{p}
              </text>
            </g>
          );
        })}
        <text x="0" y="152" fontSize="15" fill="currentColor">
          Gradient sum, in registers
        </text>
        {totals.map((v, i) => {
          const h = (v / max) * 88;
          return (
            <g key={lanes[i].id} transform={`translate(${i * PITCH} 164)`}>
              <rect width="58" height="88" rx="3" fill="var(--ifm-color-emphasis-200)" />
              <rect y={88 - h} width="58" height={h} rx="3" fill={finished ? '#c2410c' : css(lanes[i].color)} style={{transition: 'all 400ms ease'}} />
            </g>
          );
        })}
      </svg>
    </Figure>
  );
}
