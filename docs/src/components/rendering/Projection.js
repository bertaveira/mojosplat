import React from 'react';
import {Figure, useStepper} from './Figure';

const FRAMES = 90;
const F = 170; // focal length, in figure units

// Top-down view: one 3D Gaussian, its image-plane footprint from the EWA Jacobian.
export default function Projection() {
  const stepper = useStepper(FRAMES, 55);
  const t = (stepper.step / FRAMES) * 2 * Math.PI;
  const z = 300 + 90 * Math.sin(t);
  const y = 60 * Math.sin(2 * t + 0.6);
  const rot = t * 0.9 + 0.4;
  const [s1, s2] = [42, 15];
  const [c, s] = [Math.cos(rot), Math.sin(rot)];
  const czz = c * c * s1 * s1 + s * s * s2 * s2;
  const czy = c * s * (s1 * s1 - s2 * s2);
  const cyy = s * s * s1 * s1 + c * c * s2 * s2;
  // Jacobian of u = f y / z at the mean: J = (f / z) [1, -y / z]
  const j1 = F / z;
  const j2 = -(F * y) / (z * z);
  const u = (F * y) / z;
  const sigma = Math.sqrt(j1 * j1 * cyy + 2 * j1 * j2 * czy + j2 * j2 * czz);

  const camX = 40;
  const cy = 170;
  const planeX = camX + F;
  const wx = camX + z;
  const wy = cy + y;
  const top = cy + u - 2 * sigma;
  const bot = cy + u + 2 * sigma;

  return (
    <Figure
      title="Projecting one Gaussian"
      stepper={stepper}
      count={FRAMES}
      caption={`Seen from above. The Gaussian's center goes through the pinhole, and its covariance goes through the Jacobian of the projection: J Σ Jᵀ. Closer means larger; rotating changes the width. Footprint here: ±2σ = ${(2 * sigma).toFixed(1)} units.`}>
      <svg viewBox="0 0 560 340" style={{maxWidth: 640}} role="img" aria-label="A Gaussian moving in front of a pinhole camera, with its projected footprint on the image plane">
        <line x1={planeX} x2={planeX} y1={30} y2={310} stroke="currentColor" strokeOpacity="0.45" strokeWidth="2" />
        <text x={planeX} y={22} textAnchor="middle" fontSize="17" fill="currentColor">
          image plane
        </text>
        <line x1={camX} x2={wx} y1={cy} y2={wy} stroke="#c2410c" strokeDasharray="4 4" strokeOpacity="0.8" />
        <line x1={camX} x2={planeX} y1={cy} y2={top} stroke="currentColor" strokeOpacity="0.25" />
        <line x1={camX} x2={planeX} y1={cy} y2={bot} stroke="currentColor" strokeOpacity="0.25" />
        <polygon points={`${camX - 14},${cy - 11} ${camX - 14},${cy + 11} ${camX},${cy}`} fill="currentColor" />
        <text x={camX - 14} y={cy + 32} fontSize="17" fill="currentColor">
          camera
        </text>
        <g transform={`translate(${wx} ${wy}) rotate(${(rot * 180) / Math.PI})`}>
          {[2, 1.4, 0.8].map((k, i) => (
            <ellipse key={k} rx={s1 * k * 0.7} ry={s2 * k * 0.7} fill="#c2410c" fillOpacity={0.14 + i * 0.16} />
          ))}
        </g>
        <text x={wx} y={wy - 56} textAnchor="middle" fontSize="17" fill="currentColor">
          3D Gaussian
        </text>
        <line x1={planeX} x2={planeX} y1={top} y2={bot} stroke="#c2410c" strokeWidth="6" strokeLinecap="round" />
        <circle cx={planeX} cy={cy + u} r="4.5" fill="#fff" stroke="#c2410c" strokeWidth="2" />
      </svg>
    </Figure>
  );
}
