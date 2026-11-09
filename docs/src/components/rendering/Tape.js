import React from 'react';
import {Figure, useStepper} from './Figure';
import styles from './styles.module.css';

const NODES = [
  {name: 'Raw parameters', sub: 'mean, scales, quat', out: '∂L/∂ parameters'},
  {name: 'QuatScale', sub: '→ 3D covariance', out: '∂L/∂ scales, quat'},
  {name: 'PinholeEWA', sub: '→ position, 2D cov.', out: '∂L/∂ mean, covariance'},
  {name: 'Gaussian2D', sub: '→ conic', out: '∂L/∂ position, covariance'},
  {name: 'Slot gradients', sub: 'from blend backward', out: '∂L/∂ position, conic'},
];

const STEP = [
  [0, 'forward', 'Forward: start from the raw parameters of one visible Gaussian.'],
  [1, 'forward', 'QuatScale builds the 3D covariance, and keeps the rotation and scales in its fields.'],
  [2, 'forward', 'PinholeEWA projects it, and keeps its Jacobian pieces.'],
  [3, 'forward', 'Gaussian2D inverts the 2D covariance. The forward is done, and nothing was written to memory.'],
  [4, 'backward', 'Read this Gaussian’s gradients that blend backward wrote for its slot.'],
  [3, 'backward', 'Gaussian2D uses its saved covariance and determinant to turn the conic gradient into a covariance gradient.'],
  [2, 'backward', 'PinholeEWA pushes it through the Jacobian it kept.'],
  [1, 'backward', 'QuatScale turns the covariance gradient into scale and rotation gradients.'],
  [0, 'backward', 'Gradients for this Gaussian are written, in order, so memory writes are contiguous.'],
];

export default function Tape() {
  const stepper = useStepper(STEP.length, 1500);
  const [at, dir, caption] = STEP[stepper.step];
  const back = dir === 'backward';
  return (
    <Figure title="One Gaussian, forward then backward" stepper={stepper} count={STEP.length} caption={caption}>
      <div className={styles.chain}>
        {NODES.map((n, i) => (
          <React.Fragment key={n.name}>
            <div className={`${styles.node} ${i === at ? (back ? styles.nodeBackward : styles.nodeForward) : ''}`}>
              <div className={styles.nodeName}>{n.name}</div>
              <div className={styles.nodeSub}>{n.sub}</div>
            </div>
            {i < NODES.length - 1 && (
              <span className={styles.arrow} aria-hidden="true">
                {back ? '←' : '→'}
              </span>
            )}
          </React.Fragment>
        ))}
      </div>
      <div className={styles.chainNote}>{back ? `Flowing backward: ${NODES[at].out}` : 'The fields of each block are the tape.'}</div>
    </Figure>
  );
}
