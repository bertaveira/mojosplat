import React from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {Figure, useStepper} from './Figure';
import styles from './styles.module.css';

const FRAMES = [
  ['2%', '31 thousand'],
  ['5%', '78 thousand'],
  ['12%', '188 thousand'],
  ['25%', '391 thousand'],
  ['40%', '626 thousand'],
  ['60%', '939 thousand'],
  ['80%', '1.25 million'],
  ['100%', 'all 1.57 million'],
];

export default function BuildUp() {
  const stepper = useStepper(FRAMES.length, 1100);
  const base = useBaseUrl('/img/rendering/');
  const [pct, n] = FRAMES[stepper.step];
  return (
    <Figure
      title="A real render, built from the nearest Gaussians outwards"
      stepper={stepper}
      count={FRAMES.length}
      caption={`The nearest ${pct} of on-screen Gaussians (${n}), on a trained Mip-NeRF 360 garden model. Each image is a real MojoSplat render.`}>
      <img
        className={styles.photo}
        src={`${base}buildup-${stepper.step}.webp`}
        alt={`Render using the nearest ${pct} of the on-screen Gaussians`}
        width={900}
        height={583}
      />
    </Figure>
  );
}
