import React, {useCallback, useEffect, useRef, useState} from 'react';
import styles from './styles.module.css';

// Steps through a figure, playing only while it is on screen.
export function useStepper(count, ms) {
  const [step, setStep] = useState(0);
  const [paused, setPaused] = useState(false);
  const [onScreen, setOnScreen] = useState(false);
  const [reduced, setReduced] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    setReduced(window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    const io = new IntersectionObserver(([e]) => setOnScreen(e.isIntersecting), {threshold: 0.15});
    if (ref.current) io.observe(ref.current);
    return () => io.disconnect();
  }, []);

  const playing = onScreen && !paused && !reduced;

  useEffect(() => {
    if (!playing) return undefined;
    const t = setInterval(() => setStep((s) => (s + 1) % count), ms);
    return () => clearInterval(t);
  }, [playing, count, ms]);

  const go = useCallback(
    (s) => {
      setPaused(true);
      setStep(((s % count) + count) % count);
    },
    [count],
  );
  const toggle = useCallback(() => {
    if (reduced) {
      // Reduced motion only stops autoplay: an explicit Play still plays.
      setReduced(false);
      setPaused(false);
    } else {
      setPaused((p) => !p);
    }
  }, [reduced]);

  return {step, go, playing, toggle, ref};
}

export function Figure({title, stepper, count, caption, children}) {
  const {step, go, playing, toggle, ref} = stepper;
  return (
    <figure className={styles.figure} ref={ref}>
      {title && <div className={styles.figTitle}>{title}</div>}
      <div className={styles.stage}>{children}</div>
      <div className={styles.controls}>
        <button type="button" onClick={() => go(step - 1)} aria-label="Previous step">
          ‹
        </button>
        <button type="button" onClick={toggle} aria-label={playing ? 'Pause' : 'Play'} className={styles.play}>
          {playing ? 'Pause' : 'Play'}
        </button>
        <button type="button" onClick={() => go(step + 1)} aria-label="Next step">
          ›
        </button>
        <input
          type="range"
          min={0}
          max={count - 1}
          value={step}
          aria-label="Step"
          onChange={(e) => go(Number(e.target.value))}
        />
        <span className={styles.counter}>
          {step + 1} / {count}
        </span>
      </div>
      <figcaption className={styles.caption}>{caption}</figcaption>
    </figure>
  );
}
