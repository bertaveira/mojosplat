import React from 'react';
import base from '../rendering/styles.module.css';
import s from './arch.module.css';

export const cx = (...c) => c.filter(Boolean).join(' ');

export function tone(name) {
  return s[`t${name[0].toUpperCase()}${name.slice(1)}`];
}

export function Panel({title, caption, children, controls}) {
  return (
    <figure className={base.figure}>
      {title && <div className={base.figTitle}>{title}</div>}
      <div className={base.stage}>{children}</div>
      {controls}
      {caption && <figcaption className={base.caption}>{caption}</figcaption>}
    </figure>
  );
}

export function Chips({label, options, value, onChange}) {
  return (
    <div className={s.chips} role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          aria-pressed={value === o.id}
          className={cx(s.chip, value === o.id && s.chipOn)}
          onClick={() => onChange(o.id)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

const esc = (t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// `mark` maps a token to a tone name; word tokens match on word boundaries only.
export function Code({children, mark = {}}) {
  const keys = Object.keys(mark).sort((a, b) => b.length - a.length);
  if (!keys.length) {
    return (
      <pre className={s.code}>
        <code>{children}</code>
      </pre>
    );
  }
  const re = new RegExp(`(${keys.map((k) => `${/^\w/.test(k) ? '\\b' : ''}${esc(k)}${/\w$/.test(k) ? '\\b' : ''}`).join('|')})`, 'g');
  const parts = String(children).split(re);
  return (
    <pre className={s.code}>
      <code>
        {parts.map((p, i) =>
          mark[p] ? (
            <span key={i} className={cx(s.tok, tone(mark[p]))}>
              {p}
            </span>
          ) : (
            p
          ),
        )}
      </code>
    </pre>
  );
}
