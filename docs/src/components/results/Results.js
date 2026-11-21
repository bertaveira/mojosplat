import React, {useEffect, useRef, useState} from 'react';
import {Chips, cx} from '../architecture/ui';
import base from '../rendering/styles.module.css';
import s from './results.module.css';
import DATA from './data.json';

const {scenes: SCENES, methods: METHODS} = DATA;
const BY_ID = Object.fromEntries(METHODS.map((m) => [m.id, m]));

const MEASURES = [
  {id: 'psnr', label: 'PSNR'},
  {id: 'ssim', label: 'SSIM'},
  {id: 'lpips', label: 'LPIPS'},
  {id: 'train', label: 'Training time'},
  {id: 'render', label: 'Rendering speed'},
];
const QUALITY = {psnr: 0, ssim: 1, lpips: 2};
const DIGITS = {psnr: 2, ssim: 3, lpips: 3};
const UNIT = {psnr: ' dB', ssim: '', lpips: ''};
const NOISE = 0.15; // dB: a first-run gap this large is repeated before it counts

// How each method's quality numbers are scored; timing notes are in each method's section.
const METRIC_NOTE = {
  'faster-gs': 'SSIM is torchmetrics with `data_range=1.0`, and LPIPS is VGG with `normalize=True`.',
  'beta-splatting':
    'SSIM is `fused_ssim`, and LPIPS is `lpipsPyTorch` VGG, as in the official code. The repeated official PSNRs (hollow) are the float values it reports, about 0.05 dB above its 8-bit scores.',
  '3dgs-mcmc':
    'From random starts. SSIM and LPIPS are the functions of the official `metrics.py` (3DGS’s SSIM, `lpipsPyTorch` VGG).',
  'spherical-voronoi': 'SSIM is `fused_ssim`, and LPIPS is `lpipsPyTorch` VGG, as in the official code.',
  '3dgut':
    'SSIM is torchmetrics with `data_range=1.0`, and LPIPS is torchmetrics VGG, as in the official code. These runs used the port before it was merged into the current code; the merged code’s PSNRs are in the 3DGUT section.',
};

const fmt = (v, d) => v.toFixed(d);
const signed = (v, d) => (v > 0 ? '+' : v < 0 ? '−' : '±') + Math.abs(v).toFixed(d);
const times = (v) => `${v.toFixed(2)}×`;
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

function Inline({text}) {
  return text.split(/(`[^`]+`)/).map((p, i) => (p.startsWith('`') ? <code key={i}>{p.slice(1, -1)}</code> : p));
}

function useWidth(fallback = 680) {
  const ref = useRef(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    if (!ref.current || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

function niceTicks(lo, hi, count = 5) {
  const step0 = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((f) => f >= step0);
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-6; t += step) ticks.push(+t.toFixed(10));
  return {ticks, step};
}

// ---- Data accessors ---------------------------------------------------------

function quality(m, measure, scene, side) {
  return m.quality[side][scene][QUALITY[measure]];
}

// Every PSNR run of one side as {v, float}: the repeats when there are any, else the first
// run. Beta Splatting's official repeats are its float PSNR (about 0.05 dB above the 8-bit
// score), so there the 8-bit first run is kept and only the later repeats are added.
function runs(m, measure, scene, side) {
  const r = measure === 'psnr' && m.repeats[scene];
  const first = {v: quality(m, measure, scene, side), float: false};
  if (!r) return [first];
  if (side === 'official' && m.officialRepeatsFloat) {
    return [first, ...r.official.slice(1).map((v) => ({v, float: true}))];
  }
  return r[side].map((v) => ({v, float: false}));
}

// Where the strip puts zero: the official mean over 8-bit runs, else its first run.
function reference(m, measure, scene) {
  const r = measure === 'psnr' && m.repeats[scene];
  if (!r || m.officialRepeatsFloat) return quality(m, measure, scene, 'official');
  return r.officialMean ?? mean(r.official);
}

function delta(m, measure, scene) {
  return quality(m, measure, scene, 'ours') - quality(m, measure, scene, 'official');
}

function timeCell(measure, v) {
  return measure === 'train' ? `${v} s` : `${fmt(v[0], 2)} ms`;
}

function runList(xs, d, m) {
  return xs.map((x) => fmt(x, d)).join(', ') + (m != null ? ` (mean ${fmt(m, d)})` : '');
}

// ---- Shared chart pieces ----------------------------------------------------

const ROW = 30;
const TOP = 22;
const AXIS = 26;

function Frame({width, rows, label, right, x, ticks, tickFmt, children, hover, setHover, extra, head, zero}) {
  const L = label;
  const R = right;
  const h = TOP + rows.length * ROW + AXIS;
  return (
    <svg width={width} height={h} role="img" aria-label="Chart; the table view lists the same values">
      {rows.map((r, i) => (
        <g key={r.key}>
          {hover === i && <rect className={s.rowHover} x={0} y={TOP + i * ROW} width={width} height={ROW} rx={4} />}
          {r.rule && <line className={s.zero} x1={0} x2={width} y1={TOP + i * ROW} y2={TOP + i * ROW} />}
        </g>
      ))}
      {ticks.map((t) => (
        <g key={t}>
          <line className={s.grid} x1={x(t)} x2={x(t)} y1={TOP} y2={TOP + rows.length * ROW} />
          <text className={s.tick} x={x(t)} y={TOP + rows.length * ROW + 16} textAnchor="middle">
            {tickFmt(t)}
          </text>
        </g>
      ))}
      {extra}
      {zero != null && <line className={s.zero} x1={x(zero)} x2={x(zero)} y1={TOP} y2={TOP + rows.length * ROW} />}
      {head && (
        <text className={s.tick} x={width} y={TOP - 8} textAnchor="end">
          {head}
        </text>
      )}
      {rows.map((r, i) => {
        const cy = TOP + i * ROW + ROW / 2;
        return (
          <g key={r.key}>
            <text className={cx(s.rowLabel, r.strong && s.strong)} x={0} y={cy + 4}>
              {r.label}
            </text>
            {children(r, cy, i)}
            <text className={cx(r.strong && s.strong)} x={width} y={cy + 4} textAnchor="end">
              {r.right}
            </text>
            <rect
              className={s.hit}
              x={0}
              y={TOP + i * ROW}
              width={width}
              height={ROW}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
              onTouchStart={() => setHover(i)}
            />
          </g>
        );
      })}
      <line className={s.zero} x1={L} x2={width - R} y1={TOP + rows.length * ROW} y2={TOP + rows.length * ROW} />
    </svg>
  );
}

function Tooltip({hover, rows, width, children}) {
  if (hover == null) return null;
  const top = TOP + hover * ROW + ROW + 4;
  return (
    <div className={s.tooltip} style={{top, left: Math.min(width * 0.28, Math.max(0, width - 300))}}>
      {children(rows[hover])}
    </div>
  );
}

function TipRow({swatch, label, value}) {
  return (
    <div className={s.tipRow}>
      <span>
        {swatch && <i className={cx(s.swatch, swatch)} />}
        {label}
      </span>
      <span>{value}</span>
    </div>
  );
}

function Legend({items}) {
  return (
    <div className={s.legend}>
      {items.map(([cls, label]) => (
        <span key={label}>
          <i className={cx(s.swatch, cls)} />
          {label}
        </span>
      ))}
    </div>
  );
}

// ---- One method: quality as a strip of runs around the official result ------

function QualityStrip({m, measure}) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const d = DIGITS[measure];
  const rows = [...SCENES, 'mean'].map((scene) => {
    const ref0 = reference(m, measure, scene);
    return {
      key: scene,
      scene,
      label: scene,
      strong: scene === 'mean',
      rule: scene === 'mean',
      official: runs(m, measure, scene, 'official').map((p) => ({...p, v: p.v - ref0})),
      ours: runs(m, measure, scene, 'ours').map((p) => p.v - ref0),
      right: signed(delta(m, measure, scene), d),
    };
  });
  const span = Math.max(measure === 'psnr' ? 0.3 : 0.004, ...rows.flatMap((r) => [...r.official.map((p) => p.v), ...r.ours].map(Math.abs)));
  const {ticks} = niceTicks(-span, span, width < 480 ? 4 : 6);
  const lo = Math.min(-span, ticks[0]);
  const hi = Math.max(span, ticks[ticks.length - 1]);
  const L = width < 480 ? 62 : 80;
  const R = 58;
  const x = (v) => L + ((v - lo) / (hi - lo)) * (width - L - R);
  const hollow = m.officialRepeatsFloat;
  const plotH = rows.length * ROW;
  return (
    <>
      <Legend
        items={[
          [s.swOfficial, 'Official code'],
          [s.swOurs, 'MojoSplat'],
          ...(hollow && measure === 'psnr' ? [[s.swHollow, 'Later official repeat (float PSNR)']] : []),
          ...(measure === 'psnr' ? [[s.swBand, `±${NOISE} dB: a first-run gap this large is repeated`]] : []),
        ]}
      />
      <div className={s.chart} ref={ref}>
        <Frame
          width={width}
          rows={rows}
          label={L}
          right={R}
          x={x}
          ticks={ticks}
          tickFmt={(t) => (t === 0 ? '0' : signed(t, measure === 'psnr' ? 2 : 3))}
          hover={hover}
          setHover={setHover}
          head="Difference"
          zero={0}
          extra={
            measure === 'psnr' && (
              <rect className={s.band} x={x(-NOISE)} width={x(NOISE) - x(-NOISE)} y={TOP} height={plotH} />
            )
          }>
          {(r, cy) => (
            <>
              {r.official.map((p, k) => (
                <circle
                  key={`o${k}`}
                  className={p.float ? s.officialHollow : s.official}
                  cx={x(p.v)}
                  cy={cy - 4}
                  r={5}
                />
              ))}
              {r.ours.map((v, k) => (
                <circle key={`u${k}`} className={s.ours} cx={x(v)} cy={cy + 4} r={5} />
              ))}
            </>
          )}
        </Frame>
        <Tooltip hover={hover} rows={rows} width={width}>
          {(r) => {
            const rep = measure === 'psnr' && m.repeats[r.scene];
            return (
              <>
                <b>{r.scene}</b>
                <TipRow
                  swatch={s.swOfficial}
                  label="Official"
                  value={
                    rep
                      ? rep.officialNote ?? runList(rep.official, d, rep.officialMean)
                      : fmt(quality(m, measure, r.scene, 'official'), d)
                  }
                />
                <TipRow
                  swatch={s.swOurs}
                  label="MojoSplat"
                  value={rep ? runList(rep.ours, d, rep.oursMean) : fmt(quality(m, measure, r.scene, 'ours'), d)}
                />
                <TipRow label="First runs, MojoSplat − official" value={r.right + UNIT[measure]} />
              </>
            );
          }}
        </Tooltip>
      </div>
      <div className={s.note}>
        <p>
          Each dot is one training run, placed relative to the official result
          {measure === 'psnr' && !hollow ? ' (the mean of its runs where a scene was repeated)' : ''}
          {measure === 'psnr' && hollow ? ' (its first run, scored in 8 bits)' : ''}. Right: MojoSplat minus official,
          first run of each side{measure === 'lpips' ? '; lower LPIPS is better' : ''}.
        </p>
      </div>
    </>
  );
}

// ---- One method: time as paired bars ----------------------------------------

function TimeBars({m, measure}) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const table = m[measure];
  const last = measure === 'train' ? 'total' : 'mean';
  const ms = (v) => (measure === 'train' ? v : v[0]);
  const rows = SCENES.map((scene) => ({
    key: scene,
    scene,
    label: scene,
    official: ms(table[scene].official),
    ours: ms(table[scene].ours),
    right: times(table[scene].speedup),
  }));
  const hi0 = Math.max(...rows.map((r) => Math.max(r.official, r.ours)));
  const {ticks} = niceTicks(0, hi0, width < 480 ? 3 : 5);
  const hi = Math.max(hi0, ticks[ticks.length - 1]);
  const L = width < 480 ? 62 : 80;
  const R = 58;
  const x = (v) => L + (v / hi) * (width - L - R);
  const sum = table[last];
  return (
    <>
      <Legend items={[[s.swOfficial, 'Official code'], [s.swOurs, 'MojoSplat']]} />
      <div className={s.chart} ref={ref}>
        <Frame
          width={width}
          rows={rows}
          label={L}
          right={R}
          x={x}
          ticks={ticks}
          tickFmt={(t) => (measure === 'train' ? `${t} s` : `${t} ms`)}
          hover={hover}
          setHover={setHover}
          head="Speedup">
          {(r, cy) => (
            <>
              <rect className={s.barOfficial} x={L} y={cy - 9} width={Math.max(1, x(r.official) - L)} height={8} rx={2} />
              <rect className={s.barOurs} x={L} y={cy + 1} width={Math.max(1, x(r.ours) - L)} height={8} rx={2} />
            </>
          )}
        </Frame>
        <Tooltip hover={hover} rows={rows} width={width}>
          {(r) => (
            <>
              <b>{r.scene}</b>
              <TipRow swatch={s.swOfficial} label="Official" value={cellValue(measure, table[r.scene].official)} />
              <TipRow swatch={s.swOurs} label="MojoSplat" value={cellValue(measure, table[r.scene].ours)} />
              <TipRow label="Speedup" value={r.right} />
            </>
          )}
        </Tooltip>
      </div>
      <div className={s.note}>
        <p>
          {measure === 'train' ? 'Total over the seven scenes' : 'Mean over the seven scenes'}:{' '}
          <b>{cellValue(measure, sum.official)}</b> official, <b>{cellValue(measure, sum.ours)}</b> MojoSplat,{' '}
          <b>{times(sum.speedup)}</b>. Right: the speedup, official time divided by MojoSplat’s; below 1× means
          MojoSplat is slower. {measure === 'render' && 'Both renderers draw the same model.'}
        </p>
      </div>
    </>
  );
}

function cellValue(measure, v) {
  return measure === 'train' ? `${v} s` : `${fmt(v[0], 2)} ms (${v[1]} FPS)`;
}

// ---- All methods: one row per method, a dot per scene ------------------------

function AllMethods({measure}) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState(null);
  const isQ = measure in QUALITY;
  const d = DIGITS[measure];
  const rows = METHODS.map((m) => {
    const pts = SCENES.map((scene) => ({
      scene,
      v: isQ ? delta(m, measure, scene) : m[measure][scene].speedup,
    }));
    const summary = isQ ? delta(m, measure, 'mean') : m[measure][measure === 'train' ? 'total' : 'mean'].speedup;
    return {key: m.id, m, label: m.name, pts, summary, right: isQ ? signed(summary, d) : times(summary)};
  });
  const L = width < 480 ? 104 : 132;
  const R = 58;
  let x;
  let ticks;
  let tickFmt;
  if (isQ) {
    const span = Math.max(measure === 'psnr' ? 0.3 : 0.004, ...rows.flatMap((r) => r.pts.map((p) => Math.abs(p.v))));
    const nt = niceTicks(-span, span, width < 480 ? 4 : 6);
    ticks = nt.ticks;
    const lo = Math.min(-span, ticks[0]);
    const hi = Math.max(span, ticks[ticks.length - 1]);
    x = (v) => L + ((v - lo) / (hi - lo)) * (width - L - R);
    tickFmt = (t) => (t === 0 ? '0' : signed(t, measure === 'psnr' ? 2 : 3));
  } else {
    const all = rows.flatMap((r) => r.pts.map((p) => p.v));
    const lo = Math.min(0.5, ...all) / 1.1;
    const hi = Math.max(...all) * 1.1;
    ticks = [0.5, 1, 2, 4, 8, 16].filter((t) => t >= lo && t <= hi);
    x = (v) => L + ((Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * (width - L - R);
    tickFmt = (t) => `${t}×`;
  }
  const plotH = rows.length * ROW;
  const zero = isQ ? 0 : 1;
  return (
    <>
      <Legend
        items={[
          [s.swOurs, 'One scene'],
          [s.swMean, isQ ? 'Mean of the seven scenes' : measure === 'train' ? 'Total time' : 'Mean time per view'],
          ...(measure === 'psnr' ? [[s.swBand, `±${NOISE} dB: a first-run gap this large is repeated`]] : []),
        ]}
      />
      <div className={s.chart} ref={ref}>
        <Frame
          width={width}
          rows={rows}
          label={L}
          right={R}
          x={x}
          ticks={ticks}
          tickFmt={tickFmt}
          hover={hover}
          setHover={setHover}
          head={isQ ? 'Mean' : measure === 'train' ? 'Total' : 'Mean'}
          zero={zero}
          extra={
            <>
              {measure === 'psnr' && (
                <rect className={s.band} x={x(-NOISE)} width={x(NOISE) - x(-NOISE)} y={TOP} height={plotH} />
              )}
            </>
          }>
          {(r, cy) => (
            <>
              {r.pts.map((p) => (
                <circle key={p.scene} className={s.ours} cx={x(p.v)} cy={cy} r={5} opacity={0.85} />
              ))}
              <rect
                className={s.mean}
                x={x(r.summary) - 5}
                y={cy - 5}
                width={10}
                height={10}
                transform={`rotate(45 ${x(r.summary)} ${cy})`}
              />
            </>
          )}
        </Frame>
        <Tooltip hover={hover} rows={rows} width={width}>
          {(r) => (
            <>
              <b>{r.label}</b>
              {r.pts.map((p) => (
                <TipRow key={p.scene} label={p.scene} value={isQ ? signed(p.v, d) : times(p.v)} />
              ))}
              <TipRow label={isQ ? 'mean' : measure === 'train' ? 'total' : 'mean'} value={r.right} />
            </>
          )}
        </Tooltip>
      </div>
      <div className={s.note}>
        <p>
          {isQ
            ? `MojoSplat minus the official code, first run of each side${measure === 'lpips' ? '; lower LPIPS is better' : ''}. Pick a method to see the repeated runs.`
            : 'Speedup over the official code (its time divided by MojoSplat’s), on a log scale. Below 1× means MojoSplat is slower.'}
          {measure === 'lpips' && ' LPIPS values are only comparable within one method, since methods use different LPIPS implementations.'}
        </p>
      </div>
    </>
  );
}

// ---- Tables -----------------------------------------------------------------

function MethodTable({m, measure}) {
  if (measure in QUALITY) {
    const d = DIGITS[measure];
    const withRuns = measure === 'psnr' && Object.keys(m.repeats).length > 0;
    return (
      <div className={s.scroll}>
      <table className={s.table}>
        <thead>
          <tr>
            <th>Scene</th>
            <th>Official</th>
            <th>MojoSplat</th>
            <th>Difference</th>
            {withRuns && <th className={s.runs}>Repeated runs, official / MojoSplat</th>}
          </tr>
        </thead>
        <tbody>
          {[...SCENES, 'mean'].map((scene) => {
            const rep = withRuns && m.repeats[scene];
            return (
              <tr key={scene} className={cx(scene === 'mean' && s.total)}>
                <td>{scene}</td>
                <td>{fmt(quality(m, measure, scene, 'official'), d)}</td>
                <td>{fmt(quality(m, measure, scene, 'ours'), d)}</td>
                <td>{signed(delta(m, measure, scene), d)}</td>
                {withRuns && (
                  <td className={s.runs}>
                    {rep && (
                      <>
                        {rep.officialNote ?? runList(rep.official, d, rep.officialMean)}
                        <br />
                        {runList(rep.ours, d, rep.oursMean)}
                      </>
                    )}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
    );
  }
  const table = m[measure];
  const last = measure === 'train' ? 'total' : 'mean';
  return (
    <div className={s.scroll}>
      <table className={s.table}>
      <thead>
        <tr>
          <th>Scene</th>
          <th>Official</th>
          <th>MojoSplat</th>
          <th>Speedup</th>
        </tr>
      </thead>
      <tbody>
        {[...SCENES, last].map((scene) => (
          <tr key={scene} className={cx(scene === last && s.total)}>
            <td>{scene}</td>
            <td>{cellValue(measure, table[scene].official)}</td>
            <td>{cellValue(measure, table[scene].ours)}</td>
            <td>{times(table[scene].speedup)}</td>
          </tr>
        ))}
      </tbody>
    </table>
      </div>
  );
}

function AllTable({measure}) {
  const isQ = measure in QUALITY;
  const d = DIGITS[measure];
  const last = isQ || measure === 'render' ? 'mean' : 'total';
  const cell = (m, scene) =>
    isQ ? (
      <>
        {fmt(quality(m, measure, scene, 'ours'), d)}
        <span className={s.sub}>{signed(delta(m, measure, scene), d)}</span>
      </>
    ) : (
      <>
        {timeCell(measure, m[measure][scene].ours)}
        <span className={s.sub}>{times(m[measure][scene].speedup)}</span>
      </>
    );
  return (
    <div className={s.scroll}>
      <table className={s.table}>
      <thead>
        <tr>
          <th>Scene</th>
          {METHODS.map((m) => (
            <th key={m.id}>{m.name}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {[...SCENES, last].map((scene) => (
          <tr key={scene} className={cx(scene === last && s.total)}>
            <td>{scene}</td>
            {METHODS.map((m) => (
              <td key={m.id}>{cell(m, scene)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
      </div>
  );
}

// ---- The explorer -------------------------------------------------------------

export function ResultsExplorer() {
  const [method, setMethod] = useState('all');
  const [measure, setMeasure] = useState('psnr');
  const [view, setView] = useState('chart');
  const m = BY_ID[method];
  const isQ = measure in QUALITY;
  let title;
  if (!m) title = isQ ? `${measure.toUpperCase()}: MojoSplat against each official code` : `${MEASURES.find((x) => x.id === measure).label}: speedup over each official code`;
  else title = `${m.name}: ${isQ ? measure.toUpperCase() : MEASURES.find((x) => x.id === measure).label.toLowerCase()} per scene`;
  return (
    <figure className={cx(base.figure, s.root)}>
      <div className={base.figTitle}>{title}</div>
      <div className={s.controls}>
        <span>Method</span>
        <Chips
          label="Method"
          options={[{id: 'all', label: 'All methods'}, ...METHODS.map((x) => ({id: x.id, label: x.name}))]}
          value={method}
          onChange={setMethod}
        />
        <span>Measure</span>
        <Chips label="Measure" options={MEASURES} value={measure} onChange={setMeasure} />
        <span>Show</span>
        <Chips
          label="View"
          options={[
            {id: 'chart', label: 'Chart'},
            {id: 'table', label: 'Table'},
          ]}
          value={view}
          onChange={setView}
        />
      </div>
      {view === 'chart' ? (
        !m ? (
          <AllMethods measure={measure} />
        ) : isQ ? (
          <QualityStrip m={m} measure={measure} />
        ) : (
          <TimeBars m={m} measure={measure} />
        )
      ) : (
        <>
          {m ? <MethodTable m={m} measure={measure} /> : <AllTable measure={measure} />}
          <div className={s.note}>
            <p>
              {m
                ? isQ
                  ? 'First run of each side; the difference is MojoSplat minus official.'
                  : 'The speedup is the official time divided by MojoSplat’s.'
                : isQ
                  ? 'MojoSplat’s first run, with the difference from the official code’s first run in grey.'
                  : 'MojoSplat’s time, with the speedup over the official code in grey.'}
              {!m && measure === 'lpips' && ' LPIPS values are only comparable within one method.'}
            </p>
          </div>
        </>
      )}
      {m && isQ && (
        <div className={s.note}>
          <p>
            <Inline text={METRIC_NOTE[m.id]} /> More in <a href={`#${m.id}`}>the {m.name} section</a>.
          </p>
        </div>
      )}
      {m && !isQ && (
        <div className={s.note}>
          <p>
            How each side was timed: <a href={`#${m.id}`}>the {m.name} section</a>.
          </p>
        </div>
      )}
    </figure>
  );
}

// ---- The summary table at the top --------------------------------------------

export function MethodSummary() {
  return (
    <div className={s.scroll}>
      <table className={cx(s.table, s.summary)}>
        <thead>
          <tr>
            <th>Method</th>
            <th>Mean PSNR</th>
            <th>Training per scene</th>
            <th>Rendering</th>
          </tr>
        </thead>
        <tbody>
          {METHODS.map((m) => (
            <tr key={m.id}>
              <td>
                <a href={`#${m.id}`}>
                  <b>{m.name}</b>
                </a>
                <br />
                <a className={s.paper} href={m.paper.url}>
                  {m.paper.label}
                </a>
                <br />
                <code>--method {m.id}</code>
                {m.default && <span className={s.sub}>(default)</span>}
              </td>
              <td>
                {fmt(m.quality.official.mean[0], 2)} / {fmt(m.quality.ours.mean[0], 2)}
              </td>
              <td>
                {m.perScene.train}
                <span className={s.sub}>{times(m.train.total.speedup)}</span>
              </td>
              <td>
                {m.perScene.fps} FPS
                <span className={s.sub}>{times(m.render.mean.speedup)}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
