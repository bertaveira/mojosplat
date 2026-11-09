import React, {useState} from 'react';
import {Chips, Panel, cx, tone} from './ui';
import s from './arch.module.css';

const STATUS = {
  write: {label: 'Write new', tone: 'appearance'},
  edit: {label: 'Edit', tone: 'loaded'},
  none: {label: 'No change', tone: 'shape'},
  blocked: {label: 'Not a seam', tone: 'response'},
};

const VERDICT = {
  yes: {label: 'Supported by the contracts', tone: 'appearance'},
  some: {label: 'Partly: the contract exists, other code assumes today’s choice', tone: 'loaded'},
  no: {label: 'Not supported yet', tone: 'response'},
};

const COMMON_NEW = [
  ['render/pipelines.mojo', 'edit', 'Add a struct with a new ID, or add the pipeline struct next to your test.'],
  ['tests/render/test_block_gradients.mojo', 'edit', 'Finite-difference check of every backward. A generic block that nothing instantiates is not even compiled.'],
  ['Your program', 'edit', '`Trainer[MyPipeline](...)` or `rast.forward[MyPipeline](...)`. tools/train and tools/eval pick one through `--method`, which needs a `Method` that names it.'],
];

const CHANGES = {
  footprint: {
    label: 'New 2D footprint',
    verdict: 'yes',
    summary: 'A different alpha for each pixel, still built from a 2D position and covariance. Kernels are unchanged.',
    rows: [
      ['render/blocks/response/<name>.mojo', 'write', 'Implement Response and LoadedResponse, as in the walkthrough below.'],
      ...COMMON_NEW,
      ['render/preprocess.mojo, blend*.mojo, instances.mojo', 'none', 'Unchanged; they are instantiated with your block.'],
    ],
  },
  params: {
    label: 'Footprint with its own per-Gaussian parameters',
    verdict: 'yes',
    summary: 'What ScaledGaussian in tests/render/test_pipeline_seams.mojo does. Limited to 2 extra columns while the appearance uses 2 of the 4 ExtraColumns.',
    rows: [
      ['render/blocks/response/<name>.mojo', 'write', 'Also `columns()`, `EXTRA` and `extra()`, a larger `GRAD`, `grad_index`, `write_gradients`, `zero_gradients`.'],
      ...COMMON_NEW,
      ['Trainer, Adam, gather, PLY', 'none', 'Trainer adds the columns from `columns()`, Adam uses each ColumnSpec’s learning rate, gather carries them through densification, and they save as `extra_<name>_<i>`.'],
    ],
  },
  camera: {
    label: 'New camera model',
    verdict: 'some',
    summary: 'The Placement contract can hold one that still outputs a 2D position and covariance. Camera and the dataset are pinhole only.',
    rows: [
      ['render/blocks/placement/<name>.mojo', 'write', 'Implement Placement: `depth`, constructor, `mean2d`, `cov2d`, `backward`.'],
      ['render/pipelines.mojo', 'edit', 'Name it in a pipeline.'],
      ['core/camera.mojo', 'edit', '`Camera` has fx, fy, cx, cy and `packed()` fills a 32-float vector. Indices 18 to 21 and 27 to 31 are unused.'],
      ['train/dataset.mojo', 'edit', 'Builds pinhole `Camera`s from COLMAP.'],
    ],
  },
  appearance: {
    label: 'New appearance, not SH',
    verdict: 'some',
    summary: 'The Appearance trait exists, but the rest of the model and training code assumes spherical harmonics.',
    rows: [
      ['render/blocks/appearance/<name>.mojo', 'write', 'Implement Appearance: `columns`, `init_column`, `color`, `backward`, `zero_gradients`.'],
      ['model/splats.mojo', 'edit', 'The constructor always adds `sh0` and `sh_rest` as columns 4 and 5, and `Attr` hard-codes them. An appearance with other than 2 columns would misalign the Response’s columns.'],
      ['train/trainer.mojo', 'edit', 'Initializes SH from points, has SH learning rates and degree schedule, runs with `write_all=False` and calls `Adam.step(splats, ShRestGrad)`.'],
      ['train/adam.mojo', 'edit', 'The SH-rest update imports `sh_rest_basis` and reads `Attr.SH_REST`.'],
      ['contracts.mojo', 'none', '`Appearance` has `columns()` and `init_column`, and Trainer adds those columns, but after the six standard ones, while `ExtraColumns` still starts at `sh0` and `sh_rest`.'],
    ],
  },
  shape: {
    label: 'New 3D shape parameterization',
    verdict: 'no',
    summary: 'Log scales and a quaternion are assumed throughout. The shape is not part of a pipeline.',
    rows: [
      ['render/blocks/contracts.mojo', 'blocked', '`Placement.__init__` and `backward` take a `QuatScale`.'],
      ['render/preprocess*.mojo', 'blocked', 'Kernels build `QuatScale` directly from `log_scales` and `rotations`.'],
      ['core/splats.mojo, model/splats.mojo', 'blocked', 'The four core columns are fields of `SplatView` and fixed kernel arguments.'],
      ['train/densify.mojo', 'blocked', 'Reads `log_scales` and `rotations` for its clone and split decisions.'],
    ],
  },
  channels: {
    label: 'More than 3 output channels',
    verdict: 'no',
    summary: 'Blending is three channels throughout. Features would need the blend kernels and their state changed.',
    rows: [
      ['render/rasterizer.mojo', 'blocked', '`CDIM = 3`, and `colors` is `[slot, 3]`.'],
      ['render/blend.mojo', 'blocked', '`comptime assert CDIM <= 3, "slab holds 3 color channels"`.'],
      ['render/blend_backward.mojo', 'blocked', 'Channel loops are written for 3, and each snapshot is (color, T) in 4 floats.'],
      ['contracts.mojo', 'blocked', '`Appearance.color` returns `SIMD[float32, 4]`, with lane 3 unused.'],
    ],
  },
  record: {
    label: 'More than 6 floats in the record',
    verdict: 'some',
    summary: 'Use EXTRA first: up to 8 more floats per slot in a separate array. Enlarging the record itself is a change to the format.',
    rows: [
      ['render/blocks/response/<name>.mojo', 'write', '`EXTRA` up to 8. Each blend reads them from `response_extra`.'],
      ['render/record.mojo', 'blocked', '`REC = 8` and `REC_BOUNDS = 6` define the format; loads and stores are single 32-byte-aligned accesses.'],
      ['contracts.mojo', 'blocked', 'Every record-taking method is typed `SIMD[float32, REC]`.'],
    ],
  },
  ray: {
    label: 'Evaluating along a ray',
    verdict: 'no',
    summary: 'The current Response contract fits footprints evaluated in 2D. Per-hit 3D evaluation would extend it or need its own blend kernel.',
    rows: [
      ['render/blocks/contracts.mojo', 'blocked', '`Response.__init__` takes `mean2d, cov2d`, and `backward` returns `ScreenGrad` (2D position and covariance gradients).'],
      ['render/blocks/contracts.mojo', 'blocked', '`Placement.backward` takes those same 2D gradients.'],
      ['render/blend*.mojo', 'blocked', 'They own the tile walk and front-to-back compositing around `Loaded.eval(px, py)`.'],
    ],
  },
  tools: {
    label: 'Choosing a pipeline in tools/train',
    verdict: 'some',
    summary: 'The pipeline is a compile-time parameter, so the program must name it. `--method` chooses among the methods a tool names in its code.',
    rows: [
      ['tools/train.mojo, tools/eval.mojo', 'edit', '`--method` selects a `Method`, and `run[M: Method]` uses `M.Pipe`, as in `Trainer[M.Pipe](ctx, ds^, config)`. `FasterGS` is the only one registered: add a `Method` that names your pipeline and a branch for it in `main`.'],
      ['train/trainer.mojo', 'none', '`TrainConfig` has no pipeline field: a run-time option can only choose among the pipelines compiled into the tool.'],
      ['viewer/server.mojo, tools/render, bench', 'blocked', 'Each owns a `Rasterizer` and calls `render` with no pipeline, so these use Standard3DGS whatever you trained. `tools/train --viewer` raises for any other pipeline.'],
    ],
  },
};

function rich(text) {
  return text.split(/(`[^`]+`)/g).map((p, i) => (p.startsWith('`') ? <code key={i}>{p.slice(1, -1)}</code> : p));
}

export default function ChangeImpact() {
  const [id, setId] = useState('footprint');
  const c = CHANGES[id];
  const v = VERDICT[c.verdict];
  return (
    <Panel title="What a change touches" caption="Select a change. Everything here reflects the current code; none of it is a plan.">
      <Chips label="Change" options={Object.entries(CHANGES).map(([k, x]) => ({id: k, label: x.label}))} value={id} onChange={setId} />
      <div className={cx(s.verdict, tone(v.tone))}>{v.label}</div>
      <p className={s.small}>{rich(c.summary)}</p>
      {c.rows.map(([file, status, note]) => (
        <div key={file + note} className={s.touch}>
          <span className={s.touchFile}>{file}</span>
          <span>
            <span className={cx(s.tag, tone(STATUS[status].tone))}>{STATUS[status].label}</span>
          </span>
          <span>{rich(note)}</span>
        </div>
      ))}
    </Panel>
  );
}
