import React, {useState} from 'react';
import {Chips, Code, Panel, cx} from './ui';
import s from './arch.module.css';

const PIPES = {
  standard: {
    label: 'Standard3DGS',
    id: 0,
    file: 'render/pipelines.mojo',
    place: 'PinholeEWA[DType.float32]',
    resp: 'Gaussian2D',
    appear: 'SphericalHarmonics',
    extra: false,
    note: 'Gaussian2D sets EXTRA = 0, so the branch on R.EXTRA is not generated at all.',
  },
  scaled: {
    label: 'ScaledPipeline (test only)',
    id: 99,
    file: 'tests/render/test_pipeline_seams.mojo',
    place: 'PinholeEWA[DType.float32]',
    resp: 'ScaledGaussian',
    appear: 'SphericalHarmonics',
    extra: true,
    note: 'ScaledGaussian sets EXTRA = 1, so this version of the kernel also reads the block’s own column and saves one float per slot.',
  },
};

export default function Anatomy() {
  const [id, setId] = useState('standard');
  const p = PIPES[id];
  return (
    <Panel
      title="One kernel, built from a pipeline's blocks"
      caption={<>The kernel text does not change. Selecting a pipeline changes what <code>P</code>, <code>R</code> and <code>Pipe.Appear</code> stand for. {p.note}</>}>
      <Chips label="Pipeline" options={Object.entries(PIPES).map(([k, v]) => ({id: k, label: v.label}))} value={id} onChange={setId} />
      <div>
        <div>
          <span className={s.fileTag}>render/preprocess.mojo (excerpt)</span>
          <Code
            mark={{
              'P.depth': 'placement',
              'P(mean, shape, cam_v)': 'placement',
              'R.MIN_OPACITY': 'response',
              'R(mean2d, place.cov2d().f32())': 'response',
              'resp.valid()': 'response',
              'resp.record(opacity)': 'response',
              'R.tile_args(rec)': 'response',
              'R.touches_tile': 'response',
              'R.extra': 'response',
              'Pipe.Appear.color': 'appearance',
            }}>
            {`def _preprocess_kernel[..., Pipe: Pipeline](...):
    comptime P = Pipe.Place
    comptime R = Pipe.Resp
    # thread id, near/far test, warp vote: kernel
    var depth = P.depth(mean, cam_v)
    var opacity = 1.0 / (1.0 + exp(-logit))
    if opacity < R.MIN_OPACITY: active = False

    var shape = QuatScale[dt](log_scales, rotations)
    var place = P(mean, shape, cam_v)
    var resp = R(mean2d, place.cov2d().f32())
    if not resp.valid(): active = False
    var rec = resp.record(opacity)
    # exact tile count, slot allocation: kernel
    var args = R.tile_args(rec)
    R.touches_tile[tile_w, tile_h](args, tx, ty)
${p.extra ? '' : '    # not generated for this pipeline:\n'}    comptime if R.EXTRA > 0:
        var extra = R.extra(cols.skip(Pipe.Appear.NUM_COLUMNS), g)
    var rgb = Pipe.Appear.color(cols, g, dir, settings)`}
          </Code>
        </div>
        <div style={{marginTop: '0.75rem'}}>
          <div className={s.detail} style={{marginBottom: '0.5rem'}}>
            <div className={s.detailTitle}>
              <span className={s.mono}>{p.label.replace(' (test only)', '')}</span>
            </div>
            <div className={s.fileTag}>{p.file}</div>
            <div>
              <div>
                <strong>Place</strong> <span className={s.mono}>{p.place}</span>
              </div>
              <div>
                <strong>Resp</strong> <span className={s.mono}>{p.resp}</span>
              </div>
              <div>
                <strong>Appear</strong> <span className={s.mono}>{p.appear}</span>
              </div>
              <div>
                <strong>ID</strong> <span className={s.mono}>{p.id}</span>
              </div>
            </div>
          </div>
          <p className={s.small}>The excerpt is shortened: argument lists and the code between the highlighted lines are left out.</p>
        </div>
      </div>
    </Panel>
  );
}
