"""Per-Gaussian preprocessing for the tile rasterizer (one thread per Gaussian).

Ported from Faster-GS (Hahlbohm et al., CVPR 2026; Apache-2.0,
github.com/nerficg-project/faster-gaussian-splatting, `kernels_inference.cuh`).
In one pass it:

  * activates the raw parameters (exp scales, normalize quaternions,
    sigmoid opacity) -- inputs are the trainable parameters, not activated
    values;
  * projects the 3D covariance with the EWA Jacobian (+0.3 px dilation);
  * computes an opacity-aware screen-space rectangle that is exact for the
    alpha >= 1/255 region;
  * counts the tiles the Gaussian really contributes to (exact ellipse-tile
    test from StopThePop, Radl et al. 2024), large Gaussians cooperatively by
    the whole warp;
  * evaluates spherical harmonics to RGB;
  * appends visible Gaussians to a compacted (depth key, index) list and
    counts the total number of tile instances.

Outputs are *compacted*: the k-th visible Gaussian (in arbitrary order)
writes slot k of every output array, so writes are contiguous and the arrays
read by the later stages are small (they fit in L2 for multi-million splat
scenes). In training, `slot_of[g]` maps a Gaussian to its slot. Mean,
conic, opacity and pixel rectangle are packed into one 32-byte record per
slot (one memory sector per random gather).
"""

from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu.primitives import warp
from std.atomic import Atomic, Ordering
from std.bit import count_trailing_zeros, pop_count
from std.math import ceil, exp, floor

from ..core.camera import CAM_FAR, CAM_NEAR, CAM_POS_X, CAM_POS_Y, CAM_POS_Z
from ..core.linalg import V3, V4
from ..core.splats import AppearanceSettings, ExtraColumns
from ..core.types import F32Ptr, I32Ptr, KernelVector, LANE_MASK_DTYPE, LaneMask, U32Ptr, num_blocks
from .blocks.contracts import Pipeline
from .blocks.shape import QuatScale
from .pipelines import Standard3DGS
from .record import udiv, umod, N_SEQUENTIAL, pack_bounds, store_record

comptime PREPROCESS_BLOCK = 128


def _preprocess_kernel[
    tile_w: Int, tile_h: Int, train: Bool, Pipe: Pipeline
](
    means: F32Ptr,  # [N, 3]
    log_scales: F32Ptr,  # [N, 3]
    rotations: F32Ptr,  # [N, 4] (w, x, y, z), unnormalized
    opacity_logits: F32Ptr,  # [N]
    col0: F32Ptr,  # columns after the core ones (see ExtraColumns)
    col1: F32Ptr,
    col2: F32Ptr,
    col3: F32Ptr,
    cam: KernelVector[DType.float32, 32],
    depth_keys: U32Ptr,  # out [slot] depth bits (sort keys)
    slots: I32Ptr,  # out [slot] = slot (sort values)
    n_touched: U32Ptr,  # out [slot] exact tile count
    records: F32Ptr,  # out [slot, REC]
    colors: F32Ptr,  # out [slot, 3] color, not clamped
    response_extra: F32Ptr,  # out [slot, Response.EXTRA]
    counters: U32Ptr,  # [2] (n_visible, n_instances), zeroed by the caller
    slot_of: I32Ptr,  # out [N] (train only): slot of each Gaussian, -1 if culled
    n: Int32,
    grid_w: Int32,
    grid_h: Int32,
    settings: KernelVector[DType.int32, 4],
):
    var lane = lane_id()
    var idx = block_idx.x * PREPROCESS_BLOCK + thread_idx.x
    # Out-of-range threads stay alive (as inactive) for the warp-wide steps.
    var active = idx < Int(n)
    var g = min(idx, Int(n) - 1)
    comptime if train:
        if active:
            slot_of[unsafe_offset=g] = -1  # overwritten below if visible

    comptime P = Pipe.Place
    comptime R = Pipe.Resp
    comptime dt = P.dtype
    var cam_v = cam.value.cast[dt]()
    var mean = V3[dt].load(means, g)
    var depth = P.depth(mean, cam_v)
    if depth < cam_v[CAM_NEAR] or depth > cam_v[CAM_FAR]:
        active = False
    if warp.vote[LANE_MASK_DTYPE](active) == 0:
        return

    var opacity = 1.0 / (1.0 + exp(-opacity_logits[unsafe_offset=g]))
    if opacity < R.MIN_OPACITY:
        active = False

    var shape = QuatScale[dt](V3[dt].load(log_scales, g), V4[dt].load(rotations, g))
    if shape.norm_sq < 1e-8:
        active = False
    var place = P(mean, shape, cam_v)
    var mean2d = place.mean2d().f32()
    var resp = R(mean2d, place.cov2d().f32())
    if not resp.valid():
        active = False
    var rec = resp.record(opacity)
    var mx = mean2d.x
    var my = mean2d.y

    # Opacity-aware rectangle holding every visible pixel.
    var ext = resp.extent(opacity)
    var ext_x = ext[0]
    var ext_y = ext[1]
    var pad_w = Float32(Int(grid_w) * tile_w)
    var pad_h = Float32(Int(grid_h) * tile_h)
    # Clamp in float first: the mean can be far outside the image.
    var x0 = Int(min(pad_w, max(Float32(0.0), floor(mx - ext_x))))
    var x1 = Int(min(pad_w, max(Float32(0.0), ceil(mx + ext_x))))
    var y0 = Int(min(pad_h, max(Float32(0.0), floor(my - ext_y))))
    var y1 = Int(min(pad_h, max(Float32(0.0), ceil(my + ext_y))))
    # 32-bit: ptxas (CUDA 12.9, sm_86) miscompiles the 64-bit tile count when ty0 == 0.
    var tx0_32 = Int32(UInt32(x0) // UInt32(tile_w))
    var tx1_32 = Int32(UInt32(x1 + tile_w - 1) // UInt32(tile_w))
    var ty0_32 = Int32(UInt32(y0) // UInt32(tile_h))
    var ty1_32 = Int32(UInt32(y1 + tile_h - 1) // UInt32(tile_h))
    var tx0 = Int(tx0_32)
    var ty0 = Int(ty0_32)
    var tw = Int(tx1_32 - tx0_32)
    var tile_count = Int((tx1_32 - tx0_32) * (ty1_32 - ty0_32))
    if tile_count == 0:
        active = False
    if warp.vote[LANE_MASK_DTYPE](active) == 0:
        return

    # Exact tile count: a few tiles sequentially, the rest with the warp.
    var args = R.tile_args(rec)
    var touched = 0
    if active:
        for t in range(min(tile_count, N_SEQUENTIAL)):
            if R.touches_tile[tile_w, tile_h](args, tx0 + umod(t, tw), ty0 + udiv(t, tw)):
                touched += 1
    var coop = warp.vote[LANE_MASK_DTYPE](active and tile_count > N_SEQUENTIAL)
    while coop != 0:  # warp-uniform loop over lanes that need help
        var src = UInt32(count_trailing_zeros(coop))
        coop &= coop - 1
        var s_tx0 = Int(warp.shuffle_idx(Int32(tx0), src))
        var s_ty0 = Int(warp.shuffle_idx(Int32(ty0), src))
        var s_tw = Int(warp.shuffle_idx(Int32(tw), src))
        var s_count = Int(warp.shuffle_idx(Int32(tile_count), src))
        var s_args = SIMD[DType.float32, 8](0.0)
        comptime for k in range(R.TILE_ARGS):
            s_args[k] = warp.shuffle_idx(args[k], src)
        var t = N_SEQUENTIAL + lane
        while t - lane < s_count:  # uniform trip count across the warp
            var hit = t < s_count and R.touches_tile[tile_w, tile_h](
                s_args, s_tx0 + umod(t, s_tw), s_ty0 + udiv(t, s_tw)
            )
            var hits = warp.vote[LANE_MASK_DTYPE](hit)
            if lane == Int(src):
                touched += Int(pop_count(hits))
            t += WARP_SIZE

    var visible = active and touched > 0

    # Warp-aggregated slot allocation.
    var vis_mask = warp.vote[LANE_MASK_DTYPE](visible)
    if vis_mask == 0:
        return
    var leader = UInt32(count_trailing_zeros(vis_mask))
    var inst = warp.sum(UInt32(touched) if visible else UInt32(0))
    var base = UInt32(0)
    if lane == Int(leader):
        base = Atomic[UInt32].fetch_add[ordering=Ordering.RELAXED](counters, UInt32(pop_count(vis_mask)))
        _ = Atomic[UInt32].fetch_add[ordering=Ordering.RELAXED](counters.unsafe_offset(1), inst)
    base = warp.shuffle_idx(base, leader)
    if not visible:
        return

    var lt = (LaneMask(1) << LaneMask(lane)) - 1
    var slot = Int(base) + Int(pop_count(vis_mask & lt))
    depth_keys[unsafe_offset=slot] = UInt32(depth.cast[DType.float32]().to_bits())
    slots[unsafe_offset=slot] = Int32(slot)
    comptime if train:
        slot_of[unsafe_offset=g] = Int32(slot)
    n_touched[unsafe_offset=slot] = UInt32(touched)
    store_record(records, slot, rec, pack_bounds(x0, x1, y0, y1))
    var dir = SIMD[DType.float32, 4](
        mean.x.cast[DType.float32]() - cam[CAM_POS_X],
        mean.y.cast[DType.float32]() - cam[CAM_POS_Y],
        mean.z.cast[DType.float32]() - cam[CAM_POS_Z],
        0.0,
    )
    var cols = ExtraColumns(col0, col1, col2, col3)
    comptime if R.EXTRA > 0:
        var extra = R.extra(cols.skip(Pipe.Appear.NUM_COLUMNS), g)
        comptime for k in range(R.EXTRA):
            response_extra[unsafe_offset=slot * R.EXTRA + k] = extra[k]
    # Stored unclamped: blending clamps at 0, the backward needs the mask.
    var rgb = Pipe.Appear.color(cols, g, dir, settings.value)
    colors[unsafe_offset=slot * 3 + 0] = rgb[0]
    colors[unsafe_offset=slot * 3 + 1] = rgb[1]
    colors[unsafe_offset=slot * 3 + 2] = rgb[2]


def preprocess[
    tile_w: Int, tile_h: Int, train: Bool = False, Pipe: Pipeline = Standard3DGS
](
    ctx: DeviceContext,
    stream: DeviceStream,
    means: F32Ptr,
    log_scales: F32Ptr,
    rotations: F32Ptr,
    opacity_logits: F32Ptr,
    extras: ExtraColumns,
    cam: SIMD[DType.float32, 32],
    depth_keys: U32Ptr,
    slots: I32Ptr,
    n_touched: U32Ptr,
    records: F32Ptr,
    colors: F32Ptr,
    response_extra: F32Ptr,
    counters: U32Ptr,
    slot_of: I32Ptr,
    n: Int,
    grid_w: Int,
    grid_h: Int,
    settings: AppearanceSettings,
) raises:
    if n == 0:
        return
    stream.enqueue_function(
        ctx.compile_function[_preprocess_kernel[tile_w, tile_h, train, Pipe]](),
        means,
        log_scales,
        rotations,
        opacity_logits,
        extras.c0,
        extras.c1,
        extras.c2,
        extras.c3,
        KernelVector[DType.float32, 32](cam),
        depth_keys,
        slots,
        n_touched,
        records,
        colors,
        response_extra,
        counters,
        slot_of,
        Int32(n),
        Int32(grid_w),
        Int32(grid_h),
        KernelVector[DType.int32, 4](settings),
        grid_dim=num_blocks(n, PREPROCESS_BLOCK),
        block_dim=PREPROCESS_BLOCK,
    )
