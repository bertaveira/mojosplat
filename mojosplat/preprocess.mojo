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
scenes). `primitive_ids[slot]` maps a slot back to its Gaussian. Mean,
conic, opacity and pixel rectangle are packed into one 32-byte record per
slot (one memory sector per random gather).
"""

from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu.primitives import warp
from std.atomic import Atomic, Ordering
from std.bit import count_trailing_zeros, pop_count
from std.math import ceil, exp, floor, log, sqrt
from std.memory import bitcast

from .camera import CAM_CX, CAM_CY, CAM_FAR, CAM_FX, CAM_FY, CAM_H, CAM_NEAR, CAM_POS_X, CAM_POS_Y, CAM_POS_Z, CAM_W
from .common import F32Ptr, I32Ptr, LANE_MASK_DTYPE, LaneMask, U32Ptr, num_blocks
from .sh import sh_to_color

comptime MIN_ALPHA: Float32 = 1.0 / 255.0
comptime MIN_ALPHA_RCP: Float32 = 255.0
comptime DILATION: Float32 = 0.3
comptime MIN_COV2D_DET: Float32 = 1e-6
comptime CLIP_MARGIN: Float32 = 0.15  # Jacobian clamp: 15% outside the image
comptime N_SEQUENTIAL = 4  # tiles tested per thread before the warp helps
comptime PREPROCESS_BLOCK = 128


@always_inline
def pack_bounds(x0: Int, x1: Int, y0: Int, y1: Int) -> UInt64:
    """Pixel rectangle [x0, x1) x [y0, y1) as 4 x 16 bits."""
    return UInt64(x0) | (UInt64(x1) << 16) | (UInt64(y0) << 32) | (UInt64(y1) << 48)


@always_inline
def unpack_bounds(b: UInt64) -> SIMD[DType.int32, 4]:
    """-> (x0, x1, y0, y1)."""
    return SIMD[DType.int32, 4](
        Int32(b & 0xFFFF), Int32((b >> 16) & 0xFFFF), Int32((b >> 32) & 0xFFFF), Int32(b >> 48)
    )


# Packed per-slot record: [mx, my, conic a, conic b, conic c, opacity,
# bounds lo bits, bounds hi bits] (float32 x 8 = 32 bytes).
comptime REC = 8


@always_inline
def store_record(
    rec: F32Ptr, slot: Int, mx: Float32, my: Float32, ca: Float32, cb: Float32,
    cc: Float32, opacity: Float32, bounds: UInt64,
):
    var v = SIMD[DType.float32, REC](
        mx, my, ca, cb, cc, opacity,
        bitcast[DType.float32](UInt32(bounds & 0xFFFFFFFF)), bitcast[DType.float32](UInt32(bounds >> 32)),
    )
    rec.unsafe_offset(slot * REC).unsafe_bitcast[SIMD[DType.float32, REC]]()[] = v


@always_inline
def load_record(rec: F32Ptr, slot: Int) -> SIMD[DType.float32, REC]:
    return rec.unsafe_offset(slot * REC).unsafe_bitcast[SIMD[DType.float32, REC]]()[]


@always_inline
def record_bounds(r: SIMD[DType.float32, REC]) -> UInt64:
    return UInt64(r[6].to_bits()) | (UInt64(r[7].to_bits()) << 32)


@always_inline
def tile_rect[tile_w: Int, tile_h: Int](b: UInt64) -> SIMD[DType.int32, 4]:
    """Pixel rectangle -> tile rectangle (tx0, tx1, ty0, ty1), exclusive max."""
    var p = unpack_bounds(b)
    return SIMD[DType.int32, 4](
        p[0] // Int32(tile_w),
        (p[1] + Int32(tile_w - 1)) // Int32(tile_w),
        p[2] // Int32(tile_h),
        (p[3] + Int32(tile_h - 1)) // Int32(tile_h),
    )


@always_inline
def will_contribute[
    tile_w: Int, tile_h: Int
](
    mx: Float32,  # mean2d - 0.5 (pixel-center coordinates)
    my: Float32,
    ca: Float32,  # conic (a, b, c): power = 0.5 (a dx^2 + c dy^2) + b dx dy
    cb: Float32,
    cc: Float32,
    tile_x: Int,
    tile_y: Int,
    power_threshold: Float32,
) -> Bool:
    """Does the Gaussian reach alpha >= 1/255 at any pixel center of the tile?

    Evaluates it at the point of the tile closest to its center along the
    conic (StopThePop's test, as used by Faster-GS).
    """
    var rmin_x = Float32(tile_x * tile_w)
    var rmin_y = Float32(tile_y * tile_h)
    var rmax_x = Float32((tile_x + 1) * tile_w - 1)
    var rmax_y = Float32((tile_y + 1) * tile_h - 1)

    var x_min_diff = rmin_x - mx
    var x_left: Float32 = 1.0 if x_min_diff >= 0.0 else 0.0
    var not_in_x = x_left + (Float32(1.0) if mx > rmax_x else Float32(0.0))
    var y_min_diff = rmin_y - my
    var y_above: Float32 = 1.0 if y_min_diff >= 0.0 else 0.0
    var not_in_y = y_above + (Float32(1.0) if my > rmax_y else Float32(0.0))
    if not_in_x + not_in_y == 0.0:
        return True  # center inside the tile

    var corner_x = rmax_x + (rmin_x - rmax_x) * x_left
    var corner_y = rmax_y + (rmin_y - rmax_y) * y_above
    var diff_x = mx - corner_x
    var diff_y = my - corner_y
    var dx = Float32(tile_w - 1) if x_min_diff >= 0.0 else -Float32(tile_w - 1)
    var dy = Float32(tile_h - 1) if y_min_diff >= 0.0 else -Float32(tile_h - 1)
    var tx = not_in_y * min(max((dx * ca * diff_x + dx * cb * diff_y) / (dx * ca * dx), 0.0), 1.0)
    var ty = not_in_x * min(max((dy * cb * diff_x + dy * cc * diff_y) / (dy * cc * dy), 0.0), 1.0)
    var ddx = mx - (corner_x + tx * dx)
    var ddy = my - (corner_y + ty * dy)
    var power = 0.5 * (ca * ddx * ddx + cc * ddy * ddy) + cb * ddx * ddy
    return power <= power_threshold


def _preprocess_kernel[
    tile_w: Int, tile_h: Int
](
    means: F32Ptr,  # [N, 3]
    log_scales: F32Ptr,  # [N, 3]
    rotations: F32Ptr,  # [N, 4] (w, x, y, z), unnormalized
    opacity_logits: F32Ptr,  # [N]
    sh0: F32Ptr,  # [N, 3]
    sh_rest: F32Ptr,  # [N, rest_stride, 3]
    cam: SIMD[DType.float32, 32],
    depth_keys: U32Ptr,  # out [slot] depth bits (sort keys)
    slots: I32Ptr,  # out [slot] = slot (sort values)
    primitive_ids: I32Ptr,  # out [slot] Gaussian index
    n_touched: U32Ptr,  # out [slot] exact tile count
    records: F32Ptr,  # out [slot, REC]
    colors: F32Ptr,  # out [slot, 3]
    counters: U32Ptr,  # [2] (n_visible, n_instances), zeroed by the caller
    n: Int32,
    grid_w: Int32,
    grid_h: Int32,
    active_bases: Int32,
    rest_stride: Int32,
):
    var lane = lane_id()
    var idx = block_idx.x * PREPROCESS_BLOCK + thread_idx.x
    # Out-of-range threads stay alive (as inactive) for the warp-wide steps.
    var active = idx < Int(n)
    var g = min(idx, Int(n) - 1)

    var m0 = means[unsafe_offset=g * 3 + 0]
    var m1 = means[unsafe_offset=g * 3 + 1]
    var m2 = means[unsafe_offset=g * 3 + 2]
    var depth = cam[8] * m0 + cam[9] * m1 + cam[10] * m2 + cam[11]
    if depth < cam[CAM_NEAR] or depth > cam[CAM_FAR]:
        active = False
    if warp.vote[LANE_MASK_DTYPE](active) == 0:
        return

    var opacity = 1.0 / (1.0 + exp(-opacity_logits[unsafe_offset=g]))
    if opacity < MIN_ALPHA:
        active = False

    # 3D covariance R diag(s^2) R^T (quaternion normalized implicitly).
    var v0 = exp(2.0 * log_scales[unsafe_offset=g * 3 + 0])
    var v1 = exp(2.0 * log_scales[unsafe_offset=g * 3 + 1])
    var v2 = exp(2.0 * log_scales[unsafe_offset=g * 3 + 2])
    var qr = rotations[unsafe_offset=g * 4 + 0]
    var qx = rotations[unsafe_offset=g * 4 + 1]
    var qy = rotations[unsafe_offset=g * 4 + 2]
    var qz = rotations[unsafe_offset=g * 4 + 3]
    var xx = qx * qx
    var yy = qy * qy
    var zz = qz * qz
    var xy = qx * qy
    var xz = qx * qz
    var yz = qy * qz
    var rx = qr * qx
    var ry = qr * qy
    var rz = qr * qz
    var norm_sq = qr * qr + xx + yy + zz
    if norm_sq < 1e-8:
        active = False
    var s = 2.0 / norm_sq
    var R00 = 1.0 - s * (yy + zz)
    var R01 = s * (xy - rz)
    var R02 = s * (xz + ry)
    var R10 = s * (xy + rz)
    var R11 = 1.0 - s * (xx + zz)
    var R12 = s * (yz - rx)
    var R20 = s * (xz - ry)
    var R21 = s * (yz + rx)
    var R22 = 1.0 - s * (xx + yy)
    var c00 = R00 * R00 * v0 + R01 * R01 * v1 + R02 * R02 * v2
    var c01 = R00 * R10 * v0 + R01 * R11 * v1 + R02 * R12 * v2
    var c02 = R00 * R20 * v0 + R01 * R21 * v1 + R02 * R22 * v2
    var c11 = R10 * R10 * v0 + R11 * R11 * v1 + R12 * R12 * v2
    var c12 = R10 * R20 * v0 + R11 * R21 * v1 + R12 * R22 * v2
    var c22 = R20 * R20 * v0 + R21 * R21 * v1 + R22 * R22 * v2

    # Normalized image coordinates and the EWA Jacobian (clamped near the image).
    var fx = cam[CAM_FX]
    var fy = cam[CAM_FY]
    var cx = cam[CAM_CX]
    var cy = cam[CAM_CY]
    var W = cam[CAM_W]
    var H = cam[CAM_H]
    var x = (cam[0] * m0 + cam[1] * m1 + cam[2] * m2 + cam[3]) / depth
    var y = (cam[4] * m0 + cam[5] * m1 + cam[6] * m2 + cam[7]) / depth
    var x_cl = min(max(x, (-CLIP_MARGIN * W - cx) / fx), ((1.0 + CLIP_MARGIN) * W - cx) / fx)
    var y_cl = min(max(y, (-CLIP_MARGIN * H - cy) / fy), ((1.0 + CLIP_MARGIN) * H - cy) / fy)
    var j11 = fx / depth
    var j13 = -j11 * x_cl
    var j22 = fy / depth
    var j23 = -j22 * y_cl
    # Rows of J * W (W = view rotation)
    var a0 = j11 * cam[0] + j13 * cam[8]
    var a1 = j11 * cam[1] + j13 * cam[9]
    var a2 = j11 * cam[2] + j13 * cam[10]
    var b0 = j22 * cam[4] + j23 * cam[8]
    var b1 = j22 * cam[5] + j23 * cam[9]
    var b2 = j22 * cam[6] + j23 * cam[10]
    var ac0 = a0 * c00 + a1 * c01 + a2 * c02
    var ac1 = a0 * c01 + a1 * c11 + a2 * c12
    var ac2 = a0 * c02 + a1 * c12 + a2 * c22
    var bc0 = b0 * c00 + b1 * c01 + b2 * c02
    var bc1 = b0 * c01 + b1 * c11 + b2 * c12
    var bc2 = b0 * c02 + b1 * c12 + b2 * c22
    var cov_xx = ac0 * a0 + ac1 * a1 + ac2 * a2 + DILATION
    var cov_xy = ac0 * b0 + ac1 * b1 + ac2 * b2
    var cov_yy = bc0 * b0 + bc1 * b1 + bc2 * b2 + DILATION
    var det = cov_xx * cov_yy - cov_xy * cov_xy
    if det < MIN_COV2D_DET:
        active = False
    var ca = cov_yy / det
    var cb = -cov_xy / det
    var cc = cov_xx / det

    var mx = x * fx + cx
    var my = y * fy + cy

    # Opacity-aware rectangle: exact x/y extent of the alpha >= 1/255 ellipse.
    var power_threshold = log(opacity * MIN_ALPHA_RCP)
    var cutoff = 2.0 * power_threshold
    var ext_x = max(sqrt(cov_xx * cutoff) - 0.5, 0.0)
    var ext_y = max(sqrt(cov_yy * cutoff) - 0.5, 0.0)
    var pad_w = Float32(Int(grid_w) * tile_w)
    var pad_h = Float32(Int(grid_h) * tile_h)
    # Clamp in float first: the mean can be far outside the image.
    var x0 = Int(min(pad_w, max(Float32(0.0), floor(mx - ext_x))))
    var x1 = Int(min(pad_w, max(Float32(0.0), ceil(mx + ext_x))))
    var y0 = Int(min(pad_h, max(Float32(0.0), floor(my - ext_y))))
    var y1 = Int(min(pad_h, max(Float32(0.0), ceil(my + ext_y))))
    var tx0 = x0 // tile_w
    var tx1 = (x1 + tile_w - 1) // tile_w
    var ty0 = y0 // tile_h
    var ty1 = (y1 + tile_h - 1) // tile_h
    var tw = tx1 - tx0
    var tile_count = tw * (ty1 - ty0)
    if tile_count == 0:
        active = False
    if warp.vote[LANE_MASK_DTYPE](active) == 0:
        return

    # Exact tile count: a few tiles sequentially, the rest with the warp.
    var smx = mx - 0.5
    var smy = my - 0.5
    var touched = 0
    if active:
        for t in range(min(tile_count, N_SEQUENTIAL)):
            if will_contribute[tile_w, tile_h](
                smx, smy, ca, cb, cc, tx0 + t % tw, ty0 + t // tw, power_threshold
            ):
                touched += 1
    var coop = warp.vote[LANE_MASK_DTYPE](active and tile_count > N_SEQUENTIAL)
    while coop != 0:  # warp-uniform loop over lanes that need help
        var src = UInt32(count_trailing_zeros(coop))
        coop &= coop - 1
        var s_tx0 = Int(warp.shuffle_idx(Int32(tx0), src))
        var s_ty0 = Int(warp.shuffle_idx(Int32(ty0), src))
        var s_tw = Int(warp.shuffle_idx(Int32(tw), src))
        var s_count = Int(warp.shuffle_idx(Int32(tile_count), src))
        var s_mx = warp.shuffle_idx(smx, src)
        var s_my = warp.shuffle_idx(smy, src)
        var s_ca = warp.shuffle_idx(ca, src)
        var s_cb = warp.shuffle_idx(cb, src)
        var s_cc = warp.shuffle_idx(cc, src)
        var s_pt = warp.shuffle_idx(power_threshold, src)
        var t = N_SEQUENTIAL + lane
        while t - lane < s_count:  # uniform trip count across the warp
            var hit = t < s_count and will_contribute[tile_w, tile_h](
                s_mx, s_my, s_ca, s_cb, s_cc, s_tx0 + t % s_tw, s_ty0 + t // s_tw, s_pt
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
        base = Atomic[UInt32].fetch_add[ordering=Ordering.RELAXED](
            counters, UInt32(pop_count(vis_mask))
        )
        _ = Atomic[UInt32].fetch_add[ordering=Ordering.RELAXED](counters.unsafe_offset(1), inst)
    base = warp.shuffle_idx(base, leader)
    if not visible:
        return

    var lt = (LaneMask(1) << LaneMask(lane)) - 1
    var slot = Int(base) + Int(pop_count(vis_mask & lt))
    depth_keys[unsafe_offset=slot] = UInt32(depth.to_bits())
    slots[unsafe_offset=slot] = Int32(slot)
    primitive_ids[unsafe_offset=slot] = Int32(g)
    n_touched[unsafe_offset=slot] = UInt32(touched)
    store_record(records, slot, mx, my, ca, cb, cc, opacity, pack_bounds(x0, x1, y0, y1))
    var dir = SIMD[DType.float32, 4](
        m0 - cam[CAM_POS_X], m1 - cam[CAM_POS_Y], m2 - cam[CAM_POS_Z], 0.0
    )
    var rgb = sh_to_color(sh0, sh_rest, g, dir, Int(active_bases), Int(rest_stride))
    colors[unsafe_offset=slot * 3 + 0] = max(rgb[0], 0.0)
    colors[unsafe_offset=slot * 3 + 1] = max(rgb[1], 0.0)
    colors[unsafe_offset=slot * 3 + 2] = max(rgb[2], 0.0)


def preprocess[
    tile_w: Int, tile_h: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    means: F32Ptr,
    log_scales: F32Ptr,
    rotations: F32Ptr,
    opacity_logits: F32Ptr,
    sh0: F32Ptr,
    sh_rest: F32Ptr,
    cam: SIMD[DType.float32, 32],
    depth_keys: U32Ptr,
    slots: I32Ptr,
    primitive_ids: I32Ptr,
    n_touched: U32Ptr,
    records: F32Ptr,
    colors: F32Ptr,
    counters: U32Ptr,
    n: Int,
    grid_w: Int,
    grid_h: Int,
    active_bases: Int,
    rest_stride: Int,
) raises:
    if n == 0:
        return
    stream.enqueue_function(
        ctx.compile_function[_preprocess_kernel[tile_w, tile_h]](),
        means, log_scales, rotations, opacity_logits, sh0, sh_rest, cam,
        depth_keys, slots, primitive_ids, n_touched, records, colors, counters, Int32(n), Int32(grid_w), Int32(grid_h),
        Int32(active_bases), Int32(rest_stride),
        grid_dim=num_blocks(n, PREPROCESS_BLOCK), block_dim=PREPROCESS_BLOCK,
    )
