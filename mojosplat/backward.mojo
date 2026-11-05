"""Backward pass of the Faster-GS style rasterizer.

Ported from Faster-GS (Hahlbohm et al., CVPR 2026; Apache-2.0,
github.com/nerficg-project/faster-gaussian-splatting, `kernels_backward.cuh`),
adapted to MojoSplat's compacted slots and HWC images.

blend backward (per-Gaussian, after Taming-3DGS): one warp per bucket of 32
consecutive instances of a tile; lane k owns the k-th Gaussian. The tile's
pixels stream through the warp like a systolic array: lane 0 picks up a
pixel's saved state ((color, T) before the bucket, from the training
forward), applies its Gaussian and hands the updated state to lane 1 with a
shuffle, and so on. Each lane accumulates its Gaussian's gradients in
registers over all pixels and issues one set of atomics at the end
(instead of one per pixel as in a per-pixel backward).

preprocess backward (one thread per Gaussian, in Gaussian order so every
gradient write is coalesced): culled Gaussians get zero gradients, visible
ones go from their slot's 2D gradients (mean2d, conic, color) back through
the EWA projection, the activations and spherical harmonics to the raw
parameters. All gradient buffers are fully written (no zero-init needed).

Gradient conventions follow Faster-GS: the conic / 2D covariance
off-diagonal gradients are per entry of the symmetric matrix (half of the
gradient w.r.t. the single off-diagonal value).
"""

from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from max.gpu.sync import syncwarp
from std.atomic import Atomic, Ordering
from std.math import exp, exp2, sqrt
from std.memory import stack_allocation

from .camera import CAM_CX, CAM_CY, CAM_FX, CAM_FY, CAM_H, CAM_POS_X, CAM_POS_Y, CAM_POS_Z, CAM_W
from .common import BLOCK_SIZE, F32Ptr, I32Ptr, U32Ptr, num_blocks
from .preprocess import CLIP_MARGIN, DILATION, MIN_ALPHA, load_record
from .sh import sh_to_color_backward

comptime ONE_MINUS_ALPHA_EPS: Float32 = 1e-6
comptime LOG2E: Float32 = 1.4426950408889634


@always_inline
def _atomic_add(p: F32Ptr, i: Int, v: Float32):
    _ = Atomic[Float32].fetch_add[ordering=Ordering.RELAXED](p.unsafe_offset(i), v)


def _blend_backward_kernel[
    tile_w: Int, tile_h: Int
](
    tile_offsets: U32Ptr,  # [n_tiles + 1]
    bucket_start: U32Ptr,  # [n_tiles + 1]
    bucket_tile: U32Ptr,  # [n_buckets]
    tile_max_n: U32Ptr,  # [n_tiles]
    instance_ids: I32Ptr,  # [M] slots
    records: F32Ptr,  # [slot, REC]
    colors: F32Ptr,  # [slot, 3] unclamped
    background: SIMD[DType.float32, 4],
    image: F32Ptr,  # [H, W, 3] forward output
    grad_image: F32Ptr,  # [H, W, 3]
    final_T: F32Ptr,  # [H * W]
    n_processed: U32Ptr,  # [H * W]
    snapshots: F32Ptr,  # [n_buckets, BLOCK, 4]
    grad_mean2d: F32Ptr,  # out (accumulated) [slot, 2]
    grad_conic: F32Ptr,  # out (accumulated) [slot, 3]
    grad_opacity: F32Ptr,  # out (accumulated) [slot] w.r.t. the opacity logit
    grad_color: F32Ptr,  # out (accumulated) [slot, 3] w.r.t. the unclamped color
    n_tiles: Int32,
    width: Int32,
    height: Int32,
    grid_w: Int32,
):
    comptime BLOCK = tile_w * tile_h
    var bucket = block_idx.x
    if bucket >= Int(bucket_start[unsafe_offset=Int(n_tiles)]):
        return
    var tile = Int(bucket_tile[unsafe_offset=bucket])
    var start = Int(tile_offsets[unsafe_offset=tile])
    var count = Int(tile_offsets[unsafe_offset=tile + 1]) - start
    var tile_bucket = bucket - Int(bucket_start[unsafe_offset=tile])
    if tile_bucket * WARP_SIZE >= Int(tile_max_n[unsafe_offset=tile]):
        return

    var lane = lane_id()
    var tp = tile_bucket * WARP_SIZE + lane  # this lane's Gaussian, tile-local
    var valid = tp < count
    var g = 0
    var mx: Float32 = 0
    var my: Float32 = 0
    var ca: Float32 = 0
    var cb: Float32 = 0
    var cc: Float32 = 0
    var opacity: Float32 = 0
    var col = SIMD[DType.float32, 4](0.0)
    var col_factor = SIMD[DType.float32, 4](0.0)
    if valid:
        g = Int(instance_ids[unsafe_offset=start + tp])
        var rec = load_record(records, g)
        mx = rec[0]
        my = rec[1]
        ca = rec[2]
        cb = rec[3]
        cc = rec[4]
        opacity = rec[5]
        comptime for k in range(3):
            var c = colors[unsafe_offset=g * 3 + k]
            col[k] = max(c, 0.0)
            col_factor[k] = 1.0 if c >= 0.0 else 0.0
    var ea = -0.5 * LOG2E * ca
    var eb = -LOG2E * cb
    var ec = -0.5 * LOG2E * cc

    var tile_x0 = (tile % Int(grid_w)) * tile_w
    var tile_y0 = (tile // Int(grid_w)) * tile_h
    var W = Int(width)
    var H = Int(height)

    var s_last = stack_allocation[WARP_SIZE, UInt32, address_space=AddressSpace.SHARED]()
    var s_after = stack_allocation[WARP_SIZE * 4, Float32, address_space=AddressSpace.SHARED]()
    var s_grad = stack_allocation[WARP_SIZE * 4, Float32, address_space=AddressSpace.SHARED]()
    var snap = snapshots.unsafe_offset(bucket * BLOCK * 4)

    # Gradient accumulators for this lane's Gaussian.
    var d_mx: Float32 = 0
    var d_my: Float32 = 0
    var d_ca: Float32 = 0
    var d_cb: Float32 = 0
    var d_cc: Float32 = 0
    var d_op: Float32 = 0
    var d_col = SIMD[DType.float32, 4](0.0)

    # Pixel state handed from lane to lane.
    var last = UInt32(0)
    var after_r: Float32 = 0  # color still to come from this Gaussian onwards
    var after_g: Float32 = 0
    var after_b: Float32 = 0
    var T: Float32 = 0
    var gr: Float32 = 0  # dL/d(pixel color)
    var gg: Float32 = 0
    var gb: Float32 = 0
    var g_alpha_bg: Float32 = 0  # -T_final * dot(dL/dpixel, background)

    for i in range(BLOCK + WARP_SIZE - 1):
        if i % WARP_SIZE == 0:
            syncwarp()
            var local = i + lane
            if local < BLOCK:
                var x = tile_x0 + local % tile_w
                var y = tile_y0 + local // tile_w
                var s = snap.unsafe_offset(local * 4).unsafe_bitcast[SIMD[DType.float32, 4]]()[]
                var img = SIMD[DType.float32, 4](0.0)
                var gimg = SIMD[DType.float32, 4](0.0)
                var fT: Float32 = 0
                var lst = UInt32(0)
                if x < W and y < H:
                    var p = y * W + x
                    comptime for k in range(3):
                        img[k] = image[unsafe_offset=p * 3 + k]
                        gimg[k] = grad_image[unsafe_offset=p * 3 + k]
                    fT = final_T[unsafe_offset=p]
                    lst = n_processed[unsafe_offset=p]
                comptime for k in range(3):
                    s_after[unsafe_offset=lane * 4 + k] = img[k] - fT * background[k] - s[k]
                    s_grad[unsafe_offset=lane * 4 + k] = gimg[k]
                s_after[unsafe_offset=lane * 4 + 3] = s[3]
                s_grad[unsafe_offset=lane * 4 + 3] = -fT * (
                    gimg[0] * background[0] + gimg[1] * background[1] + gimg[2] * background[2]
                )
                s_last[unsafe_offset=lane] = lst
            syncwarp()

        if i > 0:
            last = warp.shuffle_up(last, UInt32(1))
            after_r = warp.shuffle_up(after_r, UInt32(1))
            after_g = warp.shuffle_up(after_g, UInt32(1))
            after_b = warp.shuffle_up(after_b, UInt32(1))
            T = warp.shuffle_up(T, UInt32(1))
            gr = warp.shuffle_up(gr, UInt32(1))
            gg = warp.shuffle_up(gg, UInt32(1))
            gb = warp.shuffle_up(gb, UInt32(1))
            g_alpha_bg = warp.shuffle_up(g_alpha_bg, UInt32(1))

        var idx = i - lane  # pixel this lane handles now
        if lane == 0 and idx < BLOCK:
            var k = i % WARP_SIZE
            last = s_last[unsafe_offset=k]
            after_r = s_after[unsafe_offset=k * 4 + 0]
            after_g = s_after[unsafe_offset=k * 4 + 1]
            after_b = s_after[unsafe_offset=k * 4 + 2]
            T = s_after[unsafe_offset=k * 4 + 3]
            gr = s_grad[unsafe_offset=k * 4 + 0]
            gg = s_grad[unsafe_offset=k * 4 + 1]
            gb = s_grad[unsafe_offset=k * 4 + 2]
            g_alpha_bg = s_grad[unsafe_offset=k * 4 + 3]

        if not valid or idx < 0 or idx >= BLOCK or tp >= Int(last):
            continue
        var x = tile_x0 + idx % tile_w
        var y = tile_y0 + idx // tile_w
        if x >= W or y >= H:
            continue

        var dx = mx - (Float32(x) + 0.5)
        var dy = my - (Float32(y) + 0.5)
        var power = dx * (ea * dx + eb * dy) + ec * dy * dy
        var gaussian = exp2(min(power, 0.0))
        var alpha = opacity * gaussian
        if alpha < MIN_ALPHA:
            continue
        var w = T * alpha
        d_col[0] += w * gr
        d_col[1] += w * gg
        d_col[2] += w * gb
        after_r -= w * col[0]
        after_g -= w * col[1]
        after_b -= w * col[2]
        var one_minus = 1.0 - alpha
        var rcp = 1.0 / max(one_minus, ONE_MINUS_ALPHA_EPS)
        var dL_dalpha = (
            (T * col[0] - after_r * rcp) * gr
            + (T * col[1] - after_g * rcp) * gg
            + (T * col[2] - after_b * rcp) * gb
            + g_alpha_bg * rcp
        )
        d_op += gaussian * dL_dalpha
        var helper = -alpha * dL_dalpha
        d_ca += 0.5 * helper * dx * dx
        d_cb += 0.5 * helper * dx * dy
        d_cc += 0.5 * helper * dy * dy
        d_mx += helper * (ca * dx + cb * dy)
        d_my += helper * (cb * dx + cc * dy)
        T *= one_minus

    if valid:
        _atomic_add(grad_mean2d, g * 2 + 0, d_mx)
        _atomic_add(grad_mean2d, g * 2 + 1, d_my)
        _atomic_add(grad_conic, g * 3 + 0, d_ca)
        _atomic_add(grad_conic, g * 3 + 1, d_cb)
        _atomic_add(grad_conic, g * 3 + 2, d_cc)
        _atomic_add(grad_opacity, g, opacity * (1.0 - opacity) * d_op)
        comptime for k in range(3):
            _atomic_add(grad_color, g * 3 + k, d_col[k] * col_factor[k])


def _preprocess_backward_kernel(
    means: F32Ptr,
    log_scales: F32Ptr,
    rotations: F32Ptr,
    sh_rest: F32Ptr,
    cam: SIMD[DType.float32, 32],
    slot_of: I32Ptr,  # [N] slot of each Gaussian, -1 if culled
    grad_mean2d: F32Ptr,  # [slot, 2]
    grad_conic: F32Ptr,  # [slot, 3]
    grad_opacity_slot: F32Ptr,  # [slot]
    grad_color: F32Ptr,  # [slot, 3]
    out_means: F32Ptr,  # [N, 3]
    out_log_scales: F32Ptr,  # [N, 3]
    out_rotations: F32Ptr,  # [N, 4]
    out_opacity: F32Ptr,  # [N]
    out_sh0: F32Ptr,  # [N, 3]
    out_sh_rest: F32Ptr,  # [N, rest_stride, 3]
    densification: F32Ptr,  # [2, N] (count, sum |dL/dmean2d_ndc|) if has_densification
    has_densification: Int32,  # Bool is not DevicePassable
    n: Int32,
    active_bases: Int32,
    rest_stride: Int32,
):
    var g = block_idx.x * BLOCK_SIZE + thread_idx.x
    if g >= Int(n):
        return
    var s = Int(slot_of[unsafe_offset=g])
    if s < 0:
        comptime for k in range(3):
            out_means[unsafe_offset=g * 3 + k] = 0.0
            out_log_scales[unsafe_offset=g * 3 + k] = 0.0
            out_sh0[unsafe_offset=g * 3 + k] = 0.0
        comptime for k in range(4):
            out_rotations[unsafe_offset=g * 4 + k] = 0.0
        out_opacity[unsafe_offset=g] = 0.0
        for k in range(Int(rest_stride) * 3):
            out_sh_rest[unsafe_offset=g * Int(rest_stride) * 3 + k] = 0.0
        return

    var m0 = means[unsafe_offset=g * 3 + 0]
    var m1 = means[unsafe_offset=g * 3 + 1]
    var m2 = means[unsafe_offset=g * 3 + 2]

    # SH -> color backward (also yields a mean gradient through the view dir).
    var gcol = SIMD[DType.float32, 4](
        grad_color[unsafe_offset=s * 3 + 0], grad_color[unsafe_offset=s * 3 + 1],
        grad_color[unsafe_offset=s * 3 + 2], 0.0,
    )
    var dir = SIMD[DType.float32, 4](m0 - cam[CAM_POS_X], m1 - cam[CAM_POS_Y], m2 - cam[CAM_POS_Z], 0.0)
    var dmean_color = sh_to_color_backward(
        sh_rest, out_sh0, out_sh_rest, g, dir, gcol, Int(active_bases), Int(rest_stride)
    )
    out_opacity[unsafe_offset=g] = grad_opacity_slot[unsafe_offset=s]

    # Recompute the forward quantities.
    var depth = cam[8] * m0 + cam[9] * m1 + cam[10] * m2 + cam[11]
    var x = (cam[0] * m0 + cam[1] * m1 + cam[2] * m2 + cam[3]) / depth
    var y = (cam[4] * m0 + cam[5] * m1 + cam[6] * m2 + cam[7]) / depth
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
    var sn = 2.0 / norm_sq
    var R00 = 1.0 - sn * (yy + zz)
    var R01 = sn * (xy - rz)
    var R02 = sn * (xz + ry)
    var R10 = sn * (xy + rz)
    var R11 = 1.0 - sn * (xx + zz)
    var R12 = sn * (yz - rx)
    var R20 = sn * (xz - ry)
    var R21 = sn * (yz + rx)
    var R22 = 1.0 - sn * (xx + yy)
    # RSS = R diag(v)
    var S00 = R00 * v0
    var S01 = R01 * v1
    var S02 = R02 * v2
    var S10 = R10 * v0
    var S11 = R11 * v1
    var S12 = R12 * v2
    var S20 = R20 * v0
    var S21 = R21 * v1
    var S22 = R22 * v2
    var c00 = S00 * R00 + S01 * R01 + S02 * R02
    var c01 = S00 * R10 + S01 * R11 + S02 * R12
    var c02 = S00 * R20 + S01 * R21 + S02 * R22
    var c11 = S10 * R10 + S11 * R11 + S12 * R12
    var c12 = S10 * R20 + S11 * R21 + S12 * R22
    var c22 = S20 * R20 + S21 * R21 + S22 * R22

    var fx = cam[CAM_FX]
    var fy = cam[CAM_FY]
    var cx = cam[CAM_CX]
    var cy = cam[CAM_CY]
    var W = cam[CAM_W]
    var H = cam[CAM_H]
    var clip_l = (-CLIP_MARGIN * W - cx) / fx
    var clip_r = ((1.0 + CLIP_MARGIN) * W - cx) / fx
    var clip_t = (-CLIP_MARGIN * H - cy) / fy
    var clip_b = ((1.0 + CLIP_MARGIN) * H - cy) / fy
    var x_cl = min(max(x, clip_l), clip_r)
    var y_cl = min(max(y, clip_t), clip_b)
    var j11 = fx / depth
    var j13 = -j11 * x_cl
    var j22 = fy / depth
    var j23 = -j22 * y_cl
    # Rows of J W
    var u0 = j11 * cam[0] + j13 * cam[8]
    var u1 = j11 * cam[1] + j13 * cam[9]
    var u2 = j11 * cam[2] + j13 * cam[10]
    var w0 = j22 * cam[4] + j23 * cam[8]
    var w1 = j22 * cam[5] + j23 * cam[9]
    var w2 = j22 * cam[6] + j23 * cam[10]
    # (J W) C rows
    var uc0 = u0 * c00 + u1 * c01 + u2 * c02
    var uc1 = u0 * c01 + u1 * c11 + u2 * c12
    var uc2 = u0 * c02 + u1 * c12 + u2 * c22
    var wc0 = w0 * c00 + w1 * c01 + w2 * c02
    var wc1 = w0 * c01 + w1 * c11 + w2 * c12
    var wc2 = w0 * c02 + w1 * c12 + w2 * c22

    # 2D covariance gradient from the conic gradient.
    var a = uc0 * u0 + uc1 * u1 + uc2 * u2 + DILATION
    var b = uc0 * w0 + uc1 * w1 + uc2 * w2
    var c = wc0 * w0 + wc1 * w1 + wc2 * w2 + DILATION
    var det = a * c - b * b
    var det_rcp_sq = 1.0 / (det * det)
    var gca = grad_conic[unsafe_offset=s * 3 + 0]
    var gcb = grad_conic[unsafe_offset=s * 3 + 1]
    var gcc = grad_conic[unsafe_offset=s * 3 + 2]
    var ga = det_rcp_sq * (2.0 * b * c * gcb - c * c * gca - b * b * gcc)
    var gb = det_rcp_sq * (b * c * gca - (a * c + b * b) * gcb + a * b * gcc)
    var gc = det_rcp_sq * (2.0 * a * b * gcb - b * b * gca - a * a * gcc)

    # 3D covariance gradient (symmetric, per entry).
    var d00 = u0 * u0 * ga + 2.0 * u0 * w0 * gb + w0 * w0 * gc
    var d01 = u0 * u1 * ga + (u0 * w1 + u1 * w0) * gb + w0 * w1 * gc
    var d02 = u0 * u2 * ga + (u0 * w2 + u2 * w0) * gb + w0 * w2 * gc
    var d11 = u1 * u1 * ga + 2.0 * u1 * w1 * gb + w1 * w1 * gc
    var d12 = u1 * u2 * ga + (u1 * w2 + u2 * w1) * gb + w1 * w2 * gc
    var d22 = u2 * u2 * ga + 2.0 * u2 * w2 * gb + w2 * w2 * gc

    # Gradient of J W, then of the non-zero entries of J.
    var du0 = 2.0 * (uc0 * ga + wc0 * gb)
    var du1 = 2.0 * (uc1 * ga + wc1 * gb)
    var du2 = 2.0 * (uc2 * ga + wc2 * gb)
    var dw0 = 2.0 * (uc0 * gb + wc0 * gc)
    var dw1 = 2.0 * (uc1 * gb + wc1 * gc)
    var dw2 = 2.0 * (uc2 * gb + wc2 * gc)
    var dj11 = cam[0] * du0 + cam[1] * du1 + cam[2] * du2
    var dj22 = cam[4] * dw0 + cam[5] * dw1 + cam[6] * dw2
    var dj13 = cam[8] * du0 + cam[9] * du1 + cam[10] * du2
    var dj23 = cam[8] * dw0 + cam[9] * dw1 + cam[10] * dw2

    var gmx = grad_mean2d[unsafe_offset=s * 2 + 0]
    var gmy = grad_mean2d[unsafe_offset=s * 2 + 1]
    if has_densification != 0:
        var nx = 0.5 * gmx * W
        var ny = 0.5 * gmy * H
        densification[unsafe_offset=g] += 1.0
        densification[unsafe_offset=Int(n) + g] += sqrt(nx * nx + ny * ny)

    # Mean gradient in camera space: from mean2d, then through J (with clamping).
    var dc0 = j11 * gmx
    var dc1 = j22 * gmy
    var dc2 = -j11 * x * gmx - j22 * y * gmy
    var valid_x = x >= clip_l and x <= clip_r
    var valid_y = y >= clip_t and y <= clip_b
    if valid_x:
        dc0 -= j11 * dj13 / depth
    if valid_y:
        dc1 -= j22 * dj23 / depth
    var fac_x: Float32 = 2.0 if valid_x else 1.0
    var fac_y: Float32 = 2.0 if valid_y else 1.0
    dc2 += (j11 * (fac_x * x_cl * dj13 - dj11) + j22 * (fac_y * y_cl * dj23 - dj22)) / depth

    out_means[unsafe_offset=g * 3 + 0] = cam[0] * dc0 + cam[4] * dc1 + cam[8] * dc2 + dmean_color[0]
    out_means[unsafe_offset=g * 3 + 1] = cam[1] * dc0 + cam[5] * dc1 + cam[9] * dc2 + dmean_color[1]
    out_means[unsafe_offset=g * 3 + 2] = cam[2] * dc0 + cam[6] * dc1 + cam[10] * dc2 + dmean_color[2]

    # Scale gradient (variance = exp(2 s)).
    var dv0 = R00 * R00 * d00 + R10 * R10 * d11 + R20 * R20 * d22 + 2.0 * (R00 * R10 * d01 + R00 * R20 * d02 + R10 * R20 * d12)
    var dv1 = R01 * R01 * d00 + R11 * R11 * d11 + R21 * R21 * d22 + 2.0 * (R01 * R11 * d01 + R01 * R21 * d02 + R11 * R21 * d12)
    var dv2 = R02 * R02 * d00 + R12 * R12 * d11 + R22 * R22 * d22 + 2.0 * (R02 * R12 * d01 + R02 * R22 * d02 + R12 * R22 * d12)
    out_log_scales[unsafe_offset=g * 3 + 0] = 2.0 * v0 * dv0
    out_log_scales[unsafe_offset=g * 3 + 1] = 2.0 * v1 * dv1
    out_log_scales[unsafe_offset=g * 3 + 2] = 2.0 * v2 * dv2

    # Rotation matrix gradient: dL/dR = 2 D R diag(v).
    var dR00 = 2.0 * (S00 * d00 + S10 * d01 + S20 * d02)
    var dR01 = 2.0 * (S01 * d00 + S11 * d01 + S21 * d02)
    var dR02 = 2.0 * (S02 * d00 + S12 * d01 + S22 * d02)
    var dR10 = 2.0 * (S00 * d01 + S10 * d11 + S20 * d12)
    var dR11 = 2.0 * (S01 * d01 + S11 * d11 + S21 * d12)
    var dR12 = 2.0 * (S02 * d01 + S12 * d11 + S22 * d12)
    var dR20 = 2.0 * (S00 * d02 + S10 * d12 + S20 * d22)
    var dR21 = 2.0 * (S01 * d02 + S11 * d12 + S21 * d22)
    var dR22 = 2.0 * (S02 * d02 + S12 * d12 + S22 * d22)

    # Quaternion gradient (through the implicit normalization).
    var d_xx = dR11 + dR22
    var d_yy = dR00 + dR22
    var d_zz = dR00 + dR11
    var d_rz = dR10 - dR01
    var d_xy = dR10 + dR01
    var d_ry = dR02 - dR20
    var d_xz = dR02 + dR20
    var d_rx = dR21 - dR12
    var d_yz = dR21 + dR12
    var two = 2.0 / norm_sq
    var helper = two * (
        xy * d_xy + xz * d_xz + yz * d_yz + rx * d_rx + ry * d_ry + rz * d_rz
        - xx * d_xx - yy * d_yy - zz * d_zz
    )
    out_rotations[unsafe_offset=g * 4 + 0] = two * (qx * d_rx + qy * d_ry + qz * d_rz - qr * helper)
    out_rotations[unsafe_offset=g * 4 + 1] = two * (qr * d_rx - 2.0 * qx * d_xx + qy * d_xy + qz * d_xz - qx * helper)
    out_rotations[unsafe_offset=g * 4 + 2] = two * (qr * d_ry + qx * d_xy - 2.0 * qy * d_yy + qz * d_yz - qy * helper)
    out_rotations[unsafe_offset=g * 4 + 3] = two * (qr * d_rz + qx * d_xz + qy * d_yz - 2.0 * qz * d_zz - qz * helper)


def blend_backward[
    tile_w: Int, tile_h: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    n_bucket_blocks: Int,
    tile_offsets: U32Ptr,
    bucket_start: U32Ptr,
    bucket_tile: U32Ptr,
    tile_max_n: U32Ptr,
    instance_ids: I32Ptr,
    records: F32Ptr,
    colors: F32Ptr,
    background: SIMD[DType.float32, 4],
    image: F32Ptr,
    grad_image: F32Ptr,
    final_T: F32Ptr,
    n_processed: U32Ptr,
    snapshots: F32Ptr,
    grad_mean2d: F32Ptr,
    grad_conic: F32Ptr,
    grad_opacity: F32Ptr,
    grad_color: F32Ptr,
    n_tiles: Int,
    width: Int,
    height: Int,
) raises:
    var grid_w = (width + tile_w - 1) // tile_w
    stream.enqueue_function(
        ctx.compile_function[_blend_backward_kernel[tile_w, tile_h]](),
        tile_offsets, bucket_start, bucket_tile, tile_max_n, instance_ids,
        records, colors, background, image, grad_image, final_T, n_processed,
        snapshots, grad_mean2d, grad_conic, grad_opacity, grad_color,
        Int32(n_tiles), Int32(width), Int32(height), Int32(grid_w),
        grid_dim=max(n_bucket_blocks, 1), block_dim=WARP_SIZE,
    )


def preprocess_backward(
    ctx: DeviceContext,
    stream: DeviceStream,
    means: F32Ptr,
    log_scales: F32Ptr,
    rotations: F32Ptr,
    sh_rest: F32Ptr,
    cam: SIMD[DType.float32, 32],
    slot_of: I32Ptr,
    grad_mean2d: F32Ptr,
    grad_conic: F32Ptr,
    grad_opacity_slot: F32Ptr,
    grad_color: F32Ptr,
    out_means: F32Ptr,
    out_log_scales: F32Ptr,
    out_rotations: F32Ptr,
    out_opacity: F32Ptr,
    out_sh0: F32Ptr,
    out_sh_rest: F32Ptr,
    densification: F32Ptr,
    has_densification: Bool,
    n: Int,
    active_bases: Int,
    rest_stride: Int,
) raises:
    if n == 0:
        return
    stream.enqueue_function(
        ctx.compile_function[_preprocess_backward_kernel](),
        means, log_scales, rotations, sh_rest, cam, slot_of, grad_mean2d,
        grad_conic, grad_opacity_slot, grad_color, out_means, out_log_scales,
        out_rotations, out_opacity, out_sh0, out_sh_rest, densification,
        Int32(1 if has_densification else 0), Int32(n), Int32(active_bases),
        Int32(rest_stride),
        grid_dim=num_blocks(n), block_dim=BLOCK_SIZE,
    )


def _fill_f32_kernel(p: F32Ptr, value: Float32, n: Int32):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i < Int(n):
        p[unsafe_offset=i] = value


def fill_f32(ctx: DeviceContext, stream: DeviceStream, p: F32Ptr, value: Float32, n: Int) raises:
    if n == 0:
        return
    stream.enqueue_function(
        ctx.compile_function[_fill_f32_kernel](), p, value, Int32(n),
        grid_dim=num_blocks(n), block_dim=BLOCK_SIZE,
    )
