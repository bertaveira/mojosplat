"""Preprocess backward: 2D gradients -> raw-parameter gradients.

Ported from Faster-GS (Apache-2.0, `kernels_backward.cuh`). One thread per
Gaussian, in Gaussian order so every gradient write is coalesced: culled
Gaussians get zero gradients, visible ones go from their slot's 2D gradients
(mean2d, conic, color) back through the EWA projection, the activations and
spherical harmonics to the raw parameters. All gradient buffers are fully
written (no zero-init needed).
"""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceStream
from std.math import exp, sqrt

from ..core.camera import CAM_CX, CAM_CY, CAM_FX, CAM_FY, CAM_H, CAM_POS_X, CAM_POS_Y, CAM_POS_Z, CAM_W
from ..core.types import BLOCK_SIZE, F32Ptr, I32Ptr, num_blocks
from .preprocess import CLIP_MARGIN, DILATION
from .sh import sh_to_color_backward


@fieldwise_init
struct ShRestGrad(ImplicitlyCopyable, Movable):
    """What an optimizer needs to recompute the higher-order SH gradients
    elementwise instead of reading them from memory:
        dL/d sh_rest[g, k, c] = sh_rest_basis(k, dir_g) * grad_color[slot_of[g], c]
    for k < active_bases - 1 and visible g (slot_of[g] >= 0), else 0, with
    dir_g the normalized (mean_g - camera position) *before* the update.
    Valid until the rasterizer's next forward / backward."""

    var slot_of: I32Ptr  # [N] slot of each Gaussian, -1 if culled
    var grad_color: F32Ptr  # [slot, 3] dL/d(color before clamping)
    var cam_pos: SIMD[DType.float32, 4]
    var active_bases: Int


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
    write_sh_rest: Int32,  # 0: leave out_sh_rest untouched (fused optimizer recomputes it)
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
        if write_sh_rest != 0:
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
        sh_rest, out_sh0, out_sh_rest, g, dir, gcol, Int(active_bases), Int(rest_stride),
        write_sh_rest != 0,
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
    write_sh_rest: Bool = True,
) raises:
    if n == 0:
        return
    stream.enqueue_function(
        ctx.compile_function[_preprocess_backward_kernel](),
        means, log_scales, rotations, sh_rest, cam, slot_of, grad_mean2d,
        grad_conic, grad_opacity_slot, grad_color, out_means, out_log_scales,
        out_rotations, out_opacity, out_sh0, out_sh_rest, densification,
        Int32(1 if has_densification else 0), Int32(1 if write_sh_rest else 0), Int32(n), Int32(active_bases),
        Int32(rest_stride),
        grid_dim=num_blocks(n), block_dim=BLOCK_SIZE,
    )
