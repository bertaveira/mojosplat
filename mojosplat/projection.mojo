"""EWA projection of 3D Gaussians to the image plane (one thread per Gaussian).

Numerically matches gsplat's `fully_fused_projection` forward pass
(eps2d = 0.3 blur, opacity-aware radius, frustum-clamped Jacobian).
"""

from max.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceStream
from std.math import sqrt, ceil, log, rsqrt

from .camera import (
    CAM_FX, CAM_FY, CAM_CX, CAM_CY, CAM_W, CAM_H,
    CAM_LIM_X_POS, CAM_LIM_X_NEG, CAM_LIM_Y_POS, CAM_LIM_Y_NEG,
    CAM_NEAR, CAM_FAR,
)
from .common import F32Ptr, I32Ptr, BLOCK_SIZE, num_blocks

comptime ALPHA_THRESHOLD: Float32 = 1.0 / 255.0
comptime EPS2D: Float32 = 0.3


def project_kernel(
    means3d: F32Ptr,    # [N, 3]
    scales: F32Ptr,     # [N, 3] linear (already exp'd) scales
    quats: F32Ptr,      # [N, 4] (w, x, y, z), need not be normalized
    opacities: F32Ptr,  # [N]
    cam: SIMD[DType.float32, 32],
    means2d: F32Ptr,    # out [N, 2]
    conics: F32Ptr,     # out [N, 3]
    depths: F32Ptr,     # out [N]
    radii: I32Ptr,      # out [N, 2]; 0 marks a culled Gaussian
    n: Int32,
):
    var g = block_idx.x * BLOCK_SIZE + thread_idx.x
    if g >= Int(n):
        return

    var Rv00 = cam[0]
    var Rv01 = cam[1]
    var Rv02 = cam[2]
    var tv0 = cam[3]
    var Rv10 = cam[4]
    var Rv11 = cam[5]
    var Rv12 = cam[6]
    var tv1 = cam[7]
    var Rv20 = cam[8]
    var Rv21 = cam[9]
    var Rv22 = cam[10]
    var tv2 = cam[11]

    # World -> camera
    var g0 = means3d[unsafe_offset=g * 3 + 0]
    var g1 = means3d[unsafe_offset=g * 3 + 1]
    var g2 = means3d[unsafe_offset=g * 3 + 2]
    var mc0 = Rv00 * g0 + Rv01 * g1 + Rv02 * g2 + tv0
    var mc1 = Rv10 * g0 + Rv11 * g1 + Rv12 * g2 + tv1
    var mc2 = Rv20 * g0 + Rv21 * g1 + Rv22 * g2 + tv2

    if mc2 <= cam[CAM_NEAR] or mc2 >= cam[CAM_FAR]:
        radii[unsafe_offset=g * 2 + 0] = 0
        radii[unsafe_offset=g * 2 + 1] = 0
        return

    var opacity = opacities[unsafe_offset=g]
    if opacity < ALPHA_THRESHOLD:
        radii[unsafe_offset=g * 2 + 0] = 0
        radii[unsafe_offset=g * 2 + 1] = 0
        return

    # Quaternion -> rotation matrix
    var w = quats[unsafe_offset=g * 4 + 0]
    var x = quats[unsafe_offset=g * 4 + 1]
    var y = quats[unsafe_offset=g * 4 + 2]
    var z = quats[unsafe_offset=g * 4 + 3]
    var inv_norm = rsqrt(x * x + y * y + z * z + w * w)
    x *= inv_norm
    y *= inv_norm
    z *= inv_norm
    w *= inv_norm
    var x2 = x * x
    var y2 = y * y
    var z2 = z * z
    var xy = x * y
    var xz = x * z
    var yz = y * z
    var wx = w * x
    var wy = w * y
    var wz = w * z
    var Q00: Float32 = 1.0 - 2.0 * (y2 + z2)
    var Q01: Float32 = 2.0 * (xy - wz)
    var Q02: Float32 = 2.0 * (xz + wy)
    var Q10: Float32 = 2.0 * (xy + wz)
    var Q11: Float32 = 1.0 - 2.0 * (x2 + z2)
    var Q12: Float32 = 2.0 * (yz - wx)
    var Q20: Float32 = 2.0 * (xz - wy)
    var Q21: Float32 = 2.0 * (yz + wx)
    var Q22: Float32 = 1.0 - 2.0 * (x2 + y2)

    # M = Q * diag(S); Mp = Rv @ M so that covar_cam = Mp @ Mp^T
    var s0 = scales[unsafe_offset=g * 3 + 0]
    var s1 = scales[unsafe_offset=g * 3 + 1]
    var s2 = scales[unsafe_offset=g * 3 + 2]
    var M00 = Q00 * s0
    var M01 = Q01 * s1
    var M02 = Q02 * s2
    var M10 = Q10 * s0
    var M11 = Q11 * s1
    var M12 = Q12 * s2
    var M20 = Q20 * s0
    var M21 = Q21 * s1
    var M22 = Q22 * s2
    var Mp00 = Rv00 * M00 + Rv01 * M10 + Rv02 * M20
    var Mp01 = Rv00 * M01 + Rv01 * M11 + Rv02 * M21
    var Mp02 = Rv00 * M02 + Rv01 * M12 + Rv02 * M22
    var Mp10 = Rv10 * M00 + Rv11 * M10 + Rv12 * M20
    var Mp11 = Rv10 * M01 + Rv11 * M11 + Rv12 * M21
    var Mp12 = Rv10 * M02 + Rv11 * M12 + Rv12 * M22
    var Mp20 = Rv20 * M00 + Rv21 * M10 + Rv22 * M20
    var Mp21 = Rv20 * M01 + Rv21 * M11 + Rv22 * M21
    var Mp22 = Rv20 * M02 + Rv21 * M12 + Rv22 * M22

    # Pinhole projection with frustum-clamped Jacobian
    var fx = cam[CAM_FX]
    var fy = cam[CAM_FY]
    var rz = rsqrt(mc2 * mc2)  # 1 / mc2 (mc2 > near > 0)
    var tx = mc2 * min(cam[CAM_LIM_X_POS], max(-cam[CAM_LIM_X_NEG], mc0 * rz))
    var ty = mc2 * min(cam[CAM_LIM_Y_POS], max(-cam[CAM_LIM_Y_NEG], mc1 * rz))
    var J00 = fx * rz
    var J02 = -J00 * tx * rz
    var J11 = fy * rz
    var J12 = -J11 * ty * rz

    # cov2d = (J @ Mp) @ (J @ Mp)^T, J = [[J00, 0, J02], [0, J11, J12]]
    var r0_0 = J00 * Mp00 + J02 * Mp20
    var r0_1 = J00 * Mp01 + J02 * Mp21
    var r0_2 = J00 * Mp02 + J02 * Mp22
    var r1_0 = J11 * Mp10 + J12 * Mp20
    var r1_1 = J11 * Mp11 + J12 * Mp21
    var r1_2 = J11 * Mp12 + J12 * Mp22
    var c00 = r0_0 * r0_0 + r0_1 * r0_1 + r0_2 * r0_2 + EPS2D
    var c01 = r0_0 * r1_0 + r0_1 * r1_1 + r0_2 * r1_2
    var c11 = r1_0 * r1_0 + r1_1 * r1_1 + r1_2 * r1_2 + EPS2D

    var m2d_x = fx * mc0 * rz + cam[CAM_CX]
    var m2d_y = fy * mc1 * rz + cam[CAM_CY]

    var det = c00 * c11 - c01 * c01
    if det <= 0.0:
        radii[unsafe_offset=g * 2 + 0] = 0
        radii[unsafe_offset=g * 2 + 1] = 0
        return
    var inv_det = Float32(1.0) / det

    # Opacity-aware extent (gsplat): 3.33 sigma max, less for faint Gaussians
    var extend = min(Float32(3.33), sqrt(2.0 * log(opacity * Float32(255.0))))
    var radius_x = ceil(extend * sqrt(c00))
    var radius_y = ceil(extend * sqrt(c11))

    if (
        m2d_x + radius_x <= 0
        or m2d_x - radius_x >= cam[CAM_W]
        or m2d_y + radius_y <= 0
        or m2d_y - radius_y >= cam[CAM_H]
    ):
        radii[unsafe_offset=g * 2 + 0] = 0
        radii[unsafe_offset=g * 2 + 1] = 0
        return

    conics[unsafe_offset=g * 3 + 0] = c11 * inv_det
    conics[unsafe_offset=g * 3 + 1] = -c01 * inv_det
    conics[unsafe_offset=g * 3 + 2] = c00 * inv_det
    radii[unsafe_offset=g * 2 + 0] = Int32(radius_x)
    radii[unsafe_offset=g * 2 + 1] = Int32(radius_y)
    means2d[unsafe_offset=g * 2 + 0] = m2d_x
    means2d[unsafe_offset=g * 2 + 1] = m2d_y
    depths[unsafe_offset=g] = mc2


def project_gaussians(
    ctx: DeviceContext,
    stream: DeviceStream,
    means3d: F32Ptr,
    scales: F32Ptr,
    quats: F32Ptr,
    opacities: F32Ptr,
    cam: SIMD[DType.float32, 32],
    means2d: F32Ptr,
    conics: F32Ptr,
    depths: F32Ptr,
    radii: I32Ptr,
    n: Int,
) raises:
    """Enqueue projection of `n` Gaussians on `stream`.

    All pointers are device pointers. Outputs of culled Gaussians (radii == 0)
    other than `radii` are left untouched.
    """
    if n == 0:
        return
    stream.enqueue_function(
        ctx.compile_function[project_kernel](),
        means3d, scales, quats, opacities, cam,
        means2d, conics, depths, radii, Int32(n),
        grid_dim=num_blocks(n),
        block_dim=BLOCK_SIZE,
    )
