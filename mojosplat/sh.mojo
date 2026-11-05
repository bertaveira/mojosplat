"""Real spherical harmonics (degree <= 3) -> RGB, as in 3DGS / gsplat.

Constants and basis ordering follow the reference 3DGS implementation; the
evaluation order is ported from Faster-GS (Apache-2.0,
github.com/nerficg-project/faster-gaussian-splatting, `sh_utils.cuh`).

Layout: `sh0` is [N, 3] (DC term), `sh_rest` is [N, rest_stride, 3] holding
the remaining (degree+1)^2 - 1 bases; `rest_stride` may exceed the number of
bases in use (e.g. a degree-3 buffer rendered at degree 1).
"""

from std.math import rsqrt

from .common import F32Ptr

comptime SH_C0: Float32 = 0.28209479177387814
comptime SH_C1: Float32 = 0.48860251190291987
comptime SH_C2A: Float32 = 1.0925484305920792
comptime SH_C2B: Float32 = 0.94617469575755997
comptime SH_C2C: Float32 = 0.31539156525251999
comptime SH_C2D: Float32 = 0.54627421529603959
comptime SH_C3A: Float32 = 0.59004358992664352
comptime SH_C3B: Float32 = 1.7701307697799304
comptime SH_C3C: Float32 = 2.8906114426405538
comptime SH_C3D: Float32 = 0.45704579946446572
comptime SH_C3E: Float32 = 2.2852289973223288
comptime SH_C3F: Float32 = 1.865881662950577
comptime SH_C3G: Float32 = 1.1195289977703462
comptime SH_C3H: Float32 = 1.4453057213202769


def sh_num_bases(degree: Int) -> Int:
    return (degree + 1) * (degree + 1)


@always_inline
def _rgb(p: F32Ptr, i: Int) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](
        p[unsafe_offset=i * 3 + 0], p[unsafe_offset=i * 3 + 1], p[unsafe_offset=i * 3 + 2], 0.0
    )


@always_inline
def sh_to_color(
    sh0: F32Ptr,
    sh_rest: F32Ptr,
    g: Int,
    dir: SIMD[DType.float32, 4],  # mean - camera center (unnormalized)
    active_bases: Int,
    rest_stride: Int,
) -> SIMD[DType.float32, 4]:
    """RGB (lane 3 unused) of Gaussian `g` seen along `dir`, before clamping."""
    var result = 0.5 + SH_C0 * _rgb(sh0, g)
    if active_bases > 1:
        var inv = rsqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
        var x = dir[0] * inv
        var y = dir[1] * inv
        var z = dir[2] * inv
        var b = g * rest_stride
        result = (
            result
            - SH_C1 * y * _rgb(sh_rest, b + 0)
            + SH_C1 * z * _rgb(sh_rest, b + 1)
            - SH_C1 * x * _rgb(sh_rest, b + 2)
        )
        if active_bases > 4:
            var xx = x * x
            var yy = y * y
            var zz = z * z
            var xy = x * y
            var xz = x * z
            var yz = y * z
            result = (
                result
                + SH_C2A * xy * _rgb(sh_rest, b + 3)
                - SH_C2A * yz * _rgb(sh_rest, b + 4)
                + (SH_C2B * zz - SH_C2C) * _rgb(sh_rest, b + 5)
                - SH_C2A * xz * _rgb(sh_rest, b + 6)
                + SH_C2D * (xx - yy) * _rgb(sh_rest, b + 7)
            )
            if active_bases > 9:
                result = (
                    result
                    + y * (SH_C3A * yy - SH_C3B * xx) * _rgb(sh_rest, b + 8)
                    + SH_C3C * xy * z * _rgb(sh_rest, b + 9)
                    + y * (SH_C3D - SH_C3E * zz) * _rgb(sh_rest, b + 10)
                    + z * (SH_C3F * zz - SH_C3G) * _rgb(sh_rest, b + 11)
                    + x * (SH_C3D - SH_C3E * zz) * _rgb(sh_rest, b + 12)
                    + SH_C3H * z * (xx - yy) * _rgb(sh_rest, b + 13)
                    + x * (SH_C3B * yy - SH_C3A * xx) * _rgb(sh_rest, b + 14)
                )
    return result
