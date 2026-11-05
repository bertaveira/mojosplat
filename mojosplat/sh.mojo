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


comptime SH_C2E: Float32 = 1.8923493915151202
comptime SH_C3I: Float32 = 3.5402615395598609
comptime SH_C3J: Float32 = 4.5704579946446566
comptime SH_C3K: Float32 = 5.597644988851731


@always_inline
def _store_rgb(p: F32Ptr, i: Int, v: SIMD[DType.float32, 4]):
    p[unsafe_offset=i * 3 + 0] = v[0]
    p[unsafe_offset=i * 3 + 1] = v[1]
    p[unsafe_offset=i * 3 + 2] = v[2]


@always_inline
def _dot3(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> Float32:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@always_inline
def sh_to_color_backward(
    sh_rest: F32Ptr,
    grad_sh0: F32Ptr,  # out [N, 3]
    grad_sh_rest: F32Ptr,  # out [N, rest_stride, 3]; unused bases get 0
    g: Int,
    dir: SIMD[DType.float32, 4],  # mean - camera center (unnormalized)
    grad_color: SIMD[DType.float32, 4],  # dL/d(color before clamping)
    active_bases: Int,
    rest_stride: Int,
) -> SIMD[DType.float32, 4]:
    """Writes SH gradients of Gaussian `g`; returns dL/d(mean) through `dir`.

    Port of Faster-GS's `convert_sh_to_color_backward` (Apache-2.0).
    """
    _store_rgb(grad_sh0, g, SH_C0 * grad_color)
    var b = g * rest_stride
    for k in range(max(active_bases - 1, 0), rest_stride):
        _store_rgb(grad_sh_rest, b + k, SIMD[DType.float32, 4](0.0))
    if active_bases <= 1:
        return SIMD[DType.float32, 4](0.0)

    var inv = rsqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
    var x = dir[0] * inv
    var y = dir[1] * inv
    var z = dir[2] * inv
    _store_rgb(grad_sh_rest, b + 0, -SH_C1 * y * grad_color)
    _store_rgb(grad_sh_rest, b + 1, SH_C1 * z * grad_color)
    _store_rgb(grad_sh_rest, b + 2, -SH_C1 * x * grad_color)
    var gx = -SH_C1 * _rgb(sh_rest, b + 2)
    var gy = -SH_C1 * _rgb(sh_rest, b + 0)
    var gz = SH_C1 * _rgb(sh_rest, b + 1)
    if active_bases > 4:
        var xx = x * x
        var yy = y * y
        var zz = z * z
        var xy = x * y
        var xz = x * z
        var yz = y * z
        _store_rgb(grad_sh_rest, b + 3, SH_C2A * xy * grad_color)
        _store_rgb(grad_sh_rest, b + 4, -SH_C2A * yz * grad_color)
        _store_rgb(grad_sh_rest, b + 5, (SH_C2B * zz - SH_C2C) * grad_color)
        _store_rgb(grad_sh_rest, b + 6, -SH_C2A * xz * grad_color)
        _store_rgb(grad_sh_rest, b + 7, SH_C2D * (xx - yy) * grad_color)
        gx = gx + SH_C2A * y * _rgb(sh_rest, b + 3) - SH_C2A * z * _rgb(sh_rest, b + 6) + SH_C2A * x * _rgb(sh_rest, b + 7)
        gy = gy + SH_C2A * x * _rgb(sh_rest, b + 3) - SH_C2A * z * _rgb(sh_rest, b + 4) - SH_C2A * y * _rgb(sh_rest, b + 7)
        gz = gz - SH_C2A * y * _rgb(sh_rest, b + 4) + SH_C2E * z * _rgb(sh_rest, b + 5) - SH_C2A * x * _rgb(sh_rest, b + 6)
        if active_bases > 9:
            _store_rgb(grad_sh_rest, b + 8, y * (SH_C3A * yy - SH_C3B * xx) * grad_color)
            _store_rgb(grad_sh_rest, b + 9, SH_C3C * xy * z * grad_color)
            _store_rgb(grad_sh_rest, b + 10, y * (SH_C3D - SH_C3E * zz) * grad_color)
            _store_rgb(grad_sh_rest, b + 11, z * (SH_C3F * zz - SH_C3G) * grad_color)
            _store_rgb(grad_sh_rest, b + 12, x * (SH_C3D - SH_C3E * zz) * grad_color)
            _store_rgb(grad_sh_rest, b + 13, SH_C3H * z * (xx - yy) * grad_color)
            _store_rgb(grad_sh_rest, b + 14, x * (SH_C3B * yy - SH_C3A * xx) * grad_color)
            gx = (
                gx - SH_C3I * xy * _rgb(sh_rest, b + 8) + SH_C3C * yz * _rgb(sh_rest, b + 9)
                + (SH_C3D - SH_C3E * zz) * _rgb(sh_rest, b + 12)
                + SH_C3C * xz * _rgb(sh_rest, b + 13)
                + SH_C3B * (yy - xx) * _rgb(sh_rest, b + 14)
            )
            gy = (
                gy + SH_C3B * (yy - xx) * _rgb(sh_rest, b + 8) + SH_C3C * xz * _rgb(sh_rest, b + 9)
                + (SH_C3D - SH_C3E * zz) * _rgb(sh_rest, b + 10)
                - SH_C3C * yz * _rgb(sh_rest, b + 13)
                + SH_C3I * xy * _rgb(sh_rest, b + 14)
            )
            gz = (
                gz + SH_C3C * xy * _rgb(sh_rest, b + 9) - SH_C3J * yz * _rgb(sh_rest, b + 10)
                + (SH_C3K * zz - SH_C3G) * _rgb(sh_rest, b + 11)
                - SH_C3J * xz * _rgb(sh_rest, b + 12)
                + SH_C3H * (xx - yy) * _rgb(sh_rest, b + 13)
            )
    # d(normalized dir)/d(dir): (I - d d^T) / |dir|
    var gdx = _dot3(gx, grad_color)
    var gdy = _dot3(gy, grad_color)
    var gdz = _dot3(gz, grad_color)
    var xr = dir[0]
    var yr = dir[1]
    var zr = dir[2]
    var xx_r = xr * xr
    var yy_r = yr * yr
    var zz_r = zr * zr
    var n2 = xx_r + yy_r + zz_r
    var s = inv * inv * inv  # |dir|^-3
    return SIMD[DType.float32, 4](
        ((yy_r + zz_r) * gdx - xr * yr * gdy - xr * zr * gdz) * s,
        (-xr * yr * gdx + (xx_r + zz_r) * gdy - yr * zr * gdz) * s,
        (-xr * zr * gdx - yr * zr * gdy + (xx_r + yy_r) * gdz) * s,
        0.0,
    )
