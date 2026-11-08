"""Real spherical harmonics (degree <= 3) -> RGB, as in 3DGS / gsplat.

Constants and basis ordering follow the reference 3DGS implementation; the
evaluation order is ported from Faster-GS (Apache-2.0,
github.com/nerficg-project/faster-gaussian-splatting, `sh_utils.cuh`).

Layout: `sh0` is [N, 3] (DC term), `sh_rest` is [N, rest_stride, 3] holding
the remaining (degree+1)^2 - 1 bases; `rest_stride` may exceed the number of
bases in use (e.g. a degree-3 buffer rendered at degree 1).
"""

from std.math import rsqrt

from ....core.math import (
    SH_C0,
    SH_C1,
    SH_C2A,
    SH_C2B,
    SH_C2C,
    SH_C2D,
    SH_C3A,
    SH_C3B,
    SH_C3C,
    SH_C3D,
    SH_C3E,
    SH_C3F,
    SH_C3G,
    SH_C3H,
    SH_C2E,
    SH_C3I,
    SH_C3J,
    SH_C3K,
)
from ....core.splats import AppearanceSettings, ExtraColumns
from ....core.types import F32Ptr, I32Ptr
from ..contracts import Appearance


@always_inline
def load_rgb(p: F32Ptr, i: Int) -> SIMD[DType.float32, 4]:
    # Avoid mixing device references and local values in the variadic SIMD
    # constructor (Mojo 1.2 Metal address-space lowering bug).
    var rgb = SIMD[DType.float32, 4](0.0)
    comptime for c in range(3):
        rgb[c] = p[unsafe_offset=i * 3 + c]
    return rgb


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
    var result = 0.5 + SH_C0 * load_rgb(sh0, g)
    if active_bases > 1:
        var inv = rsqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
        var x = dir[0] * inv
        var y = dir[1] * inv
        var z = dir[2] * inv
        var b = g * rest_stride
        result = (
            result
            - SH_C1 * y * load_rgb(sh_rest, b + 0)
            + SH_C1 * z * load_rgb(sh_rest, b + 1)
            - SH_C1 * x * load_rgb(sh_rest, b + 2)
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
                + SH_C2A * xy * load_rgb(sh_rest, b + 3)
                - SH_C2A * yz * load_rgb(sh_rest, b + 4)
                + (SH_C2B * zz - SH_C2C) * load_rgb(sh_rest, b + 5)
                - SH_C2A * xz * load_rgb(sh_rest, b + 6)
                + SH_C2D * (xx - yy) * load_rgb(sh_rest, b + 7)
            )
            if active_bases > 9:
                result = (
                    result
                    + y * (SH_C3A * yy - SH_C3B * xx) * load_rgb(sh_rest, b + 8)
                    + SH_C3C * xy * z * load_rgb(sh_rest, b + 9)
                    + y * (SH_C3D - SH_C3E * zz) * load_rgb(sh_rest, b + 10)
                    + z * (SH_C3F * zz - SH_C3G) * load_rgb(sh_rest, b + 11)
                    + x * (SH_C3D - SH_C3E * zz) * load_rgb(sh_rest, b + 12)
                    + SH_C3H * z * (xx - yy) * load_rgb(sh_rest, b + 13)
                    + x * (SH_C3B * yy - SH_C3A * xx) * load_rgb(sh_rest, b + 14)
                )
    return result


@always_inline
def _store_rgb(p: F32Ptr, i: Int, v: SIMD[DType.float32, 4]):
    p[unsafe_offset=i * 3 + 0] = v[0]
    p[unsafe_offset=i * 3 + 1] = v[1]
    p[unsafe_offset=i * 3 + 2] = v[2]


@always_inline
def _store_rest(write: Bool, p: F32Ptr, i: Int, v: SIMD[DType.float32, 4]):
    if write:
        _store_rgb(p, i, v)


@always_inline
def sh_rest_basis(k: Int, x: Float32, y: Float32, z: Float32) -> Float32:
    """Value of higher-order SH basis k + 1 (k = 0..14) at the unit direction
    (x, y, z), with the sign convention of `sh_to_color`: the color is
    0.5 + C0 * sh0 + sum_k sh_rest[k] * sh_rest_basis(k)."""
    if k == 0:
        return -SH_C1 * y
    elif k == 1:
        return SH_C1 * z
    elif k == 2:
        return -SH_C1 * x
    elif k == 3:
        return SH_C2A * x * y
    elif k == 4:
        return -SH_C2A * y * z
    elif k == 5:
        return SH_C2B * z * z - SH_C2C
    elif k == 6:
        return -SH_C2A * x * z
    elif k == 7:
        return SH_C2D * (x * x - y * y)
    elif k == 8:
        return y * (SH_C3A * y * y - SH_C3B * x * x)
    elif k == 9:
        return SH_C3C * x * y * z
    elif k == 10:
        return y * (SH_C3D - SH_C3E * z * z)
    elif k == 11:
        return z * (SH_C3F * z * z - SH_C3G)
    elif k == 12:
        return x * (SH_C3D - SH_C3E * z * z)
    elif k == 13:
        return SH_C3H * z * (x * x - y * y)
    return x * (SH_C3B * y * y - SH_C3A * x * x)


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
    write_rest: Bool = True,
) -> SIMD[DType.float32, 4]:
    """Writes SH gradients of Gaussian `g`; returns dL/d(mean) through `dir`.

    With `write_rest=False` the higher-order coefficient gradients are not
    stored (they are basis(dir) * grad_color, which the fused optimizer
    recomputes elementwise; see `sh_rest_basis`).
    Port of Faster-GS's `convert_sh_to_color_backward` (Apache-2.0).
    """
    _store_rgb(grad_sh0, g, SH_C0 * grad_color)
    var b = g * rest_stride
    if write_rest:
        for k in range(max(active_bases - 1, 0), rest_stride):
            _store_rgb(grad_sh_rest, b + k, SIMD[DType.float32, 4](0.0))
    if active_bases <= 1:
        return SIMD[DType.float32, 4](0.0)

    var inv = rsqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
    var x = dir[0] * inv
    var y = dir[1] * inv
    var z = dir[2] * inv
    _store_rest(write_rest, grad_sh_rest, b + 0, -SH_C1 * y * grad_color)
    _store_rest(write_rest, grad_sh_rest, b + 1, SH_C1 * z * grad_color)
    _store_rest(write_rest, grad_sh_rest, b + 2, -SH_C1 * x * grad_color)
    var gx = -SH_C1 * load_rgb(sh_rest, b + 2)
    var gy = -SH_C1 * load_rgb(sh_rest, b + 0)
    var gz = SH_C1 * load_rgb(sh_rest, b + 1)
    if active_bases > 4:
        var xx = x * x
        var yy = y * y
        var zz = z * z
        var xy = x * y
        var xz = x * z
        var yz = y * z
        _store_rest(write_rest, grad_sh_rest, b + 3, SH_C2A * xy * grad_color)
        _store_rest(write_rest, grad_sh_rest, b + 4, -SH_C2A * yz * grad_color)
        _store_rest(write_rest, grad_sh_rest, b + 5, (SH_C2B * zz - SH_C2C) * grad_color)
        _store_rest(write_rest, grad_sh_rest, b + 6, -SH_C2A * xz * grad_color)
        _store_rest(write_rest, grad_sh_rest, b + 7, SH_C2D * (xx - yy) * grad_color)
        gx = (
            gx
            + SH_C2A * y * load_rgb(sh_rest, b + 3)
            - SH_C2A * z * load_rgb(sh_rest, b + 6)
            + SH_C2A * x * load_rgb(sh_rest, b + 7)
        )
        gy = (
            gy
            + SH_C2A * x * load_rgb(sh_rest, b + 3)
            - SH_C2A * z * load_rgb(sh_rest, b + 4)
            - SH_C2A * y * load_rgb(sh_rest, b + 7)
        )
        gz = (
            gz
            - SH_C2A * y * load_rgb(sh_rest, b + 4)
            + SH_C2E * z * load_rgb(sh_rest, b + 5)
            - SH_C2A * x * load_rgb(sh_rest, b + 6)
        )
        if active_bases > 9:
            _store_rest(write_rest, grad_sh_rest, b + 8, y * (SH_C3A * yy - SH_C3B * xx) * grad_color)
            _store_rest(write_rest, grad_sh_rest, b + 9, SH_C3C * xy * z * grad_color)
            _store_rest(write_rest, grad_sh_rest, b + 10, y * (SH_C3D - SH_C3E * zz) * grad_color)
            _store_rest(write_rest, grad_sh_rest, b + 11, z * (SH_C3F * zz - SH_C3G) * grad_color)
            _store_rest(write_rest, grad_sh_rest, b + 12, x * (SH_C3D - SH_C3E * zz) * grad_color)
            _store_rest(write_rest, grad_sh_rest, b + 13, SH_C3H * z * (xx - yy) * grad_color)
            _store_rest(write_rest, grad_sh_rest, b + 14, x * (SH_C3B * yy - SH_C3A * xx) * grad_color)
            gx = (
                gx
                - SH_C3I * xy * load_rgb(sh_rest, b + 8)
                + SH_C3C * yz * load_rgb(sh_rest, b + 9)
                + (SH_C3D - SH_C3E * zz) * load_rgb(sh_rest, b + 12)
                + SH_C3C * xz * load_rgb(sh_rest, b + 13)
                + SH_C3B * (yy - xx) * load_rgb(sh_rest, b + 14)
            )
            gy = (
                gy
                + SH_C3B * (yy - xx) * load_rgb(sh_rest, b + 8)
                + SH_C3C * xz * load_rgb(sh_rest, b + 9)
                + (SH_C3D - SH_C3E * zz) * load_rgb(sh_rest, b + 10)
                - SH_C3C * yz * load_rgb(sh_rest, b + 13)
                + SH_C3I * xy * load_rgb(sh_rest, b + 14)
            )
            gz = (
                gz
                + SH_C3C * xy * load_rgb(sh_rest, b + 9)
                - SH_C3J * yz * load_rgb(sh_rest, b + 10)
                + (SH_C3K * zz - SH_C3G) * load_rgb(sh_rest, b + 11)
                - SH_C3J * xz * load_rgb(sh_rest, b + 12)
                + SH_C3H * (xx - yy) * load_rgb(sh_rest, b + 13)
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
    var s = inv * inv * inv  # |dir|^-3
    return SIMD[DType.float32, 4](
        ((yy_r + zz_r) * gdx - xr * yr * gdy - xr * zr * gdz) * s,
        (-xr * yr * gdx + (xx_r + zz_r) * gdy - yr * zr * gdz) * s,
        (-xr * zr * gdx - yr * zr * gdy + (xx_r + yy_r) * gdz) * s,
        0.0,
    )


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


struct SphericalHarmonics(Appearance):
    comptime NUM_COLUMNS = 2

    @always_inline
    @staticmethod
    def _sh0(cols: ExtraColumns) -> F32Ptr:
        return cols.c0  # [N, 3]

    @always_inline
    @staticmethod
    def _sh_rest(cols: ExtraColumns) -> F32Ptr:
        return cols.c1  # [N, rest_stride, 3]

    @always_inline
    @staticmethod
    def _active_bases(settings: AppearanceSettings) -> Int:
        return Int(settings[0])

    @always_inline
    @staticmethod
    def _rest_stride(settings: AppearanceSettings) -> Int:
        return Int(settings[1])

    @always_inline
    @staticmethod
    def color(
        cols: ExtraColumns, g: Int, dir: SIMD[DType.float32, 4], settings: AppearanceSettings
    ) -> SIMD[DType.float32, 4]:
        return sh_to_color(
            Self._sh0(cols), Self._sh_rest(cols), g, dir, Self._active_bases(settings), Self._rest_stride(settings)
        )

    @always_inline
    @staticmethod
    def backward(
        cols: ExtraColumns,
        grads: ExtraColumns,
        g: Int,
        dir: SIMD[DType.float32, 4],
        grad_color: SIMD[DType.float32, 4],
        settings: AppearanceSettings,
        write_all: Bool,
    ) -> SIMD[DType.float32, 4]:
        return sh_to_color_backward(
            Self._sh_rest(cols),
            Self._sh0(grads),
            Self._sh_rest(grads),
            g,
            dir,
            grad_color,
            Self._active_bases(settings),
            Self._rest_stride(settings),
            write_all,
        )

    @always_inline
    @staticmethod
    def zero_gradients(grads: ExtraColumns, g: Int, settings: AppearanceSettings, write_all: Bool):
        var grad_sh0 = Self._sh0(grads)
        comptime for k in range(3):
            grad_sh0[unsafe_offset=g * 3 + k] = 0.0
        if write_all:
            var grad_rest = Self._sh_rest(grads)
            var rest_stride = Self._rest_stride(settings)
            for k in range(rest_stride * 3):
                grad_rest[unsafe_offset=g * rest_stride * 3 + k] = 0.0
