"""Spherical Beta color of Beta Splatting.

    color = max(0.5 + C0 * sh0, 0) + sum_i rgb_i * max(dot(lobe_i, view), 0)^(4 exp(b_i))

Columns: sh0 [N, 3], sh_rest (unused, usually empty), and K lobes of
[r, g, b, theta, phi, b] raw parameters [N, K * 6]. The lobe colors go through a
softplus. Ported from the official Beta Splatting code, including one quirk: it
returns a zero gradient for the view direction, so color never moves a Gaussian.
"""

from std.math import cos, exp, log, rsqrt, sin
from std.random import random_float64

from ....core.math import SH_C0
from ....core.splats import AppearanceSettings, ExtraColumns
from ....core.types import F32Ptr
from ..contracts import Appearance, ColumnSpec

comptime PI: Float64 = 3.14159265358979323846
comptime LOBE_FLOATS = 6
comptime SOFTPLUS_BETA: Float32 = 6.931471805599453  # log(2) * 10
comptime _F4 = SIMD[DType.float32, 4]


@always_inline
def _softplus(x: Float32) -> Float32:
    return log(1.0 + exp(SOFTPLUS_BETA * x)) / SOFTPLUS_BETA


@always_inline
def _softplus_gradient(x: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-SOFTPLUS_BETA * x))


@always_inline
def _normalized(d: _F4) -> _F4:
    var inv = rsqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    var out = d * inv
    out[3] = 0.0
    return out


@always_inline
def _base(sh0: F32Ptr, g: Int, c: Int) -> Float32:
    return 0.5 + SH_C0 * sh0[unsafe_offset=g * 3 + c]


struct SphericalBetaColor[K: Int](Appearance):
    comptime NUM_COLUMNS = 3

    @staticmethod
    def columns() -> List[ColumnSpec]:
        return [ColumnSpec("sb_params", Self.K * LOBE_FLOATS, True, 0.0, 0.0025)]

    @staticmethod
    def init_column(index: Int, count: Int) -> List[Float32]:
        """Zero lobe colors and sharpness; directions uniform in (theta, phi)."""
        var out = List[Float32](length=count * Self.K * LOBE_FLOATS, fill=0.0)
        for g in range(count):
            for i in range(Self.K):
                var at = g * Self.K * LOBE_FLOATS + i * LOBE_FLOATS
                out[at + 3] = Float32(random_float64(0.0, PI))
                out[at + 4] = Float32(random_float64(0.0, 2.0 * PI))
        return out^

    @always_inline
    @staticmethod
    def color(
        cols: ExtraColumns, g: Int, dir: SIMD[DType.float32, 4], settings: AppearanceSettings
    ) -> SIMD[DType.float32, 4]:
        var out = _F4(0.0)
        comptime for c in range(3):
            out[c] = max(_base(cols.c0, g, c), 0.0)
        var v = _normalized(dir)
        comptime for i in range(Self.K):
            var at = (g * Self.K + i) * LOBE_FLOATS
            var theta = cols.c2[unsafe_offset=at + 3]
            var phi = cols.c2[unsafe_offset=at + 4]
            var st = sin(theta)
            var dot = v[0] * st * cos(phi) + v[1] * st * sin(phi) + v[2] * cos(theta)
            if dot > 0.0:
                var term = exp(4.0 * exp(cols.c2[unsafe_offset=at + 5]) * log(dot))
                comptime for c in range(3):
                    out[c] += term * _softplus(cols.c2[unsafe_offset=at + c])
        return out

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
        comptime for c in range(3):
            # Tolerance: the GPU fuses this multiply-add, so a base of exactly 0 can land just below it.
            var keep: Float32 = 1.0 if _base(cols.c0, g, c) >= -1e-7 else 0.0
            grads.c0[unsafe_offset=g * 3 + c] = SH_C0 * grad_color[c] * keep
        var v = _normalized(dir)
        comptime for i in range(Self.K):
            var at = (g * Self.K + i) * LOBE_FLOATS
            var theta = cols.c2[unsafe_offset=at + 3]
            var phi = cols.c2[unsafe_offset=at + 4]
            var raw_b = cols.c2[unsafe_offset=at + 5]
            var st = sin(theta)
            var ct = cos(theta)
            var sp = sin(phi)
            var cp = cos(phi)
            var dot = v[0] * st * cp + v[1] * st * sp + v[2] * ct
            var positive = dot > 0.0
            var e = 4.0 * exp(raw_b)
            var log_dot: Float32 = log(dot) if positive else 0.0
            var term: Float32 = exp(e * log_dot) if positive else 0.0
            var lobe_color = _F4(0.0)
            comptime for c in range(3):
                var raw = cols.c2[unsafe_offset=at + c]
                lobe_color[c] = _softplus(raw)
                grads.c2[unsafe_offset=at + c] = grad_color[c] * term * _softplus_gradient(raw)
            var color_grad = (
                grad_color[0] * lobe_color[0] + grad_color[1] * lobe_color[1] + grad_color[2] * lobe_color[2]
            )
            # d(dot^e)/d(dot) = e * dot^e / dot
            var d_term: Float32 = e * term / dot if positive else 0.0
            grads.c2[unsafe_offset=at + 3] = color_grad * d_term * (v[0] * ct * cp + v[1] * ct * sp - v[2] * st)
            grads.c2[unsafe_offset=at + 4] = color_grad * d_term * (v[1] * st * cp - v[0] * st * sp)
            grads.c2[unsafe_offset=at + 5] = color_grad * e * term * log_dot
        return _F4(0.0)  # as in the official code

    @always_inline
    @staticmethod
    def zero_gradients(grads: ExtraColumns, g: Int, settings: AppearanceSettings, write_all: Bool):
        comptime for c in range(3):
            grads.c0[unsafe_offset=g * 3 + c] = 0.0
        comptime for k in range(Self.K * LOBE_FLOATS):
            grads.c2[unsafe_offset=g * Self.K * LOBE_FLOATS + k] = 0.0
