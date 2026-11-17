"""Spherical Voronoi color (Di Sario et al., CVPR 2026; see THIRD_PARTY_NOTICES.md).

    color = 0.5 + sum_i w_i * c_i,   w = softmax(-tau_i * |s_i - v|)

evaluated once per Gaussian for the view direction `v`; blending clamps it at 0.

Columns: sh0 and sh_rest (unused, as in the official code), and K lobes of
sites [N, K, 3] (raw, normalized here), colors [N, K, 3] and log_tau [N, K].
Ported from `spherical_voronoi_cuda.cu`, including one quirk: it returns no
gradient for the view direction, so color never moves a Gaussian.
"""

from std.math import cos, exp, recip, rsqrt, sin, sqrt

from ....core.splats import AppearanceSettings, ExtraColumns
from ..contracts import Appearance, ColumnSpec

comptime PI: Float64 = 3.14159265358979323846
comptime MAX_LOBES = 16  # lanes of the local arrays; unused lanes cost nothing
comptime MIN_DIST2: Float32 = 1e-12
comptime MIN_NORM2: Float32 = 1e-24  # as F.normalize, eps = 1e-12
comptime _F4 = SIMD[DType.float32, 4]
comptime _Lobes = SIMD[DType.float32, MAX_LOBES]


@always_inline
def _dot(a: _F4, b: _F4) -> Float32:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@fieldwise_init
struct _Partition(ImplicitlyCopyable, Movable):
    """The weights of one Gaussian, with what the backward needs."""

    var view: _F4  # unit view direction
    var weights: _Lobes
    var dist: _Lobes
    var tau: _Lobes


struct SphericalVoronoi[K: Int](Appearance):
    comptime NUM_COLUMNS = 5  # sh0, sh_rest, sites, colors, log_tau

    @staticmethod
    def columns() -> List[ColumnSpec]:
        return [
            ColumnSpec("sites", Self.K * 3, True, 0.0, 0.002),
            ColumnSpec("colors", Self.K * 3, True, 0.0, 0.0008),
            ColumnSpec("log_tau", Self.K, True, 0.0, 0.006),
        ]

    @staticmethod
    def init_column(index: Int, count: Int, rgb: List[UInt8]) -> List[Float32]:
        """Sites on a Fibonacci sphere, colors from the points, temperature 1."""
        if index == 0:
            var lobes = List[Float32](capacity=Self.K * 3)
            var golden = PI * (3.0 - sqrt(5.0))
            for i in range(Self.K):
                var at = Float64(i) + 0.5
                var y = 1.0 - at / Float64(Self.K) * 2.0
                var radius = sqrt(1.0 - y * y)
                lobes.append(Float32(cos(golden * at) * radius))
                lobes.append(Float32(y))
                lobes.append(Float32(sin(golden * at) * radius))
            var out = List[Float32](capacity=count * Self.K * 3)
            for _ in range(count):
                for v in lobes:
                    out.append(v)
            return out^
        if index == 1:
            var out = List[Float32](capacity=count * Self.K * 3)
            for g in range(count):
                for _ in range(Self.K):
                    for c in range(3):
                        out.append(Float32(rgb[g * 3 + c]) / 255.0 - 0.5)
            return out^
        return List[Float32](length=count * Self.K, fill=0.0)

    @staticmethod
    def lr_scale(index: Int, iteration: Int, total: Int, frozen_last: Int) -> Float32:
        """Sites and temperatures follow a cosine down to 10% until the freeze;
        then the sites stop and the temperatures keep their last rate."""
        if index == 1:
            return 1.0
        var frozen = iteration > total - frozen_last
        if frozen and index == 0:
            return 0.0
        var at = total - frozen_last if frozen else iteration
        return Float32(0.1 + 0.45 * (1.0 + cos(PI * Float64(at) / Float64(total))))

    @always_inline
    @staticmethod
    def _site(cols: ExtraColumns, g: Int, i: Int) -> _F4:
        var at = (g * Self.K + i) * 3
        var s = _F4(0.0)
        comptime for c in range(3):
            s[c] = cols.c2[unsafe_offset=at + c]
        return s

    @always_inline
    @staticmethod
    def _unit(raw: _F4) -> _F4:
        return raw * rsqrt(max(_dot(raw, raw), MIN_NORM2))

    @always_inline
    @staticmethod
    def _color(cols: ExtraColumns, g: Int, i: Int) -> _F4:
        var at = (g * Self.K + i) * 3
        var c = _F4(0.0)
        comptime for k in range(3):
            c[k] = cols.c3[unsafe_offset=at + k]
        return c

    @always_inline
    @staticmethod
    def _partition(cols: ExtraColumns, g: Int, dir: _F4) -> _Partition:
        comptime assert 1 <= Self.K <= MAX_LOBES, "SphericalVoronoi supports 1 to 16 sites"
        var v = dir * rsqrt(_dot(dir, dir))
        v[3] = 0.0
        var dist = _Lobes(0.0)
        var tau = _Lobes(0.0)
        var logit = _Lobes(0.0)
        var largest: Float32 = -1e30
        comptime for i in range(Self.K):
            var d = Self._unit(Self._site(cols, g, i)) - v
            dist[i] = sqrt(max(_dot(d, d), MIN_DIST2))
            tau[i] = exp(cols.c4[unsafe_offset=g * Self.K + i])
            logit[i] = -tau[i] * dist[i]
            largest = max(largest, logit[i])
        var weights = _Lobes(0.0)
        var total: Float32 = 0.0
        comptime for i in range(Self.K):
            weights[i] = exp(logit[i] - largest)
            total += weights[i]
        var inv = recip(total)
        comptime for i in range(Self.K):
            weights[i] *= inv
        return _Partition(v, weights, dist, tau)

    @always_inline
    @staticmethod
    def color(cols: ExtraColumns, g: Int, dir: _F4, settings: AppearanceSettings) -> _F4:
        var p = Self._partition(cols, g, dir)
        var out = _F4(0.5, 0.5, 0.5, 0.0)
        comptime for i in range(Self.K):
            out += p.weights[i] * Self._color(cols, g, i)
        return out

    @always_inline
    @staticmethod
    def backward(
        cols: ExtraColumns,
        grads: ExtraColumns,
        g: Int,
        dir: _F4,
        grad_color: _F4,
        settings: AppearanceSettings,
        write_all: Bool,
    ) -> _F4:
        comptime for c in range(3):
            grads.c0[unsafe_offset=g * 3 + c] = 0.0
        var p = Self._partition(cols, g, dir)
        var along_color = _Lobes(0.0)  # dL/dw at fixed softmax normalization
        var mean_along: Float32 = 0.0
        comptime for i in range(Self.K):
            var c = Self._color(cols, g, i)
            var at = (g * Self.K + i) * 3
            comptime for k in range(3):
                grads.c3[unsafe_offset=at + k] = p.weights[i] * grad_color[k]
            along_color[i] = _dot(c, grad_color)
            mean_along += p.weights[i] * along_color[i]
        comptime for i in range(Self.K):
            var d_logit = p.weights[i] * (along_color[i] - mean_along)
            var at = g * Self.K + i
            grads.c4[unsafe_offset=at] = -p.dist[i] * d_logit * p.tau[i]
            var raw = Self._site(cols, g, i)
            var inv_norm = rsqrt(max(_dot(raw, raw), MIN_NORM2))
            var unit = raw * inv_norm
            # d|s - v|/ds, then through the normalization of the raw site.
            var d_site = (unit - p.view) * (-p.tau[i] * d_logit * recip(p.dist[i]))
            var d_raw = (d_site - unit * _dot(unit, d_site)) * inv_norm
            comptime for k in range(3):
                grads.c2[unsafe_offset=at * 3 + k] = d_raw[k]
        return _F4(0.0)  # as in the official code

    @always_inline
    @staticmethod
    def zero_gradients(grads: ExtraColumns, g: Int, settings: AppearanceSettings, write_all: Bool):
        comptime for c in range(3):
            grads.c0[unsafe_offset=g * 3 + c] = 0.0
        comptime for k in range(Self.K * 3):
            grads.c2[unsafe_offset=g * Self.K * 3 + k] = 0.0
            grads.c3[unsafe_offset=g * Self.K * 3 + k] = 0.0
        comptime for k in range(Self.K):
            grads.c4[unsafe_offset=g * Self.K + k] = 0.0
