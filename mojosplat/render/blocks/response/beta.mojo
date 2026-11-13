"""The Beta-kernel footprint of Beta Splatting.

    sigma = a dx^2 + c dy^2 + 2 b dx dy        (conic of the 2D covariance)
    alpha = min(0.999, opacity * (1 - sigma)^beta)    for 0 <= sigma < 1

with beta = 4 exp(raw) stored in the response's own column. Support is exactly
the ellipse sigma = 1.

Record: [mx, my, conic a, conic b, conic c, opacity, bounds, bounds].
Per-slot extra: [beta].
Slab (3 x float4): [mx, my, a, b], [c, opacity, c0, c1], [c2, beta, ...].
Slot gradients: as `Gaussian2D`, then the raw beta [slot].
"""

from std.math import exp, log, sqrt

from ....core.linalg import Sym2, V2
from ....core.splats import ExtraColumns
from ....core.types import SlabPtr
from ...record import REC
from ..contracts import ColumnSpec, Hit, LoadedResponse, Response, ScreenGrad
from ..screen import ConicFromCov, DILATION
from .gaussian2d import MIN_COV2D_DET, _will_contribute

comptime _F4 = SIMD[DType.float32, 4]
comptime _F8 = SIMD[DType.float32, 8]
comptime MAX_ALPHA: Float32 = 0.999


struct LoadedBeta(ImplicitlyCopyable, LoadedResponse, Movable):
    var mx: Float32
    var my: Float32
    var ca: Float32
    var cb: Float32
    var cc: Float32
    var opacity: Float32
    var beta: Float32
    var d_mx: Float32
    var d_my: Float32
    var d_ca: Float32
    var d_cb: Float32
    var d_cc: Float32
    var d_op: Float32
    var d_beta: Float32

    @always_inline
    def __init__(out self):
        self.mx = 0.0
        self.my = 0.0
        self.ca = 0.0
        self.cb = 0.0
        self.cc = 0.0
        self.opacity = 0.0
        self.beta = 1.0
        self.d_mx = 0.0
        self.d_my = 0.0
        self.d_ca = 0.0
        self.d_cb = 0.0
        self.d_cc = 0.0
        self.d_op = 0.0
        self.d_beta = 0.0

    @always_inline
    def __init__(out self, src: SlabPtr):
        self = Self()
        var v0 = src.unsafe_bitcast[_F4]()[]
        var v1 = src.unsafe_offset(4).unsafe_bitcast[_F4]()[]
        self.mx = v0[0]
        self.my = v0[1]
        self.ca = v0[2]
        self.cb = v0[3]
        self.cc = v1[0]
        self.opacity = v1[1]
        self.beta = src[unsafe_offset=9]

    @always_inline
    def __init__(out self, rec: SIMD[DType.float32, REC], extra: _F8):
        self = Self()
        self.mx = rec[0]
        self.my = rec[1]
        self.ca = rec[2]
        self.cb = rec[3]
        self.cc = rec[4]
        self.opacity = rec[5]
        self.beta = extra[0]

    @always_inline
    def eval(self, px: Float32, py: Float32) -> Hit:
        """-> (alpha, (1 - sigma)^beta, dx, dy, sigma, 1 if not clamped)."""
        var dx = self.mx - px
        var dy = self.my - py
        var sigma = self.ca * dx * dx + self.cc * dy * dy + 2.0 * self.cb * dx * dy
        if sigma < 0.0 or sigma >= 1.0:
            return Hit(0.0)
        var vis = exp(self.beta * log(1.0 - sigma))
        var unclamped = self.opacity * vis
        return Hit(
            min(MAX_ALPHA, unclamped),
            vis,
            dx,
            dy,
            sigma,
            Float32(1.0) if unclamped <= MAX_ALPHA else Float32(0.0),
            0.0,
            0.0,
        )

    @always_inline
    def add(mut self, hit: Hit, dL_dalpha: Float32):
        if hit[5] == 0.0:
            return  # clamped alpha: no gradient, as in the original
        var vis = hit[1]
        var dx = hit[2]
        var dy = hit[3]
        var one_minus = 1.0 - hit[4]
        var d_sigma = -dL_dalpha * self.opacity * self.beta * vis / one_minus
        self.d_ca += dx * dx * d_sigma
        self.d_cb += 2.0 * dx * dy * d_sigma
        self.d_cc += dy * dy * d_sigma
        self.d_mx += 2.0 * d_sigma * (self.ca * dx + self.cb * dy)
        self.d_my += 2.0 * d_sigma * (self.cb * dx + self.cc * dy)
        self.d_op += vis * dL_dalpha
        self.d_beta += dL_dalpha * self.opacity * vis * log(one_minus)

    @always_inline
    def gradient(self) -> _F8:
        return _F8(
            self.d_mx,
            self.d_my,
            self.d_ca,
            0.5 * self.d_cb,
            self.d_cc,
            self.opacity * (1.0 - self.opacity) * self.d_op,
            self.d_beta * self.beta,
            0.0,
        )


struct BetaFootprint(ImplicitlyCopyable, Movable, Response):
    comptime Loaded = LoadedBeta
    comptime TILE_ARGS = 6
    comptime SLAB = 12
    comptime GRAD = 7
    comptime EXTRA = 1
    comptime NUM_COLUMNS = 1
    comptime MIN_OPACITY: Float32 = 0.0
    comptime STOP_BEFORE_LAST = True

    @always_inline
    @staticmethod
    def skip_fragment(alpha: Float32) -> Bool:
        return alpha <= 0.0

    @staticmethod
    def columns() -> List[ColumnSpec]:
        # Raw beta starts at 0 (beta = 4), learning rate as in the official code.
        return [ColumnSpec("beta", 1, True, 0.0, 0.001)]

    var mean: V2[DType.float32]
    var inverse: ConicFromCov[DType.float32]

    @always_inline
    def __init__(out self, mean2d: V2[DType.float32], cov2d: Sym2[DType.float32]):
        self.mean = mean2d
        self.inverse = ConicFromCov[DType.float32](Sym2[DType.float32](cov2d.a + DILATION, cov2d.b, cov2d.c + DILATION))

    @always_inline
    def valid(self) -> Bool:
        return self.inverse.det >= MIN_COV2D_DET

    @always_inline
    def extent(self, opacity: Float32) -> SIMD[DType.float32, 2]:
        return SIMD[DType.float32, 2](
            max(sqrt(self.inverse.cov.a) - 0.5, 0.0), max(sqrt(self.inverse.cov.c) - 0.5, 0.0)
        )

    @always_inline
    def record(self, opacity: Float32) -> SIMD[DType.float32, REC]:
        var c = self.inverse.conic
        return SIMD[DType.float32, REC](self.mean.x, self.mean.y, c.a, c.b, c.c, opacity, 0.0, 0.0)

    @always_inline
    @staticmethod
    def extra(cols: ExtraColumns, g: Int) -> _F8:
        var v = _F8(0.0)
        v[0] = 4.0 * exp(cols.c0[unsafe_offset=g])
        return v

    @always_inline
    @staticmethod
    def tile_args(rec: SIMD[DType.float32, REC]) -> _F8:
        # The test compares 0.5 * sigma against the threshold: sigma < 1.
        return _F8(rec[0] - 0.5, rec[1] - 0.5, rec[2], rec[3], rec[4], 0.5, 0.0, 0.0)

    @always_inline
    @staticmethod
    def touches_tile[tile_w: Int, tile_h: Int](a: _F8, tile_x: Int, tile_y: Int) -> Bool:
        return _will_contribute[tile_w, tile_h](a[0], a[1], a[2], a[3], a[4], tile_x, tile_y, a[5])

    @always_inline
    @staticmethod
    def store_slab(dst: SlabPtr, rec: SIMD[DType.float32, REC], extra: _F8, color: SIMD[DType.float32, 4]):
        dst.unsafe_bitcast[_F4]()[] = _F4(rec[0], rec[1], rec[2], rec[3])
        dst.unsafe_offset(4).unsafe_bitcast[_F4]()[] = _F4(rec[4], rec[5], color[0], color[1])
        dst.unsafe_offset(8).unsafe_bitcast[_F4]()[] = _F4(color[2], extra[0], 0.0, 0.0)

    @always_inline
    @staticmethod
    def channel(src: SlabPtr, k: Int) -> Float32:
        if k == 2:
            return src[unsafe_offset=8]
        return src.unsafe_offset(4).unsafe_bitcast[_F4]()[][2 + k]

    @always_inline
    def backward(self, grad: _F8) -> ScreenGrad:
        return ScreenGrad(
            V2[DType.float32](grad[0], grad[1]),
            self.inverse.backward(Sym2[DType.float32](grad[2], grad[3], grad[4])),
        )

    @always_inline
    @staticmethod
    def opacity_gradient(slot_grad: _F8) -> Float32:
        return slot_grad[5]

    @always_inline
    @staticmethod
    def write_gradients(cols: ExtraColumns, grads: ExtraColumns, g: Int, slot_grad: _F8):
        grads.c0[unsafe_offset=g] = slot_grad[6]

    @always_inline
    @staticmethod
    def zero_gradients(grads: ExtraColumns, g: Int):
        grads.c0[unsafe_offset=g] = 0.0

    @always_inline
    @staticmethod
    def grad_index(k: Int, n_slots: Int, slot: Int) -> Int:
        if k < 2:
            return slot * 2 + k
        if k < 5:
            return n_slots * 2 + slot * 3 + k - 2
        if k == 5:
            return n_slots * 5 + slot
        return n_slots * 6 + slot
