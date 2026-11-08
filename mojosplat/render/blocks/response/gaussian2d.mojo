"""The 2D Gaussian footprint of classic 3DGS.

Coverage ported from Faster-GS / StopThePop (Hahlbohm et al. 2026, Radl et al.
2024; Apache-2.0). The Gaussian is evaluated with a single `exp2` by folding
-0.5 * log2(e) into the conic.

Record: [mx, my, conic a, conic b, conic c, opacity, bounds, bounds].
Slab (3 x float4): [mx, my, ea, eb], [ec, opacity, c0, c1], [c2, ...].
Slot gradients: mean2d [slot, 2], conic [slot, 3], opacity logit [slot], in
that order in the scratch buffer. Conic off-diagonal gradients are per entry
of the symmetric matrix (half of the gradient w.r.t. the single value).
"""

from std.math import exp2, log, sqrt

from ....core.linalg import Sym2, V2
from ....core.types import SlabPtr
from ...record import MIN_ALPHA_RCP, REC
from ..contracts import Response, ScreenGrad
from ..screen import ConicFromCov, DILATION

comptime MIN_COV2D_DET: Float32 = 1e-6
comptime LOG2E: Float32 = 1.4426950408889634

comptime _F4 = SIMD[DType.float32, 4]


@always_inline
def _will_contribute[
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


struct Gaussian2D(ImplicitlyCopyable, Movable, Response):
    comptime TILE_ARGS = 6
    comptime SLAB = 12
    comptime GRAD = 6

    # Preprocess
    var mean: V2[DType.float32]
    var inverse: ConicFromCov[DType.float32]
    # Blend backward: this lane's Gaussian, the current pixel, the sums so far
    var opacity: Float32
    var ca: Float32
    var cb: Float32
    var cc: Float32
    var ea: Float32
    var eb: Float32
    var ec: Float32
    var dx: Float32
    var dy: Float32
    var gaussian: Float32
    var d_mx: Float32
    var d_my: Float32
    var d_ca: Float32
    var d_cb: Float32
    var d_cc: Float32
    var d_op: Float32

    @always_inline
    def __init__(out self):
        self.mean = V2[DType.float32](0.0, 0.0)
        self.inverse = ConicFromCov[DType.float32]()
        self.opacity = 0.0
        self.ca = 0.0
        self.cb = 0.0
        self.cc = 0.0
        self.ea = 0.0
        self.eb = 0.0
        self.ec = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.gaussian = 0.0
        self.d_mx = 0.0
        self.d_my = 0.0
        self.d_ca = 0.0
        self.d_cb = 0.0
        self.d_cc = 0.0
        self.d_op = 0.0

    # --- Preprocess ---

    @always_inline
    def __init__(out self, mean2d: V2[DType.float32], cov2d: Sym2[DType.float32]):
        self = Self()
        self.mean = mean2d
        self.inverse = ConicFromCov[DType.float32](Sym2[DType.float32](cov2d.a + DILATION, cov2d.b, cov2d.c + DILATION))

    @always_inline
    def valid(self) -> Bool:
        return self.inverse.det >= MIN_COV2D_DET

    @always_inline
    def extent(self, opacity: Float32) -> SIMD[DType.float32, 2]:
        """Exact x/y extent of the alpha >= 1/255 ellipse."""
        var cutoff = 2.0 * log(opacity * MIN_ALPHA_RCP)
        return SIMD[DType.float32, 2](
            max(sqrt(self.inverse.cov.a * cutoff) - 0.5, 0.0), max(sqrt(self.inverse.cov.c * cutoff) - 0.5, 0.0)
        )

    @always_inline
    def record(self, opacity: Float32) -> SIMD[DType.float32, REC]:
        var c = self.inverse.conic
        return SIMD[DType.float32, REC](self.mean.x, self.mean.y, c.a, c.b, c.c, opacity, 0.0, 0.0)

    @always_inline
    def backward(self, grad: SIMD[DType.float32, 8]) -> ScreenGrad:
        return ScreenGrad(
            V2[DType.float32](grad[0], grad[1]),
            self.inverse.backward(Sym2[DType.float32](grad[2], grad[3], grad[4])),
        )

    @always_inline
    @staticmethod
    def tile_args(rec: SIMD[DType.float32, REC]) -> SIMD[DType.float32, 8]:
        return SIMD[DType.float32, 8](
            rec[0] - 0.5, rec[1] - 0.5, rec[2], rec[3], rec[4], log(rec[5] * MIN_ALPHA_RCP), 0.0, 0.0
        )

    @always_inline
    @staticmethod
    def touches_tile[tile_w: Int, tile_h: Int](a: SIMD[DType.float32, 8], tile_x: Int, tile_y: Int) -> Bool:
        return _will_contribute[tile_w, tile_h](a[0], a[1], a[2], a[3], a[4], tile_x, tile_y, a[5])

    # --- Blend ---

    @always_inline
    @staticmethod
    def store_slab(dst: SlabPtr, rec: SIMD[DType.float32, REC], color: SIMD[DType.float32, 4]):
        dst.unsafe_bitcast[_F4]()[] = _F4(rec[0], rec[1], -0.5 * LOG2E * rec[2], -LOG2E * rec[3])
        dst.unsafe_offset(4).unsafe_bitcast[_F4]()[] = _F4(-0.5 * LOG2E * rec[4], rec[5], color[0], color[1])
        dst.unsafe_offset(8).unsafe_bitcast[_F4]()[] = _F4(color[2], 0.0, 0.0, 0.0)

    @always_inline
    @staticmethod
    def alpha_at(src: SlabPtr, px: Float32, py: Float32) -> Float32:
        var v0 = src.unsafe_bitcast[_F4]()[]
        var v1 = src.unsafe_offset(4).unsafe_bitcast[_F4]()[]
        var dx = v0[0] - px
        var dy = v0[1] - py
        var power = dx * (v0[2] * dx + v0[3] * dy) + v1[0] * dy * dy
        return v1[1] * exp2(min(power, 0.0))

    @always_inline
    @staticmethod
    def channel(src: SlabPtr, k: Int) -> Float32:
        if k == 2:
            return src[unsafe_offset=8]
        return src.unsafe_offset(4).unsafe_bitcast[_F4]()[][2 + k]

    # --- Blend backward ---

    @always_inline
    def load(mut self, rec: SIMD[DType.float32, REC]):
        self.mean = V2[DType.float32](rec[0], rec[1])
        self.ca = rec[2]
        self.cb = rec[3]
        self.cc = rec[4]
        self.opacity = rec[5]
        self.ea = -0.5 * LOG2E * rec[2]
        self.eb = -LOG2E * rec[3]
        self.ec = -0.5 * LOG2E * rec[4]

    @always_inline
    def alpha_of_pixel(mut self, px: Float32, py: Float32) -> Float32:
        self.dx = self.mean.x - px
        self.dy = self.mean.y - py
        var power = self.dx * (self.ea * self.dx + self.eb * self.dy) + self.ec * self.dy * self.dy
        self.gaussian = exp2(min(power, 0.0))
        return self.opacity * self.gaussian

    @always_inline
    def accumulate(mut self, alpha: Float32, dL_dalpha: Float32):
        self.d_op += self.gaussian * dL_dalpha
        var helper = -alpha * dL_dalpha
        var dx = self.dx
        var dy = self.dy
        self.d_ca += helper * dx * dx  # times 0.5 in `gradient`
        self.d_cb += helper * dx * dy
        self.d_cc += helper * dy * dy
        self.d_mx += helper * (self.ca * dx + self.cb * dy)
        self.d_my += helper * (self.cb * dx + self.cc * dy)

    @always_inline
    def gradient(self) -> SIMD[DType.float32, 8]:
        return SIMD[DType.float32, 8](
            self.d_mx,
            self.d_my,
            0.5 * self.d_ca,
            0.5 * self.d_cb,
            0.5 * self.d_cc,
            self.opacity * (1.0 - self.opacity) * self.d_op,
            0.0,
            0.0,
        )

    @always_inline
    @staticmethod
    def grad_index(k: Int, n_slots: Int, slot: Int) -> Int:
        if k < 2:
            return slot * 2 + k
        if k < 5:
            return n_slots * 2 + slot * 3 + k - 2
        return n_slots * 5 + slot
