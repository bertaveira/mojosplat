"""The 3DGUT per-ray Gaussian response (Wu et al. 2025), forward and backward (pinhole).

alpha = min(MAX_ALPHA, opacity * exp(-0.5 * dist2)) with dist2 the squared
distance of the ray to the Gaussian centre in Gaussian canonical space.

For a pinhole camera all rays leave one point, so dist2 at a pixel offset
(dx, dy) from the true projected centre is a ratio of two quadratics:

    dist2 = (n11 dx^2 + 2 n12 dx dy + n22 dy^2) / (1 + 2 e.d + d^T F d)

`RayCoeffs` computes those 10 numbers and pulls gradients with respect to them
back to the 3D parameters, once per Gaussian. It uses (M x) x (M y) = cof(M) (x y) to avoid the
cross-product cancellation that makes the direct float32 form unusable for
thin Gaussians.

Record lanes hold NVIDIA's 2D proxy (bounds and tile test), extra lanes the
ray response.

Not ported: NVIDIA also accepts a hit only when its distance along the ray
lies in (tMin, tMax), the ray's span in a +-1e6 box. That distance is a norm,
so tMin = 0 rejects only a distance of exactly 0, and tMax only hits more
than about 1e6 units away; neither occurs in these scenes.

Record: [ut mx, ut my, conic a, b, c, proxy opacity w, bounds, bounds].
Extra (11): [mx, my, ea, eb, ec, 2 e1, 2 e2, f11, 2 f12, f22, opacity],
with (ea, eb, ec) the numerator folded by -0.5 log2(e) so `exp2` evaluates it.
Slab (16): [mx, my, ea, eb], [ec, 2e1, 2e2, f11], [2f12, f22, op, c0], [c1, c2, -, -].
Slot gradients (11, one array each): the extra lanes, the last as d(opacity logit).
"""

from std.math import ceil, exp2, floor, log, recip, sqrt

from ....core.camera import CAM_CX, CAM_CY, CAM_FX, CAM_FY, CAM_POS_X, CAM_POS_Y, CAM_POS_Z
from ....core.linalg import Mat3, Sym2, V2, V3
from ....core.splats import ExtraColumns
from ....core.types import SlabPtr
from ...record import MIN_ALPHA, MIN_ALPHA_RCP, REC
from ..contracts import (
    ColumnSpec,
    Extra,
    GaussianGeometry,
    GaussianGrad,
    Hit,
    LoadedResponse,
    Placement,
    Response,
    ResponseGrad,
    SlotGrad,
)
from ..shape import QuatScale, ShapeGrad

comptime DILATION: Float32 = 0.3
comptime MAX_ALPHA: Float32 = 0.99  # configs/render/3dgrt.yaml particle_kernel_max_alpha, inherited by 3dgut
comptime MIN_RESPONSE: Float32 = 0.0113
comptime LOG2E: Float32 = 1.4426950408889634
comptime LN2: Float32 = 0.6931471805599453
comptime MAX_EXTENT: Float32 = 3.33

comptime _F4 = SIMD[DType.float32, 4]
comptime _F8 = SIMD[DType.float32, 8]


@always_inline
def _dot[dt: DType, //](a: SIMD[dt, 4], b: SIMD[dt, 4]) -> Scalar[dt]:
    return (a * b).reduce_add()


@fieldwise_init
struct RayGrad[dt: DType](ImplicitlyCopyable, Movable):
    var mean: V3[Self.dt]  # dL/d(world mean)
    var log_scales: V3[Self.dt]
    var dR: Mat3[Self.dt]  # dL/d(rotation matrix entries)


struct RayCoeffs[dt: DType](ImplicitlyCopyable, Movable):
    """The ratio-form coefficients of one Gaussian for one pinhole camera, and their pullback."""

    var cam: SIMD[Self.dt, 32]
    var mx: Scalar[Self.dt]  # true projected centre
    var my: Scalar[Self.dt]
    var n11: Scalar[Self.dt]
    var n12: Scalar[Self.dt]
    var n22: Scalar[Self.dt]
    var e1: Scalar[Self.dt]
    var e2: Scalar[Self.dt]
    var f11: Scalar[Self.dt]
    var f12: Scalar[Self.dt]
    var f22: Scalar[Self.dt]
    var Px: Scalar[Self.dt]
    var Py: Scalar[Self.dt]
    var Pz: Scalar[Self.dt]
    var ipz: Scalar[Self.dt]
    var ifx: Scalar[Self.dt]
    var ify: Scalar[Self.dt]
    var kappa: Scalar[Self.dt]  # 1 / smax^2: scale-free S^-1 keeps float32 in range
    var iq: Scalar[Self.dt]
    var i: SIMD[Self.dt, 4]  # scale-free 1 / s
    var cof: SIMD[Self.dt, 4]
    var g0: SIMD[Self.dt, 4]  # columns of G (camera directions -> Gaussian axes)
    var g1: SIMD[Self.dt, 4]
    var g2: SIMD[Self.dt, 4]
    var h: SIMD[Self.dt, 4]
    var u: SIMD[Self.dt, 4]
    var a1: SIMD[Self.dt, 4]
    var a2: SIMD[Self.dt, 4]
    var w1: SIMD[Self.dt, 4]
    var w2: SIMD[Self.dt, 4]
    var v1: SIMD[Self.dt, 4]
    var v2: SIMD[Self.dt, 4]

    @always_inline
    def __init__(
        out self,
        m: V3[Self.dt],
        R: Mat3[Self.dt],
        var0: Scalar[Self.dt],
        var1: Scalar[Self.dt],
        var2: Scalar[Self.dt],
        cam: SIMD[Self.dt, 32],
    ):
        comptime V = SIMD[Self.dt, 4]
        self.cam = cam
        var Px = cam[0] * m.x + cam[1] * m.y + cam[2] * m.z + cam[3]
        var Py = cam[4] * m.x + cam[5] * m.y + cam[6] * m.z + cam[7]
        var Pz = cam[8] * m.x + cam[9] * m.y + cam[10] * m.z + cam[11]
        var ipz = recip(Pz)
        var fx = cam[CAM_FX]
        var fy = cam[CAM_FY]
        var ifx = recip(fx)
        var ify = recip(fy)
        self.Px = Px
        self.Py = Py
        self.Pz = Pz
        self.ipz = ipz
        self.ifx = ifx
        self.ify = ify
        self.mx = Px * ipz * fx + cam[CAM_CX]
        self.my = Py * ipz * fy + cam[CAM_CY]
        # G[k][j] = axis_k . view_row_j.
        var g0 = V(
            R.m00 * cam[0] + R.m10 * cam[1] + R.m20 * cam[2],
            R.m01 * cam[0] + R.m11 * cam[1] + R.m21 * cam[2],
            R.m02 * cam[0] + R.m12 * cam[1] + R.m22 * cam[2],
            0.0,
        )
        var g1 = V(
            R.m00 * cam[4] + R.m10 * cam[5] + R.m20 * cam[6],
            R.m01 * cam[4] + R.m11 * cam[5] + R.m21 * cam[6],
            R.m02 * cam[4] + R.m12 * cam[5] + R.m22 * cam[6],
            0.0,
        )
        var g2 = V(
            R.m00 * cam[8] + R.m10 * cam[9] + R.m20 * cam[10],
            R.m01 * cam[8] + R.m11 * cam[9] + R.m21 * cam[10],
            R.m02 * cam[8] + R.m12 * cam[9] + R.m22 * cam[10],
            0.0,
        )
        var s0 = sqrt(var0)
        var s1 = sqrt(var1)
        var s2 = sqrt(var2)
        var smax = max(max(s0, s1), s2)
        var ir = recip(smax)
        var i = V(recip(s0 * ir), recip(s1 * ir), recip(s2 * ir), 0.0)
        var cof = V(i[1] * i[2], i[0] * i[2], i[0] * i[1], 0.0)
        var h = (g0 * Px + g1 * Py + g2 * Pz) * ipz
        var u = i * h
        var a1 = i * g0 * ifx
        var a2 = i * g1 * ify
        var w1 = (g1 * Pz - g2 * Py) * ifx
        var w2 = (g2 * Px - g0 * Pz) * ify
        var v1 = cof * w1
        var v2 = cof * w2
        var iq = recip(_dot(u, u))
        var kappa = ir * ir
        self.kappa = kappa
        self.iq = iq
        self.i = i
        self.cof = cof
        self.g0 = g0
        self.g1 = g1
        self.g2 = g2
        self.h = h
        self.u = u
        self.a1 = a1
        self.a2 = a2
        self.w1 = w1
        self.w2 = w2
        self.v1 = v1
        self.v2 = v2
        self.n11 = _dot(v1, v1) * iq * kappa
        self.n12 = _dot(v1, v2) * iq * kappa
        self.n22 = _dot(v2, v2) * iq * kappa
        self.e1 = _dot(a1, u) * iq
        self.e2 = _dot(a2, u) * iq
        self.f11 = _dot(a1, a1) * iq
        self.f12 = _dot(a1, a2) * iq
        self.f22 = _dot(a2, a2) * iq

    @always_inline
    def backward(
        self,
        g_mx: Scalar[Self.dt],
        g_my: Scalar[Self.dt],
        g_n11: Scalar[Self.dt],
        g_n12: Scalar[Self.dt],
        g_n22: Scalar[Self.dt],
        g_e1: Scalar[Self.dt],
        g_e2: Scalar[Self.dt],
        g_f11: Scalar[Self.dt],
        g_f12: Scalar[Self.dt],
        g_f22: Scalar[Self.dt],
    ) -> RayGrad[Self.dt]:
        """dL/d(coefficient) -> dL/d(world mean, log scales, rotation matrix)."""
        var iq = self.iq
        var bn11 = g_n11 * iq * self.kappa
        var bn12 = g_n12 * iq * self.kappa
        var bn22 = g_n22 * iq * self.kappa
        var be1 = g_e1 * iq
        var be2 = g_e2 * iq
        var bf11 = g_f11 * iq
        var bf12 = g_f12 * iq
        var bf22 = g_f22 * iq
        var bq = -iq * (
            g_n11 * self.n11
            + g_n12 * self.n12
            + g_n22 * self.n22
            + g_e1 * self.e1
            + g_e2 * self.e2
            + g_f11 * self.f11
            + g_f12 * self.f12
            + g_f22 * self.f22
        )
        var u = self.u
        var a1 = self.a1
        var a2 = self.a2
        var v1 = self.v1
        var v2 = self.v2
        var vb1 = 2.0 * bn11 * v1 + bn12 * v2
        var vb2 = 2.0 * bn22 * v2 + bn12 * v1
        var ab1 = 2.0 * bf11 * a1 + bf12 * a2 + be1 * u
        var ab2 = 2.0 * bf22 * a2 + bf12 * a1 + be2 * u
        var ub = be1 * a1 + be2 * a2 + 2.0 * bq * u
        var i = self.i
        var cof = self.cof
        var g0 = self.g0
        var g1 = self.g1
        var g2 = self.g2
        var h = self.h
        var ifx = self.ifx
        var ify = self.ify
        var ipz = self.ipz
        var Px = self.Px
        var Py = self.Py
        var Pz = self.Pz
        var wb1 = cof * vb1
        var wb2 = cof * vb2
        var hb = i * ub
        var T = vb1 * self.w1 + vb2 * self.w2
        var ib = ub * h + ab1 * g0 * ifx + ab2 * g1 * ify
        ib[0] += T[1] * i[2] + T[2] * i[1]
        ib[1] += T[0] * i[2] + T[2] * i[0]
        ib[2] += T[0] * i[1] + T[1] * i[0]
        var ls = -(ib * i)
        var gb0 = ab1 * i * ifx + hb * (Px * ipz) - wb2 * (Pz * ify)
        var gb1 = ab2 * i * ify + hb * (Py * ipz) + wb1 * (Pz * ifx)
        var gb2 = hb - wb1 * (Py * ifx) + wb2 * (Px * ify)
        var cam = self.cam
        var fx = cam[CAM_FX]
        var fy = cam[CAM_FY]
        var bx = ipz * _dot(hb, g0) + ify * _dot(wb2, g2) + g_mx * fx * ipz
        var by = ipz * _dot(hb, g1) - ifx * _dot(wb1, g2) + g_my * fy * ipz
        var bz = (
            ipz * _dot(hb, g2 - h)
            + ifx * _dot(wb1, g1)
            - ify * _dot(wb2, g0)
            - (g_mx * (self.mx - cam[CAM_CX]) + g_my * (self.my - cam[CAM_CY])) * ipz
        )
        var mean = V3[Self.dt](
            cam[0] * bx + cam[4] * by + cam[8] * bz,
            cam[1] * bx + cam[5] * by + cam[9] * bz,
            cam[2] * bx + cam[6] * by + cam[10] * bz,
        )

        var r0 = gb0 * cam[0] + gb1 * cam[4] + gb2 * cam[8]  # dL/dR[m][k]: k-th lane of row m
        var r1 = gb0 * cam[1] + gb1 * cam[5] + gb2 * cam[9]
        var r2 = gb0 * cam[2] + gb1 * cam[6] + gb2 * cam[10]
        return RayGrad[Self.dt](
            mean,
            V3[Self.dt](ls[0], ls[1], ls[2]),
            Mat3[Self.dt](r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2]),
        )


struct LoadedRay3D(ImplicitlyCopyable, LoadedResponse, Movable):
    var mx: Float32
    var my: Float32
    var ea: Float32
    var eb: Float32
    var ec: Float32
    var e1: Float32
    var e2: Float32
    var f11: Float32
    var f12: Float32
    var f22: Float32
    var opacity: Float32
    var d_mx: Float32
    var d_my: Float32
    var d_ea: Float32
    var d_eb: Float32
    var d_ec: Float32
    var d_e1: Float32
    var d_e2: Float32
    var d_f11: Float32
    var d_f12: Float32
    var d_f22: Float32
    var d_op: Float32

    @always_inline
    def __init__(out self):
        self.mx = 0.0
        self.my = 0.0
        self.ea = 0.0
        self.eb = 0.0
        self.ec = 0.0
        self.e1 = 0.0
        self.e2 = 0.0
        self.f11 = 0.0
        self.f12 = 0.0
        self.f22 = 0.0
        self.opacity = 0.0
        self.d_mx = 0.0
        self.d_my = 0.0
        self.d_ea = 0.0
        self.d_eb = 0.0
        self.d_ec = 0.0
        self.d_e1 = 0.0
        self.d_e2 = 0.0
        self.d_f11 = 0.0
        self.d_f12 = 0.0
        self.d_f22 = 0.0
        self.d_op = 0.0

    @always_inline
    def __init__(out self, src: SlabPtr):
        self = Self()
        var v0 = src.unsafe_bitcast[_F4]()[]
        var v1 = src.unsafe_offset(4).unsafe_bitcast[_F4]()[]
        var v2 = src.unsafe_offset(8).unsafe_bitcast[_F4]()[]
        self.mx = v0[0]
        self.my = v0[1]
        self.ea = v0[2]
        self.eb = v0[3]
        self.ec = v1[0]
        self.e1 = v1[1]
        self.e2 = v1[2]
        self.f11 = v1[3]
        self.f12 = v2[0]
        self.f22 = v2[1]
        self.opacity = v2[2]

    @always_inline
    def __init__(out self, rec: SIMD[DType.float32, REC], extra: Extra):
        self = Self()
        self.mx = extra[0]
        self.my = extra[1]
        self.ea = extra[2]
        self.eb = extra[3]
        self.ec = extra[4]
        self.e1 = extra[5]
        self.e2 = extra[6]
        self.f11 = extra[7]
        self.f12 = extra[8]
        self.f22 = extra[9]
        self.opacity = extra[10]

    @always_inline
    def eval(self, px: Float32, py: Float32) -> Hit:
        """-> (alpha, response, dx, dy, 1/den, power, 1 if alpha is not clamped)."""
        var dx = px - self.mx
        var dy = py - self.my
        var num = dx * (self.ea * dx + self.eb * dy) + self.ec * dy * dy
        var den = 1.0 + dx * (self.e1 + self.f11 * dx + self.f12 * dy) + dy * (self.e2 + self.f22 * dy)
        var iden = recip(den)
        var power = num * iden
        var response = exp2(min(power, 0.0))
        var raw = self.opacity * response
        var alpha = min(MAX_ALPHA, raw) if response > MIN_RESPONSE else Float32(0.0)
        var free: Float32 = 1.0 if raw < MAX_ALPHA else 0.0
        return Hit(alpha, response, dx, dy, iden, power, free, 0.0)

    @always_inline
    def add(mut self, hit: Hit, dL_dalpha: Float32):
        var g = dL_dalpha * hit[6]  # a clamped alpha has no gradient (min() in NVIDIA's autodiff)
        self.d_op += hit[1] * g
        var dx = hit[2]
        var dy = hit[3]
        var iden = hit[4]
        var power = hit[5]
        # dalpha/dpower = alpha ln 2 (power is the exponent of 2)
        var h = hit[0] * LN2 * g
        var hn = h * iden
        var hd = -hn * power
        self.d_ea += hn * dx * dx
        self.d_eb += hn * dx * dy
        self.d_ec += hn * dy * dy
        self.d_e1 += hd * dx
        self.d_e2 += hd * dy
        self.d_f11 += hd * dx * dx
        self.d_f12 += hd * dx * dy
        self.d_f22 += hd * dy * dy
        var nx = 2.0 * self.ea * dx + self.eb * dy
        var ny = self.eb * dx + 2.0 * self.ec * dy
        var ddx = self.e1 + 2.0 * self.f11 * dx + self.f12 * dy
        var ddy = self.e2 + self.f12 * dx + 2.0 * self.f22 * dy
        # d(dx)/d(mx) = -1
        self.d_mx -= h * iden * (nx - power * ddx)
        self.d_my -= h * iden * (ny - power * ddy)

    @always_inline
    def gradient(self) -> SlotGrad:
        var g = SlotGrad(0.0)
        g[0] = self.d_mx
        g[1] = self.d_my
        g[2] = self.d_ea
        g[3] = self.d_eb
        g[4] = self.d_ec
        g[5] = self.d_e1
        g[6] = self.d_e2
        g[7] = self.d_f11
        g[8] = self.d_f12
        g[9] = self.d_f22
        g[10] = self.opacity * (1.0 - self.opacity) * self.d_op
        return g


struct GaussianRay3D(ImplicitlyCopyable, Movable, Response):
    comptime Loaded = LoadedRay3D
    comptime TILE_ARGS = 8
    comptime SLAB = 16
    comptime GRAD = 11
    comptime EXTRA = 11
    comptime NUM_COLUMNS = 0
    comptime MIN_OPACITY = MIN_ALPHA
    comptime STOP_BEFORE_LAST = False

    @always_inline
    @staticmethod
    def skip_fragment(alpha: Float32) -> Bool:
        return alpha <= MIN_ALPHA

    @staticmethod
    def columns() -> List[ColumnSpec]:
        return List[ColumnSpec]()

    var ok: Bool
    var ux: Float32  # unscented mean (proxy)
    var uy: Float32
    var cov: Sym2[DType.float32]  # dilated proxy covariance
    var conic: Sym2[DType.float32]
    var w_scale: Float32  # mip-splatting convolution factor of the proxy opacity
    var ray: RayCoeffs[DType.float32]
    var coef: Extra

    # --- Preprocess ---

    @always_inline
    def __init__(
        out self,
        mean2d: V2[DType.float32],
        cov2d: Sym2[DType.float32],
        geom: GaussianGeometry,
        cam: SIMD[DType.float32, 32],
    ):
        self.ux = mean2d.x + 0.5
        self.uy = mean2d.y + 0.5
        self.ok = cov2d.a >= 0.0
        var a = cov2d.a + DILATION
        var c = cov2d.c + DILATION
        var det = a * c - cov2d.b * cov2d.b
        var det_raw = cov2d.a * cov2d.c - cov2d.b * cov2d.b
        self.ok = self.ok and det != 0.0
        self.cov = Sym2[DType.float32](a, cov2d.b, c)
        var rd = recip(det)
        self.conic = Sym2[DType.float32](c * rd, -cov2d.b * rd, a * rd)
        self.w_scale = sqrt(max(Float32(0.000025), det_raw * rd))
        self.ray = RayCoeffs[DType.float32](geom.mean, geom.R, geom.variance.x, geom.variance.y, geom.variance.z, cam)
        var r = self.ray
        self.coef = Extra(0.0)
        self.coef[0] = r.mx
        self.coef[1] = r.my
        self.coef[2] = -0.5 * LOG2E * r.n11
        self.coef[3] = -LOG2E * r.n12
        self.coef[4] = -0.5 * LOG2E * r.n22
        self.coef[5] = 2.0 * r.e1
        self.coef[6] = 2.0 * r.e2
        self.coef[7] = r.f11
        self.coef[8] = 2.0 * r.f12
        self.coef[9] = r.f22

    @always_inline
    def valid(self) -> Bool:
        return self.ok

    @always_inline
    def extent(self, opacity: Float32) -> SIMD[DType.float32, 2]:
        """Half-size of a rectangle around the shifted proxy mean holding NVIDIA's proxy rectangle
        and every pixel the true response accepts. `touches_tile` applies the proxy bbox itself."""
        var w = opacity * self.w_scale
        var power = log(w * MIN_ALPHA_RCP)
        var factor = min(MAX_EXTENT, sqrt(max(Float32(0.0), 2.0 * power)))
        var a = self.cov.a
        var c = self.cov.c
        var b = self.cov.b
        var mid = 0.5 * (a + c)
        var lam = mid + sqrt(max(Float32(0.01), mid * mid - (a * c - b * b)))
        var radius = factor * sqrt(lam)
        var ex = min(factor * sqrt(a), radius)
        var ey = min(factor * sqrt(c), radius)
        # True response: dist2 <= t, a conic in (dx, dy) when the denominator stays definite.
        var r = self.ray
        var t = min(Float32(8.9660), 2.0 * log(opacity * MIN_ALPHA_RCP))
        if t <= 0.0:
            return SIMD[DType.float32, 2](ex, ey)
        var A = r.n11 - t * r.f11
        var B = r.n12 - t * r.f12
        var C = r.n22 - t * r.f22
        var D = -t * r.e1
        var E = -t * r.e2
        var d2 = A * C - B * B
        if not (A > 0.0 and C > 0.0 and d2 > 0.0):
            return SIMD[DType.float32, 2](2.0e4, 2.0e4)  # degenerate conic: every tile of the proxy bbox
        var id2 = recip(d2)
        var cx = (B * E - C * D) * id2
        var cy = (B * D - A * E) * id2
        var f = -t + D * cx + E * cy
        if f >= 0.0:
            return SIMD[DType.float32, 2](ex, ey)
        var hx = sqrt(-f * C * id2)
        var hy = sqrt(-f * A * id2)
        var ox = r.mx - self.ux
        var oy = r.my - self.uy
        var tx = max(ox + cx + hx, -(ox + cx - hx))
        var ty = max(oy + cy + hy, -(oy + cy - hy))
        return SIMD[DType.float32, 2](max(ex, tx + 0.25), max(ey, ty + 0.25))

    @always_inline
    def record(self, opacity: Float32) -> SIMD[DType.float32, REC]:
        var k = self.conic
        return SIMD[DType.float32, REC](self.ux, self.uy, k.a, k.b, k.c, opacity * self.w_scale, 0.0, 0.0)

    @always_inline
    def extra(self, opacity: Float32, cols: ExtraColumns, g: Int) -> Extra:
        var e = self.coef
        e[10] = opacity
        return e

    @always_inline
    @staticmethod
    def tile_args(rec: SIMD[DType.float32, REC]) -> _F8:
        """[ux, uy, conic a, b, c, max power, proxy extent x, y]."""
        var power = log(rec[5] * MIN_ALPHA_RCP)
        var factor = min(MAX_EXTENT, sqrt(max(Float32(0.0), 2.0 * power)))
        var id = recip(rec[2] * rec[4] - rec[3] * rec[3])
        var a = rec[4] * id
        var c = rec[2] * id
        var b = -rec[3] * id
        var mid = 0.5 * (a + c)
        var lam = mid + sqrt(max(Float32(0.01), mid * mid - (a * c - b * b)))
        var radius = factor * sqrt(lam)
        return _F8(
            rec[0], rec[1], rec[2], rec[3], rec[4], power, min(factor * sqrt(a), radius), min(factor * sqrt(c), radius)
        )

    @always_inline
    @staticmethod
    def touches_tile[tile_w: Int, tile_h: Int](a: _F8, tile_x: Int, tile_y: Int) -> Bool:
        """NVIDIA's proxy bbox and `tileMinParticlePowerResponse < maxPower` (pixel-corner tile bounds)."""
        var mx = a[0]
        var my = a[1]
        var ca = a[2]
        var cb = a[3]
        var cc = a[4]
        var tw = Float32(tile_w)
        var th = Float32(tile_h)
        var fx = Float32(tile_x)
        var fy = Float32(tile_y)
        if not (
            fx >= floor((mx - 0.5 - a[6]) * recip(tw))
            and fx < ceil((mx - 0.5 + a[6]) * recip(tw))
            and fy >= floor((my - 0.5 - a[7]) * recip(th))
            and fy < ceil((my - 0.5 + a[7]) * recip(th))
        ):
            return False
        var x_min = tw * fx
        var y_min = th * fy
        var x_max = tw + x_min
        var y_max = th + y_min
        var off_x = x_min - mx
        var off_y = y_min - my
        var left: Float32 = 1.0 if off_x > 0.0 else 0.0
        var above: Float32 = 1.0 if off_y > 0.0 else 0.0
        var nx = left + (Float32(1.0) if mx > x_max else Float32(0.0))
        var ny = above + (Float32(1.0) if my > y_max else Float32(0.0))
        var power: Float32 = 0.0
        if nx + ny > 0.0:
            var px = x_max + (x_min - x_max) * left
            var py = y_max + (y_min - y_max) * above
            var dx = tw if off_x >= 0.0 else -tw
            var dy = th if off_y >= 0.0 else -th
            var ex = mx - px
            var ey = my - py
            var rx = recip(tw * tw * ca)
            var ry = recip(th * th * cc)
            var tx = ny * min(max((dx * ca * ex + dx * cb * ey) * rx, 0.0), 1.0)
            var ty = nx * min(max((dy * cb * ex + dy * cc * ey) * ry, 0.0), 1.0)
            var qx = mx - (px + tx * dx)
            var qy = my - (py + ty * dy)
            power = 0.5 * (ca * qx * qx + cc * qy * qy) + cb * qx * qy
        return power < a[5]

    # --- Blend ---

    @always_inline
    @staticmethod
    def store_slab(dst: SlabPtr, rec: SIMD[DType.float32, REC], extra: Extra, color: SIMD[DType.float32, 4]):
        dst.unsafe_bitcast[_F4]()[] = _F4(extra[0], extra[1], extra[2], extra[3])
        dst.unsafe_offset(4).unsafe_bitcast[_F4]()[] = _F4(extra[4], extra[5], extra[6], extra[7])
        dst.unsafe_offset(8).unsafe_bitcast[_F4]()[] = _F4(extra[8], extra[9], extra[10], color[0])
        dst.unsafe_offset(12).unsafe_bitcast[_F4]()[] = _F4(color[1], color[2], 0.0, 0.0)

    @always_inline
    @staticmethod
    def channel(src: SlabPtr, k: Int) -> Float32:
        return src[unsafe_offset=11 + k]

    # --- Preprocess backward ---

    @always_inline
    def backward[
        P: Placement
    ](self, grad: SlotGrad, place: P, shape: QuatScale[P.dtype], cam: SIMD[DType.float32, 32]) -> ResponseGrad[P.dtype]:
        comptime dt = P.dtype
        # Slot gradients are per stored lane (see `extra`); unfold them to the coefficients.
        var r = self.ray.backward(
            grad[0],
            grad[1],
            -0.5 * LOG2E * grad[2],
            -LOG2E * grad[3],
            -0.5 * LOG2E * grad[4],
            2.0 * grad[5],
            2.0 * grad[6],
            grad[7],
            2.0 * grad[8],
            grad[9],
        )
        var dR = r.dR
        var quat = shape.rotation_backward(
            Mat3[dt](
                dR.m00.cast[dt](),
                dR.m01.cast[dt](),
                dR.m02.cast[dt](),
                dR.m10.cast[dt](),
                dR.m11.cast[dt](),
                dR.m12.cast[dt](),
                dR.m20.cast[dt](),
                dR.m21.cast[dt](),
                dR.m22.cast[dt](),
            )
        )
        var ls = r.log_scales
        return ResponseGrad[dt](
            GaussianGrad[dt](
                V3[dt](r.mean.x.cast[dt](), r.mean.y.cast[dt](), r.mean.z.cast[dt]()),
                ShapeGrad[dt](V3[dt](ls.x.cast[dt](), ls.y.cast[dt](), ls.z.cast[dt]()), quat),
            ),
            0.0,
        )

    @always_inline
    @staticmethod
    def densification(
        screen_norm: Float32, total_mean: V3[DType.float32], mean: V3[DType.float32], cam: SIMD[DType.float32, 32]
    ) -> SIMD[DType.float32, 2]:
        """NVIDIA's statistic: |dL/dmean| * (distance to camera) / 2, counted when the gradient is non-zero."""
        if total_mean.x == 0.0 and total_mean.y == 0.0 and total_mean.z == 0.0:
            return SIMD[DType.float32, 2](0.0)
        var cx = mean.x - cam[CAM_POS_X]
        var cy = mean.y - cam[CAM_POS_Y]
        var cz = mean.z - cam[CAM_POS_Z]
        var g2 = total_mean.x * total_mean.x + total_mean.y * total_mean.y + total_mean.z * total_mean.z
        return SIMD[DType.float32, 2](1.0, 0.5 * sqrt(g2 * (cx * cx + cy * cy + cz * cz)))

    @always_inline
    @staticmethod
    def opacity_gradient(slot_grad: SlotGrad) -> Float32:
        return slot_grad[10]

    @always_inline
    @staticmethod
    def write_gradients(cols: ExtraColumns, grads: ExtraColumns, g: Int, slot_grad: SlotGrad):
        pass

    @always_inline
    @staticmethod
    def zero_gradients(grads: ExtraColumns, g: Int):
        pass

    @always_inline
    @staticmethod
    def grad_index(k: Int, n_slots: Int, slot: Int) -> Int:
        return n_slots * k + slot
