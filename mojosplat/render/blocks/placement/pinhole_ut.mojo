"""Pinhole projection by the unscented transform (3DGUT, Wu et al. 2025).

Seven sigma points give a proxy 2D Gaussian used only for bounds and tile
lists (`ut_alpha=1, ut_beta=2, ut_kappa=0`). `Response` evaluates the true ray
response; see `response/ray3d.mojo`. Nothing is differentiated through the
proxy, as in NVIDIA's code: `backward` returns zero, and the ray response
computes the gradients itself.

Returned `mean2d` is the unscented mean minus 0.5 and `cov2d` is -1 when no
sigma point lies within the image margin: the kernel floors/ceils the mean to
pixel rectangles exactly as NVIDIA's `computeTileSpaceBBox` does on the
half-pixel-shifted centre, and the response marks a -1 covariance invalid.
"""

from std.math import sqrt

from ....core.camera import CAM_CX, CAM_CY, CAM_FX, CAM_FY, CAM_H, CAM_W
from ....core.linalg import Mat3, Sym2, Sym3, V2, V3, V4
from ..contracts import GaussianGrad, Placement
from ..shape import QuatScale, ShapeGrad

comptime IMAGE_MARGIN: Float32 = 0.1
comptime SQRT3: Float32 = 1.7320508075688772


struct PinholeUT[dt: DType](ImplicitlyCopyable, Movable, Placement):
    comptime dtype = Self.dt

    var center: V2[Self.dt]
    var cov: Sym2[Self.dt]

    @always_inline
    @staticmethod
    def depth(m: V3[Self.dt], cam: SIMD[Self.dt, 32]) -> Scalar[Self.dt]:
        return cam[8] * m.x + cam[9] * m.y + cam[10] * m.z + cam[11]

    @always_inline
    @staticmethod
    def _project(
        cam: SIMD[Self.dt, 32], x: Scalar[Self.dt], y: Scalar[Self.dt], z: Scalar[Self.dt]
    ) -> SIMD[Self.dt, 4]:
        """(u, v, valid, 0): a point at or behind the camera projects to the origin, invalid, as NVIDIA's."""
        var X = cam[0] * x + cam[1] * y + cam[2] * z + cam[3]
        var Y = cam[4] * x + cam[5] * y + cam[6] * z + cam[7]
        var Z = cam[8] * x + cam[9] * y + cam[10] * z + cam[11]
        if Z <= 0.0:
            return SIMD[Self.dt, 4](0.0)
        var u = X / Z * cam[CAM_FX] + cam[CAM_CX]
        var v = Y / Z * cam[CAM_FY] + cam[CAM_CY]
        var m = IMAGE_MARGIN.cast[Self.dt]()
        var w = cam[CAM_W]
        var h = cam[CAM_H]
        var inside = u > -m * w and v > -m * h and u < w + m * w and v < h + m * h
        var flag: Scalar[Self.dt] = 1.0 if inside else 0.0
        return SIMD[Self.dt, 4](u, v, flag, 0.0)

    @always_inline
    def __init__(out self, m: V3[Self.dt], shape: QuatScale[Self.dt], cam: SIMD[Self.dt, 32]):
        var R = shape.R
        var k = SQRT3.cast[Self.dt]()
        var s0 = k * sqrt(shape.v0)
        var s1 = k * sqrt(shape.v1)
        var s2 = k * sqrt(shape.v2)
        var d0 = V3[Self.dt](s0 * R.m00, s0 * R.m10, s0 * R.m20)
        var d1 = V3[Self.dt](s1 * R.m01, s1 * R.m11, s1 * R.m21)
        var d2 = V3[Self.dt](s2 * R.m02, s2 * R.m12, s2 * R.m22)
        var p0 = Self._project(cam, m.x, m.y, m.z)
        var p1 = Self._project(cam, m.x + d0.x, m.y + d0.y, m.z + d0.z)
        var p2 = Self._project(cam, m.x + d1.x, m.y + d1.y, m.z + d1.z)
        var p3 = Self._project(cam, m.x + d2.x, m.y + d2.y, m.z + d2.z)
        var p4 = Self._project(cam, m.x - d0.x, m.y - d0.y, m.z - d0.z)
        var p5 = Self._project(cam, m.x - d1.x, m.y - d1.y, m.z - d1.z)
        var p6 = Self._project(cam, m.x - d2.x, m.y - d2.y, m.z - d2.z)
        var n_valid = p0[2] + p1[2] + p2[2] + p3[2] + p4[2] + p5[2] + p6[2]
        var w: Scalar[Self.dt] = 1.0 / 6.0  # lambda = 0: the centre has no weight in the mean
        var mx = w * (p1[0] + p2[0] + p3[0] + p4[0] + p5[0] + p6[0])
        var my = w * (p1[1] + p2[1] + p3[1] + p4[1] + p5[1] + p6[1])
        var a: Scalar[Self.dt] = 0.0
        var b: Scalar[Self.dt] = 0.0
        var c: Scalar[Self.dt] = 0.0

        @always_inline
        def add(p: SIMD[Self.dt, 4], weight: Scalar[Self.dt]) {mut a, mut b, mut c, mx, my}:
            var ex = p[0] - mx
            var ey = p[1] - my
            a += weight * ex * ex
            b += weight * ex * ey
            c += weight * ey * ey

        add(p0, 2.0)  # weight0 = lambda/(D+lambda) + (1 - alpha^2 + beta) = 2
        add(p1, w)
        add(p2, w)
        add(p3, w)
        add(p4, w)
        add(p5, w)
        add(p6, w)
        self.center = V2[Self.dt](mx - 0.5, my - 0.5)
        self.cov = Sym2[Self.dt](a, b, c) if n_valid > 0.0 else Sym2[Self.dt](-1.0, 0.0, -1.0)

    @always_inline
    def mean2d(self) -> V2[Self.dt]:
        return self.center

    @always_inline
    def cov2d(self) -> Sym2[Self.dt]:
        return self.cov

    @always_inline
    def backward(
        self, shape: QuatScale[Self.dt], g_mean2d: V2[Self.dt], g_cov2d: Sym2[Self.dt]
    ) -> GaussianGrad[Self.dt]:
        """Zero: the gradients come from the ray response (`GaussianRay3D.backward`)."""
        return GaussianGrad[Self.dt](
            V3[Self.dt](0.0, 0.0, 0.0),
            ShapeGrad[Self.dt](V3[Self.dt](0.0, 0.0, 0.0), V4[Self.dt](0.0, 0.0, 0.0, 0.0)),
        )
