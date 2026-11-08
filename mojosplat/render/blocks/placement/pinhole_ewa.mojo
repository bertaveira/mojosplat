"""Pinhole projection with the EWA Jacobian, hand-fused in world space."""

from ....core.camera import CAM_CX, CAM_CY, CAM_FX, CAM_FY, CAM_H, CAM_W
from ....core.linalg import Sym2, Sym3, V2, V3
from ..contracts import GaussianGrad, Placement
from ..shape import QuatScale

comptime CLIP_MARGIN: Float32 = 0.15  # Jacobian clamp: 15% outside the image


struct PinholeEWA[dt: DType](ImplicitlyCopyable, Movable, Placement):
    """Pinhole projection with the EWA Jacobian (clamped near the image)."""

    comptime dtype = Self.dt

    var cam: SIMD[Self.dt, 32]
    var depth_: Scalar[Self.dt]
    var x: Scalar[Self.dt]
    var y: Scalar[Self.dt]
    var x_cl: Scalar[Self.dt]
    var y_cl: Scalar[Self.dt]
    var valid_x: Bool
    var valid_y: Bool
    var j11: Scalar[Self.dt]
    var j22: Scalar[Self.dt]
    var u: V3[Self.dt]
    var w: V3[Self.dt]
    var uc: V3[Self.dt]
    var wc: V3[Self.dt]
    var cov: Sym2[Self.dt]

    @always_inline
    @staticmethod
    def depth(m: V3[Self.dt], cam: SIMD[Self.dt, 32]) -> Scalar[Self.dt]:
        return cam[8] * m.x + cam[9] * m.y + cam[10] * m.z + cam[11]

    @always_inline
    def __init__(out self, m: V3[Self.dt], shape: QuatScale[Self.dt], cam: SIMD[Self.dt, 32]):
        var c = shape.cov
        self.cam = cam
        self.depth_ = Self.depth(m, cam)
        self.x = (cam[0] * m.x + cam[1] * m.y + cam[2] * m.z + cam[3]) / self.depth_
        self.y = (cam[4] * m.x + cam[5] * m.y + cam[6] * m.z + cam[7]) / self.depth_
        var fx = cam[CAM_FX]
        var fy = cam[CAM_FY]
        var cx = cam[CAM_CX]
        var cy = cam[CAM_CY]
        var margin = CLIP_MARGIN.cast[Self.dt]()
        var clip_l = (-margin * cam[CAM_W] - cx) / fx
        var clip_r = ((1.0 + margin) * cam[CAM_W] - cx) / fx
        var clip_t = (-margin * cam[CAM_H] - cy) / fy
        var clip_b = ((1.0 + margin) * cam[CAM_H] - cy) / fy
        self.x_cl = min(max(self.x, clip_l), clip_r)
        self.y_cl = min(max(self.y, clip_t), clip_b)
        self.valid_x = self.x >= clip_l and self.x <= clip_r
        self.valid_y = self.y >= clip_t and self.y <= clip_b
        self.j11 = fx / self.depth_
        self.j22 = fy / self.depth_
        var j13 = -self.j11 * self.x_cl
        var j23 = -self.j22 * self.y_cl
        var u = V3[Self.dt](
            self.j11 * cam[0] + j13 * cam[8], self.j11 * cam[1] + j13 * cam[9], self.j11 * cam[2] + j13 * cam[10]
        )
        var w = V3[Self.dt](
            self.j22 * cam[4] + j23 * cam[8], self.j22 * cam[5] + j23 * cam[9], self.j22 * cam[6] + j23 * cam[10]
        )
        var uc = V3[Self.dt](
            u.x * c.c00 + u.y * c.c01 + u.z * c.c02,
            u.x * c.c01 + u.y * c.c11 + u.z * c.c12,
            u.x * c.c02 + u.y * c.c12 + u.z * c.c22,
        )
        var wc = V3[Self.dt](
            w.x * c.c00 + w.y * c.c01 + w.z * c.c02,
            w.x * c.c01 + w.y * c.c11 + w.z * c.c12,
            w.x * c.c02 + w.y * c.c12 + w.z * c.c22,
        )
        self.u = u
        self.w = w
        self.uc = uc
        self.wc = wc
        self.cov = Sym2[Self.dt](
            uc.x * u.x + uc.y * u.y + uc.z * u.z,
            uc.x * w.x + uc.y * w.y + uc.z * w.z,
            wc.x * w.x + wc.y * w.y + wc.z * w.z,
        )

    @always_inline
    def mean2d(self) -> V2[Self.dt]:
        return V2[Self.dt](self.x * self.cam[CAM_FX] + self.cam[CAM_CX], self.y * self.cam[CAM_FY] + self.cam[CAM_CY])

    @always_inline
    def cov2d(self) -> Sym2[Self.dt]:
        return self.cov

    @always_inline
    def backward(self, shape: QuatScale[Self.dt], gm: V2[Self.dt], g: Sym2[Self.dt]) -> GaussianGrad[Self.dt]:
        var cam = self.cam
        var u = self.u
        var w = self.w
        var uc = self.uc
        var wc = self.wc
        # 3D covariance gradient (symmetric, per entry).
        var d = Sym3[Self.dt](
            u.x * u.x * g.a + 2.0 * u.x * w.x * g.b + w.x * w.x * g.c,
            u.x * u.y * g.a + (u.x * w.y + u.y * w.x) * g.b + w.x * w.y * g.c,
            u.x * u.z * g.a + (u.x * w.z + u.z * w.x) * g.b + w.x * w.z * g.c,
            u.y * u.y * g.a + 2.0 * u.y * w.y * g.b + w.y * w.y * g.c,
            u.y * u.z * g.a + (u.y * w.z + u.z * w.y) * g.b + w.y * w.z * g.c,
            u.z * u.z * g.a + 2.0 * u.z * w.z * g.b + w.z * w.z * g.c,
        )
        # Gradient of J W, then of the non-zero entries of J.
        var du0 = 2.0 * (uc.x * g.a + wc.x * g.b)
        var du1 = 2.0 * (uc.y * g.a + wc.y * g.b)
        var du2 = 2.0 * (uc.z * g.a + wc.z * g.b)
        var dw0 = 2.0 * (uc.x * g.b + wc.x * g.c)
        var dw1 = 2.0 * (uc.y * g.b + wc.y * g.c)
        var dw2 = 2.0 * (uc.z * g.b + wc.z * g.c)
        var dj11 = cam[0] * du0 + cam[1] * du1 + cam[2] * du2
        var dj22 = cam[4] * dw0 + cam[5] * dw1 + cam[6] * dw2
        var dj13 = cam[8] * du0 + cam[9] * du1 + cam[10] * du2
        var dj23 = cam[8] * dw0 + cam[9] * dw1 + cam[10] * dw2
        var j11 = self.j11
        var j22 = self.j22
        var depth = self.depth_
        # Mean gradient in camera space: from mean2d, then through J (with clamping).
        var dc0 = j11 * gm.x
        var dc1 = j22 * gm.y
        var dc2 = -j11 * self.x * gm.x - j22 * self.y * gm.y
        if self.valid_x:
            dc0 -= j11 * dj13 / depth
        if self.valid_y:
            dc1 -= j22 * dj23 / depth
        var fac_x: Scalar[Self.dt] = 2.0 if self.valid_x else 1.0
        var fac_y: Scalar[Self.dt] = 2.0 if self.valid_y else 1.0
        dc2 += (j11 * (fac_x * self.x_cl * dj13 - dj11) + j22 * (fac_y * self.y_cl * dj23 - dj22)) / depth
        return GaussianGrad[Self.dt](
            V3[Self.dt](
                cam[0] * dc0 + cam[4] * dc1 + cam[8] * dc2,
                cam[1] * dc0 + cam[5] * dc1 + cam[9] * dc2,
                cam[2] * dc0 + cam[6] * dc1 + cam[10] * dc2,
            ),
            shape.backward(d),
        )
