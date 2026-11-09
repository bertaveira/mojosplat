"""The shape of a Gaussian in 3D: raw parameters -> covariance, and back."""

from std.math import exp

from ...core.linalg import Mat3, Sym3, V3, V4


@fieldwise_init
struct ShapeGrad[dt: DType](ImplicitlyCopyable, Movable):
    var log_scales: V3[Self.dt]
    var quat: V4[Self.dt]


struct QuatScale[dt: DType](ImplicitlyCopyable, Movable):
    """(log scales, unnormalized wxyz quaternion) -> 3D covariance."""

    var v0: Scalar[Self.dt]
    var v1: Scalar[Self.dt]
    var v2: Scalar[Self.dt]
    var q: V4[Self.dt]
    var norm_sq: Scalar[Self.dt]
    var R: Mat3[Self.dt]
    var S: Mat3[Self.dt]  # R diag(v)
    var cov: Sym3[Self.dt]

    @always_inline
    def __init__(out self, ls: V3[Self.dt], q: V4[Self.dt]):
        comptime assert Self.dt.is_floating_point(), "needs a float type"
        self.v0 = exp(2.0 * ls.x)
        self.v1 = exp(2.0 * ls.y)
        self.v2 = exp(2.0 * ls.z)
        self.q = q
        var xx = q.x * q.x
        var yy = q.y * q.y
        var zz = q.z * q.z
        var xy = q.x * q.y
        var xz = q.x * q.z
        var yz = q.y * q.z
        var rx = q.w * q.x
        var ry = q.w * q.y
        var rz = q.w * q.z
        self.norm_sq = q.w * q.w + xx + yy + zz
        var s = 2.0 / self.norm_sq
        var R = Mat3[Self.dt](
            1.0 - s * (yy + zz),
            s * (xy - rz),
            s * (xz + ry),
            s * (xy + rz),
            1.0 - s * (xx + zz),
            s * (yz - rx),
            s * (xz - ry),
            s * (yz + rx),
            1.0 - s * (xx + yy),
        )
        var S = Mat3[Self.dt](
            R.m00 * self.v0,
            R.m01 * self.v1,
            R.m02 * self.v2,
            R.m10 * self.v0,
            R.m11 * self.v1,
            R.m12 * self.v2,
            R.m20 * self.v0,
            R.m21 * self.v1,
            R.m22 * self.v2,
        )
        self.R = R
        self.S = S
        self.cov = Sym3[Self.dt](
            S.m00 * R.m00 + S.m01 * R.m01 + S.m02 * R.m02,
            S.m00 * R.m10 + S.m01 * R.m11 + S.m02 * R.m12,
            S.m00 * R.m20 + S.m01 * R.m21 + S.m02 * R.m22,
            S.m10 * R.m10 + S.m11 * R.m11 + S.m12 * R.m12,
            S.m10 * R.m20 + S.m11 * R.m21 + S.m12 * R.m22,
            S.m20 * R.m20 + S.m21 * R.m21 + S.m22 * R.m22,
        )

    @always_inline
    def backward(self, d: Sym3[Self.dt]) -> ShapeGrad[Self.dt]:
        var R = self.R
        var S = self.S
        var dv0 = (
            R.m00 * R.m00 * d.c00
            + R.m10 * R.m10 * d.c11
            + R.m20 * R.m20 * d.c22
            + 2.0 * (R.m00 * R.m10 * d.c01 + R.m00 * R.m20 * d.c02 + R.m10 * R.m20 * d.c12)
        )
        var dv1 = (
            R.m01 * R.m01 * d.c00
            + R.m11 * R.m11 * d.c11
            + R.m21 * R.m21 * d.c22
            + 2.0 * (R.m01 * R.m11 * d.c01 + R.m01 * R.m21 * d.c02 + R.m11 * R.m21 * d.c12)
        )
        var dv2 = (
            R.m02 * R.m02 * d.c00
            + R.m12 * R.m12 * d.c11
            + R.m22 * R.m22 * d.c22
            + 2.0 * (R.m02 * R.m12 * d.c01 + R.m02 * R.m22 * d.c02 + R.m12 * R.m22 * d.c12)
        )
        # dL/dR = 2 D R diag(v)
        var dR00 = 2.0 * (S.m00 * d.c00 + S.m10 * d.c01 + S.m20 * d.c02)
        var dR01 = 2.0 * (S.m01 * d.c00 + S.m11 * d.c01 + S.m21 * d.c02)
        var dR02 = 2.0 * (S.m02 * d.c00 + S.m12 * d.c01 + S.m22 * d.c02)
        var dR10 = 2.0 * (S.m00 * d.c01 + S.m10 * d.c11 + S.m20 * d.c12)
        var dR11 = 2.0 * (S.m01 * d.c01 + S.m11 * d.c11 + S.m21 * d.c12)
        var dR12 = 2.0 * (S.m02 * d.c01 + S.m12 * d.c11 + S.m22 * d.c12)
        var dR20 = 2.0 * (S.m00 * d.c02 + S.m10 * d.c12 + S.m20 * d.c22)
        var dR21 = 2.0 * (S.m01 * d.c02 + S.m11 * d.c12 + S.m21 * d.c22)
        var dR22 = 2.0 * (S.m02 * d.c02 + S.m12 * d.c12 + S.m22 * d.c22)
        var d_xx = dR11 + dR22
        var d_yy = dR00 + dR22
        var d_zz = dR00 + dR11
        var d_rz = dR10 - dR01
        var d_xy = dR10 + dR01
        var d_ry = dR02 - dR20
        var d_xz = dR02 + dR20
        var d_rx = dR21 - dR12
        var d_yz = dR21 + dR12
        var q = self.q
        var two = 2.0 / self.norm_sq
        var helper = two * (
            q.x * q.y * d_xy
            + q.x * q.z * d_xz
            + q.y * q.z * d_yz
            + q.w * q.x * d_rx
            + q.w * q.y * d_ry
            + q.w * q.z * d_rz
            - q.x * q.x * d_xx
            - q.y * q.y * d_yy
            - q.z * q.z * d_zz
        )
        return ShapeGrad[Self.dt](
            V3[Self.dt](2.0 * self.v0 * dv0, 2.0 * self.v1 * dv1, 2.0 * self.v2 * dv2),
            V4[Self.dt](
                two * (q.x * d_rx + q.y * d_ry + q.z * d_rz - q.w * helper),
                two * (q.w * d_rx - 2.0 * q.x * d_xx + q.y * d_xy + q.z * d_xz - q.x * helper),
                two * (q.w * d_ry + q.x * d_xy - 2.0 * q.y * d_yy + q.z * d_yz - q.y * helper),
                two * (q.w * d_rz + q.x * d_xz + q.y * d_yz - 2.0 * q.z * d_zz - q.z * helper),
            ),
        )
