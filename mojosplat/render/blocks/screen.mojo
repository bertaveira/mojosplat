"""Image-space operations on a projected Gaussian."""

from ...core.linalg import Sym2

comptime DILATION: Float32 = 0.3


struct ConicFromCov[dt: DType](ImplicitlyCopyable, Movable):
    """2D covariance -> conic (its inverse)."""

    var cov: Sym2[Self.dt]
    var det: Scalar[Self.dt]
    var conic: Sym2[Self.dt]

    @always_inline
    def __init__(out self):
        self.cov = Sym2[Self.dt](0.0, 0.0, 0.0)
        self.det = 0.0
        self.conic = Sym2[Self.dt](0.0, 0.0, 0.0)

    @always_inline
    def __init__(out self, cov: Sym2[Self.dt]):
        self.cov = cov
        self.det = cov.a * cov.c - cov.b * cov.b
        self.conic = Sym2[Self.dt](cov.c / self.det, -cov.b / self.det, cov.a / self.det)

    @always_inline
    def backward(self, g: Sym2[Self.dt]) -> Sym2[Self.dt]:
        var a = self.cov.a
        var b = self.cov.b
        var c = self.cov.c
        var rs = 1.0 / (self.det * self.det)
        return Sym2[Self.dt](
            rs * (2.0 * b * c * g.b - c * c * g.a - b * b * g.c),
            rs * (b * c * g.a - (a * c + b * b) * g.b + a * b * g.c),
            rs * (2.0 * a * b * g.b - b * b * g.a - a * a * g.c),
        )
