"""Small float-generic vectors and matrices for per-Gaussian math.

Plain value types: kernels use float32, CPU gradient checks float64. Buffers
are float32, so `load` / `store` cast.
"""

from .types import F32Ptr


@fieldwise_init
struct V2[dt: DType](ImplicitlyCopyable, Movable):
    var x: Scalar[Self.dt]
    var y: Scalar[Self.dt]

    @always_inline
    @staticmethod
    def load(p: F32Ptr, i: Int) -> Self:
        return Self(p[unsafe_offset=i * 2 + 0].cast[Self.dt](), p[unsafe_offset=i * 2 + 1].cast[Self.dt]())

    @always_inline
    def f32(self) -> V2[DType.float32]:
        return V2[DType.float32](self.x.cast[DType.float32](), self.y.cast[DType.float32]())


@fieldwise_init
struct V3[dt: DType](ImplicitlyCopyable, Movable):
    var x: Scalar[Self.dt]
    var y: Scalar[Self.dt]
    var z: Scalar[Self.dt]

    @always_inline
    @staticmethod
    def load(p: F32Ptr, i: Int) -> Self:
        return Self(
            p[unsafe_offset=i * 3 + 0].cast[Self.dt](),
            p[unsafe_offset=i * 3 + 1].cast[Self.dt](),
            p[unsafe_offset=i * 3 + 2].cast[Self.dt](),
        )

    @always_inline
    def store(self, p: F32Ptr, i: Int):
        p[unsafe_offset=i * 3 + 0] = self.x.cast[DType.float32]()
        p[unsafe_offset=i * 3 + 1] = self.y.cast[DType.float32]()
        p[unsafe_offset=i * 3 + 2] = self.z.cast[DType.float32]()


@fieldwise_init
struct V4[dt: DType](ImplicitlyCopyable, Movable):
    var w: Scalar[Self.dt]
    var x: Scalar[Self.dt]
    var y: Scalar[Self.dt]
    var z: Scalar[Self.dt]

    @always_inline
    @staticmethod
    def load(p: F32Ptr, i: Int) -> Self:
        return Self(
            p[unsafe_offset=i * 4 + 0].cast[Self.dt](),
            p[unsafe_offset=i * 4 + 1].cast[Self.dt](),
            p[unsafe_offset=i * 4 + 2].cast[Self.dt](),
            p[unsafe_offset=i * 4 + 3].cast[Self.dt](),
        )

    @always_inline
    def store(self, p: F32Ptr, i: Int):
        p[unsafe_offset=i * 4 + 0] = self.w.cast[DType.float32]()
        p[unsafe_offset=i * 4 + 1] = self.x.cast[DType.float32]()
        p[unsafe_offset=i * 4 + 2] = self.y.cast[DType.float32]()
        p[unsafe_offset=i * 4 + 3] = self.z.cast[DType.float32]()


@fieldwise_init
struct Sym2[dt: DType](ImplicitlyCopyable, Movable):
    """Symmetric 2x2 matrix [[a, b], [b, c]].

    As a gradient it is per entry of the full matrix: for a scalar loss L,
    `b` is half of dL/db taken with respect to the single stored value.
    """

    var a: Scalar[Self.dt]
    var b: Scalar[Self.dt]
    var c: Scalar[Self.dt]

    @always_inline
    def f32(self) -> Sym2[DType.float32]:
        return Sym2[DType.float32](
            self.a.cast[DType.float32](), self.b.cast[DType.float32](), self.c.cast[DType.float32]()
        )

    @always_inline
    def cast[to: DType](self) -> Sym2[to]:
        return Sym2[to](self.a.cast[to](), self.b.cast[to](), self.c.cast[to]())


@fieldwise_init
struct Sym3[dt: DType](ImplicitlyCopyable, Movable):
    """Symmetric 3x3 matrix; a gradient is per entry of the full matrix."""

    var c00: Scalar[Self.dt]
    var c01: Scalar[Self.dt]
    var c02: Scalar[Self.dt]
    var c11: Scalar[Self.dt]
    var c12: Scalar[Self.dt]
    var c22: Scalar[Self.dt]


@fieldwise_init
struct Mat3[dt: DType](ImplicitlyCopyable, Movable):
    var m00: Scalar[Self.dt]
    var m01: Scalar[Self.dt]
    var m02: Scalar[Self.dt]
    var m10: Scalar[Self.dt]
    var m11: Scalar[Self.dt]
    var m12: Scalar[Self.dt]
    var m20: Scalar[Self.dt]
    var m21: Scalar[Self.dt]
    var m22: Scalar[Self.dt]
