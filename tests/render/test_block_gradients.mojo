"""Finite-difference checks of every block's backward, on the CPU.

Each output is checked on its own: with one combined objective, a wrong
derivative of one output can be hidden by the others.

`QuatScale`, `PinholeEWA` and `ConicFromCov` run in float64, so their tolerance
is tight. `Gaussian2D`, `LoadedGaussian2D` and the spherical harmonics are
float32 like the kernels they run in, so theirs is looser.

Gradients of a symmetric matrix are per entry of the full matrix: an
off-diagonal value counts twice in the objective.
"""

from std.math import exp as _exp
from std.testing import assert_raises, TestSuite

from mojosplat.core.linalg import Sym2, Sym3, V2, V3, V4
from mojosplat.core.splats import ExtraColumns
from mojosplat.core.types import F32Ptr
from mojosplat.render.blocks.appearance.sh import SphericalHarmonics
from mojosplat.render.blocks.placement.pinhole_ewa import PinholeEWA
from mojosplat.render.blocks.response.gaussian2d import Gaussian2D, LoadedGaussian2D
from mojosplat.render.blocks.screen import ConicFromCov
from mojosplat.render.blocks.shape import QuatScale

comptime F64 = DType.float64
comptime X = SIMD[F64, 16]
comptime TOL64 = 1e-6
comptime H64 = 1e-6
comptime TOL32 = 2e-2
comptime QuatScale64 = QuatScale[F64]


def check(name: String, analytic: Float64, numeric: Float64, tol: Float64, floor: Float64) raises:
    var scale = max(max(abs(analytic), abs(numeric)), floor)
    if abs(analytic - numeric) > tol * scale:
        raise Error(String(name, ": analytic ", analytic, " vs numeric ", numeric))


def x_of(a: Float64, b: Float64, c: Float64, d: Float64, e: Float64, f: Float64, g: Float64) -> X:
    var x = X(0.0)
    x[0] = a
    x[1] = b
    x[2] = c
    x[3] = d
    x[4] = e
    x[5] = f
    x[6] = g
    return x


def unit(k: Int) -> SIMD[F64, 8]:
    var w = SIMD[F64, 8](0.0)
    w[k] = 1.0
    return w


def shape_of(x: X, first: Int) -> QuatScale64:
    return QuatScale64(
        V3[F64](x[first], x[first + 1], x[first + 2]),
        V4[F64](x[first + 3], x[first + 4], x[first + 5], x[first + 6]),
    )


# --- QuatScale: (log scales, quaternion) -> covariance


def shape_objective(x: X, d: Sym3[F64], off_diagonal: Float64) -> Float64:
    var c = shape_of(x, 0).cov
    return (
        d.c00 * c.c00 + d.c11 * c.c11 + d.c22 * c.c22 + off_diagonal * (d.c01 * c.c01 + d.c02 * c.c02 + d.c12 * c.c12)
    )


def check_shape(x: X, off_diagonal: Float64) raises:
    for k in range(6):
        var w = unit(k)
        var d = Sym3[F64](w[0], w[1], w[2], w[3], w[4], w[5])
        var g = shape_of(x, 0).backward(d)
        var analytic = x_of(g.log_scales.x, g.log_scales.y, g.log_scales.z, g.quat.w, g.quat.x, g.quat.y, g.quat.z)
        for i in range(7):
            var up = x
            var dn = x
            up[i] += H64
            dn[i] -= H64
            var numeric = (shape_objective(up, d, off_diagonal) - shape_objective(dn, d, off_diagonal)) / (2.0 * H64)
            check(String("QuatScale output ", k, " input ", i), analytic[i], numeric, TOL64, 1e-6)


def shape_points() -> List[X]:
    return [
        x_of(-1.2, -0.3, -2.0, 0.8, 0.1, -0.4, 0.3),
        x_of(0.4, -1.5, -0.7, 1.7, -0.9, 0.2, 1.1),  # quaternion far from unit length
    ]


def test_quat_scale() raises:
    for x in shape_points():
        check_shape(x, 2.0)


def test_checker_catches_a_wrong_output() raises:
    """Control: treating off-diagonals as single values must be caught, and
    only an off-diagonal output can reveal it."""
    with assert_raises():
        check_shape(shape_points()[0], 1.0)


# --- ConicFromCov: covariance -> its inverse


def conic_objective(x: X, g: Sym2[F64]) -> Float64:
    var c = ConicFromCov[F64](Sym2[F64](x[0], x[1], x[2])).conic
    return g.a * c.a + g.c * c.c + 2.0 * g.b * c.b


def test_conic_from_cov() raises:
    var x = X(0.0)
    x[0] = 5.0
    x[1] = -1.3
    x[2] = 3.0
    for k in range(3):
        var w = unit(k)
        var g = Sym2[F64](w[0], w[1], w[2])
        var r = ConicFromCov[F64](Sym2[F64](x[0], x[1], x[2])).backward(g)
        var analytic = X(0.0)
        analytic[0] = r.a
        analytic[1] = 2.0 * r.b  # per entry: the single stored value counts twice
        analytic[2] = r.c
        for i in range(3):
            var up = x
            var dn = x
            up[i] += H64
            dn[i] -= H64
            var numeric = (conic_objective(up, g) - conic_objective(dn, g)) / (2.0 * H64)
            check(String("ConicFromCov output ", k, " input ", i), analytic[i], numeric, TOL64, 1e-8)


# --- PinholeEWA: (mean, shape) -> position and 2D covariance


def camera() -> SIMD[F64, 32]:
    var c = SIMD[F64, 32](0.0)
    var (cs, sn) = (0.8660254037844387, 0.5)
    c[0] = cs
    c[2] = sn
    c[3] = 0.2
    c[5] = 1.0
    c[7] = -0.1
    c[8] = -sn
    c[10] = cs
    c[11] = 3.0
    c[12] = 500.0
    c[13] = 520.0
    c[14] = 320.0
    c[15] = 240.0
    c[16] = 640.0
    c[17] = 480.0
    return c


def place_of(x: X) -> PinholeEWA[F64]:
    return PinholeEWA[F64](V3[F64](x[0], x[1], x[2]), shape_of(x, 3), camera())


def place_objective(x: X, gm: V2[F64], gc: Sym2[F64]) -> Float64:
    var p = place_of(x)
    var u = p.mean2d()
    var c = p.cov2d()
    return gm.x * u.x + gm.y * u.y + gc.a * c.a + gc.c * c.c + 2.0 * gc.b * c.b


def check_place(x: X) raises:
    for k in range(5):
        var w = unit(k)
        var gm = V2[F64](w[0], w[1])
        var gc = Sym2[F64](w[2], w[3], w[4])
        var g = place_of(x).backward(shape_of(x, 3), gm, gc)
        var a = X(0.0)
        a[0] = g.mean.x
        a[1] = g.mean.y
        a[2] = g.mean.z
        a[3] = g.shape.log_scales.x
        a[4] = g.shape.log_scales.y
        a[5] = g.shape.log_scales.z
        a[6] = g.shape.quat.w
        a[7] = g.shape.quat.x
        a[8] = g.shape.quat.y
        a[9] = g.shape.quat.z
        for i in range(10):
            var up = x
            var dn = x
            up[i] += H64
            dn[i] -= H64
            var numeric = (place_objective(up, gm, gc) - place_objective(dn, gm, gc)) / (2.0 * H64)
            check(String("PinholeEWA output ", k, " input ", i), a[i], numeric, TOL64, 1e-6)


def place_point(mx: Float64, my: Float64, mz: Float64) -> X:
    var x = X(0.0)
    x[0] = mx
    x[1] = my
    x[2] = mz
    x[3] = -1.2
    x[4] = -0.3
    x[5] = -2.0
    x[6] = 0.8
    x[7] = 0.1
    x[8] = -0.4
    x[9] = 0.3
    return x


def test_pinhole_ewa_inside_the_image() raises:
    var x = place_point(0.3, 0.2, 0.4)
    var p = place_of(x)
    if not (p.valid_x and p.valid_y):
        raise Error("expected a point inside the image")
    check_place(x)


def test_pinhole_ewa_with_clamped_jacobian() raises:
    var x = place_point(9.0, 7.0, 0.4)  # far outside the image, in x and y
    var p = place_of(x)
    if p.valid_x or p.valid_y:
        raise Error("expected a point outside the clamp margin")
    check_place(x)


# --- Gaussian2D: float32, as in the kernels


def conic_of(a: Float32, b: Float32, c: Float32) -> SIMD[DType.float32, 8]:
    var mean = V2[DType.float32](10.0, 20.0)
    return Gaussian2D(mean, Sym2[DType.float32](a, b, c)).record(0.5)


def test_gaussian2d_covariance_gradient() raises:
    var mean = V2[DType.float32](10.0, 20.0)
    var cov: SIMD[DType.float32, 4] = SIMD[DType.float32, 4](5.0, -1.3, 3.0, 0.0)
    var h: Float32 = 0.02
    for k in range(3):
        var slot = SIMD[DType.float32, 8](0.0)
        slot[2 + k] = 1.0
        var r = Gaussian2D(mean, Sym2[DType.float32](cov[0], cov[1], cov[2])).backward(slot)
        var analytic = SIMD[DType.float32, 4](r.cov2d.a, 2.0 * r.cov2d.b, r.cov2d.c, 0.0)
        var weight = 2.0 if k == 1 else 1.0
        for i in range(3):
            var up = cov
            var dn = cov
            up[i] += h
            dn[i] -= h
            var hi = conic_of(up[0], up[1], up[2])
            var lo = conic_of(dn[0], dn[1], dn[2])
            var numeric = weight * (Float64(hi[2 + k]) - Float64(lo[2 + k])) / (2.0 * Float64(h))
            check(String("Gaussian2D conic ", k, " input ", i), Float64(analytic[i]), numeric, TOL32, 1e-3)


def alpha_of(p: SIMD[DType.float32, 8], px: Float32, py: Float32) -> Float32:
    """Alpha for p = (mx, my, a, b, c, opacity logit)."""
    var rec = p
    rec[5] = 1.0 / (1.0 + Float32(_exp(-Float64(p[5]))))
    return LoadedGaussian2D(rec, SIMD[DType.float32, 8](0.0)).eval(px, py)[0]


def test_loaded_gaussian2d_pixel_gradient() raises:
    var p = SIMD[DType.float32, 8](10.0, 20.0, 0.31, -0.07, 0.22, 0.4, 0.0, 0.0)
    var pixels: List[SIMD[DType.float32, 2]] = [
        SIMD[DType.float32, 2](11.5, 19.5),
        SIMD[DType.float32, 2](8.0, 23.0),
    ]
    var h: Float32 = 0.01
    for pixel in pixels:
        var rec = p
        rec[5] = 1.0 / (1.0 + Float32(_exp(-Float64(p[5]))))
        var loaded = LoadedGaussian2D(rec, SIMD[DType.float32, 8](0.0))
        loaded.add(loaded.eval(pixel[0], pixel[1]), 1.0)
        var g = loaded.gradient()
        for i in range(6):
            var up = p
            var dn = p
            up[i] += h
            dn[i] -= h
            var numeric = (Float64(alpha_of(up, pixel[0], pixel[1])) - Float64(alpha_of(dn, pixel[0], pixel[1]))) / (
                2.0 * Float64(h)
            )
            if i == 3:
                numeric *= 0.5  # the conic's off-diagonal gradient is per entry
            check(String("LoadedGaussian2D input ", i), Float64(g[i]), numeric, TOL32, 1e-3)


# --- Spherical harmonics


def ptr(mut values: List[Float32]) -> F32Ptr:
    return values.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()


def color_of(
    mut sh0: List[Float32], mut rest: List[Float32], dir: SIMD[DType.float32, 4], channel: Int, active: Int32
) -> Float64:
    var cols = ExtraColumns(ptr(sh0), ptr(rest))
    return Float64(SphericalHarmonics.color(cols, 0, dir, SIMD[DType.int32, 4](active, 15, 0, 0))[channel])


def check_sh(active: Int32) raises:
    var sh0: List[Float32] = [0.4, -0.2, 0.7]
    var rest = List[Float32](capacity=45)
    for i in range(45):
        rest.append(0.05 * Float32(i % 7) - 0.15)
    var dir = SIMD[DType.float32, 4](0.3, -0.5, 0.8, 0.0)
    var grad0 = List[Float32](length=3, fill=0.0)
    var grad_rest = List[Float32](length=45, fill=0.0)
    var h: Float32 = 0.01
    for c in range(3):
        var gc = SIMD[DType.float32, 4](0.0)
        gc[c] = 1.0
        var cols = ExtraColumns(ptr(sh0), ptr(rest))
        var grads = ExtraColumns(ptr(grad0), ptr(grad_rest))
        var settings = SIMD[DType.int32, 4](active, 15, 0, 0)
        var dmean = SphericalHarmonics.backward(cols, grads, 0, dir, gc, settings, True)

        for k in range(3):
            var keep = sh0[k]
            sh0[k] = keep + h
            var hi = color_of(sh0, rest, dir, c, active)
            sh0[k] = keep - h
            var lo = color_of(sh0, rest, dir, c, active)
            sh0[k] = keep
            check(String("SH sh0 ", k, " channel ", c), Float64(grad0[k]), (hi - lo) / (2.0 * Float64(h)), TOL32, 1e-3)
        for k in range(45):
            var keep = rest[k]
            rest[k] = keep + h
            var hi = color_of(sh0, rest, dir, c, active)
            rest[k] = keep - h
            var lo = color_of(sh0, rest, dir, c, active)
            rest[k] = keep
            check(
                String("SH rest ", k, " channel ", c),
                Float64(grad_rest[k]),
                (hi - lo) / (2.0 * Float64(h)),
                TOL32,
                1e-3,
            )
        for k in range(3):
            var up = dir
            var dn = dir
            up[k] += h
            dn[k] -= h
            var numeric = (color_of(sh0, rest, up, c, active) - color_of(sh0, rest, dn, c, active)) / (2.0 * Float64(h))
            check(String("SH view direction ", k, " channel ", c), Float64(dmean[k]), numeric, TOL32, 1e-3)


def test_spherical_harmonics_full_degree() raises:
    check_sh(16)


def test_spherical_harmonics_partial_degree() raises:
    check_sh(4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
