"""Beta footprint: pixel gradients, and the column round trip through the rasterizer."""

from max.gpu.host import DeviceContext
from std.math import exp, log
from std.testing import assert_true, TestSuite

from mojosplat.core.camera import Camera
from mojosplat.core.types import dptr
from mojosplat.gpu.elementwise import fill_f32
from mojosplat.io.arrays import download
from mojosplat.model.splats import Splats
from mojosplat.render.blocks.response.beta import LoadedBeta
from mojosplat.core.math import SH_C0
from mojosplat.core.splats import ExtraColumns
from mojosplat.core.types import F32Ptr
from mojosplat.render.blocks.appearance.spherical_beta import SphericalBetaColor
from mojosplat.render.pipelines import BetaSplattingSH
from mojosplat.render.rasterizer import Rasterizer

comptime F8 = SIMD[DType.float32, 8]


def loaded(p: F8) -> LoadedBeta:
    """p = (mx, my, a, b, c, opacity logit, raw beta)."""
    var rec = p
    rec[5] = 1.0 / (1.0 + Float32(exp(-Float64(p[5]))))
    var extra = F8(0.0)
    extra[0] = 4.0 * Float32(exp(Float64(p[6])))
    return LoadedBeta(rec, extra)


def test_pixel_gradients_match_finite_differences() raises:
    var p = F8(10.0, 20.0, 0.06, -0.012, 0.045, 0.4, -0.9, 0.0)
    var pixels: List[SIMD[DType.float32, 2]] = [SIMD[DType.float32, 2](11.5, 19.5), SIMD[DType.float32, 2](8.0, 23.0)]
    var h: Float32 = 0.002
    for pixel in pixels:
        var l = loaded(p)
        l.add(l.eval(pixel[0], pixel[1]), 1.0)
        var g = l.gradient()
        for i in range(7):
            var up = p
            var dn = p
            up[i] += h
            dn[i] -= h
            var hi = Float64(loaded(up).eval(pixel[0], pixel[1])[0])
            var lo = Float64(loaded(dn).eval(pixel[0], pixel[1])[0])
            var numeric = (hi - lo) / (2.0 * Float64(h))
            if i == 3:
                numeric *= 0.5  # the conic's off-diagonal gradient is per entry
            var scale = max(max(abs(Float64(g[i])), abs(numeric)), 1e-3)
            assert_true(abs(Float64(g[i]) - numeric) < 0.02 * scale, String("input ", i, ": ", g[i], " vs ", numeric))


def test_support_is_exactly_the_ellipse() raises:
    var l = loaded(F8(10.0, 20.0, 0.04, 0.0, 0.04, 2.0, 0.0, 0.0))  # radius 5
    assert_true(l.eval(10.5 + 4.4, 20.0)[0] > 0.0, "inside the ellipse")
    assert_true(l.eval(10.0 + 5.1, 20.0)[0] == 0.0, "outside the ellipse")


def scene(ctx: DeviceContext, raw_beta: Float32) raises -> Splats:
    var means: List[Float32] = [0.0, 0.0, 2.0]
    var scales: List[Float32] = [-2.0, -2.0, -2.0]
    var quats: List[Float32] = [1.0, 0.0, 0.0, 0.0]
    var opacity: List[Float32] = [0.0]
    var sh0: List[Float32] = [1.0, 1.0, 1.0]
    var splats = Splats(ctx, means, scales, quats, opacity, sh0, List[Float32](), 0)
    var value: List[Float32] = [raw_beta]
    splats.add_column("beta", 1, values=value)
    return splats^


def camera() -> Camera:
    var view = SIMD[DType.float32, 16](1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    return Camera(view, 100.0, 100.0, 32.0, 32.0, 64, 64, 0.01, 100.0)


def image_sum(raw_beta: Float32) raises -> Float64:
    var ctx = DeviceContext()
    var splats = scene(ctx, raw_beta)
    var image = ctx.enqueue_create_buffer[DType.float32](64 * 64 * 3)
    var rast = Rasterizer(ctx)
    rast.render[BetaSplattingSH](splats.view(), camera(), SIMD[DType.float32, 4](0.0), dptr(image), 0)
    ctx.synchronize()
    var total: Float64 = 0.0
    for v in download(ctx, image, 64 * 64 * 3):
        total += Float64(v)
    _ = (image^, splats^)
    return total


def test_beta_gradient_reaches_its_column() raises:
    var ctx = DeviceContext()
    var raw: Float32 = -0.7
    var splats = scene(ctx, raw)
    splats.enable_training()
    var image = ctx.enqueue_create_buffer[DType.float32](64 * 64 * 3)
    var grad_image = ctx.enqueue_create_buffer[DType.float32](64 * 64 * 3)
    fill_f32(ctx, dptr(grad_image), 1.0, 64 * 64 * 3)
    var rast = Rasterizer(ctx)
    var view = splats.view()
    var st = rast.forward[BetaSplattingSH](view, camera(), SIMD[DType.float32, 4](0.0), dptr(image), 0)
    rast.backward[BetaSplattingSH](view, st, dptr(image), dptr(grad_image), splats.grad_view())
    ctx.synchronize()
    var analytic = Float64(download(ctx, splats.grads[6], 1)[0])
    var h: Float32 = 0.01
    var numeric = (image_sum(raw + h) - image_sum(raw - h)) / Float64(2.0 * h)
    assert_true(abs(analytic / numeric - 1.0) < 0.01, String(analytic, " vs ", numeric))
    _ = (image^, grad_image^, splats^)


def ptr(mut values: List[Float32]) -> F32Ptr:
    return values.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()


def sb_color(mut sh0: List[Float32], mut lobes: List[Float32], dir: SIMD[DType.float32, 4], c: Int) -> Float64:
    var rest = List[Float32](length=1, fill=0.0)
    var cols = ExtraColumns(ptr(sh0), ptr(rest), ptr(lobes))
    return Float64(SphericalBetaColor[2].color(cols, 0, dir, SIMD[DType.int32, 4](1, 0, 0, 0))[c])


def test_spherical_beta_gradients_match_finite_differences() raises:
    var sh0: List[Float32] = [0.4, -0.2, 0.7]
    # Two lobes facing the view direction, so both contribute: r, g, b, theta, phi, b
    var lobes: List[Float32] = [0.1, -0.2, 0.3, 1.0, 1.9, -0.5, 0.05, 0.2, -0.1, 1.4, 2.4, 0.2]
    var dir = SIMD[DType.float32, 4](-0.3, 0.8, 0.5, 0.0)
    var empty = List[Float32](length=1, fill=0.0)
    var grad0 = List[Float32](length=3, fill=9.0)
    var grad_l = List[Float32](length=12, fill=9.0)
    var h: Float32 = 0.002
    for c in range(3):
        var gc = SIMD[DType.float32, 4](0.0)
        gc[c] = 1.0
        var cols = ExtraColumns(ptr(sh0), ptr(empty), ptr(lobes))
        var grads = ExtraColumns(ptr(grad0), ptr(empty), ptr(grad_l))
        var dmean = SphericalBetaColor[2].backward(cols, grads, 0, dir, gc, SIMD[DType.int32, 4](1, 0, 0, 0), True)
        assert_true(dmean[0] == 0.0 and dmean[2] == 0.0, "the official code has no view-direction gradient")
        for k in range(3):
            var keep = sh0[k]
            sh0[k] = keep + h
            var hi = sb_color(sh0, lobes, dir, c)
            sh0[k] = keep - h
            var lo = sb_color(sh0, lobes, dir, c)
            sh0[k] = keep
            var numeric = (hi - lo) / (2.0 * Float64(h))
            assert_true(
                abs(Float64(grad0[k]) - numeric) < 0.02 * max(abs(numeric), 1e-3),
                String("sh0 ", k, " ", grad0[k], " vs ", numeric),
            )
        for k in range(12):
            var keep = lobes[k]
            lobes[k] = keep + h
            var hi = sb_color(sh0, lobes, dir, c)
            lobes[k] = keep - h
            var lo = sb_color(sh0, lobes, dir, c)
            lobes[k] = keep
            var numeric = (hi - lo) / (2.0 * Float64(h))
            assert_true(
                abs(Float64(grad_l[k]) - numeric) < 0.02 * max(max(abs(numeric), abs(Float64(grad_l[k]))), 2e-3),
                String("lobe param ", k, " channel ", c, ": ", grad_l[k], " vs ", numeric),
            )


def test_spherical_beta_clamps_the_base_color() raises:
    var sh0: List[Float32] = [-5.0, 0.0, 0.0]
    var lobes = List[Float32](length=12, fill=0.0)
    var dir = SIMD[DType.float32, 4](0.0, 0.0, -1.0, 0.0)  # both lobes face away (theta = 0)
    assert_true(sb_color(sh0, lobes, dir, 0) == 0.0, "negative base color clamps at 0")
    assert_true(abs(sb_color(sh0, lobes, dir, 1) - 0.5) < 1e-6, "base color")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
