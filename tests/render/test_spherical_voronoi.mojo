"""Spherical Voronoi: the color against a direct evaluation, its gradients, and its starting values."""

from max.gpu.host import DeviceContext
from std.math import cos, exp, sqrt
from std.testing import assert_equal, assert_true, TestSuite

from mojosplat.core.camera import Camera
from mojosplat.core.splats import ExtraColumns
from mojosplat.core.types import F32Ptr, dptr
from mojosplat.gpu.elementwise import fill_f32
from mojosplat.io.arrays import download
from mojosplat.model.splats import Attr, Splats
from mojosplat.render.blocks.appearance.spherical_voronoi import SphericalVoronoi
from mojosplat.render.pipelines import SphericalVoronoiBeta
from mojosplat.render.rasterizer import Rasterizer

comptime K = 3
comptime SV = SphericalVoronoi[K]
comptime NO_SETTINGS = SIMD[DType.int32, 4](0)
comptime F4 = SIMD[DType.float32, 4]


def ptr(mut values: List[Float32]) -> F32Ptr:
    return values.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()


def params() -> Tuple[List[Float32], List[Float32], List[Float32]]:
    var sites: List[Float32] = [0.3, -0.9, 0.2, -1.4, 0.7, 0.5, 0.2, 1.1, 1.6]
    var colors: List[Float32] = [0.2, -0.4, 0.1, -0.3, 0.6, 0.35, 0.05, 0.25, -0.2]
    var log_tau: List[Float32] = [0.4, -0.6, 1.1]
    return (sites^, colors^, log_tau^)


def eval_color(
    mut sites: List[Float32], mut colors: List[Float32], mut log_tau: List[Float32], dir: F4, c: Int
) -> Float64:
    var unused = List[Float32](length=3, fill=0.0)
    var cols = ExtraColumns(ptr(unused), ptr(unused), ptr(sites), ptr(colors), ptr(log_tau))
    return Float64(SV.color(cols, 0, dir, NO_SETTINGS)[c])


def test_color_matches_a_direct_evaluation() raises:
    var p = params()
    var sites = p[0].copy()
    var colors = p[1].copy()
    var log_tau = p[2].copy()
    var dir = F4(0.5, -1.2, 0.8, 0.0)
    var dn = sqrt(0.25 + 1.44 + 0.64)
    var v: List[Float64] = [0.5 / dn, -1.2 / dn, 0.8 / dn]
    var scores = List[Float64]()
    for i in range(K):
        var s0 = Float64(sites[i * 3])
        var s1 = Float64(sites[i * 3 + 1])
        var s2 = Float64(sites[i * 3 + 2])
        var n = sqrt(s0 * s0 + s1 * s1 + s2 * s2)
        var d0 = s0 / n - v[0]
        var d1 = s1 / n - v[1]
        var d2 = s2 / n - v[2]
        scores.append(exp(-exp(Float64(log_tau[i])) * sqrt(d0 * d0 + d1 * d1 + d2 * d2)))
    var total = scores[0] + scores[1] + scores[2]
    for c in range(3):
        var expected = 0.5
        for i in range(K):
            expected += scores[i] / total * Float64(colors[i * 3 + c])
        var got = eval_color(sites, colors, log_tau, dir, c)
        assert_true(abs(got - expected) < 1e-5, String(got, " vs ", expected))


def check_column(
    mut sites: List[Float32],
    mut colors: List[Float32],
    mut log_tau: List[Float32],
    which: Int,
    analytic: List[Float32],
    dir: F4,
    c: Int,
) raises:
    var h: Float32 = 0.002
    for k in range(len(analytic)):
        var hi: Float64
        var lo: Float64
        if which == 0:
            var keep = sites[k]
            sites[k] = keep + h
            hi = eval_color(sites, colors, log_tau, dir, c)
            sites[k] = keep - h
            lo = eval_color(sites, colors, log_tau, dir, c)
            sites[k] = keep
        elif which == 1:
            var keep = colors[k]
            colors[k] = keep + h
            hi = eval_color(sites, colors, log_tau, dir, c)
            colors[k] = keep - h
            lo = eval_color(sites, colors, log_tau, dir, c)
            colors[k] = keep
        else:
            var keep = log_tau[k]
            log_tau[k] = keep + h
            hi = eval_color(sites, colors, log_tau, dir, c)
            log_tau[k] = keep - h
            lo = eval_color(sites, colors, log_tau, dir, c)
            log_tau[k] = keep
        var numeric = (hi - lo) / (2.0 * Float64(h))
        var scale = max(max(abs(numeric), abs(Float64(analytic[k]))), 2e-3)
        assert_true(
            abs(Float64(analytic[k]) - numeric) < 0.02 * scale,
            String("column ", which, " entry ", k, " channel ", c, ": ", analytic[k], " vs ", numeric),
        )


def test_gradients_match_finite_differences() raises:
    var p = params()
    var sites = p[0].copy()
    var colors = p[1].copy()
    var log_tau = p[2].copy()
    var dir = F4(0.5, -1.2, 0.8, 0.0)
    var unused = List[Float32](length=3, fill=9.0)
    for c in range(3):
        var g_sites = List[Float32](length=K * 3, fill=9.0)
        var g_colors = List[Float32](length=K * 3, fill=9.0)
        var g_tau = List[Float32](length=K, fill=9.0)
        var gc = F4(0.0)
        gc[c] = 1.0
        var cols = ExtraColumns(ptr(unused), ptr(unused), ptr(sites), ptr(colors), ptr(log_tau))
        var grads = ExtraColumns(ptr(unused), ptr(unused), ptr(g_sites), ptr(g_colors), ptr(g_tau))
        var dmean = SV.backward(cols, grads, 0, dir, gc, NO_SETTINGS, True)
        assert_true(dmean[0] == 0.0 and dmean[1] == 0.0 and dmean[2] == 0.0, "no view-direction gradient")
        check_column(sites, colors, log_tau, 0, g_sites, dir, c)
        check_column(sites, colors, log_tau, 1, g_colors, dir, c)
        check_column(sites, colors, log_tau, 2, g_tau, dir, c)


def test_starting_values() raises:
    var rgb: List[UInt8] = [255, 0, 51, 128, 64, 0]
    var sites = SphericalVoronoi[8].init_column(0, 2, rgb)
    assert_equal(len(sites), 2 * 8 * 3)
    for i in range(8):
        var x = Float64(sites[i * 3])
        var y = Float64(sites[i * 3 + 1])
        var z = Float64(sites[i * 3 + 2])
        assert_true(abs(x * x + y * y + z * z - 1.0) < 1e-5, "unit sites")
        assert_true(sites[i * 3] == sites[(i + 8) * 3], "the same sites for every Gaussian")
    assert_true(abs(sites[1] - (1.0 - 1.0 / 8.0)) < 1e-6, "first site: y = 1 - 0.5 / N * 2")
    var colors = SphericalVoronoi[8].init_column(1, 2, rgb)
    assert_true(abs(colors[0] - 0.5) < 1e-6 and abs(colors[1] + 0.5) < 1e-6, "point color - 0.5")
    assert_true(abs(colors[8 * 3 + 2] + 0.5) < 1e-6, "second Gaussian")
    assert_true(abs(colors[7 * 3] - 0.5) < 1e-6, "repeated on every lobe")
    var tau = SphericalVoronoi[8].init_column(2, 2, rgb)
    assert_equal(len(tau), 16)
    assert_true(tau[15] == 0.0, "temperature 1")


def test_learning_rate_schedule() raises:
    comptime S = SphericalVoronoi[8]
    var total = 30_000
    assert_true(abs(Float64(S.lr_scale(0, 1, total, 3000)) - 1.0) < 1e-4, "starts at the full rate")
    var mid = Float64(S.lr_scale(0, 15_000, total, 3000))
    assert_true(abs(mid - (0.1 + 0.45 * (1.0 + cos(3.14159265358979 / 2.0)))) < 1e-5, "cosine")
    assert_true(S.lr_scale(1, 29_999, total, 3000) == 1.0, "colors never change rate")
    assert_true(S.lr_scale(0, 27_001, total, 3000) == 0.0, "sites freeze")
    assert_true(S.lr_scale(0, 27_000, total, 3000) > 0.0, "the last iteration before the freeze still moves them")
    assert_true(S.lr_scale(2, 29_999, total, 3000) == S.lr_scale(2, 27_000, total, 3000), "temperature rate is held")


def camera() -> Camera:
    var view = SIMD[DType.float32, 16](1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    return Camera(view, 100.0, 100.0, 32.0, 32.0, 64, 64, 0.01, 100.0)


def scene(ctx: DeviceContext, colors: List[Float32], raw_beta: Float32) raises -> Splats:
    var means: List[Float32] = [0.0, 0.0, 2.0]
    var scales: List[Float32] = [-2.0, -2.0, -2.0]
    var quats: List[Float32] = [1.0, 0.0, 0.0, 0.0]
    var opacity: List[Float32] = [0.0]
    var sh0: List[Float32] = [0.0, 0.0, 0.0]
    var splats = Splats(ctx, means, scales, quats, opacity, sh0, List[Float32](), 0)
    var rgb = List[UInt8](length=3, fill=128)
    comptime S = SphericalVoronoi[8]
    splats.add_column("sites", 24, values=S.init_column(0, 1, rgb))
    splats.add_column("colors", 24, values=colors)
    splats.add_column("log_tau", 8, values=S.init_column(2, 1, rgb))
    var beta: List[Float32] = [raw_beta]
    splats.add_column("beta", 1, values=beta)
    return splats^


def image_sum(colors: List[Float32], raw_beta: Float32) raises -> Float64:
    var ctx = DeviceContext()
    var splats = scene(ctx, colors, raw_beta)
    var image = ctx.enqueue_create_buffer[DType.float32](64 * 64 * 3)
    var rast = Rasterizer(ctx)
    rast.render[SphericalVoronoiBeta](splats.view(), camera(), F4(0.0), dptr(image), 0)
    ctx.synchronize()
    var total: Float64 = 0.0
    for v in download(ctx, image, 64 * 64 * 3):
        total += Float64(v)
    _ = (image^, splats^)
    return total


def test_columns_reach_the_gpu_kernels() raises:
    """Lobe color and Beta gradients through the rasterizer: six extra columns in the right places."""
    var ctx = DeviceContext()
    var colors = List[Float32](length=24, fill=0.1)
    var raw: Float32 = -0.7
    var splats = scene(ctx, colors, raw)
    splats.enable_training()
    var image = ctx.enqueue_create_buffer[DType.float32](64 * 64 * 3)
    var grad_image = ctx.enqueue_create_buffer[DType.float32](64 * 64 * 3)
    fill_f32(ctx, dptr(grad_image), 1.0, 64 * 64 * 3)
    var rast = Rasterizer(ctx)
    var view = splats.view()
    var st = rast.forward[SphericalVoronoiBeta](view, camera(), F4(0.0), dptr(image), 0)
    rast.backward[SphericalVoronoiBeta](view, st, dptr(image), dptr(grad_image), splats.grad_view())
    ctx.synchronize()
    var h: Float32 = 0.01
    var color_grads = download(ctx, splats.grads[Attr.COUNT + 1], 24)
    for k in range(24):
        var up = colors.copy()
        var dn = colors.copy()
        up[k] += h
        dn[k] -= h
        var numeric = (image_sum(up, raw) - image_sum(dn, raw)) / Float64(2.0 * h)
        var analytic = Float64(color_grads[k])
        assert_true(
            abs(analytic - numeric) < 0.02 * max(abs(numeric), 1e-2),
            String("color ", k, ": ", analytic, " vs ", numeric),
        )
    var beta_grad = Float64(download(ctx, splats.grads[Attr.COUNT + 3], 1)[0])
    var numeric = (image_sum(colors, raw + h) - image_sum(colors, raw - h)) / Float64(2.0 * h)
    assert_true(abs(beta_grad / numeric - 1.0) < 0.01, String("beta: ", beta_grad, " vs ", numeric))
    _ = (image^, grad_image^, splats^)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
