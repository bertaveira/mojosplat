"""End-to-end checks of binning and rendering against the golden data."""

from max.gpu.host import DeviceContext
from std.math import exp
from std.testing import assert_equal, assert_true, TestSuite

from mojosplat.common import dptr
from mojosplat.io import upload, download
from mojosplat.render import GaussianBuffers, Renderer
from golden import GoldenScene


def check_scene(name: String) raises:
    var g = GoldenScene(name)
    var ctx = DeviceContext()
    var log_scales = g.f32("log_scales.f32")
    var scales = List[Float32](capacity=len(log_scales))
    for v in log_scales:
        scales.append(exp(v))
    var gs = GaussianBuffers(
        ctx, g.f32("means3d.f32"), scales, g.f32("quats.f32"),
        g.f32("opacities.f32"), g.f32("colors.f32"),
    )
    var bg = SIMD[DType.float32, 4](g.background[0], g.background[1], g.background[2], 0.0)
    var npix = g.width * g.height * 3
    var d_image = ctx.enqueue_create_buffer[DType.float32](npix)

    var renderer = Renderer(ctx)
    renderer.render(gs, g.camera, bg, d_image)
    ctx.synchronize()

    # --- Binning: identical intersection count and per-tile ordering ---
    assert_equal(renderer.num_intersections, g.m, "number of intersections")
    var ids_buf = renderer.ids_b if renderer.sorted_in_b else renderer.ids_a
    var ids = download(ctx, ids_buf, g.m)
    var ref_ids = g.i32("flatten_ids.i32")
    var id_mismatch = 0
    for i in range(g.m):
        if ids[i] != ref_ids[i]:
            id_mismatch += 1
    var offsets = download(ctx, renderer.tile_offsets, g.camera.tiles_x(16) * g.camera.tiles_y(16) + 1)
    var ref_ranges = g.i32("tile_ranges.i32")
    for t in range(len(offsets) - 1):
        assert_equal(offsets[t], ref_ranges[2 * t], String("tile start ", t))
        assert_equal(offsets[t + 1], ref_ranges[2 * t + 1], String("tile end ", t))

    # --- Image ---
    var image = download(ctx, d_image, npix)
    var ref_image = g.f32("image.f32")
    var ref_gsplat = g.f32("image_gsplat.f32")
    var max_err: Float32 = 0.0
    var max_err_gs: Float32 = 0.0
    for i in range(npix):
        max_err = max(max_err, abs(image[i] - ref_image[i]))
        max_err_gs = max(max_err_gs, abs(image[i] - ref_gsplat[i]))
    print(
        name, ": M =", renderer.num_intersections, "id mismatches", id_mismatch,
        "| max |img - golden| =", max_err, " max |img - gsplat| =", max_err_gs,
    )
    assert_equal(id_mismatch, 0, "sorted gaussian ids differ")
    assert_true(max_err < 1e-3, String("image differs from golden: ", max_err))
    assert_true(max_err_gs < 2e-3, String("image differs from gsplat: ", max_err_gs))


def test_render_small() raises:
    check_scene("small")


def test_render_medium() raises:
    check_scene("medium")


def test_render_empty_scene() raises:
    # Everything behind the camera: no intersections, image == background.
    var ctx = DeviceContext()
    var n = 10
    var means = List[Float32](capacity=n * 3)
    var ones = List[Float32](capacity=n * 4)
    for _ in range(n):
        means.append(0.0)
        means.append(0.0)
        means.append(-5.0)
    for _ in range(n * 4):
        ones.append(0.5)
    var halves = List[Float32](capacity=n)
    for _ in range(n):
        halves.append(0.5)
    var gs = GaussianBuffers(ctx, means, ones, ones, halves, means)
    from mojosplat.camera import Camera
    var cam = Camera.translated(0.0, 0.0, 0.0, 40, 30, 50.0, 50.0)
    var d_image = ctx.enqueue_create_buffer[DType.float32](40 * 30 * 3)
    var renderer = Renderer(ctx)
    renderer.render(gs, cam, SIMD[DType.float32, 4](0.25, 0.5, 0.75, 0.0), d_image)
    ctx.synchronize()
    assert_equal(renderer.num_intersections, 0)
    var image = download(ctx, d_image, 40 * 30 * 3)
    for p in range(40 * 30):
        assert_equal(image[3 * p + 0], 0.25)
        assert_equal(image[3 * p + 1], 0.5)
        assert_equal(image[3 * p + 2], 0.75)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
