"""Faster-GS style Rasterizer vs an independent float64 brute-force reference.

Golden data: tests/data/golden/splats_* (tests/data/make_golden_splats.py).
"""

from max.gpu.host import DeviceContext
from std.testing import assert_equal, assert_true, TestSuite

from golden import GoldenScene
from mojosplat.core.camera import Camera
from mojosplat.io.arrays import download
from mojosplat.model.splats import Splats
from mojosplat.render.rasterizer import Rasterizer


def load_splats(ctx: DeviceContext, s: GoldenScene) raises -> Splats:
    return Splats(
        ctx,
        s.f32("means3d.f32"),
        s.f32("log_scales.f32"),
        s.f32("quats.f32"),
        s.f32("opacity_logits.f32"),
        s.f32("sh0.f32"),
        s.f32("shN.f32"),
        s.sh_rest,
    )


def background(s: GoldenScene) -> SIMD[DType.float32, 4]:
    var bg = SIMD[DType.float32, 4](0.0)
    for i in range(3):
        bg[i] = s.background[i]
    return bg


def render[tw: Int, th: Int](name: String) raises -> Tuple[List[Float32], List[Float32]]:
    var s = GoldenScene(name)
    var ctx = DeviceContext()
    var splats = load_splats(ctx, s)
    var image = ctx.enqueue_create_buffer[DType.float32](s.width * s.height * 3)
    var r = Rasterizer[tw, th](ctx)
    r.render(splats, s.camera, background(s), image, s.sh_degree)
    ctx.synchronize()
    return (download(ctx, image, s.width * s.height * 3), s.f32("image.f32"))


def check_close(got: List[Float32], want: List[Float32], label: String) raises:
    """Compare the float32 render with the float64 reference.

    Fragments right at the 1/255 alpha cutoff (or the 1e-4 transmittance
    stop) could flip between precisions, so a few outliers are tolerated;
    in practice the max error is ~4e-6.
    """
    assert_equal(len(got), len(want))
    var max_d: Float32 = 0
    var n_off = 0
    var sum_d: Float64 = 0
    for i in range(len(got)):
        var d = abs(got[i] - want[i])
        max_d = max(max_d, d)
        sum_d += Float64(d)
        if d > 1e-4:
            n_off += 1
    var mean_d = sum_d / Float64(len(got))
    print("  ", label, ": max", max_d, "mean", mean_d, "values >1e-4:", n_off, "/", len(got))
    assert_true(mean_d < 1e-6, String(label, ": mean error too large"))
    assert_true(max_d < 5e-3, String(label, ": max error too large"))
    assert_true(n_off * 10000 < len(got), String(label, ": too many mismatches"))


def test_small_sh3() raises:
    var r = render[16, 12]("splats_small")
    check_close(r[0], r[1], "small 16x12")


def test_medium_culling_sh2() raises:
    var r = render[16, 12]("splats_medium")
    check_close(r[0], r[1], "medium 16x12")


def test_dc_only() raises:
    var r = render[16, 12]("splats_dc")
    check_close(r[0], r[1], "dc 16x12")


def test_other_tile_shapes() raises:
    """Tile shape changes scheduling only, not the image."""
    var a = render[16, 16]("splats_medium")
    check_close(a[0], a[1], "medium 16x16")
    var b = render[8, 8]("splats_medium")
    check_close(b[0], b[1], "medium 8x8")
    var c = render[32, 8]("splats_small")
    check_close(c[0], c[1], "small 32x8")


def test_deterministic() raises:
    var a = render[16, 12]("splats_medium")
    var b = render[16, 12]("splats_medium")
    for i in range(len(a[0])):
        assert_equal(a[0][i], b[0][i])


def test_empty_scene_is_background() raises:
    var ctx = DeviceContext()
    var W = 50
    var H = 30
    var empty = List[Float32]()
    var splats = Splats(ctx, empty, empty, empty, empty, empty, empty, 0)
    var cam = Camera.translated(0.0, 0.0, 5.0, W, H, 40.0, 40.0)
    var image = ctx.enqueue_create_buffer[DType.float32](W * H * 3)
    var r = Rasterizer(ctx)
    r.render(splats, cam, SIMD[DType.float32, 4](0.25, 0.5, 0.75, 0.0), image)
    ctx.synchronize()
    assert_equal(r.num_visible, 0)
    assert_equal(r.num_instances, 0)
    var px = download(ctx, image, W * H * 3)
    for i in range(W * H):
        assert_equal(px[i * 3 + 0], 0.25)
        assert_equal(px[i * 3 + 1], 0.5)
        assert_equal(px[i * 3 + 2], 0.75)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
