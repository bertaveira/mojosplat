"""Rasterizer forward(train) + backward vs float64 autograd of the
brute-force reference (tests/data/make_golden_splats.py).

L = sum(image * grad_image) for a fixed random grad_image; every raw
parameter's gradient is compared by relative L2 error.
"""

from max.gpu.host import DeviceContext
from std.math import sqrt
from std.testing import assert_true, TestSuite

from golden import GoldenScene
from mojosplat.core.types import dptr
from mojosplat.io.arrays import download, upload
from mojosplat.core.splats import SplatParams
from mojosplat.model.splats import SplatBuffers
from mojosplat.render.rasterizer import Rasterizer


def rel_err(got: List[Float32], want: List[Float32]) -> Tuple[Float64, Float64]:
    var num: Float64 = 0
    var den: Float64 = 0
    var max_abs: Float64 = 0
    for i in range(len(want)):
        var d = Float64(got[i]) - Float64(want[i])
        num += d * d
        den += Float64(want[i]) * Float64(want[i])
        max_abs = max(max_abs, abs(d))
    return (sqrt(num / max(den, 1e-30)), max_abs)


def check_scene[tw: Int, th: Int](name: String, tol: Float64) raises:
    var s = GoldenScene(name)
    var ctx = DeviceContext()
    var splats = SplatBuffers(
        ctx, s.f32("means3d.f32"), s.f32("log_scales.f32"), s.f32("quats.f32"),
        s.f32("opacity_logits.f32"), s.f32("sh0.f32"), s.f32("shN.f32"), s.sh_rest,
    )
    var n = s.n
    var k = s.sh_rest
    var npix = s.width * s.height * 3
    var image = ctx.enqueue_create_buffer[DType.float32](npix)
    var grad_image = upload(ctx, s.f32("grad_image.f32"))
    var g_means = ctx.enqueue_create_buffer[DType.float32](n * 3)
    var g_scales = ctx.enqueue_create_buffer[DType.float32](n * 3)
    var g_quats = ctx.enqueue_create_buffer[DType.float32](n * 4)
    var g_opac = ctx.enqueue_create_buffer[DType.float32](n)
    var g_sh0 = ctx.enqueue_create_buffer[DType.float32](n * 3)
    var g_shN = ctx.enqueue_create_buffer[DType.float32](max(n * k * 3, 3))
    var grads = SplatParams(
        dptr(g_means), dptr(g_scales), dptr(g_quats), dptr(g_opac), dptr(g_sh0), dptr(g_shN), n, k
    )
    var bg = SIMD[DType.float32, 4](0.0)
    for i in range(3):
        bg[i] = s.background[i]

    var r = Rasterizer[tw, th](ctx)
    var st = r.forward(splats.view(), s.camera, bg, dptr(image), s.sh_degree)
    r.backward(splats.view(), st, dptr(image), dptr(grad_image), grads)
    ctx.synchronize()
    _ = grad_image^  # keep alive past the asynchronous backward
    _ = splats^

    # The training forward renders the same image as inference.
    var img = download(ctx, image, npix)
    var e_img = rel_err(img, s.f32("image.f32"))
    print("  ", name, tw, "x", th, ": image max", e_img[1])
    assert_true(e_img[1] < 5e-3, "training forward image differs")

    var names: List[String] = ["means3d", "log_scales", "quats", "opacity_logits", "sh0", "shN"]
    var sizes: List[Int] = [n * 3, n * 3, n * 4, n, n * 3, n * k * 3]
    var bufs = [g_means, g_scales, g_quats, g_opac, g_sh0, g_shN]
    var failed = List[String]()
    for i in range(len(names)):
        if sizes[i] == 0:
            continue
        var got = download(ctx, bufs[i], sizes[i])
        var e = rel_err(got, s.f32(String("grad_", names[i], ".f32")))
        print("     grad", names[i], ": rel", e[0], "max abs", e[1])
        if e[0] >= tol:
            failed.append(names[i])
    assert_true(len(failed) == 0, String("gradient mismatch: ", failed[0] if len(failed) > 0 else ""))


def test_small_sh3() raises:
    check_scene[16, 12]("splats_small", 1e-4)


def test_medium_culling_sh2() raises:
    check_scene[16, 12]("splats_medium", 1e-4)


def test_dc_only() raises:
    check_scene[16, 12]("splats_dc", 1e-4)


def test_other_tile_shapes() raises:
    check_scene[16, 16]("splats_medium", 1e-4)
    check_scene[8, 8]("splats_small", 1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
