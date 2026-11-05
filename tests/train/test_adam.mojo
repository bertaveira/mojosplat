"""Adam on Splats vs torch.optim.Adam with one group per attribute
(tests/data/make_golden_train.py)."""

from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from golden import GoldenScene, read_kv
from mojosplat.core.types import dptr
from mojosplat.gpu.elementwise import copy_f32
from mojosplat.io.arrays import download, read_array, upload
from mojosplat.model.splats import Attr, Splats
from mojosplat.render.rasterizer import Rasterizer
from mojosplat.train.adam import Adam

comptime D = "tests/data/golden/train/adam/"


def test_matches_torch() raises:
    var meta = read_kv(D + "meta.txt")
    var n = Int(meta["N"])
    var steps = Int(meta["steps"])
    var names: List[String] = ["means", "log_scales", "rotations", "opacity_logits", "sh0", "sh_rest"]
    var ctx = DeviceContext()
    var splats = Splats(
        ctx, read_array[DType.float32](D + "init_means.f32"),
        read_array[DType.float32](D + "init_log_scales.f32"),
        read_array[DType.float32](D + "init_rotations.f32"),
        read_array[DType.float32](D + "init_opacity_logits.f32"),
        read_array[DType.float32](D + "init_sh0.f32"),
        read_array[DType.float32](D + "init_sh_rest.f32"), Int(meta["sh_rest"]),
        capacity=n + 5,  # spare capacity must not matter
    )
    splats.enable_training()
    var lr = SIMD[DType.float32, 8](0.0)
    for a in range(Attr.COUNT):
        lr[a] = Float32(Float64(meta[String("lr_", names[a])]))
    var adam = Adam(lr)
    var grads = List[List[Float32]]()
    for a in range(Attr.COUNT):
        grads.append(read_array[DType.float32](String(D, "grads_", names[a], ".f32")))
    for s in range(steps):
        for a in range(Attr.COUNT):
            var w = splats.width(a) * n
            var step_grad = List[Float32](capacity=w)
            for i in range(w):
                step_grad.append(grads[a][s * w + i])
            var g = upload(ctx, step_grad)
            copy_f32(ctx, ctx.stream(), dptr(splats.grads[a]), dptr(g), w)
            ctx.synchronize()
        adam.step(splats)
    ctx.synchronize()
    for a in range(Attr.COUNT):
        var w = splats.width(a) * n
        var checks: List[String] = ["param_", "exp_avg_", "exp_avg_sq_"]
        for c in range(3):
            var buf = splats.params[a] if c == 0 else (splats.exp_avg[a] if c == 1 else splats.exp_avg_sq[a])
            var got = download(ctx, buf, w)
            var want = read_array[DType.float32](String(D, checks[c], names[a], ".f32"))
            var max_rel: Float64 = 0
            for i in range(w):
                var d = abs(Float64(got[i]) - Float64(want[i]))
                max_rel = max(max_rel, d / max(abs(Float64(want[i])), 1e-6))
            assert_true(max_rel < 1e-4, String(checks[c], names[a], " max rel err ", max_rel))




def _step_once(fused: Bool) raises -> List[List[Float32]]:
    """One forward/backward/Adam step on the SH-degree-3 golden scene."""
    var s = GoldenScene("splats_small")
    var ctx = DeviceContext()
    var splats = Splats(
        ctx, s.f32("means3d.f32"), s.f32("log_scales.f32"), s.f32("quats.f32"),
        s.f32("opacity_logits.f32"), s.f32("sh0.f32"), s.f32("shN.f32"), s.sh_rest,
    )
    splats.enable_training()
    var npix = s.width * s.height * 3
    var image = ctx.enqueue_create_buffer[DType.float32](npix)
    var grad_image = upload(ctx, s.f32("grad_image.f32"))
    var bg = SIMD[DType.float32, 4](0.0)
    var r = Rasterizer(ctx)
    var lr = SIMD[DType.float32, 8](0.01, 0.02, 0.003, 0.05, 0.004, 0.001, 0, 0)
    var adam = Adam(lr)
    var st = r.forward(splats.view(), s.camera, bg, dptr(image), s.sh_degree)
    r.backward(splats.view(), st, dptr(image), dptr(grad_image), splats.grad_view(), write_sh_rest=not fused)
    if fused:
        adam.step(splats, r.sh_rest_grad(st))
    else:
        adam.step(splats)
    ctx.synchronize()
    var out = List[List[Float32]]()
    for a in range(Attr.COUNT):
        var w = splats.count * splats.width(a)
        out.append(download(ctx, splats.params[a], w))
        out.append(download(ctx, splats.exp_avg[a], w))
        out.append(download(ctx, splats.exp_avg_sq[a], w))
    _ = grad_image^
    return out^


def test_fused_sh_rest_matches_unfused() raises:
    var a = _step_once(False)
    var b = _step_once(True)
    for i in range(len(a)):
        var max_d: Float64 = 0
        var scale: Float64 = 1e-12
        for j in range(len(a[i])):
            max_d = max(max_d, abs(Float64(a[i][j]) - Float64(b[i][j])))
            scale = max(scale, abs(Float64(a[i][j])))
        assert_true(max_d <= 1e-5 * scale, String("array ", i, " differs: ", max_d, " (scale ", scale, ")"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
