"""Adam on Splats vs torch.optim.Adam with one group per attribute
(tests/data/make_golden_train.py)."""

from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from golden import read_kv
from mojosplat.core.types import dptr
from mojosplat.gpu.elementwise import copy_f32
from mojosplat.io.arrays import download, read_array, upload
from mojosplat.model.splats import Attr, Splats
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
