"""One densification step (clone / split / prune) vs a torch transcription of
Faster-GS's adaptive_density_control (tests/data/make_golden_train.py)."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.testing import assert_equal, assert_true, TestSuite

from golden import read_kv
from mojosplat.core.types import dptr
from mojosplat.gpu.elementwise import copy_f32
from mojosplat.io.arrays import download, read_array, upload
from mojosplat.model.splats import Attr, Splats
from mojosplat.train.densify import Densifier

comptime D = "tests/data/golden/train/densify/"


def load(name: String) raises -> List[Float32]:
    return read_array[DType.float32](String(D, name, ".f32"))


def set_buffer(ctx: DeviceContext, dst: DeviceBuffer[DType.float32], values: List[Float32]) raises:
    var src = upload(ctx, values)
    copy_f32(ctx, ctx.stream(), dptr(dst), dptr(src), len(values))
    ctx.synchronize()


def test_matches_faster_gs() raises:
    var meta = read_kv(D + "meta.txt")
    var names: List[String] = ["means", "log_scales", "rotations", "opacity_logits", "sh0", "sh_rest"]
    var ctx = DeviceContext()
    var splats = Splats(
        ctx, load("in_means"), load("in_log_scales"), load("in_rotations"),
        load("in_opacity_logits"), load("in_sh0"), load("in_sh_rest"), Int(meta["K"]),
    )
    splats.enable_training()
    for a in range(Attr.COUNT):
        set_buffer(ctx, splats.exp_avg[a], load(String("in_m_", names[a])))
        set_buffer(ctx, splats.exp_avg_sq[a], load(String("in_v_", names[a])))
    set_buffer(ctx, splats.densification, load("info"))

    var densifier = Densifier(ctx)
    var st = densifier.densify(splats, Float32(Float64(meta["extent"])), True, load("randn"))
    assert_equal(st.cloned, Int(meta["cloned"]))
    assert_equal(st.split, Int(meta["split"]))
    assert_equal(st.pruned, Int(meta["pruned"]))
    assert_equal(splats.count, Int(meta["count"]))

    var prefixes: List[String] = ["out_", "out_m_", "out_v_"]
    for a in range(Attr.COUNT):
        for c in range(3):
            var buf = splats.params[a] if c == 0 else (splats.exp_avg[a] if c == 1 else splats.exp_avg_sq[a])
            var want = load(String(prefixes[c], names[a]))
            var got = download(ctx, buf, len(want))
            var max_err: Float64 = 0
            for i in range(len(want)):
                max_err = max(max_err, abs(Float64(got[i]) - Float64(want[i])))
            assert_true(max_err < 1e-5, String(prefixes[c], names[a], " max abs error ", max_err))


def test_reset_opacities() raises:
    var ctx = DeviceContext()
    var o: List[Float32] = [-10.0, 0.0, 5.0]
    var z3 = List[Float32](length=9, fill=0.0)
    var q: List[Float32] = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    var splats = Splats(ctx, z3, z3, q, o, z3, List[Float32](), 0)
    splats.enable_training()
    set_buffer(ctx, splats.exp_avg[Attr.OPACITY_LOGITS], [1.0, 2.0, 3.0])
    var d = Densifier(ctx)
    d.reset_opacities(splats)
    ctx.synchronize()
    var got = download(ctx, splats.params[Attr.OPACITY_LOGITS], 3)
    assert_equal(got[0], -10.0)  # already below 0.01
    assert_true(abs(got[1] - -4.59511985) < 1e-5, "capped at logit(0.01)")
    assert_true(abs(got[2] - -4.59511985) < 1e-5, "capped at logit(0.01)")
    var m = download(ctx, splats.exp_avg[Attr.OPACITY_LOGITS], 3)
    assert_equal(m[1], 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
