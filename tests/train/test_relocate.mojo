"""Relocation densification, regularizer gradients and position noise."""

from max.gpu.host import DeviceContext
from std.math import exp, log
from std.random import seed
from std.testing import assert_equal, assert_true, TestSuite

from mojosplat.core.types import dptr
from mojosplat.gpu.elementwise import fill_f32
from mojosplat.io.arrays import download
from mojosplat.model.splats import Attr, Splats
from mojosplat.train.relocate import add_gaussians, add_position_noise, add_regularizer_gradients, relocate


def logit(p: Float64) -> Float32:
    return Float32(log(p / (1.0 - p)))


def sigmoid(l: Float32) -> Float64:
    return 1.0 / (1.0 + exp(-Float64(l)))


def make(ctx: DeviceContext, opacities: List[Float64]) raises -> Splats:
    var n = len(opacities)
    var m = List[Float32]()
    var s = List[Float32]()
    var q = List[Float32]()
    var o = List[Float32]()
    var c = List[Float32]()
    for i in range(n):
        for k in range(3):
            m.append(Float32(i * 10 + k))
            s.append(-1.0)
            c.append(Float32(i))
        q.append(1.0)
        q.append(0.0)
        q.append(0.0)
        q.append(0.0)
        o.append(logit(opacities[i]))
    var splats = Splats(ctx, m, s, q, o, c, List[Float32](), 0)
    var beta = List[Float32]()
    for i in range(n):
        beta.append(Float32(i) * 0.1)
    splats.add_column("beta", 1, values=beta)
    splats.enable_training()
    return splats^


def test_relocate_moves_dead_rows_onto_live_ones() raises:
    seed(3)
    var ctx = DeviceContext()
    var splats = make(ctx, [0.8, 0.001, 0.6, 0.002])
    fill_f32(ctx, dptr(splats.exp_avg[Attr.MEANS]), 2.0, 4 * 3)
    assert_equal(relocate(splats), 2)
    var means = download(ctx, splats.params[Attr.MEANS], 12)
    var opacity = download(ctx, splats.params[Attr.OPACITY_LOGITS], 4)
    var beta = download(ctx, splats.params[Attr.COUNT], 4)
    var moments = download(ctx, splats.exp_avg[Attr.MEANS], 12)
    for dead in [1, 3]:
        var src = Int(means[dead * 3] / 10.0)
        assert_true(src == 0 or src == 2, "copied from a live row")
        assert_equal(means[dead * 3 + 2], means[src * 3 + 2])
        assert_equal(beta[dead], beta[src])  # every column is copied
        assert_equal(opacity[dead], opacity[src])  # copy and source share the opacity
        assert_true(sigmoid(opacity[dead]) < [0.8, 0.0, 0.6][src] + 1e-6, "opacity is split")
        assert_equal(moments[src * 3], 0.0)  # sources restart their optimizer state


def test_add_grows_by_five_percent_up_to_the_cap() raises:
    seed(4)
    var ctx = DeviceContext()
    var opacities = List[Float64]()
    for _ in range(40):
        opacities.append(0.5)
    var splats = make(ctx, opacities)
    assert_equal(add_gaussians(splats, 1000), 2)
    assert_equal(splats.count, 42)
    assert_equal(add_gaussians(splats, 43), 1)
    assert_equal(splats.count, 43)
    assert_equal(add_gaussians(splats, 43), 0)


def test_regularizer_gradients() raises:
    var ctx = DeviceContext()
    var splats = make(ctx, [0.5, 0.25])
    fill_f32(ctx, dptr(splats.grads[Attr.OPACITY_LOGITS]), 0.0, 2)
    fill_f32(ctx, dptr(splats.grads[Attr.LOG_SCALES]), 1.0, 6)
    add_regularizer_gradients(splats, 0.02, 0.06)
    ctx.synchronize()
    var op = download(ctx, splats.grads[Attr.OPACITY_LOGITS], 2)
    var sc = download(ctx, splats.grads[Attr.LOG_SCALES], 6)
    assert_true(abs(Float64(op[0]) - 0.02 / 2.0 * 0.25) < 1e-6, "d mean(opacity) / d logit")
    assert_true(abs(Float64(sc[0]) - (1.0 + 0.06 / 6.0 * exp(-1.0))) < 1e-6, "adds to the existing gradient")


def test_noise_moves_only_nearly_transparent_gaussians() raises:
    var ctx = DeviceContext()
    var splats = make(ctx, [0.9, 0.0001])
    add_position_noise(splats, 10.0)
    var m = download(ctx, splats.params[Attr.MEANS], 6)
    assert_true(abs(m[0] - 0.0) < 1e-5, "(1 - 0.9)^100 is effectively zero")
    assert_true(abs(m[3] - 10.0) > 1e-3 or abs(m[4] - 11.0) > 1e-3, "the transparent one moves")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
