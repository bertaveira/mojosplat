"""3DGUT's densification schedule (`GutDensification`, NVIDIA's GSStrategy):
what happens after the optimizer step of each iteration."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import exp, log
from std.random import seed
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojosplat.core.types import dptr
from mojosplat.gpu.elementwise import copy_f32
from mojosplat.io.arrays import download, upload
from mojosplat.model.splats import Attr, Splats
from mojosplat.train.densify import GutDensification, GutDensifyConfig

comptime EXTENT: Float32 = 10.0  # clone below 0.01 * 10 = 0.1


def logit(p: Float64) -> Float32:
    return Float32(log(p / (1.0 - p)))


def set_buffer(ctx: DeviceContext, dst: DeviceBuffer[DType.float32], values: List[Float32]) raises:
    var src = upload(ctx, values)
    copy_f32(ctx, dptr(dst), dptr(src), len(values))
    ctx.synchronize()
    _ = src^


def make(ctx: DeviceContext, opacities: List[Float64], info: List[Float32]) raises -> Splats:
    """Small Gaussians (scale e^-5 < 0.1, so selected ones are cloned) with the given
    opacities and densification statistics [counts..., sums...]."""
    var n = len(opacities)
    var m = List[Float32]()
    var s = List[Float32]()
    var q = List[Float32]()
    var o = List[Float32]()
    for i in range(n):
        for k in range(3):
            m.append(Float32(i * 10 + k))
            s.append(-5.0)
        q.append(1.0)
        q.append(0.0)
        q.append(0.0)
        q.append(0.0)
        o.append(logit(opacities[i]))
    var splats = Splats(ctx, m, s, q, o, List[Float32](length=n * 3, fill=0.0), List[Float32](), 0)
    splats.enable_training()
    set_buffer(ctx, splats.densification, info)
    return splats^


def test_statistics_from_the_second_iteration() raises:
    var strategy = GutDensification(DeviceContext(), GutDensifyConfig(), 30_000)
    assert_false(strategy.collects_stats(0))
    assert_true(strategy.collects_stats(1))
    assert_true(strategy.collects_stats(14_999))
    assert_false(strategy.collects_stats(15_000))


def test_nothing_before_or_at_densify_from() raises:
    var ctx = DeviceContext()
    var splats = make(ctx, [0.5, 0.003], [1, 1, 1.0, 1.0])
    var strategy = GutDensification(ctx, GutDensifyConfig(), 30_000)
    for it in [0, 300, 500]:
        strategy.after_step(it, splats, EXTENT, 1e-4)
    assert_equal(splats.count, 2)


def test_densify_then_prune() raises:
    """At 600: clone the Gaussian with a large gradient, reset the statistics,
    then prune the transparent one."""
    seed(1)
    var ctx = DeviceContext()
    var splats = make(ctx, [0.5, 0.5, 0.003], [1, 1, 1, 0.001, 0.0, 0.001])
    var strategy = GutDensification(ctx, GutDensifyConfig(), 30_000)
    strategy.after_step(600, splats, EXTENT, 1e-4)
    assert_equal(strategy.last.cloned, 2)  # rows 0 and 2
    assert_equal(splats.count, 3)  # rows 0, 1 and the clone of 0
    var means = download(ctx, splats.params[Attr.MEANS], 9)
    assert_equal(means[0], 0.0)
    assert_equal(means[3], 10.0)
    assert_equal(means[6], 0.0)
    var info = download(ctx, splats.densification, 6)
    for v in info:
        assert_equal(v, 0.0)


def test_prune_keeps_statistics_with_their_rows() raises:
    """At 700 (not a densification step) only the prune runs."""
    var ctx = DeviceContext()
    var splats = make(ctx, [0.5, 0.003, 0.5], [1, 2, 3, 0.1, 0.2, 0.3])
    var strategy = GutDensification(ctx, GutDensifyConfig(), 30_000)
    strategy.after_step(700, splats, EXTENT, 1e-4)
    assert_equal(strategy.last.cloned, 0)
    assert_equal(splats.count, 2)
    var info = download(ctx, splats.densification, 4)
    assert_equal(info[0], 1.0)
    assert_equal(info[1], 3.0)
    assert_true(abs(info[2] - 0.1) < 1e-7 and abs(info[3] - 0.3) < 1e-7, "sums follow their rows")


def test_opacity_reset() raises:
    """At 3000: densify (nothing selected), prune, then cap opacities at 0.01."""
    var ctx = DeviceContext()
    var splats = make(ctx, [0.5, 0.003, 0.9], [0, 0, 0, 0, 0, 0])
    var strategy = GutDensification(ctx, GutDensifyConfig(), 30_000)
    strategy.after_step(3000, splats, EXTENT, 1e-4)
    assert_equal(splats.count, 2)
    var o = download(ctx, splats.params[Attr.OPACITY_LOGITS], 2)
    for v in o:
        assert_true(abs(v - logit(0.01)) < 1e-5, "capped at 0.01")


def test_nothing_from_densify_until() raises:
    var ctx = DeviceContext()
    var splats = make(ctx, [0.5, 0.003], [1, 1, 1.0, 1.0])
    var strategy = GutDensification(ctx, GutDensifyConfig(), 30_000)
    strategy.after_step(15_000, splats, EXTENT, 1e-4)
    strategy.after_step(18_000, splats, EXTENT, 1e-4)
    assert_equal(splats.count, 2)
    assert_equal(strategy.finish(splats), 0)  # and nothing after training
    assert_equal(splats.count, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
