"""The schedules the Trainer applies each iteration: learning rates
(`scheduled_learning_rates`, with Spherical Voronoi's final freeze), and the
SH degree and means learning-rate steps in 3DGS's and NVIDIA's order."""

from std.testing import assert_equal, assert_true, TestSuite

from mojosplat.model.splats import Attr
from mojosplat.render.blocks.appearance.spherical_voronoi import SphericalVoronoi
from mojosplat.train.trainer import means_lr_step, scheduled_learning_rates, sh_degree_at

comptime SV = SphericalVoronoi[8]
comptime SITES = Attr.COUNT
comptime COLORS = Attr.COUNT + 1
comptime LOG_TAU = Attr.COUNT + 2
comptime BETA = Attr.COUNT + 3  # the response's column, after the appearance's
comptime N_COLUMNS = Attr.COUNT + 4


def _rates() -> SIMD[DType.float32, 16]:
    var lr = SIMD[DType.float32, 16](0.0)
    for a in range(N_COLUMNS):
        lr[a] = Float32(a + 1) * 0.001
    return lr


def _at(iteration: Int, frozen_last: Int) -> SIMD[DType.float32, 16]:
    return scheduled_learning_rates[SV](_rates(), _rates(), N_COLUMNS, 3, iteration, 30_000, frozen_last)


def test_freeze_starts_after_the_last_unfrozen_iteration() raises:
    var before = _at(27_000, 3_000)  # 1-based, as the official `iteration == T - 3000 + 1`
    for a in range(N_COLUMNS):
        assert_true(before[a] > 0.0, "column " + String(a) + " should still train at 27,000")
    var after = _at(27_001, 3_000)
    for a in range(Attr.COUNT):
        assert_equal(after[a], 0.0, "column " + String(a) + " should be frozen")
    assert_equal(after[SITES], 0.0)
    assert_equal(after[BETA], 0.0)
    assert_equal(after[COLORS], _rates()[COLORS])
    assert_equal(after[LOG_TAU], before[LOG_TAU])  # keeps its rate from the last unfrozen iteration
    var last = _at(30_000, 3_000)
    assert_equal(last[LOG_TAU], before[LOG_TAU])
    assert_equal(last[Attr.MEANS], 0.0)


def test_nothing_frozen_by_default() raises:
    var last = _at(30_000, 0)
    for a in range(N_COLUMNS):
        assert_true(last[a] > 0.0, "column " + String(a) + " should train with frozen_last = 0")
    for a in range(Attr.COUNT):
        assert_equal(last[a], _rates()[a])


def test_sh_degree_in_both_orders() raises:
    assert_equal(sh_degree_at(0, 3, 1000, False), 0)
    assert_equal(sh_degree_at(999, 3, 1000, False), 0)
    assert_equal(sh_degree_at(1000, 3, 1000, False), 1)
    assert_equal(sh_degree_at(1000, 3, 1000, True), 0)  # raised after step 1000
    assert_equal(sh_degree_at(1001, 3, 1000, True), 1)
    assert_equal(sh_degree_at(3001, 3, 1000, True), 3)
    assert_equal(sh_degree_at(29_999, 3, 1000, True), 3)


def test_means_lr_step_in_both_orders() raises:
    assert_equal(means_lr_step(0, False), 1)
    assert_equal(means_lr_step(41, False), 42)
    assert_equal(means_lr_step(0, True), 0)  # the initial rate
    assert_equal(means_lr_step(1, True), 0)  # set after step 0
    assert_equal(means_lr_step(42, True), 41)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
