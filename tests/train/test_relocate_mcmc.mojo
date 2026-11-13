"""3DGS-MCMC relocation: Equation 9, the shrink of copies, its noise and its schedule."""

from max.gpu.host import DeviceContext
from std.math import exp, log, pow
from std.random import seed
from std.testing import assert_almost_equal, assert_equal, assert_false, assert_true, TestSuite

from mojosplat.io.arrays import download
from mojosplat.model.splats import Attr, Splats
from mojosplat.train.relocate import (
    MCMCRelocation,
    RelocateConfig,
    _binomials,
    add_position_noise,
    mcmc_scale_ratio,
    relocate,
)


def _check(old: Float64, n: Int, expected: Float64) raises:
    var new = 1.0 - pow(1.0 - old, 1.0 / Float64(n))
    assert_almost_equal(mcmc_scale_ratio(old, new, n, _binomials()), expected, atol=1e-9)


def test_one_copy_keeps_the_scale() raises:
    _check(0.4, 1, 1.0)


def test_matches_equation_9() raises:
    # From the paper's formula in float64 Python.
    _check(0.4, 2, 0.964132054106)
    _check(0.9, 5, 0.797124446587)
    _check(0.05, 3, 0.995015765408)
    _check(0.7, 50, 0.852650361781)


def _logit(p: Float64) -> Float32:
    return Float32(log(p / (1.0 - p)))


def _make(ctx: DeviceContext, opacities: List[Float64]) raises -> Splats:
    var m = List[Float32]()
    var s = List[Float32]()
    var q = List[Float32]()
    var o = List[Float32]()
    var c = List[Float32]()
    for i in range(len(opacities)):
        for k in range(3):
            m.append(Float32(i * 10 + k))
            s.append(-1.0)
            c.append(0.5)
        q.append(1.0)
        q.append(0.0)
        q.append(0.0)
        q.append(0.0)
        o.append(_logit(opacities[i]))
    var splats = Splats(ctx, m, s, q, o, c, List[Float32](), 0)
    splats.enable_training()
    return splats^


def test_relocated_copies_shrink_with_their_source() raises:
    seed(5)
    var ctx = DeviceContext()
    var splats = _make(ctx, [0.8, 0.001])  # row 1 is dead and can only copy row 0
    assert_equal(relocate(splats, shrink_scales=True), 1)
    var s = download(ctx, splats.params[Attr.LOG_SCALES], 6)
    var expected = -1.0 + log(mcmc_scale_ratio(0.8, 1.0 - pow(0.2, 0.5), 2, _binomials()))
    for k in range(6):
        assert_almost_equal(Float64(s[k]), expected, atol=1e-5)


def test_sigmoid_noise_moves_only_nearly_transparent_gaussians() raises:
    var ctx = DeviceContext()
    var splats = _make(ctx, [0.9, 0.0001])
    add_position_noise(splats, 10.0, sigmoid=True)
    var m = download(ctx, splats.params[Attr.MEANS], 6)
    assert_true(abs(m[0] - 0.0) < 1e-5, "sigmoid(100 (0.1 - 0.995)) is effectively zero")
    assert_true(abs(m[3] - 10.0) > 1e-3 or abs(m[4] - 11.0) > 1e-3, "the transparent one moves")


def test_schedule() raises:
    """Optimizer steps except at relocations and on the last iteration (t = it + 1)."""
    var ctx = DeviceContext()
    var splats = _make(ctx, [0.5, 0.5])
    var config = RelocateConfig()
    config.relocate_from = 500
    config.relocate_until = 1000
    config.relocate_interval = 100
    var strategy = MCMCRelocation(ctx, config, iterations=2000)
    assert_true(strategy.after_backward(498, splats, 1e-4), "before the window")
    assert_false(strategy.after_backward(599, splats, 1e-4), "a relocation (t = 600)")
    assert_true(strategy.after_backward(600, splats, 1e-4), "between relocations")
    assert_true(strategy.after_backward(999, splats, 1e-4), "t = 1000 is past the window")
    assert_true(strategy.after_backward(1998, splats, 1e-4))
    assert_false(strategy.after_backward(1999, splats, 1e-4), "the last iteration")


def test_no_noise_after_the_last_step() raises:
    var ctx = DeviceContext()
    var splats = _make(ctx, [0.0001])
    var strategy = MCMCRelocation(ctx, RelocateConfig(), iterations=10)
    strategy.after_step(9, splats, 1.0)
    var m = download(ctx, splats.params[Attr.MEANS], 3)
    assert_equal(m[0], 0.0)
    strategy.after_step(8, splats, 1.0)
    m = download(ctx, splats.params[Attr.MEANS], 3)
    assert_true(m[0] != 0.0 or m[1] != 1.0 or m[2] != 2.0, "moved before the last iteration")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
