"""Per-scene settings of the method presets, and the options each method has."""

from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojosplat.cli import config_items
from mojosplat.methods import BetaSplattingMethod, FasterGS, MCMC
from mojosplat.train.densify import DensifyConfig
from mojosplat.train.trainer import TrainConfig


def _has_option[T: AnyType](ref config: T, name: String) raises -> Bool:
    for item in config_items[T](config):
        if item[0] == name:
            return True
    return False


def test_faster_gs_is_the_defaults() raises:
    var c = FasterGS.config("garden")
    assert_equal(c.train.iterations, TrainConfig().iterations)
    assert_equal(c.train.images, TrainConfig().images)
    assert_true(c.densify.densify)
    assert_equal(c.densify.densify_until, DensifyConfig().densify_until)


def test_beta_follows_the_official_benchmark() raises:
    var garden = BetaSplattingMethod.config("garden")
    assert_equal(garden.relocate.cap_max, 5_000_000)
    assert_equal(garden.train.images, "images_4")
    assert_equal(garden.train.sh_degree, 0)
    var room = BetaSplattingMethod.config("room")
    assert_equal(room.relocate.cap_max, 1_500_000)
    assert_equal(room.train.images, "images_2")
    assert_equal(BetaSplattingMethod.config("truck").train.images, "images")


def test_mcmc_follows_the_official_configs() raises:
    var garden = MCMC.config("garden")
    assert_equal(garden.relocate.cap_max, 5_200_000)
    assert_equal(garden.train.images, "images_4")
    assert_equal(garden.train.sh_degree, 3)
    assert_equal(garden.relocate.noise_lr, 5e5)
    assert_equal(garden.relocate.opacity_reg, 0.01)
    assert_true(garden.init.random_init)
    assert_equal(garden.init.initial_scale, 0.1)
    var bonsai = MCMC.config("bonsai")
    assert_equal(bonsai.relocate.cap_max, 1_300_000)
    assert_equal(bonsai.train.images, "images_2")
    assert_equal(MCMC.config("playroom").relocate.opacity_reg, 0.001)


def test_only_mcmc_starts_from_random_points() raises:
    assert_true(MCMC.config("garden").initialization().random_init)
    assert_false(FasterGS.config("garden").initialization().random_init)
    assert_equal(BetaSplattingMethod.config("garden").initialization().initial_scale, 1.0)


def test_each_method_has_only_its_own_options() raises:
    var fgs = FasterGS.config("garden")
    var beta = BetaSplattingMethod.config("garden")
    var mcmc = MCMC.config("garden")
    assert_true(_has_option(fgs, "grad-threshold"))
    assert_false(_has_option(fgs, "cap-max"))
    assert_true(_has_option(beta, "cap-max"))
    assert_false(_has_option(beta, "grad-threshold"))
    assert_true(_has_option(mcmc, "cap-max") and _has_option(mcmc, "random-init"))
    assert_false(_has_option(fgs, "random-init") or _has_option(beta, "random-init"))
    assert_false(_has_option(mcmc, "grad-threshold"))
    assert_true(_has_option(fgs, "iterations") and _has_option(beta, "iterations"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
