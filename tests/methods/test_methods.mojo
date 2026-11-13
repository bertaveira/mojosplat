"""Per-scene settings of the method presets, and the options each method has."""

from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojosplat.cli import config_items
from mojosplat.methods import BetaSplattingMethod, FasterGS
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


def test_each_method_has_only_its_own_options() raises:
    var fgs = FasterGS.config("garden")
    var beta = BetaSplattingMethod.config("garden")
    assert_true(_has_option(fgs, "grad-threshold"))
    assert_false(_has_option(fgs, "cap-max"))
    assert_true(_has_option(beta, "cap-max"))
    assert_false(_has_option(beta, "grad-threshold"))
    assert_true(_has_option(fgs, "iterations") and _has_option(beta, "iterations"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
