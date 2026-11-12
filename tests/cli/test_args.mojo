"""Command-line options generated from config structs (mojosplat/cli)."""

from std.testing import assert_equal, assert_false, assert_raises, assert_true, TestSuite

from mojosplat.cli import (
    Arg,
    config_items,
    parse_args_into,
    fmt_count,
    fmt_duration,
    fmt_fixed,
    fmt_thousands,
    parse_args_from,
    visible_width,
)
from mojosplat.methods import FasterGS
from mojosplat.train.densify import DensifyConfig
from mojosplat.train.trainer import TrainConfig


struct Inner(Defaultable, Movable):
    @__annotation(Arg("Learning rate"))
    var lr: Float32
    var densify: Bool
    var background: SIMD[DType.float32, 4]

    def __init__(out self):
        self.lr = 0.5
        self.densify = True
        self.background = SIMD[DType.float32, 4](0.0)


struct Opts(Defaultable, Movable):
    @__annotation(Arg("Scene", positional=True))
    var scene: String

    @__annotation(Arg("Model", positional=True))
    var model: String

    @__annotation(Arg("Output", short="o"))
    var out: String
    var max_steps: Int
    var scale: Float64

    @__annotation(Arg("Training"))
    var train: Inner

    def __init__(out self):
        self.scene = ""
        self.model = "model.ply"  # has a default: optional
        self.out = "out.png"
        self.max_steps = 10
        self.scale = 1.0
        self.train = Inner()


def parse(args: List[String]) raises -> Opts:
    return parse_args_from[Opts](args, "test")


def test_defaults_and_positionals() raises:
    var o = parse(["garden"])
    assert_equal(o.scene, "garden")
    assert_equal(o.model, "model.ply")
    assert_equal(o.max_steps, 10)
    assert_true(o.train.densify)
    o = parse(["garden", "m.ply"])
    assert_equal(o.model, "m.ply")


def test_flags() raises:
    var o = parse(
        [
            "--max-steps",
            "7",
            "garden",
            "-o",
            "x.png",
            "--scale=2.5",
            "--lr",
            "0.001",
            "--no-densify",
            "--background",
            "1,0.5,0",
        ]
    )
    assert_equal(o.scene, "garden")
    assert_equal(o.max_steps, 7)
    assert_equal(o.out, "x.png")
    assert_equal(o.scale, 2.5)
    assert_equal(o.train.lr, Float32(0.001))
    assert_false(o.train.densify)
    assert_equal(o.train.background[1], Float32(0.5))
    o = parse(["garden", "--densify=false", "--max-steps", "-3"])
    assert_false(o.train.densify)
    assert_equal(o.max_steps, -3)  # a negative number is a value, not a flag


def test_errors() raises:
    with assert_raises(contains="missing argument <scene>"):
        _ = parse([])
    with assert_raises(contains="did you mean '--max-steps'"):
        _ = parse(["garden", "--max-step", "3"])
    with assert_raises(contains="expected an integer"):
        _ = parse(["garden", "--max-steps", "abc"])
    with assert_raises(contains="expected a number"):
        _ = parse(["garden", "--scale", "x"])
    with assert_raises(contains="missing value for --out"):
        _ = parse(["garden", "--out"])
    with assert_raises(contains="unexpected argument"):
        _ = parse(["a", "b", "c"])


def test_method_settings() raises:
    """A method's settings parse as groups: the shared ones and its strategy's."""
    var c = parse_args_from[FasterGS.Config](
        ["--iterations", "7000", "--sh-degree", "1", "--no-densify", "--lr-means-init", "1e-4"], "train"
    )
    assert_equal(c.train.iterations, 7000)
    assert_equal(c.train.sh_degree, 1)
    assert_equal(c.train.lr_means_init, Float32(1e-4))
    assert_false(c.densify.densify)
    assert_equal(c.densify.densify_from, DensifyConfig().densify_from)


def test_formatting() raises:
    assert_equal(fmt_fixed(27.4312, 2), "27.43")
    assert_equal(fmt_fixed(-0.004, 2), "0.00")
    assert_equal(fmt_fixed(-1.25, 1), "-1.3")
    assert_equal(fmt_fixed(3.0, 0), "3")
    assert_equal(fmt_count(950), "950")
    assert_equal(fmt_count(138766), "138.8k")
    assert_equal(fmt_count(4219236), "4.22M")
    assert_equal(fmt_thousands(4219236), "4,219,236")
    assert_equal(fmt_thousands(12), "12")
    assert_equal(fmt_duration(12.34), "12.3s")
    assert_equal(fmt_duration(211.4), "3m31s")
    assert_equal(fmt_duration(4000.0), "1h06m")
    assert_equal(visible_width("\x1b[1mab\x1b[0m━c"), 4)


def test_presets_start_from_a_base_and_flags_override_it() raises:
    var base = Opts()
    base.scene = "preset"
    base.max_steps = 123
    var args: List[String] = ["garden", "model.ply", "--out", "x"]
    var o = parse_args_into[Opts](base^, args, "prog")
    assert_equal(o.max_steps, 123)
    assert_equal(o.scene, "garden")
    assert_equal(o.out, "x")


def test_config_items_lists_resolved_values() raises:
    var c = TrainConfig()
    c.iterations = 7
    var seen = False
    for item in config_items[TrainConfig](c):
        if item[0] == "iterations":
            seen = True
            assert_equal(item[1], "INT")
            assert_equal(item[2], "7")
    assert_true(seen, "iterations should be listed")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
