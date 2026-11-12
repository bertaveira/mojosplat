"""run.json holds the resolved settings and the results."""

from std.os.path import exists
from std.testing import assert_equal, assert_true, TestSuite

from mojosplat.methods import FasterGS
from mojosplat.train.run_record import git_state, mojosplat_root, write_run_record


def test_record_has_settings_and_metrics() raises:
    var config = FasterGS.Config()
    config.train.iterations = 1234
    config.densify.densify = False
    var metrics = List[Tuple[String, Float64]]()
    metrics.append(("test_psnr", 27.5))
    var path = String("tests/data/golden/io/_run.json")
    write_run_record(path, "faster-gs", 'data/"garden"', "GPU", config, metrics)
    var text: String
    with open(path, "r") as f:
        text = f.read()
    assert_true('"method": "faster-gs"' in text, "method")
    assert_true('"scene": "data/\\"garden\\""' in text, "escapes quotes")
    assert_true('"iterations": 1234' in text, "resolved settings")
    assert_true('"densify": false' in text, "booleans")
    assert_true('"images": "images_4"' in text, "strings")
    assert_true('"test_psnr": 27.5' in text, "metrics")
    assert_true('"commit": "' in text, "commit")
    assert_true('"dirty": ' in text, "uncommitted changes")


def test_commit_is_mojosplats_own() raises:
    """The commit comes from the checkout the code was built from, whatever the working directory."""
    var root = mojosplat_root()
    assert_true(exists(root + "/mojosplat/train/run_record.mojo"), root)
    var state = git_state(root)
    if state.commit != "unknown":
        assert_equal(state.commit.byte_length(), 40)
    assert_equal(git_state("/nonexistent").commit, "unknown")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
