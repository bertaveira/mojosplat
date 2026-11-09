"""run.json holds the resolved settings and the results."""

from std.testing import assert_true, TestSuite

from mojosplat.train.run_record import write_run_record
from mojosplat.train.trainer import TrainConfig


def test_record_has_settings_and_metrics() raises:
    var config = TrainConfig()
    config.iterations = 1234
    config.densify = False
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
