"""Training smoke test on Mip-NeRF360 garden (skipped without the dataset):
a few hundred iterations must reduce the loss and reach a sane test PSNR."""

from max.gpu.host import DeviceContext
from std.os.path import exists
from std.testing import assert_true, TestSuite

from mojosplat.train.dataset import load_colmap_dataset
from mojosplat.train.trainer import TrainConfig, Trainer

comptime GARDEN = "data/mipnerf360/garden"


def test_garden_short_run() raises:
    if not exists(GARDEN):
        print("  SKIP: needs", GARDEN)
        return
    var ctx = DeviceContext()
    var config = TrainConfig()
    config.log_interval = 50
    var trainer = Trainer(ctx, load_colmap_dataset(ctx, GARDEN), config)
    var psnr0 = trainer.evaluate(test=True)
    var first: Float64 = -1.0
    var last: Float64 = -1.0
    for _ in range(400):
        var l = trainer.step()
        if l >= 0.0:
            if first < 0.0:
                first = l
            last = l
    var psnr1 = trainer.evaluate(test=True)
    print("  loss", first, "->", last, "| test PSNR", psnr0, "->", psnr1)
    assert_true(last < first, "loss did not decrease")
    assert_true(psnr1 > psnr0 + 3.0 and psnr1 > 19.0, "test PSNR too low")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
