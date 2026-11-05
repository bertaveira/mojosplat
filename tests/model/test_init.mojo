"""3-NN initialization distances vs numpy brute force."""

from max.gpu.host import DeviceContext
from std.math import log, sqrt
from std.testing import assert_equal, assert_true, TestSuite

from mojosplat.io.arrays import read_array
from mojosplat.model.init import knn3_mean_dist2, splats_from_points
from mojosplat.model.splats import Attr

comptime D = "tests/data/golden/train/knn/"


def test_knn3() raises:
    var ctx = DeviceContext()
    var pts = read_array[DType.float32](D + "points.f32")
    var got = knn3_mean_dist2(ctx, pts)
    var want = read_array[DType.float32](D + "dist2.f32")
    assert_equal(len(got), len(want))
    var max_rel: Float64 = 0
    for i in range(len(want)):
        max_rel = max(max_rel, abs(Float64(got[i]) - Float64(want[i])) / max(Float64(want[i]), 1e-12))
    print("  max relative error", max_rel)
    assert_true(max_rel < 1e-4, "knn distances")


def test_splats_from_points() raises:
    var ctx = DeviceContext()
    var xyz: List[Float32] = [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]
    var rgb: List[UInt8] = [255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128]
    var s = splats_from_points(ctx, xyz, rgb, sh_degree=3)
    assert_equal(s.count, 4)
    assert_equal(s.rest_stride, 15)
    var h = s.to_host()
    # point 0: neighbours at squared distance 1, 4, 9 -> mean 14/3
    assert_true(abs(h.log_scales[0] - Float32(log(sqrt(14.0 / 3.0)))) < 1e-5, "scale")
    assert_equal(h.rotations[0], 1.0)
    assert_true(abs(h.opacity_logits[2] - Float32(log(0.1 / 0.9))) < 1e-6, "opacity")
    assert_true(abs(h.sh0[0] - Float32((1.0 - 0.5) / 0.28209479177387814)) < 1e-5, "color")
    assert_equal(h.sh_rest[7], 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
