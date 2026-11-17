"""Initialization: 3-NN distances vs numpy brute force, and 3DGUT's camera-distance scales."""

from max.gpu.host import DeviceContext
from std.math import log, sqrt
from std.testing import assert_equal, assert_true, TestSuite

from mojosplat.io.arrays import read_array
from mojosplat.model.init import knn3_mean_dist2, nearest_camera_distances, splats_from_points
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


def test_nearest_camera_distances() raises:
    var xyz: List[Float32] = [0, 0, 0, 10, 0, 0]
    var centers: List[SIMD[DType.float32, 4]] = [
        SIMD[DType.float32, 4](3, 4, 0, 0),
        SIMD[DType.float32, 4](10, 0, 2, 0),
    ]
    var d = nearest_camera_distances(xyz, centers)
    assert_equal(len(d), 2)
    assert_true(abs(d[0] - 5.0) < 1e-6, "nearest is the first camera")
    assert_true(abs(d[1] - 2.0) < 1e-6, "nearest is the second camera")


def test_splats_from_camera_distances() raises:
    var ctx = DeviceContext()
    var xyz: List[Float32] = [0, 0, 0, 1, 0, 0]
    var rgb: List[UInt8] = [255, 0, 0, 0, 255, 0]
    var distances: List[Float32] = [5.0, 0.0]
    var s = splats_from_points(ctx, xyz, rgb, initial_scale=0.01, distances=distances, random_rotations=True)
    var h = s.to_host()
    assert_true(abs(h.log_scales[0] - Float32(log(0.05))) < 1e-6, "scale = 0.01 x distance")
    assert_true(abs(h.log_scales[3] - Float32(log(1e-9))) < 1e-4, "distance clamped at 1e-7")
    for k in range(8):
        assert_true(h.rotations[k] >= 0.0 and h.rotations[k] < 1.0, "rotations in [0, 1)")
    assert_true(h.rotations[0] != 1.0 or h.rotations[1] != 0.0, "not the identity")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
