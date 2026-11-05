"""COLMAP binary reader vs the fixture written from the format spec
(tests/data/make_golden_io.py, colmap_tiny)."""

from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojosplat.io.colmap import read_cameras_bin, read_colmap_model, read_images_bin

comptime DIR = "tests/data/golden/io/colmap_tiny/"


def test_cameras() raises:
    var cams = read_cameras_bin(DIR + "cameras.bin")
    assert_equal(len(cams), 2)
    assert_equal(cams[0].id, 1)
    assert_equal(cams[0].model, 1)
    assert_equal(cams[0].width, 640)
    assert_equal(cams[0].height, 480)
    var k = cams[0].pinhole()
    assert_equal(k, SIMD[DType.float64, 4](500.5, 501.25, 320.0, 240.5))
    # OPENCV: 8 parameters parsed (and skipped correctly), but not pinhole.
    assert_equal(cams[1].id, 7)
    assert_equal(len(cams[1].params), 8)
    assert_equal(cams[1].params[7], 0.4)
    with assert_raises():
        _ = cams[1].pinhole()


def test_images() raises:
    var ims = read_images_bin(DIR + "images.bin")
    assert_equal(len(ims), 3)
    assert_equal(ims[0].id, 3)
    assert_equal(ims[0].name, "c.png")
    assert_equal(ims[0].qvec, SIMD[DType.float64, 4](0.5, 0.5, 0.5, 0.5))
    assert_equal(ims[0].tvec[2], 3.0)
    assert_equal(ims[2].name, "b.png")
    assert_equal(ims[2].camera_id, 7)
    # q = (0, 1, 0, 0): 180 degrees about x -> diag(1, -1, -1), t in column 3.
    var v = ims[2].view_matrix()
    assert_equal(v, SIMD[DType.float32, 16](1, 0, 0, -1, 0, -1, 0, 0.5, 0, 0, -1, 2, 0, 0, 0, 1))


def test_points_and_model() raises:
    var m = read_colmap_model("tests/data/golden/io/colmap_tiny")
    assert_equal(m.points.count(), 4)
    assert_equal(m.points.xyz[3], Float32(1.5))
    assert_equal(m.points.xyz[11], Float32(9.5))
    assert_equal(m.points.rgb[0], 255)
    assert_equal(m.points.rgb[8], 200)
    assert_equal(m.camera(7).width, 100)


def test_truncated_file_raises() raises:
    var data: List[UInt8]
    with open(DIR + "images.bin", "r") as f:
        data = f.read_bytes()
    var path = String("tests/data/golden/io/_truncated_images.bin")
    with open(path, "w") as f:
        f.write_bytes(Span(data)[0 : len(data) - 5])
    with assert_raises():
        _ = read_images_bin(path)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
