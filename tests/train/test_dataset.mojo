"""COLMAP dataset loading on Mip-NeRF360 garden vs an independent Python
summary (tests/data/golden/io/garden.txt; skipped without the dataset), and
3DGUT's median scene extent."""

from max.gpu.host import DeviceContext
from std.os.path import exists
from std.testing import assert_equal, assert_true, TestSuite

from golden import floats, read_kv
from mojosplat.io.arrays import download
from mojosplat.core.camera import Camera
from mojosplat.io.colmap import read_colmap_model
from mojosplat.train.dataset import load_colmap_dataset, median_scene_extent

comptime GARDEN = "data/mipnerf360/garden"
comptime SUMMARY = "tests/data/golden/io/garden.txt"


def close(a: Float64, b: Float64, tol: Float64) -> Bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_garden() raises:
    if not exists(GARDEN) or not exists(SUMMARY):
        print("  SKIP: needs", GARDEN, "and", SUMMARY)
        return
    var e = read_kv(SUMMARY)
    var ctx = DeviceContext()
    var ds = load_colmap_dataset(ctx, GARDEN)
    assert_equal(len(ds.train), Int(e["n_train"]))
    assert_equal(len(ds.test), Int(e["n_test"]))
    assert_equal(ds.points.count(), Int(e["n_points"]))
    assert_true(close(Float64(ds.scene_extent), Float64(e["scene_extent"]), 1e-5), "scene extent")

    # The first image (sorted by name) is the first test view.
    ref v = ds.test[0]
    assert_equal(v.name, e["first_name"])
    var c = v.camera
    assert_equal(c.width, Int(e["image_width"]))
    assert_equal(c.height, Int(e["image_height"]))
    assert_true(close(Float64(c.fx), Float64(e["fx"]), 1e-6), "fx")
    assert_true(close(Float64(c.fy), Float64(e["fy"]), 1e-6), "fy")
    assert_true(close(Float64(c.cx), Float64(e["cx"]), 1e-6), "cx")
    assert_true(close(Float64(c.cy), Float64(e["cy"]), 1e-6), "cy")
    var view = floats(e["view0"])
    for i in range(12):
        assert_true(close(Float64(c.view[i]), view[i], 1e-6), String("view[", i, "]"))

    var sums = floats(e["xyz_sum"])
    var s = SIMD[DType.float64, 4](0.0)
    for i in range(ds.points.count()):
        for k in range(3):
            s[k] += Float64(ds.points.xyz[i * 3 + k])
    for k in range(3):
        assert_true(close(s[k], sums[k], 1e-7), "point sum")

    # Decoded pixels on the GPU vs PIL.
    var px = download(ctx, v.image, c.width * c.height * 3)
    var total = 0
    for p in px:
        total += Int(p)
    var pil_total = Int(e["img0_sum"])
    print("  pixel sum", total, "PIL", pil_total)
    assert_true(abs(total - pil_total) <= c.width * c.height // 100, "decoded image differs from PIL")


def test_integer_factor() raises:
    """3dgrut's intrinsics: the COLMAP ones divided by the integer factor of the heights
    (garden images_4: 4), where the default scales each axis by the exact size ratio."""
    if not exists(GARDEN):
        print("  SKIP: needs", GARDEN)
        return
    var ctx = DeviceContext()
    var exact = load_colmap_dataset(ctx, GARDEN)
    var nv = load_colmap_dataset(ctx, GARDEN, integer_factor=True, median_extent=True)
    var model = read_colmap_model(GARDEN + "/sparse/0")
    var k = model.camera(model.images[0].camera_id).pinhole()
    ref c = nv.test[0].camera
    assert_equal(c.width, exact.test[0].camera.width)
    assert_true(close(Float64(c.fx), k[0] / 4.0, 1e-6), "fx / 4")
    assert_true(close(Float64(c.fy), k[1] / 4.0, 1e-6), "fy / 4")
    assert_true(close(Float64(c.cx), k[2] / 4.0, 1e-6), "cx / 4")
    assert_true(close(Float64(c.cy), k[3] / 4.0, 1e-6), "cy / 4")
    assert_true(abs(c.fx - exact.test[0].camera.fx) > 1e-3, "not the exact ratio")
    var cams = List[Camera]()
    for v in nv.train:
        cams.append(v.camera)
    for v in nv.test:
        cams.append(v.camera)
    assert_equal(nv.scene_extent, median_scene_extent(cams))


def _at(x: Float32, y: Float32) -> Camera:
    return Camera.translated(-x, -y, 0.0, 64, 48, 50.0, 50.0)


def test_median_scene_extent() raises:
    # Median center (1.5, 0, 0); distances 1.5, 0.5, 0.5, 8.5 -> median 1.0.
    var even: List[Camera] = [_at(0, 0), _at(1, 0), _at(2, 0), _at(10, 0)]
    assert_true(abs(median_scene_extent(even) - 1.1) < 1e-6, "even count")
    # Median center (1, 1, 0); distances 1, 1, 1, 9, sqrt(82) -> median 1.
    var odd: List[Camera] = [_at(0, 1), _at(1, 0), _at(2, 1), _at(10, 1), _at(1, 10)]
    assert_true(abs(median_scene_extent(odd) - 1.1) < 1e-6, "odd count")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
