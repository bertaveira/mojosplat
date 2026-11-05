"""JPEG decoding (libjpeg-turbo) vs PIL's decode of the same file."""

from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojosplat.io.arrays import read_array
from mojosplat.io.jpeg import JpegDecoder, read_jpeg


def test_decode_matches_pil() raises:
    var img = read_jpeg("tests/data/golden/io/tiny.jpg")
    assert_equal(img.width, 37)
    assert_equal(img.height, 23)
    var ref_px = read_array[DType.uint8]("tests/data/golden/io/tiny_rgb.u8")
    assert_equal(len(img.pixels), len(ref_px))
    var max_d = 0
    for i in range(len(ref_px)):
        max_d = max(max_d, abs(Int(img.pixels[i]) - Int(ref_px[i])))
    assert_true(max_d <= 1, String("max pixel difference vs PIL: ", max_d))


def test_decoder_reuse_and_errors() raises:
    var dec = JpegDecoder()
    var a = dec.read("tests/data/golden/io/tiny.jpg")
    var b = dec.read("tests/data/golden/io/tiny.jpg")
    assert_equal(len(a.pixels), len(b.pixels))
    with assert_raises():
        _ = dec.read("tests/data/golden/io/tiny.ply")  # not a JPEG


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
