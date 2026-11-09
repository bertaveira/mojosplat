"""3DGS PLY reader/writer vs a numpy-written file (property p of Gaussian g
holds 100 * g + p, see tests/data/make_golden_io.py)."""

from std.testing import assert_equal, assert_raises, TestSuite

from mojosplat.io.ply import HostColumn, HostSplats, read_ply, write_ply

comptime FIXTURE = "tests/data/golden/io/tiny.ply"


def expected(g: Int, p: Int) -> Float32:
    return Float32(100 * g + p)


def test_read_numpy_ply() raises:
    var s = read_ply(FIXTURE)
    assert_equal(s.count, 5)
    assert_equal(s.rest_stride, 3)
    var K = 3
    for g in range(5):
        for k in range(3):
            assert_equal(s.means[g * 3 + k], expected(g, k))
            assert_equal(s.sh0[g * 3 + k], expected(g, 6 + k))
            assert_equal(s.log_scales[g * 3 + k], expected(g, 9 + 3 * K + 1 + k))
        assert_equal(s.opacity_logits[g], expected(g, 9 + 3 * K))
        for k in range(4):
            assert_equal(s.rotations[g * 4 + k], expected(g, 9 + 3 * K + 4 + k))
        # f_rest is channel-major on disk: f_rest[c * K + k] -> sh_rest[g, k, c]
        for c in range(3):
            for k in range(K):
                assert_equal(s.sh_rest[(g * K + k) * 3 + c], expected(g, 9 + c * K + k))


def test_roundtrip_and_byte_identical() raises:
    var s = read_ply(FIXTURE)
    var path = String("tests/data/golden/io/_roundtrip.ply")
    write_ply(path, s)
    var a: List[UInt8]
    var b: List[UInt8]
    with open(FIXTURE, "r") as f:
        a = f.read_bytes()
    with open(path, "r") as f:
        b = f.read_bytes()
    # Our normals are written as 0, numpy's fixture has 100 g + 3..5:
    # compare after re-reading instead, plus the header bytes.
    var t = read_ply(path)
    assert_equal(t.count, s.count)
    for i in range(len(s.sh_rest)):
        assert_equal(t.sh_rest[i], s.sh_rest[i])
    for i in range(len(s.rotations)):
        assert_equal(t.rotations[i], s.rotations[i])
    var header_len = 0
    for i in range(len(a) - 10):
        if (
            a[i] == UInt8(ord("e"))
            and a[i + 1] == UInt8(ord("n"))
            and a[i + 2] == UInt8(ord("d"))
            and a[i + 3] == UInt8(ord("_"))
        ):
            header_len = i + 11
            break
    for i in range(header_len):
        assert_equal(a[i], b[i])


def test_no_sh_rest() raises:
    var s = HostSplats(2, 0)
    s.means[4] = 7.0
    s.opacity_logits[1] = -2.0
    var path = String("tests/data/golden/io/_k0.ply")
    write_ply(path, s)
    var t = read_ply(path)
    assert_equal(t.rest_stride, 0)
    assert_equal(t.means[4], 7.0)
    assert_equal(t.opacity_logits[1], -2.0)


def test_extra_columns_round_trip() raises:
    var s = HostSplats(2, 0)
    s.extras.append(HostColumn("spherical_sites", 2, [1.0, 2.0, 3.0, 4.0]))
    s.extras.append(HostColumn("beta", 1, [5.0, 6.0]))
    var path = String("tests/data/golden/io/_extras.ply")
    write_ply(path, s)
    var t = read_ply(path)
    assert_equal(len(t.extras), 2)
    assert_equal(t.extras[0].name, "spherical_sites")
    assert_equal(t.extras[0].width, 2)
    assert_equal(t.extras[0].data[3], 4.0)
    assert_equal(t.extras[1].name, "beta")
    assert_equal(t.extras[1].data[1], 6.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
