"""Splats storage: capacity growth keeps parameters and Adam moments."""

from max.gpu.host import DeviceContext
from std.testing import assert_equal, assert_raises, TestSuite

from mojosplat.core.types import dptr
from mojosplat.gpu.elementwise import fill_f32
from mojosplat.io.arrays import download
from mojosplat.model.splats import Attr, Splats


def make(ctx: DeviceContext, n: Int) raises -> Splats:
    var m = List[Float32]()
    var s = List[Float32]()
    var q = List[Float32]()
    var o = List[Float32]()
    var c = List[Float32]()
    var r = List[Float32]()
    for i in range(n):
        for k in range(3):
            m.append(Float32(i * 10 + k))
            s.append(Float32(-i))
            c.append(Float32(i))
        for k in range(4):
            q.append(Float32(k))
        o.append(Float32(i) * 0.5)
        for k in range(9):
            r.append(Float32(i * 100 + k))
    return Splats(ctx, m, s, q, o, c, r, 3)


def test_reserve_keeps_rows() raises:
    var ctx = DeviceContext()
    var s = make(ctx, 5)
    with assert_raises():
        _ = s.grad_view()
    s.enable_training()
    fill_f32(ctx, dptr(s.exp_avg[Attr.SH_REST]), 3.0, 5 * 9)
    s.reserve(12)
    assert_equal(s.capacity, 12)
    assert_equal(s.count, 5)
    var h = s.to_host()
    assert_equal(h.means[4 * 3 + 2], 42.0)
    assert_equal(h.sh_rest[4 * 9 + 8], 408.0)
    var m = download(ctx, s.exp_avg[Attr.SH_REST], 12 * 9)
    assert_equal(m[44], 3.0)  # kept
    assert_equal(m[45], 0.0)  # new rows are zero
    var v = s.view()
    assert_equal(v.count, 5)
    assert_equal(v.rest_stride, 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
