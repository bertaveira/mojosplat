"""Splats storage: capacity growth keeps parameters and Adam moments."""

from max.gpu.host import DeviceContext
from std.testing import assert_equal, assert_raises, TestSuite

from mojosplat.core.types import dptr
from mojosplat.gpu.elementwise import copy_f32, fill_f32
from mojosplat.io.arrays import download, upload
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


def test_columns_follow_rows() raises:
    var ctx = DeviceContext()
    var s = make(ctx, 3)
    s.enable_training()
    var beta_values: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var filter_values: List[Float32] = [7.0, 8.0, 9.0]
    s.add_column("beta", 2, values=beta_values)
    s.add_column("filter", 1, trainable=False, values=filter_values)
    assert_equal(len(s.columns), Attr.COUNT + 2)
    fill_f32(ctx, dptr(s.exp_avg[Attr.COUNT]), 3.0, 6)
    assert_equal(len(s.exp_avg[Attr.COUNT + 1]), 1)  # no moments for a non-trainable column

    var ids: List[Int32] = [2, 0]
    var index = upload(ctx, ids)
    s.gather(dptr(index), 2, 1)
    ctx.synchronize()
    _ = index^
    var beta = download(ctx, s.params[Attr.COUNT], 4)
    assert_equal(beta[0], 5.0)
    assert_equal(beta[3], 2.0)
    assert_equal(download(ctx, s.params[Attr.COUNT + 1], 2)[1], 7.0)
    var m = download(ctx, s.exp_avg[Attr.COUNT], 4)
    assert_equal(m[1], 3.0)
    assert_equal(m[2], 0.0)  # row 1 is new: zeroed moments
    var h = s.to_host()
    assert_equal(len(h.extras), 2)
    assert_equal(h.extras[0].name, "beta")
    assert_equal(h.extras[1].data[0], 9.0)


def test_reset_moments_of_arbitrary_rows() raises:
    var ctx = DeviceContext()
    var s = make(ctx, 4)
    s.enable_training()
    fill_f32(ctx, dptr(s.exp_avg[Attr.MEANS]), 3.0, 4 * 3)
    fill_f32(ctx, dptr(s.exp_avg_sq[Attr.ROTATIONS]), 5.0, 4 * 4)
    var ids: List[Int32] = [3, 1]
    var rows = upload(ctx, ids)
    s.reset_moments(dptr(rows), 2)
    _ = rows^
    var m = download(ctx, s.exp_avg[Attr.MEANS], 4 * 3)
    assert_equal(m[0], 3.0)
    assert_equal(m[3], 0.0)
    assert_equal(m[8], 3.0)
    assert_equal(m[11], 0.0)
    var v = download(ctx, s.exp_avg_sq[Attr.ROTATIONS], 4 * 4)
    assert_equal(v[4], 0.0)
    assert_equal(v[8], 5.0)
    assert_equal(v[15], 0.0)


def test_permutation_can_keep_statistics() raises:
    var ctx = DeviceContext()
    var s = make(ctx, 3)
    s.enable_training()
    var values: List[Float32] = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
    var stats = upload(ctx, values)
    copy_f32(ctx, s.densification_ptr(), dptr(stats), 6)
    var ids: List[Int32] = [2, 0, 1]
    var index = upload(ctx, ids)
    s.gather(dptr(index), 3, 3, keep_statistics=True)
    _ = (stats^, index^)
    var d = download(ctx, s.densification, 6)
    assert_equal(d[0], 3.0)
    assert_equal(d[1], 1.0)
    assert_equal(d[3], 30.0)
    assert_equal(d[5], 20.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
