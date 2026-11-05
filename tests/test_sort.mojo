from max.gpu.host import DeviceContext
from std.random import random_ui64, seed
from std.testing import assert_equal, assert_true, TestSuite

from mojosplat.common import dptr
from mojosplat.io import upload, download
from mojosplat.sort import radix_sort_pairs, sort_hist_size, sort_partials_size


def check_sort(n: Int, end_bit: Int, high_cardinality: Bool) raises:
    seed(n + end_bit)
    var ctx = DeviceContext()
    var keys = List[UInt64](capacity=n)
    var vals = List[Int32](capacity=n)
    var key_mask = (UInt64(1) << UInt64(end_bit)) - 1
    for i in range(n):
        var k: UInt64
        if high_cardinality:
            k = random_ui64(0, UInt64.MAX) & key_mask
        else:
            # Few distinct keys -> lots of ties; checks stability.
            k = (random_ui64(0, 50) << UInt64(end_bit - 8)) & key_mask
        keys.append(k)
        vals.append(Int32(i))

    var d_ka = upload(ctx, keys)
    var d_va = upload(ctx, vals)
    var d_kb = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_vb = ctx.enqueue_create_buffer[DType.int32](n)
    var d_hist = ctx.enqueue_create_buffer[DType.uint32](sort_hist_size(n))
    var d_part = ctx.enqueue_create_buffer[DType.uint32](sort_partials_size(n))
    var d_total = ctx.enqueue_create_buffer[DType.uint32](1)
    var in_b = radix_sort_pairs(
        ctx, ctx.stream(), dptr(d_ka), dptr(d_va), dptr(d_kb), dptr(d_vb),
        dptr(d_hist), dptr(d_part), dptr(d_total), n, end_bit,
    )
    ctx.synchronize()
    var out_k = download(ctx, d_kb, n) if in_b else download(ctx, d_ka, n)
    var out_v = download(ctx, d_vb, n) if in_b else download(ctx, d_va, n)

    for i in range(n):
        # Values are original indices: the pair must be preserved.
        assert_equal(out_k[i], keys[Int(out_v[i])], String("pair broken at ", i))
        if i > 0:
            assert_true(out_k[i - 1] <= out_k[i], String("not sorted at ", i))
            if out_k[i - 1] == out_k[i]:
                assert_true(out_v[i - 1] < out_v[i], String("not stable at ", i))


def test_sort_small() raises:
    for n in [2, 3, 31, 32, 33, 1000, 4095, 4096, 4097]:
        check_sort(n, 44, True)


def test_sort_large_random() raises:
    check_sort(1_000_000, 44, True)


def test_sort_full_64_bits() raises:
    check_sort(300_000, 64, True)


def test_sort_odd_end_bit() raises:
    # 40 bits -> 5 passes: result lands in the `b` buffers.
    check_sort(200_000, 40, True)


def test_sort_stability_with_ties() raises:
    check_sort(500_000, 44, False)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
