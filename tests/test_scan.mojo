from max.gpu.host import DeviceContext
from std.random import random_ui64, seed
from std.testing import assert_equal, TestSuite

from mojosplat.common import dptr
from mojosplat.io import upload, download
from mojosplat.scan import exclusive_scan, scan_workspace_size


def check_scan(n: Int, max_val: UInt64) raises:
    seed(n)
    var ctx = DeviceContext()
    var values = List[Int32](capacity=n)
    for _ in range(n):
        values.append(Int32(random_ui64(0, max_val)))
    var d_in = upload(ctx, values)
    var d_out = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var d_total = ctx.enqueue_create_buffer[DType.int32](1)
    var d_partials = ctx.enqueue_create_buffer[DType.int32](scan_workspace_size(n))
    exclusive_scan[DType.int32](
        ctx, ctx.stream(), dptr(d_in), dptr(d_out), dptr(d_total), dptr(d_partials), n
    )
    ctx.synchronize()
    var out = download(ctx, d_out, n)
    var total = download(ctx, d_total, 1)
    var running: Int32 = 0
    for i in range(n):
        assert_equal(out[i], running, String("mismatch at ", i, " (n=", n, ")"))
        running += values[i]
    assert_equal(total[0], running)


def test_scan_sizes() raises:
    for n in [1, 5, 255, 256, 257, 4095, 4096, 4097, 100_000, 1_000_003]:
        check_scan(n, 40)


def test_scan_many_tiles() raises:
    # > SCAN_BLOCK tiles so the top-level scan needs several chunks.
    check_scan(3_000_000, 3)


def test_scan_in_place() raises:
    var ctx = DeviceContext()
    var n = 10_000
    var values = List[Int32](capacity=n)
    for i in range(n):
        values.append(Int32(i % 7))
    var d = upload(ctx, values)
    var d_total = ctx.enqueue_create_buffer[DType.int32](1)
    var d_partials = ctx.enqueue_create_buffer[DType.int32](scan_workspace_size(n))
    exclusive_scan[DType.int32](
        ctx, ctx.stream(), dptr(d), dptr(d), dptr(d_total), dptr(d_partials), n
    )
    ctx.synchronize()
    var out = download(ctx, d, n)
    var running: Int32 = 0
    for i in range(n):
        assert_equal(out[i], running)
        running += values[i]


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
