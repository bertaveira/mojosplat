"""Morton reordering: a permutation along the Z-order curve that moves every
array (parameters and Adam moments) together."""

from max.gpu.host import DeviceContext
from std.random import random_float64, seed
from std.testing import assert_equal, assert_true, TestSuite

from mojosplat.core.types import dptr
from mojosplat.gpu.elementwise import copy_f32
from mojosplat.io.arrays import download, upload
from mojosplat.model.morton import morton_order
from mojosplat.model.splats import Attr, Splats


def split_by_3(a: UInt64) -> UInt64:
    var code = UInt64(0)
    for b in range(21):
        code |= ((a >> UInt64(b)) & 1) << UInt64(3 * b)
    return code


def test_morton_permutation() raises:
    seed(7)
    var n = 3000
    var ctx = DeviceContext()
    var means = List[Float32]()
    var tag = List[Float32]()
    for i in range(n):
        for _ in range(3):
            means.append(Float32(random_float64(-5.0, 5.0)))
        for _ in range(3):
            tag.append(Float32(i))
    var z1 = List[Float32](length=n, fill=0.0)
    var z3 = List[Float32](length=n * 3, fill=0.0)
    var q = List[Float32](length=n * 4, fill=1.0)
    var splats = Splats(ctx, means, z3, q, z1, tag, List[Float32](), 0)
    splats.enable_training()
    var t = upload(ctx, tag)
    copy_f32(ctx, dptr(splats.exp_avg[Attr.SH0]), dptr(t), n * 3)
    morton_order(splats)
    var m = download(ctx, splats.params[Attr.MEANS], n * 3)
    var sh0 = download(ctx, splats.params[Attr.SH0], n * 3)
    var mom = download(ctx, splats.exp_avg[Attr.SH0], n * 3)
    var seen = List[Bool](length=n, fill=False)
    var lo = SIMD[DType.float32, 4](Float32.MAX)
    var hi = SIMD[DType.float32, 4](-Float32.MAX)
    for i in range(n):
        for k in range(3):
            lo[k] = min(lo[k], m[i * 3 + k])
            hi[k] = max(hi[k], m[i * 3 + k])
    var size = max(hi[0] - lo[0], max(hi[1] - lo[1], hi[2] - lo[2]))
    var prev = UInt64(0)
    for i in range(n):
        var orig = Int(sh0[i * 3])
        assert_true(not seen[orig], "duplicate row")
        seen[orig] = True
        assert_equal(mom[i * 3], sh0[i * 3])  # moments moved with their row
        assert_equal(m[i * 3], means[orig * 3])
        var code = UInt64(0)
        for k in range(3):
            var v = min(max((m[i * 3 + k] - lo[k]) / size, 0.0), 1.0)
            code |= split_by_3(UInt64(UInt32(v * 2097151.0))) << UInt64(k)
        assert_true(code >= prev, String("not in Z-order at row ", i))
        prev = code
    _ = t^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
