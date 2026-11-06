"""Regression coverage for SIMD construction and kernel arguments on Metal."""

from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal, TestSuite

from mojosplat.core.types import F32Ptr, KernelVector, dptr
from mojosplat.gpu.reduce import bounds3
from mojosplat.io.arrays import download, upload
from mojosplat.render.sh import load_rgb


def _vector_kernel[width: Int](src: F32Ptr, dst: F32Ptr, weights: KernelVector[DType.float32, width]):
    var i = thread_idx.x
    var rgb = load_rgb(src, i)
    dst[unsafe_offset=i] = rgb[0] * weights[0] + rgb[1] * weights[width // 2] + rgb[2] * weights[width - 1]


def check_vector[width: Int]() raises:
    var ctx = DeviceContext()
    var values: List[Float32] = [1, 2, 3, 4, 5, 6]
    var src = upload(ctx, values)
    var dst = ctx.enqueue_create_buffer[DType.float32](2)
    var weights = SIMD[DType.float32, width](0.0)
    weights[0] = 2.0
    weights[width // 2] = 3.0
    weights[width - 1] = 5.0
    ctx.stream().enqueue_function(
        ctx.compile_function[_vector_kernel[width]](),
        dptr(src),
        dptr(dst),
        KernelVector[DType.float32, width](weights),
        grid_dim=1,
        block_dim=2,
    )
    var result = download(ctx, dst, 2)
    _ = src^
    assert_equal(result[0], 23.0)
    assert_equal(result[1], 53.0)


def test_vector_arguments() raises:
    check_vector[4]()
    check_vector[16]()
    check_vector[32]()


def test_bounds3() raises:
    var ctx = DeviceContext()
    var values: List[Float32] = [1, -2, 3, -4, 5, -6, 7, -8, 9]
    var src = upload(ctx, values)
    var bounds = bounds3(ctx, dptr(src), 3)
    _ = src^
    assert_equal(bounds[0][0], -4.0)
    assert_equal(bounds[0][1], -8.0)
    assert_equal(bounds[0][2], -6.0)
    assert_equal(bounds[1][0], 7.0)
    assert_equal(bounds[1][1], 5.0)
    assert_equal(bounds[1][2], 9.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
