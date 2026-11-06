"""Fill and copy device memory given raw pointers.

Enqueued on the context's stream, for code that holds `F32Ptr`s rather than
`DeviceBuffer`s. Metal uses kernels: its driver copy/fill APIs need buffer
handles and cannot wrap arbitrary raw pointers in non-owning DeviceBuffers.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu import block_idx, thread_idx
from std.sys import CompilationTarget

from ..core.types import BLOCK_SIZE, F32Ptr, U32Ptr, num_blocks


def _fill_kernel[dtype: DType](p: Pointer[Scalar[dtype], MutUntrackedOrigin], value: Scalar[dtype], n: Int32):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i < Int(n):
        p[unsafe_offset=i] = value


def _copy_kernel(dst: F32Ptr, src: F32Ptr, n: Int32):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i < Int(n):
        dst[unsafe_offset=i] = src[unsafe_offset=i]


def fill_f32(ctx: DeviceContext, p: F32Ptr, value: Float32, n: Int) raises:
    """p[0:n] = value."""
    if n > 0:
        comptime if CompilationTarget.is_macos():
            ctx.stream().enqueue_function(
                ctx.compile_function[_fill_kernel[DType.float32]](),
                p,
                value,
                Int32(n),
                grid_dim=num_blocks(n),
                block_dim=BLOCK_SIZE,
            )
        else:
            DeviceBuffer[DType.float32](ctx, p, n, owning=False).enqueue_fill(value)


def fill_u32(ctx: DeviceContext, p: U32Ptr, value: UInt32, n: Int) raises:
    """p[0:n] = value."""
    if n > 0:
        comptime if CompilationTarget.is_macos():
            ctx.stream().enqueue_function(
                ctx.compile_function[_fill_kernel[DType.uint32]](),
                p,
                value,
                Int32(n),
                grid_dim=num_blocks(n),
                block_dim=BLOCK_SIZE,
            )
        else:
            DeviceBuffer[DType.uint32](ctx, p, n, owning=False).enqueue_fill(value)


def copy_f32(ctx: DeviceContext, dst: F32Ptr, src: F32Ptr, n: Int) raises:
    """dst[0:n] = src[0:n] (device to device, non-overlapping)."""
    if n > 0:
        comptime if CompilationTarget.is_macos():
            ctx.stream().enqueue_function(
                ctx.compile_function[_copy_kernel](), dst, src, Int32(n), grid_dim=num_blocks(n), block_dim=BLOCK_SIZE
            )
        else:
            ctx.enqueue_copy(dst, src, n)
