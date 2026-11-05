"""Elementwise device helpers (fill, ...)."""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceStream

from ..core.types import BLOCK_SIZE, F32Ptr, U32Ptr, num_blocks


def _fill_f32_kernel(p: F32Ptr, value: Float32, n: Int32):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i < Int(n):
        p[unsafe_offset=i] = value


def fill_f32(ctx: DeviceContext, stream: DeviceStream, p: F32Ptr, value: Float32, n: Int) raises:
    if n == 0:
        return
    stream.enqueue_function(
        ctx.compile_function[_fill_f32_kernel](), p, value, Int32(n),
        grid_dim=num_blocks(n), block_dim=BLOCK_SIZE,
    )


def _fill_u32_kernel(p: U32Ptr, value: UInt32, n: Int32):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i < Int(n):
        p[unsafe_offset=i] = value


def fill_u32(ctx: DeviceContext, stream: DeviceStream, p: U32Ptr, value: UInt32, n: Int) raises:
    if n == 0:
        return
    stream.enqueue_function(
        ctx.compile_function[_fill_u32_kernel](), p, value, Int32(n),
        grid_dim=num_blocks(n), block_dim=BLOCK_SIZE,
    )
