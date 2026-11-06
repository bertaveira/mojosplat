"""Fill and copy device memory given raw pointers.

Thin wrappers over the driver's memset / memcpy (enqueued on the context's
stream), for code that holds `F32Ptr`s rather than `DeviceBuffer`s.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from ..core.types import F32Ptr, U32Ptr


def fill_f32(ctx: DeviceContext, p: F32Ptr, value: Float32, n: Int) raises:
    """p[0:n] = value."""
    if n > 0:
        DeviceBuffer[DType.float32](ctx, p, n, owning=False).enqueue_fill(value)


def fill_u32(ctx: DeviceContext, p: U32Ptr, value: UInt32, n: Int) raises:
    """p[0:n] = value."""
    if n > 0:
        DeviceBuffer[DType.uint32](ctx, p, n, owning=False).enqueue_fill(value)


def copy_f32(ctx: DeviceContext, dst: F32Ptr, src: F32Ptr, n: Int) raises:
    """dst[0:n] = src[0:n] (device to device, non-overlapping)."""
    if n > 0:
        ctx.enqueue_copy(dst, src, n)
