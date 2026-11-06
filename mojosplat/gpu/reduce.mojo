"""Device-wide reductions."""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, DeviceStream
from max.gpu.primitives import block

from ..core.types import F32Ptr, dptr

comptime REDUCE_BLOCK = 256
comptime REDUCE_GRID = 1024


def _block_sums_kernel(src: F32Ptr, partials: F32Ptr, n: Int32):
    var acc: Float32 = 0.0
    var i = block_idx.x * REDUCE_BLOCK + thread_idx.x
    while i < Int(n):
        acc += src[unsafe_offset=i]
        i += REDUCE_GRID * REDUCE_BLOCK
    var total = block.sum[block_size=REDUCE_BLOCK, broadcast=False](acc)
    if thread_idx.x == 0:
        partials[unsafe_offset=block_idx.x] = total


struct Summer(Movable):
    """Sums device float32 arrays (per-block partials, finished in float64 on
    the host). Synchronizes: meant for logging and evaluation, not hot loops."""

    var ctx: DeviceContext
    var partials: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext) raises:
        self.ctx = ctx
        self.partials = ctx.enqueue_create_buffer[DType.float32](REDUCE_GRID)

    def sum(mut self, src: F32Ptr, n: Int) raises -> Float64:
        if n == 0:
            return 0.0
        self.ctx.stream().enqueue_function(
            self.ctx.compile_function[_block_sums_kernel](),
            src,
            dptr(self.partials),
            Int32(n),
            grid_dim=REDUCE_GRID,
            block_dim=REDUCE_BLOCK,
        )
        var host = self.ctx.enqueue_create_host_buffer[DType.float32](REDUCE_GRID)
        self.ctx.enqueue_copy(host, self.partials)
        self.ctx.synchronize()
        var total: Float64 = 0.0
        for i in range(REDUCE_GRID):
            total += Float64(host[i])
        return total


def _block_minmax3_kernel(src: F32Ptr, partials: F32Ptr, n: Int32):
    """Per-block min/max of the x, y, z columns of src [n, 3] -> partials [block, 6]."""
    var lo = SIMD[DType.float32, 4](Float32.MAX)
    var hi = SIMD[DType.float32, 4](-Float32.MAX)
    var i = block_idx.x * REDUCE_BLOCK + thread_idx.x
    while i < Int(n):
        var v = SIMD[DType.float32, 4](
            src[unsafe_offset=i * 3], src[unsafe_offset=i * 3 + 1], src[unsafe_offset=i * 3 + 2], 0.0
        )
        lo = min(lo, v)
        hi = max(hi, v)
        i += REDUCE_GRID * REDUCE_BLOCK
    comptime for k in range(3):
        var mn = block.min[block_size=REDUCE_BLOCK, broadcast=False](lo[k])
        var mx = block.max[block_size=REDUCE_BLOCK, broadcast=False](hi[k])
        if thread_idx.x == 0:
            partials[unsafe_offset=block_idx.x * 6 + k] = mn
            partials[unsafe_offset=block_idx.x * 6 + 3 + k] = mx


def bounds3(ctx: DeviceContext, src: F32Ptr, n: Int) raises -> Tuple[SIMD[DType.float32, 4], SIMD[DType.float32, 4]]:
    """(min, max) of the rows of src [n, 3] (syncs)."""
    var partials = ctx.enqueue_create_buffer[DType.float32](REDUCE_GRID * 6)
    ctx.stream().enqueue_function(
        ctx.compile_function[_block_minmax3_kernel](),
        src,
        dptr(partials),
        Int32(n),
        grid_dim=REDUCE_GRID,
        block_dim=REDUCE_BLOCK,
    )
    var host = ctx.enqueue_create_host_buffer[DType.float32](REDUCE_GRID * 6)
    ctx.enqueue_copy(host, partials)
    ctx.synchronize()
    var lo = SIMD[DType.float32, 4](Float32.MAX)
    var hi = SIMD[DType.float32, 4](-Float32.MAX)
    for b in range(REDUCE_GRID):
        for k in range(3):
            lo[k] = min(lo[k], host[b * 6 + k])
            hi[k] = max(hi[k], host[b * 6 + 3 + k])
    return (lo, hi)
