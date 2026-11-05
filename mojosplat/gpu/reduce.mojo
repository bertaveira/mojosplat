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
            src, dptr(self.partials), Int32(n),
            grid_dim=REDUCE_GRID, block_dim=REDUCE_BLOCK,
        )
        var host = self.ctx.enqueue_create_host_buffer[DType.float32](REDUCE_GRID)
        self.ctx.enqueue_copy(host, self.partials)
        self.ctx.synchronize()
        var total: Float64 = 0.0
        for i in range(REDUCE_GRID):
            total += Float64(host[i])
        return total
