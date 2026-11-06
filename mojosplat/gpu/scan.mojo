"""Device-wide exclusive prefix sum (reduce-then-scan, 3 kernels).

Built on MAX's `block.prefix_sum` / `block.sum` primitives. Each block owns a
tile of SCAN_TILE = SCAN_BLOCK * SCAN_ITEMS consecutive elements:

  1. reduce:    tile sums -> partials[num_tiles]
  2. scan_tops: one block scans partials in place (exclusive) and writes the
                grand total
  3. downsweep: each tile rescans its elements, offset by its partial

Input and output may alias (the downsweep stages the tile in shared memory).
"""

from max.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu.memory import AddressSpace
from max.gpu.primitives import block
from max.gpu.sync import barrier
from std.memory import stack_allocation

comptime SCAN_BLOCK = 256
comptime SCAN_ITEMS = 16
comptime SCAN_TILE = SCAN_BLOCK * SCAN_ITEMS


def scan_workspace_size(n: Int) -> Int:
    """Number of elements of scratch (`partials`) needed to scan `n` items."""
    return max(1, (n + SCAN_TILE - 1) // SCAN_TILE)


def _reduce_kernel[
    dtype: DType
](input: Pointer[Scalar[dtype], MutUntrackedOrigin], partials: Pointer[Scalar[dtype], MutUntrackedOrigin], n: Int32,):
    var base = block_idx.x * SCAN_TILE
    var acc = Scalar[dtype](0)
    comptime for i in range(SCAN_ITEMS):
        var idx = base + i * SCAN_BLOCK + thread_idx.x
        if idx < Int(n):
            acc += input[unsafe_offset=idx]
    var total = block.sum[block_size=SCAN_BLOCK, broadcast=False](acc)
    if thread_idx.x == 0:
        partials[unsafe_offset=block_idx.x] = total


def _scan_tops_kernel[
    dtype: DType
](
    partials: Pointer[Scalar[dtype], MutUntrackedOrigin],
    total_out: Pointer[Scalar[dtype], MutUntrackedOrigin],
    num_tiles: Int32,
):
    # Single block: walk the partials in chunks of SCAN_BLOCK with a carry.
    var carry = Scalar[dtype](0)
    var chunks = (Int(num_tiles) + SCAN_BLOCK - 1) // SCAN_BLOCK
    for c in range(chunks):
        var idx = c * SCAN_BLOCK + thread_idx.x
        var v = Scalar[dtype](0)
        if idx < Int(num_tiles):
            v = partials[unsafe_offset=idx]
        var ex = block.prefix_sum[block_size=SCAN_BLOCK, exclusive=True](v)
        var chunk_total = block.sum[block_size=SCAN_BLOCK, broadcast=True](v)
        if idx < Int(num_tiles):
            partials[unsafe_offset=idx] = carry + ex
        carry += chunk_total
    if thread_idx.x == 0:
        total_out[unsafe_offset=0] = carry


def _downsweep_kernel[
    dtype: DType
](
    input: Pointer[Scalar[dtype], MutUntrackedOrigin],
    output: Pointer[Scalar[dtype], MutUntrackedOrigin],
    partials: Pointer[Scalar[dtype], MutUntrackedOrigin],
    n: Int32,
):
    var tile = stack_allocation[SCAN_TILE, Scalar[dtype], address_space=AddressSpace.SHARED]()
    var base = block_idx.x * SCAN_TILE
    var t = thread_idx.x

    # Coalesced load into shared memory (zero-padded past n).
    comptime for i in range(SCAN_ITEMS):
        var local = i * SCAN_BLOCK + t
        var idx = base + local
        var v = Scalar[dtype](0)
        if idx < Int(n):
            v = input[unsafe_offset=idx]
        tile[unsafe_offset=local] = v
    barrier()

    # Each thread scans SCAN_ITEMS contiguous elements serially.
    var thread_sum = Scalar[dtype](0)
    comptime for i in range(SCAN_ITEMS):
        thread_sum += tile[unsafe_offset=t * SCAN_ITEMS + i]
    var running = (
        block.prefix_sum[block_size=SCAN_BLOCK, exclusive=True](thread_sum) + partials[unsafe_offset=block_idx.x]
    )
    barrier()
    comptime for i in range(SCAN_ITEMS):
        var v = tile[unsafe_offset=t * SCAN_ITEMS + i]
        tile[unsafe_offset=t * SCAN_ITEMS + i] = running
        running += v
    barrier()

    # Coalesced store.
    comptime for i in range(SCAN_ITEMS):
        var local = i * SCAN_BLOCK + t
        var idx = base + local
        if idx < Int(n):
            output[unsafe_offset=idx] = tile[unsafe_offset=local]


def exclusive_scan[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    input: Pointer[Scalar[dtype], MutUntrackedOrigin],
    output: Pointer[Scalar[dtype], MutUntrackedOrigin],
    total: Pointer[Scalar[dtype], MutUntrackedOrigin],
    partials: Pointer[Scalar[dtype], MutUntrackedOrigin],
    n: Int,
) raises:
    """Enqueue `output = exclusive_cumsum(input)`; `total[0] = sum(input)`.

    `partials` must hold at least `scan_workspace_size(n)` elements.
    """
    var num_tiles = scan_workspace_size(n)
    stream.enqueue_function(
        ctx.compile_function[_reduce_kernel[dtype]](),
        input,
        partials,
        Int32(n),
        grid_dim=num_tiles,
        block_dim=SCAN_BLOCK,
    )
    stream.enqueue_function(
        ctx.compile_function[_scan_tops_kernel[dtype]](),
        partials,
        total,
        Int32(num_tiles),
        grid_dim=1,
        block_dim=SCAN_BLOCK,
    )
    stream.enqueue_function(
        ctx.compile_function[_downsweep_kernel[dtype]](),
        input,
        output,
        partials,
        Int32(n),
        grid_dim=num_tiles,
        block_dim=SCAN_BLOCK,
    )
