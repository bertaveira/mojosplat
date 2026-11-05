"""Tile binning: Gaussian -> tile intersections, sorted by (tile, depth).

Stages (driven by `Renderer`):
  1. count_tiles:          tiles_per_gauss[g] = #tiles overlapped by g
  2. exclusive scan:       offsets[g] (+ total M)            -> scan.mojo
  3. write_intersections:  key = (tile_id << 32) | depth_bits, value = g
  4. radix sort on the low 32 + bit_width(num_tiles) key bits -> sort.mojo
  5. tile_offsets:         first intersection of every tile (+ M at the end)
"""

from max.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceStream
from std.math import floor, ceil

from .common import F32Ptr, I32Ptr, U64Ptr, BLOCK_SIZE, num_blocks


@always_inline
def _tile_rect[
    tile_size: Int
](
    mean_x: Float32, mean_y: Float32, rx: Int32, ry: Int32, tiles_x: Int32, tiles_y: Int32
) -> SIMD[DType.int32, 4]:
    """[min_x, min_y, max_x, max_y) tile rectangle covered by a Gaussian."""
    var tx = mean_x / Float32(tile_size)
    var ty = mean_y / Float32(tile_size)
    var trx = Float32(rx) / Float32(tile_size)
    var try_ = Float32(ry) / Float32(tile_size)
    var wx = Float32(tiles_x)
    var wy = Float32(tiles_y)
    return SIMD[DType.int32, 4](
        Int32(min(max(Float32(0.0), floor(tx - trx)), wx)),
        Int32(min(max(Float32(0.0), floor(ty - try_)), wy)),
        Int32(min(max(Float32(0.0), ceil(tx + trx)), wx)),
        Int32(min(max(Float32(0.0), ceil(ty + try_)), wy)),
    )


def _count_kernel[
    tile_size: Int
](
    means2d: F32Ptr,
    radii: I32Ptr,
    tiles_per_gauss: I32Ptr,
    n: Int32,
    tiles_x: Int32,
    tiles_y: Int32,
):
    var g = block_idx.x * BLOCK_SIZE + thread_idx.x
    if g >= Int(n):
        return
    var rx = radii[unsafe_offset=g * 2 + 0]
    var ry = radii[unsafe_offset=g * 2 + 1]
    if rx <= 0 or ry <= 0:
        tiles_per_gauss[unsafe_offset=g] = 0
        return
    var r = _tile_rect[tile_size](
        means2d[unsafe_offset=g * 2 + 0], means2d[unsafe_offset=g * 2 + 1],
        rx, ry, tiles_x, tiles_y,
    )
    tiles_per_gauss[unsafe_offset=g] = (r[2] - r[0]) * (r[3] - r[1])


def _write_kernel[
    tile_size: Int
](
    means2d: F32Ptr,
    radii: I32Ptr,
    depths: F32Ptr,
    offsets: I32Ptr,
    keys: U64Ptr,
    ids: I32Ptr,
    n: Int32,
    tiles_x: Int32,
    tiles_y: Int32,
):
    var g = block_idx.x * BLOCK_SIZE + thread_idx.x
    if g >= Int(n):
        return
    var rx = radii[unsafe_offset=g * 2 + 0]
    var ry = radii[unsafe_offset=g * 2 + 1]
    if rx <= 0 or ry <= 0:
        return
    var r = _tile_rect[tile_size](
        means2d[unsafe_offset=g * 2 + 0], means2d[unsafe_offset=g * 2 + 1],
        rx, ry, tiles_x, tiles_y,
    )
    var cur = Int(offsets[unsafe_offset=g])
    # Positive float depths sort correctly by their IEEE-754 bit pattern.
    var depth_bits = UInt64(depths[unsafe_offset=g].to_bits())
    for ty in range(Int(r[1]), Int(r[3])):
        for tx in range(Int(r[0]), Int(r[2])):
            var tile_id = UInt64(ty * Int(tiles_x) + tx)
            keys[unsafe_offset=cur] = (tile_id << 32) | depth_bits
            ids[unsafe_offset=cur] = Int32(g)
            cur += 1


def _tile_offsets_kernel(
    sorted_keys: U64Ptr,
    tile_offsets: I32Ptr,  # [n_tiles + 1]
    m: Int32,
    n_tiles: Int32,
):
    """tile_offsets[t] = first index whose tile id is >= t; tile_offsets[n_tiles] = M.

    One thread per intersection: a change of tile id between idx-1 and idx
    marks idx as the start of every tile in (prev, curr].
    """
    var idx = block_idx.x * BLOCK_SIZE + thread_idx.x
    if idx >= Int(m):
        return
    var curr = Int(sorted_keys[unsafe_offset=idx] >> 32)
    if idx == 0:
        for t in range(0, curr + 1):
            tile_offsets[unsafe_offset=t] = 0
    else:
        var prev = Int(sorted_keys[unsafe_offset=idx - 1] >> 32)
        for t in range(prev + 1, curr + 1):
            tile_offsets[unsafe_offset=t] = Int32(idx)
    if idx == Int(m) - 1:
        for t in range(curr + 1, Int(n_tiles) + 1):
            tile_offsets[unsafe_offset=t] = m


def _fill_kernel(dst: I32Ptr, value: Int32, count: Int32):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i < Int(count):
        dst[unsafe_offset=i] = value


def count_tiles[
    tile_size: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    means2d: F32Ptr,
    radii: I32Ptr,
    tiles_per_gauss: I32Ptr,
    n: Int,
    tiles_x: Int,
    tiles_y: Int,
) raises:
    stream.enqueue_function(
        ctx.compile_function[_count_kernel[tile_size]](),
        means2d, radii, tiles_per_gauss, Int32(n), Int32(tiles_x), Int32(tiles_y),
        grid_dim=num_blocks(n), block_dim=BLOCK_SIZE,
    )


def write_intersections[
    tile_size: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    means2d: F32Ptr,
    radii: I32Ptr,
    depths: F32Ptr,
    offsets: I32Ptr,
    keys: U64Ptr,
    ids: I32Ptr,
    n: Int,
    tiles_x: Int,
    tiles_y: Int,
) raises:
    stream.enqueue_function(
        ctx.compile_function[_write_kernel[tile_size]](),
        means2d, radii, depths, offsets, keys, ids,
        Int32(n), Int32(tiles_x), Int32(tiles_y),
        grid_dim=num_blocks(n), block_dim=BLOCK_SIZE,
    )


def compute_tile_offsets(
    ctx: DeviceContext,
    stream: DeviceStream,
    sorted_keys: U64Ptr,
    tile_offsets: I32Ptr,
    m: Int,
    n_tiles: Int,
) raises:
    if m == 0:
        stream.enqueue_function(
            ctx.compile_function[_fill_kernel](),
            tile_offsets, Int32(0), Int32(n_tiles + 1),
            grid_dim=num_blocks(n_tiles + 1), block_dim=BLOCK_SIZE,
        )
        return
    stream.enqueue_function(
        ctx.compile_function[_tile_offsets_kernel](),
        sorted_keys, tile_offsets, Int32(m), Int32(n_tiles),
        grid_dim=num_blocks(m), block_dim=BLOCK_SIZE,
    )


def sort_end_bit(n_tiles: Int) -> Int:
    """Key bits to sort: 32 depth bits + enough bits for the largest tile id."""
    var bits = 0
    while (1 << bits) < n_tiles:
        bits += 1
    return 32 + max(bits, 1)


def _copy_kernel(src: I32Ptr, dst: I32Ptr, count: Int32):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i < Int(count):
        dst[unsafe_offset=i] = src[unsafe_offset=i]


def copy_i32(
    ctx: DeviceContext, stream: DeviceStream, src: I32Ptr, dst: I32Ptr, count: Int
) raises:
    """Device-to-device copy of `count` int32 values on `stream`."""
    if count == 0:
        return
    stream.enqueue_function(
        ctx.compile_function[_copy_kernel](),
        src, dst, Int32(count),
        grid_dim=num_blocks(count), block_dim=BLOCK_SIZE,
    )
