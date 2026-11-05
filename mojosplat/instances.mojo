"""Gaussian -> tile instances for the two-stage sort (Faster-GS / Splatshop).

Instead of sorting one (tile << 32 | depth) 64-bit key per instance, the
visible Gaussians are first sorted by depth (32-bit keys, one entry per
Gaussian), then instances are emitted in that depth order with a tile-only
key and sorted *stably* by tile. With ~5k tiles the tile key fits in 16 bits,
so the per-instance sort is 2 radix passes over 2-byte keys.

Kernels ported from Faster-GS (Apache-2.0,
github.com/nerficg-project/faster-gaussian-splatting).
"""

from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu.primitives import warp
from std.bit import count_trailing_zeros, pop_count
from std.math import log

from .common import BLOCK_SIZE, F32Ptr, I32Ptr, LANE_MASK_DTYPE, LaneMask, U32Ptr, num_blocks
from .preprocess import MIN_ALPHA_RCP, N_SEQUENTIAL, load_record, record_bounds, tile_rect, will_contribute


def _depth_ordering_kernel(
    sorted_ids: I32Ptr,  # [n_visible] slots in depth order
    n_touched: U32Ptr,  # [slot]
    counts: U32Ptr,  # out [n_visible]
    n_visible: Int32,
):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i < Int(n_visible):
        counts[unsafe_offset=i] = n_touched[unsafe_offset=Int(sorted_ids[unsafe_offset=i])]


def _create_instances_kernel[
    key_dtype: DType, tile_w: Int, tile_h: Int
](
    sorted_ids: I32Ptr,  # [n_visible] slots in depth order
    offsets: U32Ptr,  # [n_visible] exclusive scan of n_touched in depth order
    n_touched: U32Ptr,  # [slot]
    records: F32Ptr,  # [slot, REC]
    keys: Pointer[Scalar[key_dtype], MutUntrackedOrigin],  # out [M] tile ids
    instance_ids: I32Ptr,  # out [M] slots
    grid_w: Int32,
    sentinel: Int32,  # key for unused slots (>= n_tiles, sorts last)
    n_visible: Int32,
):
    var lane = lane_id()
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    var active = i < Int(n_visible)
    if warp.vote[LANE_MASK_DTYPE](active) == 0:
        return
    var ii = min(i, Int(n_visible) - 1)
    var g = Int(sorted_ids[unsafe_offset=ii])
    var rec = load_record(records, g)
    var r = tile_rect[tile_w, tile_h](record_bounds(rec))
    var tx0 = Int(r[0])
    var ty0 = Int(r[2])
    var tw = Int(r[1]) - tx0
    var tile_count = tw * (Int(r[3]) - ty0)
    var mx = rec[0] - 0.5
    var my = rec[1] - 0.5
    var ca = rec[2]
    var cb = rec[3]
    var cc = rec[4]
    var pt = log(rec[5] * MIN_ALPHA_RCP)
    var start = Int(offsets[unsafe_offset=ii])
    # Never write past this Gaussian's slots, even if the recomputed test
    # disagreed with the count from preprocessing (different FP contraction).
    var end = start + Int(n_touched[unsafe_offset=g])
    var w = start

    if active:
        for t in range(min(tile_count, N_SEQUENTIAL)):
            var tx = tx0 + t % tw
            var ty = ty0 + t // tw
            if w < end and will_contribute[tile_w, tile_h](mx, my, ca, cb, cc, tx, ty, pt):
                keys[unsafe_offset=w] = Scalar[key_dtype](ty * Int(grid_w) + tx)
                instance_ids[unsafe_offset=w] = Int32(g)
                w += 1

    var coop = warp.vote[LANE_MASK_DTYPE](active and tile_count > N_SEQUENTIAL)
    var lt = (LaneMask(1) << LaneMask(lane)) - 1
    while coop != 0:
        var src = UInt32(count_trailing_zeros(coop))
        coop &= coop - 1
        var s_g = warp.shuffle_idx(Int32(g), src)
        var s_tx0 = Int(warp.shuffle_idx(Int32(tx0), src))
        var s_ty0 = Int(warp.shuffle_idx(Int32(ty0), src))
        var s_tw = Int(warp.shuffle_idx(Int32(tw), src))
        var s_count = Int(warp.shuffle_idx(Int32(tile_count), src))
        var s_w = Int(warp.shuffle_idx(Int32(w), src))
        var s_end = Int(warp.shuffle_idx(Int32(end), src))
        var s_mx = warp.shuffle_idx(mx, src)
        var s_my = warp.shuffle_idx(my, src)
        var s_ca = warp.shuffle_idx(ca, src)
        var s_cb = warp.shuffle_idx(cb, src)
        var s_cc = warp.shuffle_idx(cc, src)
        var s_pt = warp.shuffle_idx(pt, src)
        var t = N_SEQUENTIAL + lane
        while t - lane < s_count:
            var tx = s_tx0 + t % s_tw
            var ty = s_ty0 + t // s_tw
            var hit = t < s_count and will_contribute[tile_w, tile_h](
                s_mx, s_my, s_ca, s_cb, s_cc, tx, ty, s_pt
            )
            var hits = warp.vote[LANE_MASK_DTYPE](hit)
            if hit:
                var dst = s_w + Int(pop_count(hits & lt))
                if dst < s_end:
                    keys[unsafe_offset=dst] = Scalar[key_dtype](ty * Int(grid_w) + tx)
                    instance_ids[unsafe_offset=dst] = s_g
            s_w += Int(pop_count(hits))
            t += WARP_SIZE
        if lane == Int(src):
            w = s_w

    # Pad any unused slots (only if the recount came out lower).
    if active:
        while w < end:
            keys[unsafe_offset=w] = Scalar[key_dtype](sentinel)
            instance_ids[unsafe_offset=w] = Int32(g)
            w += 1


def _tile_offsets_kernel[
    key_dtype: DType
](
    keys: Pointer[Scalar[key_dtype], MutUntrackedOrigin],  # [M] sorted by tile
    offsets: U32Ptr,  # out [n_tiles + 1]: tile t owns [offsets[t], offsets[t+1])
    m: Int32,
    n_tiles: Int32,
):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i >= Int(m):
        return
    var nt = Int(n_tiles)
    var k = min(Int(keys[unsafe_offset=i]), nt)  # sentinels count as "past the end"
    var p = -1 if i == 0 else min(Int(keys[unsafe_offset=i - 1]), nt)
    for t in range(p + 1, k + 1):
        offsets[unsafe_offset=t] = UInt32(i)
    if i == Int(m) - 1:
        for t in range(k + 1, nt + 1):
            offsets[unsafe_offset=t] = UInt32(m)


def _fill_u32_kernel(p: U32Ptr, value: UInt32, n: Int32):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i < Int(n):
        p[unsafe_offset=i] = value


def depth_ordering(
    ctx: DeviceContext,
    stream: DeviceStream,
    sorted_ids: I32Ptr,
    n_touched: U32Ptr,
    counts: U32Ptr,
    n_visible: Int,
) raises:
    stream.enqueue_function(
        ctx.compile_function[_depth_ordering_kernel](),
        sorted_ids, n_touched, counts, Int32(n_visible),
        grid_dim=num_blocks(n_visible), block_dim=BLOCK_SIZE,
    )


def create_instances[
    key_dtype: DType, tile_w: Int, tile_h: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    sorted_ids: I32Ptr,
    offsets: U32Ptr,
    n_touched: U32Ptr,
    records: F32Ptr,
    keys: Pointer[Scalar[key_dtype], MutUntrackedOrigin],
    instance_ids: I32Ptr,
    grid_w: Int,
    n_tiles: Int,
    n_visible: Int,
) raises:
    stream.enqueue_function(
        ctx.compile_function[_create_instances_kernel[key_dtype, tile_w, tile_h]](),
        sorted_ids, offsets, n_touched, records, keys, instance_ids, Int32(grid_w), Int32(n_tiles), Int32(n_visible),
        grid_dim=num_blocks(n_visible), block_dim=BLOCK_SIZE,
    )


def tile_offsets[
    key_dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    keys: Pointer[Scalar[key_dtype], MutUntrackedOrigin],
    offsets: U32Ptr,
    m: Int,
    n_tiles: Int,
) raises:
    """Per-tile instance ranges from tile-sorted keys ([n_tiles + 1])."""
    if m == 0:
        stream.enqueue_function(
            ctx.compile_function[_fill_u32_kernel](),
            offsets, UInt32(0), Int32(n_tiles + 1),
            grid_dim=num_blocks(n_tiles + 1), block_dim=BLOCK_SIZE,
        )
        return
    stream.enqueue_function(
        ctx.compile_function[_tile_offsets_kernel[key_dtype]](),
        keys, offsets, Int32(m), Int32(n_tiles),
        grid_dim=num_blocks(m), block_dim=BLOCK_SIZE,
    )


def tile_key_bits(n_tiles: Int) -> Int:
    """Bits needed for tile keys 0..n_tiles (n_tiles itself is the sentinel)."""
    var bits = 1
    while (1 << bits) <= n_tiles:
        bits += 1
    return bits
