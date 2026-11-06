"""Stable LSD radix sort of (unsigned key, Int32 value) pairs on the GPU.

Keys may be UInt16, UInt32 or UInt64 (`key_dtype`); smaller keys move fewer
bytes per pass.

Design (reduce-then-scan, 8-bit digits, one pass per digit):

  upsweep:   each partition of SORT_PART keys builds a 256-bin histogram in
             shared memory and writes it digit-major:
             hist[digit * num_parts + part]
  scan:      a device-wide exclusive scan of `hist` turns those counts into
             the global output offset of every (digit, partition) pair
  downsweep: each warp ranks its keys with `warp.match_any` (lanes holding
             the same digit), keeping per-warp digit counters in shared
             memory; a per-digit scan across warps then gives every key its
             final position. Order within a digit is preserved (stable).

Only the low `end_bit` bits are sorted (ceil(end_bit / 8) passes), which
saves whole passes when the keys use fewer bits than their type.

The approach follows the classic GPU radix sort literature (Merrill &
Grimshaw 2011; Adinets & Merrill, "Onesweep", 2022), not any specific code.
"""

from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from max.gpu.sync import barrier, syncwarp
from std.atomic import Atomic, Ordering
from std.bit import count_trailing_zeros, pop_count
from std.memory import stack_allocation

from ..core.types import LANE_MASK_DTYPE, I32Ptr, LaneMask, U32Ptr
from .scan import exclusive_scan, scan_workspace_size

comptime RADIX_BITS = 8
comptime RADIX = 1 << RADIX_BITS
comptime RADIX_MASK = RADIX - 1
comptime SORT_BLOCK = 256
comptime SORT_WARPS = SORT_BLOCK // WARP_SIZE
comptime SORT_ITEMS = 16
comptime SORT_PART = SORT_BLOCK * SORT_ITEMS
comptime WARP_SPAN = WARP_SIZE * SORT_ITEMS  # contiguous keys owned by one warp


def sort_num_parts(n: Int) -> Int:
    return max(1, (n + SORT_PART - 1) // SORT_PART)


def sort_hist_size(n: Int) -> Int:
    """Elements of `hist` scratch needed (UInt32)."""
    return RADIX * sort_num_parts(n)


def sort_partials_size(n: Int) -> Int:
    """Elements of scan `partials` scratch needed (UInt32)."""
    return scan_workspace_size(sort_hist_size(n))


def _upsweep_kernel[
    key_dtype: DType
](
    keys: Pointer[Scalar[key_dtype], MutUntrackedOrigin],
    hist: U32Ptr,  # [RADIX * num_parts], digit-major
    n: Int32,
    num_parts: Int32,
    shift: UInt32,
):
    comptime assert SORT_BLOCK == RADIX, "one thread per digit in the epilogue"
    var s_hist = stack_allocation[RADIX, UInt32, address_space=AddressSpace.SHARED]()
    var t = thread_idx.x
    s_hist[unsafe_offset=t] = 0
    barrier()

    var lane = lane_id()
    var base = block_idx.x * SORT_PART
    comptime for i in range(SORT_ITEMS):
        var idx = base + i * SORT_BLOCK + t
        var digit = UInt32(RADIX)  # sentinel for out-of-range lanes
        if idx < Int(n):
            digit = UInt32((keys[unsafe_offset=idx] >> Scalar[key_dtype](shift)) & RADIX_MASK)
        # Aggregate equal digits within the warp: one atomic per group.
        var peers = warp.match_any[mask_type=LANE_MASK_DTYPE](digit)
        if digit < RADIX and lane == Int(count_trailing_zeros(peers)):
            _ = Atomic[UInt32].fetch_add[ordering=Ordering.RELAXED](
                s_hist.unsafe_offset(Int(digit)), UInt32(pop_count(peers))
            )
    barrier()
    hist[unsafe_offset=t * Int(num_parts) + block_idx.x] = s_hist[unsafe_offset=t]


def _downsweep_kernel[
    key_dtype: DType
](
    keys_in: Pointer[Scalar[key_dtype], MutUntrackedOrigin],
    vals_in: I32Ptr,
    keys_out: Pointer[Scalar[key_dtype], MutUntrackedOrigin],
    vals_out: I32Ptr,
    offsets: U32Ptr,  # scanned hist: [RADIX * num_parts], digit-major
    n: Int32,
    num_parts: Int32,
    shift: UInt32,
):
    var s_warp_hist = stack_allocation[SORT_WARPS * RADIX, UInt32, address_space=AddressSpace.SHARED]()
    var t = thread_idx.x
    var w = t // WARP_SIZE
    var lane = lane_id()
    var hist_row = w * RADIX

    comptime for j in range(RADIX // WARP_SIZE):
        s_warp_hist[unsafe_offset=hist_row + j * WARP_SIZE + lane] = 0
    syncwarp()

    var lt_mask = (LaneMask(1) << LaneMask(lane)) - 1
    var base = block_idx.x * SORT_PART + w * WARP_SPAN
    var keys = SIMD[key_dtype, SORT_ITEMS](0)
    var vals = SIMD[DType.int32, SORT_ITEMS](0)
    var ranks = SIMD[DType.uint32, SORT_ITEMS](0)

    # Rank pass: warp w walks its contiguous WARP_SPAN keys 32 at a time, so
    # (round, lane) order == input order -> stable.
    comptime for r in range(SORT_ITEMS):
        var idx = base + r * WARP_SIZE + lane
        var digit = UInt32(RADIX)
        if idx < Int(n):
            var k = keys_in[unsafe_offset=idx]
            keys[r] = k
            vals[r] = vals_in[unsafe_offset=idx]
            digit = UInt32((k >> Scalar[key_dtype](shift)) & RADIX_MASK)
        var peers = warp.match_any[mask_type=LANE_MASK_DTYPE](digit)
        var leader = count_trailing_zeros(peers)
        var prior = UInt32(0)
        if digit < RADIX and lane == Int(leader):
            prior = s_warp_hist[unsafe_offset=hist_row + Int(digit)]
            s_warp_hist[unsafe_offset=hist_row + Int(digit)] = prior + UInt32(pop_count(peers))
        prior = warp.shuffle_idx(prior, UInt32(leader))
        ranks[r] = prior + UInt32(pop_count(peers & lt_mask))
        syncwarp()
    barrier()

    # Per digit: exclusive scan over warps, plus this partition's global base.
    var running = offsets[unsafe_offset=t * Int(num_parts) + block_idx.x]
    comptime for ww in range(SORT_WARPS):
        var c = s_warp_hist[unsafe_offset=ww * RADIX + t]
        s_warp_hist[unsafe_offset=ww * RADIX + t] = running
        running += c
    barrier()

    # Scatter.
    comptime for r in range(SORT_ITEMS):
        var idx = base + r * WARP_SIZE + lane
        if idx < Int(n):
            var k = keys[r]
            var digit = Int((k >> Scalar[key_dtype](shift)) & RADIX_MASK)
            var dst = Int(s_warp_hist[unsafe_offset=hist_row + digit] + ranks[r])
            keys_out[unsafe_offset=dst] = k
            vals_out[unsafe_offset=dst] = vals[r]


def radix_sort_pairs[
    key_dtype: DType = DType.uint64
](
    ctx: DeviceContext,
    stream: DeviceStream,
    keys_a: Pointer[Scalar[key_dtype], MutUntrackedOrigin],
    vals_a: I32Ptr,
    keys_b: Pointer[Scalar[key_dtype], MutUntrackedOrigin],
    vals_b: I32Ptr,
    hist: U32Ptr,
    partials: U32Ptr,
    total: U32Ptr,
    n: Int,
    end_bit: Int,
) raises -> Bool:
    """Sort (keys_a, vals_a) by the low `end_bit` bits of the key.

    (keys_b, vals_b) is a same-sized ping-pong buffer. Scratch sizes:
    `hist` >= sort_hist_size(n), `partials` >= sort_partials_size(n),
    `total` >= 1. Returns True if the sorted result ended up in the `b`
    buffers, False if in `a`.
    """
    comptime assert key_dtype in (DType.uint16, DType.uint32, DType.uint64), "keys must be UInt16, UInt32 or UInt64"
    if n <= 1:
        return False
    var num_parts = sort_num_parts(n)
    var passes = (end_bit + RADIX_BITS - 1) // RADIX_BITS
    var up = ctx.compile_function[_upsweep_kernel[key_dtype]]()
    var down = ctx.compile_function[_downsweep_kernel[key_dtype]]()
    for p in range(passes):
        var shift = UInt32(p * RADIX_BITS)
        var src_k = keys_a if p % 2 == 0 else keys_b
        var src_v = vals_a if p % 2 == 0 else vals_b
        var dst_k = keys_b if p % 2 == 0 else keys_a
        var dst_v = vals_b if p % 2 == 0 else vals_a
        stream.enqueue_function(
            up,
            src_k,
            hist,
            Int32(n),
            Int32(num_parts),
            shift,
            grid_dim=num_parts,
            block_dim=SORT_BLOCK,
        )
        exclusive_scan[DType.uint32](ctx, stream, hist, hist, total, partials, RADIX * num_parts)
        stream.enqueue_function(
            down,
            src_k,
            src_v,
            dst_k,
            dst_v,
            hist,
            Int32(n),
            Int32(num_parts),
            shift,
            grid_dim=num_parts,
            block_dim=SORT_BLOCK,
        )
    return passes % 2 == 1
