"""Front-to-back alpha blending of depth-sorted Gaussians per tile.

One block per tile_w x tile_h tile, one thread per pixel; each warp owns an
8 x (WARP_SIZE / 8) sub-tile (the sub-tiling of Faster-GS, Hahlbohm et al.
2026, Apache-2.0). Unlike Faster-GS's `blend_cu`, warps do not stream the
tile's Gaussians through block-wide shared-memory batches; every warp walks
the tile's list on its own:

  * each lane fetches one instance of a 32-wide chunk (one 32-byte record),
  * the warp votes on the ones whose screen rectangle overlaps its sub-tile,
  * those are shared through a warp-private shared-memory slab and
    composited by every lane.

There are no block barriers (previously the main stall: warps with little
work waited for the busiest one), duplicate fetches across a block's warps
hit L1, and a warp retires as soon as all of its pixels are saturated. On
an RTX 5090 this is ~20% faster than the batched version.

With `train=True` the kernel also saves what the per-Gaussian backward
needs (Faster-GS / Taming-3DGS): every pixel's (color, T) before each
bucket of 32 instances, its final T and the index of its last contributing
Gaussian, the per-tile maximum of that index, and each bucket's tile.

What a fragment contributes comes from the pipeline's `Response`
(blocks/contracts.mojo), which also lays out the slab. Semantics (as
Faster-GS / 3DGS): fragments with alpha < 1/255 are skipped; a pixel stops after the fragment that brings its
transmittance below 1e-4; colors are not clamped.
"""

from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from max.gpu.sync import barrier, syncwarp
from std.bit import count_trailing_zeros
from std.memory import stack_allocation

from ..core.types import F32Ptr, I32Ptr, KernelVector, LANE_MASK_DTYPE, U32Ptr, placeholder
from .blocks.contracts import Pipeline
from .pipelines import Standard3DGS
from .record import MIN_ALPHA, load_record, record_bounds, unpack_bounds

comptime T_MIN: Float32 = 1e-4
comptime SUBTILE_W = 8
comptime SUBTILE_H = WARP_SIZE // SUBTILE_W


def _blend_kernel[
    tile_w: Int, tile_h: Int, CDIM: Int, train: Bool, Pipe: Pipeline
](
    tile_offsets: U32Ptr,  # [n_tiles + 1]
    instance_ids: I32Ptr,  # [M] slots sorted by (tile, depth)
    records: F32Ptr,  # [slot, REC]
    colors: F32Ptr,  # [slot, CDIM] (clamped at 0 here)
    background_arg: KernelVector[DType.float32, 4],
    image: F32Ptr,  # out [H, W, CDIM]
    width: Int32,
    height: Int32,
    grid_w: Int32,
    # Training only (saved for the backward pass; ignored when train=False):
    bucket_start: U32Ptr,  # [n_tiles + 1] first bucket of each tile
    bucket_tile: U32Ptr,  # out [n_buckets] tile of each bucket
    snapshots: F32Ptr,  # out [n_buckets, BLOCK, 4] (color, T) before each bucket
    final_T: F32Ptr,  # out [H * W]
    n_processed: U32Ptr,  # out [H * W] tile-local index + 1 of the last used Gaussian
    tile_max_n: U32Ptr,  # out [n_tiles] max of n_processed over the tile
):
    var background = background_arg.value
    comptime BLOCK = tile_w * tile_h
    comptime WARPS = BLOCK // WARP_SIZE
    comptime SUBGRID_W = tile_w // SUBTILE_W
    comptime R = Pipe.Resp
    comptime STRIDE = R.SLAB  # floats per Gaussian in the slab
    comptime assert CDIM <= 3, "slab holds 3 color channels"
    comptime assert (
        tile_w % SUBTILE_W == 0 and tile_h % SUBTILE_H == 0
    ), "tile must split into 8 x (WARP_SIZE/8) sub-tiles"

    var slab_all = stack_allocation[WARPS * WARP_SIZE * STRIDE, Float32, address_space=AddressSpace.SHARED]()
    var rank = thread_idx.x
    var lane = lane_id()
    var warp_idx = rank // WARP_SIZE
    var slab = slab_all.unsafe_offset(warp_idx * WARP_SIZE * STRIDE)
    var tile_x0 = block_idx.x * tile_w
    var tile_y0 = block_idx.y * tile_h
    var sub_x0 = tile_x0 + (warp_idx % SUBGRID_W) * SUBTILE_W
    var sub_y0 = tile_y0 + (warp_idx // SUBGRID_W) * SUBTILE_H
    var sub_x1 = sub_x0 + SUBTILE_W
    var sub_y1 = sub_y0 + SUBTILE_H
    var x = sub_x0 + lane % SUBTILE_W
    var y = sub_y0 + lane // SUBTILE_W
    var local_idx = (y - tile_y0) * tile_w + (x - tile_x0)
    var inside = x < Int(width) and y < Int(height)
    var px = Float32(x) + 0.5
    var py = Float32(y) + 0.5

    var pix = SIMD[DType.float32, 4](0.0)
    var T: Float32 = 1.0
    var done = not inside
    var last = 0  # tile-local index + 1 of the last Gaussian that contributed

    var tile_id = block_idx.y * Int(grid_w) + block_idx.x
    var start = Int(tile_offsets[unsafe_offset=tile_id])
    var end = Int(tile_offsets[unsafe_offset=tile_id + 1])
    var first_bucket = 0
    comptime if train:
        first_bucket = Int(bucket_start[unsafe_offset=tile_id])
        var n_buckets = (end - start + WARP_SIZE - 1) // WARP_SIZE
        var b = rank
        while b < n_buckets:
            bucket_tile[unsafe_offset=first_bucket + b] = UInt32(tile_id)
            b += BLOCK

    var i = start
    while i < end:
        if warp.vote[LANE_MASK_DTYPE](not done) == 0:
            break
        comptime if train:
            if not done:
                var bucket = first_bucket + (i - start) // WARP_SIZE
                snapshots.unsafe_offset((bucket * BLOCK + local_idx) * 4).unsafe_store[alignment=16](
                    SIMD[DType.float32, 4](pix[0], pix[1], pix[2], T)
                )
        var fetch = i + lane
        var overlaps = False
        if fetch < end:
            var g = Int(instance_ids[unsafe_offset=fetch])
            var rec = load_record(records, g)
            var b = unpack_bounds(record_bounds(rec))
            overlaps = Int(b[0]) < sub_x1 and sub_x0 < Int(b[1]) and Int(b[2]) < sub_y1 and sub_y0 < Int(b[3])
            if overlaps:
                var c = SIMD[DType.float32, 4](0.0)
                comptime for k in range(CDIM):
                    c[k] = max(colors[unsafe_offset=g * CDIM + k], 0.0)
                R.store_slab(slab.unsafe_offset(lane * STRIDE), rec, c)
        var pending = warp.vote[LANE_MASK_DTYPE](overlaps)
        syncwarp()
        while not done and pending != 0:
            var j = Int(count_trailing_zeros(pending))
            pending &= pending - 1
            var src = slab.unsafe_offset(j * STRIDE)
            var alpha = R.alpha_at(src, px, py)
            if alpha < MIN_ALPHA:
                continue
            var weight = T * alpha
            comptime for k in range(CDIM):
                pix[k] += weight * R.channel(src, k)
            T *= 1.0 - alpha
            comptime if train:
                last = i - start + j + 1
            if T < T_MIN:
                done = True
        syncwarp()
        i += WARP_SIZE

    if inside:
        var p = y * Int(width) + x
        comptime for c in range(CDIM):
            image[unsafe_offset=p * CDIM + c] = pix[c] + T * background[c]
        comptime if train:
            final_T[unsafe_offset=p] = T
            n_processed[unsafe_offset=p] = UInt32(last)

    comptime if train:
        var sh_max = stack_allocation[WARPS, UInt32, address_space=AddressSpace.SHARED]()
        var m = warp.max(UInt32(last))
        if lane == 0:
            sh_max[unsafe_offset=warp_idx] = m
        barrier()
        if rank == 0:
            var mm = sh_max[unsafe_offset=0]
            comptime for w in range(1, WARPS):
                mm = max(mm, sh_max[unsafe_offset=w])
            tile_max_n[unsafe_offset=tile_id] = mm


def _bucket_count_kernel(tile_offsets: U32Ptr, counts: U32Ptr, n_tiles: Int32):
    var t = block_idx.x * 256 + thread_idx.x
    if t < Int(n_tiles):
        var n = tile_offsets[unsafe_offset=t + 1] - tile_offsets[unsafe_offset=t]
        counts[unsafe_offset=t] = (n + UInt32(WARP_SIZE - 1)) // UInt32(WARP_SIZE)


def bucket_counts(ctx: DeviceContext, stream: DeviceStream, tile_offsets: U32Ptr, counts: U32Ptr, n_tiles: Int) raises:
    """Buckets (32 instances each) per tile; scan these to get bucket_start."""
    stream.enqueue_function(
        ctx.compile_function[_bucket_count_kernel](),
        tile_offsets,
        counts,
        Int32(n_tiles),
        grid_dim=max(1, (n_tiles + 255) // 256),
        block_dim=256,
    )


def bucket_bound(n_instances: Int, n_tiles: Int) -> Int:
    """Upper bound on the number of buckets: sum ceil(c_t / 32) <= M / 32 + n_tiles."""
    return n_instances // WARP_SIZE + n_tiles + 1


def blend[
    tile_w: Int, tile_h: Int, CDIM: Int, train: Bool = False, Pipe: Pipeline = Standard3DGS
](
    ctx: DeviceContext,
    stream: DeviceStream,
    tile_offsets: U32Ptr,
    instance_ids: I32Ptr,
    records: F32Ptr,
    colors: F32Ptr,
    background: SIMD[DType.float32, 4],
    image: F32Ptr,
    width: Int,
    height: Int,
    bucket_start: U32Ptr = placeholder[DType.uint32](),
    bucket_tile: U32Ptr = placeholder[DType.uint32](),
    snapshots: F32Ptr = placeholder[DType.float32](),
    final_T: F32Ptr = placeholder[DType.float32](),
    n_processed: U32Ptr = placeholder[DType.uint32](),
    tile_max_n: U32Ptr = placeholder[DType.uint32](),
) raises:
    var grid_w = (width + tile_w - 1) // tile_w
    var grid_h = (height + tile_h - 1) // tile_h
    stream.enqueue_function(
        ctx.compile_function[_blend_kernel[tile_w, tile_h, CDIM, train, Pipe]](),
        tile_offsets,
        instance_ids,
        records,
        colors,
        KernelVector[DType.float32, 4](background),
        image,
        Int32(width),
        Int32(height),
        Int32(grid_w),
        bucket_start,
        bucket_tile,
        snapshots,
        final_T,
        n_processed,
        tile_max_n,
        grid_dim=(grid_w, grid_h),
        block_dim=tile_w * tile_h,
    )
