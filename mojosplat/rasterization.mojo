"""Tile-based front-to-back alpha compositing (one thread per pixel).

Matches gsplat's `rasterize_to_pixels_3dgs_fwd`: each block owns a
tile_size x tile_size tile, cooperatively streams that tile's depth-sorted
Gaussians through shared memory in batches of tile_size^2, and stops once
every pixel in the tile is saturated (T <= 1e-4).
"""

from max.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.atomic import Atomic, Ordering
from std.math import exp
from std.memory import stack_allocation

from .common import F32Ptr, I32Ptr

comptime ALPHA_THRESHOLD: Float32 = 1.0 / 255.0
comptime T_MIN: Float32 = 1e-4
comptime MAX_ALPHA: Float32 = 0.999


def _rasterize_kernel[
    tile_size: Int, CDIM: Int
](
    means2d: F32Ptr,      # [N, 2]
    conics: F32Ptr,       # [N, 3]
    colors: F32Ptr,       # [N, CDIM]
    opacities: F32Ptr,    # [N]
    background: SIMD[DType.float32, 4],
    tile_offsets: I32Ptr,  # [n_tiles + 1]
    flatten_ids: I32Ptr,  # [M] sorted gaussian ids
    image: F32Ptr,        # out [H, W, CDIM]
    width: Int32,
    height: Int32,
    tiles_x: Int32,
):
    comptime assert CDIM <= 4, "background is packed into a SIMD[4]"
    comptime BATCH = tile_size * tile_size
    var sh_ids = stack_allocation[BATCH, Int32, address_space=AddressSpace.SHARED]()
    var sh_xy = stack_allocation[BATCH * 2, Float32, address_space=AddressSpace.SHARED]()
    var sh_conic = stack_allocation[BATCH * 3, Float32, address_space=AddressSpace.SHARED]()
    var sh_opac = stack_allocation[BATCH, Float32, address_space=AddressSpace.SHARED]()
    var sh_done = stack_allocation[1, Int32, address_space=AddressSpace.SHARED]()

    var tile_row = block_idx.y
    var tile_col = block_idx.x
    var tid = thread_idx.y * tile_size + thread_idx.x
    var i = tile_row * tile_size + thread_idx.y
    var j = tile_col * tile_size + thread_idx.x
    var px = Float32(j) + 0.5
    var py = Float32(i) + 0.5

    var inside = i < Int(height) and j < Int(width)
    var done = not inside

    var tile_id = tile_row * Int(tiles_x) + tile_col
    var range_start = Int(tile_offsets[unsafe_offset=tile_id])
    var range_end = Int(tile_offsets[unsafe_offset=tile_id + 1])
    var num_batches = (range_end - range_start + BATCH - 1) // BATCH

    var T: Float32 = 1.0
    var pix = SIMD[DType.float32, 4](0.0)

    if tid == 0:
        sh_done[unsafe_offset=0] = 0
    barrier()

    for b in range(num_batches):
        var batch_start = range_start + BATCH * b
        # Saturated pixels report in; every thread fetches one Gaussian.
        if done:
            _ = Atomic[Int32].fetch_add[ordering=Ordering.RELAXED](sh_done, 1)
        var idx = batch_start + tid
        if idx < range_end:
            var g = Int(flatten_ids[unsafe_offset=idx])
            sh_ids[unsafe_offset=tid] = Int32(g)
            sh_xy[unsafe_offset=tid * 2 + 0] = means2d[unsafe_offset=g * 2 + 0]
            sh_xy[unsafe_offset=tid * 2 + 1] = means2d[unsafe_offset=g * 2 + 1]
            sh_conic[unsafe_offset=tid * 3 + 0] = conics[unsafe_offset=g * 3 + 0]
            sh_conic[unsafe_offset=tid * 3 + 1] = conics[unsafe_offset=g * 3 + 1]
            sh_conic[unsafe_offset=tid * 3 + 2] = conics[unsafe_offset=g * 3 + 2]
            sh_opac[unsafe_offset=tid] = opacities[unsafe_offset=g]
        barrier()

        if Int(sh_done[unsafe_offset=0]) >= BATCH:
            break

        if not done:
            var batch_size = min(BATCH, range_end - batch_start)
            for t in range(batch_size):
                var dx = sh_xy[unsafe_offset=t * 2 + 0] - px
                var dy = sh_xy[unsafe_offset=t * 2 + 1] - py
                var sigma = 0.5 * (
                    sh_conic[unsafe_offset=t * 3 + 0] * dx * dx
                    + sh_conic[unsafe_offset=t * 3 + 2] * dy * dy
                ) + sh_conic[unsafe_offset=t * 3 + 1] * dx * dy
                var alpha = min(sh_opac[unsafe_offset=t] * exp(-sigma), MAX_ALPHA)
                if sigma >= 0.0 and alpha >= ALPHA_THRESHOLD:
                    var next_T = T * (1.0 - alpha)
                    if next_T <= T_MIN:
                        done = True
                        break
                    var vis = alpha * T
                    var g = Int(sh_ids[unsafe_offset=t])
                    comptime for c in range(CDIM):
                        pix[c] += colors[unsafe_offset=g * CDIM + c] * vis
                    T = next_T

        if tid == 0:
            sh_done[unsafe_offset=0] = 0
        barrier()

    if inside:
        var base = (i * Int(width) + j) * CDIM
        comptime for c in range(CDIM):
            image[unsafe_offset=base + c] = pix[c] + T * background[c]


def rasterize[
    tile_size: Int, CDIM: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    means2d: F32Ptr,
    conics: F32Ptr,
    colors: F32Ptr,
    opacities: F32Ptr,
    background: SIMD[DType.float32, 4],
    tile_offsets: I32Ptr,
    flatten_ids: I32Ptr,
    image: F32Ptr,
    width: Int,
    height: Int,
) raises:
    var tiles_x = (width + tile_size - 1) // tile_size
    var tiles_y = (height + tile_size - 1) // tile_size
    stream.enqueue_function(
        ctx.compile_function[_rasterize_kernel[tile_size, CDIM]](),
        means2d, conics, colors, opacities, background, tile_offsets,
        flatten_ids, image, Int32(width), Int32(height), Int32(tiles_x),
        grid_dim=(tiles_x, tiles_y), block_dim=(tile_size, tile_size),
    )
