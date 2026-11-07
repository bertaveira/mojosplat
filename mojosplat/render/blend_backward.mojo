"""Per-Gaussian blend backward (Faster-GS / Taming-3DGS).

Ported from Faster-GS (Hahlbohm et al., CVPR 2026; Apache-2.0,
github.com/nerficg-project/faster-gaussian-splatting, `kernels_backward.cuh`),
adapted to MojoSplat's compacted slots and HWC images.

One warp per bucket of 32 consecutive instances of a tile; lane k owns the
k-th Gaussian. The tile's pixels stream through the warp like a systolic
array: lane 0 picks up a pixel's saved state ((color, T) before the bucket,
from the training forward), applies its Gaussian and hands the updated state
to lane 1 with a shuffle, and so on. Each lane accumulates its Gaussian's
gradients in registers over all pixels and issues one set of atomics at the
end (instead of one per pixel as in a per-pixel backward).

Outputs per visible slot: dL/d(mean2d), dL/d(conic), dL/d(opacity logit),
dL/d(color). Conic off-diagonal gradients are per entry of the symmetric
matrix (half of the gradient w.r.t. the single value), as in Faster-GS.
"""

from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.host import DeviceContext, DeviceStream
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from max.gpu.sync import syncwarp
from std.atomic import Atomic, Ordering
from std.math import exp2, recip
from std.memory import stack_allocation

from ..core.types import F32Ptr, I32Ptr, KernelVector, U32Ptr
from .preprocess import MIN_ALPHA, load_record

comptime ONE_MINUS_ALPHA_EPS: Float32 = 1e-6
comptime LOG2E: Float32 = 1.4426950408889634


@always_inline
def _atomic_add(p: F32Ptr, i: Int, v: Float32):
    _ = Atomic[Float32].fetch_add[ordering=Ordering.RELAXED](p.unsafe_offset(i), v)


comptime BUCKETS_PER_BLOCK = 4  # one warp per bucket


def _blend_backward_kernel[
    tile_w: Int, tile_h: Int
](
    tile_offsets: U32Ptr,  # [n_tiles + 1]
    bucket_start: U32Ptr,  # [n_tiles + 1]
    bucket_tile: U32Ptr,  # [n_buckets]
    tile_max_n: U32Ptr,  # [n_tiles]
    instance_ids: I32Ptr,  # [M] slots
    records: F32Ptr,  # [slot, REC]
    colors: F32Ptr,  # [slot, 3] unclamped
    background_arg: KernelVector[DType.float32, 4],
    image: F32Ptr,  # [H, W, 3] forward output
    grad_image: F32Ptr,  # [H, W, 3]
    final_T: F32Ptr,  # [H * W]
    n_processed: U32Ptr,  # [H * W]
    snapshots: F32Ptr,  # [n_buckets, BLOCK, 4]
    grad_mean2d: F32Ptr,  # out (accumulated) [slot, 2]
    grad_conic: F32Ptr,  # out (accumulated) [slot, 3]
    grad_opacity: F32Ptr,  # out (accumulated) [slot] w.r.t. the opacity logit
    grad_color: F32Ptr,  # out (accumulated) [slot, 3] w.r.t. the unclamped color
    n_tiles: Int32,
    width: Int32,
    height: Int32,
    grid_w: Int32,
):
    var background = background_arg.value
    comptime BLOCK = tile_w * tile_h
    var warp_in_block = thread_idx.x // WARP_SIZE
    var bucket = block_idx.x * BUCKETS_PER_BLOCK + warp_in_block
    if bucket >= Int(bucket_start[unsafe_offset=Int(n_tiles)]):
        return
    var tile = Int(bucket_tile[unsafe_offset=bucket])
    var start = Int(tile_offsets[unsafe_offset=tile])
    var count = Int(tile_offsets[unsafe_offset=tile + 1]) - start
    var tile_bucket = bucket - Int(bucket_start[unsafe_offset=tile])
    if tile_bucket * WARP_SIZE >= Int(tile_max_n[unsafe_offset=tile]):
        return

    var lane = lane_id()
    var tp = tile_bucket * WARP_SIZE + lane  # this lane's Gaussian, tile-local
    var valid = tp < count
    var g = 0
    var mx: Float32 = 0
    var my: Float32 = 0
    var ca: Float32 = 0
    var cb: Float32 = 0
    var cc: Float32 = 0
    var opacity: Float32 = 0
    var col = SIMD[DType.float32, 4](0.0)
    var col_factor = SIMD[DType.float32, 4](0.0)
    if valid:
        g = Int(instance_ids[unsafe_offset=start + tp])
        var rec = load_record(records, g)
        mx = rec[0]
        my = rec[1]
        ca = rec[2]
        cb = rec[3]
        cc = rec[4]
        opacity = rec[5]
        comptime for k in range(3):
            var c = colors[unsafe_offset=g * 3 + k]
            col[k] = max(c, 0.0)
            col_factor[k] = 1.0 if c >= 0.0 else 0.0
    var ea = -0.5 * LOG2E * ca
    var eb = -LOG2E * cb
    var ec = -0.5 * LOG2E * cc

    var tile_x0 = (tile % Int(grid_w)) * tile_w
    var tile_y0 = (tile // Int(grid_w)) * tile_h
    var W = Int(width)
    var H = Int(height)

    var s_last = stack_allocation[
        BUCKETS_PER_BLOCK * WARP_SIZE, UInt32, address_space=AddressSpace.SHARED
    ]().unsafe_offset(warp_in_block * WARP_SIZE)
    var s_after = stack_allocation[
        BUCKETS_PER_BLOCK * WARP_SIZE * 4, Float32, address_space=AddressSpace.SHARED
    ]().unsafe_offset(warp_in_block * WARP_SIZE * 4)
    var s_grad = stack_allocation[
        BUCKETS_PER_BLOCK * WARP_SIZE * 4, Float32, address_space=AddressSpace.SHARED
    ]().unsafe_offset(warp_in_block * WARP_SIZE * 4)
    var snap = snapshots.unsafe_offset(bucket * BLOCK * 4)

    # Gradient accumulators for this lane's Gaussian.
    var d_mx: Float32 = 0
    var d_my: Float32 = 0
    var d_ca: Float32 = 0
    var d_cb: Float32 = 0
    var d_cc: Float32 = 0
    var d_op: Float32 = 0
    var d_col = SIMD[DType.float32, 4](0.0)

    # Pixel state handed from lane to lane.
    var last = UInt32(0)
    var after_r: Float32 = 0  # color still to come from this Gaussian onwards
    var after_g: Float32 = 0
    var after_b: Float32 = 0
    var T: Float32 = 0
    var gr: Float32 = 0  # dL/d(pixel color)
    var gg: Float32 = 0
    var gb: Float32 = 0
    var g_alpha_bg: Float32 = 0  # -T_final * dot(dL/dpixel, background)

    # 32-bit loop state: 64-bit integer math and int -> float conversions are
    # several times slower on Apple GPUs.
    var lane32 = Int32(lane)
    var tp32 = UInt32(tp)
    var W32 = Int32(W)
    var H32 = Int32(H)
    var x0_32 = Int32(tile_x0)
    var y0_32 = Int32(tile_y0)
    var i = Int32(-1)
    while True:
        i += 1
        if i >= Int32(BLOCK + WARP_SIZE - 1):
            break
        var k32 = i & Int32(WARP_SIZE - 1)
        if k32 == 0:
            syncwarp()
            var local32 = i + lane32
            if local32 < Int32(BLOCK):
                var local = Int(local32)
                var x = Int(x0_32 + Int32(UInt32(local32) % UInt32(tile_w)))
                var y = Int(y0_32 + Int32(UInt32(local32) // UInt32(tile_w)))
                var s = snap.unsafe_offset(local * 4).unsafe_bitcast[SIMD[DType.float32, 4]]()[]
                var img = SIMD[DType.float32, 4](0.0)
                var gimg = SIMD[DType.float32, 4](0.0)
                var fT: Float32 = 0
                var lst = UInt32(0)
                if x < W and y < H:
                    var p = y * W + x
                    comptime for k in range(3):
                        img[k] = image[unsafe_offset=p * 3 + k]
                        gimg[k] = grad_image[unsafe_offset=p * 3 + k]
                    fT = final_T[unsafe_offset=p]
                    lst = n_processed[unsafe_offset=p]
                comptime for k in range(3):
                    s_after[unsafe_offset=lane * 4 + k] = img[k] - fT * background[k] - s[k]
                    s_grad[unsafe_offset=lane * 4 + k] = gimg[k]
                s_after[unsafe_offset=lane * 4 + 3] = s[3]
                s_grad[unsafe_offset=lane * 4 + 3] = -fT * (
                    gimg[0] * background[0] + gimg[1] * background[1] + gimg[2] * background[2]
                )
                s_last[unsafe_offset=lane] = lst
            syncwarp()

        if i > 0:
            last = warp.shuffle_up(last, UInt32(1))
            after_r = warp.shuffle_up(after_r, UInt32(1))
            after_g = warp.shuffle_up(after_g, UInt32(1))
            after_b = warp.shuffle_up(after_b, UInt32(1))
            T = warp.shuffle_up(T, UInt32(1))
            gr = warp.shuffle_up(gr, UInt32(1))
            gg = warp.shuffle_up(gg, UInt32(1))
            gb = warp.shuffle_up(gb, UInt32(1))
            g_alpha_bg = warp.shuffle_up(g_alpha_bg, UInt32(1))

        var idx = i - lane32  # pixel this lane handles now
        if lane32 == 0 and i < Int32(BLOCK):
            var k = Int(k32)
            last = s_last[unsafe_offset=k]
            after_r = s_after[unsafe_offset=k * 4 + 0]
            after_g = s_after[unsafe_offset=k * 4 + 1]
            after_b = s_after[unsafe_offset=k * 4 + 2]
            T = s_after[unsafe_offset=k * 4 + 3]
            gr = s_grad[unsafe_offset=k * 4 + 0]
            gg = s_grad[unsafe_offset=k * 4 + 1]
            gb = s_grad[unsafe_offset=k * 4 + 2]
            g_alpha_bg = s_grad[unsafe_offset=k * 4 + 3]

        if not valid or idx < 0 or idx >= Int32(BLOCK) or tp32 >= last:
            continue
        var x = x0_32 + Int32(UInt32(idx) % UInt32(tile_w))
        var y = y0_32 + Int32(UInt32(idx) // UInt32(tile_w))
        if x >= W32 or y >= H32:
            continue

        var dx = mx - (Float32(x) + 0.5)
        var dy = my - (Float32(y) + 0.5)
        var power = dx * (ea * dx + eb * dy) + ec * dy * dy
        var gaussian = exp2(min(power, 0.0))
        var alpha = opacity * gaussian
        if alpha < MIN_ALPHA:
            continue
        var w = T * alpha
        d_col[0] += w * gr
        d_col[1] += w * gg
        d_col[2] += w * gb
        after_r -= w * col[0]
        after_g -= w * col[1]
        after_b -= w * col[2]
        var one_minus = 1.0 - alpha
        var rcp = recip(max(one_minus, ONE_MINUS_ALPHA_EPS))  # approximate, like Faster-GS's fast math
        var dL_dalpha = (
            (T * col[0] - after_r * rcp) * gr
            + (T * col[1] - after_g * rcp) * gg
            + (T * col[2] - after_b * rcp) * gb
            + g_alpha_bg * rcp
        )
        d_op += gaussian * dL_dalpha
        var helper = -alpha * dL_dalpha
        d_ca += helper * dx * dx  # times 0.5 after the loop
        d_cb += helper * dx * dy
        d_cc += helper * dy * dy
        d_mx += helper * (ca * dx + cb * dy)
        d_my += helper * (cb * dx + cc * dy)
        T *= one_minus

    if valid:
        _atomic_add(grad_mean2d, g * 2 + 0, d_mx)
        _atomic_add(grad_mean2d, g * 2 + 1, d_my)
        _atomic_add(grad_conic, g * 3 + 0, 0.5 * d_ca)
        _atomic_add(grad_conic, g * 3 + 1, 0.5 * d_cb)
        _atomic_add(grad_conic, g * 3 + 2, 0.5 * d_cc)
        _atomic_add(grad_opacity, g, opacity * (1.0 - opacity) * d_op)
        comptime for k in range(3):
            _atomic_add(grad_color, g * 3 + k, d_col[k] * col_factor[k])


def blend_backward[
    tile_w: Int, tile_h: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    n_bucket_blocks: Int,
    tile_offsets: U32Ptr,
    bucket_start: U32Ptr,
    bucket_tile: U32Ptr,
    tile_max_n: U32Ptr,
    instance_ids: I32Ptr,
    records: F32Ptr,
    colors: F32Ptr,
    background: SIMD[DType.float32, 4],
    image: F32Ptr,
    grad_image: F32Ptr,
    final_T: F32Ptr,
    n_processed: U32Ptr,
    snapshots: F32Ptr,
    grad_mean2d: F32Ptr,
    grad_conic: F32Ptr,
    grad_opacity: F32Ptr,
    grad_color: F32Ptr,
    n_tiles: Int,
    width: Int,
    height: Int,
) raises:
    var grid_w = (width + tile_w - 1) // tile_w
    stream.enqueue_function(
        ctx.compile_function[_blend_backward_kernel[tile_w, tile_h]](),
        tile_offsets,
        bucket_start,
        bucket_tile,
        tile_max_n,
        instance_ids,
        records,
        colors,
        KernelVector[DType.float32, 4](background),
        image,
        grad_image,
        final_T,
        n_processed,
        snapshots,
        grad_mean2d,
        grad_conic,
        grad_opacity,
        grad_color,
        Int32(n_tiles),
        Int32(width),
        Int32(height),
        Int32(grid_w),
        grid_dim=max((n_bucket_blocks + BUCKETS_PER_BLOCK - 1) // BUCKETS_PER_BLOCK, 1),
        block_dim=BUCKETS_PER_BLOCK * WARP_SIZE,
    )
