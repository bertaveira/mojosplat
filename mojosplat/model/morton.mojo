"""Z-order (Morton) reordering of the Gaussians.

Sorting the Gaussians along a space-filling curve makes neighbours in space
neighbours in memory: warps in the preprocessing kernels see coherent
(mostly all-visible or all-culled) Gaussians, and the per-Gaussian backward
gathers and atomics hit nearby addresses. Faster-GS applies it every 5000
iterations during densification (its largest single speedup together with
the per-Gaussian backward).

Encoding as in NeRFICG's MortonEncoding (MIT; after Schuetz et al.): means
normalized into their bounding cube, 21 bits per axis interleaved into a
63-bit key.
"""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext

from ..core.types import BLOCK_SIZE, F32Ptr, I32Ptr, U64Ptr, dptr, num_blocks
from ..gpu.reduce import bounds3
from ..gpu.sort import radix_sort_pairs, sort_hist_size, sort_partials_size
from .splats import Attr, Splats


@always_inline
def _split_by_3(a: UInt32) -> UInt64:
    var x = UInt64(a) & 0x1FFFFF
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def _morton_kernel(
    means: F32Ptr, keys: U64Ptr, values: I32Ptr, lo: SIMD[DType.float32, 4], inv_size: Float32, n: Int32
):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i >= Int(n):
        return
    var code = UInt64(0)
    comptime for k in range(3):
        var t = min(max((means[unsafe_offset=i * 3 + k] - lo[k]) * inv_size, 0.0), 1.0)
        code |= _split_by_3(UInt32(t * 2097151.0)) << UInt64(k)
    keys[unsafe_offset=i] = code
    values[unsafe_offset=i] = Int32(i)


def morton_order(mut splats: Splats) raises:
    """Permute all per-Gaussian arrays (and their Adam state) into Z-order."""
    var n = splats.count
    if n < 2:
        return
    var ctx = splats.ctx
    var stream = ctx.stream()
    var b = bounds3(ctx, dptr(splats.params[Attr.MEANS]), n)
    var lo = b[0]
    var size = max(b[1][0] - lo[0], max(b[1][1] - lo[1], b[1][2] - lo[2]))
    var keys_a = ctx.enqueue_create_buffer[DType.uint64](n)
    var keys_b = ctx.enqueue_create_buffer[DType.uint64](n)
    var vals_a = ctx.enqueue_create_buffer[DType.int32](n)
    var vals_b = ctx.enqueue_create_buffer[DType.int32](n)
    var hist = ctx.enqueue_create_buffer[DType.uint32](sort_hist_size(n))
    var partials = ctx.enqueue_create_buffer[DType.uint32](sort_partials_size(n))
    var total = ctx.enqueue_create_buffer[DType.uint32](1)
    stream.enqueue_function(
        ctx.compile_function[_morton_kernel](),
        dptr(splats.params[Attr.MEANS]),
        dptr(keys_a),
        dptr(vals_a),
        lo,
        Float32(1.0) / max(size, Float32(1e-20)),
        Int32(n),
        grid_dim=num_blocks(n),
        block_dim=BLOCK_SIZE,
    )
    var in_b = radix_sort_pairs[DType.uint64](
        ctx,
        stream,
        dptr(keys_a),
        dptr(vals_a),
        dptr(keys_b),
        dptr(vals_b),
        dptr(hist),
        dptr(partials),
        dptr(total),
        n,
        63,
    )
    splats.gather(dptr(vals_b) if in_b else dptr(vals_a), n, n)
    ctx.synchronize()
    _ = (keys_a^, keys_b^, vals_a^, vals_b^, hist^, partials^, total^)
