"""The per-slot record shared by the render stages, and constants they agree on.

Preprocess compacts each visible Gaussian into one 32-byte record: six floats
owned by the pipeline's `Response` (see blocks/contracts.mojo), then the pixel
rectangle as two floats of raw bits. One record is one memory sector per
random gather.
"""

from std.memory import bitcast

from ..core.types import F32Ptr

comptime MIN_ALPHA: Float32 = 1.0 / 255.0
comptime MIN_ALPHA_RCP: Float32 = 255.0
comptime N_SEQUENTIAL = 4  # tiles tested per thread before the warp helps

comptime REC = 8
comptime REC_BOUNDS = 6  # first of the two lanes holding the pixel rectangle


@always_inline
def pack_bounds(x0: Int, x1: Int, y0: Int, y1: Int) -> UInt64:
    """Pixel rectangle [x0, x1) x [y0, y1) as 4 x 16 bits."""
    return UInt64(x0) | (UInt64(x1) << 16) | (UInt64(y0) << 32) | (UInt64(y1) << 48)


@always_inline
def unpack_bounds(b: UInt64) -> SIMD[DType.int32, 4]:
    """-> (x0, x1, y0, y1)."""
    return SIMD[DType.int32, 4](Int32(b & 0xFFFF), Int32((b >> 16) & 0xFFFF), Int32((b >> 32) & 0xFFFF), Int32(b >> 48))


@always_inline
def store_record(rec: F32Ptr, slot: Int, var payload: SIMD[DType.float32, REC], bounds: UInt64):
    payload[REC_BOUNDS] = bitcast[DType.float32](UInt32(bounds & 0xFFFFFFFF))
    payload[REC_BOUNDS + 1] = bitcast[DType.float32](UInt32(bounds >> 32))
    # Records are 32-byte aligned (allocations are 256-byte aligned): one vector store.
    rec.unsafe_offset(slot * REC).unsafe_store[alignment=REC * 4](payload)


@always_inline
def load_record(rec: F32Ptr, slot: Int) -> SIMD[DType.float32, REC]:
    return rec.unsafe_offset(slot * REC).unsafe_load[width=REC, alignment=REC * 4]()


@always_inline
def record_bounds(r: SIMD[DType.float32, REC]) -> UInt64:
    return UInt64(r[REC_BOUNDS].to_bits()) | (UInt64(r[REC_BOUNDS + 1].to_bits()) << 32)


@always_inline
def tile_rect[tile_w: Int, tile_h: Int](b: UInt64) -> SIMD[DType.int32, 4]:
    """Pixel rectangle -> tile rectangle (tx0, tx1, ty0, ty1), exclusive max."""
    var p = unpack_bounds(b)
    return SIMD[DType.int32, 4](
        p[0] // Int32(tile_w),
        (p[1] + Int32(tile_w - 1)) // Int32(tile_w),
        p[2] // Int32(tile_h),
        (p[3] + Int32(tile_h - 1)) // Int32(tile_h),
    )
