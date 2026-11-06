"""Shared type aliases and launch helpers for MojoSplat kernels."""

from max.gpu import WARP_SIZE
from max.gpu.host import DeviceBuffer

comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime U32Ptr = Pointer[UInt32, MutUntrackedOrigin]
comptime U8Ptr = Pointer[UInt8, MutUntrackedOrigin]
comptime U16Ptr = Pointer[UInt16, MutUntrackedOrigin]
comptime U64Ptr = Pointer[UInt64, MutUntrackedOrigin]

# One bit per lane, as returned by `warp.vote` / `warp.match_any`.
comptime LANE_MASK_DTYPE = DType.uint32 if WARP_SIZE <= 32 else DType.uint64
comptime LaneMask = Scalar[LANE_MASK_DTYPE]

comptime _PLACEHOLDER_ADDR = 256


@always_inline
def placeholder[dtype: DType]() -> Pointer[Scalar[dtype], MutUntrackedOrigin]:
    """A non-null pointer that must never be dereferenced (Mojo pointers are
    non-nullable); marks unused optional buffers."""
    return Pointer[Scalar[dtype], MutUntrackedOrigin](unsafe_from_address=_PLACEHOLDER_ADDR)


# Default 1D launch block size for "one thread per element" kernels.
comptime BLOCK_SIZE = 256


@always_inline
def num_blocks(n: Int, block: Int = BLOCK_SIZE) -> Int:
    """Grid size covering `n` elements; at least 1 so empty launches are valid."""
    return max(1, (n + block - 1) // block)


@always_inline
def dptr[dtype: DType](buf: DeviceBuffer[dtype]) -> Pointer[Scalar[dtype], MutUntrackedOrigin]:
    """Raw device pointer of a DeviceBuffer, usable as a kernel argument."""
    return buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
