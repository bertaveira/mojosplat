import compiler
from gpu import thread_idx, block_idx
from math import floor, ceil, ceildiv
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer

comptime block_size: Int = 256


fn isect_offset_kernel(
    # Sorted packed keys: (tile_id << 32) | depth_u32 per intersection
    sorted_keys_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],  # [M]
    offsets_ptr: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],      # [n_tiles + 1]
    M: Int,        # number of intersections
    n_tiles: Int,  # tile_width * tile_height
):
    """Compute per-tile start offsets from sorted intersection keys.

    Port of gsplat's `intersect_offset_kernel` (IntersectTile.cu:209). For each
    intersection, look at (sorted_keys[idx-1] >> 32) vs (sorted_keys[idx] >> 32)
    — a change in tile_id means `idx` is the start of a new tile run, and every
    tile between the previous one and this one is empty (start = idx).

    One thread per intersection; boundary threads also fill trailing empty
    tiles. After this kernel, offsets[t] = first index in sorted_keys where
    tile_id == t (or end-of-array if the tile is empty).
    """
    var idx = block_idx.x * block_size + thread_idx.x
    if idx >= M:
        return

    var tid_curr = Int(sorted_keys_ptr[idx] >> 32)

    if idx == 0:
        # Fill offsets[0 .. tid_curr] = 0 (tiles before the first non-empty one).
        for i in range(0, tid_curr + 1):
            offsets_ptr[i] = 0
        # If this is also the last thread (M == 1), fill the trailing region.
        if M == 1:
            for i in range(tid_curr + 1, n_tiles + 1):
                offsets_ptr[i] = Int32(M)
        return

    if idx == M - 1:
        # Fill offsets[tid_curr+1 .. n_tiles] = M (tiles after the last one).
        for i in range(tid_curr + 1, n_tiles + 1):
            offsets_ptr[i] = Int32(M)
        # Also handle the boundary at idx (falls through to the boundary write).

    var tid_prev = Int(sorted_keys_ptr[idx - 1] >> 32)
    if tid_prev != tid_curr:
        # We crossed from tile tid_prev to tid_curr. Every tile boundary in
        # (tid_prev, tid_curr] starts at position idx.
        for i in range(tid_prev + 1, tid_curr + 1):
            offsets_ptr[i] = Int32(idx)


fn isect_count_kernel[tile_size: Int](
    means2d_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],  # [N * 2]
    radii_ptr: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],      # [N * 2]
    tile_dims_ptr: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],  # [2]: [tile_width, tile_height]
    tiles_per_gauss_ptr: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],  # [N]
    N: Int,
):
    var g = block_idx.x * block_size + thread_idx.x
    if g >= N:
        return

    var radius_x = Int(radii_ptr[g * 2 + 0])
    var radius_y = Int(radii_ptr[g * 2 + 1])

    if radius_x <= 0 or radius_y <= 0:
        tiles_per_gauss_ptr[g] = 0
        return

    var tile_width = Int(tile_dims_ptr[0])
    var tile_height = Int(tile_dims_ptr[1])
    var mean_x = means2d_ptr[g * 2 + 0]
    var mean_y = means2d_ptr[g * 2 + 1]

    var tile_x = mean_x / Float32(tile_size)
    var tile_y = mean_y / Float32(tile_size)
    var tile_rx = Float32(radius_x) / Float32(tile_size)
    var tile_ry = Float32(radius_y) / Float32(tile_size)

    var tile_min_x = Int(min(max(Float32(0.0), floor(tile_x - tile_rx)), Float32(tile_width)))
    var tile_min_y = Int(min(max(Float32(0.0), floor(tile_y - tile_ry)), Float32(tile_height)))
    var tile_max_x = Int(min(max(Float32(0.0),  ceil(tile_x + tile_rx)), Float32(tile_width)))
    var tile_max_y = Int(min(max(Float32(0.0),  ceil(tile_y + tile_ry)), Float32(tile_height)))

    tiles_per_gauss_ptr[g] = Int32((tile_max_y - tile_min_y) * (tile_max_x - tile_min_x))


fn isect_write_kernel[tile_size: Int](
    means2d_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],   # [N * 2]
    radii_ptr: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],       # [N * 2]
    depths_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],    # [N]
    offsets_ptr: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],     # [N] exclusive prefix sum
    tile_dims_ptr: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],   # [2]: [tile_width, tile_height]
    isect_ids_ptr: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],   # [M]
    flatten_ids_ptr: UnsafePointer[Scalar[DType.int32], MutAnyOrigin], # [M]
    N: Int,
    M: Int,
):
    var g = block_idx.x * block_size + thread_idx.x
    if g >= N:
        return

    var radius_x = Int(radii_ptr[g * 2 + 0])
    var radius_y = Int(radii_ptr[g * 2 + 1])

    if radius_x <= 0 or radius_y <= 0:
        return

    var tile_width = Int(tile_dims_ptr[0])
    var tile_height = Int(tile_dims_ptr[1])
    var mean_x = means2d_ptr[g * 2 + 0]
    var mean_y = means2d_ptr[g * 2 + 1]

    var tile_x = mean_x / Float32(tile_size)
    var tile_y = mean_y / Float32(tile_size)
    var tile_rx = Float32(radius_x) / Float32(tile_size)
    var tile_ry = Float32(radius_y) / Float32(tile_size)

    var tile_min_x = Int(min(max(Float32(0.0), floor(tile_x - tile_rx)), Float32(tile_width)))
    var tile_min_y = Int(min(max(Float32(0.0), floor(tile_y - tile_ry)), Float32(tile_height)))
    var tile_max_x = Int(min(max(Float32(0.0),  ceil(tile_x + tile_rx)), Float32(tile_width)))
    var tile_max_y = Int(min(max(Float32(0.0),  ceil(tile_y + tile_ry)), Float32(tile_height)))

    var cur_idx = Int(offsets_ptr[g])

    # Reinterpret float depth bits as uint32 for monotone integer sort
    var depth_u32 = (depths_ptr + g).bitcast[Scalar[DType.uint32]]()[0].cast[DType.int64]()

    for ty in range(tile_min_y, tile_max_y):
        for tx in range(tile_min_x, tile_max_x):
            var tile_id = Int64(ty * tile_width + tx)
            isect_ids_ptr[cur_idx] = (tile_id << 32) | depth_u32
            flatten_ids_ptr[cur_idx] = Int32(g)
            cur_idx += 1


# --------------------------------------------------------------------------
# MAX Engine Kernel Definitions
# --------------------------------------------------------------------------

@compiler.register("isect_count")
struct IsectCount:
    @staticmethod
    fn execute[
        tile_size: Int,
        target: StaticString,
    ](
        # Dummy DPS output (1 scalar) — prevents dead code elimination
        dummy: OutputTensor[dtype=DType.float32, rank=1, ...],
        # "Output" passed as InputTensor (written in-place via rebind)
        tiles_per_gauss: InputTensor[dtype=DType.int32, rank=1, ...],  # (N,)
        # Inputs
        means2d:   InputTensor[dtype=DType.float32, rank=2, ...],  # (N, 2)
        radii:     InputTensor[dtype=DType.int32,   rank=2, ...],  # (N, 2)
        tile_dims: InputTensor[dtype=DType.int32,   rank=1, ...],  # (2,)
        ctx: DeviceContextPtr
    ) raises:
        var N = means2d.dim_size(0)

        var means2d_ptr         = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](means2d.to_layout_tensor().ptr)
        var radii_ptr           = rebind[UnsafePointer[Scalar[DType.int32],   MutAnyOrigin]](radii.to_layout_tensor().ptr)
        var tile_dims_ptr       = rebind[UnsafePointer[Scalar[DType.int32],   MutAnyOrigin]](tile_dims.to_layout_tensor().ptr)
        var tiles_per_gauss_ptr = rebind[UnsafePointer[Scalar[DType.int32],   MutAnyOrigin]](tiles_per_gauss.to_layout_tensor().ptr)

        @parameter
        if target == "cpu":
            raise Error("IsectCount CPU target not implemented.")
        elif target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var grid = (ceildiv(N, block_size),)
            var block = (block_size,)
            gpu_ctx.enqueue_function_unchecked[isect_count_kernel[tile_size]](
                means2d_ptr, radii_ptr, tile_dims_ptr, tiles_per_gauss_ptr,
                N,
                grid_dim=grid, block_dim=block,
            )
        else:
            raise Error("Unsupported target:", target)


@compiler.register("isect_offset")
struct IsectOffset:
    """Port of gsplat's `intersect_offset_kernel` — replaces Python-side
    `searchsorted(arange(n_tiles+1))` + `torch.stack` with a single Mojo
    kernel. n_tiles is inferred from the offsets tensor length (n_tiles+1)."""

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        # Dummy DPS output (1 scalar) — prevents dead code elimination
        dummy: OutputTensor[dtype=DType.float32, rank=1, ...],
        # "Output" passed as InputTensor (written in-place via rebind)
        offsets: InputTensor[dtype=DType.int32, rank=1, ...],     # (n_tiles + 1,)
        # Inputs
        sorted_keys: InputTensor[dtype=DType.int64, rank=1, ...], # (M,)
        ctx: DeviceContextPtr
    ) raises:
        var M = sorted_keys.dim_size(0)
        var n_tiles = offsets.dim_size(0) - 1

        var sorted_keys_ptr = rebind[UnsafePointer[Scalar[DType.int64], MutAnyOrigin]](sorted_keys.to_layout_tensor().ptr)
        var offsets_ptr     = rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](offsets.to_layout_tensor().ptr)

        @parameter
        if target == "cpu":
            raise Error("IsectOffset CPU target not implemented.")
        elif target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var grid = (ceildiv(M, block_size),)
            var block = (block_size,)
            gpu_ctx.enqueue_function_unchecked[isect_offset_kernel](
                sorted_keys_ptr, offsets_ptr, M, n_tiles,
                grid_dim=grid, block_dim=block,
            )
        else:
            raise Error("Unsupported target:", target)


@compiler.register("isect_write")
struct IsectWrite:
    @staticmethod
    fn execute[
        tile_size: Int,
        target: StaticString,
    ](
        # Dummy DPS output (1 scalar) — prevents dead code elimination
        dummy: OutputTensor[dtype=DType.float32, rank=1, ...],
        # "Outputs" passed as InputTensor (written in-place via rebind)
        isect_ids:   InputTensor[dtype=DType.int64, rank=1, ...],  # (M,)
        flatten_ids: InputTensor[dtype=DType.int32, rank=1, ...],  # (M,)
        # Inputs
        means2d:   InputTensor[dtype=DType.float32, rank=2, ...],  # (N, 2)
        radii:     InputTensor[dtype=DType.int32,   rank=2, ...],  # (N, 2)
        depths:    InputTensor[dtype=DType.float32, rank=1, ...],  # (N,)
        offsets:   InputTensor[dtype=DType.int32,   rank=1, ...],  # (N,)
        tile_dims: InputTensor[dtype=DType.int32,   rank=1, ...],  # (2,)
        ctx: DeviceContextPtr
    ) raises:
        var N = means2d.dim_size(0)
        var M = isect_ids.dim_size(0)

        var means2d_ptr     = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](means2d.to_layout_tensor().ptr)
        var radii_ptr       = rebind[UnsafePointer[Scalar[DType.int32],   MutAnyOrigin]](radii.to_layout_tensor().ptr)
        var depths_ptr      = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](depths.to_layout_tensor().ptr)
        var offsets_ptr     = rebind[UnsafePointer[Scalar[DType.int32],   MutAnyOrigin]](offsets.to_layout_tensor().ptr)
        var tile_dims_ptr   = rebind[UnsafePointer[Scalar[DType.int32],   MutAnyOrigin]](tile_dims.to_layout_tensor().ptr)
        var isect_ids_ptr   = rebind[UnsafePointer[Scalar[DType.int64],   MutAnyOrigin]](isect_ids.to_layout_tensor().ptr)
        var flatten_ids_ptr = rebind[UnsafePointer[Scalar[DType.int32],   MutAnyOrigin]](flatten_ids.to_layout_tensor().ptr)

        @parameter
        if target == "cpu":
            raise Error("IsectWrite CPU target not implemented.")
        elif target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var grid = (ceildiv(N, block_size),)
            var block = (block_size,)
            gpu_ctx.enqueue_function_unchecked[isect_write_kernel[tile_size]](
                means2d_ptr, radii_ptr, depths_ptr, offsets_ptr, tile_dims_ptr,
                isect_ids_ptr, flatten_ids_ptr,
                N, M,
                grid_dim=grid, block_dim=block,
            )
        else:
            raise Error("Unsupported target:", target)
