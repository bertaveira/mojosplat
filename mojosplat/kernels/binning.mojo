import compiler
from gpu import thread_idx, block_idx
from math import floor, ceil, ceildiv
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer

comptime block_size: Int = 256


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

    if radius_x <= 0 and radius_y <= 0:
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

    if radius_x <= 0 and radius_y <= 0:
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
        # Output
        tiles_per_gauss: OutputTensor[dtype=DType.int32, rank=1],  # (N,)
        # Inputs
        means2d:   InputTensor[dtype=DType.float32, rank=2],  # (N, 2)
        radii:     InputTensor[dtype=DType.int32,   rank=2],  # (N, 2)
        tile_dims: InputTensor[dtype=DType.int32,   rank=1],  # (2,)
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


@compiler.register("isect_write")
struct IsectWrite:
    @staticmethod
    fn execute[
        tile_size: Int,
        target: StaticString,
    ](
        # Outputs
        isect_ids:   OutputTensor[dtype=DType.int64, rank=1],  # (M,)
        flatten_ids: OutputTensor[dtype=DType.int32, rank=1],  # (M,)
        # Inputs
        means2d:   InputTensor[dtype=DType.float32, rank=2],  # (N, 2)
        radii:     InputTensor[dtype=DType.int32,   rank=2],  # (N, 2)
        depths:    InputTensor[dtype=DType.float32, rank=1],  # (N,)
        offsets:   InputTensor[dtype=DType.int32,   rank=1],  # (N,)
        tile_dims: InputTensor[dtype=DType.int32,   rank=1],  # (2,)
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
