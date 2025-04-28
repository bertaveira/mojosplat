from gpu import thread_idx, block_idx, warp, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace
from memory import stack_allocation
from layout import Layout, LayoutTensor
from math import iota
from sys import sizeof

from tensor import InputTensor, OutputTensor, ManagedTensorSlice


fn _rasterize_to_pixels_3dgs_fwd_kernel[
    tile_size: Int,
    tile_grid_width: Int,
    tile_grid_height: Int,
    # packed: Bool,
](
    # Constants - passed as arguments
    C: UInt32,
    N: UInt32,
    n_isects: UInt32,
    image_width: UInt32,
    image_height: UInt32,
    # Inputs - Using placeholder LayoutTensors
    means2d: ManagedTensorSlice[type=DType.float32, rank=3],         # [C, N, 2] or [nnz, 2]
    conics: ManagedTensorSlice[type=DType.float32, rank=3],          # [C, N, 3] or [nnz, 3]
    colors: ManagedTensorSlice[type=DType.float32, rank=3],      # [C, N, CDIM] or [nnz, CDIM]
    opacities: ManagedTensorSlice[type=DType.float32, rank=2],   # [C, N] or [nnz]
    backgrounds: ManagedTensorSlice[type=DType.float32, rank=2], # Optional [C, CDIM] - Handle optionality
    has_backgrounds: Bool,                      # Flag for optional backgrounds
    masks: ManagedTensorSlice[type=DType.bool, rank=3],           # Optional [C, tile_height, tile_width] - Handle optionality
    has_masks: Bool,                            # Flag for optional masks
    tile_ranges: ManagedTensorSlice[type=DType.int32, rank=4], # [C, tile_height, tile_width, 2]
    flatten_ids: ManagedTensorSlice[type=DType.int32, rank=1],  # [n_isects]
    # Outputs - Using placeholder LayoutTensors
    render_colors: ManagedTensorSlice[type=DType.float32, rank=4], # [C, image_height, image_width, CDIM]
    render_alphas: ManagedTensorSlice[type=DType.float32, rank=4], # [C, image_height, image_width, 1]
    last_ids: ManagedTensorSlice[type=DType.int32, rank=3],        # [C, image_height, image_width]
):
    var sh_gaussian_ids = stack_allocation[
        tile_size * tile_size,
        DType.int32,
        address_space = AddressSpace.SHARED,
    ]()
    var sh_means = stack_allocation[
        tile_size * tile_size * 2,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()
    var sh_conics = stack_allocation[
        tile_size * tile_size * 3,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()
    var sh_opacities = stack_allocation[
        tile_size * tile_size,
        DType.float32,
        address_space = AddressSpace.SHARED,
    ]()
    
    # Get block and thread IDs
    alias camera_id: UInt32 = block_idx.x # Corresponds to grid.x
    alias tile_row: UInt32 = block_idx.y # Tile id row
    alias tile_col: UInt32 = block_idx.z # Tile id column

    alias thread_row: UInt32 = thread_idx.y # Pixel row within tile #TODO: Double check this order!
    alias thread_col: UInt32 = thread_idx.x # Pixel column within tile
    alias thread_count: UInt32 = tile_size * tile_size
    var thread_id: UInt32 = thread_row * tile_size + thread_col # Flat thread id within tile

    # var tile_id: UInt32 = block_row * tile_grid_width + block_col
    var i: UInt32 = tile_row * tile_size + thread_row # Absolute image row
    var j: UInt32 = tile_col * tile_size + thread_col # Absolute image column

    var px: Float32 = j.cast[DType.float32]() + 0.5
    var py: Float32 = i.cast[DType.float32]() + 0.5
    var pix_id: UInt32 = i * image_width + j # Flat pixel index within the camera's image

    # Return if out of bounds
    var inside: Bool = (i < image_height) and (j < image_width)
    var done: Bool = not inside

    # When the mask is provided, render the background color and return
    # if this tile is labeled as False
    if has_masks and inside:
        # TODO: Masking
        pass
    
    # Have all threads in tile process the same gaussians in batches
    # First collect gaussians between range.x and range.y in batches
    # Which gaussians to look through in this tile
    var range_start = tile_ranges[camera_id, tile_row, tile_col, 0]
    var range_end = tile_ranges[camera_id, tile_row, tile_col, 1]
    var num_batches: UInt32 = (range_end - range_start + thread_count - 1) / thread_count

    # Pixel Transmittance
    var T: Float32 = 1.0
    # Pixel Color
    var c: Vec3 = {0.0, 0.0, 0.0}
    var last_id: Int32 = -1

    # Collect gaussians in batches
    for batch in range(num_batches):
        # Sync all threads before starting next batch
        barrier()

        # Each thread loads one gaussian from front to back
        var batch_start = range_start + thread_count * batch
        var idx = batch_start + thread_id
        if idx < range_end:
            var g = flatten_ids[idx]
            sh_gaussian_ids[thread_id] = g
            sh_means[thread_id] = means2d[g, :]
            sh_conics[thread_id] = conics[g, :]
            sh_opacities[thread_id] = opacities[g]

        # Wait for all threads to load gaussians
        barrier()

        # Rasterize gaussians for this pixel
        # We iterate over the gaussians in the current batch
        var batch_size = min(thread_count, range_end - batch_start)
        for t in range(batch_size):
            var g = sh_gaussian_ids[t]
            var mean: Vec2 = sh_means[t]
            var conic: Vec3 = sh_conics[t]
            var opacity: Float32 = sh_opacities[t]

            var delta: Vec2 = {mean[0] - px, mean[1] - py}
            var sigma: Float32 = 0.5 * (conic[0] * delta[0] * delta[0] +
                                        conic[2] * delta[1] * delta[1]) +
                                        conic[1] * delta[0] * delta[1]
            var alpha: Float32 = min(0.999, opacity * exp(-sigma))
            if sigma < 0.0 or alpha < ALPHA_THRESHOLD:
                continue

            var next_T: Float32 = T * (1.0 - alpha)
            if next_T <= 1e-4:
                done = True
                break

            var vis: Float32 = alpha * T

            var c: Vec3 = colors[g, :]
            c = c * vis

            T = next_T
            last_id = last_id + 1

    if inside:
        render_colors[pix_id, :] = c
        render_alphas[pix_id] = T
        last_ids[pix_id] = last_id
            


# --------------------------------------------------------------------------
# MAX Engine Kernel Definition
# --------------------------------------------------------------------------

@compiler.register("rasterize_to_pixels_3dgs_fwd")
struct RasterizeToPixels3DGSFwd:
    @staticmethod
    fn execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
        # Compile-time constants for the kernel
        tile_size: Int,
        CDIM: Int
    ](
        # Inputs
        means2d: InputTensor[type=DType.float32, rank=2],        # [nnz, 2]
        conics: InputTensor[type=DType.float32, rank=2],         # [nnz, 3]
        colors: InputTensor[type=DType.float32, rank=2],         # [nnz, CDIM]
        opacities: InputTensor[type=DType.float32, rank=1],      # [nnz]
        backgrounds: InputTensor[type=DType.float32, rank=2],    # [C, CDIM]
        masks: InputTensor[type=DType.bool, rank=3],             # [C, tile_h, tile_w]
        tile_ranges: InputTensor[type=DType.int32, rank=4],      # [C, tile_h, tile_w, 2]
        flatten_ids: InputTensor[type=DType.int32, rank=1],      # [n_isects]
        # Outputs
        render_colors: OutputTensor[type=DType.float32, rank=4], # [C, H, W, CDIM]
        render_alphas: OutputTensor[type=DType.float32, rank=4], # [C, H, W, 1]
        last_ids: OutputTensor[type=DType.int32, rank=3],        # [C, H, W]
        # Runtime parameters (passed explicitly)
        image_height: Int,
        image_width: Int,
        has_backgrounds: Bool, # Pass flags explicitly
        has_masks: Bool,
        # Context
        ctx: DeviceContextPtr,
    ) raises:
        # Determine grid and block dimensions based on output image and tile size
        var C = render_colors.dim_size(0) # Number of cameras
        # Assuming image_height, image_width are passed correctly
        var tile_grid_height = ceildiv(image_height, tile_size)
        var tile_grid_width = ceildiv(image_width, tile_size)

        # Check Tensor dimensions (optional but recommended)
        # TODO: Add assertions for tensor shapes consistency

        # Dispatch based on target device
        @parameter
        if target == "cpu":
            # TODO: Implement CPU version if needed
             raise Error("Rasterize3DGS CPU target not implemented yet.")
        elif target == "gpu":
            # Get GPU context
            var gpu_ctx = ctx.get_device_context()

            # Define grid and block dimensions for the kernel launch
            # Grid: (Num Cameras, Num Tiles Height, Num Tiles Width)
            let grid = (C, tile_grid_height, tile_grid_width)
            # Block: (Tile Width, Tile Height, 1) - Assuming tile_size x tile_size threads per tile
            let block = (tile_size, tile_size, 1) # Note: thread_idx.x maps to tile width, thread_idx.y to height

            # Launch the GPU kernel
            gpu_ctx.enqueue_function[
                 rasterize_kernel[tile_size, CDIM, ALPHA_THRESHOLD] # Pass compile-time params
            ](
                # Pass runtime params
                image_width.cast[DType.uint32](),
                image_height.cast[DType.uint32](),
                tile_grid_width,
                tile_grid_height,
                # Pass tensor slices (implicitly converted from Input/OutputTensor)
                means2d,
                conics,
                colors,
                opacities,
                backgrounds,
                has_backgrounds,
                masks,
                has_masks,
                tile_ranges,
                flatten_ids,
                render_colors,
                render_alphas,
                last_ids,
                # Grid and block dimensions
                grid_dim=grid,
                block_dim=block
            )
        else:
            raise Error("Unsupported target:", target)


 
