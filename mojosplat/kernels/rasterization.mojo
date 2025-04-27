from gpu import thread_idx, block_idx, warp, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace
from memory import stack_allocation
from layout import Layout, LayoutTensor
from math import iota
from sys import sizeof

from tensor import Tensor

# Placeholder aliases - We need to define the actual layouts based on usage
alias InputLayoutTensor_Means2D = LayoutTensor[DType.float32, Layout.row_major(1, 1), MutableAnyOrigin] # Placeholder [C, N, 2] or [nnz, 2]
alias InputLayoutTensor_Conics = LayoutTensor[DType.float32, Layout.row_major(1, 1), MutableAnyOrigin] # Placeholder [C, N, 3] or [nnz, 3]
alias InputLayoutTensor_Colors = LayoutTensor[DType.float32, Layout.row_major(1, 1), MutableAnyOrigin] # Placeholder [C, N, CDIM] or [nnz, CDIM]
alias InputLayoutTensor_Opacities = LayoutTensor[DType.float32, Layout.row_major(1, 1), MutableAnyOrigin] # Placeholder [C, N] or [nnz]
alias InputLayoutTensor_Backgrounds = LayoutTensor[DType.float32, Layout.row_major(1, 1), MutableAnyOrigin] # Placeholder [C, CDIM]
alias InputLayoutTensor_Masks = LayoutTensor[DType.bool, Layout.row_major(1, 1), MutableAnyOrigin]       # Placeholder [C, tile_height, tile_width]
alias InputLayoutTensor_TileOffsets = LayoutTensor[DType.int32, Layout.row_major(1, 1), MutableAnyOrigin] # Placeholder [C, tile_height, tile_width]
alias InputLayoutTensor_FlattenIDs = LayoutTensor[DType.int32, Layout.row_major(1), MutableAnyOrigin]   # Placeholder [n_isects]

alias OutputLayoutTensor_RenderColors = LayoutTensor[DType.float32, Layout.row_major(1, 1), MutableAnyOrigin] # Placeholder [C, image_height, image_width, CDIM]
alias OutputLayoutTensor_RenderAlphas = LayoutTensor[DType.float32, Layout.row_major(1, 1), MutableAnyOrigin] # Placeholder [C, image_height, image_width, 1]
alias OutputLayoutTensor_LastIDs = LayoutTensor[DType.int32, Layout.row_major(1, 1), MutableAnyOrigin]      # Placeholder [C, image_height, image_width]

alias Vec2 = SIMD[DType.float32, 2]
alias Vec3 = SIMD[DType.float32, 3]

fn rasterize_to_pixels_3dgs_fwd(
    means2d: Tensor[DType.float32],
    opacities: Tensor[DType.float32],
    
):
    var means2d_shape = means2d.shape
    var opacities_shape = opacities.shape

    alias ILTensor_Means2D = LayoutTensor[DType.float32, Layout.row_major(means2d_shape), MutableAnyOrigin]
    alias ILTensor_Conics = LayoutTensor[DType.float32, Layout.row_major(conics_shape), MutableAnyOrigin]
    alias ILTensor_Colors = LayoutTensor[DType.float32, Layout.row_major(colors_shape), MutableAnyOrigin]
    alias ILTensor_Opacities = LayoutTensor[DType.float32, Layout.row_major(opacities_shape), MutableAnyOrigin]



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
        means2d: InputLayoutTensor_Means2D,         # [C, N, 2] or [nnz, 2]
        conics: InputLayoutTensor_Conics,          # [C, N, 3] or [nnz, 3]
        colors: InputLayoutTensor_Colors,      # [C, N, CDIM] or [nnz, CDIM]
        opacities: InputLayoutTensor_Opacities,   # [C, N] or [nnz]
        backgrounds: InputLayoutTensor_Backgrounds, # Optional [C, CDIM] - Handle optionality
        has_backgrounds: Bool,                      # Flag for optional backgrounds
        masks: InputLayoutTensor_Masks,           # Optional [C, tile_height, tile_width] - Handle optionality
        has_masks: Bool,                            # Flag for optional masks
        tile_ranges: InputLayoutTensor_TileOffsets, # [C, tile_height, tile_width, 2]
        flatten_ids: InputLayoutTensor_FlattenIDs,  # [n_isects]
        # Outputs - Using placeholder LayoutTensors
        render_colors: OutputLayoutTensor_RenderColors, # [C, image_height, image_width, CDIM]
        render_alphas: OutputLayoutTensor_RenderAlphas, # [C, image_height, image_width, 1]
        last_ids: OutputLayoutTensor_LastIDs        # [C, image_height, image_width]
    ):
        var sh_gaussian_ids = stack_allocation[
            tile_size * tile_size * sizeof[DType.int32],
            SIMD[DType.int32, 1],
            address_space = AddressSpace.SHARED,
        ]()
        var sh_means = stack_allocation[
            tile_size * tile_size * 2 * sizeof[DType.float32],
            SIMD[DType.float32, 2],
            address_space = AddressSpace.SHARED,
        ]()
        var sh_conics = stack_allocation[
            tile_size * tile_size * 3 * sizeof[DType.float32],
            SIMD[DType.float32, 3],
            address_space = AddressSpace.SHARED,
        ]()
        var sh_opacities = stack_allocation[
            tile_size * tile_size * sizeof[DType.float32],
            SIMD[DType.float32, 1],
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
                
                
