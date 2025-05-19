import compiler
from gpu import thread_idx, block_idx, barrier
from gpu.host import DeviceBuffer
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from layout.tensor_builder import LayoutTensorBuild as tb
from runtime.asyncrt import DeviceContextPtr
from math import exp, ceildiv
from sys import sizeof

from tensor import InputTensor, OutputTensor, ManagedTensorSlice

alias Dyn1DLayout = Layout.row_major(UNKNOWN_VALUE)
alias Dyn2DLayout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
alias Dyn3DLayout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
alias Dyn4DLayout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
alias dtype = DType.float32

alias ALPHA_THRESHOLD = 1.0 / 255.0

alias Means3DLayout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, 2)
alias Conics3DLayout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, 3)
alias TileRanges4DLayout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, 2)

fn rasterize_to_pixels_3dgs_fwd_kernel[
    tile_size: Int,
    tile_grid_width: Int,
    tile_grid_height: Int,
    CDIM: Int, # Number of color channels
    # packed: Bool,
](
    # Constants - passed as arguments
    C: UInt, # Number of cameras (aka batch size)
    N: UInt, # Number of gaussians
    n_isects: UInt, # Number of intersections
    image_width: UInt,
    image_height: UInt,
    # Inputs - Using placeholder LayoutTensors
    means2d: LayoutTensor[dtype, Means3DLayout, MutableAnyOrigin],         # [C, N, 2]
    conics: LayoutTensor[dtype, Conics3DLayout, MutableAnyOrigin],          # [C, N, 3]
    colors: LayoutTensor[dtype, Dyn3DLayout, MutableAnyOrigin],      # [C, N, CDIM]
    opacities: LayoutTensor[dtype, Dyn2DLayout, MutableAnyOrigin],   # [C, N]
    backgrounds: LayoutTensor[dtype, Dyn2DLayout, MutableAnyOrigin], # Optional [C, CDIM] - Handle optionality
    has_backgrounds: Bool,                      # Flag for optional backgrounds
    tile_ranges: LayoutTensor[DType.int32, TileRanges4DLayout, MutableAnyOrigin], # [C, tile_height, tile_width, 2]
    flatten_ids: LayoutTensor[DType.int32, Dyn1DLayout, MutableAnyOrigin],  # [n_isects]
    # Outputs - Using placeholder LayoutTensors
    render_colors: LayoutTensor[dtype, Dyn4DLayout, MutableAnyOrigin], # [C, image_height, image_width, CDIM]
    render_alphas: LayoutTensor[dtype, Dyn3DLayout, MutableAnyOrigin], # [C, image_height, image_width]
    last_ids: LayoutTensor[DType.int32, Dyn3DLayout, MutableAnyOrigin],        # [C, image_height, image_width]
):
    sh_gaussian_ids = tb[DType.int32]().row_major[tile_size * tile_size]().shared().alloc()
    sh_means = tb[dtype]().row_major[tile_size * tile_size, 2]().shared().alloc()
    sh_conics = tb[dtype]().row_major[tile_size * tile_size, 3]().shared().alloc()
    sh_opacities = tb[dtype]().row_major[tile_size * tile_size]().shared().alloc()

    alias FloatType = opacities.element_type
    
    # Get block and thread IDs
    camera_id = block_idx.x # Corresponds to grid.x
    tile_row = block_idx.y # Tile id row
    tile_col = block_idx.z # Tile id column

    thread_row = thread_idx.y # Pixel row within tile #TODO: Double check this order!
    thread_col = thread_idx.x # Pixel column within tile
    thread_count = tile_size * tile_size
    thread_id = thread_row * tile_size + thread_col # Flat thread id within tile

    # var tile_id: UInt32 = block_row * tile_grid_width + block_col
    i = tile_row * tile_size + thread_row # Absolute image row
    j = tile_col * tile_size + thread_col # Absolute image column

    px = Float32(j) + 0.5
    py = Float32(i) + 0.5
    # pix_id = i * image_width + j # Flat pixel index within the camera's image

    # Return if out of bounds
    var inside: Bool = (i < image_height) and (j < image_width)
    var done: Bool = not inside

    # Have all threads in tile process the same gaussians in batches
    # First collect gaussians between range.x and range.y in batches
    # Which gaussians to look through in this tile
    var range_start = tile_ranges[camera_id, tile_row, tile_col, 0]
    var range_end = tile_ranges[camera_id, tile_row, tile_col, 1]
    var num_batches = (range_end - range_start + thread_count - 1) / thread_count

    # Pixel Transmittance
    var T: FloatType = 1.0
    # Pixel Color
    # var pix_out: SIMD[dtype, CDIM] = SIMD[dtype, CDIM](0.0)
    var pix_out = tb[dtype]().row_major[CDIM]().alloc()
    for c in range(CDIM):
        pix_out[c] = 0.0
    var last_id: Int32 = -1

    # Collect gaussians in batches
    for batch in range(num_batches):
        # Sync all threads before starting next batch
        barrier()

        # Each thread loads one gaussian from front to back
        var batch_start = range_start + thread_count * batch
        var idx = batch_start + thread_id
        if idx < range_end:
            g = Int(flatten_ids[Int(idx)])
            sh_gaussian_ids[thread_id] = g
            sh_means[thread_id] = means2d[camera_id, g]
            sh_conics[thread_id] = conics[camera_id, g]
            sh_opacities[thread_id] = opacities[camera_id, g]

        # Wait for all threads in block to load gaussians
        barrier()

        # Rasterize gaussians for this pixel
        # We iterate over the gaussians in the current batch
        var batch_size = min(thread_count, range_end - batch_start)
        for t in range(batch_size):
            var g = Int(sh_gaussian_ids[t])
            var mean = sh_means[t]
            var conic = sh_conics[t]
            var opacity: FloatType = sh_opacities[t]

            var delta = SIMD[dtype, 2](mean[0] - px, mean[1] - py)
            var sigma: FloatType = 0.5 * (conic[0] * delta[0] * delta[0] +
                                        conic[2] * delta[1] * delta[1]) +
                                        conic[1] * delta[0] * delta[1]
            var alpha = min(opacity * exp(-sigma), 0.999)
            if sigma < 0.0 or alpha < ALPHA_THRESHOLD:
                continue

            var next_T = T * (1.0 - alpha)
            if next_T <= 1e-4:
                done = True
                break

            var vis = alpha * T

            for c in range(CDIM):
                pix_out[c] += colors[camera_id, g, c] * vis

            T = next_T
            last_id = last_id + 1

    if inside:
        for c in range(CDIM):
            render_colors[camera_id, i, j, c] = pix_out[c]
        render_alphas[camera_id, i, j] = T
        last_ids[camera_id, i, j] = last_id
            


# --------------------------------------------------------------------------
# MAX Engine Kernel Definition
# --------------------------------------------------------------------------

@compiler.register("rasterize_to_pixels_3dgs_fwd")
struct RasterizeToPixels3DGSFwd:
    @staticmethod
    fn execute[
        target: StaticString,
        tile_size: Int,
        image_height: Int,
        image_width: Int,
        CDIM: Int
    ](
        # Outputs
        render_colors: OutputTensor[type=DType.float32, rank=4],
        render_alphas: OutputTensor[type=DType.float32, rank=4],
        last_ids: OutputTensor[type=DType.int32, rank=3],
        # Inputs
        means2d: InputTensor[type=DType.float32, rank=3],
        conics: InputTensor[type=DType.float32, rank=3],
        colors: InputTensor[type=DType.float32, rank=3],
        opacities: InputTensor[type=DType.float32, rank=2],
        backgrounds: InputTensor[type=DType.float32, rank=2],
        has_backgrounds: Bool,
        tile_ranges: InputTensor[type=DType.int32, rank=4],
        flatten_ids: InputTensor[type=DType.int32, rank=1],
        # Context
        ctx: DeviceContextPtr
    ) raises:
        # Determine grid and block dimensions based on output image and tile size
        var C = render_colors.dim_size(0) # Number of cameras
        alias tile_grid_height = ceildiv(image_height, tile_size)
        alias tile_grid_width = ceildiv(image_width, tile_size)

        means2d_tensor = means2d.to_layout_tensor()
        conics_tensor = conics.to_layout_tensor()
        colors_tensor = colors.to_layout_tensor()
        opacities_tensor = opacities.to_layout_tensor()
        backgrounds_tensor = backgrounds.to_layout_tensor()
        tile_ranges_tensor = tile_ranges.to_layout_tensor()

        render_colors_tensor = render_colors.to_layout_tensor()
        render_alphas_tensor = render_alphas.to_layout_tensor()
        last_ids_tensor = last_ids.to_layout_tensor()

        @parameter
        if target == "cpu":
            raise Error("Rasterize3DGS CPU target not implemented yet.")
        elif target == "gpu":
            # Get GPU context
            var gpu_ctx = ctx.get_device_context()

            # Define grid and block dimensions for the kernel launch
            var grid = (C, tile_grid_height, tile_grid_width)
            var block = (tile_size, tile_size, 1)

            gpu_ctx.enqueue_function[
                rasterize_to_pixels_3dgs_fwd_kernel[
                    tile_size, tile_grid_width, tile_grid_height, CDIM
                ]
            ](
                render_colors_tensor,
                render_alphas_tensor,
                last_ids_tensor,
                means2d_tensor,
                conics_tensor,
                colors_tensor,
                opacities_tensor,
                backgrounds_tensor,
                has_backgrounds,
                tile_ranges_tensor,
                flatten_ids,
                grid_dim=grid,
                block_dim=block,
            )

        else:
            raise Error("Unsupported target:", target)
