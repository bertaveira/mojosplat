import compiler
from gpu import thread_idx, block_idx, barrier
from gpu.host import DeviceBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from runtime.asyncrt import DeviceContextPtr
from math import exp, ceildiv

from tensor import InputTensor, OutputTensor

comptime dtype = DType.float32

comptime ALPHA_THRESHOLD = 1.0 / 255.0


fn rasterize_to_pixels_3dgs_fwd_kernel[
    tile_size: Int,
    tile_grid_width: Int,
    tile_grid_height: Int,
    CDIM: Int, # Number of color channels
    C: Int, # Number of cameras
    N: Int, # Number of gaussians
    NIntersections: Int, # Number of intersections
    image_width: Int,
    image_height: Int,
](
    means2d: LayoutTensor[dtype, Layout.row_major(C, N, 2), MutAnyOrigin],        # [C, N, 2]
    conics: LayoutTensor[dtype, Layout.row_major(C, N, 3), MutAnyOrigin],         # [C, N, 3]
    colors: LayoutTensor[dtype, Layout.row_major(C, N, CDIM), MutAnyOrigin],      # [C, N, CDIM]
    opacities: LayoutTensor[dtype, Layout.row_major(C, N), MutAnyOrigin],         # [C, N]
    backgrounds: LayoutTensor[dtype, Layout.row_major(C, CDIM), MutAnyOrigin],    # [C, CDIM]
    # has_backgrounds always True: Bool is not DevicePassable in Mojo 26
    tile_ranges: LayoutTensor[DType.int32, Layout.row_major(C, tile_grid_height, tile_grid_width, 2), MutAnyOrigin], # [C, tile_height, tile_width, 2]
    flatten_ids: LayoutTensor[DType.int32, Layout.row_major(C, NIntersections), MutAnyOrigin],  # [C, n_isects]
    # Outputs
    render_colors: LayoutTensor[dtype, Layout.row_major(C, image_height, image_width, CDIM), MutAnyOrigin], # [C, image_height, image_width, CDIM]
):
    sh_gaussian_ids = LayoutTensor[
        DType.int32,
        Layout.row_major(tile_size * tile_size),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    sh_means = LayoutTensor[
        dtype,
        Layout.row_major(tile_size * tile_size, 2),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    sh_conics = LayoutTensor[
        dtype,
        Layout.row_major(tile_size * tile_size, 3),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    sh_opacities = LayoutTensor[
        dtype,
        Layout.row_major(tile_size * tile_size),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    comptime FloatType = opacities.element_type

    # Get block and thread IDs
    camera_id = block_idx.x # Corresponds to grid.x
    tile_row = block_idx.y # Tile id row
    tile_col = block_idx.z # Tile id column

    thread_row = thread_idx.y # Pixel row within tile #TODO: Double check this order!
    thread_col = thread_idx.x # Pixel column within tile
    thread_count = Int32(tile_size * tile_size)
    thread_id = Int32(thread_row * tile_size + thread_col) # Flat thread id within tile

    i = tile_row * tile_size + thread_row # Absolute image row
    j = tile_col * tile_size + thread_col # Absolute image column

    px = Float32(j) + 0.5
    py = Float32(i) + 0.5

    # Return if out of bounds
    var inside: Bool = (i < image_height) and (j < image_width)
    var done: Bool = not inside

    # Have all threads in tile process the same gaussians in batches
    # First collect gaussians between range.x and range.y in batches
    # Which gaussians to look through in this tile
    range_start = tile_ranges[camera_id, tile_row, tile_col, 0][0]
    range_end = tile_ranges[camera_id, tile_row, tile_col, 1][0]
    num_batches = (range_end - range_start + thread_count - 1) / thread_count

    # Pixel Transmittance
    var T: FloatType = 1.0
    # Pixel Color (start at zero; background blended at the end weighted by remaining T)
    var pix_out = LayoutTensor[dtype, Layout.row_major(CDIM), MutAnyOrigin].stack_allocation()
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
            var g = Int(flatten_ids[camera_id, Int(idx)])
            if g >= 0 and g < N:
                sh_gaussian_ids[thread_id] = g
                sh_means[thread_id, 0] = means2d[camera_id, g, 0]
                sh_means[thread_id, 1] = means2d[camera_id, g, 1]
                sh_conics[thread_id, 0] = conics[camera_id, g, 0]
                sh_conics[thread_id, 1] = conics[camera_id, g, 1]
                sh_conics[thread_id, 2] = conics[camera_id, g, 2]
                sh_opacities[thread_id] = opacities[camera_id, g]

        # Wait for all threads in block to load gaussians
        barrier()

        # Rasterize gaussians for this pixel
        # We iterate over the gaussians in the current batch
        if inside and not done:
            var batch_size = min(thread_count, range_end - batch_start)
            for t in range(batch_size):
                var g = Int(sh_gaussian_ids[t])
                # Access 2D tensor elements individually instead of getting row slices
                var mean_x: FloatType = sh_means[t, 0]
                var mean_y: FloatType = sh_means[t, 1]
                var conic_xx: FloatType = sh_conics[t, 0]
                var conic_xy: FloatType = sh_conics[t, 1]
                var conic_yy: FloatType = sh_conics[t, 2]
                var opacity: FloatType = sh_opacities[t]

                var delta_x = mean_x - px
                var delta_y = mean_y - py
                var sigma: FloatType = 0.5 * (conic_xx * delta_x * delta_x +
                                            conic_yy * delta_y * delta_y) +
                                            conic_xy * delta_x * delta_y
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
            render_colors[camera_id, i, j, c] = pix_out[c] + T * backgrounds[camera_id, c][0]


# --------------------------------------------------------------------------
# MAX Engine Kernel Definition
# --------------------------------------------------------------------------

@compiler.register("rasterize_to_pixels_3dgs_fwd")
struct RasterizeToPixels3DGSFwd:
    @staticmethod
    fn execute[
        tile_size: Int,
        image_height: Int,
        image_width: Int,
        CDIM: Int,
        C: Int,
        N: Int,
        NIntersections: Int,
        target: StaticString,
    ](
        # Outputs
        render_colors: OutputTensor[dtype=DType.float32, rank=4],
        # Inputs
        means2d: InputTensor[dtype=DType.float32, rank=3],
        conics: InputTensor[dtype=DType.float32, rank=3],
        colors: InputTensor[dtype=DType.float32, rank=3],
        opacities: InputTensor[dtype=DType.float32, rank=2],
        backgrounds: InputTensor[dtype=DType.float32, rank=2],
        tile_ranges: InputTensor[dtype=DType.int32, rank=4],
        flatten_ids: InputTensor[dtype=DType.int32, rank=2],
        # Context
        ctx: DeviceContextPtr
    ) raises:
        # Determine grid and block dimensions based on output image and tile size
        comptime tile_grid_height = ceildiv(image_height, tile_size)
        comptime tile_grid_width = ceildiv(image_width, tile_size)

        has_backgrounds = True #FIXME: when mojo fix it, accept this bool as argument

        means2d_tensor = means2d.to_layout_tensor()
        conics_tensor = conics.to_layout_tensor()
        colors_tensor = colors.to_layout_tensor()
        opacities_tensor = opacities.to_layout_tensor()
        backgrounds_tensor = backgrounds.to_layout_tensor()
        tile_ranges_tensor = tile_ranges.to_layout_tensor()
        flatten_ids_tensor = flatten_ids.to_layout_tensor()

        render_colors_tensor = render_colors.to_layout_tensor()

        @parameter
        if target == "cpu":
            raise Error("Rasterize3DGS CPU target not implemented yet.")
        elif target == "gpu":
            # Get GPU context
            var gpu_ctx = ctx.get_device_context()

            # Define grid and block dimensions for the kernel launch
            var grid = (C, tile_grid_height, tile_grid_width)
            var block = (tile_size, tile_size, 1)

            gpu_ctx.enqueue_function_unchecked[
                rasterize_to_pixels_3dgs_fwd_kernel[
                    tile_size, tile_grid_width, tile_grid_height, CDIM, C, N, NIntersections, image_width, image_height
                ]
            ](
                means2d_tensor,
                conics_tensor,
                colors_tensor,
                opacities_tensor,
                backgrounds_tensor,
                tile_ranges_tensor,
                flatten_ids_tensor,
                render_colors_tensor,
                grid_dim=grid,
                block_dim=block,
            )

        else:
            raise Error("Unsupported target:", target)
