import compiler
from gpu import thread_idx, block_idx, barrier
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from collections import InlineArray
from runtime.asyncrt import DeviceContextPtr
from math import exp, ceildiv
from memory import UnsafePointer
from os import Atomic
from tensor import InputTensor, OutputTensor

comptime dtype = DType.float32

comptime ALPHA_THRESHOLD = 1.0 / 255.0


fn rasterize_to_pixels_3dgs_fwd_kernel[
    tile_size: Int,
    CDIM: Int,  # kept compile-time — shared memory and pix_out need it
](
    means2d_ptr:       UnsafePointer[Scalar[DType.float32], MutAnyOrigin],  # [C * N * 2]
    conics_ptr:        UnsafePointer[Scalar[DType.float32], MutAnyOrigin],  # [C * N * 3]
    colors_ptr:        UnsafePointer[Scalar[DType.float32], MutAnyOrigin],  # [C * N * CDIM]
    opacities_ptr:     UnsafePointer[Scalar[DType.float32], MutAnyOrigin],  # [C * N]
    backgrounds_ptr:   UnsafePointer[Scalar[DType.float32], MutAnyOrigin],  # [C * CDIM]
    tile_ranges_ptr:   UnsafePointer[Scalar[DType.int32],   MutAnyOrigin],  # [C * TGH * TGW * 2]
    flatten_ids_ptr:   UnsafePointer[Scalar[DType.int32],   MutAnyOrigin],  # [C * NIntersections]
    render_colors_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],  # [C * H * W * CDIM]
    N: Int, C: Int, NIntersections: Int,
    image_width: Int, image_height: Int,
    tile_grid_width: Int, tile_grid_height: Int,
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
    sh_done_count = LayoutTensor[
        DType.int32,
        Layout.row_major(1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Get block and thread IDs
    camera_id = block_idx.x  # Corresponds to grid.x
    tile_row = block_idx.y   # Tile id row
    tile_col = block_idx.z   # Tile id column

    thread_row = thread_idx.y  # Pixel row within tile
    thread_col = thread_idx.x  # Pixel column within tile
    thread_count = Int32(tile_size * tile_size)
    thread_id = Int32(thread_row * tile_size + thread_col)  # Flat thread id within tile

    i = tile_row * tile_size + thread_row  # Absolute image row
    j = tile_col * tile_size + thread_col  # Absolute image column

    px = Float32(j) + 0.5
    py = Float32(i) + 0.5

    # Return if out of bounds
    var inside: Bool = (i < image_height) and (j < image_width)
    var done: Bool = not inside

    # Which gaussians to look through in this tile
    var tile_base = (camera_id * tile_grid_height * tile_grid_width + tile_row * tile_grid_width + tile_col) * 2
    var range_start = tile_ranges_ptr[tile_base + 0]
    var range_end   = tile_ranges_ptr[tile_base + 1]
    var num_batches = (range_end - range_start + thread_count - 1) / thread_count

    # Pixel Transmittance
    var T: Float32 = 1.0
    # Pixel Color: InlineArray stays in registers (no AddressSpace.GENERIC spilling)
    var pix_out = InlineArray[Float32, CDIM](0.0)
    var last_id: Int32 = -1

    if thread_id == 0:
        sh_done_count[0] = 0
    barrier()

    # Collect gaussians in batches
    for batch in range(num_batches):
        var batch_start = range_start + thread_count * batch

        # Phase 1 (overlapped): done threads increment the counter AND every
        # thread loads its gaussian. This reduces what would be 2 barriers per batch to 1.
        if done:
            _ = Atomic[DType.int32].fetch_add(sh_done_count.ptr, Int32(1))

        var idx = batch_start + thread_id
        if idx < range_end:
            var g = Int(flatten_ids_ptr[camera_id * NIntersections + Int(idx)])
            if g >= 0 and g < N:
                sh_gaussian_ids[thread_id] = g
                sh_means[thread_id, 0] = means2d_ptr[(camera_id * N + g) * 2 + 0]
                sh_means[thread_id, 1] = means2d_ptr[(camera_id * N + g) * 2 + 1]
                sh_conics[thread_id, 0] = conics_ptr[(camera_id * N + g) * 3 + 0]
                sh_conics[thread_id, 1] = conics_ptr[(camera_id * N + g) * 3 + 1]
                sh_conics[thread_id, 2] = conics_ptr[(camera_id * N + g) * 3 + 2]
                sh_opacities[thread_id] = opacities_ptr[camera_id * N + g]

        # Barrier 1 of 2: synchronises both the done-count atomics and the
        # shared-memory gaussian loads in one shot.
        barrier()

        if sh_done_count[0][0] >= thread_count:
            break

        # Rasterize gaussians for this pixel
        if inside and not done:
            var batch_size = min(thread_count, range_end - batch_start)
            for t in range(batch_size):
                var g = Int(sh_gaussian_ids[t][0])
                var mean_x: Float32 = sh_means[t, 0][0]
                var mean_y: Float32 = sh_means[t, 1][0]
                var conic_xx: Float32 = sh_conics[t, 0][0]
                var conic_xy: Float32 = sh_conics[t, 1][0]
                var conic_yy: Float32 = sh_conics[t, 2][0]
                var opacity: Float32 = sh_opacities[t][0]

                var delta_x = mean_x - px
                var delta_y = mean_y - py
                var sigma: Float32 = 0.5 * (conic_xx * delta_x * delta_x +
                                            conic_yy * delta_y * delta_y) +
                                            conic_xy * delta_x * delta_y
                var alpha = min(opacity * exp(-sigma), 0.999)

                # Nested if instead of `continue` — avoids the double-label PTX
                # pattern where the compiler emits a second loop header to
                # recompute 5 address-arithmetic instructions for every gaussian
                # that actually contributes (the fast-path skip jumps to the
                # inner label directly, bypassing the recompute; the slow path
                # always falls through the outer label and pays the penalty).
                if sigma >= 0.0 and alpha >= ALPHA_THRESHOLD:
                    var next_T = T * (1.0 - alpha)
                    if next_T <= 1e-4:
                        done = True
                        break

                    var vis = alpha * T

                    @parameter
                    for c in range(CDIM):
                        pix_out[c] = pix_out[c] + colors_ptr[(camera_id * N + g) * CDIM + c] * vis

                    T = next_T
                    last_id = last_id + 1

        # Barrier 2 of 2: reset done counter for the next batch.  This barrier
        # doubles as the "previous-batch done" sync that the next iteration
        # needs before it writes atomics, so no extra barrier is required there.
        if thread_id == 0:
            sh_done_count[0] = 0
        barrier()

    if inside:
        var pixel_base = (camera_id * image_height + i) * image_width * CDIM + j * CDIM
        @parameter
        for c in range(CDIM):
            render_colors_ptr[pixel_base + c] = pix_out[c] + T * backgrounds_ptr[camera_id * CDIM + c]


# --------------------------------------------------------------------------
# MAX Engine Kernel Definition
# --------------------------------------------------------------------------

@compiler.register("rasterize_to_pixels_3dgs_fwd")
struct RasterizeToPixels3DGSFwd:
    @staticmethod
    fn execute[
        tile_size: Int,
        CDIM: Int,
        target: StaticString,
    ](
        # Outputs
        render_colors: OutputTensor[dtype=DType.float32, rank=4],  # (C, H, W, CDIM)
        # Inputs
        means2d:     InputTensor[dtype=DType.float32, rank=3],
        conics:      InputTensor[dtype=DType.float32, rank=3],
        colors:      InputTensor[dtype=DType.float32, rank=3],
        opacities:   InputTensor[dtype=DType.float32, rank=2],
        backgrounds: InputTensor[dtype=DType.float32, rank=2],
        tile_ranges: InputTensor[dtype=DType.int32, rank=4],
        flatten_ids: InputTensor[dtype=DType.int32, rank=2],
        # Context
        ctx: DeviceContextPtr
    ) raises:
        var C              = render_colors.dim_size(0)
        var image_height   = render_colors.dim_size(1)
        var image_width    = render_colors.dim_size(2)
        var N              = means2d.dim_size(1)
        var NIntersections = flatten_ids.dim_size(1)
        var tile_grid_height = tile_ranges.dim_size(1)
        var tile_grid_width  = tile_ranges.dim_size(2)

        # Extract UnsafePointers (same rebind pattern as projection.mojo)
        var means2d_ptr       = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](means2d.to_layout_tensor().ptr)
        var conics_ptr        = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](conics.to_layout_tensor().ptr)
        var colors_ptr        = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](colors.to_layout_tensor().ptr)
        var opacities_ptr     = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](opacities.to_layout_tensor().ptr)
        var backgrounds_ptr   = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](backgrounds.to_layout_tensor().ptr)
        var tile_ranges_ptr   = rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](tile_ranges.to_layout_tensor().ptr)
        var flatten_ids_ptr   = rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](flatten_ids.to_layout_tensor().ptr)
        var render_colors_ptr = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](render_colors.to_layout_tensor().ptr)

        @parameter
        if target == "cpu":
            raise Error("Rasterize3DGS CPU target not implemented yet.")
        elif target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var grid = (C, tile_grid_height, tile_grid_width)
            var block = (tile_size, tile_size, 1)
            gpu_ctx.enqueue_function_unchecked[
                rasterize_to_pixels_3dgs_fwd_kernel[tile_size, CDIM]
            ](
                means2d_ptr, conics_ptr, colors_ptr, opacities_ptr,
                backgrounds_ptr, tile_ranges_ptr, flatten_ids_ptr, render_colors_ptr,
                N, C, NIntersections, image_width, image_height, tile_grid_width, tile_grid_height,
                grid_dim=grid, block_dim=block,
            )
        else:
            raise Error("Unsupported target:", target)
