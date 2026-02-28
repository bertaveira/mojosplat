import compiler
from gpu import thread_idx, block_idx, barrier
from layout import Layout, LayoutTensor
from utils.index import IndexList
from math import sqrt, ceil, ceildiv, log
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer

comptime radius_clip: Float32 = 0.0
comptime block_size: Int = 256


fn project_ewa_kernel(
    means3d_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],      # [N * 3]
    scales_ptr:  UnsafePointer[Scalar[DType.float32], MutAnyOrigin],      # [N * 3]
    quats_ptr:   UnsafePointer[Scalar[DType.float32], MutAnyOrigin],      # [N * 4]
    opacities_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],    # [N]
    view_matrices_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin], # [C * 16]
    ks_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],           # [C * 11]
    radii_ptr:   UnsafePointer[Scalar[DType.int32], MutAnyOrigin],        # [C * N * 2]
    means2d_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],      # [C * N * 2]
    depths_ptr:  UnsafePointer[Scalar[DType.float32], MutAnyOrigin],      # [C * N]
    conics_ptr:  UnsafePointer[Scalar[DType.float32], MutAnyOrigin],      # [C * N * 3]
    N: Int,
    C: Int,
):
    # This kernel is parallelized over N * C
    var idx = block_idx.x * block_size + thread_idx.x
    var gaussian_idx = Int(idx % N)
    var camera_idx   = Int(idx // N)

    if idx >= N * C:
        return

    # Extract R (3x3) and t (3) from the view matrix — skip the unused bottom row
    var R_view = LayoutTensor[DType.float32, Layout.row_major(3, 3), MutAnyOrigin].stack_allocation()
    var t_view = LayoutTensor[DType.float32, Layout.row_major(3), MutAnyOrigin].stack_allocation()
    for i in range(3):
        for j in range(3):
            R_view[i, j] = view_matrices_ptr[camera_idx * 16 + i * 4 + j]
        t_view[i] = view_matrices_ptr[camera_idx * 16 + i * 4 + 3]


    ########### Gaussian World to Camera ########### FIXME: Replace by function (not possible it seems right now)
    # mean_c = R * mean + t
    var mean_c = LayoutTensor[DType.float32, Layout.row_major(3), MutAnyOrigin].stack_allocation()
    for i in range(3):
        mean_c[i] = 0.0
        for j in range(3):
            mean_c[i] += R_view[i, j] * means3d_ptr[gaussian_idx * 3 + j]
        mean_c[i] += t_view[i]

    comptime near_plane: Float32 = 0.1
    if mean_c[2][0] <= near_plane:
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0
        # Set means2d to 0 instead of NaN for culled Gaussians
        means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0.0
        means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0.0
        depths_ptr[camera_idx * N + gaussian_idx] = 0.0
        # Set conics to 0 for culled Gaussians
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 0] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 1] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 2] = 0.0
        return

    # Opacity-based culling (matches GSplat CUDA kernel)
    comptime ALPHA_THRESHOLD: Float32 = 1.0 / 255.0  # Standard 3DGS threshold
    var opacity = opacities_ptr[gaussian_idx]
    if opacity < ALPHA_THRESHOLD:
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0
        # Set means2d to 0 instead of NaN for culled Gaussians
        means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0.0
        means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0.0
        depths_ptr[camera_idx * N + gaussian_idx] = 0.0
        # Set conics to 0 for culled Gaussians
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 0] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 1] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 2] = 0.0
        return

    ########### Quaternion to Rotation Matrix ########### FIXME: Replace by function (not possible it seems right now)
    var R = LayoutTensor[DType.float32, Layout.row_major(3, 3), MutAnyOrigin].stack_allocation()
    # Extract quaternion components
    var w = quats_ptr[gaussian_idx * 4 + 0]
    var x = quats_ptr[gaussian_idx * 4 + 1]
    var y = quats_ptr[gaussian_idx * 4 + 2]
    var z = quats_ptr[gaussian_idx * 4 + 3]

    # Normalize quaternion
    var norm_sq = x * x + y * y + z * z + w * w
    var inv_norm = 1.0 / sqrt(norm_sq)
    x *= inv_norm
    y *= inv_norm
    z *= inv_norm
    w *= inv_norm

    # Compute intermediate values
    var x2 = x * x
    var y2 = y * y
    var z2 = z * z
    var xy = x * y
    var xz = x * z
    var yz = y * z
    var wx = w * x
    var wy = w * y
    var wz = w * z

    R[0, 0] = 1.0 - 2.0 * (y2 + z2)
    R[0, 1] = 2.0 * (xy - wz)
    R[0, 2] = 2.0 * (xz + wy)

    R[1, 0] = 2.0 * (xy + wz)
    R[1, 1] = 1.0 - 2.0 * (x2 + z2)
    R[1, 2] = 2.0 * (yz - wx)

    R[2, 0] = 2.0 * (xz - wy)
    R[2, 1] = 2.0 * (yz + wx)
    R[2, 2] = 1.0 - 2.0 * (x2 + y2)


    ########### Rotation Matrix to Covariance Matrix ########### FIXME: Replace by function (not possible it seems right now)
    # S is diagonal so M = R * S simplifies to M[i,j] = R[i,j] * scale[j]
    var M = LayoutTensor[DType.float32, Layout.row_major(3, 3), MutAnyOrigin].stack_allocation()
    for i in range(3):
        for j in range(3):
            M[i, j] = R[i, j] * scales_ptr[gaussian_idx * 3 + j]

    var covar = LayoutTensor[DType.float32, Layout.row_major(3, 3), MutAnyOrigin].stack_allocation()
    for i in range(3):
        for j in range(3):
            covar[i, j] = 0.0
            for k in range(3):
                covar[i, j] += M[i, k] * M[j, k]


    ########### Covariance World to Camera ###########
    # covar_c = R_view * covar * R_view^T
    var covar_c = LayoutTensor[DType.float32, Layout.row_major(3, 3), MutAnyOrigin].stack_allocation()
    for i in range(3):
        for j in range(3):
            var tmp: Float32 = 0.0
            for l in range(3):
                var tmp2: Float32 = 0.0
                for k in range(3):
                    tmp2 += R_view[i, k][0] * covar[k, l][0]
                tmp += tmp2 * R_view[j, l][0]

            covar_c[i, j] = tmp

    ########### Pinhole Camera Projection ########### FIXME: Replace by function (not possible it seems right now)
    # Read image dimensions from ks (indices 4 and 5)
    var image_width  = Int(ks_ptr[camera_idx * 11 + 4])
    var image_height = Int(ks_ptr[camera_idx * 11 + 5])

    # Read camera intrinsics
    var fx: Float32 = ks_ptr[camera_idx * 11 + 0]
    var fy: Float32 = ks_ptr[camera_idx * 11 + 1]
    var cx: Float32 = ks_ptr[camera_idx * 11 + 2]
    var cy: Float32 = ks_ptr[camera_idx * 11 + 3]

    # 2D projection
    var mean2d = LayoutTensor[DType.float32, Layout.row_major(2), MutAnyOrigin].stack_allocation()

    var tan_fov_x: Float32 = 0.5 * Float32(image_width) / fx
    var tan_fov_y: Float32 = 0.5 * Float32(image_height) / fy
    var lim_x_pos: Float32 = (Float32(image_width) - cx) / fx + 0.3 * tan_fov_x
    var lim_x_neg: Float32 = cx / fx + 0.3 * tan_fov_x
    var lim_y_pos: Float32 = (Float32(image_height) - cy) / fy + 0.3 * tan_fov_y
    var lim_y_neg: Float32 = cy / fy + 0.3 * tan_fov_y

    var rz: Float32 = 1.0 / mean_c[2][0]
    var rz2: Float32 = rz * rz

    var tx: Float32 = mean_c[2][0] * min(lim_x_pos, max(-lim_x_neg, mean_c[0][0] * rz))
    var ty: Float32 = mean_c[2][0] * min(lim_y_pos, max(-lim_y_neg, mean_c[1][0] * rz))

    var J = LayoutTensor[DType.float32, Layout.row_major(2, 3), MutAnyOrigin].stack_allocation()
    J[0, 0] = fx * rz
    J[0, 1] = 0.0
    J[0, 2] = -fx * tx * rz2
    J[1, 0] = 0.0
    J[1, 1] = fy * rz
    J[1, 2] = -fy * ty * rz2

    # cov2d = J (2x3) * covar_c (3x3) * J^T (3x2) = (2x2)
    var cov2d = LayoutTensor[DType.float32, Layout.row_major(2, 2), MutAnyOrigin].stack_allocation()
    for i in range(2):
        for j in range(2):
            var total_sum: Float32 = 0.0
            for l in range(3):
                var temp_il: Float32 = 0.0
                for k in range(3):
                    temp_il += J[i, k][0] * covar_c[k, l][0]
                total_sum += temp_il * J[j, l][0]

            cov2d[i, j] = total_sum

    mean2d[0] = fx * mean_c[0] * rz + cx
    mean2d[1] = fy * mean_c[1] * rz + cy

    # Add eps2d to diagonal to prevent gaussians from being too small (to match gsplat)
    comptime eps2d: Float32 = 0.3
    cov2d[0, 0] += eps2d
    cov2d[1, 1] += eps2d

    ########### Opacity-aware radius calculation (matches CUDA kernel) ###########
    var extend: Float32 = 3.33  # default extend
    var opacity2 = opacities_ptr[gaussian_idx]  # Use different name to avoid duplicate
    if opacity2 >= ALPHA_THRESHOLD:
        # Compute opacity-aware bounding box: extend = min(extend, sqrt(2.0f * log(opacity / ALPHA_THRESHOLD)))
        var log_ratio = log(opacity2 / ALPHA_THRESHOLD)
        var opacity_extend = sqrt(2.0 * log_ratio)
        # Manual min implementation since min is not available in Mojo math
        # Extract scalar value from SIMD type
        if opacity_extend[0] < extend:
            extend = opacity_extend[0]

    var radius_x: Float32 = ceil(extend * sqrt(cov2d[0, 0][0]))
    var radius_y: Float32 = ceil(extend * sqrt(cov2d[1, 1][0]))

    if radius_x <= radius_clip and radius_y <= radius_clip:
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0
        # Set all outputs to 0 for culled Gaussians (matches GSplat behavior)
        means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0.0
        means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0.0
        depths_ptr[camera_idx * N + gaussian_idx] = 0.0  # Zero depth for culled Gaussians
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 0] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 1] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 2] = 0.0
        return

    # Viewport culling
    if mean2d[0] + radius_x <= 0 or mean2d[0] - radius_x >= Float32(image_width) or mean2d[1] + radius_y <= 0 or mean2d[1] - radius_y >= Float32(image_height):
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0
        return

    ########### Conic calculation (inlined 2x2 matrix inverse) ###########
    var det_val: Float32 = cov2d[0, 0][0] * cov2d[1, 1][0] - cov2d[0, 1][0] * cov2d[1, 0][0]
    var inv_det: Float32 = Float32(1.0) / det_val
    conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 0] = cov2d[1, 1][0] * inv_det   # cov2d[1,1] / det
    conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 1] = -cov2d[0, 1][0] * inv_det  # -cov2d[0,1] / det
    conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 2] = cov2d[0, 0][0] * inv_det   # cov2d[0,0] / det

    radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = radius_x.cast[DType.int32]()
    radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = radius_y.cast[DType.int32]()
    means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = mean2d[0][0]
    means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = mean2d[1][0]
    depths_ptr[camera_idx * N + gaussian_idx] = mean_c[2][0]  # Depth is positive distance along viewing direction


@compiler.register("project_gaussians")
struct ProjectGaussians:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        # Outputs
        means2d: OutputTensor[dtype=DType.float32, rank=3],
        conics:  OutputTensor[dtype=DType.float32, rank=3],
        depths:  OutputTensor[dtype=DType.float32, rank=2],
        radii:   OutputTensor[dtype=DType.int32,   rank=3],
        # Inputs
        means3d:       InputTensor[dtype=DType.float32, rank=2],
        scales:        InputTensor[dtype=DType.float32, rank=2],
        quats:         InputTensor[dtype=DType.float32, rank=2],
        opacities:     InputTensor[dtype=DType.float32, rank=1],
        view_matrices: InputTensor[dtype=DType.float32, rank=3],
        ks:            InputTensor[dtype=DType.float32, rank=2],  # (C, 11) now
        # Context
        ctx: DeviceContextPtr
    ) raises:
        var N = means3d.dim_size(0)
        var C = view_matrices.dim_size(0)

        # Get raw pointers from tensors
        var means3d_ptr       = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](means3d.to_layout_tensor().ptr)
        var scales_ptr        = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](scales.to_layout_tensor().ptr)
        var quats_ptr         = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](quats.to_layout_tensor().ptr)
        var opacities_ptr     = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](opacities.to_layout_tensor().ptr)
        var view_matrices_ptr = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](view_matrices.to_layout_tensor().ptr)
        var ks_ptr            = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](ks.to_layout_tensor().ptr)
        var radii_ptr         = rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](radii.to_layout_tensor().ptr)
        var means2d_ptr       = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](means2d.to_layout_tensor().ptr)
        var depths_ptr        = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](depths.to_layout_tensor().ptr)
        var conics_ptr        = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](conics.to_layout_tensor().ptr)

        @parameter
        if target == "cpu":
            raise Error("ProjectGaussians CPU target not implemented yet.")
        elif target == "gpu":
            # Get GPU context
            var gpu_ctx = ctx.get_device_context()

            # Define grid and block dimensions for the kernel launch
            # Kernel processes N * C threads total
            var grid  = (ceildiv(N * C, block_size))
            var block = (block_size)

            gpu_ctx.enqueue_function_unchecked[project_ewa_kernel](
                means3d_ptr, scales_ptr, quats_ptr, opacities_ptr,
                view_matrices_ptr, ks_ptr,
                radii_ptr, means2d_ptr, depths_ptr, conics_ptr,
                N, C,
                grid_dim=grid, block_dim=block,
            )

            # gpu_ctx.synchronize()
