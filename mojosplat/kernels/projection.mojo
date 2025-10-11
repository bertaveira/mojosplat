import compiler
from gpu import thread_idx, block_idx, barrier
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from utils.index import IndexList
from math import sqrt, ceil, ceildiv, log
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

alias radius_clip: Float32 = 0.0
alias block_size: Int = 256


fn project_ewa_kernel[
    C: Int,
    N: Int,
    # TODO: Support batching different scenes with same number of gaussians?
](
    means3d: LayoutTensor[DType.float32, Layout.row_major(N, 3), MutableAnyOrigin],
    scales: LayoutTensor[DType.float32, Layout.row_major(N, 3), MutableAnyOrigin],
    quats: LayoutTensor[DType.float32, Layout.row_major(N, 4), MutableAnyOrigin],
    opacities: LayoutTensor[DType.float32, Layout.row_major(N), MutableAnyOrigin],
    view_matrices: LayoutTensor[DType.float32, Layout.row_major(C, 4, 4), MutableAnyOrigin],
    ks: LayoutTensor[DType.float32, Layout.row_major(C, 9), MutableAnyOrigin], # camera params: [fx, fy, cx, cy, k1, k2, k3, k4, k5]
    image_width: Int,
    image_height: Int,
    # Outputs - Using placeholder LayoutTensors
    radii: LayoutTensor[DType.int32, Layout.row_major(C, N, 2), MutableAnyOrigin],
    means2d: LayoutTensor[DType.float32, Layout.row_major(C, N, 2), MutableAnyOrigin],
    depths: LayoutTensor[DType.float32, Layout.row_major(C, N), MutableAnyOrigin],
    conics: LayoutTensor[DType.float32, Layout.row_major(C, N, 3), MutableAnyOrigin],
):
    # This kernel is paralellized over N * C
    var idx = thread_idx.x + block_idx.x * block_size
    var gaussian_idx = Int(idx % N)
    var camera_idx = Int(idx // N)

    if idx >= N * C:
        return

    # var mean = means3d.tile[1, 3](gaussian_idx, 0)
    # var scale = scales.tile[1, 3](gaussian_idx, 0)
    # var quat = quats.tile[1, 4](gaussian_idx, 0)
    var mean = means3d.slice_1d[Slice(0, 3), IndexList[1](1)](gaussian_idx) # [3] LayoutTensor
    var scale = scales.slice_1d[Slice(0, 3), IndexList[1](1)](gaussian_idx) # [3] LayoutTensor
    var quat = quats.slice_1d[Slice(0, 4), IndexList[1](1)](gaussian_idx) # [4] LayoutTensor
    var opacity = opacities[gaussian_idx]
    
    # Extract the 4x4 view matrix for the current camera
    var view_matrix = tb[DType.float32]().row_major[4, 4]().alloc()
    for i in range(4):
        for j in range(4):
            view_matrix[i, j] = view_matrices[camera_idx, i, j]

    ########### Gaussian Wolrd to Camera ########### FIXME: Replace by function (not possible it seems right now)
    # mean_c = R * mean + t
    var mean_c = tb[DType.float32]().row_major[3]().alloc()
    for i in range(3):
        mean_c[i] = 0.0
        for j in range(3):
            mean_c[i] += view_matrix[i, j] * mean[j]
        mean_c[i] += view_matrix[i, 3]

    alias near_plane: Float32 = 0.1
    alias far_plane: Float32 = 1e10
    if mean_c[2][0] <= near_plane or mean_c[2][0] >= far_plane:
        radii[camera_idx, gaussian_idx, 0] = 0
        radii[camera_idx, gaussian_idx, 1] = 0
        # Set means2d to 0 instead of NaN for culled Gaussians  
        # means2d[camera_idx, gaussian_idx, 0] = 0.0
        # means2d[camera_idx, gaussian_idx, 1] = 0.0
        # depths[camera_idx, gaussian_idx] = 0.0
        # # Set conics to 0 for culled Gaussians
        # conics[camera_idx, gaussian_idx, 0] = 0.0
        # conics[camera_idx, gaussian_idx, 1] = 0.0
        # conics[camera_idx, gaussian_idx, 2] = 0.0
        return

    ########### Quaternion to Rotation Matrix ########### FIXME: Replace by function (not possible it seems right now)
    var R = tb[DType.float32]().row_major[3, 3]().alloc()
    # Extract quaternion components
    var w = quat[0]
    var x = quat[1] 
    var y = quat[2]
    var z = quat[3]
    
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
    R[0, 1] = 2.0 * (xy + wz)
    R[0, 2] = 2.0 * (xz - wy)
    
    R[1, 0] = 2.0 * (xy - wz)
    R[1, 1] = 1.0 - 2.0 * (x2 + z2)
    R[1, 2] = 2.0 * (yz + wx)

    R[2, 0] = 2.0 * (xz + wy)
    R[2, 1] = 2.0 * (yz - wx)
    R[2, 2] = 1.0 - 2.0 * (x2 + y2)


    ########### Rotation Matrix to Covariance Matrix ########### FIXME: Replace by function (not possible it seems right now)
    # C = R * S * S * Rt

    # create scaling matrix # TODO: avoid using tensor aloc for faster computation?
    var S = tb[DType.float32]().row_major[3, 3]().alloc()
    for i in range(3):
        for j in range(3):
            if i == j:
                S[i, j] = scale[i]
            else:
                S[i, j] = 0.0
    
    # M = R * S
    var M = tb[DType.float32]().row_major[3, 3]().alloc()
    for i in range(3):
        for j in range(3):
            M[i, j] = 0.0
            for k in range(3):
                M[i, j] += R[i, k] * S[k, j]
    
    # C = M * M^T
    var covar = tb[DType.float32]().row_major[3, 3]().alloc()
    for i in range(3):
        for j in range(3):
            covar[i, j] = 0.0
            for k in range(3):
                covar[i, j] += M[i, k] * M[j, k]
    
    
    ########### Covariance World to Camera ###########
    # Apply camera rotation to covariance: covar_c = R_view * covar * R_view^T
    var R_view = tb[DType.float32]().row_major[3, 3]().alloc()
    for i in range(3):
        for j in range(3):
            R_view[i, j] = view_matrix[i, j]
    var covar_c = tb[DType.float32]().row_major[3, 3]().alloc()
    for i in range(3):
        for j in range(3):
            var tmp: Float32 = 0.0
            for l in range(3):
                # (R_view * covar)[i, l]
                var tmp2: Float32 = 0.0
                for k in range(3):
                    tmp2 += R_view[i, k][0] * covar[k, l][0]
                # Multiply by R_view^T
                tmp += tmp2 * R_view[l, j][0]
            covar_c[i, j] = tmp

    ########### Pinhole Camera Projection ########### FIXME: Replace by function (not possible it seems right now)
    # 2D projection
    var mean2d = tb[DType.float32]().row_major[2]().alloc()
    
    var tan_fov_x: Float32 = 0.5 * Float32(image_width) / ks[camera_idx, 0][0]
    var tan_fov_y: Float32 = 0.5 * Float32(image_height) / ks[camera_idx, 1][0]
    var lim_x_pos: Float32 = (Float32(image_width) - ks[camera_idx, 2][0]) / ks[camera_idx, 0][0] + 0.3 * tan_fov_x
    var lim_x_neg: Float32 = ks[camera_idx, 2][0] / ks[camera_idx, 0][0] + 0.3 * tan_fov_x
    var lim_y_pos: Float32 = (Float32(image_height) - ks[camera_idx, 3][0]) / ks[camera_idx, 1][0] + 0.3 * tan_fov_y
    var lim_y_neg: Float32 = ks[camera_idx, 3][0] / ks[camera_idx, 1][0] + 0.3 * tan_fov_y

    var rz: Float32 = 1.0 / mean_c[2][0]
    var rz2: Float32 = rz * rz

    var tx: Float32 = mean_c[2][0] * min(lim_x_pos, max(-lim_x_neg, mean_c[0][0] * rz))
    var ty: Float32 = mean_c[2][0] * min(lim_y_pos, max(-lim_y_neg, mean_c[1][0] * rz))
    
    # Jacobian d([fx*x/z + cx, fy*y/z + cy]) / d([x, y, z]) has shape (2, 3)
    var J = tb[DType.float32]().row_major[2, 3]().alloc()
    # Row 0
    J[0, 0] = ks[camera_idx, 0] * rz                 # fx / z
    J[0, 1] = 0.0                                     # 0
    J[0, 2] = -ks[camera_idx, 0] * tx * rz2           # -fx * x / z^2
    # Row 1
    J[1, 0] = 0.0                                     # 0
    J[1, 1] = ks[camera_idx, 1] * rz                  # fy / z
    J[1, 2] = -ks[camera_idx, 1] * ty * rz2           # -fy * y / z^2


    # cov2d = J * covar_c * J^T
    var cov2d = tb[DType.float32]().row_major[2, 2]().alloc()
    for i in range(2):
        for j in range(2):
            var total_sum: Float32 = 0.0
            for l in range(3):
                var temp_il: Float32 = 0.0
                for k in range(3):
                    temp_il += J[i, k][0] * covar_c[k, l][0]
                total_sum += temp_il * J[j, l][0]
            cov2d[i, j] = total_sum

    mean2d[0] = ks[camera_idx, 0] * mean_c[0] * rz + ks[camera_idx, 2]
    mean2d[1] = ks[camera_idx, 1] * mean_c[1] * rz + ks[camera_idx, 3]  

    ########### Add blur to covariance ###########
    var det_orig: Float32 = cov2d[0, 0][0] * cov2d[1, 1][0] - cov2d[0, 1][0] * cov2d[1, 0][0]
    # Add eps2d to diagonal to prevent gaussians from being too small (to match gsplat)
    alias eps2d: Float32 = 0.3
    cov2d[0, 0] += eps2d
    cov2d[1, 1] += eps2d
    var det_blur: Float32 = cov2d[0, 0][0] * cov2d[1, 1][0] - cov2d[0, 1][0] * cov2d[1, 0][0]

    if det_blur <= 0.0:
        radii[camera_idx, gaussian_idx, 0] = 0
        radii[camera_idx, gaussian_idx, 1] = 0
        return

    ########### Opacity-aware radius calculation (matches CUDA kernel) ###########
    alias ALPHA_THRESHOLD: Float32 = 1.0 / 255.0  # Standard 3DGS threshold
    var extend: Float32 = 3.33  # default extend
    if opacity < ALPHA_THRESHOLD:
        radii[camera_idx, gaussian_idx, 0] = 0
        radii[camera_idx, gaussian_idx, 1] = 0
        return
    
    extend = min(Float32(extend), sqrt(Float32(2.0) * log(opacity / ALPHA_THRESHOLD)[0]))
    
    var radius_x: Float32 = ceil(extend * sqrt(cov2d[0, 0][0]))
    var radius_y: Float32 = ceil(extend * sqrt(cov2d[1, 1][0]))

    if radius_x <= radius_clip and radius_y <= radius_clip:
        radii[camera_idx, gaussian_idx, 0] = 0
        radii[camera_idx, gaussian_idx, 1] = 0
        # Set all outputs to 0 for culled Gaussians (matches GSplat behavior)
        means2d[camera_idx, gaussian_idx, 0] = 0.0
        means2d[camera_idx, gaussian_idx, 1] = 0.0
        depths[camera_idx, gaussian_idx] = 0.0  # Zero depth for culled Gaussians
        conics[camera_idx, gaussian_idx, 0] = 0.0
        conics[camera_idx, gaussian_idx, 1] = 0.0
        conics[camera_idx, gaussian_idx, 2] = 0.0
        return

    # Viewport culling
    if mean2d[0] + radius_x <= 0 or mean2d[0] - radius_x >= image_width or mean2d[1] + radius_y <= 0 or mean2d[1] - radius_y >= image_height:
        radii[camera_idx, gaussian_idx, 0] = 0
        radii[camera_idx, gaussian_idx, 1] = 0
        return
    
    ########### Conic calculation ###########
    var (a, b, c, d) = inverse_2x2(cov2d[0, 0][0], cov2d[0, 1][0], cov2d[1, 0][0], cov2d[1, 1][0])
    conics[camera_idx, gaussian_idx, 0] = a
    conics[camera_idx, gaussian_idx, 1] = b
    conics[camera_idx, gaussian_idx, 2] = d

    radii[camera_idx, gaussian_idx, 0] = Int(radius_x)
    radii[camera_idx, gaussian_idx, 1] = Int(radius_y)
    means2d[camera_idx, gaussian_idx, 0] = mean2d[0]
    means2d[camera_idx, gaussian_idx, 1] = mean2d[1]
    depths[camera_idx, gaussian_idx] = mean_c[2][0]  # Depth is positive distance along viewing direction?


fn inverse_2x2(
    a: Float32,
    b: Float32,
    c: Float32,
    d: Float32,
) -> (Float32, Float32, Float32, Float32):
    var det = a * d - b * c
    var inv_det = 1.0 / det
    return (d * inv_det, -b * inv_det, -c * inv_det, a * inv_det)


@compiler.register("project_gaussians")
struct ProjectGaussians:
    @staticmethod
    fn execute[
        C: Int,
        N: Int,
        image_width: Int,
        image_height: Int,
        target: StaticString,
    ](
        # Outputs
        means2d: OutputTensor[dtype=DType.float32, rank=3],
        conics: OutputTensor[dtype=DType.float32, rank=3],
        depths: OutputTensor[dtype=DType.float32, rank=2],
        radii: OutputTensor[dtype=DType.int32, rank=3],
        # Inputs
        means3d: InputTensor[dtype=DType.float32, rank=2],
        scales: InputTensor[dtype=DType.float32, rank=2],
        quats: InputTensor[dtype=DType.float32, rank=2],
        opacities: InputTensor[dtype=DType.float32, rank=1],
        view_matrices: InputTensor[dtype=DType.float32, rank=3],
        ks: InputTensor[dtype=DType.float32, rank=2],
        # Context
        ctx: DeviceContextPtr
    ) raises:
        # Outputs
        means2d_tensor = means2d.to_layout_tensor()
        conics_tensor = conics.to_layout_tensor()
        depths_tensor = depths.to_layout_tensor()
        radii_tensor = radii.to_layout_tensor()

        # Inputs
        means3d_tensor = means3d.to_layout_tensor()
        scales_tensor = scales.to_layout_tensor()
        quats_tensor = quats.to_layout_tensor()
        opacities_tensor = opacities.to_layout_tensor()
        view_matrices_tensor = view_matrices.to_layout_tensor()
        ks_tensor = ks.to_layout_tensor()

        @parameter
        if target == "cpu":
            raise Error("Rasterize3DGS CPU target not implemented yet.")
        elif target == "gpu":
            # Get GPU context
            var gpu_ctx = ctx.get_device_context()

            # Define grid and block dimensions for the kernel launch
            # Kernel processes N * C threads total
            var grid = (ceildiv(N * C, block_size))
            var block = (block_size)

            gpu_ctx.enqueue_function[
                project_ewa_kernel[
                    C, N,
                ]
            ](
                means3d_tensor,
                scales_tensor,
                quats_tensor,
                opacities_tensor,
                view_matrices_tensor,
                ks_tensor,
                image_width,
                image_height,
                radii_tensor,
                means2d_tensor,
                depths_tensor,
                conics_tensor,
                grid_dim=grid,
                block_dim=block,
            )

            # gpu_ctx.synchronize()