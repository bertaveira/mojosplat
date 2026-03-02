import compiler
from gpu import thread_idx, block_idx, barrier
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

    # Extract R (3x3) and t (3) from the view matrix (row-major 4x4, skip bottom row)
    var Rv00: Float32 = view_matrices_ptr[camera_idx * 16 + 0]
    var Rv01: Float32 = view_matrices_ptr[camera_idx * 16 + 1]
    var Rv02: Float32 = view_matrices_ptr[camera_idx * 16 + 2]
    var tv0:  Float32 = view_matrices_ptr[camera_idx * 16 + 3]
    var Rv10: Float32 = view_matrices_ptr[camera_idx * 16 + 4]
    var Rv11: Float32 = view_matrices_ptr[camera_idx * 16 + 5]
    var Rv12: Float32 = view_matrices_ptr[camera_idx * 16 + 6]
    var tv1:  Float32 = view_matrices_ptr[camera_idx * 16 + 7]
    var Rv20: Float32 = view_matrices_ptr[camera_idx * 16 + 8]
    var Rv21: Float32 = view_matrices_ptr[camera_idx * 16 + 9]
    var Rv22: Float32 = view_matrices_ptr[camera_idx * 16 + 10]
    var tv2:  Float32 = view_matrices_ptr[camera_idx * 16 + 11]

    ########### Gaussian World to Camera ###########
    # mean_c = R_view * mean + t
    var g0: Float32 = means3d_ptr[gaussian_idx * 3 + 0]
    var g1: Float32 = means3d_ptr[gaussian_idx * 3 + 1]
    var g2: Float32 = means3d_ptr[gaussian_idx * 3 + 2]
    var mc0: Float32 = Rv00 * g0 + Rv01 * g1 + Rv02 * g2 + tv0
    var mc1: Float32 = Rv10 * g0 + Rv11 * g1 + Rv12 * g2 + tv1
    var mc2: Float32 = Rv20 * g0 + Rv21 * g1 + Rv22 * g2 + tv2

    comptime near_plane: Float32 = 0.1
    if mc2 <= near_plane:
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0
        means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0.0
        means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0.0
        depths_ptr[camera_idx * N + gaussian_idx] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 0] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 1] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 2] = 0.0
        return

    # Opacity-based culling (matches GSplat CUDA kernel)
    comptime ALPHA_THRESHOLD: Float32 = 1.0 / 255.0
    var opacity: Float32 = opacities_ptr[gaussian_idx]
    if opacity < ALPHA_THRESHOLD:
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0
        means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0.0
        means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0.0
        depths_ptr[camera_idx * N + gaussian_idx] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 0] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 1] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 2] = 0.0
        return

    ########### Quaternion to Rotation Matrix ###########
    var w: Float32 = quats_ptr[gaussian_idx * 4 + 0]
    var x: Float32 = quats_ptr[gaussian_idx * 4 + 1]
    var y: Float32 = quats_ptr[gaussian_idx * 4 + 2]
    var z: Float32 = quats_ptr[gaussian_idx * 4 + 3]

    # Normalize quaternion
    var norm_sq: Float32 = x * x + y * y + z * z + w * w
    var inv_norm: Float32 = 1.0 / sqrt(norm_sq)
    x *= inv_norm
    y *= inv_norm
    z *= inv_norm
    w *= inv_norm

    var x2: Float32 = x * x
    var y2: Float32 = y * y
    var z2: Float32 = z * z
    var xy: Float32 = x * y
    var xz: Float32 = x * z
    var yz: Float32 = y * z
    var wx: Float32 = w * x
    var wy: Float32 = w * y
    var wz: Float32 = w * z

    var Q00: Float32 = 1.0 - 2.0 * (y2 + z2)
    var Q01: Float32 = 2.0 * (xy - wz)
    var Q02: Float32 = 2.0 * (xz + wy)
    var Q10: Float32 = 2.0 * (xy + wz)
    var Q11: Float32 = 1.0 - 2.0 * (x2 + z2)
    var Q12: Float32 = 2.0 * (yz - wx)
    var Q20: Float32 = 2.0 * (xz - wy)
    var Q21: Float32 = 2.0 * (yz + wx)
    var Q22: Float32 = 1.0 - 2.0 * (x2 + y2)

    ########### Rotation Matrix to Covariance Matrix ###########
    # S is diagonal so M = Q * S simplifies to M[i,j] = Q[i,j] * scale[j]
    var s0: Float32 = scales_ptr[gaussian_idx * 3 + 0]
    var s1: Float32 = scales_ptr[gaussian_idx * 3 + 1]
    var s2: Float32 = scales_ptr[gaussian_idx * 3 + 2]
    var M00: Float32 = Q00 * s0;  var M01: Float32 = Q01 * s1;  var M02: Float32 = Q02 * s2
    var M10: Float32 = Q10 * s0;  var M11: Float32 = Q11 * s1;  var M12: Float32 = Q12 * s2
    var M20: Float32 = Q20 * s0;  var M21: Float32 = Q21 * s1;  var M22: Float32 = Q22 * s2

    # covar = M @ M^T  (symmetric)
    var Cv00: Float32 = M00*M00 + M01*M01 + M02*M02
    var Cv01: Float32 = M00*M10 + M01*M11 + M02*M12
    var Cv02: Float32 = M00*M20 + M01*M21 + M02*M22
    var Cv10: Float32 = Cv01
    var Cv11: Float32 = M10*M10 + M11*M11 + M12*M12
    var Cv12: Float32 = M10*M20 + M11*M21 + M12*M22
    var Cv20: Float32 = Cv02
    var Cv21: Float32 = Cv12
    var Cv22: Float32 = M20*M20 + M21*M21 + M22*M22

    ########### Covariance World to Camera ###########
    # covar_c = R_view @ covar @ R_view^T
    # Step 1: tmp = R_view @ covar
    var tmp00: Float32 = Rv00*Cv00 + Rv01*Cv10 + Rv02*Cv20
    var tmp01: Float32 = Rv00*Cv01 + Rv01*Cv11 + Rv02*Cv21
    var tmp02: Float32 = Rv00*Cv02 + Rv01*Cv12 + Rv02*Cv22
    var tmp10: Float32 = Rv10*Cv00 + Rv11*Cv10 + Rv12*Cv20
    var tmp11: Float32 = Rv10*Cv01 + Rv11*Cv11 + Rv12*Cv21
    var tmp12: Float32 = Rv10*Cv02 + Rv11*Cv12 + Rv12*Cv22
    var tmp20: Float32 = Rv20*Cv00 + Rv21*Cv10 + Rv22*Cv20
    var tmp21: Float32 = Rv20*Cv01 + Rv21*Cv11 + Rv22*Cv21
    var tmp22: Float32 = Rv20*Cv02 + Rv21*Cv12 + Rv22*Cv22
    # Step 2: covar_c = tmp @ R_view^T  (Cc[i,j] = sum_l tmp[i,l] * R_view[j,l])
    var Cc00: Float32 = tmp00*Rv00 + tmp01*Rv01 + tmp02*Rv02
    var Cc01: Float32 = tmp00*Rv10 + tmp01*Rv11 + tmp02*Rv12
    var Cc02: Float32 = tmp00*Rv20 + tmp01*Rv21 + tmp02*Rv22
    var Cc10: Float32 = tmp10*Rv00 + tmp11*Rv01 + tmp12*Rv02
    var Cc11: Float32 = tmp10*Rv10 + tmp11*Rv11 + tmp12*Rv12
    var Cc12: Float32 = tmp10*Rv20 + tmp11*Rv21 + tmp12*Rv22
    var Cc20: Float32 = tmp20*Rv00 + tmp21*Rv01 + tmp22*Rv02
    var Cc21: Float32 = tmp20*Rv10 + tmp21*Rv11 + tmp22*Rv12
    var Cc22: Float32 = tmp20*Rv20 + tmp21*Rv21 + tmp22*Rv22

    ########### Pinhole Camera Projection ###########
    # Read image dimensions from ks (indices 4 and 5)
    var image_width:  Int = Int(ks_ptr[camera_idx * 11 + 4])
    var image_height: Int = Int(ks_ptr[camera_idx * 11 + 5])

    var fx: Float32 = ks_ptr[camera_idx * 11 + 0]
    var fy: Float32 = ks_ptr[camera_idx * 11 + 1]
    var cx: Float32 = ks_ptr[camera_idx * 11 + 2]
    var cy: Float32 = ks_ptr[camera_idx * 11 + 3]

    var tan_fov_x: Float32 = 0.5 * Float32(image_width) / fx
    var tan_fov_y: Float32 = 0.5 * Float32(image_height) / fy
    var lim_x_pos: Float32 = (Float32(image_width) - cx) / fx + 0.3 * tan_fov_x
    var lim_x_neg: Float32 = cx / fx + 0.3 * tan_fov_x
    var lim_y_pos: Float32 = (Float32(image_height) - cy) / fy + 0.3 * tan_fov_y
    var lim_y_neg: Float32 = cy / fy + 0.3 * tan_fov_y

    var rz:  Float32 = 1.0 / mc2
    var rz2: Float32 = rz * rz

    var tx: Float32 = mc2 * min(lim_x_pos, max(-lim_x_neg, mc0 * rz))
    var ty: Float32 = mc2 * min(lim_y_pos, max(-lim_y_neg, mc1 * rz))

    var J00: Float32 = fx * rz
    var J02: Float32 = -fx * tx * rz2
    var J11: Float32 = fy * rz
    var J12: Float32 = -fy * ty * rz2
    # J01 = J10 = 0.0

    # cov2d = J (2x3) @ covar_c (3x3) @ J^T (3x2)
    # Step 1: t2 = J @ covar_c  (only non-zero J entries: J[0,0], J[0,2], J[1,1], J[1,2])
    var t2_00: Float32 = J00*Cc00 + J02*Cc20
    var t2_01: Float32 = J00*Cc01 + J02*Cc21
    var t2_02: Float32 = J00*Cc02 + J02*Cc22
    var t2_10: Float32 = J11*Cc10 + J12*Cc20
    var t2_11: Float32 = J11*Cc11 + J12*Cc21
    var t2_12: Float32 = J11*Cc12 + J12*Cc22
    # Step 2: cov2d = t2 @ J^T  (c2[i,j] = sum_l t2[i,l]*J[j,l])
    var c2_00: Float32 = t2_00*J00 + t2_02*J02
    var c2_01: Float32 = t2_01*J11 + t2_02*J12
    var c2_10: Float32 = t2_10*J00 + t2_12*J02
    var c2_11: Float32 = t2_11*J11 + t2_12*J12

    var m2d_x: Float32 = fx * mc0 * rz + cx
    var m2d_y: Float32 = fy * mc1 * rz + cy

    # Add eps2d to diagonal to prevent gaussians from being too small (to match gsplat)
    comptime eps2d: Float32 = 0.3
    c2_00 += eps2d
    c2_11 += eps2d

    ########### Opacity-aware radius calculation (matches CUDA kernel) ###########
    var extend: Float32 = 3.33
    if opacity >= ALPHA_THRESHOLD:
        var log_ratio = log(opacity / ALPHA_THRESHOLD)
        var opacity_extend = sqrt(2.0 * log_ratio)
        if opacity_extend[0] < extend:
            extend = opacity_extend[0]

    var radius_x: Float32 = ceil(extend * sqrt(c2_00))
    var radius_y: Float32 = ceil(extend * sqrt(c2_11))

    if radius_x <= radius_clip and radius_y <= radius_clip:
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0
        means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0.0
        means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0.0
        depths_ptr[camera_idx * N + gaussian_idx] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 0] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 1] = 0.0
        conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 2] = 0.0
        return

    # Viewport culling
    if m2d_x + radius_x <= 0 or m2d_x - radius_x >= Float32(image_width) or m2d_y + radius_y <= 0 or m2d_y - radius_y >= Float32(image_height):
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0
        return

    ########### Conic calculation (inlined 2x2 matrix inverse) ###########
    var det_val: Float32 = c2_00 * c2_11 - c2_01 * c2_10
    var inv_det: Float32 = Float32(1.0) / det_val
    conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 0] = c2_11 * inv_det   # cov2d[1,1] / det
    conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 1] = -c2_01 * inv_det  # -cov2d[0,1] / det
    conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 2] = c2_00 * inv_det   # cov2d[0,0] / det

    radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = radius_x.cast[DType.int32]()
    radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = radius_y.cast[DType.int32]()
    means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = m2d_x
    means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = m2d_y
    depths_ptr[camera_idx * N + gaussian_idx] = mc2  # Depth is positive distance along viewing direction


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
        ks:            InputTensor[dtype=DType.float32, rank=2],  # (C, 11)
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
            var gpu_ctx = ctx.get_device_context()
            var grid  = (ceildiv(N * C, block_size))
            var block = (block_size)

            gpu_ctx.enqueue_function_unchecked[project_ewa_kernel](
                means3d_ptr, scales_ptr, quats_ptr, opacities_ptr,
                view_matrices_ptr, ks_ptr,
                radii_ptr, means2d_ptr, depths_ptr, conics_ptr,
                N, C,
                grid_dim=grid, block_dim=block,
            )
