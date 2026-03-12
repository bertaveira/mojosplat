import compiler
from gpu import thread_idx, block_idx, barrier
from math import sqrt, ceil, ceildiv, log, rsqrt
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer

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
    var far_plane: Float32 = ks_ptr[camera_idx * 11 + 10]
    if mc2 <= near_plane or mc2 >= far_plane:
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0
        return

    # Opacity-based culling (matches GSplat CUDA kernel)
    comptime ALPHA_THRESHOLD: Float32 = 1.0 / 255.0
    var opacity: Float32 = opacities_ptr[gaussian_idx]
    if opacity < ALPHA_THRESHOLD:
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0
        return

    ########### Quaternion to Rotation Matrix ###########
    var w: Float32 = quats_ptr[gaussian_idx * 4 + 0]
    var x: Float32 = quats_ptr[gaussian_idx * 4 + 1]
    var y: Float32 = quats_ptr[gaussian_idx * 4 + 2]
    var z: Float32 = quats_ptr[gaussian_idx * 4 + 3]

    # Normalize quaternion
    var norm_sq: Float32 = x * x + y * y + z * z + w * w
    var inv_norm: Float32 = rsqrt(norm_sq)
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

    ########### Rotation Matrix to Camera-Space Factor ###########
    # M = Q * S  (S diagonal, so M[i,j] = Q[i,j] * scale[j])
    var s0: Float32 = scales_ptr[gaussian_idx * 3 + 0]
    var s1: Float32 = scales_ptr[gaussian_idx * 3 + 1]
    var s2: Float32 = scales_ptr[gaussian_idx * 3 + 2]
    var M00: Float32 = Q00 * s0;  var M01: Float32 = Q01 * s1;  var M02: Float32 = Q02 * s2
    var M10: Float32 = Q10 * s0;  var M11: Float32 = Q11 * s1;  var M12: Float32 = Q12 * s2
    var M20: Float32 = Q20 * s0;  var M21: Float32 = Q21 * s1;  var M22: Float32 = Q22 * s2

    # Mp = Rv @ M  (fused view-rotation × scale; covar_c = Mp @ Mp^T)
    # After this, Rv and M are no longer needed.
    var Mp00: Float32 = Rv00*M00 + Rv01*M10 + Rv02*M20
    var Mp01: Float32 = Rv00*M01 + Rv01*M11 + Rv02*M21
    var Mp02: Float32 = Rv00*M02 + Rv01*M12 + Rv02*M22
    var Mp10: Float32 = Rv10*M00 + Rv11*M10 + Rv12*M20
    var Mp11: Float32 = Rv10*M01 + Rv11*M11 + Rv12*M21
    var Mp12: Float32 = Rv10*M02 + Rv11*M12 + Rv12*M22
    var Mp20: Float32 = Rv20*M00 + Rv21*M10 + Rv22*M20
    var Mp21: Float32 = Rv20*M01 + Rv21*M11 + Rv22*M21
    var Mp22: Float32 = Rv20*M02 + Rv21*M12 + Rv22*M22

    ########### Pinhole Camera Projection ###########
    var fx: Float32 = ks_ptr[camera_idx * 11 + 0]
    var fy: Float32 = ks_ptr[camera_idx * 11 + 1]
    var cx: Float32 = ks_ptr[camera_idx * 11 + 2]
    var cy: Float32 = ks_ptr[camera_idx * 11 + 3]
    # Read image dimensions as Float32 (ks[4], ks[5]) — avoids 64-bit Int registers
    var image_width_f:  Float32 = ks_ptr[camera_idx * 11 + 4]
    var image_height_f: Float32 = ks_ptr[camera_idx * 11 + 5]
    # ks[6:9] = lim_x_pos, lim_x_neg, lim_y_pos, lim_y_neg (precomputed in Python)
    var lim_x_pos: Float32 = ks_ptr[camera_idx * 11 + 6]
    var lim_x_neg: Float32 = ks_ptr[camera_idx * 11 + 7]
    var lim_y_pos: Float32 = ks_ptr[camera_idx * 11 + 8]
    var lim_y_neg: Float32 = ks_ptr[camera_idx * 11 + 9]

    var rz: Float32 = rsqrt(mc2 * mc2)   # 1/mc2 via rsqrt.approx (mc2 > 0 by near-plane check)

    var tx: Float32 = mc2 * min(lim_x_pos, max(-lim_x_neg, mc0 * rz))
    var ty: Float32 = mc2 * min(lim_y_pos, max(-lim_y_neg, mc1 * rz))

    var J00: Float32 = fx * rz
    var J02: Float32 = -J00 * tx * rz   # = -fx * tx * rz^2, reuses J00
    var J11: Float32 = fy * rz
    var J12: Float32 = -J11 * ty * rz   # = -fy * ty * rz^2, reuses J11
    # J01 = J10 = 0.0

    # cov2d = (J @ Mp) @ (J @ Mp)^T  [identity: J@covar_c@J^T = J@Mp@Mp^T@J^T]
    # J = [[J00, 0, J02], [0, J11, J12]]  (sparse 2x3)
    var r0_0: Float32 = J00*Mp00 + J02*Mp20
    var r0_1: Float32 = J00*Mp01 + J02*Mp21
    var r0_2: Float32 = J00*Mp02 + J02*Mp22
    var r1_0: Float32 = J11*Mp10 + J12*Mp20
    var r1_1: Float32 = J11*Mp11 + J12*Mp21
    var r1_2: Float32 = J11*Mp12 + J12*Mp22
    var c2_00: Float32 = r0_0*r0_0 + r0_1*r0_1 + r0_2*r0_2
    var c2_01: Float32 = r0_0*r1_0 + r0_1*r1_1 + r0_2*r1_2
    var c2_11: Float32 = r1_0*r1_0 + r1_1*r1_1 + r1_2*r1_2

    var m2d_x: Float32 = fx * mc0 * rz + cx
    var m2d_y: Float32 = fy * mc1 * rz + cy

    # Add eps2d to diagonal to prevent gaussians from being too small (to match gsplat)
    comptime eps2d: Float32 = 0.3
    c2_00 += eps2d
    c2_11 += eps2d

    # Early exit if covariance is degenerate (matches gsplat's add_blur det check)
    var det_val: Float32 = c2_00 * c2_11 - c2_01 * c2_01
    if det_val <= 0.0:
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0
        return
    var inv_det: Float32 = Float32(1.0) / det_val

    ########### Radius calculation (opacity-based, matches gsplat) ###########
    var extend: Float32 = min(Float32(3.33), sqrt(2.0 * log(opacity * Float32(255.0)))[0])
    var radius_x: Float32 = ceil(extend * sqrt(c2_00))
    var radius_y: Float32 = ceil(extend * sqrt(c2_11))

    # Viewport culling
    if m2d_x + radius_x <= 0 or m2d_x - radius_x >= image_width_f or m2d_y + radius_y <= 0 or m2d_y - radius_y >= image_height_f:
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = 0
        radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = 0
        return

    ########### Conic calculation (inlined 2x2 matrix inverse) ###########
    conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 0] = c2_11 * inv_det   # cov2d[1,1] / det
    conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 1] = -c2_01 * inv_det  # -cov2d[0,1] / det
    conics_ptr[camera_idx * N * 3 + gaussian_idx * 3 + 2] = c2_00 * inv_det   # cov2d[0,0] / det

    radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = radius_x.cast[DType.int32]()
    radii_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = radius_y.cast[DType.int32]()
    means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 0] = m2d_x
    means2d_ptr[camera_idx * N * 2 + gaussian_idx * 2 + 1] = m2d_y
    depths_ptr[camera_idx * N + gaussian_idx] = mc2  # Depth is positive distance along viewing direction


@compiler.register("project_gaussians_inplace")
struct ProjectGaussiansInplace:
    """Projection kernel with 1 tiny dummy DPS output to prevent DCE.

    Real "output" tensors are passed as InputTensor and written via rebind
    to mutable pointers. This avoids the 4 large DPS buffer_store copy
    kernels (~240µs), replacing them with 1 negligible scalar copy.
    """

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        # Dummy DPS output (1 scalar) — prevents dead code elimination
        dummy: OutputTensor[dtype=DType.float32, rank=1],
        # "Outputs" passed as InputTensor (written in-place via rebind)
        means2d: InputTensor[dtype=DType.float32, rank=3],
        conics:  InputTensor[dtype=DType.float32, rank=3],
        depths:  InputTensor[dtype=DType.float32, rank=2],
        radii:   InputTensor[dtype=DType.int32,   rank=3],
        # Inputs
        means3d:       InputTensor[dtype=DType.float32, rank=2],
        scales:        InputTensor[dtype=DType.float32, rank=2],
        quats:         InputTensor[dtype=DType.float32, rank=2],
        opacities:     InputTensor[dtype=DType.float32, rank=1],
        view_matrices: InputTensor[dtype=DType.float32, rank=3],
        ks:            InputTensor[dtype=DType.float32, rank=2],
        # Context
        ctx: DeviceContextPtr
    ) raises:
        var N = means3d.dim_size(0)
        var C = view_matrices.dim_size(0)

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
            raise Error("ProjectGaussiansInplace CPU target not implemented yet.")
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
