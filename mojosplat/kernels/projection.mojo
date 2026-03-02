import compiler
from gpu import thread_idx, block_idx
from math import sqrt, rsqrt, recip, ceil, log, fma
from math import ceildiv
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer
from gpu.intrinsics import ldg
from pathlib import Path

comptime radius_clip: Float32 = 0.0
comptime block_size: Int = 256


fn project_ewa_kernel(
    means3d_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],       # [N * 3]
    scales_ptr:  UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],       # [N * 3]
    quats_ptr:   UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],       # [N * 4]
    opacities_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],     # [N]
    view_matrices_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin], # [C * 16]
    ks_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],            # [C * 11]
    radii_ptr:   UnsafePointer[Scalar[DType.int32], MutAnyOrigin],           # [C * N * 2]
    means2d_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],         # [C * N * 2]
    depths_ptr:  UnsafePointer[Scalar[DType.float32], MutAnyOrigin],         # [C * N]
    conics_ptr:  UnsafePointer[Scalar[DType.float32], MutAnyOrigin],         # [C * N * 3]
    N: UInt32,
    C: UInt32,
):
    var idx: UInt32 = UInt32(block_idx.x) * UInt32(block_size) + UInt32(thread_idx.x)
    if idx >= N * C:
        return

    var gaussian_idx: UInt32 = idx % N
    var camera_idx:   UInt32 = idx // N

    var g_off:   Int = Int(gaussian_idx)
    var cam_off: Int = Int(camera_idx)
    var io_off:  Int = Int(idx)

    var means3d_g   = means3d_ptr   + g_off   * 3
    var quats_g     = quats_ptr     + g_off   * 4
    var scales_g    = scales_ptr    + g_off   * 3
    var opacities_g = opacities_ptr + g_off
    var vm_g        = view_matrices_ptr + cam_off * 16
    var ks_g        = ks_ptr            + cam_off * 11
    var radii_out   = radii_ptr   + io_off * 2
    var means2d_out = means2d_ptr + io_off * 2
    var depths_out  = depths_ptr  + io_off
    var conics_out  = conics_ptr  + io_off * 3

    # ── World-to-camera via FMA chains (4-term dot product = 3 FMA each) ─────
    # ldg() uses ld.global.nc (NC/read-only cache path), same as CUDA __ldg().
    # This matches how gsplat uses const float* inputs.
    var g0: Float32 = ldg(means3d_g)
    var g1: Float32 = ldg(means3d_g + 1)
    var g2: Float32 = ldg(means3d_g + 2)
    # Load view matrix row 0 and compute W2C + keep for Mp row 0
    var r0: Float32 = ldg(vm_g);      var r1: Float32 = ldg(vm_g + 1)
    var r2: Float32 = ldg(vm_g + 2);  var tr0: Float32 = ldg(vm_g + 3)
    var mc0: Float32 = fma(r0, g0, fma(r1, g1, fma(r2, g2, tr0)))
    # Load view matrix row 1
    var r3: Float32 = ldg(vm_g + 4);  var r4: Float32 = ldg(vm_g + 5)
    var r5: Float32 = ldg(vm_g + 6);  var tr1: Float32 = ldg(vm_g + 7)
    var mc1: Float32 = fma(r3, g0, fma(r4, g1, fma(r5, g2, tr1)))
    # Load view matrix row 2
    var r6: Float32 = ldg(vm_g + 8);  var r7: Float32 = ldg(vm_g + 9)
    var r8: Float32 = ldg(vm_g + 10); var tr2: Float32 = ldg(vm_g + 11)
    var mc2: Float32 = fma(r6, g0, fma(r7, g1, fma(r8, g2, tr2)))
    # g0, g1, g2, tr0, tr1, tr2 dead; r0..r8 KEPT for Mp below

    comptime near_plane: Float32 = 0.1
    if mc2 <= near_plane:
        radii_out[0] = Int32(0)
        radii_out[1] = Int32(0)
        return

    var opacity: Float32 = ldg(opacities_g)
    comptime ALPHA_THRESHOLD: Float32 = 1.0 / 255.0
    if opacity < ALPHA_THRESHOLD:
        radii_out[0] = Int32(0)
        radii_out[1] = Int32(0)
        return

    # ── Quaternion normalization with FMA for norm_sq ─────────────────────────
    # Quats have stride-4 (16 bytes) → 16-byte aligned → use 128-bit v4 NC load
    var q_vec = ldg[width=4](quats_g)
    var w: Float32 = q_vec[0]; var x: Float32 = q_vec[1]
    var y: Float32 = q_vec[2]; var z: Float32 = q_vec[3]
    var inv_norm: Float32 = rsqrt(fma(w, w, fma(x, x, fma(y, y, z*z))))
    w *= inv_norm; x *= inv_norm; y *= inv_norm; z *= inv_norm

    var x2: Float32 = x*x;  var y2: Float32 = y*y;  var z2: Float32 = z*z
    var xy: Float32 = x*y;  var xz: Float32 = x*z;  var yz: Float32 = y*z
    var wx: Float32 = w*x;  var wy: Float32 = w*y;  var wz: Float32 = w*z

    var s0: Float32 = ldg(scales_g)
    var s1: Float32 = ldg(scales_g + 1)
    var s2: Float32 = ldg(scales_g + 2)

    # ── Q rotation matrix columns (precomputed once, reused for all 3 Mp rows) ──
    # fma() requires all args to be Float32 — bare literals cause type errors,
    # so we use plain arithmetic here. The big FMA wins are in Mp/Cc/t2/c2.
    var q0: Float32 = 1.0 - 2.0*(y2+z2)   # 1-2(y2+z2)
    var q1: Float32 = 2.0*xy + 2.0*wz      # 2(xy+wz)
    var q2: Float32 = 2.0*xz - 2.0*wy      # 2(xz-wy)
    var q3: Float32 = 2.0*xy - 2.0*wz      # 2(xy-wz)
    var q4: Float32 = 1.0 - 2.0*(x2+z2)   # 1-2(x2+z2)
    var q5: Float32 = 2.0*yz + 2.0*wx      # 2(yz+wx)
    var q6: Float32 = 2.0*xz + 2.0*wy      # 2(xz+wy)
    var q7: Float32 = 2.0*yz - 2.0*wx      # 2(yz-wx)
    var q8: Float32 = 1.0 - 2.0*(x2+y2)   # 1-2(x2+y2)
    # x2,y2,z2,xy,xz,yz,wx,wy,wz dead

    # ── Mp = Rv @ Q @ diag(S) using r0..r8 already loaded above ─────────────
    # r0..r8 are alive from the W2C computation — NO redundant loads needed.
    var Mp00: Float32 = s0 * fma(r0, q0, fma(r1, q1, r2*q2))
    var Mp01: Float32 = s1 * fma(r0, q3, fma(r1, q4, r2*q5))
    var Mp02: Float32 = s2 * fma(r0, q6, fma(r1, q7, r2*q8))

    var Mp10: Float32 = s0 * fma(r3, q0, fma(r4, q1, r5*q2))
    var Mp11: Float32 = s1 * fma(r3, q3, fma(r4, q4, r5*q5))
    var Mp12: Float32 = s2 * fma(r3, q6, fma(r4, q7, r5*q8))

    var Mp20: Float32 = s0 * fma(r6, q0, fma(r7, q1, r8*q2))
    var Mp21: Float32 = s1 * fma(r6, q3, fma(r7, q4, r8*q5))
    var Mp22: Float32 = s2 * fma(r6, q6, fma(r7, q7, r8*q8))
    # q0..q8, s0,s1,s2, r0..r8 dead

    # ── Cc = Mp @ Mp^T with FMA chains (3-term → 1 MUL + 2 FMA each) ─────────
    var Cc00: Float32 = fma(Mp00, Mp00, fma(Mp01, Mp01, Mp02*Mp02))
    var Cc01: Float32 = fma(Mp00, Mp10, fma(Mp01, Mp11, Mp02*Mp12))
    var Cc02: Float32 = fma(Mp00, Mp20, fma(Mp01, Mp21, Mp02*Mp22))
    var Cc11: Float32 = fma(Mp10, Mp10, fma(Mp11, Mp11, Mp12*Mp12))
    var Cc12: Float32 = fma(Mp10, Mp20, fma(Mp11, Mp21, Mp12*Mp22))
    var Cc22: Float32 = fma(Mp20, Mp20, fma(Mp21, Mp21, Mp22*Mp22))
    # Mp00..Mp22 dead

    # ── Pinhole projection with fast recip ────────────────────────────────────
    var fx: Float32 = ldg(ks_g);      var fy: Float32 = ldg(ks_g + 1)
    var cx: Float32 = ldg(ks_g + 2);  var cy: Float32 = ldg(ks_g + 3)
    var W: Float32  = ldg(ks_g + 4);  var H: Float32  = ldg(ks_g + 5)
    var rfx: Float32 = recip(fx)
    var rfy: Float32 = recip(fy)

    # Frustum limits: lim_x_pos = (W-cx)/fx + 0.3*(0.5*W/fx)
    #                            = (W-cx+0.15*W)/fx = (1.15*W - cx)/fx
    # This avoids computing tan_fov_x/y as intermediates (saves 2 regs & 2 MULs).
    var lim_x_pos: Float32 = fma(Float32(1.15), W, -cx) * rfx
    var lim_x_neg: Float32 = fma(Float32(0.15), W,  cx) * rfx
    var lim_y_pos: Float32 = fma(Float32(1.15), H, -cy) * rfy
    var lim_y_neg: Float32 = fma(Float32(0.15), H,  cy) * rfy

    var rz:  Float32 = recip(mc2)
    var rz2: Float32 = rz * rz
    var tx: Float32 = mc2 * min(lim_x_pos, max(-lim_x_neg, mc0 * rz))
    var ty: Float32 = mc2 * min(lim_y_pos, max(-lim_y_neg, mc1 * rz))

    # Jacobian J (2×3, J01=J10=0)
    var J00: Float32 = fx * rz;        var J02: Float32 = -fx * tx * rz2
    var J11: Float32 = fy * rz;        var J12: Float32 = -fy * ty * rz2

    # cov2d = J @ Cc @ J^T  (exploit sparsity, FMA for each 2-term sum)
    var t2_00: Float32 = fma(J00, Cc00, J02*Cc02)
    var t2_01: Float32 = fma(J00, Cc01, J02*Cc12)
    var t2_02: Float32 = fma(J00, Cc02, J02*Cc22)
    var t2_10: Float32 = fma(J11, Cc01, J12*Cc02)
    var t2_11: Float32 = fma(J11, Cc11, J12*Cc12)
    var t2_12: Float32 = fma(J11, Cc12, J12*Cc22)
    var c2_00: Float32 = fma(t2_00, J00, t2_02*J02)
    var c2_01: Float32 = fma(t2_01, J11, t2_02*J12)
    var c2_10: Float32 = fma(t2_10, J00, t2_12*J02)
    var c2_11: Float32 = fma(t2_11, J11, t2_12*J12)

    # Projected mean: J00*mc0 + cx = fx*mc0/mc2 + cx  (1 FMA, no extra MUL)
    var m2d_x: Float32 = fma(J00, mc0, cx)
    var m2d_y: Float32 = fma(J11, mc1, cy)

    comptime eps2d: Float32 = 0.3
    c2_00 += eps2d
    c2_11 += eps2d

    # ── Opacity-aware radius ──────────────────────────────────────────────────
    # opacity >= ALPHA_THRESHOLD is guaranteed (we returned above otherwise).
    # Replace division with multiplication to eliminate the last div.rn.f32.
    var log_ratio: Float32 = log(opacity * 255.0)
    var extend: Float32 = min(Float32(3.33), sqrt(2.0 * log_ratio))

    var radius_x: Float32 = ceil(extend * sqrt(c2_00))
    var radius_y: Float32 = ceil(extend * sqrt(c2_11))

    if radius_x <= radius_clip and radius_y <= radius_clip:
        radii_out[0] = Int32(0)
        radii_out[1] = Int32(0)
        return

    if m2d_x + radius_x <= 0.0 or m2d_x - radius_x >= W or \
       m2d_y + radius_y <= 0.0 or m2d_y - radius_y >= H:
        radii_out[0] = Int32(0)
        radii_out[1] = Int32(0)
        return

    # ── Conic (2×2 inverse) ───────────────────────────────────────────────────
    var det_val: Float32 = fma(c2_00, c2_11, -(c2_01 * c2_10))
    var inv_det: Float32 = recip(det_val)
    conics_out[0]  =  c2_11 * inv_det
    conics_out[1]  = -c2_01 * inv_det
    conics_out[2]  =  c2_00 * inv_det

    radii_out[0]   = radius_x.cast[DType.int32]()
    radii_out[1]   = radius_y.cast[DType.int32]()
    means2d_out[0] = m2d_x
    means2d_out[1] = m2d_y
    depths_out[0]  = mc2


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

        var means3d_ptr       = rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](means3d.to_layout_tensor().ptr)
        var scales_ptr        = rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](scales.to_layout_tensor().ptr)
        var quats_ptr         = rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](quats.to_layout_tensor().ptr)
        var opacities_ptr     = rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](opacities.to_layout_tensor().ptr)
        var view_matrices_ptr = rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](view_matrices.to_layout_tensor().ptr)
        var ks_ptr            = rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](ks.to_layout_tensor().ptr)
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

            gpu_ctx.enqueue_function_unchecked[project_ewa_kernel, dump_asm = Path("/tmp/proj_kernel.ptx")](
                means3d_ptr, scales_ptr, quats_ptr, opacities_ptr,
                view_matrices_ptr, ks_ptr,
                radii_ptr, means2d_ptr, depths_ptr, conics_ptr,
                UInt32(N), UInt32(C),
                grid_dim=grid, block_dim=block,
            )
