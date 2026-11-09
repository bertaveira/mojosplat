"""Preprocess backward: 2D gradients -> raw-parameter gradients.

Ported from Faster-GS (Apache-2.0, `kernels_backward.cuh`). One thread per
Gaussian, in Gaussian order so every gradient write is coalesced: culled
Gaussians get zero gradients, visible ones go from their slot's 2D gradients
(mean2d, conic, color) back through the EWA projection, the activations and
spherical harmonics to the raw parameters. All gradient buffers are fully
written (no zero-init needed).
"""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceStream
from std.math import sqrt

from ..core.camera import CAM_H, CAM_POS_X, CAM_POS_Y, CAM_POS_Z, CAM_W
from ..core.types import BLOCK_SIZE, F32Ptr, I32Ptr, KernelVector, num_blocks
from ..core.linalg import V2, V3, V4
from ..core.splats import AppearanceSettings, ExtraColumns
from .blocks.contracts import Pipeline
from .blocks.shape import QuatScale
from .pipelines import Standard3DGS


def _preprocess_backward_kernel[
    Pipe: Pipeline
](
    means: F32Ptr,
    log_scales: F32Ptr,
    rotations: F32Ptr,
    col0: F32Ptr,  # columns after the core ones (see ExtraColumns)
    col1: F32Ptr,
    col2: F32Ptr,
    col3: F32Ptr,
    cam: KernelVector[DType.float32, 32],
    slot_of: I32Ptr,  # [N] slot of each Gaussian, -1 if culled
    grad_resp: F32Ptr,  # slot gradients of the Response (see its `grad_index`)
    n_slots: Int32,
    grad_color: F32Ptr,  # [slot, 3]
    out_means: F32Ptr,  # [N, 3]
    out_log_scales: F32Ptr,  # [N, 3]
    out_rotations: F32Ptr,  # [N, 4]
    out_opacity: F32Ptr,  # [N]
    gcol0: F32Ptr,  # out: gradients of those columns
    gcol1: F32Ptr,
    gcol2: F32Ptr,
    gcol3: F32Ptr,
    densification: F32Ptr,  # [2, N] (count, sum |dL/dmean2d_ndc|) if has_densification
    has_densification: Int32,  # Bool is not DevicePassable
    write_all: Int32,  # 0: skip what a fused optimizer recomputes (see the appearance)
    n: Int32,
    settings: KernelVector[DType.int32, 4],
):
    var cols = ExtraColumns(col0, col1, col2, col3)
    var out_cols = ExtraColumns(gcol0, gcol1, gcol2, gcol3)
    var g = block_idx.x * BLOCK_SIZE + thread_idx.x
    if g >= Int(n):
        return
    var s = Int(slot_of[unsafe_offset=g])
    if s < 0:
        comptime for k in range(3):
            out_means[unsafe_offset=g * 3 + k] = 0.0
            out_log_scales[unsafe_offset=g * 3 + k] = 0.0
        comptime for k in range(4):
            out_rotations[unsafe_offset=g * 4 + k] = 0.0
        out_opacity[unsafe_offset=g] = 0.0
        Pipe.Appear.zero_gradients(out_cols, g, settings.value, write_all != 0)
        Pipe.Resp.zero_gradients(out_cols.skip(Pipe.Appear.NUM_COLUMNS), g)
        return

    comptime P = Pipe.Place
    comptime R = Pipe.Resp
    comptime dt = P.dtype
    var cam_v = cam.value.cast[dt]()
    var mean = V3[dt].load(means, g)

    # SH -> color backward (also yields a mean gradient through the view dir).
    var gcol = SIMD[DType.float32, 4](0.0)
    comptime for c in range(3):
        gcol[c] = grad_color[unsafe_offset=s * 3 + c]
    var dir = SIMD[DType.float32, 4](
        mean.x.cast[DType.float32]() - cam[CAM_POS_X],
        mean.y.cast[DType.float32]() - cam[CAM_POS_Y],
        mean.z.cast[DType.float32]() - cam[CAM_POS_Z],
        0.0,
    )
    var dmean_color = Pipe.Appear.backward(
        cols,
        out_cols,
        g,
        dir,
        gcol,
        settings.value,
        write_all != 0,
    )
    var slot_grad = SIMD[DType.float32, 8](0.0)
    comptime for k in range(R.GRAD):
        slot_grad[k] = grad_resp[unsafe_offset=R.grad_index(k, Int(n_slots), s)]
    out_opacity[unsafe_offset=g] = R.opacity_gradient(slot_grad)
    R.write_gradients(cols.skip(Pipe.Appear.NUM_COLUMNS), out_cols.skip(Pipe.Appear.NUM_COLUMNS), g, slot_grad)

    # Forward, top to bottom.
    var shape = QuatScale[dt](V3[dt].load(log_scales, g), V4[dt].load(rotations, g))
    var place = P(mean, shape, cam_v)
    var resp = R(place.mean2d().f32(), place.cov2d().f32())

    var screen = resp.backward(slot_grad)
    if has_densification != 0:
        var nx = 0.5 * screen.mean2d.x * cam[CAM_W]
        var ny = 0.5 * screen.mean2d.y * cam[CAM_H]
        densification[unsafe_offset=g] += 1.0
        densification[unsafe_offset=Int(n) + g] += sqrt(nx * nx + ny * ny)

    # Backward, bottom to top.
    var grad = place.backward(
        shape,
        V2[dt](screen.mean2d.x.cast[dt](), screen.mean2d.y.cast[dt]()),
        screen.cov2d.cast[dt](),
    )

    out_means[unsafe_offset=g * 3 + 0] = grad.mean.x.cast[DType.float32]() + dmean_color[0]
    out_means[unsafe_offset=g * 3 + 1] = grad.mean.y.cast[DType.float32]() + dmean_color[1]
    out_means[unsafe_offset=g * 3 + 2] = grad.mean.z.cast[DType.float32]() + dmean_color[2]
    grad.shape.log_scales.store(out_log_scales, g)
    grad.shape.quat.store(out_rotations, g)


def preprocess_backward[
    Pipe: Pipeline = Standard3DGS
](
    ctx: DeviceContext,
    stream: DeviceStream,
    means: F32Ptr,
    log_scales: F32Ptr,
    rotations: F32Ptr,
    extras: ExtraColumns,
    cam: SIMD[DType.float32, 32],
    slot_of: I32Ptr,
    grad_resp: F32Ptr,
    n_slots: Int,
    grad_color: F32Ptr,
    out_means: F32Ptr,
    out_log_scales: F32Ptr,
    out_rotations: F32Ptr,
    out_opacity: F32Ptr,
    out_extras: ExtraColumns,
    densification: F32Ptr,
    has_densification: Bool,
    n: Int,
    settings: AppearanceSettings,
    write_all: Bool = True,
) raises:
    if n == 0:
        return
    stream.enqueue_function(
        ctx.compile_function[_preprocess_backward_kernel[Pipe]](),
        means,
        log_scales,
        rotations,
        extras.c0,
        extras.c1,
        extras.c2,
        extras.c3,
        KernelVector[DType.float32, 32](cam),
        slot_of,
        grad_resp,
        Int32(n_slots),
        grad_color,
        out_means,
        out_log_scales,
        out_rotations,
        out_opacity,
        out_extras.c0,
        out_extras.c1,
        out_extras.c2,
        out_extras.c3,
        densification,
        Int32(1 if has_densification else 0),
        Int32(1 if write_all else 0),
        Int32(n),
        KernelVector[DType.int32, 4](settings),
        grid_dim=num_blocks(n),
        block_dim=BLOCK_SIZE,
    )
