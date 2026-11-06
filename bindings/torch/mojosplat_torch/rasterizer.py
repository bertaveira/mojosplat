"""Faster-GS style forward rasterizer (Mojo `Rasterizer`) for torch tensors.

Takes the *raw* trainable parameters, as stored by 3DGS / gsplat trainers:

    means      [N, 3]
    quats      [N, 4]  (w, x, y, z), need not be normalized
    scales     [N, 3]  log-space
    opacities  [N]     logits (pre-sigmoid)
    sh0        [N, 3] or [N, 1, 3]   DC SH coefficient
    shN        [N, K, 3] or None     higher-order SH (K = 3, 8 or 15)

Activations, projection, SH evaluation, sorting and blending all run in one
Mojo call. Differentiable: when any parameter requires grad, the training
forward saves its state (in torch-allocated buffers) and the Mojo backward
returns gradients for all six parameters (per-Gaussian backward of
Faster-GS). Optionally accumulates 3DGS densification statistics.
"""

from __future__ import annotations

import torch

from ._ext import camera_list, check, context, sync_torch
from .utils import Camera

__all__ = ["render_splats", "SH_C0", "rgb_to_sh0"]

SH_C0 = 0.28209479177387814


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    """RGB in [0, 1] -> DC SH coefficient (inverse of 0.5 + C0 * sh0)."""
    return (rgb - 0.5) / SH_C0


def _prepare(means, quats, scales, opacities, sh0, shN, sh_degree):
    n = means.shape[0]
    dev = means.device
    means = check(means, "means", torch.float32)
    quats = check(quats, "quats", torch.float32)
    scales = check(scales, "scales", torch.float32)
    opacities = check(opacities.reshape(n), "opacities", torch.float32)
    sh0 = check(sh0.reshape(n, 3), "sh0", torch.float32)
    if shN is None or shN.numel() == 0:
        rest = 0
        shN_t = torch.zeros(3, device=dev, dtype=torch.float32)
    else:
        rest = shN.shape[1]
        shN_t = check(shN.reshape(n, rest, 3), "shN", torch.float32)
    max_degree = {0: 0, 3: 1, 8: 2, 15: 3}.get(rest)
    if max_degree is None:
        raise ValueError(f"shN must hold 3, 8 or 15 bases per Gaussian, got {rest}")
    degree = max_degree if sh_degree is None else sh_degree
    if not 0 <= degree <= max_degree:
        raise ValueError(f"sh_degree {degree} needs more SH bases than shN holds ({rest})")
    if means.shape != (n, 3) or quats.shape != (n, 4) or scales.shape != (n, 3):
        raise ValueError("expected means [N,3], quats [N,4], scales [N,3]")
    return (means, quats, scales, opacities, sh0, shN_t), rest, degree


def _ptrs(params):
    means, quats, scales, opacities, sh0, shN_t = params
    return (
        means.data_ptr(), scales.data_ptr(), quats.data_ptr(), opacities.data_ptr(),
        sh0.data_ptr(), shN_t.data_ptr(),
    )


class _RenderSplats(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, quats, scales, opacities, sh0, shN_t, cam_bg, sizes, densification, stats_out):
        params = (means, quats, scales, opacities, sh0, shN_t)
        H, W = cam_bg[0][21], cam_bg[0][20]
        image = torch.empty(H, W, 3, device=means.device, dtype=torch.float32)
        buffers = []

        def alloc(nbytes):
            t = torch.empty(max(int(nbytes), 1), dtype=torch.uint8, device=means.device)
            buffers.append(t)
            return t.data_ptr()

        sync_torch()
        state = context().forward_train(_ptrs(params) + (image.data_ptr(),), cam_bg, sizes, alloc)
        ctx.state = state
        ctx.buffers = buffers
        ctx.cam_bg = cam_bg
        ctx.sizes = sizes
        ctx.densification = densification
        stats_out.extend((int(state[1]), int(state[2])))
        ctx.save_for_backward(*params, image)
        return image

    @staticmethod
    def backward(ctx, grad_image):
        *params, image = ctx.saved_tensors
        grad_image = grad_image.contiguous()
        grads = [torch.empty_like(p) for p in params]  # fully written by the kernel
        dens = ctx.densification
        sync_torch()
        context().backward(
            _ptrs(params) + (image.data_ptr(), grad_image.data_ptr()),
            tuple(g.data_ptr() for g in (grads[0], grads[2], grads[1], grads[3], grads[4], grads[5]))
            + (0 if dens is None else dens.data_ptr(),),
            ctx.cam_bg, ctx.state, ctx.sizes,
        )
        ctx.buffers = None  # release the saved forward state
        return (*grads, None, None, None, None)


def render_splats(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor | None,
    camera: Camera,
    sh_degree: int | None = None,
    background: torch.Tensor | None = None,
    return_stats: bool = False,
    densification: torch.Tensor | None = None,
):
    """Render to an [H, W, 3] float32 image (differentiable).

    `sh_degree` defaults to the highest degree `shN` holds. With
    `return_stats=True` also returns (num_visible, num_tile_instances).
    `densification` (optional float32 [2, N], CUDA) accumulates, during the
    backward pass, a visibility count and the norm of the screen-space mean
    gradient (in NDC) per Gaussian, as used by 3DGS densification.
    """
    n = means.shape[0]
    params, rest, degree = _prepare(means, quats, scales, opacities, sh0, shN, sh_degree)
    bg = [0.0, 0.0, 0.0] if background is None else [float(v) for v in background.reshape(-1).tolist()]
    cam_bg = (camera_list(camera), bg)
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (means, quats, scales, opacities, sh0) + ((shN,) if shN is not None else ())
    )
    if needs_grad:
        if densification is not None:
            densification = check(densification, "densification", torch.float32)
            if densification.shape != (2, n):
                raise ValueError("densification must be [2, N]")
        stats = []
        image = _RenderSplats.apply(*params, cam_bg, (n, rest, degree), densification, stats)
        stats = tuple(stats)
    else:
        image = torch.empty(camera.H, camera.W, 3, device=means.device, dtype=torch.float32)
        sync_torch()
        stats = context().render_splats(_ptrs(params) + (image.data_ptr(),), cam_bg[0], bg, (n, rest, degree))
        stats = (int(stats[0]), int(stats[1]))
    if return_stats:
        return image, stats
    return image
