"""Faster-GS style forward rasterizer (Mojo `Rasterizer`) for torch tensors.

Takes the *raw* trainable parameters, as stored by 3DGS / gsplat trainers:

    means      [N, 3]
    quats      [N, 4]  (w, x, y, z), need not be normalized
    scales     [N, 3]  log-space
    opacities  [N]     logits (pre-sigmoid)
    sh0        [N, 3] or [N, 1, 3]   DC SH coefficient
    shN        [N, K, 3] or None     higher-order SH (K = 3, 8 or 15)

Activations, projection, SH evaluation, sorting and blending all run in one
Mojo call. Forward only (no autograd yet).
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
):
    """Render to an [H, W, 3] float32 image.

    `sh_degree` defaults to the highest degree `shN` holds. With
    `return_stats=True` also returns (num_visible, num_tile_instances).
    """
    n = means.shape[0]
    dev = means.device
    means = check(means, "means", torch.float32)
    quats = check(quats, "quats", torch.float32)
    scales = check(scales, "scales", torch.float32)
    opacities = check(opacities.reshape(n), "opacities", torch.float32)
    sh0 = check(sh0.reshape(n, 3), "sh0", torch.float32)
    if shN is None or shN.numel() == 0:
        rest = 0
        shN_t = torch.empty(3, device=dev, dtype=torch.float32)
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

    bg = [0.0, 0.0, 0.0] if background is None else [float(v) for v in background.reshape(-1).tolist()]
    image = torch.empty(camera.H, camera.W, 3, device=dev, dtype=torch.float32)
    sync_torch()
    stats = context().render_splats(
        (
            means.data_ptr(), scales.data_ptr(), quats.data_ptr(), opacities.data_ptr(),
            sh0.data_ptr(), shN_t.data_ptr(), image.data_ptr(),
        ),
        camera_list(camera),
        bg,
        (n, rest, degree),
    )
    if return_stats:
        return image, (int(stats[0]), int(stats[1]))
    return image
