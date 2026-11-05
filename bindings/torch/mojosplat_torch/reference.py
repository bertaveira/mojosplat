"""gsplat reference renderer with the same raw-parameter interface.

Used by tests, benchmarks and the viewer to compare against
`render_splats`. Requires the optional `gsplat` package.
"""

from __future__ import annotations

import torch

from .utils import Camera

__all__ = ["render_gsplat"]


def render_gsplat(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor | None,
    camera: Camera,
    sh_degree: int | None = None,
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    """gsplat.rasterization on raw parameters; returns [H, W, 3]."""
    from gsplat import rasterization

    n = means.shape[0]
    colors = sh0.reshape(n, 1, 3)
    if shN is not None and shN.numel() > 0:
        colors = torch.cat([colors, shN.reshape(n, -1, 3)], 1)
    max_degree = {1: 0, 4: 1, 9: 2, 16: 3}[colors.shape[1]]
    img, alpha, _ = rasterization(
        means, quats, torch.exp(scales), torch.sigmoid(opacities.reshape(n)), colors,
        camera.view_matrix[None], camera.Ks[None], camera.W, camera.H,
        near_plane=camera.near, far_plane=camera.far,
        sh_degree=max_degree if sh_degree is None else sh_degree,
    )
    img = img[0]
    if background is not None:
        img = img + (1.0 - alpha[0]) * background.reshape(1, 1, 3).to(img)
    return img
