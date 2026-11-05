from pathlib import Path
import torch
from typing_extensions import Literal

from .utils import Camera
from . import _ext


def rasterize_gaussians(
    means2d: torch.Tensor,  # (N, 2) 2D projected means
    conics: torch.Tensor,   # (N, 3) 2D covariance matrices (flattened upper triangle)
    colors: torch.Tensor,   # (N, C) Color features
    opacities: torch.Tensor, # (N,) Opacity values
    background_color: torch.Tensor, # (C,) Background color
    tile_ranges: torch.Tensor, # Tile ranges from binning (tile_height, tile_width, 2)
    sorted_gaussian_indices: torch.Tensor, # Sorted gaussian indices from binning (n_tiles,)
    camera: Camera,
    tile_size: int = 16,
    backend: Literal["torch", "gsplat", "mojo"] = "mojo",
) -> torch.Tensor:
    """Rasterizes 2D Gaussians to pixels.

    Args:
        means2d: 2D projected means in pixel coordinates
        conics: 2D covariance matrices (flattened upper triangle)
        colors: Color features for each Gaussian
        opacities: Opacity values for each Gaussian
        background_color: Background color tensor
        tile_ranges: Tile ranges from binning step
        sorted_gaussian_indices: Sorted Gaussian indices from binning step
        camera: Camera object with image dimensions
        backend: Backend to use ("torch", "gsplat", "mojo")

    Returns:
        rendered_image: (H, W, C) Rendered image
    """
    if backend == "torch":
        return rasterize_gaussians_torch(
            means2d, conics, colors, opacities, background_color,
            tile_ranges, sorted_gaussian_indices, camera, tile_size
        )
    elif backend == "gsplat":
        return rasterize_gaussians_gsplat(
            means2d, conics, colors, opacities, background_color,
            tile_ranges, sorted_gaussian_indices, camera, tile_size
        )
    elif backend == "mojo":
        return rasterize_gaussians_mojo(
            means2d, conics, colors, opacities, background_color,
            tile_ranges, sorted_gaussian_indices, camera, tile_size
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")


def rasterize_gaussians_torch(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    background_color: torch.Tensor,
    tile_ranges: torch.Tensor,
    sorted_gaussian_indices: torch.Tensor,
    camera: Camera,
    tile_size: int = 16,
) -> torch.Tensor:
    """PyTorch implementation of Gaussian rasterization."""

    # Throw warning that this is not implemented yet
    print("Warning: PyTorch backend for rasterization not implemented yet")
    return rasterize_gaussians_gsplat(
        means2d, conics, colors, opacities, background_color,
        tile_ranges, sorted_gaussian_indices, camera, tile_size
    )


def rasterize_gaussians_gsplat(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    background_color: torch.Tensor,
    tile_ranges: torch.Tensor,
    sorted_gaussian_indices: torch.Tensor,
    camera: Camera,
    tile_size: int = 16,
) -> torch.Tensor:
    """GSplat implementation of Gaussian rasterization."""
    from gsplat import rasterize_to_pixels

    means2d = means2d.unsqueeze(0).contiguous()
    conics = conics.unsqueeze(0).contiguous()
    colors = colors.unsqueeze(0).contiguous()
    # GSplat expects opacities to be (batch, N) not (batch, N, 1)
    if opacities.dim() == 1:
        opacities = opacities.unsqueeze(0).contiguous()  # (N,) -> (1, N)
    else:
        opacities = opacities.squeeze(-1).unsqueeze(0).contiguous()  # (N, 1) -> (N,) -> (1, N)
    background_color = background_color.unsqueeze(0).contiguous()
    tile_ranges = tile_ranges.unsqueeze(0).contiguous()
    sorted_gaussian_indices = sorted_gaussian_indices.contiguous()  # gsplat expects 1D [n_isects]

    tile_ranges = tile_ranges[:, :, :, 0]

    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        camera.W,
        camera.H,
        tile_size,
        tile_ranges,
        sorted_gaussian_indices,
        backgrounds=background_color,
        packed=False,
        absgrad=False,
    )

    return render_colors.squeeze(0)


def rasterize_gaussians_mojo(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    background_color: torch.Tensor,
    tile_ranges: torch.Tensor,
    sorted_gaussian_indices: torch.Tensor,
    camera: Camera,
    tile_size: int = 16,
) -> torch.Tensor:
    """Mojo rasterization; returns an (H, W, C) image."""
    if tile_size != _ext.TILE_SIZE:
        raise ValueError(f"mojo backend is compiled for tile_size={_ext.TILE_SIZE}")
    if colors.shape[-1] != _ext.CHANNELS:
        raise ValueError(f"mojo backend is compiled for {_ext.CHANNELS} color channels")
    device = means2d.device
    means2d = _ext.check(means2d.reshape(-1, 2), "means2d", torch.float32)
    conics = _ext.check(conics.reshape(-1, 3), "conics", torch.float32)
    colors = _ext.check(colors.reshape(-1, _ext.CHANNELS), "colors", torch.float32)
    opacities = _ext.check(opacities.reshape(-1), "opacities", torch.float32)
    ids = _ext.check(sorted_gaussian_indices.reshape(-1).to(torch.int32), "ids", torch.int32)
    # tile_ranges [TH, TW, 2] -> start offsets [n_tiles + 1]
    tr = tile_ranges.reshape(-1, 2).to(torch.int32)
    tile_offsets = torch.cat([tr[:, 0], tr[-1:, 1]]).contiguous()
    if ids.numel() == 0:
        ids = torch.zeros(1, dtype=torch.int32, device=device)

    image = torch.empty(camera.H, camera.W, _ext.CHANNELS, dtype=torch.float32, device=device)
    _ext.sync_torch()
    _ext.context().rasterize(
        (means2d.data_ptr(), conics.data_ptr(), colors.data_ptr(), opacities.data_ptr(),
         tile_offsets.data_ptr(), ids.data_ptr(), image.data_ptr()),
        background_color.reshape(-1).tolist(),
        camera.W, camera.H,
    )
    return image
