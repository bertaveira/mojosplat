from pathlib import Path
import torch
from typing_extensions import Literal

from max.torch import graph_op
from max.graph import ops as graph_ops, TensorType as GraphTensorType, DeviceRef
from max.dtype import DType as MaxDType
from .utils import Camera

mojo_kernels = Path(__file__).parent / "kernels"
_gpu = DeviceRef.GPU()

_rasterize_graph_ops: dict = {}


def _get_rasterize_graph_op(tile_size: int, cdim: int):
    key = (tile_size, cdim)
    if key in _rasterize_graph_ops:
        return _rasterize_graph_ops[key]

    _N = "N"
    _M = "M"
    _TGH = "TGH"
    _TGW = "TGW"
    _H = "ImgH"
    _W = "ImgW"

    @graph_op(
        name=f"rasterize_to_pixels_3dgs_fwd_ts{tile_size}_c{cdim}",
        kernel_library=mojo_kernels,
        input_types=[
            GraphTensorType(MaxDType.float32, (1, _N, 2),       device=_gpu),  # means2d
            GraphTensorType(MaxDType.float32, (1, _N, 3),       device=_gpu),  # conics
            GraphTensorType(MaxDType.float32, (1, _N, cdim),    device=_gpu),  # colors
            GraphTensorType(MaxDType.float32, (1, _N),          device=_gpu),  # opacities
            GraphTensorType(MaxDType.float32, (1, cdim),        device=_gpu),  # backgrounds
            GraphTensorType(MaxDType.int32,   (1, _TGH, _TGW, 2), device=_gpu),  # tile_ranges
            GraphTensorType(MaxDType.int32,   (1, _M),          device=_gpu),  # flatten_ids
        ],
        output_types=[
            GraphTensorType(MaxDType.float32, (1, _H, _W, cdim), device=_gpu),  # render_colors
        ],
    )
    def _rasterize_graph(means2d, conics, colors, opacities, backgrounds, tile_ranges, flatten_ids):
        return graph_ops.custom(
            "rasterize_to_pixels_3dgs_fwd",
            _gpu,
            [means2d, conics, colors, opacities, backgrounds, tile_ranges, flatten_ids],
            out_types=[GraphTensorType(MaxDType.float32, (1, _H, _W, cdim), device=_gpu)],
            parameters={"tile_size": tile_size, "CDIM": cdim},
        )

    _rasterize_graph_ops[key] = _rasterize_graph
    return _rasterize_graph


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
    """Mojo implementation of Gaussian rasterization."""
    # Ensure proper tensor shapes and types for Mojo kernel
    if means2d.dim() == 2:
        means2d = means2d.unsqueeze(0)  # (N, 2) -> (1, N, 2)
    if conics.dim() == 2:
        conics = conics.unsqueeze(0)    # (N, 3) -> (1, N, 3)
    if colors.dim() == 2:
        colors = colors.unsqueeze(0)    # (N, C) -> (1, N, C)
    if opacities.dim() == 1:
        opacities = opacities.unsqueeze(0)  # (N,) -> (1, N)
    elif opacities.dim() == 2:
        opacities = opacities.squeeze(-1).unsqueeze(0)  # (N, 1) -> (N,) -> (1, N)
    if background_color.dim() == 1:
        background_color = background_color.unsqueeze(0)  # (C,) -> (1, C)
    if tile_ranges.dim() == 3:
        tile_ranges = tile_ranges.unsqueeze(0)  # (TGH, TGW, 2) -> (1, TGH, TGW, 2)
    if sorted_gaussian_indices.dim() == 1:
        sorted_gaussian_indices = sorted_gaussian_indices.unsqueeze(0)  # (M,) -> (1, M)

    # Ensure contiguous tensors and correct dtypes
    means2d = means2d.contiguous()
    conics = conics.contiguous()
    colors = colors.contiguous()
    background_color = background_color.contiguous()
    tile_ranges = tile_ranges.to(torch.int32).contiguous()
    sorted_gaussian_indices = sorted_gaussian_indices.to(torch.int32).contiguous()

    num_channels = colors.shape[-1]
    result = torch.zeros(1, camera.H, camera.W, num_channels, device=means2d.device, dtype=means2d.dtype)

    rasterize_kernel = _get_rasterize_graph_op(tile_size, num_channels)
    rasterize_kernel(
        result, means2d, conics, colors, opacities, background_color,
        tile_ranges, sorted_gaussian_indices
    )

    return result.squeeze(0)
