
import os
import sys
import sysconfig
import max.torch as mtorch
from pathlib import Path
import torch
import numpy as np
import math

from max import engine
from max.driver import Accelerator

from .projection import Camera, project_gaussians
from .binning import bin_gaussians_to_tiles

TILE_SIZE = 16

# Register Mojo kernels in Torch
mojo_kernels = Path(__file__).parent / "kernels"
inference_session = engine.InferenceSession(
    devices=[Accelerator()],
    custom_extensions=[mojo_kernels],
)
with torch.no_grad():
    mtorch.register_custom_ops(inference_session)

@torch.no_grad()
def render_gaussians(
    means3d: torch.Tensor, # (N, 3) World coordinates
    scales: torch.Tensor, # (N, 3) Scale factors (log-space)
    rotations: torch.Tensor, # (N, 4) Quaternions for orientation (w, x, y, z)
    opacities: torch.Tensor, # (N, 1) Opacity features (often pre-activation)
    features: torch.Tensor, # (N, C) Color features (e.g., RGB or SH coefficients)
    camera: Camera, # Camera object with intrinsics/extrinsics, H, W, near, far
    backend: str = "triton",
    # Optional args
    sh_degree: int | None = None, # Degree of Spherical Harmonics if used
    background_color: torch.Tensor | None = None, # (C,) Background color
    tile_size: int = TILE_SIZE, # Allow overriding tile size
) -> torch.Tensor:
    """Main function to render 3D Gaussians.

    Orchestrates projection, rasterization, and blending.
    """
    assert backend == "triton", "Only Triton backend is supported for now."
    required_tensors = [means3d, scales, rotations, opacities, features]
    if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in required_tensors):
        raise ValueError("All input gaussian tensors must be CUDA tensors.")

    # Prepare background color tensor
    num_channels = features.shape[-1]
    if background_color is None:
        background_color_tensor = torch.zeros(num_channels, device=means3d.device, dtype=features.dtype)
    elif not isinstance(background_color, torch.Tensor):
         background_color_tensor = torch.tensor(background_color, device=means3d.device, dtype=features.dtype)
    else:
        background_color_tensor = background_color.to(device=means3d.device, dtype=features.dtype)

    if background_color_tensor.shape[0] != num_channels:
         raise ValueError(f"Background color channels ({background_color_tensor.shape[0]}) must match gaussian color channels ({num_channels})")

    # --- 1. Projection ---
    means2d, covs2d, depths, projected_opacities, radii = project_gaussians(
        means3d, scales, rotations, opacities, camera
    )

    # --- 2. Binning & Sorting ---
    sorted_gaussian_indices, tile_pointers, tile_ranges = bin_gaussians_to_tiles(
        means2d, radii, depths, camera.H, camera.W, tile_size
    )

    # --- 3. Cull Gaussians (Optional - Placeholder) ---
    # Culling would ideally happen *before* projection/binning
    colors = features # Placeholder - may need SH evaluation first

    # --- 4. Evaluate SH (Optional - Placeholder) ---
    if sh_degree is not None:
        # TODO: Implement SH evaluation
        print("WARN: SH evaluation not implemented yet.")
        if features.shape[-1] > 3:
             colors = features[..., :3] # Crude approximation

    # --- 5. Rasterization & Blending --- 
    # Call the kernel wrapper from kernels.py
    # final_image = rasterize_gaussians_forward(
    #     means2d=means2d,
    #     covs2d=covs2d,
    #     depths=depths,
    #     opacities=projected_opacities,
    #     colors=colors,
    #     sorted_gaussian_indices=sorted_gaussian_indices,
    #     tile_ranges=tile_ranges,
    #     img_height=camera.H,
    #     img_width=camera.W,
    #     background_color=background_color_tensor,
    #     tile_size=tile_size,
    # )
    means2d = means2d.unsqueeze(0)
    covs2d = covs2d.unsqueeze(0)
    colors = colors.unsqueeze(0)
    projected_opacities = projected_opacities.unsqueeze(0)
    background_color_tensor = background_color_tensor.unsqueeze(0)
    tile_ranges = tile_ranges.unsqueeze(0)
    sorted_gaussian_indices = sorted_gaussian_indices.unsqueeze(0)
    print(f"means2d: {means2d.shape}, covs2d: {covs2d.shape}, colors: {colors.shape}, projected_opacities: {projected_opacities.shape}, background_color_tensor: {background_color_tensor.shape}, tile_ranges: {tile_ranges.shape}, sorted_gaussian_indices: {sorted_gaussian_indices.shape}")

    template = torch.zeros(1, camera.H, camera.W, num_channels, device=means2d.device, dtype=colors.dtype)
    final_image = torch.ops.modular_ops.rasterize_to_pixels_3dgs_fwd(
        means2d,
        covs2d,
        colors,
        projected_opacities,
        background_color_tensor,
        tile_ranges,
        sorted_gaussian_indices,
        template,
        mojo_parameters={
            "tile_size": tile_size,
            "image_height": camera.H,
            "image_width": camera.W,
            "CDIM": num_channels,
        }
    )

    print(f"final_image: {final_image.shape}")
    print(f"max_val: {final_image.max()}")

    # --- 6. Blending (Removed - Done in rasterizer) ---
    # final_image = alpha_blend(rasterized_output, camera)

    # Add background color (Removed - Applied in kernel)

    return final_image
    