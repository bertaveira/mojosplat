
import torch
from typing_extensions import Literal

from .projection import Camera, project_gaussians
from .binning import bin_gaussians_to_tiles
from .rasterization import rasterize_gaussians

TILE_SIZE = 16

@torch.no_grad()
def render_gaussians(
    means3d: torch.Tensor, # (N, 3) World coordinates
    scales: torch.Tensor, # (N, 3) Scale factors (log-space)
    quats: torch.Tensor, # (N, 4) Quaternions for orientation (w, x, y, z)
    opacities: torch.Tensor, # (N) Opacity features (often pre-activation)
    features: torch.Tensor, # (N, C) Color features (e.g., RGB or SH coefficients)
    camera: Camera, # Camera object with intrinsics/extrinsics, H, W, near, far
    # Optional args
    sh_degree: int | None = None, # Degree of Spherical Harmonics if used
    background_color: torch.Tensor | None = None, # (C,) Background color
    tile_size: int = TILE_SIZE, # Allow overriding tile size
    backend: Literal["torch", "gsplat", "mojo"] = "mojo", # Backend to use for projection, binning, and rasterization
) -> torch.Tensor:
    """Main function to render 3D Gaussians.

    Orchestrates projection, binning, and rasterization with consistent backend usage.
    
    Args:
        means3d: (N, 3) World coordinates of Gaussian centers
        scales: (N, 3) Scale factors in log-space
        quats: (N, 4) Quaternions for orientation (w, x, y, z)
        opacities: (N, 1) Opacity features (often pre-activation)
        features: (N, C) Color features (e.g., RGB or SH coefficients)
        camera: Camera object with intrinsics/extrinsics
        sh_degree: Degree of Spherical Harmonics if used (optional)
        background_color: (C,) Background color tensor (optional)
        tile_size: Size of square tiles in pixels (default: 16)
        backend: Backend to use for projection, binning, and rasterization ("torch", "gsplat", "mojo")
        
    Returns:
        final_image: (H, W, C) Rendered image
    """
    required_tensors = [means3d, scales, quats, opacities, features]
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

    assert opacities.shape == (means3d.shape[0],)

    # --- 1. Projection ---
    means2d, covs2d, depths, radii = project_gaussians(
        means3d, scales, quats, opacities, camera, backend=backend
    )

    # --- 2. Binning & Sorting ---
    sorted_gaussian_indices, tile_ranges = bin_gaussians_to_tiles(
        means2d, radii, depths, camera.H, camera.W, tile_size, backend=backend
    )
    
    # Validate binning outputs
    if sorted_gaussian_indices.numel() == 0:
        print("Warning: No Gaussian overlaps found. This might indicate all Gaussians are outside the image or culled.")
        # Return black image
        return torch.zeros(camera.H, camera.W, num_channels, device=means3d.device, dtype=features.dtype)

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
    print(f"means2d: {means2d.shape}, covs2d: {covs2d.shape}, colors: {colors.shape}, opacities: {opacities.shape}, background_color_tensor: {background_color_tensor.shape}, tile_ranges: {tile_ranges.shape}, sorted_gaussian_indices: {sorted_gaussian_indices.shape}")

    final_image = rasterize_gaussians(
        means2d,
        covs2d,
        colors,
        opacities,
        background_color_tensor,
        tile_ranges,
        sorted_gaussian_indices,
        camera,
        tile_size=tile_size,
        backend=backend,
    )
    print("Done")
    print(f"final_image: {final_image.shape}")
    print(f"max_val: {final_image.max()}")

    return final_image
    