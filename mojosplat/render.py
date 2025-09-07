
from pathlib import Path
import torch

from max.torch import CustomOpLibrary

from .projection import Camera, project_gaussians
from .binning import bin_gaussians_to_tiles

TILE_SIZE = 16

# Register Mojo kernels in Torch
mojo_kernels = Path(__file__).parent / "kernels"
op_library = CustomOpLibrary(mojo_kernels)
# rasterize_to_pixels_3dgs_fwd_kernel = op_library.rasterize_to_pixels_3dgs_fwd[
#     {
#         "tile_size": TILE_SIZE,
#         "image_height": 512,
#         "image_width": 512,
#         "CDIM": 3,
#         "C": 1,
#         "N": 1,
#         "NIntersections": 1,
#         "image_width": 512,
#         "image_height": 512,
#     }
# ]

def rasterize_to_pixels_3dgs_fwd(
    means2d: torch.Tensor,
    covs2d: torch.Tensor,
    colors: torch.Tensor,
    projected_opacities: torch.Tensor,
    background_color_tensor: torch.Tensor,
    tile_ranges: torch.Tensor,
    sorted_gaussian_indices: torch.Tensor,
    camera: Camera,
    num_channels: int,
) -> torch.Tensor:
    result = torch.zeros(1, camera.H, camera.W, num_channels, device=means2d.device, dtype=means2d.dtype)
    rasterize_to_pixels_3dgs_fwd_kernel = op_library.rasterize_to_pixels_3dgs_fwd[
        {
            "tile_size": TILE_SIZE,
            "image_height": camera.H,
            "image_width": camera.W,
            "CDIM": 3,
            "C": 1,
            "N": means2d.shape[1],
            "NIntersections": sorted_gaussian_indices.shape[1],
        }
    ]
    rasterize_to_pixels_3dgs_fwd_kernel(result, means2d, covs2d, colors, projected_opacities, background_color_tensor, tile_ranges, sorted_gaussian_indices)

    print("Back in python")
    return result

@torch.no_grad()
def render_gaussians(
    means3d: torch.Tensor, # (N, 3) World coordinates
    scales: torch.Tensor, # (N, 3) Scale factors (log-space)
    quats: torch.Tensor, # (N, 4) Quaternions for orientation (w, x, y, z)
    opacities: torch.Tensor, # (N, 1) Opacity features (often pre-activation)
    features: torch.Tensor, # (N, C) Color features (e.g., RGB or SH coefficients)
    camera: Camera, # Camera object with intrinsics/extrinsics, H, W, near, far
    # Optional args
    sh_degree: int | None = None, # Degree of Spherical Harmonics if used
    background_color: torch.Tensor | None = None, # (C,) Background color
    tile_size: int = TILE_SIZE, # Allow overriding tile size
    backend: str = "torch", # Backend to use for projection and binning
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
        backend: Backend to use for projection and binning ("torch", "gsplat", "mojo")
        
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

    # --- 1. Projection ---
    means2d, covs2d, depths, radii = project_gaussians(
        means3d, scales, quats, opacities, camera, backend=backend
    )

    # --- 2. Binning & Sorting ---
    sorted_gaussian_indices, tile_ranges = bin_gaussians_to_tiles(
        means2d, radii, depths, camera.H, camera.W, tile_size, backend="gsplat"
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

    # Prepare tensors for rasterization kernel
    means2d = means2d.unsqueeze(0).contiguous()
    covs2d = covs2d.unsqueeze(0).contiguous()
    colors = colors.unsqueeze(0).contiguous()
    background_color_tensor = background_color_tensor.unsqueeze(0).contiguous()
    tile_ranges = tile_ranges.unsqueeze(0).to(torch.int32).contiguous()
    sorted_gaussian_indices = sorted_gaussian_indices.unsqueeze(0).to(torch.int32).contiguous()
    print(f"means2d: {means2d.shape}, covs2d: {covs2d.shape}, colors: {colors.shape}, opacities: {opacities.shape}, background_color_tensor: {background_color_tensor.shape}, tile_ranges: {tile_ranges.shape}, sorted_gaussian_indices: {sorted_gaussian_indices.shape}")

    final_image = rasterize_to_pixels_3dgs_fwd(
        means2d,
        covs2d,
        colors,
        opacities,
        background_color_tensor,
        tile_ranges,
        sorted_gaussian_indices,
        camera,
        num_channels,
    )
    print("Done")
    print(f"final_image: {final_image.shape}")
    print(f"max_val: {final_image.max()}")

    # --- 6. Blending (Removed - Done in rasterizer) ---
    # final_image = alpha_blend(rasterized_output, camera)

    # Add background color (Removed - Applied in kernel)

    # Remove batch dimension to return (H, W, C) instead of (1, H, W, C)
    final_image = final_image.squeeze(0)

    return final_image
    