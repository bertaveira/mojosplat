from mojosplat.render import render_gaussians
from mojosplat.projection import Camera

import torch
import os
from PIL import Image
import numpy as np
import time


# Helper function for camera view matrix
def look_at(eye, target, up):
    """Calculates the view matrix for a camera looking at a target."""
    eye = eye.float()
    target = target.float()
    up = up.float()
    device = eye.device

    forward = torch.nn.functional.normalize(target - eye, dim=0)
    right = torch.nn.functional.normalize(torch.cross(forward, up, dim=0), dim=0)
    down = torch.cross(right, forward, dim=0)

    # gsplat convention: +X right, +Y down, +Z forward (into scene)
    R_t = torch.stack([right, down, forward], dim=0)
    t = -torch.matmul(R_t, eye)

    view_matrix = torch.eye(4, device=device, dtype=eye.dtype)
    view_matrix[:3, :3] = R_t
    view_matrix[:3, 3] = t
    return view_matrix


def main():
    # --- Configuration ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("Warning: CUDA not available, running on CPU (slow).")
        print("Triton backend requires CUDA.")
        # If you want to proceed on CPU for testing setup (but rendering will fail)
        # return 
        # For now, let's try to proceed assuming CUDA is available for tensor creation
        # but acknowledge the render call will likely fail.
        # Better: Exit if no CUDA
        if not torch.cuda.is_available():
             print("Error: CUDA device not found. Triton backend requires CUDA.")
             return


    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "render_example.png")

    img_height = 1080
    img_width = 1920
    num_gaussians = 10000 # Number of Gaussians to render
    num_channels = 3 # RGB

    # --- Camera Setup ---
    # Simple perspective camera looking at the origin
    cam_pos = torch.tensor([0., 1.5, 5.0], device=device)
    cam_target = torch.tensor([0., 0., 0.], device=device)
    cam_up = torch.tensor([0., 1., 0.], device=device)
    view_matrix = look_at(cam_pos, cam_target, cam_up)

    focal_length = 500.0
    fx = focal_length
    fy = focal_length
    cx = img_width / 2.0
    cy = img_height / 2.0
    near_plane = 0.1
    far_plane = 100.0

    # Extract R and T from the view matrix
    R = view_matrix[:3, :3]
    T = view_matrix[:3, 3]

    camera = Camera(
        H=img_height, W=img_width,
        fx=fx, fy=fy, cx=cx, cy=cy,
        R=R, T=T, # Pass R and T instead of view_matrix
        near=near_plane, far=far_plane,
        # device=device # Device is inferred from R/T
    )

    # --- Generate Random Gaussian Data ---
    torch.manual_seed(42)
    print(f"Generating {num_gaussians} random Gaussians...")

    # Means centered around origin, spread out
    means3d = torch.randn(num_gaussians, 3, device=device) * 2.0

    log_scales = torch.ones(num_gaussians, 3, device=device) * -2.0  # exp(-2) ≈ 0.14
    log_scales += torch.randn(num_gaussians, 3, device=device) * 0.3

    # Random quaternions (w, x, y, z format)
    random_quats = torch.randn(num_gaussians, 4, device=device)
    quats = torch.nn.functional.normalize(random_quats, dim=1)

    opacities = torch.sigmoid(torch.randn(num_gaussians, device=device) + 1.0)
    
    # RGB Colors
    colors = torch.rand(num_gaussians, num_channels, device=device)

    # Ensure correct dtypes
    means3d = means3d.float()
    log_scales = log_scales.float()
    quats = quats.float()
    opacities = opacities.float()
    colors = colors.float()

    # --- Rendering ---
    print("Rendering...")
    print(f"Input shapes: means3d={means3d.shape}, scales={log_scales.shape}, quats={quats.shape}, opacities={opacities.shape}, colors={colors.shape}")
    
    rendered_image = render_gaussians(
        means3d=means3d,
        scales=log_scales, # Pass log-scales directly
        quats=quats, # Pass quaternions (w, x, y, z)
        opacities=opacities,
        features=colors,
        camera=camera,
        background_color=torch.tensor([0.1, 0.1, 0.1], device=device), # Dark gray background
    )
    
    print(f"Rendered image shape: {rendered_image.shape}")
    print(f"Rendered image range: [{rendered_image.min().item():.4f}, {rendered_image.max().item():.4f}]")

    # --- Save Output ---
    print(f"Saving image to {output_path}...")
    # Convert from torch tensor (H, W, C) on GPU to numpy array (H, W, C) on CPU
    output_image_np = (rendered_image.cpu().numpy() * 255).astype(np.uint8)
    
    # Create PIL image and save
    pil_image = Image.fromarray(output_image_np)
    pil_image.save(output_path)
    print("Done.")

if __name__ == "__main__":
    main() 