from mojosplat.render import render_gaussians
from mojosplat.projection import Camera

import torch
import os
from PIL import Image
import numpy as np


# Helper function for camera view matrix
def look_at(eye, target, up):
    """Calculates the view matrix for a camera looking at a target."""
    eye = eye.float()
    target = target.float()
    up = up.float()
    device = eye.device

    z_axis = torch.nn.functional.normalize(eye - target, dim=0)
    x_axis = torch.nn.functional.normalize(torch.cross(up, z_axis, dim=0), dim=0)
    y_axis = torch.nn.functional.normalize(torch.cross(z_axis, x_axis, dim=0), dim=0)

    # World-to-camera rotation
    R_t = torch.stack([x_axis, y_axis, z_axis], dim=0)
    # World-to-camera translation
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

    img_height = 512
    img_width = 512
    num_gaussians = 100 # Number of Gaussians to render
    num_channels = 3 # RGB

    # --- Camera Setup ---
    # Simple perspective camera looking at the origin
    cam_pos = torch.tensor([0., 1.5, 5.0], device=device)
    cam_target = torch.tensor([0., 0., 0.], device=device)
    cam_up = torch.tensor([0., 1., 0.], device=device)
    view_matrix = look_at(cam_pos, cam_target, cam_up)

    # Approximate perspective projection
    focal_length = 300.0 # Adjust for zoom
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
    print(f"Generating {num_gaussians} random Gaussians...")

    # Means centered around origin, spread out
    means3d = torch.randn(num_gaussians, 3, device=device) * 2.0

    # Log-scales, mostly small
    # Use small, fixed scales instead of random ones
    log_scales = torch.ones(num_gaussians, 3, device=device) * -3.0  # exp(-3) ~= 0.05, even smaller
    # Add some slight variation to make it more natural
    log_scales += torch.randn(num_gaussians, 3, device=device) * 0.1  # Smaller random variation

    # Random rotations (unit quaternions w, x, y, z)
    random_rots = torch.randn(num_gaussians, 4, device=device)
    rotations = torch.nn.functional.normalize(random_rots, dim=1)

    # Opacities (pre-activation, sigmoid is often applied later)
    # Let's provide values that would be reasonable post-sigmoid, e.g., 0.1 to 0.9
    # To achieve this with sigmoid, pre-activation needs to span roughly -2 to 2
    # Simpler: just use random values and apply sigmoid in projection if needed
    # Let's assume render_gaussians expects pre-sigmoid opacities
    opacities_pre_sigmoid = torch.randn(num_gaussians, 1, device=device) # Centered around 0
    
    # RGB Colors
    colors = torch.rand(num_gaussians, num_channels, device=device)

    # Ensure correct dtypes
    means3d = means3d.float()
    log_scales = log_scales.float()
    rotations = rotations.float()
    opacities_pre_sigmoid = opacities_pre_sigmoid.float()
    colors = colors.float()

    # --- Rendering ---
    print("Rendering...")
    rendered_image = render_gaussians(
        means3d=means3d,
        scales=log_scales, # Pass log-scales directly
        rotations=rotations,
        opacities=opacities_pre_sigmoid, # Pass pre-activation opacities
        features=colors,
        camera=camera,
        background_color=torch.tensor([0.1, 0.1, 0.1], device=device), # Dark gray background
    )

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