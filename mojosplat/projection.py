from pathlib import Path
import torch
from typing import Literal
from max.torch import CustomOpLibrary
from .utils import Camera

mojo_kernels = Path(__file__).parent / "kernels"
op_library = CustomOpLibrary(mojo_kernels)

def project_gaussians(
    means3d: torch.Tensor, # (N, 3)
    scales: torch.Tensor, # (N, 3)
    quats: torch.Tensor, # (N, 4)
    opacity_features: torch.Tensor, # (N, 1)
    camera: Camera,
    backend: Literal["torch", "gsplat", "mojo"] = "torch"
) -> tuple:
    """Projects 3D Gaussians to 2D image plane.

    Args:
        means3d: (N, 3)
        scales: (N, 3)
        quats: (N, 4) w,x,y,z quaternion
        opacity_features: (N, 1)
        camera: Camera
        backend: Literal["torch", "gsplat", "mojo"]

    Returns:
        means2d: (N, 2) xy coordinates in pixel space.
        conics: (N, 3) inverse of the projected covariances. Return the flattend upper triangle
        depths: (N,) depth of each Gaussian center in camera space.
        opacities: (N,) sigmoid applied opacity.
        radii: (N,) estimated radius in pixels for conservative rasterization bounds.
    """
    if backend == "torch":
        means2d, conics, depths, radii = project_gaussians_torch(means3d, scales, quats, opacity_features, camera)
    elif backend == "gsplat":
        means2d, conics, depths, radii = project_gaussians_gsplat(means3d, scales, quats, opacity_features, camera)
    elif backend == "mojo":
        means2d, conics, depths, radii = project_gaussians_mojo(means3d, scales, quats, opacity_features, camera)
    else:
        raise ValueError(f"Invalid backend: {backend}")
    return means2d, conics, depths, radii 


#    d888888b  .d88b.  d8888b.  .o88b. db   db 
#    `~~88~~' .8P  Y8. 88  `8D d8P  Y8 88   88 
#       88    88    88 88oobY' 8P      88ooo88 
#       88    88    88 88`8b   8b      88~~~88 
#       88    `8b  d8' 88 `88. Y8b  d8 88   88 
#       YP     `Y88P'  88   YD  `Y88P' YP   YP 

def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = torch.unbind(quat, -1)
    N = quat.shape[0]
    mat = torch.zeros((N, 3, 3), device=quat.device, dtype=quat.dtype)

    mat[:, 0, 0] = 1 - 2*y*y - 2*z*z
    mat[:, 0, 1] = 2*x*y - 2*z*w
    mat[:, 0, 2] = 2*x*z + 2*y*w

    mat[:, 1, 0] = 2*x*y + 2*z*w
    mat[:, 1, 1] = 1 - 2*x*x - 2*z*z
    mat[:, 1, 2] = 2*y*z - 2*x*w

    mat[:, 2, 0] = 2*x*z - 2*y*w
    mat[:, 2, 1] = 2*y*z + 2*x*w
    mat[:, 2, 2] = 1 - 2*x*x - 2*y*y
    return mat

def build_covariance(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """Build 3D covariance matrix from scales and normalized quaternions."""
    N = scales.shape[0]
    # Ensure rotations are normalized
    quats = torch.nn.functional.normalize(quats, p=2, dim=-1)
    # Convert scales from log space if necessary (assuming they are radii now)
    # scales = torch.exp(scales) # Use if scales are log-radii
    S = torch.diag_embed(scales * scales) # Covariance is Scale * Scale^T = diag(scale^2)
    R = quat_to_rotmat(quats) # (N, 3, 3)
    # Sigma = R @ S @ R.transpose(-1, -2)
    # Optimized: Compute R @ S directly
    # RS = R * scales.unsqueeze(-1) # Incorrect logic
    # Sigma = RS @ R.transpose(-1, -2) # Incorrect logic

    # Correct batch matrix multiplication: R @ S @ R.T
    # S is (N, 3, 3), R is (N, 3, 3)
    Sigma = torch.bmm(torch.bmm(R, S), R.transpose(-1, -2))
    return Sigma

# --- Projection Function ---
def project_gaussians_torch(
    means3d: torch.Tensor, # (N, 3)
    scales: torch.Tensor, # (N, 3) log-scales
    quats: torch.Tensor, # (N, 4) w,x,y,z quaternion
    opacity_features: torch.Tensor, # (N, 1)
    camera: Camera,
) -> tuple:
    """Projects 3D Gaussians to 2D image plane.

    Computes 2D means, projected 2D covariance, depths, and opacity.
    Handles view transformation and perspective projection.

    Returns:
        Tuple containing:
        - means2d: (N, 2) xy coordinates in pixel space.
        - covs2d: (N, 3) upper triangular part of 2D covariance matrix (conic: a, b, c).
        - depths: (N,) depth of each Gaussian center in camera space.
        - opacities: (N,) sigmoid applied opacity.
        - radii: (N,) estimated radius in pixels for conservative rasterization bounds.
    """
    N = means3d.shape[0]
    device = means3d.device
    dtype = means3d.dtype

    # --- 1. Transform means ---
    means_cam = means3d @ camera.R.T + camera.T.unsqueeze(0)
    # depths = means_cam[:, 2] # Z in camera space (can be negative)
    z = means_cam[:, 2] # Z in camera space (can be negative)
    depths_for_return = -z # Depth is positive distance along viewing direction

    # --- 2. Project to Pixels ---
    z_proj = torch.clamp(z, max=-1e-6) # Ensure z is negative and non-zero

    means_cam_proj = means_cam @ camera.Ks.transpose(0, 1)
    # Use z_proj for division
    means2d_x = means_cam_proj[:, 0] / z_proj
    means2d_y = means_cam_proj[:, 1] / z_proj
    means2d = torch.stack([means2d_x, means2d_y], dim=-1)

    # --- 3. Project Covariance ---
    Sigma3D = build_covariance(torch.exp(scales), quats) # Assuming scales are log-scales
    W = camera.R # View rotation matrix
    Sigma_cam = torch.bmm(torch.bmm(W.unsqueeze(0).expand(N, -1, -1), Sigma3D), W.T.unsqueeze(0).expand(N, -1, -1))

    # Compute Jacobian J
    x, y = means_cam[:, 0], means_cam[:, 1]
    # Use z_proj here as well
    inv_z = 1.0 / z_proj
    inv_z2 = inv_z * inv_z
    J = torch.zeros((N, 2, 3), device=device, dtype=dtype)
    J[:, 0, 0] = camera.fx * inv_z
    J[:, 0, 2] = -camera.fx * x * inv_z2
    J[:, 1, 1] = camera.fy * inv_z
    J[:, 1, 2] = -camera.fy * y * inv_z2

    # Project covariance: Sigma_2D = J @ Sigma_cam @ J.T
    Sigma_2D = torch.bmm(torch.bmm(J, Sigma_cam), J.transpose(-1, -2))
    covs2d = torch.stack([Sigma_2D[:, 0, 0], Sigma_2D[:, 0, 1], Sigma_2D[:, 1, 1]], dim=-1)

    # Compute conics (inverse of the projected covariances)
    Sigma_2D_inv = torch.linalg.inv(Sigma_2D)
    conics = torch.stack([Sigma_2D_inv[:, 0, 0], Sigma_2D_inv[:, 0, 1], Sigma_2D_inv[:, 1, 1]], dim=-1)

    # --- 4. Calculate Radii ---
    a, b, c = covs2d[:, 0], covs2d[:, 1], covs2d[:, 2]
    trace = a + c
    discriminant = torch.clamp((a - c)**2 + 4 * b**2, min=1e-8)
    max_eigenvalue = 0.5 * (trace + torch.sqrt(discriminant))
    radii = torch.ceil(3.0 * torch.sqrt(max_eigenvalue))

    return means2d, conics, depths_for_return, radii 



#     d888b  .d8888. d8888b. db       .d8b.  d888888b 
#    88' Y8b 88'  YP 88  `8D 88      d8' `8b `~~88~~' 
#    88      `8bo.   88oodD' 88      88ooo88    88    
#    88  ooo   `Y8b. 88~~~   88      88~~~88    88    
#    88. ~8~ db   8D 88      88booo. 88   88    88    
#     Y888P  `8888Y' 88      Y88888P YP   YP    YP   

def project_gaussians_gsplat(
    means3d: torch.Tensor, # (N, 3)
    scales: torch.Tensor, # (N, 3)
    quats: torch.Tensor, # (N, 4)
    opacities: torch.Tensor, # (N, 1)
    camera: Camera,
) -> tuple:
    """Projects 3D Gaussians to 2D image plane."""
    from gsplat import fully_fused_projection

    view_matrix = camera.view_matrix.unsqueeze(0).unsqueeze(0)
    Ks = camera.Ks.unsqueeze(0).unsqueeze(0)

    means3d = means3d.unsqueeze(0)
    scales = scales.unsqueeze(0)
    quats = quats.unsqueeze(0)
    opacities = opacities.unsqueeze(0).view(1, -1)

    proj_results = fully_fused_projection(
        means3d,
        None,
        quats,
        scales,
        view_matrix,
        Ks,
        camera.W,
        camera.H,
        packed=False,
        # camera_model="pinhole",
        opacities=opacities,  # use opacities to compute a tigher bound for radii.
    )

    radii, means2d, depths, conics, _ = proj_results

    return means2d, conics, depths, radii



#    .88b  d88.  .d88b.     d88b  .d88b.  
#    88'YbdP`88 .8P  Y8.    `8P' .8P  Y8. 
#    88  88  88 88    88     88  88    88 
#    88  88  88 88    88     88  88    88 
#    88  88  88 `8b  d8' db. 88  `8b  d8' 
#    YP  YP  YP  `Y88P'  Y8888P   `Y88P'  

def project_gaussians_mojo(
    means3d: torch.Tensor, # (N, 3)
    scales: torch.Tensor, # (N, 3)
    quats: torch.Tensor, # (N, 4)
    opacities: torch.Tensor, # (N, 1)
    camera: Camera,
) -> tuple:
    """Projects 3D Gaussians to 2D image plane."""
    
    project_gaussians_kernel = op_library.project_gaussians[
        {
            "C": 1,
            "N": means3d.shape[0],
            "image_width": camera.W,
            "image_height": camera.H,
        }
    ]

    means2d = torch.zeros((1, means3d.shape[0], 2), dtype=torch.float32, device=means3d.device).contiguous()
    conics = torch.zeros((1, means3d.shape[0], 3), dtype=torch.float32, device=means3d.device).contiguous()
    depth = torch.zeros((1, means3d.shape[0]), dtype=torch.float32, device=means3d.device).contiguous()  # Shape (C, N) not (C, N, 1)
    radii = torch.zeros((1, means3d.shape[0], 2), dtype=torch.int32, device=means3d.device).contiguous()  # int32 and shape (C, N, 2)

    view_matrix = camera.view_matrix.unsqueeze(0).contiguous()  # Shape: (1, 4, 4)
    # Convert camera intrinsics to the format expected by Mojo kernel: [fx, fy, cx, cy, k1, k2, k3, k4, k5]
    # For pinhole camera, distortion coefficients k1-k5 are zero
    ks_flat = torch.tensor([[camera.fx, camera.fy, camera.cx, camera.cy, 0.0, 0.0, 0.0, 0.0, 0.0]], 
                          device=means3d.device, dtype=torch.float32).view(1, -1).contiguous()

    means3d = means3d.contiguous()
    scales = scales.contiguous()
    quats = quats.contiguous()
    opacities = opacities.view(-1).contiguous()

    project_gaussians_kernel(means2d, conics, depth, radii, means3d, scales, quats, opacities, view_matrix, ks_flat)

    return means2d, conics, depth, radii
