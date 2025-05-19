import torch
import math

# --- Helper Functions ---

def quat_to_rotmat(quat):
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

def build_covariance(scales, rotations):
    """Build 3D covariance matrix from scales and normalized quaternions."""
    N = scales.shape[0]
    # Ensure rotations are normalized
    rotations = torch.nn.functional.normalize(rotations, p=2, dim=-1)
    # Convert scales from log space if necessary (assuming they are radii now)
    # scales = torch.exp(scales) # Use if scales are log-radii
    S = torch.diag_embed(scales * scales) # Covariance is Scale * Scale^T = diag(scale^2)
    R = quat_to_rotmat(rotations) # (N, 3, 3)
    # Sigma = R @ S @ R.transpose(-1, -2)
    # Optimized: Compute R @ S directly
    # RS = R * scales.unsqueeze(-1) # Incorrect logic
    # Sigma = RS @ R.transpose(-1, -2) # Incorrect logic

    # Correct batch matrix multiplication: R @ S @ R.T
    # S is (N, 3, 3), R is (N, 3, 3)
    Sigma = torch.bmm(torch.bmm(R, S), R.transpose(-1, -2))
    return Sigma

def build_view_matrix(R, T):
    """Builds the 4x4 view matrix."""
    view_matrix = torch.eye(4, device=R.device, dtype=R.dtype)
    view_matrix[:3, :3] = R
    view_matrix[:3, 3] = T
    return view_matrix

# --- Camera Class ---
class Camera:
    def __init__(self, R, T, fx, fy, cx, cy, H, W, near=0.1, far=100.0):
        self.R = R # Rotation (3, 3) world-to-camera
        self.T = T # Translation (3,) world-to-camera
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.H = H
        self.W = W
        self.near = near
        self.far = far
        # Store precomputed matrices
        self.view_matrix = build_view_matrix(self.R, self.T)
        # Store intrinsics tensor for convenience
        self.K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=R.device, dtype=R.dtype)

# --- Projection Function ---
def project_gaussians(
    means3d: torch.Tensor, # (N, 3)
    scales: torch.Tensor, # (N, 3) log-scales
    rotations: torch.Tensor, # (N, 4) w,x,y,z quaternion
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
    # z = torch.clamp(depths, min=1e-6) # Avoid zero (Old)
    # Clamp z for projection to avoid issues with points behind camera or exactly at center
    z_proj = torch.clamp(z, max=-1e-6) # Ensure z is negative and non-zero

    means_cam_proj = means_cam @ camera.K.T
    # Use z_proj for division
    means2d_x = means_cam_proj[:, 0] / z_proj
    means2d_y = means_cam_proj[:, 1] / z_proj
    means2d = torch.stack([means2d_x, means2d_y], dim=-1)

    # --- 3. Project Covariance ---
    Sigma3D = build_covariance(torch.exp(scales), rotations) # Assuming scales are log-scales
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

    # --- 4. Calculate Radii ---
    a, b, c = covs2d[:, 0], covs2d[:, 1], covs2d[:, 2]
    trace = a + c
    discriminant = torch.clamp((a - c)**2 + 4 * b**2, min=1e-8)
    max_eigenvalue = 0.5 * (trace + torch.sqrt(discriminant))
    radii = torch.ceil(3.0 * torch.sqrt(max_eigenvalue))

    # --- 5. Opacity ---
    opacities = torch.sigmoid(opacity_features).squeeze(-1) # (N,)

    # --- 6. Return ---
    # Return positive depth
    return means2d, covs2d, depths_for_return, opacities, radii 