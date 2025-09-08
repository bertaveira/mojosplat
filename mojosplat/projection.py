from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from typing_extensions import Literal

from pathlib import Path
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
        radii: (N, 2) estimated radius in pixels for conservative rasterization bounds.
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


def _quat_to_rotmat(quats: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return R.reshape(quats.shape[:-1] + (3, 3))


def _quat_scale_to_covar_preci(
    quats: Tensor,  # [..., 4],
    scales: Tensor,  # [..., 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.quat_scale_to_covar_preci()`."""
    batch_dims = quats.shape[:-1]
    assert quats.shape == batch_dims + (4,), quats.shape
    assert scales.shape == batch_dims + (3,), scales.shape
    R = _quat_to_rotmat(quats)  # [..., 3, 3]

    if compute_covar:
        M = R * scales[..., None, :]  # [..., 3, 3]
        covars = torch.einsum("...ij,...kj -> ...ik", M, M)  # [..., 3, 3]
        if triu:
            covars = covars.reshape(batch_dims + (9,))  # [..., 9]
            covars = (
                covars[..., [0, 1, 2, 4, 5, 8]] + covars[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # [..., 6]
    if compute_preci:
        P = R * (1 / scales[..., None, :])  # [..., 3, 3]
        precis = torch.einsum("...ij,...kj -> ...ik", P, P)  # [..., 3, 3]
        if triu:
            precis = precis.reshape(batch_dims + (9,))  # [..., 9]
            precis = (
                precis[..., [0, 1, 2, 4, 5, 8]] + precis[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # [..., 6]

    return covars if compute_covar else None, precis if compute_preci else None


def _persp_proj(
    means: Tensor,  # [..., C, N, 3]
    covars: Tensor,  # [..., C, N, 3, 3]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of perspective projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [..., C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [..., C, N, 3, 3].
        Ks: Camera intrinsics. [..., C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [..., C, N, 2].
        - **cov2d**: Projected covariances. [..., C, N, 2, 2].
    """
    batch_dims = means.shape[:-3]
    C, N = means.shape[-3:-1]
    assert means.shape == batch_dims + (C, N, 3), means.shape
    assert covars.shape == batch_dims + (C, N, 3, 3), covars.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [..., C, N]
    tz2 = tz**2  # [..., C, N]

    fx = Ks[..., 0, 0, None]  # [..., C, 1]
    fy = Ks[..., 1, 1, None]  # [..., C, 1]
    cx = Ks[..., 0, 2, None]  # [..., C, 1]
    cy = Ks[..., 1, 2, None]  # [..., C, 1]
    tan_fovx = 0.5 * width / fx  # [..., C, 1]
    tan_fovy = 0.5 * height / fy  # [..., C, 1]

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

    O = torch.zeros(batch_dims + (C, N), device=means.device, dtype=means.dtype)
    J = torch.stack(
        [fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1
    ).reshape(batch_dims + (C, N, 2, 3))

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = torch.einsum(
        "...ij,...nj->...ni", Ks[..., :2, :3], means
    )  # [..., C, N, 2]
    means2d = means2d / tz[..., None]  # [..., C, N, 2]
    return means2d, cov2d  # [..., C, N, 2], [..., C, N, 2, 2]


def _world_to_cam(
    means: Tensor,  # [..., N, 3]
    covars: Tensor,  # [..., N, 3, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of world to camera transformation on Gaussians.

    Args:
        means: Gaussian means in world coordinate system. [..., N, 3].
        covars: Gaussian covariances in world coordinate system. [..., N, 3, 3].
        viewmats: world to camera transformation matrices. [..., C, 4, 4].

    Returns:
        A tuple:

        - **means_c**: Gaussian means in camera coordinate system. [..., C, N, 3].
        - **covars_c**: Gaussian covariances in camera coordinate system. [..., C, N, 3, 3].
    """
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert covars.shape == batch_dims + (N, 3, 3), covars.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape

    R = viewmats[..., :3, :3]  # [..., C, 3, 3]
    t = viewmats[..., :3, 3]  # [..., C, 3]
    means_c = (
        torch.einsum("...cij,...nj->...cni", R, means) + t[..., None, :]
    )  # [..., C, N, 3]
    covars_c = torch.einsum(
        "...cij,...njk,...clk->...cnil", R, covars, R
    )  # [..., C, N, 3, 3]
    return means_c, covars_c


def _fully_fused_projection(
    means: Tensor,  # [..., N, 3]
    covars: Tensor,  # [..., N, 3, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert covars.shape == batch_dims + (N, 3, 3), covars.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape

    assert (
        camera_model != "ftheta"
    ), "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"

    means_c, covars_c = _world_to_cam(means, covars, viewmats)

    if camera_model == "pinhole":
        means2d, covars2d = _persp_proj(means_c, covars_c, Ks, width, height)
    else:
        raise ValueError(f"Unsupported camera model: {camera_model}")

    det_orig = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = det.clamp(min=1e-10)

    if calc_compensations:
        compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0))
    else:
        compensations = None

    conics = torch.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # [..., C, N, 3]

    depths = means_c[..., 2]  # [..., C, N]

    radius_x = torch.ceil(3.33 * torch.sqrt(covars2d[..., 0, 0]))
    radius_y = torch.ceil(3.33 * torch.sqrt(covars2d[..., 1, 1]))

    radius = torch.stack([radius_x, radius_y], dim=-1)  # [..., C, N, 2]

    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < width)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < height)
    )
    radius[~inside] = 0.0

    radii = radius.int()
    return radii, means2d, depths, conics, compensations

def project_gaussians_torch(
    means3d: torch.Tensor, # (N, 3)
    scales: torch.Tensor, # (N, 3)
    quats: torch.Tensor, # (N, 4) w,x,y,z quaternion
    opacity_features: torch.Tensor, # (N, 1)
    camera: Camera,
) -> tuple:
    """Projects 3D Gaussians to 2D image plane.

    Args:
        means3d: (N, 3)
        scales: (N, 3)
        quats: (N, 4) w,x,y,z quaternion
        opacity_features: (N, 1)
        camera: Camera

    Returns:
        means2d: (N, 2) xy coordinates in pixel space.
        conics: (N, 3) inverse of the projected covariances. Return the flattened upper triangle
        depths: (N,) depth of each Gaussian center in camera space.
        radii: (N, 2) estimated radius in pixels for conservative rasterization bounds.
    """
    N = means3d.shape[0]
    device = means3d.device
    dtype = means3d.dtype

    # Convert camera parameters to match expected format
    # Add batch and camera dimensions: [..., N, 3] -> [1, N, 3]
    means = means3d.unsqueeze(0)  # [1, N, 3]
    
    # Build covariance matrices from quaternions and scales  
    # Note: scales are in log space, need to convert to linear
    covars, _ = _quat_scale_to_covar_preci(
        quats, torch.exp(scales), compute_covar=True, compute_preci=False
    )  # [N, 3, 3]
    covars = covars.unsqueeze(0)  # [1, N, 3, 3]
    
    # Convert camera view matrix and intrinsics to match expected format
    viewmats = camera.view_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
    Ks = camera.Ks.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    
    # Use the fully fused projection with same parameters as GSplat backend
    radii, means2d, depths, conics, _ = _fully_fused_projection(
        means,
        covars,
        viewmats,
        Ks,
        camera.W,
        camera.H,
        eps2d=0.3,                  # Match GSplat backend
        near_plane=camera.near,     # Match GSplat backend  
        far_plane=camera.far,       # Match GSplat backend
        camera_model="pinhole"
    )
    
    # Remove batch dimensions to match expected output format
    means2d = means2d.squeeze(0).squeeze(0)  # [N, 2]
    conics = conics.squeeze(0).squeeze(0)    # [N, 3]
    depths = depths.squeeze(0).squeeze(0)    # [N]
    radii = radii.squeeze(0).squeeze(0)      # [N, 2]
    
    return means2d, conics, depths, radii 



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

    # GSplat expects specific dimensions:
    # means: [..., N, 3] - for single batch, this is [N, 3]
    # viewmats: [..., C, 4, 4] - for single camera, this is [1, 4, 4]  
    # Ks: [..., C, 3, 3] - for single camera, this is [1, 3, 3]
    view_matrix = camera.view_matrix.unsqueeze(0)  # [1, 4, 4]
    Ks = camera.Ks.unsqueeze(0)  # [1, 3, 3]

    # Input data is already in correct format:
    # means3d: [N, 3]
    # scales: [N, 3] - need to convert from log to linear
    # quats: [N, 4]
    # opacities: [N, 1] -> [N]
    opacities = opacities.view(-1)  # [N]

    proj_results = fully_fused_projection(
        means3d,                    # [N, 3]
        None,                       # No covariance input
        quats,                      # [N, 4] 
        torch.exp(scales),          # [N, 3] - convert from log to linear scale
        view_matrix,                # [1, 4, 4]
        Ks,                         # [1, 3, 3]
        camera.W,
        camera.H,
        eps2d=0.3,                  # Default epsilon for 2D covariance
        near_plane=camera.near,     # Use camera's near plane
        far_plane=camera.far,       # Use camera's far plane
        radius_clip=0.0,            # No radius clipping
        packed=False,
        camera_model="pinhole",
        opacities=opacities,      # [N] - GSplat culls gaussians with low opacity
    )

    # Function returns: (radii, means, depths, conics, compensations) when packed=False
    # Expected shapes: radii [1, N, 2], means [1, N, 2], depths [1, N], conics [1, N, 3], compensations [1, N]
    radii, means2d, depths, conics, compensations = proj_results
    
    # GSplat returns shapes [C=1, N, ...], squeeze camera dimension to get [N, ...]
    means2d = means2d.squeeze(0)  # [N, 2]
    conics = conics.squeeze(0)    # [N, 3]
    depths = depths.squeeze(0)    # [N]
    radii = radii.squeeze(0)      # [N, 2]
    
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

    project_gaussians_kernel(means2d, conics, depth, radii, means3d, torch.exp(scales), quats, opacities, view_matrix, ks_flat)

    # Remove batch dimension to match expected output format: (N, 2), (N, 3), (N), (N, 2)
    means2d = means2d.squeeze(0)  # (1, N, 2) -> (N, 2)
    conics = conics.squeeze(0)    # (1, N, 3) -> (N, 3)
    depth = depth.squeeze(0)      # (1, N) -> (N)
    radii = radii.squeeze(0)      # (1, N, 2) -> (N, 2)

    return means2d, conics, depth, radii
