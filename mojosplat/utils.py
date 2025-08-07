import torch
from dataclasses import dataclass

# --- Camera Class ---
@dataclass
class Camera:
    R: torch.Tensor # Rotation (3, 3) world-to-camera
    T: torch.Tensor # Translation (3,) world-to-camera
    H: int
    W: int
    fx: float
    fy: float
    cx: float
    cy: float
    near: float = 0.1
    far: float = 100.0
    # Precomputed matrices
    view_matrix: torch.Tensor = None
    Ks: torch.Tensor = None

    def __post_init__(self):
        """Builds the 4x4 view matrix."""
        if self.view_matrix is None:
            # Build world-to-camera view matrix as expected by GSplat
            # GSplat expects: viewmats: The world-to-cam transformation of the cameras. [..., C, 4, 4]
            # Standard world-to-camera matrix: [R | T; 0 0 0 1] where R,T are world-to-camera
            self.view_matrix = torch.eye(4, device=self.R.device, dtype=self.R.dtype)
            self.view_matrix[:3, :3] = self.R    # World-to-camera rotation
            self.view_matrix[:3, 3] = self.T     # World-to-camera translation
        if self.Ks is None:
            self.Ks = torch.tensor([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], device=self.R.device, dtype=self.R.dtype)
