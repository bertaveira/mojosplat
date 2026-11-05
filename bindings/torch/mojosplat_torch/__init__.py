"""PyTorch bindings for MojoSplat.

The rendering kernels are written in Mojo (see the repository's `mojosplat/`
package); this package wraps them for torch tensors and keeps reference
`torch` and `gsplat` backends for comparison.
"""

from .utils import Camera
from .projection import project_gaussians
from .binning import bin_gaussians_to_tiles
from .rasterization import rasterize_gaussians
from .render import render_gaussians
from .rasterizer import render_splats, rgb_to_sh0

__all__ = [
    "Camera",
    "project_gaussians",
    "bin_gaussians_to_tiles",
    "rasterize_gaussians",
    "render_gaussians",
    "render_splats",
    "rgb_to_sh0",
]
