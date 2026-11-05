"""PyTorch bindings for MojoSplat.

The rendering kernels are written in Mojo (see the repository's `mojosplat/`
package); `render_splats` exposes the rasterizer to torch tensors, with
autograd. `render_gsplat` is a gsplat reference with the same interface.
"""

from .io import load_splat
from .rasterizer import render_splats, rgb_to_sh0
from .reference import render_gsplat
from .utils import Camera

__all__ = ["Camera", "load_splat", "render_gsplat", "render_splats", "rgb_to_sh0"]
