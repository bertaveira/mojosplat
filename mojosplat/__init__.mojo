"""MojoSplat: 3D Gaussian Splatting (forward and backward) in Mojo.

Packages:
    core    plain types: pointers, Camera, SplatParams view, SH constants
    gpu     generic GPU building blocks: scan, radix sort, fills, arenas
    render  the Faster-GS style Rasterizer (forward + backward)
    model   owning storage of the Gaussians
    io      raw arrays, .splat scenes, image output
"""

from .core.camera import Camera
from .core.splats import SplatParams
from .io.arrays import download, read_array, upload, write_array
from .io.image import write_png, write_ppm
from .io.splat import SplatScene, load_splat
from .model.splats import SplatBuffers
from .render.rasterizer import ForwardState, Rasterizer
