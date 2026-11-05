"""MojoSplat: 3D Gaussian Splatting (forward and backward) in Mojo.

Packages:
    core    plain types: pointers, Camera, SplatParams view, SH constants
    gpu     generic GPU building blocks: scan, radix sort, fills, arenas
    render  the Faster-GS style Rasterizer (forward + backward)
    model   owning storage of the Gaussians
    io      raw arrays, .splat / .ply scenes, COLMAP models, JPEG decoding, image output
    train   datasets (COLMAP scenes); loss, optimizer and trainer to come
"""

from .core.camera import Camera
from .core.splats import SplatParams
from .io.arrays import download, read_array, upload, write_array
from .io.colmap import read_colmap_model
from .io.image import write_png, write_ppm
from .io.jpeg import JpegDecoder, read_jpeg
from .io.ply import HostSplats, read_ply, write_ply
from .io.splat import SplatScene, load_splat
from .model.splats import SplatBuffers
from .render.rasterizer import ForwardState, Rasterizer
from .train.dataset import Dataset, View, load_colmap_dataset
