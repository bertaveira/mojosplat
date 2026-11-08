"""Named renderers: which blocks the kernels are built from."""

from .blocks.appearance.sh import SphericalHarmonics
from .blocks.contracts import Pipeline
from .blocks.placement.pinhole_ewa import PinholeEWA
from .blocks.response.gaussian2d import Gaussian2D


struct Standard3DGS(Pipeline):
    """Classic 3DGS: linearized pinhole projection, 2D Gaussians, SH color."""

    comptime ID = 0
    comptime Place = PinholeEWA[DType.float32]
    comptime Resp = Gaussian2D
    comptime Appear = SphericalHarmonics
