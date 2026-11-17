"""Named renderers: which blocks the kernels are built from."""

from max.gpu import WARP_SIZE

from .blocks.appearance.sh import SphericalHarmonics
from .blocks.appearance.spherical_beta import SphericalBetaColor
from .blocks.appearance.spherical_voronoi import SphericalVoronoi
from .blocks.contracts import Pipeline
from .blocks.placement.pinhole_ewa import PinholeEWA
from .blocks.placement.pinhole_ut import PinholeUT
from .blocks.response.beta import BetaFootprint
from .blocks.response.gaussian2d import Gaussian2D
from .blocks.response.ray3d import GaussianRay3D

comptime DEFAULT_TILE_H = 12 if WARP_SIZE == 32 else 16  # tiles are 16 wide; `Rasterizer`'s default


struct Standard3DGS(Pipeline):
    """Classic 3DGS: linearized pinhole projection, 2D Gaussians, SH color."""

    comptime ID = 0
    comptime TILE_H = DEFAULT_TILE_H
    comptime Place = PinholeEWA[DType.float32]
    comptime Resp = Gaussian2D
    comptime Appear = SphericalHarmonics


struct BetaSplattingSH(Pipeline):
    """The Beta-kernel footprint with SH color: an ablation of Beta Splatting, used by the tests."""

    comptime ID = 1
    comptime TILE_H = DEFAULT_TILE_H
    comptime Place = PinholeEWA[DType.float32]
    comptime Resp = BetaFootprint
    comptime Appear = SphericalHarmonics


struct BetaSplatting(Pipeline):
    """Beta Splatting: the Beta-kernel footprint with Spherical Beta color (`--method beta-splatting`)."""

    comptime ID = 2
    comptime TILE_H = DEFAULT_TILE_H
    comptime Place = PinholeEWA[DType.float32]
    comptime Resp = BetaFootprint
    comptime Appear = SphericalBetaColor[2]


struct SphericalVoronoiBeta(Pipeline):
    """Spherical Voronoi: the Beta-kernel footprint with Spherical Voronoi color (`--method spherical-voronoi`)."""

    comptime ID = 3
    comptime TILE_H = DEFAULT_TILE_H
    comptime Place = PinholeEWA[DType.float32]
    comptime Resp = BetaFootprint
    comptime Appear = SphericalVoronoi[8]


struct Gut3D(Pipeline):
    """3DGUT: unscented-transform tile bounds, the 3D Gaussian response along each pixel's ray, SH color
    (`--method 3dgut`)."""

    comptime ID = 4
    comptime TILE_H = 16  # NVIDIA's 16x16 tiles: which Gaussians a pixel sees depends on them
    comptime Place = PinholeUT[DType.float32]
    comptime Resp = GaussianRay3D
    comptime Appear = SphericalHarmonics
