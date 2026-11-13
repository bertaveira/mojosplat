"""Named renderers: which blocks the kernels are built from."""

from .blocks.appearance.sh import SphericalHarmonics
from .blocks.appearance.spherical_beta import SphericalBetaColor
from .blocks.contracts import Pipeline
from .blocks.placement.pinhole_ewa import PinholeEWA
from .blocks.response.beta import BetaFootprint
from .blocks.response.gaussian2d import Gaussian2D


struct Standard3DGS(Pipeline):
    """Classic 3DGS: linearized pinhole projection, 2D Gaussians, SH color."""

    comptime ID = 0
    comptime Place = PinholeEWA[DType.float32]
    comptime Resp = Gaussian2D
    comptime Appear = SphericalHarmonics


struct BetaSplattingSH(Pipeline):
    """The Beta-kernel footprint with SH color: an ablation of Beta Splatting, used by the tests."""

    comptime ID = 1
    comptime Place = PinholeEWA[DType.float32]
    comptime Resp = BetaFootprint
    comptime Appear = SphericalHarmonics


struct BetaSplatting(Pipeline):
    """Beta Splatting: the Beta-kernel footprint with Spherical Beta color (`--method beta-splatting`)."""

    comptime ID = 2
    comptime Place = PinholeEWA[DType.float32]
    comptime Resp = BetaFootprint
    comptime Appear = SphericalBetaColor[2]
