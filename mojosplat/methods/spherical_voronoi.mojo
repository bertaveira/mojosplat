"""Spherical Voronoi (Di Sario et al., CVPR 2026): the settings of the official `radiance/` code.

Beta Splatting's footprint, relocation, Gaussian caps and image folders, with
Spherical Voronoi color (render/blocks/appearance/spherical_voronoi.mojo) and
the official configs: 20,000 iterations outdoors and 30,000 indoors, relocation
after the optimizer step, and only the colors and temperatures trained over the
last 3,000 iterations.
"""

from ..cli.args import Arg
from ..render.pipelines import SphericalVoronoiBeta
from ..train.relocate import RelocateConfig, Relocation
from ..train.trainer import ScheduleConfig, TrainConfig
from .beta_splatting import BetaSplattingMethod
from .method import Method, MethodConfig


def _is_outdoor(scene: String) -> Bool:
    """The scenes of the official `config/outdoor.json`."""
    return scene == "bicycle" or scene == "garden" or scene == "stump" or scene == "flowers" or scene == "treehill"


struct SphericalVoronoiConfig(MethodConfig):
    comptime Strategy = Relocation[step_first=True]

    @__annotation(Arg("Training"))
    var train: TrainConfig

    @__annotation(Arg("Relocation"))
    var relocate: RelocateConfig

    @__annotation(Arg("Schedule"))
    var schedule: ScheduleConfig

    def __init__(out self):
        var beta = BetaSplattingMethod.Config()
        self.train = beta.train.copy()
        self.train.iterations = 30_000
        self.relocate = beta.relocate.copy()
        self.schedule = ScheduleConfig()
        self.schedule.frozen_last = 3_000

    def training(self) -> TrainConfig:
        return self.train.copy()

    def densification(self) -> RelocateConfig:
        return self.relocate.copy()

    def schedules(self) -> ScheduleConfig:
        return self.schedule.copy()


struct SphericalVoronoiMethod(Method):
    comptime Pipe = SphericalVoronoiBeta
    comptime Config = SphericalVoronoiConfig

    @staticmethod
    def name() -> String:
        return "spherical-voronoi"

    @staticmethod
    def config(scene: String) -> SphericalVoronoiConfig:
        var beta = BetaSplattingMethod.config(scene)  # Gaussian caps and image folders
        var c = SphericalVoronoiConfig()
        c.relocate.cap_max = beta.relocate.cap_max
        c.train.images = beta.train.images
        if _is_outdoor(scene):
            c.train.iterations = 20_000
            c.relocate.relocate_until = 15_000
        return c^
