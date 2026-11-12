"""Beta Splatting (Liu et al., 2025): the settings of the official benchmark.py.

Scales come from the prebuilt image folders, where benchmark.py resizes the
full-resolution images with `-r`.
"""

from ..cli.args import Arg
from ..render.pipelines import BetaSplatting
from ..train.relocate import RelocateConfig, Relocation
from ..train.trainer import TrainConfig
from .method import Method, MethodConfig


def _cap_max(scene: String) -> Int:
    if scene == "bicycle":
        return 6_000_000
    if scene == "flowers":
        return 3_000_000
    if scene == "garden":
        return 5_000_000
    if scene == "stump":
        return 4_500_000
    if scene == "treehill" or scene == "drjohnson":
        return 3_500_000
    if scene == "truck" or scene == "playroom":
        return 2_500_000
    if scene == "room" or scene == "counter" or scene == "kitchen" or scene == "bonsai":
        return 1_500_000
    return 1_000_000  # also the official default for any other scene


def _images(scene: String) -> String:
    if scene == "room" or scene == "counter" or scene == "kitchen" or scene == "bonsai":
        return "images_2"
    if scene == "truck" or scene == "train" or scene == "drjohnson" or scene == "playroom":
        return "images"
    return "images_4"


struct BetaSplattingConfig(MethodConfig):
    comptime Strategy = Relocation

    @__annotation(Arg("Training"))
    var train: TrainConfig

    @__annotation(Arg("Relocation"))
    var relocate: RelocateConfig

    def __init__(out self):
        self.train = TrainConfig()
        self.train.initial_opacity = 0.5
        self.train.lr_opacity = 0.05
        self.train.sh_degree = 0
        # The official renderer's planes (gsplat's defaults).
        self.train.near = 0.01
        self.train.far = 1e10
        self.relocate = RelocateConfig()

    def training(self) -> TrainConfig:
        return self.train.copy()

    def densification(self) -> RelocateConfig:
        return self.relocate.copy()


struct BetaSplattingMethod(Method):
    comptime Pipe = BetaSplatting
    comptime Config = BetaSplattingConfig

    @staticmethod
    def name() -> String:
        return "beta-splatting"

    @staticmethod
    def config(scene: String) -> BetaSplattingConfig:
        var c = BetaSplattingConfig()
        c.relocate.cap_max = _cap_max(scene)
        c.train.images = _images(scene)
        return c^
