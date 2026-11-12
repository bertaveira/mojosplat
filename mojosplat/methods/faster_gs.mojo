"""Faster-GS (Hahlbohm et al., CVPR 2026): the default settings, with classic densification."""

from ..cli.args import Arg
from ..render.pipelines import Standard3DGS
from ..train.densify import ClassicDensification, DensifyConfig
from ..train.trainer import TrainConfig
from .method import Method, MethodConfig


struct FasterGSConfig(MethodConfig):
    comptime Strategy = ClassicDensification

    @__annotation(Arg("Training"))
    var train: TrainConfig

    @__annotation(Arg("Densification"))
    var densify: DensifyConfig

    def __init__(out self):
        self.train = TrainConfig()
        self.densify = DensifyConfig()

    def training(self) -> TrainConfig:
        return self.train.copy()

    def densification(self) -> DensifyConfig:
        return self.densify.copy()


struct FasterGS(Method):
    comptime Pipe = Standard3DGS
    comptime Config = FasterGSConfig

    @staticmethod
    def name() -> String:
        return "faster-gs"

    @staticmethod
    def config(scene: String) -> FasterGSConfig:
        return FasterGSConfig()
