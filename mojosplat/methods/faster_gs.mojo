"""Faster-GS: the settings `TrainConfig` defaults to (Hahlbohm et al., CVPR 2026)."""

from ..render.pipelines import Standard3DGS
from ..train.trainer import TrainConfig
from .method import Method


struct FasterGS(Method):
    comptime Pipe = Standard3DGS

    @staticmethod
    def name() -> String:
        return "faster-gs"

    @staticmethod
    def config(scene: String) -> TrainConfig:
        return TrainConfig()
