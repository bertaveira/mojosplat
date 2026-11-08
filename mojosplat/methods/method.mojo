"""A method: which renderer it uses and how it trains, for any scene."""

from ..render.blocks.contracts import Pipeline
from ..train.trainer import TrainConfig


trait Method:
    comptime Pipe: Pipeline

    @staticmethod
    def name() -> String:
        """The name given to `--method`."""
        ...

    @staticmethod
    def config(scene: String) -> TrainConfig:
        """Settings for a scene (its directory name, e.g. "garden")."""
        ...
