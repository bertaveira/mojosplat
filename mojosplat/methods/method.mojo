"""A method: which renderer it uses, how it densifies, and its settings for any scene."""

from ..render.blocks.contracts import Pipeline
from ..train.strategy import Densification
from ..train.trainer import InitConfig, TrainConfig


trait MethodConfig(Copyable, Defaultable, Deinitable, Movable):
    """A method's settings. Every field is a command-line option of `pixi run train`
    (nested structs are option groups), and `run.json` records them all.

    Usually a `TrainConfig` field, its strategy's config, and any settings of the
    method's own (see `FasterGSConfig`)."""

    comptime Strategy: Densification

    def training(self) -> TrainConfig:
        """The settings every method shares."""
        ...

    def densification(self) -> Self.Strategy.Config:
        """The densification strategy's settings."""
        ...

    def initialization(self) -> InitConfig:
        """Where training starts: 3DGS's SfM start unless the method defines it."""
        return InitConfig()


trait Method:
    comptime Pipe: Pipeline
    comptime Config: MethodConfig

    @staticmethod
    def name() -> String:
        """The name given to `--method`."""
        ...

    @staticmethod
    def config(scene: String) -> Self.Config:
        """Settings for a scene (its directory name, e.g. "garden")."""
        ...
