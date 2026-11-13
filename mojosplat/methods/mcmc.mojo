"""3DGS-MCMC (Kheradmand et al., "3D Gaussian Splatting as Markov Chain Monte Carlo",
NeurIPS 2024): the official per-scene settings.

3DGS rendering with MCMC relocation (train/relocate.mojo). Training starts from
random points, as the official code does by default; `--no-random-init` starts
from the SfM points instead (the paper reports both). Scales come from the
prebuilt image folders.
"""

from ..cli.args import Arg
from ..render.pipelines import Standard3DGS
from ..train.relocate import MCMCRelocation, RelocateConfig
from ..train.trainer import InitConfig, TrainConfig
from .method import Method, MethodConfig


def _cap_max(scene: String) -> Int:
    if scene == "bicycle":
        return 5_900_000
    if scene == "garden":
        return 5_200_000
    if scene == "stump":
        return 4_750_000
    if scene == "room":
        return 1_500_000
    if scene == "counter":
        return 1_200_000
    if scene == "kitchen":
        return 1_800_000
    if scene == "bonsai":
        return 1_300_000
    if scene == "truck":
        return 2_600_000
    if scene == "train":
        return 1_100_000
    if scene == "drjohnson":
        return 3_400_000
    if scene == "playroom":
        return 2_500_000
    return 1_000_000  # the official code has no setting for other scenes; pass --cap-max


def _images(scene: String) -> String:
    if scene == "room" or scene == "counter" or scene == "kitchen" or scene == "bonsai":
        return "images_2"
    if scene == "truck" or scene == "train" or scene == "drjohnson" or scene == "playroom":
        return "images"
    return "images_4"


struct MCMCConfig(MethodConfig):
    comptime Strategy = MCMCRelocation

    @__annotation(Arg("Training"))
    var train: TrainConfig

    @__annotation(Arg("Initialization"))
    var init: InitConfig

    @__annotation(Arg("Relocation"))
    var relocate: RelocateConfig

    def __init__(out self):
        self.train = TrainConfig()
        self.train.initial_opacity = 0.5
        self.train.lr_opacity = 0.05
        self.init = InitConfig()
        self.init.random_init = True
        self.init.initial_scale = 0.1
        self.relocate = RelocateConfig()
        self.relocate.noise_lr = 5e5

    def training(self) -> TrainConfig:
        return self.train.copy()

    def densification(self) -> RelocateConfig:
        return self.relocate.copy()

    def initialization(self) -> InitConfig:
        return self.init.copy()


struct MCMC(Method):
    comptime Pipe = Standard3DGS
    comptime Config = MCMCConfig

    @staticmethod
    def name() -> String:
        return "3dgs-mcmc"

    @staticmethod
    def config(scene: String) -> MCMCConfig:
        var c = MCMCConfig()
        c.relocate.cap_max = _cap_max(scene)
        c.train.images = _images(scene)
        if scene == "drjohnson" or scene == "playroom":
            c.relocate.opacity_reg = 0.001
        return c^
