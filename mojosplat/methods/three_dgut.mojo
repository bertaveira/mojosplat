"""3DGUT (Wu et al., "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian
Splatting", CVPR 2025): NVIDIA's 3dgrut settings for its unsorted pinhole renderer.

Settings from configs/paper/3dgut/unsorted_colmap.yaml and everything it
includes (base_gs.yaml, render/3dgut.yaml, strategy/gs.yaml,
initialization/colmap.yaml, dataset/colmap.yaml); the image folders from
scripts/benchmark/mipnerf360.sh. Rendering is render/blocks/placement/pinhole_ut.mojo
and render/blocks/response/ray3d.mojo; densification is `GutDensification`.
"""

from ..cli.args import Arg
from ..render.pipelines import Gut3D
from ..train.densify import GutDensification, GutDensifyConfig
from ..train.trainer import InitConfig, ScheduleConfig, TrainConfig
from .method import Method, MethodConfig


struct ThreeDGUTConfig(MethodConfig):
    comptime Strategy = GutDensification

    @__annotation(Arg("Training"))
    var train: TrainConfig

    @__annotation(Arg("Initialization"))
    var init: InitConfig

    @__annotation(Arg("Densification"))
    var densify: GutDensifyConfig

    @__annotation(Arg("Schedule"))
    var schedule: ScheduleConfig

    def __init__(out self):
        self.train = TrainConfig()
        self.train.lr_opacity = 0.05
        self.train.far = 1e10  # no far plane
        self.train.ssim_valid = True
        self.train.integer_factor = True
        self.train.median_extent = True
        self.init = InitConfig()
        self.init.observer_scale = 0.01
        self.init.random_rotations = True
        self.densify = GutDensifyConfig()
        self.schedule = ScheduleConfig()
        self.schedule.after_step = True

    def training(self) -> TrainConfig:
        return self.train.copy()

    def densification(self) -> GutDensifyConfig:
        return self.densify.copy()

    def initialization(self) -> InitConfig:
        return self.init.copy()

    def schedules(self) -> ScheduleConfig:
        return self.schedule.copy()


struct ThreeDGUT(Method):
    comptime Pipe = Gut3D
    comptime Config = ThreeDGUTConfig

    @staticmethod
    def name() -> String:
        return "3dgut"

    @staticmethod
    def config(scene: String) -> ThreeDGUTConfig:
        var c = ThreeDGUTConfig()
        if scene == "bonsai" or scene == "counter" or scene == "kitchen" or scene == "room":
            c.train.images = "images_2"
        return c^
