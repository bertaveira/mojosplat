"""The training loop (3DGS / Faster-GS recipe).

One iteration `it` (0-based, Faster-GS's callback order):
    strategy.before_step(it)   densification (see strategy.mojo; densify.mojo for the classic one)
    Morton reorder             if it <= morton_until and it % morton_interval == 0 (model/morton.mojo)
    view  = next training view (random order, reshuffled every epoch)
    image, state = rasterizer.forward(splats, view.camera)       # saves backward state
    grad  = loss.backward(image, view.image)                      # L1 + D-SSIM
    rasterizer.backward(state, grad) -> splats.grads              # all raw parameters
    adam.step(splats)                                             # per-attribute lrs
plus the schedules: the means learning rate decays exponentially (scaled by
the scene extent) and the SH degree grows by one every 1000 iterations.
The backward accumulates densification statistics while the strategy asks for them.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import exp, log
from std.random import random_ui64, seed

from ..cli.args import Arg
from ..core.types import dptr
from ..io.ply import write_ply
from ..model.init import splats_from_points
from ..model.morton import morton_order
from ..model.splats import Attr, Splats
from ..render.blocks.contracts import Pipeline
from ..render.pipelines import Standard3DGS
from ..render.rasterizer import Rasterizer
from .adam import Adam
from .dataset import Dataset
from .densify import ClassicDensification
from .loss import PhotometricLoss
from .strategy import Densification


struct TrainConfig(Copyable, Defaultable, Movable):
    """Settings every method shares; defaults are Faster-GS's (= 3DGS's, opacity lr halved).
    Densification settings belong to the strategy (`Densification.Config`).
    Every field is also a command-line option of `pixi run train`
    (`sh_degree` -> `--sh-degree`); the `Arg` annotations are its help."""

    @__annotation(Arg("Number of training iterations"))
    var iterations: Int

    @__annotation(Arg("Maximum spherical-harmonics degree (0-3)"))
    var sh_degree: Int

    @__annotation(Arg("Raise the active SH degree by one every N iterations"))
    var sh_degree_interval: Int

    @__annotation(Arg("Weight of D-SSIM in the loss (the rest is L1)"))
    var lambda_dssim: Float32

    @__annotation(Arg("Initial learning rate of the means (times the scene extent)"))
    var lr_means_init: Float32

    @__annotation(Arg("Final learning rate of the means (times the scene extent)"))
    var lr_means_final: Float32

    @__annotation(Arg("Iterations over which the means learning rate decays"))
    var lr_means_steps: Int

    @__annotation(Arg("Learning rate of the log scales"))
    var lr_log_scales: Float32

    @__annotation(Arg("Learning rate of the rotations"))
    var lr_rotations: Float32

    @__annotation(Arg("Learning rate of the opacity logits"))
    var lr_opacity: Float32

    @__annotation(Arg("Learning rate of the SH DC coefficients"))
    var lr_sh0: Float32

    @__annotation(Arg("Learning rate of the higher SH coefficients"))
    var lr_sh_rest: Float32

    @__annotation(Arg("Background color (RGB in [0, 1])"))
    var background: SIMD[DType.float32, 4]

    @__annotation(Arg("Initial opacity of every Gaussian"))
    var initial_opacity: Float32

    @__annotation(Arg("Image folder inside the scene"))
    var images: String

    @__annotation(Arg("Near plane of the cameras"))
    var near: Float32

    @__annotation(Arg("Far plane of the cameras"))
    var far: Float32

    @__annotation(Arg("Random seed"))
    var seed: Int

    @__annotation(Arg("Read back the loss every N iterations (syncs the GPU)"))
    var log_interval: Int

    @__annotation(Arg("Z-order (Morton) reordering every N iterations (0 = off)"))
    var morton_interval: Int

    @__annotation(Arg("Last Morton reordering iteration (before the final one)"))
    var morton_until: Int

    def __init__(out self):
        self.iterations = 30_000
        self.sh_degree = 3
        self.sh_degree_interval = 1000
        self.lambda_dssim = 0.2
        self.lr_means_init = 0.00016
        self.lr_means_final = 0.0000016
        self.lr_means_steps = 30_000
        self.lr_log_scales = 0.005
        self.lr_rotations = 0.001
        self.lr_opacity = 0.025
        self.lr_sh0 = 0.0025
        self.lr_sh_rest = 0.0025 / 20.0
        self.background = SIMD[DType.float32, 4](0.0)
        self.initial_opacity = 0.1
        self.images = "images_4"
        self.near = 0.2
        self.far = 10000.0
        self.seed = 0
        self.log_interval = 100
        self.morton_interval = 5000
        self.morton_until = 15_000


def exponential_lr(init: Float32, final: Float32, step: Int, max_steps: Int) -> Float32:
    """3DGS's get_expon_lr_func (without delay): log-linear interpolation."""
    var t = min(max(Float64(step) / Float64(max(max_steps, 1)), 0.0), 1.0)
    return Float32(exp(log(Float64(init)) * (1.0 - t) + log(Float64(final)) * t))


struct Trainer[Pipe: Pipeline = Standard3DGS, Strategy: Densification = ClassicDensification](Movable):
    var ctx: DeviceContext
    var config: TrainConfig
    var dataset: Dataset
    var splats: Splats
    var rasterizer: Rasterizer[]
    var adam: Adam
    var loss: PhotometricLoss
    var strategy: Self.Strategy
    var image: DeviceBuffer[DType.float32]
    var grad_image: DeviceBuffer[DType.float32]
    var order: List[Int]
    var order_pos: Int
    var iteration: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        var dataset: Dataset,
        config: TrainConfig,
        strategy: Self.Strategy.Config = Self.Strategy.Config(),
    ) raises:
        self.ctx = ctx
        self.config = config.copy()
        seed(config.seed)
        self.splats = splats_from_points(
            ctx, dataset.points.xyz, dataset.points.rgb, config.sh_degree, initial_opacity=config.initial_opacity
        )
        var specs = Self.Pipe.Appear.columns()
        for i in range(len(specs)):
            ref s = specs[i]
            self.splats.add_column(
                s.name, s.width, trainable=s.trainable, values=Self.Pipe.Appear.init_column(i, self.splats.count)
            )
        var response_specs = Self.Pipe.Resp.columns()
        for s in response_specs:
            specs.append(s.copy())
            self.splats.add_column(
                s.name,
                s.width,
                trainable=s.trainable,
                values=List[Float32](length=self.splats.count * s.width, fill=s.init_value),
            )
        self.splats.enable_training(sh_rest_grad=False)  # Adam recomputes it (write_all=False)
        var lr = SIMD[DType.float32, 8](0.0)
        lr[Attr.MEANS] = config.lr_means_init * dataset.scene_extent
        lr[Attr.LOG_SCALES] = config.lr_log_scales
        lr[Attr.ROTATIONS] = config.lr_rotations
        lr[Attr.OPACITY_LOGITS] = config.lr_opacity
        lr[Attr.SH0] = config.lr_sh0
        lr[Attr.SH_REST] = config.lr_sh_rest
        for i in range(len(specs)):
            lr[Attr.COUNT + i] = specs[i].lr
        self.adam = Adam(lr)
        self.rasterizer = Rasterizer(ctx)
        self.loss = PhotometricLoss(ctx, config.lambda_dssim)
        self.strategy = Self.Strategy(ctx, strategy)
        var max_px = 1
        for v in dataset.train:
            max_px = max(max_px, v.camera.width * v.camera.height)
        for v in dataset.test:
            max_px = max(max_px, v.camera.width * v.camera.height)
        self.image = ctx.enqueue_create_buffer[DType.float32](max_px * 3)
        self.grad_image = ctx.enqueue_create_buffer[DType.float32](max_px * 3)
        self.dataset = dataset^
        self.order = List[Int]()
        self.order_pos = 0
        self.iteration = 0

    def active_sh_degree(self) -> Int:
        """SH degree used by the next iteration."""
        return min(self.config.sh_degree, self.iteration // self.config.sh_degree_interval)

    def _next_view(mut self) -> Int:
        if self.order_pos >= len(self.order):
            var n = len(self.dataset.train)
            self.order = List[Int](capacity=n)
            for i in range(n):
                self.order.append(i)
            for i in range(n - 1, 0, -1):  # Fisher-Yates
                var j = Int(random_ui64(0, UInt64(i)))
                self.order.swap_elements(i, j)
            self.order_pos = 0
        var v = self.order[self.order_pos]
        self.order_pos += 1
        return v

    def _maybe_morton(mut self, it: Int) raises:
        var c = self.config.copy()
        if c.morton_interval > 0 and it <= c.morton_until and it % c.morton_interval == 0:
            morton_order(self.splats)

    def step(mut self) raises -> Float64:
        """One training iteration. Returns the loss on logging iterations
        (every config.log_interval-th), else -1 (no extra sync)."""
        var it = self.iteration
        self.strategy.before_step(it, self.splats, self.dataset.scene_extent)
        self._maybe_morton(it)
        var vi = self._next_view()
        ref view = self.dataset.train[vi]
        var cam = view.camera
        var image = dptr(self.image)
        var grad = dptr(self.grad_image)
        var st = self.rasterizer.forward[Self.Pipe](
            self.splats.view(), cam, self.config.background, image, self.active_sh_degree()
        )
        var log_now = self.config.log_interval > 0 and (it + 1) % self.config.log_interval == 0
        var value = self.loss.backward(image, dptr(view.image), grad, cam.width, cam.height, compute_loss=log_now)
        var stats = self.strategy.collects_stats(it)
        self.rasterizer.backward[Self.Pipe](
            self.splats.view(),
            st,
            image,
            grad,
            self.splats.grad_view(),
            self.splats.densification_ptr(),
            stats,
            write_all=False,
        )
        self.adam.lr[Attr.MEANS] = exponential_lr(
            self.config.lr_means_init * self.dataset.scene_extent,
            self.config.lr_means_final * self.dataset.scene_extent,
            it + 1,
            self.config.lr_means_steps,
        )
        self.adam.step(self.splats, self.rasterizer.sh_rest_grad[Self.Pipe](st))
        self.iteration += 1
        return value if log_now else -1.0

    def finish(mut self) raises -> Int:
        """After training: the strategy's last step (the classic one prunes Gaussians
        that can no longer affect any pixel), then Z-order them (faster rendering
        later). Returns the number removed."""
        var removed = self.strategy.finish(self.splats)
        morton_order(self.splats)
        return removed

    def evaluate(mut self, test: Bool = True) raises -> Float64:
        """Mean PSNR over the test (or training) views at the current state."""
        var n = len(self.dataset.test) if test else len(self.dataset.train)
        if n == 0:
            return 0.0
        var total: Float64 = 0.0
        for i in range(n):
            var cam = self.dataset.test[i].camera if test else self.dataset.train[i].camera
            var target = dptr(self.dataset.test[i].image) if test else dptr(self.dataset.train[i].image)
            self.rasterizer.render[Self.Pipe](
                self.splats.view(), cam, self.config.background, dptr(self.image), self.config.sh_degree
            )
            total += self.loss.psnr(dptr(self.image), target, cam.width, cam.height)
        return total / Float64(n)

    def save_ply(self, path: String) raises:
        write_ply(path, self.splats.to_host())
