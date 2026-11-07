"""The training loop (3DGS / Faster-GS recipe).

One iteration `it` (0-based, Faster-GS's callback order):
    densify           if 600 <= it <= 14900 and it % 100 == 0 (see densify.mojo)
    Morton reorder    if it <= 15000 and it % 5000 == 0 (model/morton.mojo)
    reset opacities   if 3000 <= it <= 14900 and it % 3000 == 0
    view  = next training view (random order, reshuffled every epoch)
    image, state = rasterizer.forward(splats, view.camera)       # saves backward state
    grad  = loss.backward(image, view.image)                      # L1 + D-SSIM
    rasterizer.backward(state, grad) -> splats.grads              # all raw parameters
    adam.step(splats)                                             # per-attribute lrs
plus the schedules: the means learning rate decays exponentially (scaled by
the scene extent) and the SH degree grows by one every 1000 iterations.
The backward accumulates densification statistics until densification ends.
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
from ..render.rasterizer import Rasterizer
from .adam import Adam
from .dataset import Dataset
from .densify import DensifyStats, Densifier
from .loss import PhotometricLoss


struct TrainConfig(Copyable, Defaultable, Movable):
    """Hyperparameters; defaults are Faster-GS's (= 3DGS's, opacity lr halved).
    Every field is also a command-line option of `tools/train.mojo`
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

    @__annotation(Arg("Random seed"))
    var seed: Int

    @__annotation(Arg("Read back the loss every N iterations (syncs the GPU)"))
    var log_interval: Int

    @__annotation(Arg("Clone, split and prune Gaussians (classic 3DGS densification)"))
    var densify: Bool

    @__annotation(Arg("First densification iteration"))
    var densify_from: Int

    @__annotation(Arg("Last densification iteration"))
    var densify_until: Int

    @__annotation(Arg("Densify every N iterations"))
    var densify_interval: Int

    @__annotation(Arg("Reset opacities every N iterations (during densification)"))
    var opacity_reset_interval: Int

    @__annotation(Arg("Densify Gaussians whose mean screen-space gradient exceeds this"))
    var grad_threshold: Float32

    @__annotation(Arg("Clone below / split above this size (fraction of the scene extent)"))
    var percent_dense: Float32

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
        self.seed = 0
        self.log_interval = 100
        self.densify = True
        self.densify_from = 600
        self.densify_until = 14_900
        self.densify_interval = 100
        self.opacity_reset_interval = 3000
        self.grad_threshold = 0.0002
        self.percent_dense = 0.01
        self.morton_interval = 5000
        self.morton_until = 15_000


def exponential_lr(init: Float32, final: Float32, step: Int, max_steps: Int) -> Float32:
    """3DGS's get_expon_lr_func (without delay): log-linear interpolation."""
    var t = min(max(Float64(step) / Float64(max(max_steps, 1)), 0.0), 1.0)
    return Float32(exp(log(Float64(init)) * (1.0 - t) + log(Float64(final)) * t))


struct Trainer(Movable):
    var ctx: DeviceContext
    var config: TrainConfig
    var dataset: Dataset
    var splats: Splats
    var rasterizer: Rasterizer[]
    var adam: Adam
    var loss: PhotometricLoss
    var densifier: Densifier
    var last_densify: DensifyStats
    var image: DeviceBuffer[DType.float32]
    var grad_image: DeviceBuffer[DType.float32]
    var order: List[Int]
    var order_pos: Int
    var iteration: Int

    def __init__(out self, ctx: DeviceContext, var dataset: Dataset, config: TrainConfig) raises:
        self.ctx = ctx
        self.config = config.copy()
        self.splats = splats_from_points(ctx, dataset.points.xyz, dataset.points.rgb, config.sh_degree)
        self.splats.enable_training(sh_rest_grad=False)  # Adam recomputes it (write_sh_rest=False)
        var lr = SIMD[DType.float32, 8](0.0)
        lr[Attr.MEANS] = config.lr_means_init * dataset.scene_extent
        lr[Attr.LOG_SCALES] = config.lr_log_scales
        lr[Attr.ROTATIONS] = config.lr_rotations
        lr[Attr.OPACITY_LOGITS] = config.lr_opacity
        lr[Attr.SH0] = config.lr_sh0
        lr[Attr.SH_REST] = config.lr_sh_rest
        self.adam = Adam(lr)
        self.rasterizer = Rasterizer(ctx)
        self.loss = PhotometricLoss(ctx, config.lambda_dssim)
        self.densifier = Densifier(ctx)
        self.densifier.grad_threshold = config.grad_threshold
        self.densifier.percent_dense = config.percent_dense
        self.last_densify = DensifyStats()
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
        seed(config.seed)

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

    def _densification_callbacks(mut self, it: Int) raises:
        var c = self.config.copy()
        if not c.densify or it < c.densify_from or it > c.densify_until:
            self._maybe_morton(it)
            return
        if (it - c.densify_from) % c.densify_interval == 0:
            self.last_densify = self.densifier.densify(
                self.splats, self.dataset.scene_extent, it > c.opacity_reset_interval
            )
            if it < c.densify_until:
                self.splats.reset_densification()
        self._maybe_morton(it)
        if it >= c.opacity_reset_interval and it % c.opacity_reset_interval == 0:
            self.densifier.reset_opacities(self.splats)

    def _maybe_morton(mut self, it: Int) raises:
        var c = self.config.copy()
        if c.morton_interval > 0 and it <= c.morton_until and it % c.morton_interval == 0:
            morton_order(self.splats)

    def step(mut self) raises -> Float64:
        """One training iteration. Returns the loss on logging iterations
        (every config.log_interval-th), else -1 (no extra sync)."""
        var it = self.iteration
        self._densification_callbacks(it)
        var vi = self._next_view()
        ref view = self.dataset.train[vi]
        var cam = view.camera
        var image = dptr(self.image)
        var grad = dptr(self.grad_image)
        var st = self.rasterizer.forward(
            self.splats.view(), cam, self.config.background, image, self.active_sh_degree()
        )
        var log_now = self.config.log_interval > 0 and (it + 1) % self.config.log_interval == 0
        var value = self.loss.backward(image, dptr(view.image), grad, cam.width, cam.height, compute_loss=log_now)
        var stats = self.config.densify and it < self.config.densify_until
        self.rasterizer.backward(
            self.splats.view(),
            st,
            image,
            grad,
            self.splats.grad_view(),
            self.splats.densification_ptr(),
            stats,
            write_sh_rest=False,
        )
        self.adam.lr[Attr.MEANS] = exponential_lr(
            self.config.lr_means_init * self.dataset.scene_extent,
            self.config.lr_means_final * self.dataset.scene_extent,
            it + 1,
            self.config.lr_means_steps,
        )
        self.adam.step(self.splats, self.rasterizer.sh_rest_grad(st))
        self.iteration += 1
        return value if log_now else -1.0

    def finish(mut self) raises -> Int:
        """After training: prune Gaussians that can no longer affect any pixel
        (opacity < 1/255) or are degenerate, then Z-order them (faster
        rendering later). Returns the number removed."""
        var removed = self.densifier.prune_final(self.splats)
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
            self.rasterizer.render(
                self.splats.view(), cam, self.config.background, dptr(self.image), self.config.sh_degree
            )
            total += self.loss.psnr(dptr(self.image), target, cam.width, cam.height)
        return total / Float64(n)

    def save_ply(self, path: String) raises:
        write_ply(path, self.splats.to_host())
