"""The training loop (3DGS / Faster-GS recipe, no densification yet).

One iteration:
    view  = next training view (random order, reshuffled every epoch)
    image, state = rasterizer.forward(splats, view.camera)       # saves backward state
    grad  = loss.backward(image, view.image)                      # L1 + D-SSIM
    rasterizer.backward(state, grad) -> splats.grads              # all raw parameters
    adam.step(splats)                                             # per-attribute lrs
plus the schedules: the means learning rate decays exponentially (scaled by
the scene extent) and the SH degree grows by one every 1000 iterations.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import exp, log
from std.random import random_ui64, seed

from ..core.types import dptr
from ..io.ply import write_ply
from ..model.init import splats_from_points
from ..model.splats import Attr, Splats
from ..render.rasterizer import Rasterizer
from .adam import Adam
from .dataset import Dataset
from .loss import PhotometricLoss


struct TrainConfig(Copyable, Movable):
    """Hyperparameters; defaults are Faster-GS's (= 3DGS's, opacity lr halved)."""

    var iterations: Int
    var sh_degree: Int
    var sh_degree_interval: Int
    var lambda_dssim: Float32
    var lr_means_init: Float32  # multiplied by the scene extent
    var lr_means_final: Float32
    var lr_means_steps: Int
    var lr_log_scales: Float32
    var lr_rotations: Float32
    var lr_opacity: Float32
    var lr_sh0: Float32
    var lr_sh_rest: Float32
    var background: SIMD[DType.float32, 4]
    var seed: Int
    var log_interval: Int

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
    var image: DeviceBuffer[DType.float32]
    var grad_image: DeviceBuffer[DType.float32]
    var order: List[Int]
    var order_pos: Int
    var iteration: Int

    def __init__(out self, ctx: DeviceContext, var dataset: Dataset, config: TrainConfig) raises:
        self.ctx = ctx
        self.config = config.copy()
        self.splats = splats_from_points(ctx, dataset.points.xyz, dataset.points.rgb, config.sh_degree)
        self.splats.enable_training()
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

    def step(mut self) raises -> Float64:
        """One training iteration. Returns the loss on logging iterations
        (multiples of config.log_interval), else -1 (no extra sync)."""
        self.iteration += 1
        var vi = self._next_view()
        ref view = self.dataset.train[vi]
        var cam = view.camera
        var image = dptr(self.image)
        var grad = dptr(self.grad_image)
        var st = self.rasterizer.forward(
            self.splats.view(), cam, self.config.background, image, self.active_sh_degree()
        )
        var log_now = self.config.log_interval > 0 and self.iteration % self.config.log_interval == 0
        var value = self.loss.backward(
            image, dptr(view.image), grad, cam.width, cam.height, compute_loss=log_now
        )
        self.rasterizer.backward(self.splats.view(), st, image, grad, self.splats.grad_view())
        self.adam.lr[Attr.MEANS] = exponential_lr(
            self.config.lr_means_init * self.dataset.scene_extent,
            self.config.lr_means_final * self.dataset.scene_extent,
            self.iteration, self.config.lr_means_steps,
        )
        self.adam.step(self.splats)
        return value if log_now else -1.0

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
