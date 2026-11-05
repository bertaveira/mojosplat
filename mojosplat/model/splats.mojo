"""Owning device storage of a set of Gaussians (raw parameters)."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import log

from ..core.splats import SplatParams
from ..core.types import dptr
from ..io.arrays import upload
from ..io.splat import SplatScene
from ..core.math import SH_C0


struct SplatBuffers(Movable):
    """Owns N Gaussians' raw parameters in device memory.

    Pass this (not a `SplatParams` view) to `Rasterizer.render` from Mojo code
    so the buffers outlive the enqueued work.
    """

    var means: DeviceBuffer[DType.float32]
    var log_scales: DeviceBuffer[DType.float32]
    var rotations: DeviceBuffer[DType.float32]
    var opacity_logits: DeviceBuffer[DType.float32]
    var sh0: DeviceBuffer[DType.float32]
    var sh_rest: DeviceBuffer[DType.float32]
    var count: Int
    var rest_stride: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        means: List[Float32],
        log_scales: List[Float32],
        rotations: List[Float32],
        opacity_logits: List[Float32],
        sh0: List[Float32],
        sh_rest: List[Float32],
        rest_stride: Int,
    ) raises:
        self.count = len(opacity_logits)
        self.rest_stride = rest_stride
        self.means = upload(ctx, means)
        self.log_scales = upload(ctx, log_scales)
        self.rotations = upload(ctx, rotations)
        self.opacity_logits = upload(ctx, opacity_logits)
        self.sh0 = upload(ctx, sh0)
        if len(sh_rest) > 0:
            self.sh_rest = upload(ctx, sh_rest)
        else:
            self.sh_rest = ctx.enqueue_create_buffer[DType.float32](3)

    @staticmethod
    def from_scene(ctx: DeviceContext, scene: SplatScene) raises -> SplatBuffers:
        """Convert a .splat scene (activated values, RGB) to raw parameters."""
        var n = scene.count
        var log_scales = List[Float32](capacity=n * 3)
        for s in scene.scales:
            log_scales.append(log(max(s, 1e-30)))
        var logits = List[Float32](capacity=n)
        for o in scene.opacities:
            var p = min(max(o, 1e-6), 1.0 - 1e-6)
            logits.append(log(p / (1.0 - p)))
        var sh0 = List[Float32](capacity=n * 3)
        for c in scene.colors:
            sh0.append((c - 0.5) / SH_C0)
        return SplatBuffers(
            ctx, scene.means, log_scales, scene.quats, logits, sh0, List[Float32](), 0
        )

    def view(self) -> SplatParams:
        return SplatParams(
            dptr(self.means), dptr(self.log_scales), dptr(self.rotations),
            dptr(self.opacity_logits), dptr(self.sh0), dptr(self.sh_rest),
            self.count, self.rest_stride,
        )
