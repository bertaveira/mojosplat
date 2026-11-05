"""Splats: the owner of a set of Gaussians on the GPU.

Holds the six raw parameter arrays (see `SplatView` for their layout) with
room for `capacity` Gaussians, of which the first `count` are live. Training
state is allocated only by `enable_training()`: per attribute a gradient
buffer and the Adam moments (exp_avg, exp_avg_sq), plus the densification
statistics. Every per-Gaussian array is indexed by `Attr`, so operations
that must touch all of them (growing the capacity now; append / compact /
permute for densification) are one loop.

Mojo code passes `Splats` around (it keeps the memory alive); kernels get a
`SplatView` of raw pointers.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import log

from ..core.math import SH_C0
from ..core.splats import SplatView
from ..core.types import F32Ptr, dptr
from ..gpu.elementwise import copy_f32, fill_f32
from ..io.arrays import download
from ..io.ply import HostSplats
from ..io.splat import SplatScene


struct Attr:
    """Indices of the per-Gaussian parameter arrays."""

    comptime MEANS = 0
    comptime LOG_SCALES = 1
    comptime ROTATIONS = 2
    comptime OPACITY_LOGITS = 3
    comptime SH0 = 4
    comptime SH_REST = 5
    comptime COUNT = 6


def attr_width(attr: Int, rest_stride: Int) -> Int:
    """Floats per Gaussian of attribute `attr`."""
    if attr == Attr.MEANS or attr == Attr.LOG_SCALES or attr == Attr.SH0:
        return 3
    elif attr == Attr.ROTATIONS:
        return 4
    elif attr == Attr.OPACITY_LOGITS:
        return 1
    return rest_stride * 3


def _upload_padded(ctx: DeviceContext, values: List[Float32], capacity_floats: Int) raises -> DeviceBuffer[DType.float32]:
    var n = max(capacity_floats, 1)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(len(values)):
        host[i] = values[i]
    for i in range(len(values), n):
        host[i] = 0.0
    var dev = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dev, host)
    return dev


struct Splats(Movable):
    var ctx: DeviceContext
    var count: Int
    var capacity: Int
    var rest_stride: Int  # SH bases per Gaussian in sh_rest (0, 3, 8 or 15)
    var params: List[DeviceBuffer[DType.float32]]  # [Attr.COUNT], capacity * width each
    # Training state (empty until enable_training)
    var training: Bool
    var grads: List[DeviceBuffer[DType.float32]]
    var exp_avg: List[DeviceBuffer[DType.float32]]
    var exp_avg_sq: List[DeviceBuffer[DType.float32]]
    var densification: DeviceBuffer[DType.float32]  # [2, count]: visibility count, |dL/dmean2d| sum

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
        capacity: Int = 0,
    ) raises:
        """Upload host arrays (raw parameters, SoA; see SplatView)."""
        self.ctx = ctx
        self.count = len(opacity_logits)
        self.capacity = max(capacity, self.count, 1)
        self.rest_stride = rest_stride
        self.params = List[DeviceBuffer[DType.float32]]()
        self.params.append(_upload_padded(ctx, means, self.capacity * 3))
        self.params.append(_upload_padded(ctx, log_scales, self.capacity * 3))
        self.params.append(_upload_padded(ctx, rotations, self.capacity * 4))
        self.params.append(_upload_padded(ctx, opacity_logits, self.capacity))
        self.params.append(_upload_padded(ctx, sh0, self.capacity * 3))
        self.params.append(_upload_padded(ctx, sh_rest, self.capacity * rest_stride * 3))
        self.training = False
        self.grads = List[DeviceBuffer[DType.float32]]()
        self.exp_avg = List[DeviceBuffer[DType.float32]]()
        self.exp_avg_sq = List[DeviceBuffer[DType.float32]]()
        self.densification = ctx.enqueue_create_buffer[DType.float32](1)

    @staticmethod
    def from_scene(ctx: DeviceContext, scene: SplatScene) raises -> Splats:
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
        return Splats(ctx, scene.means, log_scales, scene.quats, logits, sh0, List[Float32](), 0)

    @staticmethod
    def from_host(ctx: DeviceContext, h: HostSplats, capacity: Int = 0) raises -> Splats:
        return Splats(
            ctx, h.means, h.log_scales, h.rotations, h.opacity_logits, h.sh0,
            h.sh_rest, h.rest_stride, capacity,
        )

    def to_host(self) raises -> HostSplats:
        var h = HostSplats(self.count, self.rest_stride)
        h.means = download(self.ctx, self.params[Attr.MEANS], self.count * 3)
        h.log_scales = download(self.ctx, self.params[Attr.LOG_SCALES], self.count * 3)
        h.rotations = download(self.ctx, self.params[Attr.ROTATIONS], self.count * 4)
        h.opacity_logits = download(self.ctx, self.params[Attr.OPACITY_LOGITS], self.count)
        h.sh0 = download(self.ctx, self.params[Attr.SH0], self.count * 3)
        if self.rest_stride > 0:
            h.sh_rest = download(self.ctx, self.params[Attr.SH_REST], self.count * self.rest_stride * 3)
        return h^

    def width(self, attr: Int) -> Int:
        return attr_width(attr, self.rest_stride)

    def _view(self, bufs: List[DeviceBuffer[DType.float32]]) -> SplatView:
        return SplatView(
            dptr(bufs[0]), dptr(bufs[1]), dptr(bufs[2]), dptr(bufs[3]),
            dptr(bufs[4]), dptr(bufs[5]), self.count, self.rest_stride,
        )

    def view(self) -> SplatView:
        """Raw pointers to the parameters (for kernels)."""
        return self._view(self.params)

    def grad_view(self) raises -> SplatView:
        """Raw pointers to the gradient buffers (laid out like the parameters)."""
        if not self.training:
            raise Error("Splats.grad_view: call enable_training() first")
        return self._view(self.grads)

    def densification_ptr(self) -> F32Ptr:
        return dptr(self.densification)

    def _zeros(self, n: Int) raises -> DeviceBuffer[DType.float32]:
        var b = self.ctx.enqueue_create_buffer[DType.float32](max(n, 1))
        fill_f32(self.ctx, self.ctx.stream(), dptr(b), 0.0, max(n, 1))
        return b

    def enable_training(mut self) raises:
        """Allocate gradients, Adam moments (zeroed) and densification stats."""
        if self.training:
            return
        for a in range(Attr.COUNT):
            var n = self.capacity * self.width(a)
            self.grads.append(self._zeros(n))
            self.exp_avg.append(self._zeros(n))
            self.exp_avg_sq.append(self._zeros(n))
        self.densification = self._zeros(2 * self.capacity)
        self.training = True

    def reset_densification(mut self) raises:
        fill_f32(self.ctx, self.ctx.stream(), dptr(self.densification), 0.0, 2 * self.count)

    def reserve(mut self, capacity: Int) raises:
        """Grow every per-Gaussian array to hold `capacity` Gaussians,
        keeping the first `count` rows (new rows are zero)."""
        if capacity <= self.capacity:
            return
        var stream = self.ctx.stream()
        for a in range(Attr.COUNT):
            var w = self.width(a)
            var grown = self._zeros(capacity * w)
            copy_f32(self.ctx, stream, dptr(grown), dptr(self.params[a]), self.count * w)
            self.params[a] = grown
            if self.training:
                var g = self._zeros(capacity * w)
                var m = self._zeros(capacity * w)
                var v = self._zeros(capacity * w)
                copy_f32(self.ctx, stream, dptr(m), dptr(self.exp_avg[a]), self.count * w)
                copy_f32(self.ctx, stream, dptr(v), dptr(self.exp_avg_sq[a]), self.count * w)
                self.grads[a] = g
                self.exp_avg[a] = m
                self.exp_avg_sq[a] = v
        if self.training:
            var d = self._zeros(2 * capacity)
            self.densification = d
        self.capacity = capacity
