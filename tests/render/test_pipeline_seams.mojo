"""A second pipeline, for tests only: proves the seams compile and carry data.

`ScaledGaussian` is `Gaussian2D` with alpha multiplied by a per-Gaussian value
from its own column: that value has to travel from the column through the
slot, into blend, and its gradient back to the column.
"""

from max.gpu.host import DeviceContext
from std.testing import assert_raises, assert_true, TestSuite

from mojosplat.core.camera import Camera
from mojosplat.core.linalg import Sym2, V2
from mojosplat.core.splats import ExtraColumns
from mojosplat.core.types import SlabPtr, dptr
from mojosplat.gpu.elementwise import fill_f32
from mojosplat.io.arrays import download
from mojosplat.model.splats import Splats
from mojosplat.render.blocks.appearance.sh import SphericalHarmonics
from mojosplat.render.blocks.contracts import Hit, LoadedResponse, Pipeline, Response, ScreenGrad
from mojosplat.render.blocks.placement.pinhole_ewa import PinholeEWA
from mojosplat.render.blocks.response.gaussian2d import Gaussian2D, LoadedGaussian2D
from mojosplat.render.pipelines import Standard3DGS
from mojosplat.render.rasterizer import Rasterizer
from mojosplat.render.record import REC

comptime _F8 = SIMD[DType.float32, 8]


struct LoadedScaled(ImplicitlyCopyable, LoadedResponse, Movable):
    var inner: LoadedGaussian2D
    var s: Float32
    var d_s: Float32

    @always_inline
    def __init__(out self):
        self.inner = LoadedGaussian2D()
        self.s = 1.0
        self.d_s = 0.0

    @always_inline
    def __init__(out self, src: SlabPtr):
        self.inner = LoadedGaussian2D(src)
        self.s = src[unsafe_offset=9]
        self.d_s = 0.0

    @always_inline
    def __init__(out self, rec: SIMD[DType.float32, REC], extra: _F8):
        self.inner = LoadedGaussian2D(rec, extra)
        self.s = extra[0]
        self.d_s = 0.0

    @always_inline
    def eval(self, px: Float32, py: Float32) -> Hit:
        var h = self.inner.eval(px, py)
        h[0] *= self.s
        h[1] *= self.s
        return h

    @always_inline
    def add(mut self, hit: Hit, dL_dalpha: Float32):
        self.inner.add(hit, dL_dalpha)
        self.d_s += hit[0] / self.s * dL_dalpha

    @always_inline
    def gradient(self) -> _F8:
        var g = self.inner.gradient()
        g[6] = self.d_s
        return g


struct ScaledGaussian(ImplicitlyCopyable, Movable, Response):
    comptime Loaded = LoadedScaled
    comptime TILE_ARGS = Gaussian2D.TILE_ARGS
    comptime SLAB = Gaussian2D.SLAB
    comptime GRAD = 7
    comptime EXTRA = 1
    comptime NUM_COLUMNS = 1

    var inner: Gaussian2D

    @always_inline
    def __init__(out self, mean2d: V2[DType.float32], cov2d: Sym2[DType.float32]):
        self.inner = Gaussian2D(mean2d, cov2d)

    @always_inline
    def valid(self) -> Bool:
        return self.inner.valid()

    @always_inline
    def extent(self, opacity: Float32) -> SIMD[DType.float32, 2]:
        return self.inner.extent(opacity)

    @always_inline
    def record(self, opacity: Float32) -> SIMD[DType.float32, REC]:
        return self.inner.record(opacity)

    @always_inline
    @staticmethod
    def extra(cols: ExtraColumns, g: Int) -> _F8:
        var v = _F8(0.0)
        v[0] = cols.c0[unsafe_offset=g]
        return v

    @always_inline
    @staticmethod
    def tile_args(rec: SIMD[DType.float32, REC]) -> _F8:
        return Gaussian2D.tile_args(rec)

    @always_inline
    @staticmethod
    def touches_tile[tile_w: Int, tile_h: Int](args: _F8, tile_x: Int, tile_y: Int) -> Bool:
        return Gaussian2D.touches_tile[tile_w, tile_h](args, tile_x, tile_y)

    @always_inline
    @staticmethod
    def store_slab(dst: SlabPtr, rec: SIMD[DType.float32, REC], extra: _F8, color: SIMD[DType.float32, 4]):
        Gaussian2D.store_slab(dst, rec, extra, color)
        dst[unsafe_offset=9] = extra[0]

    @always_inline
    @staticmethod
    def channel(src: SlabPtr, k: Int) -> Float32:
        return Gaussian2D.channel(src, k)

    @always_inline
    def backward(self, grad: _F8) -> ScreenGrad:
        return self.inner.backward(grad)

    @always_inline
    @staticmethod
    def write_gradients(cols: ExtraColumns, grads: ExtraColumns, g: Int, slot_grad: _F8):
        grads.c0[unsafe_offset=g] = slot_grad[6]

    @always_inline
    @staticmethod
    def zero_gradients(grads: ExtraColumns, g: Int):
        grads.c0[unsafe_offset=g] = 0.0

    @always_inline
    @staticmethod
    def grad_index(k: Int, n_slots: Int, slot: Int) -> Int:
        if k == 6:
            return n_slots * 6 + slot
        return Gaussian2D.grad_index(k, n_slots, slot)


struct ScaledPipeline(Pipeline):
    comptime ID = 99
    comptime Place = PinholeEWA[DType.float32]
    comptime Resp = ScaledGaussian
    comptime Appear = SphericalHarmonics


def scene(ctx: DeviceContext, s: Float32) raises -> Splats:
    var means: List[Float32] = [0.0, 0.0, 2.0]
    var scales: List[Float32] = [-2.5, -2.5, -2.5]
    var quats: List[Float32] = [1.0, 0.0, 0.0, 0.0]
    var opacity: List[Float32] = [0.0]
    var sh0: List[Float32] = [1.0, 1.0, 1.0]
    var splats = Splats(ctx, means, scales, quats, opacity, sh0, List[Float32](), 0)
    var value: List[Float32] = [s]
    splats.add_column("scale_of_alpha", 1, values=value)
    return splats^


def camera() -> Camera:
    var view = SIMD[DType.float32, 16](1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    return Camera(view, 100.0, 100.0, 32.0, 32.0, 64, 64, 0.01, 100.0)


def loss_of(s: Float32) raises -> Float64:
    """L = sum of all image values, rendered through the second pipeline."""
    var ctx = DeviceContext()
    var splats = scene(ctx, s)
    var image = ctx.enqueue_create_buffer[DType.float32](64 * 64 * 3)
    var rast = Rasterizer(ctx)
    rast.render[ScaledPipeline](splats.view(), camera(), SIMD[DType.float32, 4](0.0), dptr(image), 0)
    ctx.synchronize()
    var total: Float64 = 0.0
    for v in download(ctx, image, 64 * 64 * 3):
        total += Float64(v)
    _ = (image^, splats^)
    return total


def test_extra_column_reaches_blend() raises:
    var half = loss_of(0.5)
    var full = loss_of(1.0)
    assert_true(full > 1.5 * half and half > 0.0, "alpha should follow the response's own column")


def test_gradient_reaches_the_column() raises:
    var ctx = DeviceContext()
    var s: Float32 = 0.6
    var splats = scene(ctx, s)
    splats.enable_training()
    var image = ctx.enqueue_create_buffer[DType.float32](64 * 64 * 3)
    var grad_image = ctx.enqueue_create_buffer[DType.float32](64 * 64 * 3)
    fill_f32(ctx, dptr(grad_image), 1.0, 64 * 64 * 3)  # L = sum of the image
    var rast = Rasterizer(ctx)
    var view = splats.view()
    var st = rast.forward[ScaledPipeline](view, camera(), SIMD[DType.float32, 4](0.0), dptr(image), 0)
    rast.backward[ScaledPipeline](view, st, dptr(image), dptr(grad_image), splats.grad_view())
    ctx.synchronize()
    var analytic = Float64(download(ctx, splats.grads[6], 1)[0])
    var h: Float32 = 0.01
    var numeric = (loss_of(s + h) - loss_of(s - h)) / Float64(2.0 * h)
    assert_true(abs(analytic / numeric - 1.0) < 0.01, String("gradient ", analytic, " vs ", numeric))
    _ = (image^, grad_image^, splats^)


def test_backward_rejects_another_pipelines_state() raises:
    var ctx = DeviceContext()
    var splats = scene(ctx, 1.0)
    splats.enable_training()
    var image = ctx.enqueue_create_buffer[DType.float32](64 * 64 * 3)
    var grad_image = ctx.enqueue_create_buffer[DType.float32](64 * 64 * 3)
    var rast = Rasterizer(ctx)
    var view = splats.view()
    var st = rast.forward[ScaledPipeline](view, camera(), SIMD[DType.float32, 4](0.0), dptr(image), 0)
    with assert_raises():
        rast.backward[Standard3DGS](view, st, dptr(image), dptr(grad_image), splats.grad_view())
    ctx.synchronize()
    _ = (image^, grad_image^, splats^)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
