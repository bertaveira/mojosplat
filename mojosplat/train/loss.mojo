"""Photometric loss: (1 - lambda) * L1 + lambda * (1 - SSIM), and PSNR.

SSIM as in 3DGS (Wang et al. 2004): 11x11 Gaussian window (sigma 1.5),
C1 = 0.01^2, C2 = 0.03^2, zero padding, averaged over pixels and channels.
The window is separable, so the forward is a horizontal then a vertical
pass over the five local moments (I, T, I^2, T^2, I T).

Backward (as in fused-ssim, Mallick et al.): with the target T constant,
SSIM(p) depends on the image through mu1 = G*I, E11 = G*I^2, E12 = G*(I T),
so
    dL/dI(x) = (G*dmu1)(x) + 2 I(x) (G*dE11)(x) + T(x) (G*dE12)(x)
where d* are the per-pixel partial derivatives of the loss; two more
separable passes. Images are [H, W, 3] float32; targets are uint8 (the
dataset keeps them on the GPU as 8-bit) and are converted on the fly.
"""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import exp, log10

from ..core.types import BLOCK_SIZE, F32Ptr, U8Ptr, dptr, num_blocks
from ..gpu.reduce import Summer

comptime WINDOW = 11
comptime HALF = WINDOW // 2
comptime SIGMA: Float64 = 1.5
comptime C1: Float32 = 0.01 * 0.01
comptime C2: Float32 = 0.03 * 0.03


def gaussian_window() -> SIMD[DType.float32, 16]:
    """Normalized 1D Gaussian weights (first WINDOW lanes)."""
    var w = SIMD[DType.float32, 16](0.0)
    var total: Float64 = 0.0
    for k in range(WINDOW):
        var d = Float64(k - HALF)
        total += exp(-d * d / (2.0 * SIGMA * SIGMA))
    for k in range(WINDOW):
        var d = Float64(k - HALF)
        w[k] = Float32(exp(-d * d / (2.0 * SIGMA * SIGMA)) / total)
    return w


@always_inline
def _target(t: U8Ptr, i: Int) -> Float32:
    return Float32(t[unsafe_offset=i]) * (1.0 / 255.0)


def _moments_h_kernel(
    image: F32Ptr, target: U8Ptr, h5: F32Ptr, width: Int32, height: Int32, gw: SIMD[DType.float32, 16]
):
    """Horizontal Gaussian pass of (I, T, I^2, T^2, I T) -> h5 [5, N]."""
    var n = Int(width) * Int(height) * 3
    var idx = block_idx.x * BLOCK_SIZE + thread_idx.x
    if idx >= n:
        return
    var c = idx % 3
    var x = (idx // 3) % Int(width)
    var row = idx - (x * 3 + c)  # start of this row ([y, 0, 0])
    var acc = SIMD[DType.float32, 8](0.0)
    comptime for k in range(WINDOW):
        var xx = x + k - HALF
        if xx >= 0 and xx < Int(width):
            var j = row + xx * 3 + c
            var a = image[unsafe_offset=j]
            var b = _target(target, j)
            acc += gw[k] * SIMD[DType.float32, 8](a, b, a * a, b * b, a * b, 0, 0, 0)
    comptime for q in range(5):
        h5[unsafe_offset=q * n + idx] = acc[q]


def _ssim_v_kernel(
    h5: F32Ptr, image: F32Ptr, target: U8Ptr, d3: F32Ptr, loss_px: F32Ptr,
    width: Int32, height: Int32, gw: SIMD[DType.float32, 16], lam: Float32, inv_n: Float32,
):
    """Vertical pass -> SSIM and its partials; d3 [3, N] = dL/d(mu1, E11, E12),
    loss_px [N] = per-element loss contribution (before averaging)."""
    var n = Int(width) * Int(height) * 3
    var idx = block_idx.x * BLOCK_SIZE + thread_idx.x
    if idx >= n:
        return
    var stride = Int(width) * 3
    var y = idx // stride
    var acc = SIMD[DType.float32, 8](0.0)
    comptime for k in range(WINDOW):
        var yy = y + k - HALF
        if yy >= 0 and yy < Int(height):
            var j = idx + (k - HALF) * stride
            acc += gw[k] * SIMD[DType.float32, 8](
                h5[unsafe_offset=j], h5[unsafe_offset=n + j], h5[unsafe_offset=2 * n + j],
                h5[unsafe_offset=3 * n + j], h5[unsafe_offset=4 * n + j], 0, 0, 0,
            )
    var mu1 = acc[0]
    var mu2 = acc[1]
    var s11 = acc[2] - mu1 * mu1
    var s22 = acc[3] - mu2 * mu2
    var s12 = acc[4] - mu1 * mu2
    var a1 = 2.0 * mu1 * mu2 + C1
    var a2 = 2.0 * s12 + C2
    var b1 = mu1 * mu1 + mu2 * mu2 + C1
    var b2 = s11 + s22 + C2
    var ssim = (a1 * a2) / (b1 * b2)
    # dL/dssim = -lambda / N
    var gs = -lam * inv_n
    d3[unsafe_offset=idx] = gs * ssim * (2.0 * mu2 / a1 - 2.0 * mu2 / a2 - 2.0 * mu1 / b1 + 2.0 * mu1 / b2)
    d3[unsafe_offset=n + idx] = gs * (-ssim / b2)
    d3[unsafe_offset=2 * n + idx] = gs * (2.0 * ssim / a2)
    var diff = image[unsafe_offset=idx] - _target(target, idx)
    loss_px[unsafe_offset=idx] = (1.0 - lam) * abs(diff) + lam * (1.0 - ssim)


def _grad_h_kernel(d3: F32Ptr, hd3: F32Ptr, width: Int32, height: Int32, gw: SIMD[DType.float32, 16]):
    var n = Int(width) * Int(height) * 3
    var idx = block_idx.x * BLOCK_SIZE + thread_idx.x
    if idx >= n:
        return
    var c = idx % 3
    var x = (idx // 3) % Int(width)
    var row = idx - (x * 3 + c)
    var acc = SIMD[DType.float32, 4](0.0)
    comptime for k in range(WINDOW):
        var xx = x + k - HALF
        if xx >= 0 and xx < Int(width):
            var j = row + xx * 3 + c
            acc += gw[k] * SIMD[DType.float32, 4](
                d3[unsafe_offset=j], d3[unsafe_offset=n + j], d3[unsafe_offset=2 * n + j], 0
            )
    comptime for q in range(3):
        hd3[unsafe_offset=q * n + idx] = acc[q]


def _grad_v_kernel(
    hd3: F32Ptr, image: F32Ptr, target: U8Ptr, grad: F32Ptr, width: Int32, height: Int32,
    gw: SIMD[DType.float32, 16], lam: Float32, inv_n: Float32,
):
    var n = Int(width) * Int(height) * 3
    var idx = block_idx.x * BLOCK_SIZE + thread_idx.x
    if idx >= n:
        return
    var stride = Int(width) * 3
    var y = idx // stride
    var acc = SIMD[DType.float32, 4](0.0)
    comptime for k in range(WINDOW):
        var yy = y + k - HALF
        if yy >= 0 and yy < Int(height):
            var j = idx + (k - HALF) * stride
            acc += gw[k] * SIMD[DType.float32, 4](
                hd3[unsafe_offset=j], hd3[unsafe_offset=n + j], hd3[unsafe_offset=2 * n + j], 0
            )
    var i = image[unsafe_offset=idx]
    var t = _target(target, idx)
    var d = i - t
    var sign = Float32(1.0) if d > 0.0 else (Float32(-1.0) if d < 0.0 else Float32(0.0))
    grad[unsafe_offset=idx] = acc[0] + 2.0 * i * acc[1] + t * acc[2] + (1.0 - lam) * inv_n * sign


def _sq_err_kernel(image: F32Ptr, target: U8Ptr, sq: F32Ptr, n: Int32):
    var idx = block_idx.x * BLOCK_SIZE + thread_idx.x
    if idx < Int(n):
        var d = min(max(image[unsafe_offset=idx], 0.0), 1.0) - _target(target, idx)
        sq[unsafe_offset=idx] = d * d


struct PhotometricLoss(Movable):
    """Loss + image gradient with reusable workspaces."""

    var ctx: DeviceContext
    var lam: Float32  # SSIM weight (0.2 in 3DGS)
    var cap: Int
    var h5: DeviceBuffer[DType.float32]
    var d3: DeviceBuffer[DType.float32]
    var hd3: DeviceBuffer[DType.float32]
    var px: DeviceBuffer[DType.float32]
    var summer: Summer
    var gw: SIMD[DType.float32, 16]

    def __init__(out self, ctx: DeviceContext, lambda_dssim: Float32 = 0.2) raises:
        self.ctx = ctx
        self.lam = lambda_dssim
        self.cap = 0
        self.h5 = ctx.enqueue_create_buffer[DType.float32](1)
        self.d3 = ctx.enqueue_create_buffer[DType.float32](1)
        self.hd3 = ctx.enqueue_create_buffer[DType.float32](1)
        self.px = ctx.enqueue_create_buffer[DType.float32](1)
        self.summer = Summer(ctx)
        self.gw = gaussian_window()

    def _reserve(mut self, n: Int) raises:
        if n <= self.cap:
            return
        self.h5 = self.ctx.enqueue_create_buffer[DType.float32](5 * n)
        self.d3 = self.ctx.enqueue_create_buffer[DType.float32](3 * n)
        self.hd3 = self.ctx.enqueue_create_buffer[DType.float32](3 * n)
        self.px = self.ctx.enqueue_create_buffer[DType.float32](n)
        self.cap = n

    def backward(
        mut self, image: F32Ptr, target: U8Ptr, grad: F32Ptr, width: Int, height: Int,
        compute_loss: Bool = False,
    ) raises -> Float64:
        """Write dL/d(image) into `grad` ([H, W, 3]). Returns the loss value if
        `compute_loss` (adds a sync), else 0."""
        var n = width * height * 3
        self._reserve(n)
        var ctx = self.ctx
        var stream = ctx.stream()
        var inv_n = Float32(1.0 / Float64(n))
        var g = num_blocks(n)
        stream.enqueue_function(
            ctx.compile_function[_moments_h_kernel](), image, target, dptr(self.h5),
            Int32(width), Int32(height), self.gw, grid_dim=g, block_dim=BLOCK_SIZE,
        )
        stream.enqueue_function(
            ctx.compile_function[_ssim_v_kernel](), dptr(self.h5), image, target,
            dptr(self.d3), dptr(self.px), Int32(width), Int32(height), self.gw,
            self.lam, inv_n, grid_dim=g, block_dim=BLOCK_SIZE,
        )
        stream.enqueue_function(
            ctx.compile_function[_grad_h_kernel](), dptr(self.d3), dptr(self.hd3),
            Int32(width), Int32(height), self.gw, grid_dim=g, block_dim=BLOCK_SIZE,
        )
        stream.enqueue_function(
            ctx.compile_function[_grad_v_kernel](), dptr(self.hd3), image, target, grad,
            Int32(width), Int32(height), self.gw, self.lam, inv_n,
            grid_dim=g, block_dim=BLOCK_SIZE,
        )
        if not compute_loss:
            return 0.0
        return self.summer.sum(dptr(self.px), n) / Float64(n)

    def psnr(mut self, image: F32Ptr, target: U8Ptr, width: Int, height: Int) raises -> Float64:
        """PSNR of the image clamped to [0, 1] against the 8-bit target."""
        var n = width * height * 3
        self._reserve(n)
        self.ctx.stream().enqueue_function(
            self.ctx.compile_function[_sq_err_kernel](), image, target, dptr(self.px),
            Int32(n), grid_dim=num_blocks(n), block_dim=BLOCK_SIZE,
        )
        var mse = self.summer.sum(dptr(self.px), n) / Float64(n)
        return -10.0 * log10(max(mse, 1e-20))
