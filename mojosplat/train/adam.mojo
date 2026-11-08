"""Adam over the parameter arrays of `Splats`, matching torch.optim.Adam.

Each attribute (means, scales, ...) is its own parameter group with its own
learning rate. Bias corrections are computed on the host and folded into
the step size, as in PyTorch, llm.c and Faster-GS. The per-element update is
a separate function so it can later be fused into the backward pass.
"""

from max.gpu import block_idx, thread_idx
from std.math import rsqrt, sqrt

from ..core.types import BLOCK_SIZE, F32Ptr, I32Ptr, KernelVector, dptr, num_blocks
from ..model.splats import Attr, Splats
from ..render.preprocess_backward import ShRestGrad
from ..render.blocks.appearance.sh import sh_rest_basis


@always_inline
def adam_update(
    p: Float32,
    g: Float32,
    m: Float32,
    v: Float32,
    step_size: Float32,
    bc2_sqrt_rcp: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
) -> SIMD[DType.float32, 4]:
    """One Adam step for one element: returns (param, exp_avg, exp_avg_sq)."""
    var m2 = beta1 * m + (1.0 - beta1) * g
    var v2 = beta2 * v + (1.0 - beta2) * g * g
    var denom = sqrt(v2) * bc2_sqrt_rcp + eps
    return SIMD[DType.float32, 4](p - step_size * m2 / denom, m2, v2, 0.0)


def _adam_kernel(
    param: F32Ptr,
    grad: F32Ptr,
    exp_avg: F32Ptr,
    exp_avg_sq: F32Ptr,
    n: Int32,
    step_size: Float32,
    bc2_sqrt_rcp: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i >= Int(n):
        return
    var r = adam_update(
        param[unsafe_offset=i],
        grad[unsafe_offset=i],
        exp_avg[unsafe_offset=i],
        exp_avg_sq[unsafe_offset=i],
        step_size,
        bc2_sqrt_rcp,
        beta1,
        beta2,
        eps,
    )
    param[unsafe_offset=i] = r[0]
    exp_avg[unsafe_offset=i] = r[1]
    exp_avg_sq[unsafe_offset=i] = r[2]


def _adam_sh_rest_kernel(
    param: F32Ptr,
    exp_avg: F32Ptr,
    exp_avg_sq: F32Ptr,
    means: F32Ptr,
    slot_of: I32Ptr,
    grad_color: F32Ptr,
    cam_pos: KernelVector[DType.float32, 4],
    n: Int32,
    rest_stride: Int32,
    active_bases: Int32,
    step_size: Float32,
    bc2_sqrt_rcp: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
):
    """Adam on sh_rest [N, K, 3] with the gradient recomputed per element
    (basis(dir) * dL/dcolor) instead of read from a gradient buffer. Bands
    at or above `active_bases` are skipped: callers must never lower the
    active degree once it has been raised."""
    var e = block_idx.x * BLOCK_SIZE + thread_idx.x
    var per = Int(rest_stride) * 3
    if e >= Int(n) * per:
        return
    # 32-bit index math: 64-bit division is very slow on Apple GPUs.
    var g = Int(UInt32(e) // UInt32(per))
    var rem = UInt32(e) - UInt32(g) * UInt32(per)
    var k = Int(rem // 3)
    var c = Int(rem % 3)
    # Bands above the active SH degree have never had a gradient (the degree
    # only grows), so their moments are 0 and Adam would leave them unchanged.
    if k >= Int(active_bases) - 1:
        return
    var grad: Float32 = 0.0
    var slot = Int(slot_of[unsafe_offset=g])
    if slot >= 0:
        var dx = means[unsafe_offset=g * 3 + 0] - cam_pos[0]
        var dy = means[unsafe_offset=g * 3 + 1] - cam_pos[1]
        var dz = means[unsafe_offset=g * 3 + 2] - cam_pos[2]
        var inv = rsqrt(dx * dx + dy * dy + dz * dz)
        grad = sh_rest_basis(k, dx * inv, dy * inv, dz * inv) * grad_color[unsafe_offset=slot * 3 + c]
    var r = adam_update(
        param[unsafe_offset=e],
        grad,
        exp_avg[unsafe_offset=e],
        exp_avg_sq[unsafe_offset=e],
        step_size,
        bc2_sqrt_rcp,
        beta1,
        beta2,
        eps,
    )
    param[unsafe_offset=e] = r[0]
    exp_avg[unsafe_offset=e] = r[1]
    exp_avg_sq[unsafe_offset=e] = r[2]


struct Adam(Movable):
    """Adam state shared by all groups: hyperparameters and the step count.
    The moments live in `Splats` (exp_avg / exp_avg_sq), next to the rows
    they belong to."""

    var lr: SIMD[DType.float32, 8]  # per Attr index
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var step_count: Int

    def __init__(
        out self,
        lr: SIMD[DType.float32, 8],
        beta1: Float32 = 0.9,
        beta2: Float32 = 0.999,
        eps: Float32 = 1e-15,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0

    def step(mut self, mut splats: Splats) raises:
        """One step on the first `count` rows, reading every gradient from
        `splats.grads`."""
        if not splats.has_sh_rest_grad:
            raise Error("Adam.step: no sh_rest gradient buffer (pass the ShRestGrad)")
        self._step(splats, Optional[ShRestGrad]())

    def step(mut self, mut splats: Splats, sh_rest: ShRestGrad) raises:
        """One step where the sh_rest gradients are recomputed elementwise
        from `sh_rest` (the backward ran with write_sh_rest=False): saves
        writing and reading the largest gradient buffer."""
        self._step(splats, Optional(sh_rest))

    def _step(mut self, mut splats: Splats, sh_rest: Optional[ShRestGrad]) raises:
        if not splats.training:
            raise Error("Adam.step: call splats.enable_training() first")
        self.step_count += 1
        var t = Float64(self.step_count)
        var bc1 = 1.0 - Float64(self.beta1) ** t
        var bc2 = 1.0 - Float64(self.beta2) ** t
        var bc2_sqrt_rcp = Float32(1.0 / sqrt(bc2))
        var ctx = splats.ctx
        var stream = ctx.stream()

        var step_size = SIMD[DType.float32, 8](0.0)  # lr / bias correction 1, per Attr
        for a in range(Attr.COUNT):
            step_size[a] = Float32(Float64(self.lr[a]) / bc1)

        # sh_rest first: its recomputed gradient needs the means before they move.
        if sh_rest:
            var sg = sh_rest.value()
            var n = splats.count * splats.width(Attr.SH_REST)
            if n > 0:
                stream.enqueue_function(
                    ctx.compile_function[_adam_sh_rest_kernel](),
                    dptr(splats.params[Attr.SH_REST]),
                    dptr(splats.exp_avg[Attr.SH_REST]),
                    dptr(splats.exp_avg_sq[Attr.SH_REST]),
                    dptr(splats.params[Attr.MEANS]),
                    sg.slot_of,
                    sg.grad_color,
                    KernelVector[DType.float32, 4](sg.cam_pos),
                    Int32(splats.count),
                    Int32(splats.rest_stride),
                    Int32(sg.active_bases),
                    step_size[Attr.SH_REST],
                    bc2_sqrt_rcp,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    grid_dim=num_blocks(n),
                    block_dim=BLOCK_SIZE,
                )
        var kernel = ctx.compile_function[_adam_kernel]()
        for a in range(Attr.COUNT):
            if a == Attr.SH_REST and sh_rest:
                continue
            var n = splats.count * splats.width(a)
            if n == 0:
                continue
            stream.enqueue_function(
                kernel,
                dptr(splats.params[a]),
                dptr(splats.grads[a]),
                dptr(splats.exp_avg[a]),
                dptr(splats.exp_avg_sq[a]),
                Int32(n),
                step_size[a],
                bc2_sqrt_rcp,
                self.beta1,
                self.beta2,
                self.eps,
                grid_dim=num_blocks(n),
                block_dim=BLOCK_SIZE,
            )
