"""Adam over the parameter arrays of `Splats`, matching torch.optim.Adam.

Each attribute (means, scales, ...) is its own parameter group with its own
learning rate. Bias corrections are computed on the host and folded into
the step size, as in PyTorch, llm.c and Faster-GS. The per-element update is
a separate function so it can later be fused into the backward pass.
"""

from max.gpu import block_idx, thread_idx
from std.math import sqrt

from ..core.types import BLOCK_SIZE, F32Ptr, dptr, num_blocks
from ..model.splats import Attr, Splats


@always_inline
def adam_update(
    p: Float32, g: Float32, m: Float32, v: Float32,
    step_size: Float32, bc2_sqrt_rcp: Float32, beta1: Float32, beta2: Float32, eps: Float32,
) -> SIMD[DType.float32, 4]:
    """One Adam step for one element: returns (param, exp_avg, exp_avg_sq)."""
    var m2 = beta1 * m + (1.0 - beta1) * g
    var v2 = beta2 * v + (1.0 - beta2) * g * g
    var denom = sqrt(v2) * bc2_sqrt_rcp + eps
    return SIMD[DType.float32, 4](p - step_size * m2 / denom, m2, v2, 0.0)


def _adam_kernel(
    param: F32Ptr, grad: F32Ptr, exp_avg: F32Ptr, exp_avg_sq: F32Ptr, n: Int32,
    step_size: Float32, bc2_sqrt_rcp: Float32, beta1: Float32, beta2: Float32, eps: Float32,
):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i >= Int(n):
        return
    var r = adam_update(
        param[unsafe_offset=i], grad[unsafe_offset=i], exp_avg[unsafe_offset=i],
        exp_avg_sq[unsafe_offset=i], step_size, bc2_sqrt_rcp, beta1, beta2, eps,
    )
    param[unsafe_offset=i] = r[0]
    exp_avg[unsafe_offset=i] = r[1]
    exp_avg_sq[unsafe_offset=i] = r[2]


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
        out self, lr: SIMD[DType.float32, 8], beta1: Float32 = 0.9, beta2: Float32 = 0.999,
        eps: Float32 = 1e-15,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0

    def step(mut self, mut splats: Splats) raises:
        """Apply one step to the first `count` rows using `splats.grads`."""
        if not splats.training:
            raise Error("Adam.step: call splats.enable_training() first")
        self.step_count += 1
        var t = Float64(self.step_count)
        var bc1 = 1.0 - Float64(self.beta1) ** t
        var bc2 = 1.0 - Float64(self.beta2) ** t
        var bc2_sqrt_rcp = Float32(1.0 / sqrt(bc2))
        var ctx = splats.ctx
        var stream = ctx.stream()
        var kernel = ctx.compile_function[_adam_kernel]()
        for a in range(Attr.COUNT):
            var n = splats.count * splats.width(a)
            if n == 0:
                continue
            stream.enqueue_function(
                kernel, dptr(splats.params[a]), dptr(splats.grads[a]),
                dptr(splats.exp_avg[a]), dptr(splats.exp_avg_sq[a]), Int32(n),
                Float32(Float64(self.lr[a]) / bc1), bc2_sqrt_rcp, self.beta1,
                self.beta2, self.eps,
                grid_dim=num_blocks(n), block_dim=BLOCK_SIZE,
            )
