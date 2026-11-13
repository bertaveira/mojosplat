"""Relocation densification, regularizers and position noise.

The relocation of 3DGS-MCMC (Kheradmand et al., "3D Gaussian Splatting as
Markov Chain Monte Carlo", NeurIPS 2024), with the settings Beta Splatting
uses. Every densification step, Gaussians with opacity <= 0.005 are moved onto
live ones, sampled by opacity; then 5% more are added the same way, up to
`cap_max`. A sampled Gaussian with opacity o and its copies each get
1 - (1 - o)^(1 / (copies + 1)), so together they keep its opacity, and its Adam
moments are reset. Nothing is cloned by gradient, split or pruned.

Sampling is on the host (every 100 iterations): one multinomial draw per moved
Gaussian from a cumulative table.
"""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import exp, log, pow
from std.random import random_float64, random_ui64
from std.random.philox import NormalRandom

from ..cli.args import Arg
from ..core.linalg import V3, V4
from ..core.types import BLOCK_SIZE, F32Ptr, I32Ptr, dptr, num_blocks
from ..io.arrays import download, upload
from ..model.splats import Attr, Splats
from ..render.blocks.shape import QuatScale
from .strategy import Densification

comptime DEAD_OPACITY: Float64 = 0.005
comptime MAX_OPACITY: Float64 = 1.0 - 1.1920929e-07


def _logit(p: Float64) -> Float32:
    return Float32(log(p / (1.0 - p)))


def _copy_rows_kernel(p: F32Ptr, dst: I32Ptr, src: I32Ptr, n_rows: Int32, width: Int32):
    """p[dst[i], :] = p[src[i], :]."""
    var e = block_idx.x * BLOCK_SIZE + thread_idx.x
    if e >= Int(n_rows) * Int(width):
        return
    var i = Int(UInt32(e) // UInt32(width))
    var k = e - i * Int(width)
    p[unsafe_offset=Int(dst[unsafe_offset=i]) * Int(width) + k] = p[
        unsafe_offset=Int(src[unsafe_offset=i]) * Int(width) + k
    ]


def _set_rows_kernel(p: F32Ptr, rows: I32Ptr, values: F32Ptr, n: Int32):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i < Int(n):
        p[unsafe_offset=Int(rows[unsafe_offset=i])] = values[unsafe_offset=i]


def _noise_kernel(
    means: F32Ptr,
    log_scales: F32Ptr,
    rotations: F32Ptr,
    opacity_logits: F32Ptr,
    n: Int32,
    seed: UInt64,
    scale: Float32,
):
    """means += Cov * z * (1 - opacity)^100 * scale, z ~ N(0, I)."""
    var g = block_idx.x * BLOCK_SIZE + thread_idx.x
    if g >= Int(n):
        return
    comptime dt = DType.float32
    var rng = NormalRandom(seed=seed, subsequence=UInt64(g))
    var z = rng.step_normal()
    var one_minus = 1.0 - 1.0 / (1.0 + exp(-opacity_logits[unsafe_offset=g]))
    var factor = exp(100.0 * log(max(one_minus, 1e-30))) * scale
    var c = QuatScale[dt](V3[dt].load(log_scales, g), V4[dt].load(rotations, g)).cov
    var nx = z[0] * factor
    var ny = z[1] * factor
    var nz = z[2] * factor
    means[unsafe_offset=g * 3 + 0] += c.c00 * nx + c.c01 * ny + c.c02 * nz
    means[unsafe_offset=g * 3 + 1] += c.c01 * nx + c.c11 * ny + c.c12 * nz
    means[unsafe_offset=g * 3 + 2] += c.c02 * nx + c.c12 * ny + c.c22 * nz


def _opacity_reg_kernel(grad: F32Ptr, logits: F32Ptr, n: Int32, weight: Float32):
    """Adds the gradient of weight * mean(opacity) with respect to the logits."""
    var g = block_idx.x * BLOCK_SIZE + thread_idx.x
    if g < Int(n):
        var o = 1.0 / (1.0 + exp(-logits[unsafe_offset=g]))
        grad[unsafe_offset=g] += weight * o * (1.0 - o)


def _scale_reg_kernel(grad: F32Ptr, log_scales: F32Ptr, n_floats: Int32, weight: Float32):
    """Adds the gradient of weight * mean(scale) with respect to the log scales."""
    var e = block_idx.x * BLOCK_SIZE + thread_idx.x
    if e < Int(n_floats):
        grad[unsafe_offset=e] += weight * exp(log_scales[unsafe_offset=e])


def add_regularizer_gradients(mut splats: Splats, opacity_reg: Float32, scale_reg: Float32) raises:
    """Adds d/d(params) of opacity_reg * mean(opacity) + scale_reg * mean(scale)."""
    var ctx = splats.ctx
    var n = splats.count
    ctx.stream().enqueue_function(
        ctx.compile_function[_opacity_reg_kernel](),
        dptr(splats.grads[Attr.OPACITY_LOGITS]),
        dptr(splats.params[Attr.OPACITY_LOGITS]),
        Int32(n),
        opacity_reg / Float32(n),
        grid_dim=num_blocks(n),
        block_dim=BLOCK_SIZE,
    )
    ctx.stream().enqueue_function(
        ctx.compile_function[_scale_reg_kernel](),
        dptr(splats.grads[Attr.LOG_SCALES]),
        dptr(splats.params[Attr.LOG_SCALES]),
        Int32(n * 3),
        scale_reg / Float32(n * 3),
        grid_dim=num_blocks(n * 3),
        block_dim=BLOCK_SIZE,
    )


def add_position_noise(mut splats: Splats, scale: Float32) raises:
    var ctx = splats.ctx
    var n = splats.count
    ctx.stream().enqueue_function(
        ctx.compile_function[_noise_kernel](),
        dptr(splats.params[Attr.MEANS]),
        dptr(splats.params[Attr.LOG_SCALES]),
        dptr(splats.params[Attr.ROTATIONS]),
        dptr(splats.params[Attr.OPACITY_LOGITS]),
        Int32(n),
        random_ui64(0, UInt64.MAX),
        scale,
        grid_dim=num_blocks(n),
        block_dim=BLOCK_SIZE,
    )
    ctx.synchronize()


def _opacities(ctx: DeviceContext, splats: Splats) raises -> List[Float64]:
    var logits = download(ctx, splats.params[Attr.OPACITY_LOGITS], splats.count)
    var out = List[Float64](capacity=len(logits))
    for l in logits:
        out.append(1.0 / (1.0 + exp(-Float64(l))))
    return out^


def _sample(weights: List[Float64], indices: List[Int], count: Int) -> List[Int]:
    """`count` draws (with replacement) from `indices`, proportional to `weights`."""
    var cdf = List[Float64](capacity=len(weights))
    var total: Float64 = 0.0
    for w in weights:
        total += w
        cdf.append(total)
    var out = List[Int](capacity=count)
    for _ in range(count):
        var r = random_float64(0.0, total)
        var lo = 0
        var hi = len(cdf) - 1
        while lo < hi:
            var mid = (lo + hi) // 2
            if cdf[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        out.append(indices[lo])
    return out^


def _shared_opacities(opacity: List[Float64], sampled: List[Int]) -> List[Float32]:
    """New logit of each sampled row: opacity split among it and its copies."""
    var counts = List[Int](length=len(opacity), fill=0)
    for s in sampled:
        counts[s] += 1
    var out = List[Float32](capacity=len(sampled))
    for s in sampled:
        var shared = 1.0 - pow(1.0 - opacity[s], 1.0 / Float64(counts[s] + 1))
        out.append(_logit(min(max(shared, DEAD_OPACITY), MAX_OPACITY)))
    return out^


def _set_opacity(mut splats: Splats, rows: List[Int32], logits: List[Float32]) raises:
    var ctx = splats.ctx
    var r = upload(ctx, rows)
    var v = upload(ctx, logits)
    ctx.stream().enqueue_function(
        ctx.compile_function[_set_rows_kernel](),
        dptr(splats.params[Attr.OPACITY_LOGITS]),
        dptr(r),
        dptr(v),
        Int32(len(rows)),
        grid_dim=num_blocks(len(rows)),
        block_dim=BLOCK_SIZE,
    )
    ctx.synchronize()
    _ = (r^, v^)


def _as_i32(values: List[Int]) -> List[Int32]:
    var out = List[Int32](capacity=len(values))
    for v in values:
        out.append(Int32(v))
    return out^


def relocate(mut splats: Splats) raises -> Int:
    """Moves dead Gaussians onto sampled live ones. Returns how many moved."""
    var ctx = splats.ctx
    var opacity = _opacities(ctx, splats)
    var dead = List[Int]()
    var alive = List[Int]()
    var probs = List[Float64]()
    for g in range(len(opacity)):
        if opacity[g] <= DEAD_OPACITY:
            dead.append(g)
        else:
            alive.append(g)
            probs.append(opacity[g])
    if len(dead) == 0 or len(alive) == 0:
        return 0
    var sampled = _sample(probs, alive, len(dead))
    var logits = _shared_opacities(opacity, sampled)
    var dst = _as_i32(dead)
    var src = _as_i32(sampled)
    var d = upload(ctx, dst)
    var s = upload(ctx, src)
    var n = len(dead)
    for a in range(len(splats.columns)):
        var w = splats.width(a)
        if w == 0:
            continue
        ctx.stream().enqueue_function(
            ctx.compile_function[_copy_rows_kernel](),
            dptr(splats.params[a]),
            dptr(d),
            dptr(s),
            Int32(n),
            Int32(w),
            grid_dim=num_blocks(n * w),
            block_dim=BLOCK_SIZE,
        )
    ctx.synchronize()
    _set_opacity(splats, dst, logits)
    _set_opacity(splats, src, logits)
    splats.reset_moments(dptr(s), n)
    _ = (d^, s^)
    return n


def add_gaussians(mut splats: Splats, cap_max: Int) raises -> Int:
    """Adds 5% more Gaussians, sampled by opacity, up to `cap_max`."""
    var ctx = splats.ctx
    var n = splats.count
    var target = min(cap_max, Int(1.05 * Float64(n)))
    var added = max(0, target - n)
    if added == 0:
        return 0
    var opacity = _opacities(ctx, splats)
    var all_rows = List[Int](capacity=n)
    for g in range(n):
        all_rows.append(g)
    var sampled = _sample(opacity, all_rows, added)
    var logits = _shared_opacities(opacity, sampled)
    var index = List[Int32](capacity=n + added)
    for g in range(n):
        index.append(Int32(g))
    var new_rows = List[Int32](capacity=added)
    for i in range(added):
        index.append(Int32(sampled[i]))
        new_rows.append(Int32(n + i))
    var idx = upload(ctx, index)
    splats.gather(dptr(idx), n + added, n)
    ctx.synchronize()
    _ = idx^
    var src = _as_i32(sampled)
    _set_opacity(splats, src, logits)
    _set_opacity(splats, new_rows, logits)
    var s = upload(ctx, src)
    splats.reset_moments(dptr(s), added)
    _ = s^
    return added


struct RelocateConfig(Copyable, Defaultable, Movable):
    """Settings of `Relocation`."""

    @__annotation(Arg("Maximum number of Gaussians"))
    var cap_max: Int

    @__annotation(Arg("Relocate after this iteration"))
    var relocate_from: Int

    @__annotation(Arg("Relocate before this iteration"))
    var relocate_until: Int

    @__annotation(Arg("Relocate every N iterations"))
    var relocate_interval: Int

    @__annotation(Arg("Position noise strength at each relocation"))
    var noise_lr: Float32

    @__annotation(Arg("Weight of the mean opacity penalty"))
    var opacity_reg: Float32

    @__annotation(Arg("Weight of the mean scale penalty"))
    var scale_reg: Float32

    def __init__(out self):
        self.cap_max = 5_000_000
        self.relocate_from = 500
        self.relocate_until = 25_000
        self.relocate_interval = 100
        self.noise_lr = 5e4
        self.opacity_reg = 0.01
        self.scale_reg = 0.01


struct Relocation(Densification):
    """Relocation densification, on Beta Splatting's schedule. With t = it + 1
    (the official code counts from 1), inside relocate_from < t < relocate_until:

        t % relocate_interval == 0   relocate, add 5%, add position noise, and
                                     skip the optimizer step (as the official code)
        otherwise                    add the regularizer gradients, then step

    Nothing is pruned after training.
    """

    comptime Config = RelocateConfig

    var config: RelocateConfig

    def __init__(out self, ctx: DeviceContext, config: RelocateConfig) raises:
        self.config = config.copy()

    def before_step(mut self, it: Int, mut splats: Splats, scene_extent: Float32) raises:
        pass

    def collects_stats(self, it: Int) -> Bool:
        return False

    def after_backward(mut self, it: Int, mut splats: Splats, means_lr: Float32) raises -> Bool:
        var c = self.config.copy()
        var t = it + 1
        if t <= c.relocate_from or t >= c.relocate_until:
            return True
        if t % c.relocate_interval == 0:
            _ = relocate(splats)
            _ = add_gaussians(splats, c.cap_max)
            add_position_noise(splats, c.noise_lr * means_lr)
            return False
        add_regularizer_gradients(splats, c.opacity_reg, c.scale_reg)
        return True

    def finish(mut self, mut splats: Splats) raises -> Int:
        return 0
