"""Relocation densification, regularizers and position noise.

The relocation of 3DGS-MCMC (Kheradmand et al., "3D Gaussian Splatting as
Markov Chain Monte Carlo", NeurIPS 2024). Every densification step, Gaussians
with opacity <= 0.005 are moved onto live ones, sampled by opacity; then 5% more
are added the same way, up to `cap_max`. A sampled Gaussian with opacity o and
its copies each get 1 - (1 - o)^(1 / (copies + 1)), so together they keep its
opacity, and its Adam moments are reset. Nothing is cloned by gradient, split
or pruned.

Two strategies use it:

    Relocation       Beta Splatting's schedule: position noise only at relocations
    MCMCRelocation   the paper's: copies also shrink (its Equation 9), and position
                     noise and the regularizers act on every iteration

Sampling is on the host (every 100 iterations): one multinomial draw per moved
Gaussian from a cumulative table.
"""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import exp, log, pow, sqrt
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
comptime MCMC_N_MAX = 51  # Equation 9 is evaluated for at most this many copies


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
    sigmoid: Int32,
):
    """means += Cov * z * f(1 - opacity) * scale, z ~ N(0, I).

    f(x) is x^100 (Beta Splatting) or 1 / (1 + exp(-100 (x - 0.995))) (3DGS-MCMC).
    """
    var g = block_idx.x * BLOCK_SIZE + thread_idx.x
    if g >= Int(n):
        return
    comptime dt = DType.float32
    var rng = NormalRandom(seed=seed, subsequence=UInt64(g))
    var z = rng.step_normal()
    var one_minus = 1.0 - 1.0 / (1.0 + exp(-opacity_logits[unsafe_offset=g]))
    var shape: Float32
    if sigmoid != 0:
        shape = 1.0 / (1.0 + exp(-100.0 * (one_minus - 0.995)))
    else:
        shape = exp(100.0 * log(max(one_minus, 1e-30)))
    var factor = shape * scale
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


def add_position_noise(mut splats: Splats, scale: Float32, sigmoid: Bool = False) raises:
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
        Int32(1 if sigmoid else 0),
        grid_dim=num_blocks(n),
        block_dim=BLOCK_SIZE,
    )


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


def _binomials() -> List[Float64]:
    """Pascal's triangle, row-major in a MCMC_N_MAX x MCMC_N_MAX table."""
    comptime M = MCMC_N_MAX
    var b = List[Float64](length=M * M, fill=0.0)
    for n in range(M):
        b[n * M] = 1.0
        for k in range(1, n + 1):
            b[n * M + k] = b[(n - 1) * M + k - 1] + b[(n - 1) * M + k]
    return b^


def mcmc_scale_ratio(opacity_old: Float64, opacity_new: Float64, n: Int, binomials: List[Float64]) -> Float64:
    """Equation 9 of 3DGS-MCMC: new scale / old scale when a Gaussian becomes `n` copies
    of opacity `opacity_new` each, so that together they render as it did."""
    var total: Float64 = 0.0
    for i in range(1, n + 1):
        var sign: Float64 = 1.0
        var power = opacity_new
        for k in range(i):
            total += binomials[(i - 1) * MCMC_N_MAX + k] * sign / sqrt(Float64(k + 1)) * power
            sign = -sign
            power *= opacity_new
    return opacity_old / total


def _copies(opacity: List[Float64], sampled: List[Int], cap: Int) -> List[Int]:
    """How many Gaussians each sampled row becomes (itself and its copies), at most `cap`."""
    var counts = List[Int](length=len(opacity), fill=0)
    for s in sampled:
        counts[s] += 1
    var out = List[Int](capacity=len(sampled))
    for s in sampled:
        out.append(min(counts[s] + 1, cap))
    return out^


def _shared_opacities(opacity: List[Float64], sampled: List[Int], copies: List[Int]) -> List[Float32]:
    """New logit of each sampled row: opacity split among it and its copies."""
    var out = List[Float32](capacity=len(sampled))
    for i in range(len(sampled)):
        var shared = 1.0 - pow(1.0 - opacity[sampled[i]], 1.0 / Float64(copies[i]))
        out.append(_logit(min(max(shared, DEAD_OPACITY), MAX_OPACITY)))
    return out^


def _log_scale_shifts(opacity: List[Float64], sampled: List[Int], copies: List[Int]) -> List[Float32]:
    var binomials = _binomials()
    var out = List[Float32](capacity=len(sampled))
    for i in range(len(sampled)):
        var old = opacity[sampled[i]]
        var new = 1.0 - pow(1.0 - old, 1.0 / Float64(copies[i]))
        out.append(Float32(log(mcmc_scale_ratio(old, new, copies[i], binomials))))
    return out^


def _shift_rows_kernel(log_scales: F32Ptr, rows: I32Ptr, shifts: F32Ptr, n_floats: Int32):
    var e = block_idx.x * BLOCK_SIZE + thread_idx.x
    if e < Int(n_floats):
        var i = Int(UInt32(e) // 3)
        log_scales[unsafe_offset=Int(rows[unsafe_offset=i]) * 3 + e - i * 3] += shifts[unsafe_offset=i]


def _shrink_copies(mut splats: Splats, copies: List[Int32], sources: List[Int32], shifts: List[Float32]) raises:
    """Applies the shift to the copies, then gives each source its copy's scale.

    A source can appear several times, so it is assigned, not shifted again.
    """
    var ctx = splats.ctx
    var c = upload(ctx, copies)
    var s = upload(ctx, sources)
    var v = upload(ctx, shifts)
    var n = len(copies)
    ctx.stream().enqueue_function(
        ctx.compile_function[_shift_rows_kernel](),
        dptr(splats.params[Attr.LOG_SCALES]),
        dptr(c),
        dptr(v),
        Int32(n * 3),
        grid_dim=num_blocks(n * 3),
        block_dim=BLOCK_SIZE,
    )
    ctx.synchronize()
    ctx.stream().enqueue_function(
        ctx.compile_function[_copy_rows_kernel](),
        dptr(splats.params[Attr.LOG_SCALES]),
        dptr(s),
        dptr(c),
        Int32(n),
        Int32(3),
        grid_dim=num_blocks(n * 3),
        block_dim=BLOCK_SIZE,
    )
    ctx.synchronize()
    _ = (c^, s^, v^)


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


def relocate(mut splats: Splats, shrink_scales: Bool = False) raises -> Int:
    """Moves dead Gaussians onto sampled live ones. Returns how many moved.
    With `shrink_scales`, a source and its copies get the scale of Equation 9."""
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
    var copies = _copies(opacity, sampled, MCMC_N_MAX - 1 if shrink_scales else Int.MAX)
    var logits = _shared_opacities(opacity, sampled, copies)
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
    if shrink_scales:
        _shrink_copies(splats, dst, src, _log_scale_shifts(opacity, sampled, copies))
    splats.reset_moments(dptr(s), n)
    _ = (d^, s^)
    return n


def add_gaussians(mut splats: Splats, cap_max: Int, shrink_scales: Bool = False) raises -> Int:
    """Adds 5% more Gaussians, sampled by opacity, up to `cap_max`.
    With `shrink_scales`, a source and its copies get the scale of Equation 9."""
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
    var copies = _copies(opacity, sampled, MCMC_N_MAX - 1 if shrink_scales else Int.MAX)
    var logits = _shared_opacities(opacity, sampled, copies)
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
    if shrink_scales:
        _shrink_copies(splats, new_rows, src, _log_scale_shifts(opacity, sampled, copies))
    var s = upload(ctx, src)
    splats.reset_moments(dptr(s), added)
    _ = s^
    return added


struct RelocateConfig(Copyable, Defaultable, Movable):
    """Settings of `Relocation` and `MCMCRelocation`. The defaults are Beta Splatting's."""

    @__annotation(Arg("Maximum number of Gaussians"))
    var cap_max: Int

    @__annotation(Arg("Relocate after this iteration"))
    var relocate_from: Int

    @__annotation(Arg("Relocate before this iteration"))
    var relocate_until: Int

    @__annotation(Arg("Relocate every N iterations"))
    var relocate_interval: Int

    @__annotation(Arg("Position noise strength (times the means learning rate)"))
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

    def __init__(out self, ctx: DeviceContext, config: RelocateConfig, iterations: Int) raises:
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
            splats.ctx.synchronize()
            return False
        add_regularizer_gradients(splats, c.opacity_reg, c.scale_reg)
        return True

    def finish(mut self, mut splats: Splats) raises -> Int:
        return 0


struct MCMCRelocation(Densification):
    """Relocation as in 3DGS-MCMC's training loop. With t = it + 1 (the official
    code counts from 1) and T the number of iterations:

        relocate_from < t < relocate_until and t % relocate_interval == 0:
            relocate and add 5%, shrinking the copies (Equation 9), and skip the
            optimizer step (the official code replaces the parameter tensors, so
            they have no gradient that iteration)
        otherwise:
            add the regularizer gradients; the optimizer steps unless t == T
        after the optimizer, while t < T:
            position noise, scaled by a sigmoid of (1 - opacity)

    Nothing is pruned after training.
    """

    comptime Config = RelocateConfig

    var config: RelocateConfig
    var iterations: Int

    def __init__(out self, ctx: DeviceContext, config: RelocateConfig, iterations: Int) raises:
        self.config = config.copy()
        self.iterations = iterations

    def before_step(mut self, it: Int, mut splats: Splats, scene_extent: Float32) raises:
        pass

    def collects_stats(self, it: Int) -> Bool:
        return False

    def after_backward(mut self, it: Int, mut splats: Splats, means_lr: Float32) raises -> Bool:
        var c = self.config.copy()
        var t = it + 1
        if t > c.relocate_from and t < c.relocate_until and t % c.relocate_interval == 0:
            _ = relocate(splats, shrink_scales=True)
            _ = add_gaussians(splats, c.cap_max, shrink_scales=True)
            return False
        add_regularizer_gradients(splats, c.opacity_reg, c.scale_reg)
        return t < self.iterations

    def after_step(mut self, it: Int, mut splats: Splats, means_lr: Float32) raises:
        if it + 1 < self.iterations:
            add_position_noise(splats, self.config.noise_lr * means_lr, sigmoid=True)

    def finish(mut self, mut splats: Splats) raises -> Int:
        return 0
