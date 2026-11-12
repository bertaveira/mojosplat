"""Classic 3DGS adaptive density control (as in Faster-GS's
`adaptive_density_control`, Apache-2.0; originally Kerbl et al. 2023).

Every densification step, from the statistics the backward pass accumulates
(per Gaussian: number of views it was visible in, and the summed norm of
its screen-space mean gradient in NDC):

1. select Gaussians whose mean gradient is >= `grad_threshold`;
2. clone the selected small ones (largest scale <= percent_dense * extent):
   an exact copy;
3. split the selected large ones into two children: each child's mean is
   moved by a sample of the parent's Gaussian (R diag(s) z, z ~ N(0, I)) and
   its scales are divided by 1.6; the parent is removed;
4. prune: split parents, opacity < `min_opacity`, degenerate rotations and,
   after the first opacity reset, Gaussians larger than 10% of the extent.

All row bookkeeping goes through `Splats.gather` (new rows get zeroed Adam
moments, pruned rows drop theirs). The random normals for the split come
from a counter-based RNG (Philox, one stream per child) seeded from the host
RNG, so a step is reproducible bit-for-bit for a given `random.seed`.
"""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.math import exp, log, sqrt
from std.random import random_ui64
from std.random.philox import NormalRandom

from ..cli.args import Arg
from ..core.types import BLOCK_SIZE, F32Ptr, I32Ptr, U32Ptr, dptr, num_blocks
from ..gpu.elementwise import fill_f32, fill_u32
from ..gpu.scan import exclusive_scan, scan_workspace_size
from ..io.arrays import upload
from ..model.splats import Attr, Splats
from .strategy import Densification


@always_inline
def _max3(p: F32Ptr, g: Int) -> Float32:
    return max(p[unsafe_offset=g * 3], max(p[unsafe_offset=g * 3 + 1], p[unsafe_offset=g * 3 + 2]))


def _classify_kernel(
    log_scales: F32Ptr,
    info: F32Ptr,
    dup: U32Ptr,
    split: U32Ptr,
    n: Int32,
    grad_threshold: Float32,
    log_small: Float32,
):
    var g = block_idx.x * BLOCK_SIZE + thread_idx.x
    if g >= Int(n):
        return
    var count = max(info[unsafe_offset=g], 1.0)
    var selected = info[unsafe_offset=Int(n) + g] >= grad_threshold * count
    var small = _max3(log_scales, g) <= log_small
    dup[unsafe_offset=g] = 1 if selected and small else 0
    split[unsafe_offset=g] = 1 if selected and not small else 0


def _build_index_kernel(
    dup: U32Ptr,
    dup_off: U32Ptr,
    split: U32Ptr,
    split_off: U32Ptr,
    index: I32Ptr,
    n: Int32,
    n_dup: Int32,
    n_split: Int32,
):
    """index = [0..n) ++ cloned rows ++ split rows ++ split rows."""
    var g = block_idx.x * BLOCK_SIZE + thread_idx.x
    if g >= Int(n):
        return
    index[unsafe_offset=g] = Int32(g)
    if dup[unsafe_offset=g] != 0:
        index[unsafe_offset=Int(n) + Int(dup_off[unsafe_offset=g])] = Int32(g)
    if split[unsafe_offset=g] != 0:
        var o = Int(n) + Int(n_dup) + Int(split_off[unsafe_offset=g])
        index[unsafe_offset=o] = Int32(g)
        index[unsafe_offset=o + Int(n_split)] = Int32(g)


def _split_children_kernel(
    means: F32Ptr,
    log_scales: F32Ptr,
    rotations: F32Ptr,
    randn: F32Ptr,
    use_randn: Int32,
    seed: UInt64,
    first: Int32,
    count: Int32,
    log_shrink: Float32,
):
    """Move child rows [first, first + count) by R diag(s) z, shrink scales.
    z ~ N(0, I) from Philox (stream `j` of `seed`), or read from `randn`
    when `use_randn` != 0 (tests)."""
    var j = block_idx.x * BLOCK_SIZE + thread_idx.x
    if j >= Int(count):
        return
    var r = Int(first) + j
    var zn: SIMD[DType.float32, 8]
    if use_randn != 0:
        zn = SIMD[DType.float32, 8](0.0)
        comptime for k in range(3):
            zn[k] = randn[unsafe_offset=j * 3 + k]
    else:
        var rng = NormalRandom(seed=seed, subsequence=UInt64(j))
        zn = rng.step_normal()
    var s0 = exp(log_scales[unsafe_offset=r * 3 + 0]) * zn[0]
    var s1 = exp(log_scales[unsafe_offset=r * 3 + 1]) * zn[1]
    var s2 = exp(log_scales[unsafe_offset=r * 3 + 2]) * zn[2]
    var w = rotations[unsafe_offset=r * 4 + 0]
    var x = rotations[unsafe_offset=r * 4 + 1]
    var y = rotations[unsafe_offset=r * 4 + 2]
    var z = rotations[unsafe_offset=r * 4 + 3]
    var inv = 1.0 / sqrt(w * w + x * x + y * y + z * z)
    w *= inv
    x *= inv
    y *= inv
    z *= inv
    means[unsafe_offset=r * 3 + 0] += (
        (1 - 2 * (y * y + z * z)) * s0 + 2 * (x * y - w * z) * s1 + 2 * (x * z + w * y) * s2
    )
    means[unsafe_offset=r * 3 + 1] += (
        2 * (x * y + w * z) * s0 + (1 - 2 * (x * x + z * z)) * s1 + 2 * (y * z - w * x) * s2
    )
    means[unsafe_offset=r * 3 + 2] += (
        2 * (x * z - w * y) * s0 + 2 * (y * z + w * x) * s1 + (1 - 2 * (x * x + y * y)) * s2
    )
    comptime for k in range(3):
        log_scales[unsafe_offset=r * 3 + k] += log_shrink


def _prune_flags_kernel(
    log_scales: F32Ptr,
    rotations: F32Ptr,
    opacity_logits: F32Ptr,
    split: U32Ptr,
    keep: U32Ptr,
    n_out: Int32,
    n_orig: Int32,
    min_opacity_logit: Float32,
    prune_large: Int32,
    log_max_scale: Float32,
):
    var g = block_idx.x * BLOCK_SIZE + thread_idx.x
    if g >= Int(n_out):
        return
    var prune = g < Int(n_orig) and split[unsafe_offset=g] != 0
    prune = prune or opacity_logits[unsafe_offset=g] < min_opacity_logit
    var q2: Float32 = 0.0
    comptime for k in range(4):
        q2 += rotations[unsafe_offset=g * 4 + k] * rotations[unsafe_offset=g * 4 + k]
    prune = prune or q2 < 1e-8
    if prune_large != 0:
        prune = prune or _max3(log_scales, g) > log_max_scale
    keep[unsafe_offset=g] = 0 if prune else 1


def _compact_index_kernel(keep: U32Ptr, off: U32Ptr, index: I32Ptr, n: Int32):
    var g = block_idx.x * BLOCK_SIZE + thread_idx.x
    if g < Int(n) and keep[unsafe_offset=g] != 0:
        index[unsafe_offset=Int(off[unsafe_offset=g])] = Int32(g)


def _cap_kernel(p: F32Ptr, n: Int32, cap: Float32):
    var g = block_idx.x * BLOCK_SIZE + thread_idx.x
    if g < Int(n):
        p[unsafe_offset=g] = min(p[unsafe_offset=g], cap)


def _logit(p: Float64) -> Float32:
    return Float32(log(p / (1.0 - p)))


struct DensifyStats(ImplicitlyCopyable, Movable):
    var cloned: Int
    var split: Int
    var pruned: Int
    var count: Int  # Gaussians after the step

    def __init__(out self):
        self.cloned = 0
        self.split = 0
        self.pruned = 0
        self.count = 0


struct Densifier(Movable):
    """Classic densification with Faster-GS's defaults."""

    var ctx: DeviceContext
    var grad_threshold: Float32
    var percent_dense: Float32
    var min_opacity: Float64
    var max_scale_fraction: Float32  # prune Gaussians larger than this * extent

    def __init__(out self, ctx: DeviceContext):
        self.ctx = ctx
        self.grad_threshold = 0.0002
        self.percent_dense = 0.01
        self.min_opacity = 0.005
        self.max_scale_fraction = 0.1

    def _scan(self, flags: U32Ptr, offsets: U32Ptr, n: Int) raises -> Int:
        """Exclusive scan of 0/1 flags; returns the total (syncs)."""
        var partials = self.ctx.enqueue_create_buffer[DType.uint32](scan_workspace_size(max(n, 1)))
        var total = self.ctx.enqueue_create_buffer[DType.uint32](1)
        exclusive_scan[DType.uint32](self.ctx, self.ctx.stream(), flags, offsets, dptr(total), dptr(partials), n)
        var host = self.ctx.enqueue_create_host_buffer[DType.uint32](1)
        self.ctx.enqueue_copy(host, total)
        self.ctx.synchronize()
        _ = partials^  # keep the scan scratch alive until the kernels finished
        return Int(host[0])

    def densify(
        mut self,
        mut splats: Splats,
        scene_extent: Float32,
        prune_large: Bool,
        randn_override: List[Float32] = List[Float32](),
    ) raises -> DensifyStats:
        """One clone / split / prune step. Uses (and does not reset) the
        densification statistics in `splats`. Split offsets are drawn on the
        GPU (Philox, seeded from `std.random`, so `random.seed` makes runs
        repeatable); `randn_override` (3 normals per split child) replaces
        them, for testing."""
        var ctx = self.ctx
        var stream = ctx.stream()
        var n = splats.count
        var st = DensifyStats()
        var dup = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
        var split = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
        var dup_off = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
        var split_off = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
        stream.enqueue_function(
            ctx.compile_function[_classify_kernel](),
            dptr(splats.params[Attr.LOG_SCALES]),
            splats.densification_ptr(),
            dptr(dup),
            dptr(split),
            Int32(n),
            self.grad_threshold,
            Float32(log(Float64(self.percent_dense * scene_extent))),
            grid_dim=num_blocks(n),
            block_dim=BLOCK_SIZE,
        )
        st.cloned = self._scan(dptr(dup), dptr(dup_off), n)
        st.split = self._scan(dptr(split), dptr(split_off), n)

        # Append clones and split children (zeroed moments for new rows).
        var n_out = n + st.cloned + 2 * st.split
        var index = ctx.enqueue_create_buffer[DType.int32](max(n_out, 1))
        stream.enqueue_function(
            ctx.compile_function[_build_index_kernel](),
            dptr(dup),
            dptr(dup_off),
            dptr(split),
            dptr(split_off),
            dptr(index),
            Int32(n),
            Int32(st.cloned),
            Int32(st.split),
            grid_dim=num_blocks(n),
            block_dim=BLOCK_SIZE,
        )
        splats.gather(dptr(index), n_out, n)
        var n_children = 2 * st.split
        if n_children > 0:
            var use_override = len(randn_override) > 0
            if use_override and len(randn_override) < 3 * n_children:
                raise Error("randn_override needs 3 values per split child")
            var dz = upload(ctx, randn_override)  # 1 element when unused
            stream.enqueue_function(
                ctx.compile_function[_split_children_kernel](),
                dptr(splats.params[Attr.MEANS]),
                dptr(splats.params[Attr.LOG_SCALES]),
                dptr(splats.params[Attr.ROTATIONS]),
                dptr(dz),
                Int32(1 if use_override else 0),
                random_ui64(0, UInt64.MAX),
                Int32(n + st.cloned),
                Int32(n_children),
                Float32(log(0.625)),
                grid_dim=num_blocks(n_children),
                block_dim=BLOCK_SIZE,
            )
            ctx.synchronize()
            _ = dz^

        # Prune (keeps order; moments follow their rows).
        var keep = ctx.enqueue_create_buffer[DType.uint32](max(n_out, 1))
        var keep_off = ctx.enqueue_create_buffer[DType.uint32](max(n_out, 1))
        stream.enqueue_function(
            ctx.compile_function[_prune_flags_kernel](),
            dptr(splats.params[Attr.LOG_SCALES]),
            dptr(splats.params[Attr.ROTATIONS]),
            dptr(splats.params[Attr.OPACITY_LOGITS]),
            dptr(split),
            dptr(keep),
            Int32(n_out),
            Int32(n),
            _logit(self.min_opacity),
            Int32(1 if prune_large else 0),
            Float32(log(Float64(self.max_scale_fraction * scene_extent))),
            grid_dim=num_blocks(n_out),
            block_dim=BLOCK_SIZE,
        )
        var n_keep = self._scan(dptr(keep), dptr(keep_off), n_out)
        var keep_index = ctx.enqueue_create_buffer[DType.int32](max(n_keep, 1))
        stream.enqueue_function(
            ctx.compile_function[_compact_index_kernel](),
            dptr(keep),
            dptr(keep_off),
            dptr(keep_index),
            Int32(n_out),
            grid_dim=num_blocks(n_out),
            block_dim=BLOCK_SIZE,
        )
        splats.gather(dptr(keep_index), n_keep, n_keep)
        ctx.synchronize()
        st.pruned = n_out - n_keep
        st.count = n_keep
        _ = (dup^, split^, dup_off^, split_off^, index^, keep^, keep_off^, keep_index^)
        return st

    def reset_opacities(mut self, mut splats: Splats, max_opacity: Float64 = 0.01) raises:
        """Clamp opacities to at most `max_opacity`; reset their Adam moments."""
        var n = splats.count
        var ctx = self.ctx
        ctx.stream().enqueue_function(
            ctx.compile_function[_cap_kernel](),
            dptr(splats.params[Attr.OPACITY_LOGITS]),
            Int32(n),
            _logit(max_opacity),
            grid_dim=num_blocks(n),
            block_dim=BLOCK_SIZE,
        )
        fill_f32(ctx, dptr(splats.exp_avg[Attr.OPACITY_LOGITS]), 0.0, n)
        fill_f32(ctx, dptr(splats.exp_avg_sq[Attr.OPACITY_LOGITS]), 0.0, n)

    def prune_final(mut self, mut splats: Splats, min_opacity: Float64 = 1.0 / 255.0) raises -> Int:
        """After training: drop nearly transparent / degenerate Gaussians."""
        var n = splats.count
        var ctx = self.ctx
        var split = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
        fill_u32(ctx, dptr(split), 0, n)
        var keep = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
        var keep_off = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
        ctx.stream().enqueue_function(
            ctx.compile_function[_prune_flags_kernel](),
            dptr(splats.params[Attr.LOG_SCALES]),
            dptr(splats.params[Attr.ROTATIONS]),
            dptr(splats.params[Attr.OPACITY_LOGITS]),
            dptr(split),
            dptr(keep),
            Int32(n),
            Int32(n),
            _logit(min_opacity),
            Int32(0),
            Float32(0.0),
            grid_dim=num_blocks(n),
            block_dim=BLOCK_SIZE,
        )
        var n_keep = self._scan(dptr(keep), dptr(keep_off), n)
        var keep_index = ctx.enqueue_create_buffer[DType.int32](max(n_keep, 1))
        ctx.stream().enqueue_function(
            ctx.compile_function[_compact_index_kernel](),
            dptr(keep),
            dptr(keep_off),
            dptr(keep_index),
            Int32(n),
            grid_dim=num_blocks(n),
            block_dim=BLOCK_SIZE,
        )
        splats.gather(dptr(keep_index), n_keep, n_keep)
        ctx.synchronize()
        _ = (split^, keep^, keep_off^, keep_index^)
        return n - n_keep


struct DensifyConfig(Copyable, Defaultable, Movable):
    """Settings of `ClassicDensification`; defaults are Faster-GS's."""

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

    def __init__(out self):
        self.densify = True
        self.densify_from = 600
        self.densify_until = 14_900
        self.densify_interval = 100
        self.opacity_reset_interval = 3000
        self.grad_threshold = 0.0002
        self.percent_dense = 0.01


struct ClassicDensification(Densification):
    """3DGS's adaptive density control, on Faster-GS's schedule (iteration `it`, 0-based):

        densify           if densify_from <= it <= densify_until and
                          (it - densify_from) % densify_interval == 0
        reset opacities   if it % opacity_reset_interval == 0, it >= opacity_reset_interval,
                          and it is inside the densification window above

    The backward collects statistics until `densify_until`. After training,
    nearly transparent and degenerate Gaussians are pruned.
    """

    comptime Config = DensifyConfig

    var config: DensifyConfig
    var densifier: Densifier
    var last: DensifyStats

    def __init__(out self, ctx: DeviceContext, config: DensifyConfig) raises:
        self.config = config.copy()
        self.densifier = Densifier(ctx)
        self.densifier.grad_threshold = config.grad_threshold
        self.densifier.percent_dense = config.percent_dense
        self.last = DensifyStats()

    def before_step(mut self, it: Int, mut splats: Splats, scene_extent: Float32) raises:
        var c = self.config.copy()
        if not c.densify or it < c.densify_from or it > c.densify_until:
            return
        if (it - c.densify_from) % c.densify_interval == 0:
            self.last = self.densifier.densify(splats, scene_extent, it > c.opacity_reset_interval)
            if it < c.densify_until:
                splats.reset_densification()
        if it >= c.opacity_reset_interval and it % c.opacity_reset_interval == 0:
            self.densifier.reset_opacities(splats)

    def collects_stats(self, it: Int) -> Bool:
        return self.config.densify and it < self.config.densify_until

    def after_backward(mut self, it: Int, mut splats: Splats, means_lr: Float32) raises -> Bool:
        return True

    def finish(mut self, mut splats: Splats) raises -> Int:
        return self.densifier.prune_final(splats)
