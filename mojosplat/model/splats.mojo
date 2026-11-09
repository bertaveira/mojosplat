"""Splats: the owner of a set of Gaussians on the GPU.

Holds one array per `Column` with room for `capacity` Gaussians, of which the
first `count` are live. Core columns come first (means, log scales, rotations,
opacity logits); the rest belong to the appearance. Training state is
allocated only by `enable_training()`: per trainable column a gradient buffer
and the Adam moments (exp_avg, exp_avg_sq), plus the densification statistics.
The sh_rest gradient, the largest buffer, is optional: a fused optimizer can
recompute it (`Adam.step(splats, sh_rest)`). Operations that must touch every
column are one loop: `reserve` grows the capacity, and `gather` (rows by
index, optionally zeroing the moments of new rows) is the single row operation
behind clone / split, pruning and Morton reordering.

Mojo code passes `Splats` around (it keeps the memory alive); kernels get a
`SplatView` of raw pointers.
"""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import log

from ..core.math import SH_C0
from ..core.splats import ExtraColumns, NUM_EXTRA_COLUMNS, SplatView
from ..core.types import BLOCK_SIZE, F32Ptr, I32Ptr, dptr, num_blocks
from ..gpu.elementwise import copy_f32, fill_f32
from ..io.arrays import download
from ..io.ply import HostColumn, HostSplats
from ..io.splat import SplatScene


def _gather_rows_kernel(dst: F32Ptr, src: F32Ptr, index: I32Ptr, n_rows: Int32, width: Int32, zero_from: Int32):
    """dst[i, :] = src[index[i], :] for rows i < zero_from, 0 for the rest."""
    var e = block_idx.x * BLOCK_SIZE + thread_idx.x
    var w = Int(width)
    if e >= Int(n_rows) * w:
        return
    var row = e // w
    if row >= Int(zero_from):
        dst[unsafe_offset=e] = 0.0
    else:
        dst[unsafe_offset=e] = src[unsafe_offset=Int(index[unsafe_offset=row]) * w + e % w]


def _zero_rows_kernel(p: F32Ptr, rows: I32Ptr, n_rows: Int32, width: Int32):
    """p[rows[i], :] = 0."""
    var e = block_idx.x * BLOCK_SIZE + thread_idx.x
    if e >= Int(n_rows) * Int(width):
        return
    var row = UInt32(e) // UInt32(width)  # 32-bit: 64-bit division is very slow on Apple GPUs
    var k = UInt32(e) - row * UInt32(width)
    p[unsafe_offset=Int(rows[unsafe_offset=Int(row)]) * Int(width) + Int(k)] = 0.0


struct Attr:
    """Indices of the standard columns: the four core ones, then the columns
    of the spherical-harmonics appearance."""

    comptime MEANS = 0
    comptime LOG_SCALES = 1
    comptime ROTATIONS = 2
    comptime OPACITY_LOGITS = 3
    comptime NUM_CORE = 4
    comptime SH0 = 4
    comptime SH_REST = 5
    comptime COUNT = 6


@fieldwise_init
struct Column(ImplicitlyCopyable, Movable):
    """One per-Gaussian array of `Splats`."""

    var name: String
    var width: Int  # floats per Gaussian
    var trainable: Bool  # False: no gradient or Adam moments are allocated


def _upload_padded(
    ctx: DeviceContext, values: List[Float32], capacity_floats: Int
) raises -> DeviceBuffer[DType.float32]:
    var dev = ctx.enqueue_create_buffer[DType.float32](max(capacity_floats, 1))
    dev.enqueue_fill(0.0)
    if len(values) > 0:
        ctx.enqueue_copy(dev.create_sub_buffer[DType.float32](0, len(values)), Span(values))
        ctx.synchronize()  # `values` may be freed once we return
    return dev


struct Splats(Movable):
    var ctx: DeviceContext
    var count: Int
    var capacity: Int
    var rest_stride: Int  # SH bases per Gaussian in sh_rest (0, 3, 8 or 15)
    var columns: List[Column]
    var params: List[DeviceBuffer[DType.float32]]  # one per column, capacity * width each
    # Training state (empty until enable_training); one-float placeholders for non-trainable columns
    var training: Bool
    var has_sh_rest_grad: Bool  # False: grads[SH_REST] is a 1-float placeholder
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
        self.columns = List[Column]()
        self.params = List[DeviceBuffer[DType.float32]]()
        self.training = False
        self.has_sh_rest_grad = False
        self.grads = List[DeviceBuffer[DType.float32]]()
        self.exp_avg = List[DeviceBuffer[DType.float32]]()
        self.exp_avg_sq = List[DeviceBuffer[DType.float32]]()
        self.densification = ctx.enqueue_create_buffer[DType.float32](1)
        self.add_column("means", 3, values=means)
        self.add_column("log_scales", 3, values=log_scales)
        self.add_column("rotations", 4, values=rotations)
        self.add_column("opacity_logits", 1, values=opacity_logits)
        self.add_column("sh0", 3, values=sh0)
        self.add_column("sh_rest", rest_stride * 3, values=sh_rest)

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
        var s = Splats(
            ctx,
            h.means,
            h.log_scales,
            h.rotations,
            h.opacity_logits,
            h.sh0,
            h.sh_rest,
            h.rest_stride,
            capacity,
        )
        for c in h.extras:
            s.add_column(c.name, c.width, trainable=False, values=c.data)
        return s^

    def to_host(self) raises -> HostSplats:
        var h = HostSplats(0, self.rest_stride)  # every array replaced below
        h.count = self.count
        h.means = download(self.ctx, self.params[Attr.MEANS], self.count * 3)
        h.log_scales = download(self.ctx, self.params[Attr.LOG_SCALES], self.count * 3)
        h.rotations = download(self.ctx, self.params[Attr.ROTATIONS], self.count * 4)
        h.opacity_logits = download(self.ctx, self.params[Attr.OPACITY_LOGITS], self.count)
        h.sh0 = download(self.ctx, self.params[Attr.SH0], self.count * 3)
        if self.rest_stride > 0:
            h.sh_rest = download(self.ctx, self.params[Attr.SH_REST], self.count * self.rest_stride * 3)
        for a in range(Attr.COUNT, len(self.columns)):
            var w = self.columns[a].width
            h.extras.append(HostColumn(self.columns[a].name, w, download(self.ctx, self.params[a], self.count * w)))
        return h^

    def width(self, column: Int) -> Int:
        """Floats per Gaussian of a column."""
        return self.columns[column].width

    def add_column(
        mut self, name: String, width: Int, trainable: Bool = True, values: List[Float32] = List[Float32]()
    ) raises:
        """Add a per-Gaussian array after the existing ones. `values` holds
        count * width floats, or is empty for zeros."""
        if len(values) > 0 and len(values) != self.count * width:
            raise Error(String("add_column ", name, ": expected count * width values"))
        self.columns.append(Column(name, width, trainable))
        self.params.append(_upload_padded(self.ctx, values, self.capacity * width))
        if self.training:
            self._add_training_state(len(self.columns) - 1)

    def _view(self, bufs: List[DeviceBuffer[DType.float32]]) -> SplatView:
        var extras = ExtraColumns()
        for a in range(Attr.NUM_CORE, min(len(self.columns), Attr.NUM_CORE + NUM_EXTRA_COLUMNS)):
            extras.set(a - Attr.NUM_CORE, dptr(bufs[a]))
        return SplatView(
            dptr(bufs[Attr.MEANS]),
            dptr(bufs[Attr.LOG_SCALES]),
            dptr(bufs[Attr.ROTATIONS]),
            dptr(bufs[Attr.OPACITY_LOGITS]),
            extras,
            self.count,
            self.rest_stride,
        )

    def view(self) -> SplatView:
        """Raw pointers to the parameters (for kernels)."""
        return self._view(self.params)

    def grad_view(self) raises -> SplatView:
        """Raw pointers to the gradient buffers (laid out like the parameters).
        Without an sh_rest gradient buffer, the backward must run with
        `write_all=False`."""
        if not self.training:
            raise Error("Splats.grad_view: call enable_training() first")
        return self._view(self.grads)

    def densification_ptr(self) -> F32Ptr:
        return dptr(self.densification)

    def _zeros(self, n: Int) raises -> DeviceBuffer[DType.float32]:
        var b = self.ctx.enqueue_create_buffer[DType.float32](max(n, 1))
        fill_f32(self.ctx, dptr(b), 0.0, max(n, 1))
        return b

    def _has_moments(self, column: Int) -> Bool:
        return self.columns[column].trainable

    def _grad_floats(self, column: Int, rows: Int) -> Int:
        if not self.columns[column].trainable or (column == Attr.SH_REST and not self.has_sh_rest_grad):
            return 1
        return max(rows * self.width(column), 1)

    def _moment_floats(self, column: Int, rows: Int) -> Int:
        return rows * self.width(column) if self._has_moments(column) else 1

    def _add_training_state(mut self, column: Int) raises:
        var n = self._moment_floats(column, self.capacity)
        self.grads.append(self._zeros(self._grad_floats(column, self.capacity)))
        self.exp_avg.append(self._zeros(n))
        self.exp_avg_sq.append(self._zeros(n))

    def enable_training(mut self, sh_rest_grad: Bool = True) raises:
        """Allocate gradients, Adam moments (zeroed) and densification stats.
        With `sh_rest_grad=False` there is no sh_rest gradient buffer (for an
        optimizer that recomputes it, see `Adam.step(splats, sh_rest)`)."""
        if self.training:
            return
        self.has_sh_rest_grad = sh_rest_grad
        for a in range(len(self.columns)):
            self._add_training_state(a)
        self.densification = self._zeros(2 * self.capacity)
        self.training = True

    def reset_densification(mut self) raises:
        fill_f32(self.ctx, dptr(self.densification), 0.0, 2 * self.count)

    def reset_moments(mut self, rows: I32Ptr, n_rows: Int) raises:
        """Zero the Adam moments of the given rows in every trainable column."""
        if not self.training or n_rows == 0:
            return
        var kernel = self.ctx.compile_function[_zero_rows_kernel]()
        for a in range(len(self.columns)):
            if not self._has_moments(a):
                continue
            var w = self.width(a)
            if w == 0:
                continue
            for moments in [dptr(self.exp_avg[a]), dptr(self.exp_avg_sq[a])]:
                self.ctx.stream().enqueue_function(
                    kernel,
                    moments,
                    rows,
                    Int32(n_rows),
                    Int32(w),
                    grid_dim=num_blocks(n_rows * w),
                    block_dim=BLOCK_SIZE,
                )
        self.ctx.synchronize()

    def reserve(mut self, capacity: Int) raises:
        """Grow every per-Gaussian array to hold `capacity` Gaussians,
        keeping the first `count` rows (new rows are zero)."""
        if capacity <= self.capacity:
            return
        for a in range(len(self.columns)):
            var w = self.width(a)
            var grown = self._zeros(capacity * w)
            copy_f32(self.ctx, dptr(grown), dptr(self.params[a]), self.count * w)
            self.params[a] = grown
            if self.training and self._has_moments(a):
                var g = self._zeros(self._grad_floats(a, capacity))
                var m = self._zeros(capacity * w)
                var v = self._zeros(capacity * w)
                copy_f32(self.ctx, dptr(m), dptr(self.exp_avg[a]), self.count * w)
                copy_f32(self.ctx, dptr(v), dptr(self.exp_avg_sq[a]), self.count * w)
                self.grads[a] = g
                self.exp_avg[a] = m
                self.exp_avg_sq[a] = v
        if self.training:
            var d = self._zeros(2 * capacity)
            self.densification = d
        self.capacity = capacity

    def _gather_into(self, dst: F32Ptr, src: F32Ptr, index: I32Ptr, n_out: Int, width: Int, zero_from: Int) raises:
        if n_out * width > 0:
            self.ctx.stream().enqueue_function(
                self.ctx.compile_function[_gather_rows_kernel](),
                dst,
                src,
                index,
                Int32(n_out),
                Int32(width),
                Int32(zero_from),
                grid_dim=num_blocks(n_out * width),
                block_dim=BLOCK_SIZE,
            )

    def _gathered(
        self, src: DeviceBuffer[DType.float32], index: I32Ptr, n_out: Int, width: Int, zero_from: Int
    ) raises -> DeviceBuffer[DType.float32]:
        var dst = self.ctx.enqueue_create_buffer[DType.float32](max(n_out * width, 1))
        self._gather_into(dptr(dst), dptr(src), index, n_out, width, zero_from)
        return dst

    def gather(mut self, index: I32Ptr, n_out: Int, zero_moments_from: Int, keep_statistics: Bool = False) raises:
        """Rebuild every per-Gaussian array: row i becomes old row index[i].

        The one primitive behind densification: append (clone / split:
        index = [0..n) ++ copies), compact (prune: index = kept rows) and
        permute (reordering). Adam moments of rows >= `zero_moments_from`
        are zeroed (new Gaussians start with fresh optimizer state);
        gradients are reallocated (the next backward refills them).
        Densification statistics are zeroed, unless `keep_statistics` (for a
        permutation) moves them with their rows.
        """
        # Synchronize after each array: freed memory returns to the pool only
        # once the GPU is done with it, and without the syncs the old and new
        # arrays are all alive at once (out of memory on 10 GB GPUs).
        for a in range(len(self.columns)):
            var w = self.width(a)
            self.params[a] = self._gathered(self.params[a], index, n_out, w, n_out)
            self.ctx.synchronize()
            if self.training and self._has_moments(a):
                self.exp_avg[a] = self._gathered(self.exp_avg[a], index, n_out, w, zero_moments_from)
                self.ctx.synchronize()
                self.exp_avg_sq[a] = self._gathered(self.exp_avg_sq[a], index, n_out, w, zero_moments_from)
                self.ctx.synchronize()
                self.grads[a] = self.ctx.enqueue_create_buffer[DType.float32](1)  # free the old one first
                self.ctx.synchronize()
                self.grads[a] = self.ctx.enqueue_create_buffer[DType.float32](self._grad_floats(a, n_out))
        if self.training:
            if keep_statistics:
                var d = self.ctx.enqueue_create_buffer[DType.float32](2 * max(n_out, 1))
                var old = dptr(self.densification)
                self._gather_into(dptr(d), old, index, n_out, 1, n_out)
                self._gather_into(dptr(d).unsafe_offset(n_out), old.unsafe_offset(self.count), index, n_out, 1, n_out)
                self.densification = d
                self.ctx.synchronize()
            else:
                self.densification = self._zeros(2 * max(n_out, 1))
        self.count = n_out
        self.capacity = max(n_out, 1)
