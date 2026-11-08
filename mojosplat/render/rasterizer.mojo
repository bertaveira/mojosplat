"""Faster-GS style rasterizer: forward (inference and training) and backward.

Inference (`render`) and training (`forward` + `backward`) share the
pipeline (one host sync, to size the instance buffers):

  preprocess       activations, EWA projection, exact tile count, SH -> RGB,
                   compacted (depth, id) list of visible Gaussians
  depth sort       radix sort of the visible Gaussians by depth (32-bit keys)
  instances        exclusive scan of tile counts in depth order, then one
                   (tile key, id) per Gaussian-tile pair
  tile sort        stable radix sort by tile only (16-bit keys up to 64k tiles)
  tile offsets     per-tile instance ranges
  blend            per-tile front-to-back compositing with sub-tile skipping

`forward` additionally saves the state the per-Gaussian backward needs
(see blend.mojo / blend_backward.mojo) into buffers obtained from a
`StateAllocator`, and returns it as a `ForwardState`, so several forwards can
be in flight before their backward passes (e.g. torch autograd).

Takes the *raw* (trainable) parameters as in 3DGS/gsplat training: log
scales, unnormalized quaternions, opacity logits and SH coefficients.
Design and kernels follow Faster-GS (Hahlbohm et al., CVPR 2026, Apache-2.0).
"""

from max.gpu import WARP_SIZE
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.memory import ArcPointer
from std.time import perf_counter_ns

from ..core.splats import SplatView
from ..gpu.arena import DeviceArena, StateAllocator
from ..gpu.elementwise import fill_f32
from ..model.splats import Splats
from .blend_backward import blend_backward
from .preprocess_backward import ShRestGrad, preprocess_backward
from .blend import blend, bucket_bound, bucket_counts
from ..core.camera import CAM_POS_X, CAM_POS_Y, CAM_POS_Z, Camera
from ..core.types import F32Ptr, I32Ptr, U16Ptr, U32Ptr, dptr, placeholder
from .instances import create_instances, depth_ordering, tile_key_bits, tile_offsets
from .pipelines import Standard3DGS
from .preprocess import preprocess
from .record import REC
from ..gpu.scan import exclusive_scan, scan_workspace_size
from ..core.math import sh_num_bases
from ..gpu.sort import radix_sort_pairs, sort_hist_size, sort_partials_size


struct ForwardState(ImplicitlyCopyable, Movable):
    """What a training forward saves for its backward (raw device pointers).

    Slot arrays are indexed by visible-Gaussian slot; `slot_of` maps a
    Gaussian to its slot.
    """

    var n: Int
    var n_visible: Int
    var n_instances: Int
    var n_tiles: Int
    var n_bucket_blocks: Int  # upper bound on buckets (grid of the backward)
    var width: Int
    var height: Int
    var grid_w: Int
    var grid_h: Int
    var active_bases: Int
    var cam: SIMD[DType.float32, 32]
    var background: SIMD[DType.float32, 4]
    var slot_of: I32Ptr  # [N] slot of each Gaussian, -1 if culled
    var records: F32Ptr  # [N, REC]
    var colors: F32Ptr  # [N, 3]
    var instance_slots: I32Ptr  # [M] sorted by (tile, depth)
    var inst_a: I32Ptr
    var inst_b: I32Ptr
    var tile_offsets: U32Ptr  # [n_tiles + 1]
    var bucket_start: U32Ptr  # [n_tiles + 1]
    var bucket_tile: U32Ptr  # [n_bucket_blocks]
    var snapshots: F32Ptr  # [n_bucket_blocks, tile_w * tile_h, 4]
    var final_T: F32Ptr  # [H * W]
    var n_processed: U32Ptr  # [H * W]
    var tile_max_n: U32Ptr  # [n_tiles]

    def __init__(out self):
        self.n = 0
        self.n_visible = 0
        self.n_instances = 0
        self.n_tiles = 0
        self.n_bucket_blocks = 0
        self.width = 0
        self.height = 0
        self.grid_w = 0
        self.grid_h = 0
        self.active_bases = 1
        self.cam = SIMD[DType.float32, 32](0.0)
        self.background = SIMD[DType.float32, 4](0.0)
        self.slot_of = placeholder[DType.int32]()
        self.records = placeholder[DType.float32]()
        self.colors = placeholder[DType.float32]()
        self.instance_slots = placeholder[DType.int32]()
        self.inst_a = placeholder[DType.int32]()
        self.inst_b = placeholder[DType.int32]()
        self.tile_offsets = placeholder[DType.uint32]()
        self.bucket_start = placeholder[DType.uint32]()
        self.bucket_tile = placeholder[DType.uint32]()
        self.snapshots = placeholder[DType.float32]()
        self.final_T = placeholder[DType.float32]()
        self.n_processed = placeholder[DType.uint32]()
        self.tile_max_n = placeholder[DType.uint32]()


struct Rasterizer[tile_w: Int = 16, tile_h: Int = 12 if WARP_SIZE == 32 else 16](Movable):
    """Forward renderer and its backward; owns and reuses scratch buffers."""

    comptime CDIM = 3
    comptime SLOT_GRADS = Standard3DGS.Resp.GRAD  # floats per slot before the color gradient
    comptime BLOCK = Self.tile_w * Self.tile_h
    var ctx: DeviceContext
    # Per-Gaussian scratch (slot arrays, records and colors come from `arena`)
    var n_cap: Int
    var depth_keys_a: DeviceBuffer[DType.uint32]
    var depth_keys_b: DeviceBuffer[DType.uint32]
    var slots_a: DeviceBuffer[DType.int32]
    var slots_b: DeviceBuffer[DType.int32]
    var n_touched: DeviceBuffer[DType.uint32]
    var offsets: DeviceBuffer[DType.uint32]
    var scan_cap: Int
    var scan_partials: DeviceBuffer[DType.uint32]
    var scan_total: DeviceBuffer[DType.uint32]
    var counters: DeviceBuffer[DType.uint32]
    var host_counters: HostBuffer[DType.uint32]
    # Per-instance (keys stored as u32; 16-bit keys reuse the same memory)
    var m_cap: Int
    var keys_a: DeviceBuffer[DType.uint32]
    var keys_b: DeviceBuffer[DType.uint32]
    # Sort scratch, sized for max(N, M)
    var sort_cap: Int
    var sort_hist: DeviceBuffer[DType.uint32]
    var sort_partials: DeviceBuffer[DType.uint32]
    var sort_total: DeviceBuffer[DType.uint32]
    # Backward scratch: per-slot 2D gradients
    var grad_cap: Int
    var grad_scratch: DeviceBuffer[DType.float32]
    # Default state allocator: `render`, and `forward` without an explicit one
    var arena: ArcPointer[DeviceArena]
    # Results of the last render / forward
    var num_visible: Int
    var num_instances: Int
    # Optional per-stage timing (adds syncs)
    var profile: Bool
    var stage_ms: SIMD[DType.float64, 8]  # preprocess, depth sort, instances, tile sort, blend, -, -, total

    def __init__(out self, ctx: DeviceContext) raises:
        self.ctx = ctx
        self.n_cap = 0
        self.depth_keys_a = ctx.enqueue_create_buffer[DType.uint32](1)
        self.depth_keys_b = ctx.enqueue_create_buffer[DType.uint32](1)
        self.slots_a = ctx.enqueue_create_buffer[DType.int32](1)
        self.slots_b = ctx.enqueue_create_buffer[DType.int32](1)
        self.n_touched = ctx.enqueue_create_buffer[DType.uint32](1)
        self.offsets = ctx.enqueue_create_buffer[DType.uint32](1)
        self.scan_cap = 1
        self.scan_partials = ctx.enqueue_create_buffer[DType.uint32](scan_workspace_size(1))
        self.scan_total = ctx.enqueue_create_buffer[DType.uint32](1)
        self.counters = ctx.enqueue_create_buffer[DType.uint32](2)
        self.host_counters = ctx.enqueue_create_host_buffer[DType.uint32](2)
        self.m_cap = 0
        self.keys_a = ctx.enqueue_create_buffer[DType.uint32](1)
        self.keys_b = ctx.enqueue_create_buffer[DType.uint32](1)
        self.sort_cap = 0
        self.sort_hist = ctx.enqueue_create_buffer[DType.uint32](1)
        self.sort_partials = ctx.enqueue_create_buffer[DType.uint32](1)
        self.sort_total = ctx.enqueue_create_buffer[DType.uint32](1)
        self.grad_cap = 0
        self.grad_scratch = ctx.enqueue_create_buffer[DType.float32](1)
        self.arena = ArcPointer(DeviceArena(ctx))
        self.num_visible = 0
        self.num_instances = 0
        self.profile = False
        self.stage_ms = SIMD[DType.float64, 8](0)

    @staticmethod
    def _grow(cap: Int, need: Int) -> Int:
        return max(need, cap * 3 // 2)

    def _reserve_gaussians(mut self, n: Int) raises:
        if n <= self.n_cap:
            return
        var cap = Self._grow(self.n_cap, n)
        var ctx = self.ctx
        self.depth_keys_a = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.depth_keys_b = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.slots_a = ctx.enqueue_create_buffer[DType.int32](cap)
        self.slots_b = ctx.enqueue_create_buffer[DType.int32](cap)
        self.n_touched = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.offsets = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.n_cap = cap
        self._reserve_scan(cap)
        self._reserve_sort(cap)

    def _reserve_scan(mut self, n: Int) raises:
        if n <= self.scan_cap:
            return
        self.scan_cap = Self._grow(self.scan_cap, n)
        self.scan_partials = self.ctx.enqueue_create_buffer[DType.uint32](scan_workspace_size(self.scan_cap))

    def _reserve_instances(mut self, m: Int) raises:
        if m <= self.m_cap:
            return
        var cap = Self._grow(self.m_cap, m)
        var ctx = self.ctx
        self.keys_a = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.keys_b = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.m_cap = cap
        self._reserve_sort(cap)

    def _reserve_sort(mut self, n: Int) raises:
        if n <= self.sort_cap:
            return
        self.sort_hist = self.ctx.enqueue_create_buffer[DType.uint32](sort_hist_size(n))
        self.sort_partials = self.ctx.enqueue_create_buffer[DType.uint32](sort_partials_size(n))
        self.sort_cap = n

    @staticmethod
    def _alloc_slot_state[A: StateAllocator](mut st: ForwardState, mut alloc: A) raises:
        """Per-slot arrays (before preprocessing). `render` and `forward`
        allocate in the same order so they share the arena's buffers."""
        var n = max(st.n, 1)
        st.slot_of = I32Ptr(unsafe_from_address=alloc.alloc(n * 4))
        st.records = F32Ptr(unsafe_from_address=alloc.alloc(n * REC * 4))
        st.colors = F32Ptr(unsafe_from_address=alloc.alloc(n * 3 * 4))

    @staticmethod
    def _alloc_instance_state[A: StateAllocator](mut st: ForwardState, mut alloc: A) raises:
        """Per-instance and per-tile arrays (after preprocessing)."""
        var m = max(st.n_instances, 1)
        st.inst_a = I32Ptr(unsafe_from_address=alloc.alloc(m * 4))
        st.inst_b = I32Ptr(unsafe_from_address=alloc.alloc(m * 4))
        st.tile_offsets = U32Ptr(unsafe_from_address=alloc.alloc((st.n_tiles + 1) * 4))

    def _mark(self, mut t: Int, slot: Int, mut ms: SIMD[DType.float64, 8]) raises:
        if self.profile:
            self.ctx.synchronize()
            var now = Int(perf_counter_ns())
            ms[slot] = Float64(now - t) / 1e6
            t = now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        mut self,
        splats: Splats,
        camera: Camera,
        background: SIMD[DType.float32, 4],
        image: DeviceBuffer[DType.float32],
        sh_degree: Int = 3,
    ) raises:
        """Render owned splats into an owned [H, W, 3] image buffer."""
        if len(image) < camera.width * camera.height * Self.CDIM:
            raise Error("image buffer too small")
        self.render(splats.view(), camera, background, dptr(image), sh_degree)

    def render(
        mut self,
        splats: SplatView,
        camera: Camera,
        background: SIMD[DType.float32, 4],
        image: F32Ptr,
        sh_degree: Int = 3,
    ) raises:
        """Render into `image` ([H, W, 3] float32 device memory).

        `sh_degree` is clamped to what `splats.rest_stride` holds. The caller
        keeps the memory behind `splats`/`image` alive until the work is
        done. Blocks once (after preprocessing) to size the instance buffers.
        Uses the same scratch memory as `forward()`: a render between a
        `forward()` and its `backward()` invalidates that forward's state.
        """
        var arena = self.arena.copy()
        arena[].reset()
        var st = self._begin(splats, camera, background, sh_degree)
        Self._alloc_slot_state(st, arena[])
        var ms = SIMD[DType.float64, 8](0)
        var t_start = Int(perf_counter_ns())
        var t = t_start
        self._preprocess(splats, st)
        self._mark(t, 0, ms)
        Self._alloc_instance_state(st, arena[])
        self._raster[False](st, image, t, ms)
        if self.profile:
            ms[7] = Float64(t - t_start) / 1e6
            self.stage_ms = ms

    def forward(
        mut self,
        splats: SplatView,
        camera: Camera,
        background: SIMD[DType.float32, 4],
        image: F32Ptr,
        sh_degree: Int = 3,
    ) raises -> ForwardState:
        """Training forward using the rasterizer's own reusable state buffers.

        The returned state stays valid until the next `forward` or `render`
        call on this rasterizer (one forward/backward in flight at a time).
        """
        var arena = self.arena.copy()
        arena[].reset()
        return self.forward(splats, camera, background, image, sh_degree, arena[])

    def forward[
        A: StateAllocator
    ](
        mut self,
        splats: SplatView,
        camera: Camera,
        background: SIMD[DType.float32, 4],
        image: F32Ptr,
        sh_degree: Int,
        mut alloc: A,
    ) raises -> ForwardState:
        """Training forward: renders like `render` and returns the state for
        `backward`. Persistent buffers come from `alloc` and must stay alive
        until the backward has run (keep the allocator alive too: Mojo
        destroys values after their last use)."""
        var st = self._begin(splats, camera, background, sh_degree)
        Self._alloc_slot_state(st, alloc)
        var ms = SIMD[DType.float64, 8](0)
        var t = Int(perf_counter_ns())
        self._preprocess[True](splats, st)
        st.n_bucket_blocks = bucket_bound(st.n_instances, st.n_tiles)
        var pixels = st.width * st.height
        Self._alloc_instance_state(st, alloc)
        st.bucket_start = U32Ptr(unsafe_from_address=alloc.alloc((st.n_tiles + 1) * 4))
        st.tile_max_n = U32Ptr(unsafe_from_address=alloc.alloc(st.n_tiles * 4))
        st.bucket_tile = U32Ptr(unsafe_from_address=alloc.alloc(st.n_bucket_blocks * 4))
        st.snapshots = F32Ptr(unsafe_from_address=alloc.alloc(st.n_bucket_blocks * Self.BLOCK * 16))
        st.final_T = F32Ptr(unsafe_from_address=alloc.alloc(pixels * 4))
        st.n_processed = U32Ptr(unsafe_from_address=alloc.alloc(pixels * 4))
        self._raster[True](st, image, t, ms)
        return st

    def backward(
        mut self,
        splats: SplatView,
        st: ForwardState,
        image: F32Ptr,
        grad_image: F32Ptr,
        grads: SplatView,
        densification: F32Ptr = placeholder[DType.float32](),
        has_densification: Bool = False,
        write_sh_rest: Bool = True,
    ) raises:
        """Gradients of the loss w.r.t. the raw parameters.

        `image` is the forward's output and `grad_image` dL/d(image), both
        [H, W, 3]. `grads` points at gradient buffers laid out like `splats`
        (same rest_stride); every element is written (culled Gaussians get
        0), so they need no initialization. If
        `has_densification`, `densification` [2, N] accumulates (count,
        |dL/dmean2d| in NDC) per Gaussian, as used by 3DGS densification.
        With `write_sh_rest=False`, `grads.sh_rest` is left untouched: a fused
        optimizer recomputes those gradients from `sh_rest_grad(st)`.
        """
        var ctx = self.ctx
        var stream = ctx.stream()
        var n = splats.count
        var nv = max(st.n_visible, 1)
        var grad_floats = nv * (Self.SLOT_GRADS + Self.CDIM)
        if grad_floats > self.grad_cap:
            self.grad_cap = Self._grow(self.grad_cap, grad_floats)
            self.grad_scratch = ctx.enqueue_create_buffer[DType.float32](self.grad_cap)
        var g_resp = dptr(self.grad_scratch)
        var g_color = g_resp.unsafe_offset(nv * Self.SLOT_GRADS)
        fill_f32(ctx, g_resp, 0.0, grad_floats)
        if st.n_instances > 0:
            blend_backward[Self.tile_w, Self.tile_h](
                ctx,
                stream,
                st.n_bucket_blocks,
                st.tile_offsets,
                st.bucket_start,
                st.bucket_tile,
                st.tile_max_n,
                st.instance_slots,
                st.records,
                st.colors,
                st.background,
                image,
                grad_image,
                st.final_T,
                st.n_processed,
                st.snapshots,
                g_resp,
                nv,
                g_color,
                st.n_tiles,
                st.width,
                st.height,
            )
        preprocess_backward(
            ctx,
            stream,
            splats.means,
            splats.log_scales,
            splats.rotations,
            splats.sh_rest,
            st.cam,
            st.slot_of,
            g_resp,
            nv,
            g_color,
            grads.means,
            grads.log_scales,
            grads.rotations,
            grads.opacity_logits,
            grads.sh0,
            grads.sh_rest,
            densification,
            has_densification,
            n,
            st.active_bases,
            splats.rest_stride,
            write_sh_rest,
        )

    def sh_rest_grad(self, st: ForwardState) -> ShRestGrad:
        """Inputs to recompute the higher-order SH gradients of the last
        `backward` (on `st`) elementwise; valid until the next forward."""
        var nv = max(st.n_visible, 1)
        return ShRestGrad(
            st.slot_of,
            dptr(self.grad_scratch).unsafe_offset(nv * Self.SLOT_GRADS),
            SIMD[DType.float32, 4](st.cam[CAM_POS_X], st.cam[CAM_POS_Y], st.cam[CAM_POS_Z], 0.0),
            st.active_bases,
        )

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _begin(
        mut self,
        splats: SplatView,
        camera: Camera,
        background: SIMD[DType.float32, 4],
        sh_degree: Int,
    ) raises -> ForwardState:
        var st = ForwardState()
        st.n = splats.count
        st.width = camera.width
        st.height = camera.height
        if st.width >= 65536 or st.height >= 65536:
            raise Error("image dimensions must be < 65536")
        st.grid_w = (st.width + Self.tile_w - 1) // Self.tile_w
        st.grid_h = (st.height + Self.tile_h - 1) // Self.tile_h
        st.n_tiles = st.grid_w * st.grid_h
        st.active_bases = sh_num_bases(min(max(sh_degree, 0), splats.max_sh_degree()))
        st.cam = camera.packed()
        st.background = background
        self._reserve_gaussians(max(st.n, 1))
        self._reserve_scan(st.n_tiles)
        return st

    def _preprocess[train: Bool = False](mut self, splats: SplatView, mut st: ForwardState) raises:
        """Preprocess into st's slot arrays; one host sync for the counts."""
        var ctx = self.ctx
        self.counters.enqueue_fill(0)
        preprocess[Self.tile_w, Self.tile_h, train](
            ctx,
            ctx.stream(),
            splats.means,
            splats.log_scales,
            splats.rotations,
            splats.opacity_logits,
            splats.sh0,
            splats.sh_rest,
            st.cam,
            dptr(self.depth_keys_a),
            dptr(self.slots_a),
            dptr(self.n_touched),
            st.records,
            st.colors,
            dptr(self.counters),
            st.slot_of,
            st.n,
            st.grid_w,
            st.grid_h,
            st.active_bases,
            splats.rest_stride,
        )
        ctx.enqueue_copy(self.host_counters, self.counters)
        ctx.synchronize()
        st.n_visible = Int(self.host_counters[0])
        st.n_instances = Int(self.host_counters[1])
        self.num_visible = st.n_visible
        self.num_instances = st.n_instances
        self._reserve_instances(max(st.n_instances, 1))

    def _raster[
        train: Bool
    ](mut self, mut st: ForwardState, image: F32Ptr, mut t: Int, mut ms: SIMD[DType.float64, 8]) raises:
        var ctx = self.ctx
        var stream = ctx.stream()
        var n_vis = st.n_visible
        var m = st.n_instances
        var n_tiles = st.n_tiles

        # Depth sort of the visible Gaussians.
        var ids_sorted = dptr(self.slots_a)
        if n_vis > 1:
            var in_b = radix_sort_pairs[DType.uint32](
                ctx,
                stream,
                dptr(self.depth_keys_a),
                dptr(self.slots_a),
                dptr(self.depth_keys_b),
                dptr(self.slots_b),
                dptr(self.sort_hist),
                dptr(self.sort_partials),
                dptr(self.sort_total),
                n_vis,
                32,
            )
            ids_sorted = dptr(self.slots_b) if in_b else dptr(self.slots_a)
        self._mark(t, 1, ms)

        # Instances in depth order.
        var key_bits = tile_key_bits(n_tiles)
        if m > 0:
            depth_ordering(ctx, stream, ids_sorted, dptr(self.n_touched), dptr(self.offsets), n_vis)
            exclusive_scan[DType.uint32](
                ctx,
                stream,
                dptr(self.offsets),
                dptr(self.offsets),
                dptr(self.scan_total),
                dptr(self.scan_partials),
                n_vis,
            )
            if key_bits <= 16:
                create_instances[DType.uint16, Self.tile_w, Self.tile_h](
                    ctx,
                    stream,
                    ids_sorted,
                    dptr(self.offsets),
                    dptr(self.n_touched),
                    st.records,
                    self._keys16(False),
                    st.inst_a,
                    st.grid_w,
                    n_tiles,
                    n_vis,
                )
            else:
                create_instances[DType.uint32, Self.tile_w, Self.tile_h](
                    ctx,
                    stream,
                    ids_sorted,
                    dptr(self.offsets),
                    dptr(self.n_touched),
                    st.records,
                    dptr(self.keys_a),
                    st.inst_a,
                    st.grid_w,
                    n_tiles,
                    n_vis,
                )
        self._mark(t, 2, ms)

        # Stable sort by tile, then per-tile ranges.
        var in_b = False
        if key_bits <= 16:
            if m > 1:
                in_b = radix_sort_pairs[DType.uint16](
                    ctx,
                    stream,
                    self._keys16(False),
                    st.inst_a,
                    self._keys16(True),
                    st.inst_b,
                    dptr(self.sort_hist),
                    dptr(self.sort_partials),
                    dptr(self.sort_total),
                    m,
                    key_bits,
                )
            tile_offsets[DType.uint16](ctx, stream, self._keys16(in_b), st.tile_offsets, m, n_tiles)
        else:
            if m > 1:
                in_b = radix_sort_pairs[DType.uint32](
                    ctx,
                    stream,
                    dptr(self.keys_a),
                    st.inst_a,
                    dptr(self.keys_b),
                    st.inst_b,
                    dptr(self.sort_hist),
                    dptr(self.sort_partials),
                    dptr(self.sort_total),
                    m,
                    key_bits,
                )
            var keys = dptr(self.keys_b) if in_b else dptr(self.keys_a)
            tile_offsets[DType.uint32](ctx, stream, keys, st.tile_offsets, m, n_tiles)
        st.instance_slots = st.inst_b if in_b else st.inst_a
        self._mark(t, 3, ms)

        comptime if train:
            # Bucket layout: ceil(instances / 32) per tile, scanned.
            bucket_counts(ctx, stream, st.tile_offsets, st.bucket_start, n_tiles)
            exclusive_scan[DType.uint32](
                ctx,
                stream,
                st.bucket_start,
                st.bucket_start,
                st.bucket_start.unsafe_offset(n_tiles),
                dptr(self.scan_partials),
                n_tiles,
            )
            blend[Self.tile_w, Self.tile_h, Self.CDIM, True](
                ctx,
                stream,
                st.tile_offsets,
                st.instance_slots,
                st.records,
                st.colors,
                st.background,
                image,
                st.width,
                st.height,
                st.bucket_start,
                st.bucket_tile,
                st.snapshots,
                st.final_T,
                st.n_processed,
                st.tile_max_n,
            )
        else:
            blend[Self.tile_w, Self.tile_h, Self.CDIM](
                ctx,
                stream,
                st.tile_offsets,
                st.instance_slots,
                st.records,
                st.colors,
                st.background,
                image,
                st.width,
                st.height,
            )
        self._mark(t, 4, ms)

    def _keys16(self, b: Bool) -> U16Ptr:
        var p = dptr(self.keys_b) if b else dptr(self.keys_a)
        return U16Ptr(unsafe_from_address=Int(p))
