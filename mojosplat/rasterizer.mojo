"""Faster-GS style forward rasterizer (inference path).

Pipeline (one host sync, to size the instance buffers):

  preprocess       activations, EWA projection, exact tile count, SH -> RGB,
                   compacted (depth, id) list of visible Gaussians
  depth sort       radix sort of the visible Gaussians by depth (32-bit keys)
  instances        exclusive scan of tile counts in depth order, then one
                   (tile key, id) per Gaussian-tile pair
  tile sort        stable radix sort by tile only (16-bit keys up to 64k tiles)
  tile offsets     per-tile instance ranges
  blend            per-tile front-to-back compositing with sub-tile skipping

Takes the *raw* (trainable) parameters as in 3DGS/gsplat training: log
scales, unnormalized quaternions, opacity logits and SH coefficients.
Design and kernels follow Faster-GS (Hahlbohm et al., CVPR 2026, Apache-2.0).
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.math import log
from std.time import perf_counter_ns

from .blend import blend
from .camera import Camera
from .common import F32Ptr, I32Ptr, U16Ptr, U32Ptr, dptr
from .instances import create_instances, depth_ordering, tile_key_bits, tile_offsets
from .io import SplatScene, upload
from .preprocess import REC, preprocess
from .scan import exclusive_scan, scan_workspace_size
from .sh import SH_C0, sh_num_bases
from .sort import radix_sort_pairs, sort_hist_size, sort_partials_size


@fieldwise_init
struct SplatParams(ImplicitlyCopyable, Movable):
    """Device pointers to N Gaussians' raw parameters (float32, SoA).

    means [N,3], log_scales [N,3], rotations [N,4] (w,x,y,z; any norm),
    opacity_logits [N], sh0 [N,3], sh_rest [N, rest_stride, 3].
    """

    var means: F32Ptr
    var log_scales: F32Ptr
    var rotations: F32Ptr
    var opacity_logits: F32Ptr
    var sh0: F32Ptr
    var sh_rest: F32Ptr
    var count: Int
    var rest_stride: Int  # SH bases stored per Gaussian in sh_rest

    def max_sh_degree(self) -> Int:
        var d = 0
        while sh_num_bases(d + 1) - 1 <= self.rest_stride and d < 3:
            d += 1
        return d


struct SplatBuffers(Movable):
    """Owns N Gaussians' raw parameters in device memory.

    Pass this (not a `SplatParams` view) to `Rasterizer.render` from Mojo code
    so the buffers outlive the enqueued work.
    """

    var means: DeviceBuffer[DType.float32]
    var log_scales: DeviceBuffer[DType.float32]
    var rotations: DeviceBuffer[DType.float32]
    var opacity_logits: DeviceBuffer[DType.float32]
    var sh0: DeviceBuffer[DType.float32]
    var sh_rest: DeviceBuffer[DType.float32]
    var count: Int
    var rest_stride: Int

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
    ) raises:
        self.count = len(opacity_logits)
        self.rest_stride = rest_stride
        self.means = upload(ctx, means)
        self.log_scales = upload(ctx, log_scales)
        self.rotations = upload(ctx, rotations)
        self.opacity_logits = upload(ctx, opacity_logits)
        self.sh0 = upload(ctx, sh0)
        if len(sh_rest) > 0:
            self.sh_rest = upload(ctx, sh_rest)
        else:
            self.sh_rest = ctx.enqueue_create_buffer[DType.float32](3)

    @staticmethod
    def from_scene(ctx: DeviceContext, scene: SplatScene) raises -> SplatBuffers:
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
        return SplatBuffers(
            ctx, scene.means, log_scales, scene.quats, logits, sh0, List[Float32](), 0
        )

    def view(self) -> SplatParams:
        return SplatParams(
            dptr(self.means), dptr(self.log_scales), dptr(self.rotations),
            dptr(self.opacity_logits), dptr(self.sh0), dptr(self.sh_rest),
            self.count, self.rest_stride,
        )


struct Rasterizer[tile_w: Int = 16, tile_h: Int = 12](Movable):
    """Forward renderer; owns and reuses all intermediate GPU buffers."""

    comptime CDIM = 3
    var ctx: DeviceContext
    # Per-Gaussian
    var n_cap: Int
    # Per visible Gaussian ("slot"; compacted by preprocess)
    var depth_keys_a: DeviceBuffer[DType.uint32]
    var depth_keys_b: DeviceBuffer[DType.uint32]
    var slots_a: DeviceBuffer[DType.int32]
    var slots_b: DeviceBuffer[DType.int32]
    var primitive_ids: DeviceBuffer[DType.int32]
    var n_touched: DeviceBuffer[DType.uint32]
    var records: DeviceBuffer[DType.float32]
    var colors: DeviceBuffer[DType.float32]
    var offsets: DeviceBuffer[DType.uint32]
    var scan_partials: DeviceBuffer[DType.uint32]
    var scan_total: DeviceBuffer[DType.uint32]
    var counters: DeviceBuffer[DType.uint32]
    var host_counters: HostBuffer[DType.uint32]
    # Per-instance (keys stored as u32; 16-bit keys reuse the same memory)
    var m_cap: Int
    var keys_a: DeviceBuffer[DType.uint32]
    var keys_b: DeviceBuffer[DType.uint32]
    var inst_a: DeviceBuffer[DType.int32]
    var inst_b: DeviceBuffer[DType.int32]
    # Sort scratch, sized for max(N, M)
    var sort_cap: Int
    var sort_hist: DeviceBuffer[DType.uint32]
    var sort_partials: DeviceBuffer[DType.uint32]
    var sort_total: DeviceBuffer[DType.uint32]
    # Per-tile
    var tiles_cap: Int
    var tile_offsets: DeviceBuffer[DType.uint32]
    # Results of the last render
    var num_visible: Int
    var num_instances: Int
    var inst_in_b: Bool
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
        self.primitive_ids = ctx.enqueue_create_buffer[DType.int32](1)
        self.n_touched = ctx.enqueue_create_buffer[DType.uint32](1)
        self.records = ctx.enqueue_create_buffer[DType.float32](REC)
        self.colors = ctx.enqueue_create_buffer[DType.float32](3)
        self.offsets = ctx.enqueue_create_buffer[DType.uint32](1)
        self.scan_partials = ctx.enqueue_create_buffer[DType.uint32](scan_workspace_size(1))
        self.scan_total = ctx.enqueue_create_buffer[DType.uint32](1)
        self.counters = ctx.enqueue_create_buffer[DType.uint32](2)
        self.host_counters = ctx.enqueue_create_host_buffer[DType.uint32](2)
        self.m_cap = 0
        self.keys_a = ctx.enqueue_create_buffer[DType.uint32](1)
        self.keys_b = ctx.enqueue_create_buffer[DType.uint32](1)
        self.inst_a = ctx.enqueue_create_buffer[DType.int32](1)
        self.inst_b = ctx.enqueue_create_buffer[DType.int32](1)
        self.sort_cap = 0
        self.sort_hist = ctx.enqueue_create_buffer[DType.uint32](1)
        self.sort_partials = ctx.enqueue_create_buffer[DType.uint32](1)
        self.sort_total = ctx.enqueue_create_buffer[DType.uint32](1)
        self.tiles_cap = 0
        self.tile_offsets = ctx.enqueue_create_buffer[DType.uint32](1)
        self.num_visible = 0
        self.num_instances = 0
        self.inst_in_b = False
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
        self.primitive_ids = ctx.enqueue_create_buffer[DType.int32](cap)
        self.n_touched = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.records = ctx.enqueue_create_buffer[DType.float32](cap * REC)
        self.colors = ctx.enqueue_create_buffer[DType.float32](cap * 3)
        self.offsets = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.scan_partials = ctx.enqueue_create_buffer[DType.uint32](scan_workspace_size(cap))
        self.n_cap = cap
        self._reserve_sort(cap)

    def _reserve_instances(mut self, m: Int) raises:
        if m <= self.m_cap:
            return
        var cap = Self._grow(self.m_cap, m)
        var ctx = self.ctx
        self.keys_a = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.keys_b = ctx.enqueue_create_buffer[DType.uint32](cap)
        self.inst_a = ctx.enqueue_create_buffer[DType.int32](cap)
        self.inst_b = ctx.enqueue_create_buffer[DType.int32](cap)
        self.m_cap = cap
        self._reserve_sort(cap)

    def _reserve_sort(mut self, n: Int) raises:
        if n <= self.sort_cap:
            return
        self.sort_hist = self.ctx.enqueue_create_buffer[DType.uint32](sort_hist_size(n))
        self.sort_partials = self.ctx.enqueue_create_buffer[DType.uint32](sort_partials_size(n))
        self.sort_cap = n

    def _reserve_tiles(mut self, n_tiles: Int) raises:
        if n_tiles <= self.tiles_cap:
            return
        self.tile_offsets = self.ctx.enqueue_create_buffer[DType.uint32](n_tiles + 1)
        self.tiles_cap = n_tiles

    def _mark(self, mut t: Int, slot: Int, mut ms: SIMD[DType.float64, 8]) raises:
        if self.profile:
            self.ctx.synchronize()
            var now = Int(perf_counter_ns())
            ms[slot] = Float64(now - t) / 1e6
            t = now

    def instance_ids(self) -> I32Ptr:
        """Slots of the last render sorted by (tile, depth); map a slot to its
        Gaussian with `primitive_ids`."""
        return dptr(self.inst_b) if self.inst_in_b else dptr(self.inst_a)

    def render(
        mut self,
        splats: SplatBuffers,
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
        splats: SplatParams,
        camera: Camera,
        background: SIMD[DType.float32, 4],
        image: F32Ptr,
        sh_degree: Int = 3,
    ) raises:
        """Render into `image` ([H, W, 3] float32 device memory).

        `sh_degree` is clamped to what `splats.rest_stride` holds. The caller
        keeps the memory behind `splats`/`image` alive until the work is
        done. Blocks once (after preprocessing) to size the instance buffers.
        """
        var ctx = self.ctx
        var stream = ctx.stream()
        var n = splats.count
        var W = camera.width
        var H = camera.height
        if W >= 65536 or H >= 65536:
            raise Error("image dimensions must be < 65536")
        var grid_w = (W + Self.tile_w - 1) // Self.tile_w
        var grid_h = (H + Self.tile_h - 1) // Self.tile_h
        var n_tiles = grid_w * grid_h
        var degree = min(max(sh_degree, 0), splats.max_sh_degree())
        self._reserve_gaussians(max(n, 1))
        self._reserve_tiles(n_tiles)
        var ms = SIMD[DType.float64, 8](0)
        var t_start = Int(perf_counter_ns())
        var t = t_start

        self.counters.enqueue_fill(0)
        preprocess[Self.tile_w, Self.tile_h](
            ctx, stream, splats.means, splats.log_scales, splats.rotations,
            splats.opacity_logits, splats.sh0, splats.sh_rest, camera.packed(),
            dptr(self.depth_keys_a), dptr(self.slots_a), dptr(self.primitive_ids),
            dptr(self.n_touched), dptr(self.records), dptr(self.colors),
            dptr(self.counters), n, grid_w, grid_h,
            sh_num_bases(degree), splats.rest_stride,
        )
        ctx.enqueue_copy(self.host_counters, self.counters)
        ctx.synchronize()
        var n_vis = Int(self.host_counters[0])
        var m = Int(self.host_counters[1])
        self.num_visible = n_vis
        self.num_instances = m
        self._mark(t, 0, ms)

        # Depth sort of the visible Gaussians.
        var ids_sorted = dptr(self.slots_a)
        if n_vis > 1:
            var in_b = radix_sort_pairs[DType.uint32](
                ctx, stream, dptr(self.depth_keys_a), dptr(self.slots_a),
                dptr(self.depth_keys_b), dptr(self.slots_b), dptr(self.sort_hist),
                dptr(self.sort_partials), dptr(self.sort_total), n_vis, 32,
            )
            ids_sorted = dptr(self.slots_b) if in_b else dptr(self.slots_a)
        self._mark(t, 1, ms)

        # Instances in depth order.
        self._reserve_instances(max(m, 1))
        var key_bits = tile_key_bits(n_tiles)
        if m > 0:
            depth_ordering(ctx, stream, ids_sorted, dptr(self.n_touched), dptr(self.offsets), n_vis)
            exclusive_scan[DType.uint32](
                ctx, stream, dptr(self.offsets), dptr(self.offsets),
                dptr(self.scan_total), dptr(self.scan_partials), n_vis,
            )
            if key_bits <= 16:
                create_instances[DType.uint16, Self.tile_w, Self.tile_h](
                    ctx, stream, ids_sorted, dptr(self.offsets), dptr(self.n_touched),
                    dptr(self.records), self._keys16(False), dptr(self.inst_a),
                    grid_w, n_tiles, n_vis,
                )
            else:
                create_instances[DType.uint32, Self.tile_w, Self.tile_h](
                    ctx, stream, ids_sorted, dptr(self.offsets), dptr(self.n_touched),
                    dptr(self.records), dptr(self.keys_a), dptr(self.inst_a),
                    grid_w, n_tiles, n_vis,
                )
        self._mark(t, 2, ms)

        # Stable sort by tile, then per-tile ranges.
        self.inst_in_b = False
        if key_bits <= 16:
            if m > 1:
                self.inst_in_b = radix_sort_pairs[DType.uint16](
                    ctx, stream, self._keys16(False), dptr(self.inst_a),
                    self._keys16(True), dptr(self.inst_b), dptr(self.sort_hist),
                    dptr(self.sort_partials), dptr(self.sort_total), m, key_bits,
                )
            tile_offsets[DType.uint16](
                ctx, stream, self._keys16(self.inst_in_b), dptr(self.tile_offsets), m, n_tiles
            )
        else:
            if m > 1:
                self.inst_in_b = radix_sort_pairs[DType.uint32](
                    ctx, stream, dptr(self.keys_a), dptr(self.inst_a),
                    dptr(self.keys_b), dptr(self.inst_b), dptr(self.sort_hist),
                    dptr(self.sort_partials), dptr(self.sort_total), m, key_bits,
                )
            var keys = dptr(self.keys_b) if self.inst_in_b else dptr(self.keys_a)
            tile_offsets[DType.uint32](ctx, stream, keys, dptr(self.tile_offsets), m, n_tiles)
        self._mark(t, 3, ms)

        blend[Self.tile_w, Self.tile_h, Self.CDIM](
            ctx, stream, dptr(self.tile_offsets), self.instance_ids(),
            dptr(self.records), dptr(self.colors), background, image, W, H,
        )
        self._mark(t, 4, ms)
        if self.profile:
            ms[7] = Float64(t - t_start) / 1e6
            self.stage_ms = ms

    def _keys16(self, b: Bool) -> U16Ptr:
        var p = dptr(self.keys_b) if b else dptr(self.keys_a)
        return U16Ptr(unsafe_from_address=Int(p))
