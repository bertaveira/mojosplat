"""High-level forward renderer: projection -> binning -> rasterization.

`Renderer` owns all intermediate GPU buffers and grows them on demand, so
rendering many frames (or a training loop) does not reallocate. Inputs and
the output image are raw device pointers, which lets the same code serve the
pure-Mojo examples and the PyTorch bindings (torch-owned memory).
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.time import perf_counter_ns

from .binning import (
    compute_tile_offsets,
    count_tiles,
    sort_end_bit,
    write_intersections,
)
from .camera import Camera
from .common import F32Ptr, I32Ptr, dptr
from .io import SplatScene, upload
from .projection import project_gaussians
from .rasterization import rasterize
from .scan import exclusive_scan, scan_workspace_size
from .sort import radix_sort_pairs, sort_hist_size, sort_partials_size


@fieldwise_init
struct Gaussians(ImplicitlyCopyable, Movable):
    """Device pointers to N Gaussians (struct-of-arrays, float32).

    scales are linear (not log), quats (w, x, y, z), opacities in [0, 1],
    colors [N, channels].
    """

    var means: F32Ptr
    var scales: F32Ptr
    var quats: F32Ptr
    var opacities: F32Ptr
    var colors: F32Ptr
    var count: Int


struct GaussianBuffers(Movable):
    """Owns N Gaussians in device memory.

    Pass this (not a `Gaussians` view) to `Renderer.render` from Mojo code:
    Mojo destroys values after their last use, so a raw-pointer view alone
    does not keep the underlying buffers alive.
    """

    var means: DeviceBuffer[DType.float32]
    var scales: DeviceBuffer[DType.float32]
    var quats: DeviceBuffer[DType.float32]
    var opacities: DeviceBuffer[DType.float32]
    var colors: DeviceBuffer[DType.float32]
    var count: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        means: List[Float32],
        scales: List[Float32],
        quats: List[Float32],
        opacities: List[Float32],
        colors: List[Float32],
    ) raises:
        """Upload host arrays (linear scales, (w, x, y, z) quats)."""
        self.count = len(opacities)
        self.means = upload(ctx, means)
        self.scales = upload(ctx, scales)
        self.quats = upload(ctx, quats)
        self.opacities = upload(ctx, opacities)
        self.colors = upload(ctx, colors)

    @staticmethod
    def from_scene(ctx: DeviceContext, scene: SplatScene) raises -> GaussianBuffers:
        return GaussianBuffers(
            ctx, scene.means, scene.scales, scene.quats, scene.opacities, scene.colors
        )

    def view(self) -> Gaussians:
        return Gaussians(
            dptr(self.means), dptr(self.scales), dptr(self.quats),
            dptr(self.opacities), dptr(self.colors), self.count,
        )


struct Renderer[tile_size: Int = 16, channels: Int = 3](Movable):
    var ctx: DeviceContext
    # Per-Gaussian buffers
    var n_cap: Int
    var means2d: DeviceBuffer[DType.float32]
    var conics: DeviceBuffer[DType.float32]
    var depths: DeviceBuffer[DType.float32]
    var radii: DeviceBuffer[DType.int32]
    var tiles_per_gauss: DeviceBuffer[DType.int32]
    var offsets: DeviceBuffer[DType.int32]
    var scan_partials: DeviceBuffer[DType.int32]
    var scan_total: DeviceBuffer[DType.int32]
    var host_total: HostBuffer[DType.int32]
    # Per-intersection buffers
    var m_cap: Int
    var keys_a: DeviceBuffer[DType.uint64]
    var keys_b: DeviceBuffer[DType.uint64]
    var ids_a: DeviceBuffer[DType.int32]
    var ids_b: DeviceBuffer[DType.int32]
    var sort_hist: DeviceBuffer[DType.uint32]
    var sort_partials: DeviceBuffer[DType.uint32]
    var sort_total: DeviceBuffer[DType.uint32]
    # Per-tile buffers
    var tiles_cap: Int
    var tile_offsets: DeviceBuffer[DType.int32]
    # Result of the last `render`
    var num_intersections: Int
    var sorted_in_b: Bool
    # Optional per-stage timing (adds a sync after every stage).
    var profile: Bool
    var stage_ms: SIMD[DType.float64, 4]  # projection, binning, raster, total

    def __init__(out self, ctx: DeviceContext) raises:
        self.ctx = ctx
        self.n_cap = 1
        self.means2d = ctx.enqueue_create_buffer[DType.float32](2)
        self.conics = ctx.enqueue_create_buffer[DType.float32](3)
        self.depths = ctx.enqueue_create_buffer[DType.float32](1)
        self.radii = ctx.enqueue_create_buffer[DType.int32](2)
        self.tiles_per_gauss = ctx.enqueue_create_buffer[DType.int32](1)
        self.offsets = ctx.enqueue_create_buffer[DType.int32](1)
        self.scan_partials = ctx.enqueue_create_buffer[DType.int32](1)
        self.scan_total = ctx.enqueue_create_buffer[DType.int32](1)
        self.host_total = ctx.enqueue_create_host_buffer[DType.int32](1)
        self.m_cap = 1
        self.keys_a = ctx.enqueue_create_buffer[DType.uint64](1)
        self.keys_b = ctx.enqueue_create_buffer[DType.uint64](1)
        self.ids_a = ctx.enqueue_create_buffer[DType.int32](1)
        self.ids_b = ctx.enqueue_create_buffer[DType.int32](1)
        self.sort_hist = ctx.enqueue_create_buffer[DType.uint32](1)
        self.sort_partials = ctx.enqueue_create_buffer[DType.uint32](1)
        self.sort_total = ctx.enqueue_create_buffer[DType.uint32](1)
        self.tiles_cap = 1
        self.tile_offsets = ctx.enqueue_create_buffer[DType.int32](2)
        self.num_intersections = 0
        self.sorted_in_b = False
        self.profile = False
        self.stage_ms = SIMD[DType.float64, 4](0)

    def _reserve_gaussians(mut self, n: Int) raises:
        if n <= self.n_cap:
            return
        var cap = max(n, self.n_cap * 3 // 2)
        self.means2d = self.ctx.enqueue_create_buffer[DType.float32](cap * 2)
        self.conics = self.ctx.enqueue_create_buffer[DType.float32](cap * 3)
        self.depths = self.ctx.enqueue_create_buffer[DType.float32](cap)
        self.radii = self.ctx.enqueue_create_buffer[DType.int32](cap * 2)
        self.tiles_per_gauss = self.ctx.enqueue_create_buffer[DType.int32](cap)
        self.offsets = self.ctx.enqueue_create_buffer[DType.int32](cap)
        self.scan_partials = self.ctx.enqueue_create_buffer[DType.int32](
            scan_workspace_size(cap)
        )
        self.n_cap = cap

    def _reserve_intersections(mut self, m: Int) raises:
        if m <= self.m_cap:
            return
        var cap = max(m, self.m_cap * 3 // 2)
        self.keys_a = self.ctx.enqueue_create_buffer[DType.uint64](cap)
        self.keys_b = self.ctx.enqueue_create_buffer[DType.uint64](cap)
        self.ids_a = self.ctx.enqueue_create_buffer[DType.int32](cap)
        self.ids_b = self.ctx.enqueue_create_buffer[DType.int32](cap)
        self.sort_hist = self.ctx.enqueue_create_buffer[DType.uint32](sort_hist_size(cap))
        self.sort_partials = self.ctx.enqueue_create_buffer[DType.uint32](
            sort_partials_size(cap)
        )
        self.m_cap = cap

    def _reserve_tiles(mut self, n_tiles: Int) raises:
        if n_tiles <= self.tiles_cap:
            return
        self.tile_offsets = self.ctx.enqueue_create_buffer[DType.int32](n_tiles + 1)
        self.tiles_cap = n_tiles

    def sorted_ids(self) -> I32Ptr:
        """Gaussian ids of the last render, sorted by (tile, depth)."""
        return dptr(self.ids_b) if self.sorted_in_b else dptr(self.ids_a)

    def render(
        mut self,
        gaussians: GaussianBuffers,
        camera: Camera,
        background: SIMD[DType.float32, 4],
        image: DeviceBuffer[DType.float32],
    ) raises:
        """Render owned Gaussians into an owned [H, W, channels] image buffer."""
        if len(image) < camera.width * camera.height * Self.channels:
            raise Error("image buffer too small")
        self.render(gaussians.view(), camera, background, dptr(image))

    def render(
        mut self,
        gaussians: Gaussians,
        camera: Camera,
        background: SIMD[DType.float32, 4],
        image: F32Ptr,
    ) raises:
        """Render into `image` ([H, W, channels] float32, device memory).

        Low-level entry point on raw device pointers (used by the bindings).
        The caller must keep the memory behind `gaussians` and `image` alive
        until the work completes.

        Enqueued on the context's stream. Blocks once (to read back the
        number of tile intersections, needed to size the sort).
        """
        var n = gaussians.count
        self._reserve_gaussians(n)
        var t0 = perf_counter_ns()
        self.project(gaussians, camera, dptr(self.means2d), dptr(self.conics),
                     dptr(self.depths), dptr(self.radii))
        if self.profile:
            self.ctx.synchronize()
        var t1 = perf_counter_ns()
        _ = self.bin(dptr(self.means2d), dptr(self.radii), dptr(self.depths), n,
                     camera.width, camera.height)
        if self.profile:
            self.ctx.synchronize()
        var t2 = perf_counter_ns()
        self.rasterize(dptr(self.means2d), dptr(self.conics), gaussians.colors,
                       gaussians.opacities, background, self.tile_offsets_ptr(),
                       self.sorted_ids(), image, camera.width, camera.height)
        if self.profile:
            self.ctx.synchronize()
            var t3 = perf_counter_ns()
            self.stage_ms = SIMD[DType.float64, 4](
                Float64(t1 - t0), Float64(t2 - t1), Float64(t3 - t2), Float64(t3 - t0)
            ) / 1e6

    # ------------------------------------------------------------------
    # Individual stages (raw device pointers; enqueued on ctx's stream)
    # ------------------------------------------------------------------

    def project(
        mut self,
        gaussians: Gaussians,
        camera: Camera,
        means2d: F32Ptr,
        conics: F32Ptr,
        depths: F32Ptr,
        radii: I32Ptr,
    ) raises:
        """Projection into caller-provided [N,2]/[N,3]/[N]/[N,2] buffers."""
        project_gaussians(
            self.ctx, self.ctx.stream(), gaussians.means, gaussians.scales,
            gaussians.quats, gaussians.opacities, camera.packed(), means2d,
            conics, depths, radii, gaussians.count,
        )

    def bin(
        mut self,
        means2d: F32Ptr,
        radii: I32Ptr,
        depths: F32Ptr,
        n: Int,
        width: Int,
        height: Int,
    ) raises -> Int:
        """Tile binning. Returns the number of intersections M.

        Results stay in renderer memory: `sorted_ids()` ([M]) and
        `tile_offsets_ptr()` ([num_tiles + 1]). Blocks once to read M.
        """
        var ctx = self.ctx
        var stream = ctx.stream()
        var tiles_x = (width + Self.tile_size - 1) // Self.tile_size
        var tiles_y = (height + Self.tile_size - 1) // Self.tile_size
        var n_tiles = tiles_x * tiles_y
        self._reserve_gaussians(n)
        self._reserve_tiles(n_tiles)
        var m = 0
        if n > 0:
            count_tiles[Self.tile_size](
                ctx, stream, means2d, radii, dptr(self.tiles_per_gauss), n,
                tiles_x, tiles_y,
            )
            exclusive_scan[DType.int32](
                ctx, stream, dptr(self.tiles_per_gauss), dptr(self.offsets),
                dptr(self.scan_total), dptr(self.scan_partials), n,
            )
            ctx.enqueue_copy(self.host_total, self.scan_total)
            ctx.synchronize()
            m = Int(self.host_total[0])
        self._reserve_intersections(m)
        self.num_intersections = m
        self.sorted_in_b = False
        if m > 0:
            write_intersections[Self.tile_size](
                ctx, stream, means2d, radii, depths, dptr(self.offsets),
                dptr(self.keys_a), dptr(self.ids_a), n, tiles_x, tiles_y,
            )
            self.sorted_in_b = radix_sort_pairs(
                ctx, stream, dptr(self.keys_a), dptr(self.ids_a),
                dptr(self.keys_b), dptr(self.ids_b), dptr(self.sort_hist),
                dptr(self.sort_partials), dptr(self.sort_total), m,
                sort_end_bit(n_tiles),
            )
        var sorted_keys = dptr(self.keys_b) if self.sorted_in_b else dptr(self.keys_a)
        compute_tile_offsets(ctx, stream, sorted_keys, dptr(self.tile_offsets), m, n_tiles)
        return m

    def tile_offsets_ptr(self) -> I32Ptr:
        """Per-tile start offsets of the last `bin` ([num_tiles + 1])."""
        return dptr(self.tile_offsets)

    def rasterize(
        mut self,
        means2d: F32Ptr,
        conics: F32Ptr,
        colors: F32Ptr,
        opacities: F32Ptr,
        background: SIMD[DType.float32, 4],
        tile_offsets: I32Ptr,
        sorted_ids: I32Ptr,
        image: F32Ptr,
        width: Int,
        height: Int,
    ) raises:
        rasterize[Self.tile_size, Self.channels](
            self.ctx, self.ctx.stream(), means2d, conics, colors, opacities,
            background, tile_offsets, sorted_ids, image, width, height,
        )
