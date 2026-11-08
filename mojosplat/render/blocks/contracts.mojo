"""What the render kernels require of a block. Read this file first.

A kernel handles GPU work: threads, culling, memory, tile lists, compositing.
A block is the math for one Gaussian, forward and backward kept together.
Constructing a block runs its forward and keeps what its backward needs, and
`backward` is the pullback, so a kernel chains blocks top to bottom, then
calls `backward` bottom to top. Local variables are the tape: nothing goes to
memory, and everything is inlined, so a block costs what the same expressions
cost written out flat.

Blocks import no GPU primitives. A block touches memory only where the
contract says so: `Response` lays out its own warp-local slab, and `Response`
and `Appearance` read their own columns.

Add a block by implementing one of these traits and naming it in a pipeline
(`render/pipelines.mojo`). Metal rules for anything running in a kernel are in
docs/design.md under "Portability".
"""

from ...core.linalg import Sym2, V2, V3, V4
from ...core.splats import AppearanceSettings, ExtraColumns
from ...core.types import SlabPtr
from ..record import REC
from .shape import QuatScale, ShapeGrad


# What `LoadedResponse.eval` returns: lane 0 is alpha, the rest is the block's own.
comptime Hit = SIMD[DType.float32, 8]


@fieldwise_init
struct GaussianGrad[dt: DType](ImplicitlyCopyable, Movable):
    """dL/d(mean) and dL/d(shape parameters)."""

    var mean: V3[Self.dt]
    var shape: ShapeGrad[Self.dt]


@fieldwise_init
struct ColumnSpec(Copyable, Movable):
    """A `Splats` column a response needs for training."""

    var name: String
    var width: Int
    var trainable: Bool
    var init_value: Float32
    var lr: Float32


@fieldwise_init
struct ScreenGrad(ImplicitlyCopyable, Movable):
    """dL/d(image-space position and covariance) of one Gaussian."""

    var mean2d: V2[DType.float32]
    var cov2d: Sym2[DType.float32]


trait Placement(Deinitable):
    """Where a Gaussian lands: 3D Gaussian -> image-space position and size."""

    comptime dtype: DType

    @staticmethod
    def depth(m: V3[Self.dtype], cam: SIMD[Self.dtype, 32]) -> Scalar[Self.dtype]:
        """Value tested against the near/far planes and used as the sort key."""
        ...

    def __init__(out self, m: V3[Self.dtype], shape: QuatScale[Self.dtype], cam: SIMD[Self.dtype, 32]):
        ...

    def mean2d(self) -> V2[Self.dtype]:
        ...

    def cov2d(self) -> Sym2[Self.dtype]:
        ...

    def backward(
        self, shape: QuatScale[Self.dtype], g_mean2d: V2[Self.dtype], g_cov2d: Sym2[Self.dtype]
    ) -> GaussianGrad[Self.dtype]:
        ...


trait LoadedResponse(Deinitable):
    """One Gaussian ready to be evaluated at pixels, in both blend kernels.

    `eval` returns a `Hit` for the backward to pass to `add`, so the gradient
    sums need no hidden per-pixel state.
    """

    def __init__(out self):
        """An empty lane, for the lanes with no Gaussian."""
        ...

    def __init__(out self, src: SlabPtr):
        """From the blend slab (`Response.store_slab`)."""
        ...

    def __init__(out self, rec: SIMD[DType.float32, REC], extra: SIMD[DType.float32, 8]):
        """From the record and the response's per-slot values."""
        ...

    def eval(self, px: Float32, py: Float32) -> Hit:
        ...

    def add(mut self, hit: Hit, dL_dalpha: Float32):
        """Backward: add this pixel's contribution to the gradient sums."""
        ...

    def gradient(self) -> SIMD[DType.float32, 8]:
        """The summed slot gradients (`Response.GRAD` lanes)."""
        ...


trait Response(Deinitable):
    """What a pixel sees of a Gaussian. Float32: this runs inside blend.

    A footprint's coverage test, pixel evaluation and derivatives stay together
    in one file. The blend kernels stay hand-tuned around it: they own the tile
    walk, sub-tile skipping, transmittance and snapshots.
    """

    comptime Loaded: LoadedResponse
    comptime TILE_ARGS: Int  # lanes of `tile_args` that are used
    comptime SLAB: Int  # floats per Gaussian in the blend slab
    comptime GRAD: Int  # per-slot gradient values, see `grad_index`
    comptime EXTRA: Int  # per-slot floats saved besides the record (<= 8)
    comptime NUM_COLUMNS: Int  # `Splats` columns owned by the response
    comptime MIN_OPACITY: Float32  # Gaussians below this are culled
    comptime STOP_BEFORE_LAST: Bool  # True: a fragment that would saturate the pixel is not applied

    @staticmethod
    def skip_fragment(alpha: Float32) -> Bool:
        """True if a fragment with this alpha contributes nothing."""
        ...

    @staticmethod
    def columns() -> List[ColumnSpec]:
        """The columns `Trainer` adds for this response, after the standard ones."""
        ...

    # --- Preprocess ---

    def __init__(out self, mean2d: V2[DType.float32], cov2d: Sym2[DType.float32]):
        ...

    def valid(self) -> Bool:
        """False for a degenerate footprint, which is culled."""
        ...

    def extent(self, opacity: Float32) -> SIMD[DType.float32, 2]:
        """Half-size in pixels of the rectangle holding every visible pixel."""
        ...

    def record(self, opacity: Float32) -> SIMD[DType.float32, REC]:
        """The per-slot payload (bounds lanes are filled in by the kernel)."""
        ...

    @staticmethod
    def extra(cols: ExtraColumns, g: Int) -> SIMD[DType.float32, 8]:
        """The `EXTRA` per-slot values of Gaussian `g`, from the response's columns."""
        ...

    @staticmethod
    def tile_args(rec: SIMD[DType.float32, REC]) -> SIMD[DType.float32, 8]:
        """What `touches_tile` needs, shared across a warp when it helps."""
        ...

    @staticmethod
    def touches_tile[tile_w: Int, tile_h: Int](args: SIMD[DType.float32, 8], tile_x: Int, tile_y: Int) -> Bool:
        ...

    # --- Blend ---

    @staticmethod
    def store_slab(
        dst: SlabPtr, rec: SIMD[DType.float32, REC], extra: SIMD[DType.float32, 8], color: SIMD[DType.float32, 4]
    ):
        ...

    @staticmethod
    def channel(src: SlabPtr, k: Int) -> Float32:
        """Color channel `k` as stored by `store_slab`."""
        ...

    # --- Preprocess backward ---

    def backward(self, grad: SIMD[DType.float32, 8]) -> ScreenGrad:
        """Slot gradients (`GRAD` lanes of `Loaded.gradient`) -> screen gradients."""
        ...

    @staticmethod
    def opacity_gradient(slot_grad: SIMD[DType.float32, 8]) -> Float32:
        """dL/d(opacity logit) from the slot gradients."""
        ...

    @staticmethod
    def write_gradients(cols: ExtraColumns, grads: ExtraColumns, g: Int, slot_grad: SIMD[DType.float32, 8]):
        """Write the gradients of the response's own columns."""
        ...

    @staticmethod
    def zero_gradients(grads: ExtraColumns, g: Int):
        """Write zero gradients of the response's columns for a culled Gaussian."""
        ...

    @staticmethod
    def grad_index(k: Int, n_slots: Int, slot: Int) -> Int:
        """Where slot gradient `k` lives in the scratch buffer."""
        ...


trait Appearance:
    """Per-Gaussian color for a view direction, and its pullback.

    An appearance owns the first `NUM_COLUMNS` columns after the core ones,
    and up to four run-time integers.
    """

    comptime NUM_COLUMNS: Int

    @staticmethod
    def columns() -> List[ColumnSpec]:
        """Columns `Trainer` adds for this appearance, before the response's."""
        ...

    @staticmethod
    def init_column(index: Int, count: Int) -> List[Float32]:
        """Starting values of column `index` of `columns()`."""
        ...

    @staticmethod
    def color(
        cols: ExtraColumns, g: Int, dir: SIMD[DType.float32, 4], settings: AppearanceSettings
    ) -> SIMD[DType.float32, 4]:
        ...

    @staticmethod
    def backward(
        cols: ExtraColumns,
        grads: ExtraColumns,
        g: Int,
        dir: SIMD[DType.float32, 4],
        grad_color: SIMD[DType.float32, 4],
        settings: AppearanceSettings,
        write_all: Bool,
    ) -> SIMD[DType.float32, 4]:
        """Writes the gradients of `g`; returns dL/d(mean) through `dir`.
        `write_all=False` skips columns a fused optimizer recomputes."""
        ...

    @staticmethod
    def zero_gradients(grads: ExtraColumns, g: Int, settings: AppearanceSettings, write_all: Bool):
        """Writes zero gradients for a culled Gaussian."""
        ...


trait Pipeline:
    """A renderer: which blocks the kernels are built from."""

    comptime ID: Int  # checked between a forward and its backward
    comptime Place: Placement
    comptime Resp: Response
    comptime Appear: Appearance
