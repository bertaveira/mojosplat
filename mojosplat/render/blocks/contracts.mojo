"""What the render kernels require of a block. Read this file first.

A kernel handles GPU work: threads, culling, memory, tile lists, compositing.
A block is the math for one Gaussian, forward and backward kept together.
Constructing a block runs its forward and keeps what its backward needs, and
`backward` is the pullback, so a kernel chains blocks top to bottom, then
calls `backward` bottom to top. Local variables are the tape: nothing goes to
memory, and everything is inlined, so a block costs what the same expressions
cost written out flat.

Blocks import no GPU primitives. A block touches memory only where the
contract says so: `Response` lays out its own warp-local slab, and an
`Appearance` reads its own columns (higher SH bands are loaded only if active).

Add a block by implementing one of these traits and naming it in a pipeline
(`render/pipelines.mojo`). Metal rules for anything running in a kernel are in
docs/design.md under "Portability".
"""

from ...core.linalg import Sym2, V2, V3, V4
from ...core.types import F32Ptr, SlabPtr
from ..record import REC
from .shape import QuatScale, ShapeGrad


@fieldwise_init
struct GaussianGrad[dt: DType](ImplicitlyCopyable, Movable):
    """dL/d(mean) and dL/d(shape parameters)."""

    var mean: V3[Self.dt]
    var shape: ShapeGrad[Self.dt]


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


trait Response(Deinitable):
    """What a pixel sees of a Gaussian. Float32: this runs inside blend.

    One block spans three kernels, so a footprint's coverage test, its pixel
    evaluation and their derivatives stay together. The blend kernels stay
    hand-tuned around it: they own the tile walk, sub-tile skipping,
    transmittance and snapshots.
    """

    comptime TILE_ARGS: Int  # lanes of `tile_args` that are used
    comptime SLAB: Int  # floats per Gaussian in the blend slab
    comptime GRAD: Int  # per-slot gradient values, see `grad_index`

    # --- Preprocess (forward and backward) ---

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

    def backward(self, grad: SIMD[DType.float32, 8]) -> ScreenGrad:
        """Slot gradients (`GRAD` lanes of `gradient`) -> screen gradients."""
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
    def store_slab(dst: SlabPtr, rec: SIMD[DType.float32, REC], color: SIMD[DType.float32, 4]):
        ...

    @staticmethod
    def alpha_at(src: SlabPtr, px: Float32, py: Float32) -> Float32:
        ...

    @staticmethod
    def channel(src: SlabPtr, k: Int) -> Float32:
        """Color channel `k` as stored by `store_slab`."""
        ...

    # --- Blend backward: one per lane, accumulated over the tile's pixels ---

    def __init__(out self):
        """An empty lane, for the lanes with no Gaussian."""
        ...

    def load(mut self, rec: SIMD[DType.float32, REC]):
        ...

    def alpha_of_pixel(mut self, px: Float32, py: Float32) -> Float32:
        """Evaluates the pixel, keeping what `accumulate` needs."""
        ...

    def accumulate(mut self, alpha: Float32, dL_dalpha: Float32):
        ...

    def gradient(self) -> SIMD[DType.float32, 8]:
        ...

    @staticmethod
    def grad_index(k: Int, n_slots: Int, slot: Int) -> Int:
        """Where slot gradient `k` lives in the scratch buffer."""
        ...


trait Appearance:
    """Per-Gaussian color for a view direction, and its pullback.

    Takes the spherical-harmonics columns for now; `Splats` columns owned by
    the appearance replace `sh0` / `sh_rest`.
    """

    @staticmethod
    def color(
        sh0: F32Ptr, sh_rest: F32Ptr, g: Int, dir: SIMD[DType.float32, 4], active_bases: Int, rest_stride: Int
    ) -> SIMD[DType.float32, 4]:
        ...

    @staticmethod
    def backward(
        sh_rest: F32Ptr,
        grad_sh0: F32Ptr,
        grad_sh_rest: F32Ptr,
        g: Int,
        dir: SIMD[DType.float32, 4],
        grad_color: SIMD[DType.float32, 4],
        active_bases: Int,
        rest_stride: Int,
        write_rest: Bool,
    ) -> SIMD[DType.float32, 4]:
        """Writes the appearance gradients of `g`; returns dL/d(mean) through `dir`."""
        ...


trait Pipeline:
    """A renderer: which blocks the kernels are built from."""

    comptime Place: Placement
    comptime Resp: Response
    comptime Appear: Appearance
