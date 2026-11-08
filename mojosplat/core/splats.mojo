"""Raw-pointer view of N Gaussians' parameters."""

from .math import sh_num_bases
from .types import F32Ptr, placeholder

comptime NUM_EXTRA_COLUMNS = 4  # kernel arguments reserved for columns after the core ones
comptime AppearanceSettings = SIMD[DType.int32, 4]  # run-time integers of an appearance


struct ExtraColumns(ImplicitlyCopyable, Movable):
    """Device pointers to the columns after the core ones: the appearance's
    first, then the response's. Unused ones are placeholders; the fixed count
    keeps the kernel signatures plain pointers."""

    var c0: F32Ptr
    var c1: F32Ptr
    var c2: F32Ptr
    var c3: F32Ptr

    @always_inline
    def __init__(
        out self,
        c0: F32Ptr = placeholder[DType.float32](),
        c1: F32Ptr = placeholder[DType.float32](),
        c2: F32Ptr = placeholder[DType.float32](),
        c3: F32Ptr = placeholder[DType.float32](),
    ):
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    @always_inline
    def skip(self, k: Int) -> Self:
        """The columns after the first `k`, so a block sees its own from c0."""
        if k == 0:
            return self
        if k == 1:
            return Self(self.c1, self.c2, self.c3)
        if k == 2:
            return Self(self.c2, self.c3)
        if k == 3:
            return Self(self.c3)
        return Self()

    def set(mut self, i: Int, p: F32Ptr):
        if i == 0:
            self.c0 = p
        elif i == 1:
            self.c1 = p
        elif i == 2:
            self.c2 = p
        else:
            self.c3 = p


struct SplatView(ImplicitlyCopyable, Movable):
    """Device pointers to N Gaussians' raw parameters (float32, SoA).

    Core columns: means [N,3], log_scales [N,3], rotations [N,4] (w,x,y,z; any
    norm), opacity_logits [N]. The rest belong to the appearance and the response;
    for spherical harmonics, sh0 [N,3] and sh_rest [N, rest_stride, 3].
    """

    var means: F32Ptr
    var log_scales: F32Ptr
    var rotations: F32Ptr
    var opacity_logits: F32Ptr
    var extras: ExtraColumns
    var count: Int
    var rest_stride: Int  # SH bases stored per Gaussian in sh_rest

    def __init__(
        out self,
        means: F32Ptr,
        log_scales: F32Ptr,
        rotations: F32Ptr,
        opacity_logits: F32Ptr,
        extras: ExtraColumns,
        count: Int,
        rest_stride: Int = 0,
    ):
        self.means = means
        self.log_scales = log_scales
        self.rotations = rotations
        self.opacity_logits = opacity_logits
        self.extras = extras
        self.count = count
        self.rest_stride = rest_stride

    def __init__(
        out self,
        means: F32Ptr,
        log_scales: F32Ptr,
        rotations: F32Ptr,
        opacity_logits: F32Ptr,
        sh0: F32Ptr,
        sh_rest: F32Ptr,
        count: Int,
        rest_stride: Int,
    ):
        """The spherical-harmonics layout."""
        self = Self(means, log_scales, rotations, opacity_logits, ExtraColumns(sh0, sh_rest), count, rest_stride)

    def max_sh_degree(self) -> Int:
        var d = 0
        while sh_num_bases(d + 1) - 1 <= self.rest_stride and d < 3:
            d += 1
        return d
