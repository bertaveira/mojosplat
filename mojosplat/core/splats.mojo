"""Raw-pointer view of N Gaussians' trainable parameters."""

from .types import F32Ptr
from .math import sh_num_bases


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
