"""Initializing Gaussians from a point cloud (as 3DGS does from SfM points).

Each point becomes an isotropic Gaussian whose scale is the root mean
squared distance to its 3 nearest neighbours (3DGS's simple-knn
`distCUDA2`), identity rotation, opacity 0.1, and the point color as the
DC spherical-harmonics term (higher-order terms zero).

The 3-NN search is brute force on the GPU (exact; every query scans all
points through shared-memory tiles): O(N^2), a few seconds for a few
hundred thousand SfM points, run once.
"""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.math import log, sqrt
from std.memory import stack_allocation

from ..core.math import SH_C0
from ..core.types import F32Ptr, dptr, num_blocks
from ..io.arrays import download, upload
from .splats import Splats

comptime KNN_BLOCK = 256


def _knn3_kernel(points: F32Ptr, dist2: F32Ptr, n: Int32):
    """dist2[i] = mean squared distance from point i to its 3 nearest others."""
    var tile = stack_allocation[KNN_BLOCK * 3, Float32, address_space=AddressSpace.SHARED]()
    var i = block_idx.x * KNN_BLOCK + thread_idx.x
    var active = i < Int(n)
    var px: Float32 = 0
    var py: Float32 = 0
    var pz: Float32 = 0
    if active:
        px = points[unsafe_offset=i * 3 + 0]
        py = points[unsafe_offset=i * 3 + 1]
        pz = points[unsafe_offset=i * 3 + 2]
    var b0 = Float32.MAX
    var b1 = Float32.MAX
    var b2 = Float32.MAX
    var start = 0
    while start < Int(n):
        var j = start + thread_idx.x
        if j < Int(n):
            tile[unsafe_offset=thread_idx.x * 3 + 0] = points[unsafe_offset=j * 3 + 0]
            tile[unsafe_offset=thread_idx.x * 3 + 1] = points[unsafe_offset=j * 3 + 1]
            tile[unsafe_offset=thread_idx.x * 3 + 2] = points[unsafe_offset=j * 3 + 2]
        barrier()
        var m = min(KNN_BLOCK, Int(n) - start)
        if active:
            for k in range(m):
                if start + k == i:
                    continue
                var dx = tile[unsafe_offset=k * 3 + 0] - px
                var dy = tile[unsafe_offset=k * 3 + 1] - py
                var dz = tile[unsafe_offset=k * 3 + 2] - pz
                var d = dx * dx + dy * dy + dz * dz
                if d < b2:
                    if d < b1:
                        b2 = b1
                        if d < b0:
                            b1 = b0
                            b0 = d
                        else:
                            b1 = d
                    else:
                        b2 = d
        barrier()
        start += KNN_BLOCK
    if active:
        var cnt = min(Int(n) - 1, 3)
        var s: Float32 = 0
        if cnt >= 1:
            s += b0
        if cnt >= 2:
            s += b1
        if cnt >= 3:
            s += b2
        dist2[unsafe_offset=i] = s / Float32(max(cnt, 1))


def knn3_mean_dist2(ctx: DeviceContext, xyz: List[Float32]) raises -> List[Float32]:
    """Mean squared distance of every point to its 3 nearest neighbours."""
    var n = len(xyz) // 3
    if n == 0:
        return List[Float32]()
    var pts = upload(ctx, xyz)
    var out = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_function[_knn3_kernel](
        dptr(pts), dptr(out), Int32(n), grid_dim=num_blocks(n, KNN_BLOCK), block_dim=KNN_BLOCK
    )
    var r = download(ctx, out, n)
    _ = pts^
    return r^


def splats_from_points(
    ctx: DeviceContext,
    xyz: List[Float32],
    rgb: List[UInt8],
    sh_degree: Int = 3,
    initial_opacity: Float32 = 0.1,
    capacity: Int = 0,
) raises -> Splats:
    """3DGS initialization from a colored point cloud (xyz [N, 3], rgb [N, 3])."""
    var n = len(xyz) // 3
    var dist2 = knn3_mean_dist2(ctx, xyz)
    var log_scales = List[Float32](capacity=n * 3)
    var rotations = List[Float32](capacity=n * 4)
    var opacity = List[Float32](capacity=n)
    var sh0 = List[Float32](capacity=n * 3)
    var logit = log(initial_opacity / (1.0 - initial_opacity))
    for i in range(n):
        var s = log(sqrt(max(dist2[i], 1e-7)))
        for _ in range(3):
            log_scales.append(s)
        rotations.append(1.0)
        rotations.append(0.0)
        rotations.append(0.0)
        rotations.append(0.0)
        opacity.append(logit)
        for k in range(3):
            sh0.append((Float32(rgb[i * 3 + k]) / 255.0 - 0.5) / SH_C0)
    var rest = (sh_degree + 1) * (sh_degree + 1) - 1
    var sh_rest = List[Float32](length=n * rest * 3, fill=0.0)
    return Splats(ctx, xyz, log_scales, rotations, opacity, sh0, sh_rest, rest, capacity)
