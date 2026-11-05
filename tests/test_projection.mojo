from max.gpu.host import DeviceContext
from std.math import exp
from std.testing import assert_equal, assert_true, TestSuite

from mojosplat.common import dptr
from mojosplat.io import upload, download
from mojosplat.projection import project_gaussians
from golden import GoldenScene


def check_projection(name: String) raises:
    var g = GoldenScene(name)
    var ctx = DeviceContext()
    var log_scales = g.f32("log_scales.f32")
    var scales = List[Float32](capacity=len(log_scales))
    for v in log_scales:
        scales.append(exp(v))

    var d_means3d = upload(ctx, g.f32("means3d.f32"))
    var d_scales = upload(ctx, scales)
    var d_quats = upload(ctx, g.f32("quats.f32"))
    var d_opac = upload(ctx, g.f32("opacities.f32"))
    var n = g.n
    var d_means2d = ctx.enqueue_create_buffer[DType.float32](n * 2)
    var d_conics = ctx.enqueue_create_buffer[DType.float32](n * 3)
    var d_depths = ctx.enqueue_create_buffer[DType.float32](n)
    var d_radii = ctx.enqueue_create_buffer[DType.int32](n * 2)

    project_gaussians(
        ctx, ctx.stream(),
        dptr(d_means3d), dptr(d_scales), dptr(d_quats),
        dptr(d_opac), g.camera.packed(),
        dptr(d_means2d), dptr(d_conics), dptr(d_depths),
        dptr(d_radii), n,
    )
    ctx.synchronize()

    var radii = download(ctx, d_radii, n * 2)
    var means2d = download(ctx, d_means2d, n * 2)
    var conics = download(ctx, d_conics, n * 3)
    var depths = download(ctx, d_depths, n)
    var ref_radii = g.i32("radii.i32")
    var ref_means2d = g.f32("means2d.f32")
    var ref_conics = g.f32("conics.f32")
    var ref_depths = g.f32("depths.f32")

    var radius_mismatch = 0
    var visible = 0
    var max_err: Float32 = 0.0
    for i in range(n):
        var vis = radii[2 * i] > 0 and radii[2 * i + 1] > 0
        var ref_vis = ref_radii[2 * i] > 0 and ref_radii[2 * i + 1] > 0
        assert_equal(vis, ref_vis, String("visibility differs for gaussian ", i))
        if not vis:
            continue
        visible += 1
        if radii[2 * i] != ref_radii[2 * i] or radii[2 * i + 1] != ref_radii[2 * i + 1]:
            radius_mismatch += 1
        for k in range(2):
            max_err = max(max_err, abs(means2d[2 * i + k] - ref_means2d[2 * i + k]))
        for k in range(3):
            var r = ref_conics[3 * i + k]
            max_err = max(max_err, abs(conics[3 * i + k] - r) / max(abs(r), 1e-3))
        max_err = max(max_err, abs(depths[i] - ref_depths[i]))
    print(name, ": visible", visible, "radius mismatches", radius_mismatch, "max err", max_err)
    assert_equal(radius_mismatch, 0)
    assert_true(max_err < 1e-4, String("max error too large: ", max_err))


def test_projection_small() raises:
    check_projection("small")


def test_projection_medium() raises:
    check_projection("medium")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
