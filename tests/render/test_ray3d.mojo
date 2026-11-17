"""3DGUT's ray response: against NVIDIA's definition evaluated directly in float64, its
bounds, finite differences of the ratio-form coefficients (float64), of the per-pixel
gradients and of the whole response backward (float32), the densification statistic,
the unscented placement, and the Gut3D pipeline on the GPU against a direct
compositing of the same definition."""

from max.gpu.host import DeviceContext
from std.math import ceil, exp, floor, log, recip, sqrt
from std.testing import assert_equal, assert_true, TestSuite

from mojosplat.core.camera import Camera
from mojosplat.core.linalg import Mat3, V3, V4
from mojosplat.core.math import SH_C0
from mojosplat.core.splats import ExtraColumns
from mojosplat.core.types import dptr
from mojosplat.gpu.elementwise import fill_f32
from mojosplat.io.arrays import download, upload
from mojosplat.model.splats import Attr, Splats
from mojosplat.render.blocks.contracts import GaussianGeometry
from mojosplat.render.blocks.placement.pinhole_ewa import PinholeEWA
from mojosplat.render.blocks.placement.pinhole_ut import PinholeUT
from mojosplat.render.blocks.response.ray3d import GaussianRay3D, LoadedRay3D, RayCoeffs
from mojosplat.render.blocks.shape import QuatScale
from mojosplat.render.pipelines import Gut3D
from mojosplat.render.rasterizer import Rasterizer

comptime F64 = DType.float64
comptime F32 = DType.float32


def camera() -> SIMD[F64, 32]:
    var c = SIMD[F64, 32](0.0)
    # A slightly rotated camera.
    c[0] = 0.8
    c[1] = 0.6
    c[2] = 0.0
    c[3] = 0.1
    c[4] = -0.36
    c[5] = 0.48
    c[6] = 0.8
    c[7] = -0.3
    c[8] = 0.48
    c[9] = -0.64
    c[10] = 0.6
    c[11] = 2.0
    c[12] = 700.0
    c[13] = 690.0
    c[14] = 640.0
    c[15] = 360.0
    return c


comptime X = SIMD[F64, 16]


def coeffs(x: X) -> SIMD[F64, 16]:
    var shape = QuatScale[F64](V3[F64](x[3], x[4], x[5]), V4[F64](x[6], x[7], x[8], x[9]))
    var r = RayCoeffs[F64](V3[F64](x[0], x[1], x[2]), shape.R, shape.v0, shape.v1, shape.v2, camera())
    var o = SIMD[F64, 16](0.0)
    o[0] = r.mx
    o[1] = r.my
    o[2] = r.n11
    o[3] = r.n12
    o[4] = r.n22
    o[5] = r.e1
    o[6] = r.e2
    o[7] = r.f11
    o[8] = r.f12
    o[9] = r.f22
    return o


def test_ray_coefficient_pullback() raises:
    var x = X(0.31, -0.42, 0.9, -2.2, -3.1, -4.0, 0.4, -0.7, 0.2, 0.55, 0, 0, 0, 0, 0, 0)
    var h: Float64 = 1e-6
    var worst: Float64 = 0.0
    for k in range(10):
        var seed = SIMD[F64, 16](0.0)
        seed[k] = 1.0
        var shape = QuatScale[F64](V3[F64](x[3], x[4], x[5]), V4[F64](x[6], x[7], x[8], x[9]))
        var r = RayCoeffs[F64](V3[F64](x[0], x[1], x[2]), shape.R, shape.v0, shape.v1, shape.v2, camera())
        var g = r.backward(seed[0], seed[1], seed[2], seed[3], seed[4], seed[5], seed[6], seed[7], seed[8], seed[9])
        var q = shape.rotation_backward(g.dR)
        var analytic = SIMD[F64, 16](
            g.mean.x,
            g.mean.y,
            g.mean.z,
            g.log_scales.x,
            g.log_scales.y,
            g.log_scales.z,
            q.w,
            q.x,
            q.y,
            q.z,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        for i in range(10):
            var up = x
            var dn = x
            up[i] += h
            dn[i] -= h
            var numeric = (coeffs(up)[k] - coeffs(dn)[k]) / (2.0 * h)
            var scale = max(max(abs(numeric), abs(analytic[i])), 1e-3 * max(abs(coeffs(x)[k]), 1e-9))
            var err = abs(numeric - analytic[i]) / scale
            worst = max(worst, err)
            assert_true(err < 2e-4, String("coefficient ", k, " input ", i, ": ", analytic[i], " vs ", numeric))
    print("RayCoeffs pullback worst relative error", worst)


def extra_of(p: SIMD[F32, 16]) -> SIMD[F32, 16]:
    return p


def alpha_of(p: SIMD[F32, 16], px: Float32, py: Float32) -> Float32:
    var e = p
    e[10] = Float32(1.0 / (1.0 + exp(-Float64(p[10]))))
    return LoadedRay3D(SIMD[F32, 8](0.0), e).eval(px, py)[0]


def test_loaded_ray_pixel_gradient() raises:
    var p = SIMD[F32, 16](10.0, 20.0, -0.11, 0.03, -0.09, 0.012, -0.007, 0.0004, 0.0002, 0.0005, 0.7, 0, 0, 0, 0, 0)
    var h: Float32 = 0.005
    var worst: Float64 = 0.0
    var pixels: List[SIMD[F32, 2]] = [SIMD[F32, 2](11.5, 19.5), SIMD[F32, 2](8.0, 23.0)]
    for pixel in pixels:
        var e = p
        e[10] = Float32(1.0 / (1.0 + exp(-Float64(p[10]))))
        var loaded = LoadedRay3D(SIMD[F32, 8](0.0), e)
        loaded.add(loaded.eval(pixel[0], pixel[1]), 1.0)
        var g = loaded.gradient()
        for i in range(11):
            var up = p
            var dn = p
            var step = h if i != 7 and i != 8 and i != 9 else 2e-4
            up[i] += step
            dn[i] -= step
            var numeric = (Float64(alpha_of(up, pixel[0], pixel[1])) - Float64(alpha_of(dn, pixel[0], pixel[1]))) / (
                2.0 * Float64(step)
            )
            var scale = max(abs(numeric), abs(Float64(g[i])), 1e-3)
            var err = abs(numeric - Float64(g[i])) / scale
            worst = max(worst, err)
            assert_true(err < 1e-2, String("lane ", i, ": ", g[i], " vs ", numeric))
    print("LoadedRay3D worst relative error", worst)


# --- The whole response, against NVIDIA's definition ---

comptime P11 = SIMD[F64, 16]  # mean (3), log scales (3), quaternion wxyz (4), opacity logit (1)


def gut_camera() -> Camera:
    """A rotated pinhole camera, 128 x 96 (rows of the rotation are orthonormal)."""
    var v = SIMD[F32, 16](0.8, 0.6, 0.0, 0.1, -0.36, 0.48, 0.8, -0.3, 0.48, -0.64, 0.6, 2.0, 0, 0, 0, 1)
    return Camera(v, 140.0, 138.0, 64.0, 48.0, 128, 96, 0.2, 1e10)


def sigmoid(x: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-x))


def nvidia_alpha(p: P11, cam: Camera, px: Float64, py: Float64) -> Float64:
    """3dgrut's kernel-degree-2 hit, written out in float64: the ray through pixel
    (px, py) from the camera center, its squared distance to the Gaussian center in
    canonical space |d x o|^2, response exp(-dist2 / 2), alpha = min(0.99, opacity *
    response), accepted when response > 0.0113 and alpha > 1/255. The rotation is
    3dgrut's `quaternion_to_so3`."""
    var v = cam.view
    var Rc = SIMD[F64, 16](0.0)
    for i in range(12):
        Rc[i] = Float64(v[i])
    # Camera center -R^T t and the world direction R^T d_cam.
    var dc = SIMD[F64, 4]((px - Float64(cam.cx)) / Float64(cam.fx), (py - Float64(cam.cy)) / Float64(cam.fy), 1.0, 0.0)
    var o = SIMD[F64, 4](0.0)
    var d = SIMD[F64, 4](0.0)
    for k in range(3):
        o[k] = -(Rc[k] * Rc[3] + Rc[4 + k] * Rc[7] + Rc[8 + k] * Rc[11])
        d[k] = Rc[k] * dc[0] + Rc[4 + k] * dc[1] + Rc[8 + k] * dc[2]
    var n = sqrt(p[6] * p[6] + p[7] * p[7] + p[8] * p[8] + p[9] * p[9])
    var r = p[6] / n
    var x = p[7] / n
    var y = p[8] / n
    var z = p[9] / n
    var R = SIMD[F64, 16](
        1 - 2 * (y * y + z * z),
        2 * (x * y - r * z),
        2 * (x * z + r * y),
        0,
        2 * (x * y + r * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z - r * x),
        0,
        2 * (x * z - r * y),
        2 * (y * z + r * x),
        1 - 2 * (x * x + y * y),
        0,
        0,
        0,
        0,
        0,
    )
    # Canonical origin and direction: S^-1 R^T (o - m), S^-1 R^T d.
    var oc = SIMD[F64, 4](0.0)
    var dd = SIMD[F64, 4](0.0)
    for k in range(3):
        var s = exp(p[3 + k])
        var a: Float64 = 0.0
        var b: Float64 = 0.0
        for j in range(3):
            a += R[j * 4 + k] * (o[j] - p[j])
            b += R[j * 4 + k] * d[j]
        oc[k] = a / s
        dd[k] = b / s
    dd /= sqrt((dd * dd).reduce_add())
    var cx = dd[1] * oc[2] - dd[2] * oc[1]
    var cy = dd[2] * oc[0] - dd[0] * oc[2]
    var cz = dd[0] * oc[1] - dd[1] * oc[0]
    var response = exp(-0.5 * (cx * cx + cy * cy + cz * cz))
    var alpha = min(0.99, sigmoid(p[10]) * response)
    if response > 0.0113 and alpha > 1.0 / 255.0:
        return alpha
    return 0.0


@fieldwise_init
struct Ours(Movable):
    var place: PinholeUT[F32]
    var shape: QuatScale[F32]
    var resp: GaussianRay3D
    var loaded: LoadedRay3D
    var opacity: Float32


def ours(p: P11, cam: SIMD[F32, 32]) -> Ours:
    """The response as the kernels build it, from parameters `p`."""
    var m = V3[F32](Float32(p[0]), Float32(p[1]), Float32(p[2]))
    var shape = QuatScale[F32](
        V3[F32](Float32(p[3]), Float32(p[4]), Float32(p[5])),
        V4[F32](Float32(p[6]), Float32(p[7]), Float32(p[8]), Float32(p[9])),
    )
    var place = PinholeUT[F32](m, shape, cam)
    var resp = GaussianRay3D(place.mean2d(), place.cov2d(), GaussianGeometry.of(m, shape), cam)
    var opacity = Float32(sigmoid(p[10]))
    var loaded = LoadedRay3D(resp.record(opacity), resp.extra(opacity, ExtraColumns(), 0))
    return Ours(place, shape, resp, loaded, opacity)


def gaussians() -> List[P11]:
    var g = List[P11]()
    g.append(P11(0.31, -0.42, 0.9, -2.2, -3.1, -4.0, 0.4, -0.7, 0.2, 0.55, 0.4, 0, 0, 0, 0, 0))
    g.append(P11(-0.2, 0.1, 0.4, -1.6, -1.9, -2.3, 0.9, 0.1, 0.3, -0.2, 2.5, 0, 0, 0, 0, 0))  # nearly opaque
    g.append(P11(0.05, 0.2, 1.2, -1.5, -6.5, -2.0, 0.2, 0.6, -0.5, 0.1, 1.0, 0, 0, 0, 0, 0))  # thin
    return g^


def test_alpha_matches_nvidia_definition() raises:
    var cam = gut_camera()
    var c = cam.packed()
    var worst: Float64 = 0.0
    var hits = 0
    for p in gaussians():
        var o = ours(p, c)
        for y in range(cam.height):
            for x in range(cam.width):
                var px = Float64(x) + 0.5
                var py = Float64(y) + 0.5
                var want = nvidia_alpha(p, cam, px, py)
                var got = Float64(o.loaded.eval(Float32(px), Float32(py))[0])
                if want > 0.0:
                    hits += 1
                # Pixels at the 0.0113 cutoff may flip; elsewhere alpha must agree.
                if (want > 0.0) != (got > 0.0):
                    assert_true(abs(want - got) < 0.02, String("acceptance at ", x, ", ", y, ": ", got, " vs ", want))
                    continue
                worst = max(worst, abs(got - want))
    print("  alpha vs NVIDIA's definition: worst abs error", worst, "over", hits, "accepted pixels")
    assert_true(hits > 500, "the Gaussians cover pixels")
    assert_true(worst < 2e-5, "alpha")


def test_bounds_hold_every_accepted_pixel() raises:
    """Preprocess keeps a pixel x when floor(mx - ext) <= x < ceil(mx + ext); blend skips the rest."""
    var cam = gut_camera()
    var c = cam.packed()
    for p in gaussians():
        var o = ours(p, c)
        var ext = o.resp.extent(o.opacity)
        var mx = o.place.mean2d().x
        var my = o.place.mean2d().y
        for y in range(cam.height):
            for x in range(cam.width):
                if nvidia_alpha(p, cam, Float64(x) + 0.5, Float64(y) + 0.5) > 0.0:
                    assert_true(floor(mx - ext[0]) <= Float32(x) and Float32(x) < ceil(mx + ext[0]), "x in bounds")
                    assert_true(floor(my - ext[1]) <= Float32(y) and Float32(y) < ceil(my + ext[1]), "y in bounds")


def alpha32(p: P11, cam: SIMD[F32, 32], px: Float32, py: Float32) -> Float64:
    return Float64(ours(p, cam).loaded.eval(px, py)[0])


def test_response_backward_matches_finite_differences() raises:
    """Slot gradients -> `GaussianRay3D.backward` -> mean, log scales, quaternion, opacity logit."""
    var c = gut_camera().packed()
    var worst: Float64 = 0.0
    var gs = gaussians()
    for gi in range(2):
        var p = gs[gi]
        var o = ours(p, c)
        var centre = o.place.mean2d()
        var pixels: List[SIMD[F32, 2]] = [
            SIMD[F32, 2](centre.x + 2.5, centre.y - 1.5),
            SIMD[F32, 2](centre.x - 4.5, centre.y + 3.5),
        ]
        for pixel in pixels:
            var hit = o.loaded.eval(pixel[0], pixel[1])
            assert_true(hit[0] > 0.0 and hit[6] > 0.0, "an accepted, unclamped pixel")
            var loaded = o.loaded
            loaded.add(hit, 1.0)
            var slot = loaded.gradient()
            var rg = o.resp.backward(slot, o.place, o.shape, c)
            var g = rg.grad
            var analytic = SIMD[F64, 16](
                Float64(g.mean.x),
                Float64(g.mean.y),
                Float64(g.mean.z),
                Float64(g.shape.log_scales.x),
                Float64(g.shape.log_scales.y),
                Float64(g.shape.log_scales.z),
                Float64(g.shape.quat.w),
                Float64(g.shape.quat.x),
                Float64(g.shape.quat.y),
                Float64(g.shape.quat.z),
                Float64(GaussianRay3D.opacity_gradient(slot)),
                0,
                0,
                0,
                0,
                0,
            )
            assert_equal(rg.screen_norm, 0.0)
            for i in range(11):
                var h: Float64 = 2e-3
                var up = p
                var dn = p
                up[i] += h
                dn[i] -= h
                var numeric = (alpha32(up, c, pixel[0], pixel[1]) - alpha32(dn, c, pixel[0], pixel[1])) / (2.0 * h)
                var scale = max(abs(numeric), abs(analytic[i]), 1e-2)
                var err = abs(numeric - analytic[i]) / scale
                worst = max(worst, err)
                assert_true(err < 2e-2, String("Gaussian ", gi, " input ", i, ": ", analytic[i], " vs ", numeric))
    print("  GaussianRay3D backward worst relative error", worst)


def test_densification_statistic() raises:
    """NVIDIA's: |dL/dmean| x distance to the camera / 2, counted when the gradient is not zero."""
    var c = gut_camera().packed()
    var m = V3[F32](1.0, 2.0, 3.0)
    var zero = GaussianRay3D.densification(0.0, V3[F32](0.0, 0.0, 0.0), m, c)
    assert_equal(zero[0], 0.0)
    assert_equal(zero[1], 0.0)
    var s = GaussianRay3D.densification(0.0, V3[F32](0.3, 0.0, -0.4), m, c)
    var dx = Float64(1.0 - c[24])
    var dy = Float64(2.0 - c[25])
    var dz = Float64(3.0 - c[26])
    assert_equal(s[0], 1.0)
    assert_true(abs(Float64(s[1]) - 0.5 * 0.5 * sqrt(dx * dx + dy * dy + dz * dz)) < 1e-5, "statistic")


def test_unscented_placement_of_a_small_gaussian() raises:
    """For a Gaussian small against its depth the projection is nearly linear, so the
    unscented proxy matches the linearized (EWA) one, shifted by half a pixel."""
    var c = gut_camera().packed()
    var m = V3[F32](0.2, -0.1, 0.5)
    var shape = QuatScale[F32](V3[F32](-6.0, -6.5, -7.0), V4[F32](0.9, 0.2, -0.3, 0.1))
    var ut = PinholeUT[F32](m, shape, c)
    var ewa = PinholeEWA[F32](m, shape, c)
    assert_true(abs(ut.mean2d().x + 0.5 - ewa.mean2d().x) < 1e-3, "mean x")
    assert_true(abs(ut.mean2d().y + 0.5 - ewa.mean2d().y) < 1e-3, "mean y")
    var a = ut.cov2d()
    var b = ewa.cov2d()
    var size = max(abs(b.a), abs(b.c))
    assert_true(abs(a.a - b.a) < 1e-2 * size, String("covariance a ", a.a, " vs ", b.a))
    assert_true(abs(a.b - b.b) < 1e-2 * size, String("covariance b ", a.b, " vs ", b.b))
    assert_true(abs(a.c - b.c) < 1e-2 * size, String("covariance c ", a.c, " vs ", b.c))


# --- The Gut3D pipeline on the GPU ---


def splats_of(ctx: DeviceContext, ps: List[P11], colors: List[Float32]) raises -> Splats:
    var m = List[Float32]()
    var s = List[Float32]()
    var q = List[Float32]()
    var o = List[Float32]()
    var sh0 = List[Float32]()
    for i in range(len(ps)):
        var p = ps[i]
        for k in range(3):
            m.append(Float32(p[k]))
            s.append(Float32(p[3 + k]))
            sh0.append((colors[i * 3 + k] - 0.5) / Float32(SH_C0))
        for k in range(4):
            q.append(Float32(p[6 + k]))
        o.append(Float32(p[10]))
    return Splats(ctx, m, s, q, o, sh0, List[Float32](), 0)


def depth(p: P11, cam: Camera) -> Float64:
    var v = cam.view
    return Float64(v[8]) * p[0] + Float64(v[9]) * p[1] + Float64(v[10]) * p[2] + Float64(v[11])


def test_gut3d_render_matches_direct_compositing() raises:
    """Every pixel against the alpha of NVIDIA's definition, composited front to back
    (global depth order, colors from SH degree 0, black background)."""
    var cam = gut_camera()
    var ps = gaussians()
    var colors: List[Float32] = [0.9, 0.2, 0.1, 0.1, 0.8, 0.3, 0.2, 0.3, 0.95]
    var ctx = DeviceContext()
    var splats = splats_of(ctx, ps, colors)
    var npx = cam.width * cam.height * 3
    var image = ctx.enqueue_create_buffer[DType.float32](npx)
    var r = Rasterizer[16, Gut3D.TILE_H](ctx)
    r.render[Gut3D](splats.view(), cam, SIMD[F32, 4](0.0), dptr(image), 0)
    ctx.synchronize()
    var got = download(ctx, image, npx)
    var order: List[Int] = [0, 1, 2]
    for i in range(3):  # sort by depth
        for j in range(2 - i):
            if depth(ps[order[j]], cam) > depth(ps[order[j + 1]], cam):
                order.swap_elements(j, j + 1)
    var worst: Float64 = 0.0
    for y in range(cam.height):
        for x in range(cam.width):
            var T: Float64 = 1.0
            var rgb = SIMD[F64, 4](0.0)
            for g in order:
                var a = nvidia_alpha(ps[g], cam, Float64(x) + 0.5, Float64(y) + 0.5)
                for k in range(3):
                    rgb[k] += Float64(colors[g * 3 + k]) * a * T
                T *= 1.0 - a
            for k in range(3):
                worst = max(worst, abs(Float64(got[(y * cam.width + x) * 3 + k]) - rgb[k]))
    print("  Gut3D image vs direct compositing: worst abs error", worst)
    assert_true(worst < 0.02, "image")  # pixels at the acceptance cutoff may differ by one alpha step
    _ = (image^, splats^)


def test_gut3d_backward_matches_the_block_chain() raises:
    """The GPU blend and preprocess backward against the same chain on the CPU: front-to-back
    compositing gradients per pixel (dL/dalpha), `LoadedRay3D.add`, then
    `GaussianRay3D.backward`. (Finite differences of the whole image would also count
    pixels crossing the acceptance cutoff, where alpha jumps.)"""
    var cam = gut_camera()
    var c = cam.packed()
    var ps = gaussians()
    var colors: List[Float32] = [0.9, 0.2, 0.1, 0.1, 0.8, 0.3, 0.2, 0.3, 0.95]
    var ctx = DeviceContext()
    var splats = splats_of(ctx, ps, colors)
    splats.enable_training()
    var npx = cam.width * cam.height * 3
    var image = ctx.enqueue_create_buffer[DType.float32](npx)
    var weights = List[Float32](capacity=npx)
    for i in range(npx):
        weights.append(Float32(1 + i % 3))
    var grad_image = upload(ctx, weights)
    var r = Rasterizer[16, Gut3D.TILE_H](ctx)
    var view = splats.view()
    var st = r.forward[Gut3D](view, cam, SIMD[F32, 4](0.0), dptr(image), 0)
    r.backward[Gut3D](view, st, dptr(image), dptr(grad_image), splats.grad_view())
    ctx.synchronize()
    var gpu = List[List[Float32]]()
    for a in [Attr.MEANS, Attr.LOG_SCALES, Attr.ROTATIONS, Attr.OPACITY_LOGITS, Attr.SH0]:
        gpu.append(
            download(ctx, splats.grads[a], 3 * (4 if a == Attr.ROTATIONS else (1 if a == Attr.OPACITY_LOGITS else 3)))
        )

    var order: List[Int] = [0, 1, 2]
    for i in range(3):
        for j in range(2 - i):
            if depth(ps[order[j]], cam) > depth(ps[order[j + 1]], cam):
                order.swap_elements(j, j + 1)
    var blocks = List[Ours]()
    var acc = List[LoadedRay3D]()
    for g in range(3):
        blocks.append(ours(ps[g], c))
        acc.append(blocks[g].loaded)
    var d_color = List[Float64](length=9, fill=0.0)
    for y in range(cam.height):
        for x in range(cam.width):
            var w = SIMD[F64, 4](1.0, 2.0, 3.0, 0.0)
            var hits = List[SIMD[F32, 8]]()
            var alphas = SIMD[F64, 4](0.0)
            for k in range(3):
                var h = blocks[order[k]].loaded.eval(Float32(x) + 0.5, Float32(y) + 0.5)
                if h[0] <= 1.0 / 255.0:
                    h[0] = 0.0
                hits.append(h)
                alphas[k] = Float64(h[0])
            var T = SIMD[F64, 4](1.0, 0.0, 0.0, 0.0)
            for k in range(1, 3):
                T[k] = T[k - 1] * (1.0 - alphas[k - 1])
            for k in range(3):
                if alphas[k] == 0.0:
                    continue
                var g = order[k]
                var after: Float64 = 0.0  # w . (color composited behind k)
                for m in range(k + 1, 3):
                    for ch in range(3):
                        after += w[ch] * Float64(colors[order[m] * 3 + ch]) * alphas[m] * T[m]
                var dalpha = -after / (1.0 - alphas[k])
                for ch in range(3):
                    dalpha += w[ch] * Float64(colors[g * 3 + ch]) * T[k]
                    d_color[g * 3 + ch] += w[ch] * alphas[k] * T[k]
                acc[g].add(hits[k], Float32(dalpha))
    var worst: Float64 = 0.0
    for g in range(3):
        var slot = acc[g].gradient()
        var rg = blocks[g].resp.backward(slot, blocks[g].place, blocks[g].shape, c)
        var want = List[Float64]()
        var got = List[Float64]()
        var m = rg.grad.mean
        var ls = rg.grad.shape.log_scales
        var q = rg.grad.shape.quat
        for v in [m.x, m.y, m.z, ls.x, ls.y, ls.z, q.w, q.x, q.y, q.z, GaussianRay3D.opacity_gradient(slot)]:
            want.append(Float64(v))
        for k in range(3):
            got.append(Float64(gpu[0][g * 3 + k]))
        for k in range(3):
            got.append(Float64(gpu[1][g * 3 + k]))
        for k in range(4):
            got.append(Float64(gpu[2][g * 4 + k]))
        got.append(Float64(gpu[3][g]))
        for ch in range(3):
            want.append(d_color[g * 3 + ch] * Float64(SH_C0))
            got.append(Float64(gpu[4][g * 3 + ch]))
        var size: Float64 = 0.0
        for v in want:
            size = max(size, abs(v))
        for k in range(len(want)):
            var err = abs(got[k] - want[k]) / max(abs(want[k]), 1e-2 * size)
            worst = max(worst, err)
            assert_true(err < 2e-3, String("Gaussian ", g, " output ", k, ": GPU ", got[k], " vs CPU ", want[k]))
    print("  Gut3D backward vs the CPU chain: worst relative error", worst)
    _ = (image^, grad_image^, splats^)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
