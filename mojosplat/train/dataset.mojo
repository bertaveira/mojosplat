"""Posed-image datasets for training (COLMAP scenes, e.g. Mip-NeRF360).

Images are decoded once (libjpeg-turbo) and kept on the GPU as 8-bit RGB
([H, W, 3]); the loss kernels convert to float on the fly.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import sqrt

from ..core.camera import Camera
from ..io.arrays import upload
from ..io.colmap import ColmapImage, ColmapPoints, read_colmap_model
from ..io.jpeg import JpegDecoder


@fieldwise_init
struct View(Copyable, Movable):
    """One posed image: camera (intrinsics scaled to the image) + pixels."""

    var name: String
    var camera: Camera
    var image: DeviceBuffer[DType.uint8]  # [H, W, 3] RGB


struct Dataset(Movable):
    var train: List[View]
    var test: List[View]
    var points: ColmapPoints  # sparse SfM points (initialization)
    var scene_extent: Float32  # 1.1 x max camera distance from the mean camera center

    def __init__(
        out self, var train: List[View], var test: List[View], var points: ColmapPoints, scene_extent: Float32
    ):
        self.train = train^
        self.test = test^
        self.points = points^
        self.scene_extent = scene_extent


def _sort_by_name(mut images: List[ColmapImage]):
    # Insertion sort keeps this dependency-free; a few hundred images.
    for i in range(1, len(images)):
        var j = i
        while j > 0 and images[j - 1].name > images[j].name:
            images.swap_elements(j - 1, j)
            j -= 1


def scene_extent(cameras: List[Camera]) -> Float32:
    """3DGS's normalization radius: 1.1 x the largest distance of a camera
    center from the mean camera center."""
    if len(cameras) == 0:
        return 1.0
    var mean = SIMD[DType.float32, 4](0.0)
    for c in cameras:
        mean += c.position()
    mean /= Float32(len(cameras))
    var r: Float32 = 0.0
    for c in cameras:
        var d = c.position() - mean
        r = max(r, sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))
    return 1.1 * r


def _median(var v: List[Float64]) -> Float64:
    for i in range(1, len(v)):
        var x = v[i]
        var j = i - 1
        while j >= 0 and v[j] > x:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = x
    var n = len(v)
    return v[n // 2] if n % 2 == 1 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def median_scene_extent(cameras: List[Camera]) -> Float32:
    """NVIDIA's `get_center_and_diag` * 1.1: the median distance of the camera
    centers from their per-axis median (3dgrut, over train and test cameras)."""
    if len(cameras) == 0:
        return 1.0
    var center = SIMD[DType.float64, 4](0.0)
    for k in range(3):
        var values = List[Float64](capacity=len(cameras))
        for c in cameras:
            values.append(Float64(c.position()[k]))
        center[k] = _median(values^)
    var dist = List[Float64](capacity=len(cameras))
    for c in cameras:
        var p = c.position()
        var dx = Float64(p[0]) - center[0]
        var dy = Float64(p[1]) - center[1]
        var dz = Float64(p[2]) - center[2]
        dist.append(sqrt(dx * dx + dy * dy + dz * dz))
    return Float32(_median(dist^) * 1.1)


def load_colmap_dataset(
    ctx: DeviceContext,
    root: String,
    images_dir: String = "images_4",
    test_every: Int = 8,
    near: Float32 = 0.2,
    far: Float32 = 10000.0,
    integer_factor: Bool = False,
    median_extent: Bool = False,
) raises -> Dataset:
    """Load `<root>/sparse/0` and the images in `<root>/<images_dir>`.

    Images are sorted by name; with `test_every` > 0 every `test_every`-th
    one (index 0, 8, 16, ...) is held out for testing (the standard
    Mip-NeRF360 protocol). Intrinsics are scaled from the COLMAP camera
    size to the actual image size (e.g. the 4x downsampled `images_4`), or,
    with `integer_factor`, divided by the nearest integer factor of the
    heights on both axes (NVIDIA's 3dgrut). The scene extent is 3DGS's, over
    the training cameras, or with `median_extent` 3dgrut's
    (`median_scene_extent`, over every camera).
    """
    var r = root if root.endswith("/") else root + "/"
    var model = read_colmap_model(r + "sparse/0")
    _sort_by_name(model.images)
    var decoder = JpegDecoder()
    var train = List[View]()
    var test = List[View]()
    var train_cams = List[Camera]()
    var all_cams = List[Camera]()
    for i in range(len(model.images)):
        ref im = model.images[i]
        var img = decoder.read(String(r, images_dir, "/", im.name))
        var cam = model.camera(im.camera_id)
        var k = cam.pinhole()
        var sx = Float64(img.width) / Float64(cam.width)
        var sy = Float64(img.height) / Float64(cam.height)
        if integer_factor:  # NVIDIA's convention: the nearest integer factor, the same on both axes
            sx = 1.0 / round(Float64(cam.height) / Float64(img.height))
            sy = sx
        var camera = Camera(
            im.view_matrix(),
            Float32(k[0] * sx),
            Float32(k[1] * sy),
            Float32(k[2] * sx),
            Float32(k[3] * sy),
            img.width,
            img.height,
            near,
            far,
        )
        all_cams.append(camera)
        var view = View(im.name, camera, upload(ctx, img.pixels))
        if test_every > 0 and i % test_every == 0:
            test.append(view^)
        else:
            train_cams.append(camera)
            train.append(view^)
    var extent = median_scene_extent(all_cams) if median_extent else scene_extent(train_cams)
    if len(train) == 0:
        raise Error(String(root, ": no training views (", len(test), " test views)"))
    return Dataset(train^, test^, model.points.copy(), extent)
