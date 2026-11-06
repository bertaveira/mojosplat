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

    def __init__(out self, var train: List[View], var test: List[View], var points: ColmapPoints, scene_extent: Float32):
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


def load_colmap_dataset(
    ctx: DeviceContext,
    root: String,
    images_dir: String = "images_4",
    test_every: Int = 8,
    near: Float32 = 0.2,
    far: Float32 = 10000.0,
) raises -> Dataset:
    """Load `<root>/sparse/0` and the images in `<root>/<images_dir>`.

    Images are sorted by name; with `test_every` > 0 every `test_every`-th
    one (index 0, 8, 16, ...) is held out for testing (the standard
    Mip-NeRF360 protocol). Intrinsics are scaled from the COLMAP camera
    size to the actual image size (e.g. the 4x downsampled `images_4`).
    """
    var r = root if root.endswith("/") else root + "/"
    var model = read_colmap_model(r + "sparse/0")
    _sort_by_name(model.images)
    var decoder = JpegDecoder()
    var train = List[View]()
    var test = List[View]()
    var train_cams = List[Camera]()
    for i in range(len(model.images)):
        ref im = model.images[i]
        var img = decoder.read(String(r, images_dir, "/", im.name))
        var cam = model.camera(im.camera_id)
        var k = cam.pinhole()
        var sx = Float64(img.width) / Float64(cam.width)
        var sy = Float64(img.height) / Float64(cam.height)
        var camera = Camera(
            im.view_matrix(), Float32(k[0] * sx), Float32(k[1] * sy), Float32(k[2] * sx),
            Float32(k[3] * sy), img.width, img.height, near, far,
        )
        var view = View(im.name, camera, upload(ctx, img.pixels))
        if test_every > 0 and i % test_every == 0:
            test.append(view^)
        else:
            train_cams.append(camera)
            train.append(view^)
    var extent = scene_extent(train_cams)
    if len(train) == 0:
        raise Error(String(root, ": no training views (", len(test), " test views)"))
    return Dataset(train^, test^, model.points.copy(), extent)
