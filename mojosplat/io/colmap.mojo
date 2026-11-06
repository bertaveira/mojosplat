"""COLMAP sparse model reader (binary format: cameras.bin, images.bin,
points3D.bin), following COLMAP's `src/colmap/scene/reconstruction_io.cc`.

Every reader checks that it consumed the whole file, so a format mismatch
fails loudly instead of producing garbage.
"""

from std.math import sqrt

from .binary import ByteReader


def camera_model_num_params(model_id: Int) raises -> Int:
    """Number of intrinsic parameters of a COLMAP camera model."""
    if model_id == 0:  # SIMPLE_PINHOLE: f, cx, cy
        return 3
    elif model_id == 1:  # PINHOLE: fx, fy, cx, cy
        return 4
    elif model_id == 2:  # SIMPLE_RADIAL
        return 4
    elif model_id == 3:  # RADIAL
        return 5
    elif model_id == 4:  # OPENCV
        return 8
    elif model_id == 5:  # OPENCV_FISHEYE
        return 8
    elif model_id == 6:  # FULL_OPENCV
        return 12
    elif model_id == 7:  # FOV
        return 5
    elif model_id == 8:  # SIMPLE_RADIAL_FISHEYE
        return 4
    elif model_id == 9:  # RADIAL_FISHEYE
        return 5
    elif model_id == 10:  # THIN_PRISM_FISHEYE
        return 12
    elif model_id == 11:  # RAD_TAN_THIN_PRISM_FISHEYE
        return 16
    raise Error(String("unknown COLMAP camera model id ", model_id))


@fieldwise_init
struct ColmapCamera(Copyable, Movable):
    var id: Int
    var model: Int  # COLMAP model id (0 SIMPLE_PINHOLE, 1 PINHOLE, ...)
    var width: Int
    var height: Int
    var params: List[Float64]

    def pinhole(self) raises -> SIMD[DType.float64, 4]:
        """(fx, fy, cx, cy) for the undistorted pinhole models."""
        if self.model == 0:
            return SIMD[DType.float64, 4](self.params[0], self.params[0], self.params[1], self.params[2])
        elif self.model == 1:
            return SIMD[DType.float64, 4](self.params[0], self.params[1], self.params[2], self.params[3])
        raise Error(
            String(
                "camera ",
                self.id,
                ": only undistorted SIMPLE_PINHOLE/PINHOLE models are supported (model id ",
                self.model,
                "); undistort the images with COLMAP first",
            )
        )


@fieldwise_init
struct ColmapImage(Copyable, Movable):
    """A registered image: world-to-camera rotation (quaternion w, x, y, z)
    and translation."""

    var id: Int
    var qvec: SIMD[DType.float64, 4]
    var tvec: SIMD[DType.float64, 4]  # lane 3 unused
    var camera_id: Int
    var name: String

    def view_matrix(self) -> SIMD[DType.float32, 16]:
        """Row-major 4x4 world-to-camera matrix [R | t; 0 0 0 1]."""
        var q = self.qvec
        q /= sqrt((q * q).reduce_add())
        var w = q[0]
        var x = q[1]
        var y = q[2]
        var z = q[3]
        var m = SIMD[DType.float32, 16](0.0)
        m[0] = Float32(1 - 2 * (y * y + z * z))
        m[1] = Float32(2 * (x * y - w * z))
        m[2] = Float32(2 * (x * z + w * y))
        m[4] = Float32(2 * (x * y + w * z))
        m[5] = Float32(1 - 2 * (x * x + z * z))
        m[6] = Float32(2 * (y * z - w * x))
        m[8] = Float32(2 * (x * z - w * y))
        m[9] = Float32(2 * (y * z + w * x))
        m[10] = Float32(1 - 2 * (x * x + y * y))
        m[3] = Float32(self.tvec[0])
        m[7] = Float32(self.tvec[1])
        m[11] = Float32(self.tvec[2])
        m[15] = 1.0
        return m


struct ColmapPoints(Copyable, Movable):
    """Sparse 3D points: xyz [N, 3] and rgb [N, 3] (0-255)."""

    var xyz: List[Float32]
    var rgb: List[UInt8]

    def __init__(out self):
        self.xyz = List[Float32]()
        self.rgb = List[UInt8]()

    def count(self) -> Int:
        return len(self.xyz) // 3


struct ColmapModel(Movable):
    var cameras: List[ColmapCamera]
    var images: List[ColmapImage]
    var points: ColmapPoints

    def __init__(out self, var cameras: List[ColmapCamera], var images: List[ColmapImage], var points: ColmapPoints):
        self.cameras = cameras^
        self.images = images^
        self.points = points^

    def camera(self, camera_id: Int) raises -> ColmapCamera:
        for i in range(len(self.cameras)):
            if self.cameras[i].id == camera_id:
                return self.cameras[i].copy()
        raise Error(String("COLMAP camera id ", camera_id, " not found"))


def _check_end(r: ByteReader) raises:
    if not r.at_end():
        raise Error(String(r.name, ": ", r.remaining(), " unexpected trailing bytes (not a COLMAP binary model?)"))


def read_cameras_bin(path: String) raises -> List[ColmapCamera]:
    var r = ByteReader.open(path)
    var n = r.count(24)  # camera_id, model_id, width, height
    var out = List[ColmapCamera](capacity=n)
    for _ in range(n):
        var cid = Int(r.i32())
        var model = Int(r.i32())
        var w = Int(r.u64())
        var h = Int(r.u64())
        var params = List[Float64]()
        for _ in range(camera_model_num_params(model)):
            params.append(r.f64())
        out.append(ColmapCamera(cid, model, w, h, params^))
    _check_end(r)
    return out^


def read_images_bin(path: String) raises -> List[ColmapImage]:
    var r = ByteReader.open(path)
    var n = r.count(69)  # id, qvec, tvec, camera_id, name NUL, n2d
    var out = List[ColmapImage](capacity=n)
    for _ in range(n):
        var iid = Int(r.i32())
        var q = SIMD[DType.float64, 4](0.0)
        for k in range(4):
            q[k] = r.f64()
        var t = SIMD[DType.float64, 4](0.0)
        for k in range(3):
            t[k] = r.f64()
        var cid = Int(r.i32())
        var name = r.cstring()
        var n2d = r.count(24)
        r.skip(n2d * 24)  # (x: f64, y: f64, point3D_id: i64) per 2D point
        out.append(ColmapImage(iid, q, t, cid, name^))
    _check_end(r)
    return out^


def read_points3d_bin(path: String) raises -> ColmapPoints:
    var r = ByteReader.open(path)
    var n = r.count(51)  # id, xyz, rgb, error, track length
    var pts = ColmapPoints()
    pts.xyz.reserve(n * 3)
    pts.rgb.reserve(n * 3)
    for _ in range(n):
        _ = r.u64()  # point3D_id
        for _ in range(3):
            pts.xyz.append(Float32(r.f64()))
        for _ in range(3):
            pts.rgb.append(r.u8())
        _ = r.f64()  # reprojection error
        var track = r.count(8)
        r.skip(track * 8)  # (image_id: i32, point2D_idx: i32)
    _check_end(r)
    return pts^


def read_colmap_model(sparse_dir: String) raises -> ColmapModel:
    """Read `<sparse_dir>/{cameras,images,points3D}.bin`."""
    var d = sparse_dir if sparse_dir.endswith("/") else sparse_dir + "/"
    return ColmapModel(
        read_cameras_bin(d + "cameras.bin"),
        read_images_bin(d + "images.bin"),
        read_points3d_bin(d + "points3D.bin"),
    )
