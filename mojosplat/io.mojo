"""Host-side I/O: raw binary arrays, .splat scenes, and image output."""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.sys import size_of


# ---------------------------------------------------------------------------
# Raw little-endian arrays
# ---------------------------------------------------------------------------


def read_array[dtype: DType](path: String) raises -> List[Scalar[dtype]]:
    """Read a raw little-endian binary file of `dtype` values."""
    var data: List[UInt8]
    with open(path, "r") as f:
        data = f.read_bytes()
    comptime size = size_of[Scalar[dtype]]()
    if len(data) % size != 0:
        raise Error(String("file size not a multiple of ", size, ": ", path))
    var n = len(data) // size
    var src = data.unsafe_ptr().unsafe_bitcast[Scalar[dtype]]()
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(src[unsafe_offset=i])
    return out^


def write_array[dtype: DType](path: String, values: List[Scalar[dtype]]) raises:
    """Write values as a raw little-endian binary file."""
    comptime size = size_of[Scalar[dtype]]()
    var bytes = List[UInt8](capacity=len(values) * size)
    var src = values.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(len(values) * size):
        bytes.append(src[unsafe_offset=i])
    with open(path, "w") as f:
        f.write_bytes(bytes)


# ---------------------------------------------------------------------------
# Host <-> device helpers
# ---------------------------------------------------------------------------


def upload[
    dtype: DType
](ctx: DeviceContext, values: List[Scalar[dtype]]) raises -> DeviceBuffer[dtype]:
    """Copy a host list into a new device buffer (enqueued on ctx's stream)."""
    var n = max(len(values), 1)
    var host = ctx.enqueue_create_host_buffer[dtype](n)
    ctx.synchronize()
    for i in range(len(values)):
        host[i] = values[i]
    var dev = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(dev, host)
    ctx.synchronize()
    return dev^


def download[
    dtype: DType
](ctx: DeviceContext, buf: DeviceBuffer[dtype], count: Int) raises -> List[Scalar[dtype]]:
    """Copy the first `count` elements of a device buffer back to the host."""
    var out = List[Scalar[dtype]](capacity=count)
    if count == 0:
        return out^
    var host = ctx.enqueue_create_host_buffer[dtype](len(buf))
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    for i in range(count):
        out.append(host[i])
    return out^


# ---------------------------------------------------------------------------
# .splat scenes
# ---------------------------------------------------------------------------


struct SplatScene(Movable):
    """Gaussians in host memory, laid out as the pipeline expects.

    scales are linear (not log), quats are (w, x, y, z), opacities in [0, 1].
    """

    var count: Int
    var means: List[Float32]  # [N, 3]
    var scales: List[Float32]  # [N, 3]
    var quats: List[Float32]  # [N, 4]
    var opacities: List[Float32]  # [N]
    var colors: List[Float32]  # [N, 3]

    def __init__(out self, count: Int):
        self.count = count
        self.means = List[Float32](capacity=count * 3)
        self.scales = List[Float32](capacity=count * 3)
        self.quats = List[Float32](capacity=count * 4)
        self.opacities = List[Float32](capacity=count)
        self.colors = List[Float32](capacity=count * 3)


def load_splat(path: String) raises -> SplatScene:
    """Load an antimatter15-style `.splat` file (32 bytes per Gaussian).

    Layout: position xyz (3 x f32), scale xyz (3 x f32, linear), RGBA (4 x u8,
    alpha = activated opacity), rotation (4 x u8, (q * 128 + 128), order w x y z).
    """
    var data: List[UInt8]
    with open(path, "r") as f:
        data = f.read_bytes()
    if len(data) % 32 != 0 or len(data) == 0:
        raise Error(String("invalid .splat file: ", path))
    var n = len(data) // 32
    var scene = SplatScene(n)
    var base = data.unsafe_ptr()
    for i in range(n):
        var rec = base + i * 32
        var f = rec.unsafe_bitcast[Float32]()
        for k in range(3):
            scene.means.append(f[unsafe_offset=k])
        for k in range(3):
            scene.scales.append(f[unsafe_offset=3 + k])
        for k in range(3):
            scene.colors.append(Float32(rec[unsafe_offset=24 + k]) / 255.0)
        scene.opacities.append(Float32(rec[unsafe_offset=27]) / 255.0)
        var q0 = (Float32(rec[unsafe_offset=28]) - 128.0) / 128.0
        var q1 = (Float32(rec[unsafe_offset=29]) - 128.0) / 128.0
        var q2 = (Float32(rec[unsafe_offset=30]) - 128.0) / 128.0
        var q3 = (Float32(rec[unsafe_offset=31]) - 128.0) / 128.0
        var norm = max((q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3) ** 0.5, 1e-10)
        scene.quats.append(q0 / norm)
        scene.quats.append(q1 / norm)
        scene.quats.append(q2 / norm)
        scene.quats.append(q3 / norm)
    return scene^


# ---------------------------------------------------------------------------
# Image output
# ---------------------------------------------------------------------------


def write_ppm(path: String, rgb: List[Float32], width: Int, height: Int) raises:
    """Write an [H, W, 3] float image (clamped to [0, 1]) as binary PPM (P6)."""
    var header = String("P6\n", width, " ", height, "\n255\n")
    var bytes = List[UInt8](capacity=header.byte_length() + width * height * 3)
    for b in header.as_bytes():
        bytes.append(b)
    for i in range(width * height * 3):
        var v = min(max(rgb[i], 0.0), 1.0)
        bytes.append(UInt8(Int(v * 255.0 + 0.5)))
    with open(path, "w") as f:
        f.write_bytes(bytes)
