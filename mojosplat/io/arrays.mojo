"""Raw little-endian arrays on disk, and host <-> device copies."""

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
