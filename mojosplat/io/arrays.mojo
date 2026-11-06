"""Raw little-endian arrays on disk, and host <-> device copies."""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import unsafe_memcpy
from std.sys import size_of

from .binary import read_file


# ---------------------------------------------------------------------------
# Raw little-endian arrays
# ---------------------------------------------------------------------------


def read_array[dtype: DType](path: String) raises -> List[Scalar[dtype]]:
    """Read a raw little-endian binary file of `dtype` values."""
    var data = read_file(path)
    comptime size = size_of[Scalar[dtype]]()
    if len(data) % size != 0:
        raise Error(String("file size not a multiple of ", size, ": ", path))
    var n = len(data) // size
    var out = List[Scalar[dtype]](unsafe_uninit_length=n)
    unsafe_memcpy(dest=out.unsafe_ptr().unsafe_bitcast[UInt8](), src=data.unsafe_ptr(), count=n * size)
    return out^


def write_array[dtype: DType](path: String, values: List[Scalar[dtype]]) raises:
    """Write values as a raw little-endian binary file."""
    comptime size = size_of[Scalar[dtype]]()
    var n = len(values) * size
    var bytes = List[UInt8](unsafe_uninit_length=n)
    unsafe_memcpy(dest=bytes.unsafe_ptr(), src=values.unsafe_ptr().unsafe_bitcast[UInt8](), count=n)
    with open(path, "w") as f:
        f.write_bytes(bytes)


# ---------------------------------------------------------------------------
# Host <-> device helpers
# ---------------------------------------------------------------------------


def upload[dtype: DType](ctx: DeviceContext, values: List[Scalar[dtype]]) raises -> DeviceBuffer[dtype]:
    """Copy a host list into a new device buffer (enqueued on ctx's stream)."""
    var dev = ctx.enqueue_create_buffer[dtype](max(len(values), 1))
    if len(values) > 0:
        ctx.enqueue_copy(dev.create_sub_buffer[dtype](0, len(values)), Span(values))
        ctx.synchronize()  # `values` may be freed once we return
    return dev^


def download[dtype: DType](ctx: DeviceContext, buf: DeviceBuffer[dtype], count: Int) raises -> List[Scalar[dtype]]:
    """Copy the first `count` elements of a device buffer back to the host."""
    var out = List[Scalar[dtype]](unsafe_uninit_length=count)
    if count > 0:
        ctx.enqueue_copy(Span(out), buf.create_sub_buffer[dtype](0, count))
        ctx.synchronize()
    return out^
