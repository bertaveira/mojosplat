"""Little-endian binary reading/writing helpers for file formats."""

from std.memory import bitcast
from std.os import SEEK_END, SEEK_SET


def read_file(path: String) raises -> List[UInt8]:
    """The whole file in one read. (`read_bytes()` without a size grows its
    buffer by doubling, which copies a 1 GB PLY ~20 times.)"""
    with open(path, "r") as f:
        var size = Int(f.seek(0, SEEK_END))
        _ = f.seek(0, SEEK_SET)
        var data = f.read_bytes(size)
        if len(data) != size:
            raise Error(String("short read: ", path))
        return data^


struct ByteReader(Movable):
    """Sequential little-endian reader over a byte buffer (whole file)."""

    var data: List[UInt8]
    var pos: Int
    var name: String  # for error messages

    def __init__(out self, var data: List[UInt8], name: String = ""):
        self.data = data^
        self.pos = 0
        self.name = name

    @staticmethod
    def open(path: String) raises -> ByteReader:
        return ByteReader(read_file(path), path)

    def remaining(self) -> Int:
        return len(self.data) - self.pos

    def at_end(self) -> Bool:
        return self.pos == len(self.data)

    def _need(self, n: Int) raises:
        if self.pos + n > len(self.data):
            raise Error(String("unexpected end of file in ", self.name, " at byte ", self.pos))

    def skip(mut self, n: Int) raises:
        if n < 0:
            raise Error(String("negative skip in ", self.name, " at byte ", self.pos))
        self._need(n)
        self.pos += n

    def count(mut self, min_record_bytes: Int) raises -> Int:
        """Read a u64 element count, checking that `count` records of at least
        `min_record_bytes` each fit in the rest of the file (so a corrupt
        count raises instead of over-allocating or wrapping negative)."""
        var at = self.pos
        var n = self.u64()
        if n > UInt64(self.remaining() // max(min_record_bytes, 1)):
            raise Error(String("corrupt count ", n, " in ", self.name, " at byte ", at))
        return Int(n)

    def u8(mut self) raises -> UInt8:
        self._need(1)
        var v = self.data[self.pos]
        self.pos += 1
        return v

    def _u64_at(self, p: Int, n: Int) -> UInt64:
        var v = UInt64(0)
        for i in range(n):
            v |= UInt64(self.data[p + i]) << UInt64(8 * i)
        return v

    def u32(mut self) raises -> UInt32:
        self._need(4)
        var v = UInt32(self._u64_at(self.pos, 4))
        self.pos += 4
        return v

    def i32(mut self) raises -> Int32:
        return bitcast[DType.int32](self.u32())

    def u64(mut self) raises -> UInt64:
        self._need(8)
        var v = self._u64_at(self.pos, 8)
        self.pos += 8
        return v

    def i64(mut self) raises -> Int64:
        return bitcast[DType.int64](self.u64())

    def f32(mut self) raises -> Float32:
        return bitcast[DType.float32](self.u32())

    def f64(mut self) raises -> Float64:
        return bitcast[DType.float64](self.u64())

    def cstring(mut self) raises -> String:
        """NUL-terminated string."""
        var start = self.pos
        while True:
            self._need(1)
            if self.data[self.pos] == 0:
                break
            self.pos += 1
        var s = String(unsafe_from_utf8=Span(self.data)[start : self.pos])
        self.pos += 1
        return s

    def line(mut self) raises -> String:
        """Bytes up to (excluding) the next '\\n'; a trailing '\\r' is dropped."""
        var start = self.pos
        while True:
            self._need(1)
            if self.data[self.pos] == UInt8(ord("\n")):
                break
            self.pos += 1
        var end = self.pos
        if end > start and self.data[end - 1] == UInt8(ord("\r")):
            end -= 1
        var s = String(unsafe_from_utf8=Span(self.data)[start:end])
        self.pos += 1
        return s


def append_u32(mut out: List[UInt8], v: UInt32):
    for i in range(4):
        out.append(UInt8((v >> UInt32(8 * i)) & 0xFF))


def append_f32(mut out: List[UInt8], v: Float32):
    append_u32(out, bitcast[DType.uint32](v))


def append_text(mut out: List[UInt8], s: String):
    for b in s.as_bytes():
        out.append(b)
