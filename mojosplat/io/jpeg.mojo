"""JPEG decoding through libjpeg-turbo's TurboJPEG 3 C API.

libjpeg-turbo is a pixi dependency (conda-forge). The library is loaded at
runtime from the active environment (`$CONDA_PREFIX/lib`, set by pixi), so
nothing needs to be linked at build time. The TurboJPEG 3 API reports
errors through return codes (no setjmp/longjmp), which makes it safe to call
from Mojo.
"""

from std.ffi import OwnedDLHandle, c_int
from std.os import getenv

from .binary import read_file


# turbojpeg.h enums (TurboJPEG 3)
comptime _TJINIT_COMPRESS = 0
comptime _TJINIT_DECOMPRESS = 1
comptime _TJPARAM_NOREALLOC = 2
comptime _TJPARAM_QUALITY = 3
comptime _TJPARAM_SUBSAMP = 4
comptime _TJPARAM_JPEGWIDTH = 5
comptime _TJPARAM_JPEGHEIGHT = 6
comptime _TJPARAM_FASTDCT = 10
comptime _TJPF_RGB = 0
comptime _TJSAMP_444 = 0
comptime _TJSAMP_420 = 2


@fieldwise_init
struct Image8(Movable):
    """An 8-bit RGB image in host memory, [height, width, 3] row-major."""

    var width: Int
    var height: Int
    var pixels: List[UInt8]


def _load_turbojpeg() raises -> OwnedDLHandle:
    var candidates = List[String]()
    var prefix = getenv("CONDA_PREFIX")
    if prefix.byte_length() > 0:
        candidates.append(prefix + "/lib/libturbojpeg.so.0")
        candidates.append(prefix + "/lib/libturbojpeg.dylib")
    candidates.append("libturbojpeg.so.0")
    candidates.append("libturbojpeg.dylib")
    for path in candidates:
        try:
            return OwnedDLHandle(path)
        except:
            pass
    raise Error(
        "libturbojpeg not found: run inside the pixi environment (pixi run ...) "
        "or install libjpeg-turbo"
    )


struct JpegDecoder(Movable):
    """Reusable TurboJPEG decompressor (one per thread)."""

    var lib: OwnedDLHandle
    var handle: Int  # tjhandle (an opaque C pointer, passed by address)

    def __init__(out self) raises:
        self.lib = _load_turbojpeg()
        var init = self.lib.get_function[Int]("tj3Init")
        self.handle = init(c_int(_TJINIT_DECOMPRESS))
        if self.handle == 0:
            raise Error("tj3Init failed")

    def __deinit__(deinit self):
        try:
            var destroy = self.lib.get_function[NoneType]("tj3Destroy")
            _ = destroy(self.handle)
        except:
            pass

    def _error(self, what: String) raises -> Error:
        var code = self.lib.get_function[c_int]("tj3GetErrorCode")(self.handle)
        return Error(String(what, " (turbojpeg error code ", code, ")"))

    def decode(mut self, data: List[UInt8], name: String = "<memory>") raises -> Image8:
        """Decode a JPEG byte stream to 8-bit RGB."""
        var header = self.lib.get_function[c_int]("tj3DecompressHeader")
        var tj_get = self.lib.get_function[c_int]("tj3Get")
        var decompress = self.lib.get_function[c_int]("tj3Decompress8")
        var src = data.unsafe_ptr().as_unsafe_any_origin()
        if header(self.handle, src, len(data)) != 0:
            var err = self._error(String("not a readable JPEG: ", name))
            raise err^
        var w = Int(tj_get(self.handle, c_int(_TJPARAM_JPEGWIDTH)))
        var h = Int(tj_get(self.handle, c_int(_TJPARAM_JPEGHEIGHT)))
        var pixels = List[UInt8](length=w * h * 3, fill=0)
        var rc = decompress(
            self.handle, src, len(data), pixels.unsafe_ptr().as_unsafe_any_origin(),
            c_int(0), c_int(_TJPF_RGB),
        )
        # tj3Decompress8 returns -1 for recoverable warnings too (e.g. extraneous
        # bytes before a marker); the image is complete then (TJERR_WARNING = 0).
        if rc != 0 and self.lib.get_function[c_int]("tj3GetErrorCode")(self.handle) != 0:
            var err = self._error(String("JPEG decoding failed: ", name))
            raise err^
        return Image8(w, h, pixels^)

    def read(mut self, path: String) raises -> Image8:
        return self.decode(read_file(path), path)


def read_jpeg(path: String) raises -> Image8:
    """Decode one JPEG file to 8-bit RGB (creates a temporary decoder)."""
    var dec = JpegDecoder()
    return dec.read(path)


struct JpegEncoder(Movable):
    """Reusable TurboJPEG compressor: 8-bit RGB -> JPEG (4:2:0, fast DCT).
    The result stays in `data` (first `size` bytes) until the next encode."""

    var lib: OwnedDLHandle
    var handle: Int
    var data: List[UInt8]
    var size: Int

    def __init__(out self, quality: Int = 85) raises:
        self.lib = _load_turbojpeg()
        self.handle = self.lib.get_function[Int]("tj3Init")(c_int(_TJINIT_COMPRESS))
        if self.handle == 0:
            raise Error("tj3Init failed")
        self.data = List[UInt8]()
        self.size = 0
        var tj_set = self.lib.get_function[c_int]("tj3Set")
        _ = tj_set(self.handle, c_int(_TJPARAM_QUALITY), c_int(quality))
        _ = tj_set(self.handle, c_int(_TJPARAM_SUBSAMP), c_int(_TJSAMP_420))
        _ = tj_set(self.handle, c_int(_TJPARAM_FASTDCT), c_int(1))
        _ = tj_set(self.handle, c_int(_TJPARAM_NOREALLOC), c_int(1))  # we own the buffer

    def __deinit__(deinit self):
        try:
            _ = self.lib.get_function[NoneType]("tj3Destroy")(self.handle)
        except:
            pass

    def encode(mut self, rgb: List[UInt8], width: Int, height: Int) raises -> Int:
        """Compress [height, width, 3] RGB; returns the JPEG size in bytes."""
        if len(rgb) < width * height * 3:
            raise Error("JpegEncoder.encode: buffer smaller than width * height * 3")
        var bound = Int(self.lib.get_function[UInt]("tj3JPEGBufSize")(c_int(width), c_int(height), c_int(_TJSAMP_444)))
        if len(self.data) < bound:
            self.data = List[UInt8](unsafe_uninit_length=bound)
        # tj3Compress8 takes unsigned char **jpegBuf and size_t *jpegSize.
        var buf_addr = Int(self.data.unsafe_ptr())
        var size = UInt(bound)
        var rc = self.lib.get_function[c_int]("tj3Compress8")(
            self.handle, rgb.unsafe_ptr().as_unsafe_any_origin(), c_int(width), c_int(0), c_int(height),
            c_int(_TJPF_RGB), Pointer(to=buf_addr), Pointer(to=size),
        )
        if rc != 0 and self.lib.get_function[c_int]("tj3GetErrorCode")(self.handle) != 0:
            raise Error("JPEG encoding failed")
        self.size = Int(size)
        return self.size
