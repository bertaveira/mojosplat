"""JPEG decoding through libjpeg-turbo's TurboJPEG 3 C API.

libjpeg-turbo is a pixi dependency (conda-forge). The library is loaded at
runtime from the active environment (`$CONDA_PREFIX/lib`, set by pixi), so
nothing needs to be linked at build time. The TurboJPEG 3 API reports
errors through return codes (no setjmp/longjmp), which makes it safe to call
from Mojo.
"""

from std.ffi import OwnedDLHandle, c_int
from std.os import getenv
from std.pathlib import Path


# turbojpeg.h enums (TurboJPEG 3)
comptime _TJINIT_DECOMPRESS = 1
comptime _TJPARAM_JPEGWIDTH = 5
comptime _TJPARAM_JPEGHEIGHT = 6
comptime _TJPF_RGB = 0


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
        if rc != 0:
            var err = self._error(String("JPEG decoding failed: ", name))
            raise err^
        return Image8(w, h, pixels^)

    def read(mut self, path: String) raises -> Image8:
        return self.decode(Path(path).read_bytes(), path)


def read_jpeg(path: String) raises -> Image8:
    """Decode one JPEG file to 8-bit RGB (creates a temporary decoder)."""
    var dec = JpegDecoder()
    return dec.read(path)
