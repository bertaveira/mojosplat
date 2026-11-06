"""3D Gaussian Splatting PLY files (the INRIA / gsplat / viewer layout).

Binary little-endian, one `vertex` element with float properties:
    x y z nx ny nz f_dc_0..2 f_rest_0..(3K-1) opacity scale_0..2 rot_0..3
Values are the raw parameters (log scales, opacity logits, unnormalized
(w, x, y, z) quaternions). `f_rest` is stored channel-major: f_rest[c*K + k]
is SH basis k + 1 of color channel c; in memory we use [N, K, 3].
"""

from max.algorithm import parallelize
from std.sys.info import Endian

from .binary import ByteReader, append_text

comptime _CHUNK = 4096  # Gaussians per parallel work item


struct HostSplats(Movable):
    """Raw Gaussian parameters in host memory (same layout as SplatView)."""

    var count: Int
    var rest_stride: Int  # K: SH bases per Gaussian in sh_rest (0, 3, 8 or 15)
    var means: List[Float32]  # [N, 3]
    var log_scales: List[Float32]  # [N, 3]
    var rotations: List[Float32]  # [N, 4] (w, x, y, z)
    var opacity_logits: List[Float32]  # [N]
    var sh0: List[Float32]  # [N, 3]
    var sh_rest: List[Float32]  # [N, K, 3]

    def __init__(out self, count: Int, rest_stride: Int):
        self.count = count
        self.rest_stride = rest_stride
        self.means = List[Float32](length=count * 3, fill=0.0)
        self.log_scales = List[Float32](length=count * 3, fill=0.0)
        self.rotations = List[Float32](length=count * 4, fill=0.0)
        self.opacity_logits = List[Float32](length=count, fill=0.0)
        self.sh0 = List[Float32](length=count * 3, fill=0.0)
        self.sh_rest = List[Float32](length=count * rest_stride * 3, fill=0.0)

    @staticmethod
    def uninitialized(count: Int, rest_stride: Int) -> HostSplats:
        """Like `HostSplats(count, rest_stride)` without zeroing (for code
        that overwrites every value, e.g. loaders)."""
        var s = HostSplats(0, rest_stride)
        s.count = count
        s.means = List[Float32](unsafe_uninit_length=count * 3)
        s.log_scales = List[Float32](unsafe_uninit_length=count * 3)
        s.rotations = List[Float32](unsafe_uninit_length=count * 4)
        s.opacity_logits = List[Float32](unsafe_uninit_length=count)
        s.sh0 = List[Float32](unsafe_uninit_length=count * 3)
        s.sh_rest = List[Float32](unsafe_uninit_length=count * rest_stride * 3)
        return s^


def _ply_property_names(rest_stride: Int) -> List[String]:
    var names: List[String] = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    for i in range(rest_stride * 3):
        names.append(String("f_rest_", i))
    names.append("opacity")
    for i in range(3):
        names.append(String("scale_", i))
    for i in range(4):
        names.append(String("rot_", i))
    return names^


def write_ply(path: String, s: HostSplats) raises:
    """Write Gaussians as a binary little-endian 3DGS PLY."""
    comptime assert Endian.native() == Endian.little, "PLY rows are written as host floats"
    var names = _ply_property_names(s.rest_stride)
    var header = String("ply\nformat binary_little_endian 1.0\nelement vertex ", s.count, "\n")
    for name in names:
        header += String("property float ", name, "\n")
    header += "end_header\n"
    var P = len(names)
    var K = s.rest_stride
    var n = s.count
    var out = List[UInt8](capacity=header.byte_length() + n * P * 4)
    append_text(out, header)
    var h = len(out)
    out.resize(h + n * P * 4, 0)
    # The rows start right after the text header, so they are not 4-byte
    # aligned: unaligned stores.
    var dst = out.unsafe_ptr().unsafe_offset(h).unsafe_bitcast[Float32]()
    var means = s.means.unsafe_ptr()
    var sh0 = s.sh0.unsafe_ptr()
    var log_scales = s.log_scales.unsafe_ptr()
    var sh_rest = s.sh_rest.unsafe_ptr()
    var opacity = s.opacity_logits.unsafe_ptr()
    var rotations = s.rotations.unsafe_ptr()

    def rows(
        chunk: Int,
    ) {var dst, var means, var sh0, var log_scales, var sh_rest, var opacity, var rotations, var P, var K, var n}:
        for g in range(chunk * _CHUNK, min((chunk + 1) * _CHUNK, n)):
            var o = dst.unsafe_offset(g * P)
            for k in range(3):
                o.unsafe_offset(k).unsafe_store[alignment=1](means[unsafe_offset=g * 3 + k])
                o.unsafe_offset(6 + k).unsafe_store[alignment=1](sh0[unsafe_offset=g * 3 + k])
                o.unsafe_offset(9 + 3 * K + 1 + k).unsafe_store[alignment=1](log_scales[unsafe_offset=g * 3 + k])
            for c in range(3):
                for k in range(K):
                    o.unsafe_offset(9 + c * K + k).unsafe_store[alignment=1](sh_rest[unsafe_offset=(g * K + k) * 3 + c])
            o.unsafe_offset(9 + 3 * K).unsafe_store[alignment=1](opacity[unsafe_offset=g])
            for k in range(4):
                o.unsafe_offset(9 + 3 * K + 4 + k).unsafe_store[alignment=1](rotations[unsafe_offset=g * 4 + k])

    parallelize(rows, (n + _CHUNK - 1) // _CHUNK)
    with open(path, "w") as f:
        f.write_bytes(out)


def _col(names: List[String], name: String) -> Int:
    """Index of property `name` (-1 if missing)."""
    for i in range(len(names)):
        if names[i] == name:
            return i
    return -1


def read_ply(path: String) raises -> HostSplats:
    """Read a binary little-endian 3DGS PLY (float vertex properties only)."""
    comptime assert Endian.native() == Endian.little, "PLY rows are read as host floats"
    var r = ByteReader.open(path)
    if r.line() != "ply":
        raise Error(String(path, ": not a PLY file"))
    var count = -1
    var names = List[String]()
    while True:
        var line = r.line()
        if line == "end_header":
            break
        var parts = List[String]()
        for p in line.split(" "):
            parts.append(String(p))
        if len(parts) == 0 or parts[0] == "comment" or parts[0] == "obj_info":
            continue
        if parts[0] == "format":
            if len(parts) < 2 or parts[1] != "binary_little_endian":
                raise Error(String(path, ": only binary_little_endian PLY is supported"))
        elif parts[0] == "element":
            if len(parts) != 3 or parts[1] != "vertex" or count >= 0:
                raise Error(String(path, ": expected a single 'vertex' element"))
            count = Int(parts[2])
            if count < 0:
                raise Error(String(path, ": negative vertex count"))
        elif parts[0] == "property":
            if len(parts) != 3 or parts[1] != "float":
                raise Error(String(path, ": only 'property float <name>' is supported, got: ", line))
            names.append(parts[2])
    if count < 0:
        raise Error(String(path, ": no vertex element"))

    var P = len(names)
    var n_rest = 0
    for name in names:
        if name.startswith("f_rest_"):
            n_rest += 1
    if n_rest % 3 != 0:
        raise Error(String(path, ": number of f_rest properties is not a multiple of 3"))
    var K = n_rest // 3

    var c_xyz = SIMD[DType.int32, 4](Int32(_col(names, "x")), Int32(_col(names, "y")), Int32(_col(names, "z")), 0)
    var c_dc = SIMD[DType.int32, 4](
        Int32(_col(names, "f_dc_0")), Int32(_col(names, "f_dc_1")), Int32(_col(names, "f_dc_2")), 0
    )
    var c_sc = SIMD[DType.int32, 4](
        Int32(_col(names, "scale_0")), Int32(_col(names, "scale_1")), Int32(_col(names, "scale_2")), 0
    )
    var c_rot = SIMD[DType.int32, 4](
        Int32(_col(names, "rot_0")),
        Int32(_col(names, "rot_1")),
        Int32(_col(names, "rot_2")),
        Int32(_col(names, "rot_3")),
    )
    var c_op = _col(names, "opacity")
    var c_rest = List[Int](capacity=n_rest)
    for i in range(n_rest):
        c_rest.append(_col(names, String("f_rest_", i)))
    for k in range(3):
        if c_xyz[k] < 0 or c_dc[k] < 0 or c_sc[k] < 0:
            raise Error(String(path, ": missing position, f_dc or scale properties"))
    for k in range(4):
        if c_rot[k] < 0:
            raise Error(String(path, ": missing rot_* properties"))
    if c_op < 0:
        raise Error(String(path, ": missing opacity property"))
    for c in c_rest:
        if c < 0:
            raise Error(String(path, ": f_rest properties are not contiguously numbered"))
    if r.remaining() < count * P * 4:
        raise Error(String(path, ": file shorter than its header says"))

    var s = HostSplats.uninitialized(count, K)
    # Rows follow the text header (not 4-byte aligned): unaligned loads.
    var src = r.data.unsafe_ptr().unsafe_offset(r.pos).unsafe_bitcast[Float32]()
    var rest_cols = c_rest.unsafe_ptr()
    var means = s.means.unsafe_ptr()
    var sh0 = s.sh0.unsafe_ptr()
    var log_scales = s.log_scales.unsafe_ptr()
    var sh_rest = s.sh_rest.unsafe_ptr()
    var opacity = s.opacity_logits.unsafe_ptr()
    var rotations = s.rotations.unsafe_ptr()

    def rows(
        chunk: Int,
    ) {
        var src,
        var rest_cols,
        var means,
        var sh0,
        var log_scales,
        var sh_rest,
        var opacity,
        var rotations,
        var c_xyz,
        var c_dc,
        var c_sc,
        var c_rot,
        var c_op,
        var P,
        var K,
        var count,
    }:
        for g in range(chunk * _CHUNK, min((chunk + 1) * _CHUNK, count)):
            var row = src.unsafe_offset(g * P)
            for k in range(3):
                means[unsafe_offset=g * 3 + k] = row.unsafe_offset(Int(c_xyz[k])).unsafe_load[alignment=1]()
                sh0[unsafe_offset=g * 3 + k] = row.unsafe_offset(Int(c_dc[k])).unsafe_load[alignment=1]()
                log_scales[unsafe_offset=g * 3 + k] = row.unsafe_offset(Int(c_sc[k])).unsafe_load[alignment=1]()
            for k in range(4):
                rotations[unsafe_offset=g * 4 + k] = row.unsafe_offset(Int(c_rot[k])).unsafe_load[alignment=1]()
            opacity[unsafe_offset=g] = row.unsafe_offset(c_op).unsafe_load[alignment=1]()
            for c in range(3):
                for k in range(K):
                    sh_rest[unsafe_offset=(g * K + k) * 3 + c] = row.unsafe_offset(
                        rest_cols[unsafe_offset=c * K + k]
                    ).unsafe_load[alignment=1]()

    parallelize(rows, (count + _CHUNK - 1) // _CHUNK)
    _ = r^  # the file bytes must outlive the parallel loop
    _ = c_rest^
    return s^
