"""3D Gaussian Splatting PLY files (the INRIA / gsplat / viewer layout).

Binary little-endian, one `vertex` element with float properties:
    x y z nx ny nz f_dc_0..2 f_rest_0..(3K-1) opacity scale_0..2 rot_0..3
Values are the raw parameters (log scales, opacity logits, unnormalized
(w, x, y, z) quaternions). `f_rest` is stored channel-major: f_rest[c*K + k]
is SH basis k + 1 of color channel c; in memory we use [N, K, 3].
"""

from std.memory import bitcast

from .binary import ByteReader, append_text


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
    var names = _ply_property_names(s.rest_stride)
    var header = String("ply\nformat binary_little_endian 1.0\nelement vertex ", s.count, "\n")
    for name in names:
        header += String("property float ", name, "\n")
    header += "end_header\n"
    var P = len(names)
    var K = s.rest_stride
    var row = List[Float32](length=s.count * P, fill=0.0)
    for g in range(s.count):
        var o = g * P
        for k in range(3):
            row[o + k] = s.means[g * 3 + k]
            row[o + 6 + k] = s.sh0[g * 3 + k]
            row[o + 9 + 3 * K + 1 + k] = s.log_scales[g * 3 + k]
        for c in range(3):
            for k in range(K):
                row[o + 9 + c * K + k] = s.sh_rest[(g * K + k) * 3 + c]
        row[o + 9 + 3 * K] = s.opacity_logits[g]
        for k in range(4):
            row[o + 9 + 3 * K + 4 + k] = s.rotations[g * 4 + k]
    var out = List[UInt8](capacity=header.byte_length() + len(row) * 4)
    append_text(out, header)
    var src = row.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(len(row) * 4):
        out.append(src[unsafe_offset=i])
    with open(path, "w") as f:
        f.write_bytes(out)


def _col(names: List[String], name: String) -> Int:
    """Index of property `name` (-1 if missing)."""
    for i in range(len(names)):
        if names[i] == name:
            return i
    return -1


@always_inline
def _le_f32(data: List[UInt8], row: Int, c: Int) -> Float32:
    var p = row + 4 * c
    var v = UInt32(data[p]) | (UInt32(data[p + 1]) << 8) | (UInt32(data[p + 2]) << 16) | (UInt32(data[p + 3]) << 24)
    return bitcast[DType.float32](v)


def read_ply(path: String) raises -> HostSplats:
    """Read a binary little-endian 3DGS PLY (float vertex properties only)."""
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
    var c_dc = SIMD[DType.int32, 4](Int32(_col(names, "f_dc_0")), Int32(_col(names, "f_dc_1")), Int32(_col(names, "f_dc_2")), 0)
    var c_sc = SIMD[DType.int32, 4](Int32(_col(names, "scale_0")), Int32(_col(names, "scale_1")), Int32(_col(names, "scale_2")), 0)
    var c_rot = SIMD[DType.int32, 4](
        Int32(_col(names, "rot_0")), Int32(_col(names, "rot_1")), Int32(_col(names, "rot_2")), Int32(_col(names, "rot_3"))
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

    var s = HostSplats(count, K)
    var base = r.pos
    ref data = r.data
    for g in range(count):
        for k in range(3):
            s.means[g * 3 + k] = _le_f32(data, base + 4 * g * P, Int(c_xyz[k]))
            s.sh0[g * 3 + k] = _le_f32(data, base + 4 * g * P, Int(c_dc[k]))
            s.log_scales[g * 3 + k] = _le_f32(data, base + 4 * g * P, Int(c_sc[k]))
        for k in range(4):
            s.rotations[g * 4 + k] = _le_f32(data, base + 4 * g * P, Int(c_rot[k]))
        s.opacity_logits[g] = _le_f32(data, base + 4 * g * P, c_op)
        for c in range(3):
            for k in range(K):
                s.sh_rest[(g * K + k) * 3 + c] = _le_f32(data, base + 4 * g * P, c_rest[c * K + k])
    return s^
