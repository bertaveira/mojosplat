"""Image output: binary PPM and PNG (stored/uncompressed deflate)."""


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


def _crc32_table() -> List[UInt32]:
    var table = List[UInt32](capacity=256)
    for n in range(256):
        var c = UInt32(n)
        for _ in range(8):
            if c & 1:
                c = UInt32(0xEDB88320) ^ (c >> 1)
            else:
                c = c >> 1
        table.append(c)
    return table^


def _append_u32_be(mut out: List[UInt8], v: UInt32):
    out.append(UInt8((v >> 24) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


def _append_chunk(mut out: List[UInt8], kind: StringSlice, data: List[UInt8], table: List[UInt32]):
    _append_u32_be(out, UInt32(len(data)))
    var crc = UInt32(0xFFFFFFFF)
    for b in kind.as_bytes():
        out.append(b)
        crc = table[Int((crc ^ UInt32(b)) & 0xFF)] ^ (crc >> 8)
    for b in data:
        out.append(b)
        crc = table[Int((crc ^ UInt32(b)) & 0xFF)] ^ (crc >> 8)
    _append_u32_be(out, crc ^ 0xFFFFFFFF)


def write_png(path: String, rgb: List[Float32], width: Int, height: Int) raises:
    """Write an [H, W, 3] float image (clamped to [0, 1]) as an 8-bit RGB PNG.

    Uses uncompressed ("stored") deflate blocks, so no zlib is needed; files
    are about the size of the raw pixels.
    """
    # Raw scanlines: filter byte 0 + RGB bytes.
    var raw = List[UInt8](capacity=height * (1 + width * 3))
    for y in range(height):
        raw.append(0)
        for i in range(width * 3):
            var v = min(max(rgb[y * width * 3 + i], 0.0), 1.0)
            raw.append(UInt8(Int(v * 255.0 + 0.5)))

    # zlib stream with stored blocks (max 65535 bytes each).
    var z = List[UInt8](capacity=len(raw) + len(raw) // 65535 * 5 + 16)
    z.append(0x78)
    z.append(0x01)
    var pos = 0
    while True:
        var block = min(65535, len(raw) - pos)
        var final = pos + block >= len(raw)
        z.append(UInt8(1) if final else UInt8(0))
        z.append(UInt8(block & 0xFF))
        z.append(UInt8((block >> 8) & 0xFF))
        z.append(UInt8(~block & 0xFF))
        z.append(UInt8((~block >> 8) & 0xFF))
        for i in range(block):
            z.append(raw[pos + i])
        pos += block
        if final:
            break
    var a = UInt32(1)
    var b = UInt32(0)
    for v in raw:
        a = (a + UInt32(v)) % 65521
        b = (b + a) % 65521
    _append_u32_be(z, (b << 16) | a)

    var table = _crc32_table()
    var ihdr = List[UInt8]()
    _append_u32_be(ihdr, UInt32(width))
    _append_u32_be(ihdr, UInt32(height))
    for v in [UInt8(8), UInt8(2), UInt8(0), UInt8(0), UInt8(0)]:  # 8-bit RGB
        ihdr.append(v)
    var out: List[UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    _append_chunk(out, "IHDR", ihdr, table)
    _append_chunk(out, "IDAT", z, table)
    _append_chunk(out, "IEND", List[UInt8](), table)
    with open(path, "w") as f:
        f.write_bytes(out)
