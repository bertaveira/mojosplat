"""The parts of HTTP/1.1 and WebSocket (RFC 6455) the viewer needs."""

from std.base64 import b64encode

comptime OP_TEXT = 1
comptime OP_BINARY = 2
comptime OP_CLOSE = 8
comptime OP_PING = 9
comptime OP_PONG = 10


@always_inline
def _rotl(x: UInt32, n: UInt32) -> UInt32:
    return (x << n) | (x >> (32 - n))


def sha1(msg: List[UInt8]) -> List[UInt8]:
    """SHA-1 digest (20 bytes). Only used for the WebSocket handshake."""
    var data = msg.copy()
    var bit_len = UInt64(len(msg)) * 8
    data.append(0x80)
    while len(data) % 64 != 56:
        data.append(0)
    for i in range(8):
        data.append(UInt8((bit_len >> UInt64(56 - 8 * i)) & 0xFF))
    var h0: UInt32 = 0x67452301
    var h1: UInt32 = 0xEFCDAB89
    var h2: UInt32 = 0x98BADCFE
    var h3: UInt32 = 0x10325476
    var h4: UInt32 = 0xC3D2E1F0
    var w = List[UInt32](length=80, fill=0)
    for chunk in range(0, len(data), 64):
        for t in range(16):
            var b = chunk + 4 * t
            w[t] = (UInt32(data[b]) << 24) | (UInt32(data[b + 1]) << 16) | (UInt32(data[b + 2]) << 8) | UInt32(data[b + 3])
        for t in range(16, 80):
            w[t] = _rotl(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1)
        var a = h0
        var b = h1
        var c = h2
        var d = h3
        var e = h4
        for t in range(80):
            var f: UInt32
            var k: UInt32
            if t < 20:
                f = (b & c) | (~b & d)
                k = 0x5A827999
            elif t < 40:
                f = b ^ c ^ d
                k = 0x6ED9EBA1
            elif t < 60:
                f = (b & c) | (b & d) | (c & d)
                k = 0x8F1BBCDC
            else:
                f = b ^ c ^ d
                k = 0xCA62C1D6
            var tmp = _rotl(a, 5) + f + e + k + w[t]
            e = d
            d = c
            c = _rotl(b, 30)
            b = a
            a = tmp
        h0 += a
        h1 += b
        h2 += c
        h3 += d
        h4 += e
    var out = List[UInt8](capacity=20)
    for h in [h0, h1, h2, h3, h4]:
        for s in [24, 16, 8, 0]:
            out.append(UInt8((h >> UInt32(s)) & 0xFF))
    return out^


def accept_key(client_key: String) -> String:
    """Sec-WebSocket-Accept for a Sec-WebSocket-Key."""
    var s = client_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    var bytes = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        bytes.append(b)
    var digest = sha1(bytes)
    return b64encode(Span(digest))


def header_value(request: String, name: String) -> String:
    """Value of an HTTP header (case-insensitive name), or ""."""
    var want = name.lower() + ":"
    for line in request.split("\r\n"):
        var l = String(line)
        if l.lower().startswith(want):
            return String(String(l[byte = want.byte_length() :]).strip())
    return ""


def http_response(status: String, content_type: String, body: String) -> List[UInt8]:
    var head = String(
        "HTTP/1.1 ", status, "\r\nContent-Type: ", content_type, "\r\nContent-Length: ", body.byte_length(),
        "\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
    )
    var out = List[UInt8](capacity=head.byte_length() + body.byte_length())
    for b in head.as_bytes():
        out.append(b)
    for b in body.as_bytes():
        out.append(b)
    return out^


def upgrade_response(client_key: String) -> List[UInt8]:
    var head = String(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n",
        "Sec-WebSocket-Accept: ", accept_key(client_key), "\r\n\r\n",
    )
    var out = List[UInt8](capacity=head.byte_length())
    for b in head.as_bytes():
        out.append(b)
    return out^


def append_frame(mut out: List[UInt8], opcode: Int, payload: Span[UInt8, _]):
    """Append one unmasked (server -> client) frame."""
    out.append(UInt8(0x80 | opcode))
    var n = len(payload)
    if n < 126:
        out.append(UInt8(n))
    elif n < 65536:
        out.append(126)
        out.append(UInt8(n >> 8))
        out.append(UInt8(n & 0xFF))
    else:
        out.append(127)
        for s in range(7, -1, -1):
            out.append(UInt8((n >> (8 * s)) & 0xFF))
    out.extend(payload)


def append_text(mut out: List[UInt8], text: String):
    append_frame(out, OP_TEXT, text.as_bytes())


@fieldwise_init
struct Frame(Movable):
    var opcode: Int
    var payload: List[UInt8]


def take_frame(mut buf: List[UInt8]) raises -> Optional[Frame]:
    """Remove and return the first complete (client -> server, masked) frame
    from `buf`, or None if it is incomplete. Fragmented messages are not
    supported (browsers do not fragment small messages)."""
    if len(buf) < 2:
        return None
    var opcode = Int(buf[0] & 0x0F)
    var masked = (buf[1] & 0x80) != 0
    var n = Int(buf[1] & 0x7F)
    var p = 2
    if n == 126:
        if len(buf) < 4:
            return None
        n = (Int(buf[2]) << 8) | Int(buf[3])
        p = 4
    elif n == 127:
        if len(buf) < 10:
            return None
        n = 0
        for i in range(8):
            n = (n << 8) | Int(buf[2 + i])
        p = 10
    if n > 1 << 20:
        raise Error("WebSocket message too large")
    var mask = SIMD[DType.uint8, 4](0)
    if masked:
        if len(buf) < p + 4:
            return None
        for i in range(4):
            mask[i] = buf[p + i]
        p += 4
    if len(buf) < p + n:
        return None
    var payload = List[UInt8](capacity=n)
    for i in range(n):
        payload.append(buf[p + i] ^ mask[i % 4])
    var rest = List[UInt8](capacity=len(buf) - p - n)
    for i in range(p + n, len(buf)):
        rest.append(buf[i])
    buf = rest^
    return Frame(opcode, payload^)
