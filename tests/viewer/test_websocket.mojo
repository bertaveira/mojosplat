"""HTTP / WebSocket helpers of the viewer (no network needed)."""

from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojosplat.viewer.server import json_string
from mojosplat.viewer.websocket import OP_BINARY, OP_TEXT, accept_key, append_frame, header_value, sha1, take_frame


def _hex(b: List[UInt8]) -> String:
    comptime digits = "0123456789abcdef"
    var s = String()
    for x in b:
        s += String(digits[byte = Int(x >> 4)]) + String(digits[byte = Int(x & 15)])
    return s


def _bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    for b in s.as_bytes():
        out.append(b)
    return out^


def test_sha1() raises:
    assert_equal(_hex(sha1(List[UInt8]())), "da39a3ee5e6b4b0d3255bfef95601890afd80709")
    assert_equal(_hex(sha1(_bytes("abc"))), "a9993e364706816aba3e25717850c26c9cd0d89d")
    # two blocks
    assert_equal(
        _hex(sha1(_bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))),
        "84983e441c3bd26ebaae4aa1f95129e5e54670f1",
    )


def test_accept_key() raises:
    # RFC 6455, section 1.3
    assert_equal(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")


def test_header_value() raises:
    var req = "GET /ws HTTP/1.1\r\nHost: x\r\nupgrade: WebSocket\r\nSec-WebSocket-Key:  abc== \r\n"
    assert_equal(header_value(req, "Upgrade"), "WebSocket")
    assert_equal(header_value(req, "sec-websocket-key"), "abc==")
    assert_equal(header_value(req, "Missing"), "")


def test_frames() raises:
    # A masked client text frame "hi", then the start of a second frame.
    var buf: List[UInt8] = [0x81, 0x82, 1, 2, 3, 4, UInt8(ord("h") ^ 1), UInt8(ord("i") ^ 2), 0x81]
    var f = take_frame(buf)
    assert_true(Bool(f))
    var frame = f.take()
    assert_equal(frame.opcode, OP_TEXT)
    assert_equal(String(unsafe_from_utf8=Span(frame.payload)), "hi")
    assert_equal(len(buf), 1)
    assert_false(Bool(take_frame(buf)))  # incomplete
    # Server frames: 7-bit, 16-bit and 64-bit lengths.
    for n in [5, 300, 70000]:
        var payload = List[UInt8](length=n, fill=7)
        var out = List[UInt8]()
        append_frame(out, OP_BINARY, Span(payload))
        var header = 2 if n < 126 else (4 if n < 65536 else 10)
        assert_equal(len(out), header + n)
        assert_equal(Int(out[0]), 0x80 | OP_BINARY)
        var back = take_frame(out)  # unmasked frames parse too
        assert_equal(len(back.take().payload), n)


def test_json_string() raises:
    assert_equal(json_string('a "b"\\c'), '"a \\"b\\"\\\\c"')


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
