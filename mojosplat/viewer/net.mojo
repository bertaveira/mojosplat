"""Minimal non-blocking TCP over libc (Linux and macOS).

Only non-variadic C functions are called (socket, setsockopt, bind, listen,
accept, recv, send, poll, close): variadic ones such as fcntl use a
different calling convention on Apple arm64, which Mojo's `external_call`
does not model. Non-blocking I/O therefore comes from `poll` plus
MSG_DONTWAIT on every recv/send.
"""

from std.ffi import c_int, external_call, get_errno, ErrNo
from std.sys.info import CompilationTarget

comptime _AF_INET = 2
comptime _SOCK_STREAM = 1
comptime _SOL_SOCKET = 0xFFFF if CompilationTarget.is_macos() else 1
comptime _SO_REUSEADDR = 4 if CompilationTarget.is_macos() else 2
comptime _SO_NOSIGPIPE = 0x1022  # macOS only
comptime _MSG_DONTWAIT = 0x80 if CompilationTarget.is_macos() else 0x40
comptime _MSG_NOSIGNAL = 0 if CompilationTarget.is_macos() else 0x4000
comptime POLLIN = 1
comptime POLLOUT = 4
comptime POLLERR = 8
comptime POLLHUP = 16


def _sockaddr_in(host: String, port: Int) raises -> List[UInt8]:
    """struct sockaddr_in for an IPv4 dotted-quad address."""
    var a = List[UInt8](length=16, fill=0)
    comptime if CompilationTarget.is_macos():
        a[0] = 16  # sin_len
        a[1] = _AF_INET
    else:
        a[0] = _AF_INET
    a[2] = UInt8((port >> 8) & 0xFF)  # network byte order
    a[3] = UInt8(port & 0xFF)
    var parts = host.split(".")
    if len(parts) != 4:
        raise Error(String("invalid IPv4 address '", host, "'"))
    for i in range(4):
        var v = Int(String(parts[i]))
        if v < 0 or v > 255:
            raise Error(String("invalid IPv4 address '", host, "'"))
        a[4 + i] = UInt8(v)
    return a^


def _set_int_option(fd: c_int, option: Int):
    var one = Int32(1)
    _ = external_call["setsockopt", c_int](fd, c_int(_SOL_SOCKET), c_int(option), Pointer(to=one), c_int(4))


def listen_tcp(host: String, port: Int) raises -> c_int:
    """A listening socket on host:port."""
    var fd = external_call["socket", c_int](c_int(_AF_INET), c_int(_SOCK_STREAM), c_int(0))
    if fd < 0:
        raise Error(String("socket() failed: ", get_errno()))
    _set_int_option(fd, _SO_REUSEADDR)
    var addr = _sockaddr_in(host, port)
    if external_call["bind", c_int](fd, addr.unsafe_ptr(), c_int(16)) != 0:
        var err = get_errno()
        close_fd(fd)
        raise Error(String("cannot listen on ", host, ":", port, ": ", err))
    if external_call["listen", c_int](fd, c_int(16)) != 0:
        var err = get_errno()
        close_fd(fd)
        raise Error(String("listen() failed: ", err))
    return fd


def accept_client(listen_fd: c_int) -> c_int:
    """Accept one pending connection (-1 if none)."""
    var fd = external_call["accept", c_int](listen_fd, Int(0), Int(0))
    if fd >= 0:
        comptime if CompilationTarget.is_macos():
            _set_int_option(fd, _SO_NOSIGPIPE)
    return fd


def close_fd(fd: c_int):
    _ = external_call["close", c_int](fd)


def would_block() -> Bool:
    var e = get_errno()
    return e == ErrNo.EAGAIN or e == ErrNo.EWOULDBLOCK or e == ErrNo.EINTR


def recv_some(fd: c_int, mut into: List[UInt8], max_bytes: Int = 65536) -> Int:
    """Append up to `max_bytes` received bytes to `into`. Returns the count,
    0 when the peer closed, -1 on error, -2 when nothing is available."""
    var old = len(into)
    into.resize(unsafe_uninit_length=old + max_bytes)
    var n = external_call["recv", Int](fd, into.unsafe_ptr().unsafe_offset(old), max_bytes, c_int(_MSG_DONTWAIT))
    into.resize(unsafe_uninit_length=old + max(n, 0))
    if n < 0:
        return -2 if would_block() else -1
    return n


def send_some(fd: c_int, data: List[UInt8], start: Int) -> Int:
    """Send data[start:]; returns bytes sent (0 if the socket is full), -1 on error."""
    var n = external_call["send", Int](
        fd, data.unsafe_ptr().unsafe_offset(start), len(data) - start, c_int(_MSG_DONTWAIT | _MSG_NOSIGNAL)
    )
    if n < 0:
        return 0 if would_block() else -1
    return n


struct PollSet(Movable):
    """struct pollfd[] for `poll`: per entry one Int32 fd and one Int32 that
    holds `short events` (low half) and `short revents` (high half)."""

    var words: List[Int32]

    def __init__(out self):
        self.words = List[Int32]()

    def clear(mut self):
        self.words.clear()

    def add(mut self, fd: c_int, events: Int):
        self.words.append(Int32(fd))
        self.words.append(Int32(events))

    def wait(mut self, timeout_ms: Int) -> Int:
        if len(self.words) == 0:
            return 0
        return Int(external_call["poll", c_int](self.words.unsafe_ptr(), len(self.words) // 2, c_int(timeout_ms)))

    def revents(self, i: Int) -> Int:
        return Int((self.words[2 * i + 1] >> 16) & 0xFFFF)
