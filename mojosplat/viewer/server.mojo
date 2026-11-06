"""The viewer server: browsers connect over HTTP + WebSocket, send their
camera, and get JPEG frames rendered by MojoSplat.

Single-threaded and non-blocking: the owner calls `update()` regularly (the
training loop after every step), which accepts connections, reads camera
messages, and renders a frame for each browser whose camera moved, or
periodically while the scene changes (`scene_changed=True`, i.e. training).
A browser gets a new frame only once its previous one was fully sent, so a
slow connection just gets fewer frames.
"""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.ffi import c_int
from std.math import sqrt
from std.time import perf_counter_ns

from ..cli.args import Arg
from ..cli.term import cyan, done, dim
from ..core.camera import Camera
from ..core.types import BLOCK_SIZE, F32Ptr, U8Ptr, dptr, num_blocks
from ..io.arrays import download
from ..io.jpeg import JpegEncoder
from ..model.splats import Attr, Splats
from ..render.rasterizer import Rasterizer
from .net import POLLERR, POLLHUP, POLLIN, POLLOUT, PollSet, accept_client, close_fd, listen_tcp, recv_some, send_some
from .page import PAGE
from .websocket import (
    OP_BINARY,
    OP_CLOSE,
    OP_PING,
    OP_PONG,
    OP_TEXT,
    append_frame,
    append_text,
    header_value,
    http_response,
    take_frame,
    upgrade_response,
)


struct ViewerConfig(Copyable, Defaultable, Movable):
    @__annotation(Arg("Viewer port"))
    var port: Int

    @__annotation(Arg("Address to listen on (0.0.0.0: reachable from other machines)"))
    var host: String

    @__annotation(Arg("Maximum frames per second per browser"))
    var max_fps: Int

    @__annotation(Arg("Frames per second while only the scene changes (training)"))
    var refresh_fps: Int

    @__annotation(Arg("JPEG quality of the frames (1-100)"))
    var jpeg_quality: Int

    @__annotation(Arg("Longest image side rendered for a browser (pixels)"))
    var max_size: Int

    def __init__(out self):
        self.port = 8080
        self.host = "0.0.0.0"
        self.max_fps = 30
        self.refresh_fps = 5
        self.jpeg_quality = 85
        self.max_size = 1920


def _to_rgb8_kernel(src: F32Ptr, dst: U8Ptr, n: Int32):
    var i = block_idx.x * BLOCK_SIZE + thread_idx.x
    if i < Int(n):
        var v = min(max(src[unsafe_offset=i], Float32(0.0)), Float32(1.0))
        dst[unsafe_offset=i] = UInt8(Int(v * 255.0 + 0.5))


struct _Client(Movable):
    var fd: c_int
    var inbuf: List[UInt8]
    var out: List[UInt8]
    var sent: Int
    var websocket: Bool
    var close_after_send: Bool
    var dead: Bool
    var camera: Camera
    var has_camera: Bool
    var camera_moved: Bool
    var last_frame: Int

    def __init__(out self, fd: c_int):
        self.fd = fd
        self.inbuf = List[UInt8]()
        self.out = List[UInt8]()
        self.sent = 0
        self.websocket = False
        self.close_after_send = False
        self.dead = False
        self.camera = Camera.translated(0.0, 0.0, 0.0, 16, 16, 16.0, 16.0)
        self.has_camera = False
        self.camera_moved = False
        self.last_frame = 0

    def pending(self) -> Int:
        return len(self.out) - self.sent


def _json_floats(v: SIMD[DType.float32, _], n: Int) -> String:
    var s = String("[")
    for i in range(n):
        if i > 0:
            s += ","
        s += String(v[i])
    return s + "]"


def json_string(s: String) -> String:
    """`s` as a JSON string literal."""
    var out = String('"')
    for cp in s.codepoint_slices():
        if cp == '"':
            out += '\\"'
        elif cp == "\\":
            out += "\\\\"
        elif cp == "\n":
            out += "\\n"
        else:
            out += cp
    return out + '"'


def _view_json(cam: Camera, dist: Float32) -> String:
    """A camera as the page's view object: camera-to-world rotation (row-major),
    position, fy / height, and orbit distance."""
    var v = cam.view
    var rot = SIMD[DType.float32, 16](0.0)
    for i in range(3):
        for j in range(3):
            rot[i * 3 + j] = v[j * 4 + i]  # transpose
    var pos = cam.position()
    return String(
        '{"rot":',
        _json_floats(rot, 9),
        ',"pos":',
        _json_floats(pos, 3),
        ',"fy":',
        cam.fy / Float32(cam.height),
        ',"dist":',
        dist,
        "}",
    )


def _focus_point(views: List[Camera], fallback: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    """The point closest (least squares) to all cameras' optical axes: what a
    360-degree capture looks at. `fallback` if the axes are near parallel
    (forward-facing captures) or the point is implausibly far away."""
    var A = SIMD[DType.float64, 16](0.0)  # sum of (I - f f^T), 3x3 in a 4x4
    var b = SIMD[DType.float64, 4](0.0)
    var mean_p = SIMD[DType.float64, 4](0.0)
    for c in views:
        var p = c.position().cast[DType.float64]()
        var f = SIMD[DType.float64, 4](Float64(c.view[8]), Float64(c.view[9]), Float64(c.view[10]), 0.0)
        mean_p += p
        for i in range(3):
            for j in range(3):
                var m = (1.0 if i == j else 0.0) - f[i] * f[j]
                A[i * 4 + j] += m
                b[i] += m * p[j]
    var n = Float64(len(views))
    mean_p /= n
    var det = (
        A[0] * (A[5] * A[10] - A[6] * A[9]) - A[1] * (A[4] * A[10] - A[6] * A[8]) + A[2] * (A[4] * A[9] - A[5] * A[8])
    )
    if abs(det) < 1e-3 * n * n * n:
        return fallback
    var x = SIMD[DType.float64, 4](0.0)
    for k in range(3):  # Cramer's rule
        var M = A
        for i in range(3):
            M[i * 4 + k] = b[i]
        x[k] = (
            M[0] * (M[5] * M[10] - M[6] * M[9])
            - M[1] * (M[4] * M[10] - M[6] * M[8])
            + M[2] * (M[4] * M[9] - M[5] * M[8])
        ) / det
    var spread: Float64 = 0
    for c in views:
        var d = c.position().cast[DType.float64]() - mean_p
        spread += sqrt((d * d).reduce_add())
    spread = max(spread / n, 1e-6)
    var off = x - mean_p
    if sqrt((off * off).reduce_add()) > 10.0 * spread:
        return fallback
    return x.cast[DType.float32]()


struct Viewer(Movable):
    """Serves a web viewer for one set of Gaussians (see the module doc)."""

    var ctx: DeviceContext
    var config: ViewerConfig
    var listen_fd: c_int
    var clients: List[_Client]
    var polls: PollSet
    var rasterizer: Rasterizer[]
    var image: DeviceBuffer[DType.float32]
    var rgb_dev: DeviceBuffer[DType.uint8]
    var rgb: List[UInt8]
    var encoder: JpegEncoder
    var views: List[Camera]
    var center: SIMD[DType.float32, 4]
    var init_json: String
    var stats_json: String
    var last_stats: Int
    var paused: Bool
    var training: Bool
    var background: SIMD[DType.float32, 4]
    var frames_sent: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        config: ViewerConfig,
        splats: Splats,
        views: List[Camera],
        title: String,
        training: Bool,
    ) raises:
        self.ctx = ctx
        self.config = config.copy()
        self.listen_fd = listen_tcp(config.host, config.port)
        self.clients = List[_Client]()
        self.polls = PollSet()
        self.rasterizer = Rasterizer(ctx)
        self.image = ctx.enqueue_create_buffer[DType.float32](3)
        self.rgb_dev = ctx.enqueue_create_buffer[DType.uint8](3)
        self.rgb = List[UInt8]()
        self.encoder = JpegEncoder(config.jpeg_quality)
        self.views = views.copy()
        self.center = SIMD[DType.float32, 4](0.0)
        self.init_json = ""
        self.stats_json = '{"type":"stats","items":[],"progress":-1,"paused":false}'
        self.last_stats = 0
        self.paused = False
        self.training = training
        self.background = SIMD[DType.float32, 4](0.0)
        self.frames_sent = 0
        self._make_init(splats, title)
        var shown_host = config.host if config.host != "0.0.0.0" else String("localhost")
        done(
            String(
                "Viewer at ",
                cyan(String("http://", shown_host, ":", config.port)),
                dim(
                    String(
                        " (listening on ",
                        config.host,
                        ")" if config.host != "0.0.0.0" else ", reachable from other machines)",
                    )
                ),
            )
        )

    def __deinit__(deinit self):
        for i in range(len(self.clients)):
            close_fd(self.clients[i].fd)
        close_fd(self.listen_fd)

    # ------------------------------------------------------------------
    # Initial camera
    # ------------------------------------------------------------------

    def _make_init(mut self, splats: Splats, title: String) raises:
        """Scene center (median of the means), world up and the first view."""
        var n = splats.count
        var means = download(self.ctx, splats.params[Attr.MEANS], n * 3)
        var stride = max(n // 20000, 1)
        var xs = List[Float32]()
        var ys = List[Float32]()
        var zs = List[Float32]()
        for i in range(0, n, stride):
            var x = means[i * 3]
            var y = means[i * 3 + 1]
            var z = means[i * 3 + 2]
            if abs(x) < 1e30 and abs(y) < 1e30 and abs(z) < 1e30:  # skips NaN / inf
                xs.append(x)
                ys.append(y)
                zs.append(z)
        if len(xs) == 0:
            xs.append(0.0)
            ys.append(0.0)
            zs.append(0.0)
        sort(xs)
        sort(ys)
        sort(zs)
        var m = len(xs) // 2
        var center = SIMD[DType.float32, 4](xs[m], ys[m], zs[m], 0.0)
        if len(self.views) > 0:
            center = _focus_point(self.views, center)
        self.center = center

        var up = SIMD[DType.float32, 4](0.0, -1.0, 0.0, 0.0)  # OpenCV / COLMAP: -y
        var view: Camera
        var dist: Float32
        if len(self.views) > 0:
            up = SIMD[DType.float32, 4](0.0)
            for c in self.views:
                for k in range(3):
                    up[k] -= c.view[4 + k]  # minus the camera's down axis
            up /= sqrt((up * up).reduce_add())
            view = self.views[0]
            var p = view.position()
            var f = SIMD[DType.float32, 4](view.view[8], view.view[9], view.view[10], 0.0)
            dist = max(((center - p) * f).reduce_add(), Float32(0.1))
        else:
            var radii = List[Float32]()
            for i in range(len(xs)):
                var d = SIMD[DType.float32, 4](xs[i], ys[i], zs[i], 0.0) - center
                radii.append(sqrt((d * d).reduce_add()))
            sort(radii)
            # 360-degree captures are surrounded by a shell of background
            # Gaussians; ~0.6x their median radius is about where the cameras were.
            dist = max(radii[len(radii) // 2] * 0.6, Float32(0.1))
            # Look along +z (or +x if up is along z), with `up` as up.
            var d = -up
            var e = SIMD[DType.float32, 4](0.0, 0.0, 1.0, 0.0)
            if abs((e * d).reduce_add()) > 0.9:
                e = SIMD[DType.float32, 4](1.0, 0.0, 0.0, 0.0)
            var f = e - d * (e * d).reduce_add()
            f /= sqrt((f * f).reduce_add())
            var r = SIMD[DType.float32, 4](
                d[1] * f[2] - d[2] * f[1], d[2] * f[0] - d[0] * f[2], d[0] * f[1] - d[1] * f[0], 0.0
            )
            var p = center - f * dist
            var v = SIMD[DType.float32, 16](0.0)
            for k in range(3):
                v[k] = r[k]
                v[4 + k] = d[k]
                v[8 + k] = f[k]
            v[3] = -(r * p).reduce_add()
            v[7] = -(d * p).reduce_add()
            v[11] = -(f * p).reduce_add()
            v[15] = 1.0
            view = Camera(v, 900.0, 900.0, 640.0, 360.0, 1280, 720, 0.1, 100.0)
        self.init_json = String(
            '{"type":"init","title":',
            json_string(title),
            ',"training":',
            "true" if self.training else "false",
            ',"views":',
            len(self.views),
            ',"max_size":',
            self.config.max_size,
            ',"up":',
            _json_floats(up, 3),
            ',"view":',
            _view_json(view, dist),
            "}",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_stats(mut self, items: List[Tuple[String, String]], progress: Float64 = -1.0):
        """Rows (label, value) and a progress fraction (< 0: none) shown in the
        page's panel; sent to the browsers a few times a second."""
        var s = String('{"type":"stats","items":[')
        for i in range(len(items)):
            if i > 0:
                s += ","
            s += String("[", json_string(items[i][0]), ",", json_string(items[i][1]), "]")
        s += String('],"progress":', progress, ',"paused":', "true" if self.paused else "false", "}")
        self.stats_json = s

    def num_clients(self) -> Int:
        var n = 0
        for i in range(len(self.clients)):
            n += 1 if self.clients[i].websocket else 0
        return n

    def update(mut self, splats: Splats, sh_degree: Int = 3, scene_changed: Bool = False, timeout_ms: Int = 0) raises:
        """Handle network events and send due frames. Waits at most
        `timeout_ms` for network activity (0: never blocks)."""
        self.polls.clear()
        self.polls.add(self.listen_fd, POLLIN)
        var wait_ms = timeout_ms
        var start = Int(perf_counter_ns())
        for i in range(len(self.clients)):
            ref c = self.clients[i]
            self.polls.add(c.fd, POLLIN | (POLLOUT if c.pending() > 0 else 0))
            # Wake up when this browser's next frame is due, not at the end of the timeout.
            if c.websocket and c.has_camera and c.camera_moved and c.pending() == 0:
                var due_in_ns = c.last_frame + 1_000_000_000 // max(self.config.max_fps, 1) - start
                wait_ms = min(wait_ms, max(0, (due_in_ns + 999_999) // 1_000_000))
        if self.polls.wait(wait_ms) > 0:
            for i in range(len(self.clients)):
                var ev = self.polls.revents(i + 1)
                if ev & (POLLIN | POLLERR | POLLHUP):
                    self._read(i)
            if self.polls.revents(0) & POLLIN:
                var fd = accept_client(self.listen_fd)
                if fd >= 0:
                    self.clients.append(_Client(fd))

        var now = Int(perf_counter_ns())
        var send_stats = now - self.last_stats > 250_000_000
        if send_stats:
            self.last_stats = now
        for i in range(len(self.clients)):
            if self.clients[i].dead or not self.clients[i].websocket:
                continue
            if send_stats:
                append_text(self.clients[i].out, self.stats_json)
            if self.clients[i].has_camera and self.clients[i].pending() == 0:
                var since = now - self.clients[i].last_frame
                var due = self.clients[i].camera_moved and since * self.config.max_fps >= 1_000_000_000
                due = due or (scene_changed and since * self.config.refresh_fps >= 1_000_000_000)
                if due:
                    self._render(i, splats, sh_degree)
                    self.clients[i].camera_moved = False
                    # Schedule from the decision time, not the end of the render: otherwise
                    # every frame also waits for the render, and the rate drops below max_fps.
                    self.clients[i].last_frame = now
        for i in range(len(self.clients)):
            self._flush(i)
        self._drop_dead()

    def serve_forever(mut self, splats: Splats, sh_degree: Int = 3) raises:
        """Serve until the process is interrupted (Ctrl+C)."""
        while True:
            self.update(splats, sh_degree, False, 10)

    def wait_while_paused(mut self, splats: Splats, sh_degree: Int = 3) raises:
        """Keep serving (no scene changes) while a browser has paused training."""
        while self.paused:
            self.update(splats, sh_degree, False, 10)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _render(mut self, i: Int, splats: Splats, sh_degree: Int) raises:
        var cam = self.clients[i].camera
        var n = cam.width * cam.height * 3
        if len(self.image) < n:
            self.image = self.ctx.enqueue_create_buffer[DType.float32](n)
            self.rgb_dev = self.ctx.enqueue_create_buffer[DType.uint8](n)
        self.rasterizer.render(splats, cam, self.background, self.image, sh_degree)
        self.ctx.enqueue_function[_to_rgb8_kernel](
            dptr(self.image), dptr(self.rgb_dev), Int32(n), grid_dim=num_blocks(n), block_dim=BLOCK_SIZE
        )
        self.rgb.resize(unsafe_uninit_length=n)
        self.ctx.enqueue_copy(Span(self.rgb), self.rgb_dev.create_sub_buffer[DType.uint8](0, n))
        self.ctx.synchronize()
        var size = self.encoder.encode(self.rgb, cam.width, cam.height)
        append_frame(self.clients[i].out, OP_BINARY, Span(self.encoder.data)[:size])
        self.frames_sent += 1

    def _read(mut self, i: Int) raises:
        ref c = self.clients[i]
        var n = recv_some(c.fd, c.inbuf)
        if n == 0 or n == -1:
            c.dead = True
            return
        if len(c.inbuf) > 1 << 20:
            c.dead = True  # nothing we accept is that large
            return
        if not c.websocket:
            self._read_http(i)
        else:
            self._read_websocket(i)

    def _read_http(mut self, i: Int) raises:
        ref c = self.clients[i]
        var text = String(unsafe_from_utf8=Span(c.inbuf))
        var end = text.find("\r\n\r\n")
        if end < 0:
            return
        var request = String(text[byte=:end])
        c.inbuf.clear()
        var line = request.split("\r\n")[0]
        var parts = line.split(" ")
        var path = String(parts[1]) if len(parts) >= 2 else String("/")
        if path == "/ws" and header_value(request, "Upgrade").lower() == "websocket":
            var key = header_value(request, "Sec-WebSocket-Key")
            c.out = upgrade_response(key)
            c.sent = 0
            c.websocket = True
            # Training may have finished since _make_init built this message.
            append_text(
                c.out,
                self.init_json.replace('"training":true', '"training":true' if self.training else '"training":false'),
            )
            append_text(c.out, self.stats_json)
        elif path == "/" or path == "/index.html":
            c.out = http_response("200 OK", "text/html; charset=utf-8", String(PAGE))
            c.close_after_send = True
        else:
            c.out = http_response("404 Not Found", "text/plain", "not found")
            c.close_after_send = True

    def _read_websocket(mut self, i: Int) raises:
        while True:
            var f = take_frame(self.clients[i].inbuf)
            if not f:
                return
            var frame = f.take()
            if frame.opcode == OP_CLOSE:
                self.clients[i].dead = True
                return
            elif frame.opcode == OP_PING:
                append_frame(self.clients[i].out, OP_PONG, Span(frame.payload))
            elif frame.opcode == OP_TEXT:
                self._command(i, String(unsafe_from_utf8=Span(frame.payload)))

    def _command(mut self, i: Int, msg: String) raises:
        var parts = msg.split(" ")
        if len(parts) == 0:
            return
        var cmd = String(parts[0])
        if cmd == "cam" and len(parts) == 16:
            var W = max(Int(String(parts[1])), 16)
            var H = max(Int(String(parts[2])), 16)
            var s = min(Float64(self.config.max_size) / Float64(max(W, H)), 1.0)
            W = max(Int(Float64(W) * s), 16)
            H = max(Int(Float64(H) * s), 16)
            var fy = Float32(Float64(String(parts[3])) * s)
            var v = SIMD[DType.float32, 16](0.0)
            for k in range(12):
                v[k] = Float32(Float64(String(parts[4 + k])))
            v[15] = 1.0
            ref c = self.clients[i]
            c.camera = Camera(v, fy, fy, Float32(W) / 2.0, Float32(H) / 2.0, W, H, 0.01, 1000.0)
            c.has_camera = True
            c.camera_moved = True
        elif cmd == "pause" and self.training:
            self.paused = True
            self._stats_paused_flag()
        elif cmd == "resume":
            self.paused = False
            self._stats_paused_flag()
        elif cmd == "view" and len(parts) == 2 and len(self.views) > 0:
            var k = Int(String(parts[1])) % len(self.views)
            var cam = self.views[k]
            var p = cam.position()
            var f = SIMD[DType.float32, 4](cam.view[8], cam.view[9], cam.view[10], 0.0)
            var dist = max(((self.center - p) * f).reduce_add(), Float32(0.1))
            append_text(self.clients[i].out, String('{"type":"view","view":', _view_json(cam, dist), "}"))

    def _stats_paused_flag(mut self):
        var a = '"paused":true' if not self.paused else '"paused":false'
        var b = '"paused":true' if self.paused else '"paused":false'
        self.stats_json = self.stats_json.replace(a, b)
        self.last_stats = 0  # send soon

    def _flush(mut self, i: Int):
        ref c = self.clients[i]
        while c.pending() > 0 and not c.dead:
            var n = send_some(c.fd, c.out, c.sent)
            if n < 0:
                c.dead = True
            elif n == 0:
                break
            c.sent += max(n, 0)
        if c.pending() == 0 and len(c.out) > 0:
            c.out.clear()
            c.sent = 0
            if c.close_after_send:
                c.dead = True

    def _drop_dead(mut self):
        var i = 0
        while i < len(self.clients):
            if self.clients[i].dead:
                close_fd(self.clients[i].fd)
                _ = self.clients.pop(i)
            else:
                i += 1
