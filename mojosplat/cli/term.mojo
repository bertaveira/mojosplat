"""Terminal output: colors, number formatting, a progress bar and tables.

Colors and the redrawn progress bar are used only when stdout is a terminal
(and `NO_COLOR` is unset); otherwise output is plain lines, so logs and
pipes stay readable.
"""

from std.ffi import c_int, external_call
from std.os import getenv
from std.sys import stderr
from std.time import perf_counter_ns


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------


def is_terminal(fd: Int = 1) -> Bool:
    """Whether file descriptor `fd` (default: stdout) is a terminal."""
    return external_call["isatty", c_int](c_int(fd)) == 1


def colors_enabled() -> Bool:
    return is_terminal() and getenv("NO_COLOR").byte_length() == 0 and getenv("TERM") != "dumb"


def _sgr(code: StaticString, s: String) -> String:
    if not colors_enabled():
        return s
    return String("\x1b[", code, "m", s, "\x1b[0m")


def bold(s: String) -> String:
    return _sgr("1", s)


def dim(s: String) -> String:
    return _sgr("2", s)


def red(s: String) -> String:
    return _sgr("31", s)


def green(s: String) -> String:
    return _sgr("32", s)


def yellow(s: String) -> String:
    return _sgr("33", s)


def cyan(s: String) -> String:
    return _sgr("36", s)


def magenta(s: String) -> String:
    return _sgr("35", s)


def done(msg: String):
    """Print a completed step: a green check mark and the message."""
    print(green("✓"), msg)


def warn(msg: String):
    print(yellow("!"), msg)


def error(msg: String):
    """Print an error message to stderr."""
    print(red(bold("error:")), msg, file=stderr)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def fmt_fixed(x: Float64, digits: Int) -> String:
    """`x` with exactly `digits` decimals (rounded half away from zero)."""
    if x != x:
        return "nan"
    var scale: Float64 = 1.0
    for _ in range(digits):
        scale *= 10.0
    var neg = x < 0
    var r = Int(abs(x) * scale + 0.5)
    var whole = r // Int(scale)
    var frac = r % Int(scale)
    var s = String("-" if neg and r != 0 else "", whole)
    if digits > 0:
        var f = String(frac)
        s += "."
        for _ in range(digits - f.byte_length()):
            s += "0"
        s += f
    return s


def fmt_count(n: Int) -> String:
    """Compact count: 950, 138.8k, 4.21M."""
    if n < 1000:
        return String(n)
    if n < 1_000_000:
        return String(fmt_fixed(Float64(n) / 1e3, 1), "k")
    return String(fmt_fixed(Float64(n) / 1e6, 2), "M")


def fmt_thousands(n: Int) -> String:
    """4219236 -> 4,219,236."""
    var digits = String(abs(n))
    var out = String("-" if n < 0 else "")
    var k = digits.byte_length()
    for i in range(k):
        out += String(digits[byte=i])
        var left = k - 1 - i
        if left > 0 and left % 3 == 0:
            out += ","
    return out


def fmt_duration(seconds: Float64) -> String:
    """12.3s, 3m31s, 1h05m."""
    if seconds < 60.0:
        return String(fmt_fixed(seconds, 1), "s")
    var s = Int(seconds + 0.5)
    if s < 3600:
        var sec = s % 60
        return String(s // 60, "m", "0" if sec < 10 else "", sec, "s")
    var m = (s // 60) % 60
    return String(s // 3600, "h", "0" if m < 10 else "", m, "m")


def visible_width(s: String) -> Int:
    """Terminal columns of `s`: code points, not counting ANSI escapes."""
    var n = 0
    var in_escape = False
    for byte in s.as_bytes():
        var b = Int(byte)
        if in_escape:
            if (b >= ord("a") and b <= ord("z")) or (b >= ord("A") and b <= ord("Z")):
                in_escape = False
        elif b == 0x1B:
            in_escape = True
        elif (b & 0xC0) != 0x80:  # not a UTF-8 continuation byte
            n += 1
    return n


def spaces(n: Int) -> String:
    var s = String()
    for _ in range(n):
        s += " "
    return s


def seconds_since(t0_ns: Int) -> Float64:
    return Float64(Int(perf_counter_ns()) - t0_ns) / 1e9


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


struct ProgressBar(Movable):
    """A single redrawn line: bar, count, rate, elapsed<remaining, and a
    caller-provided info string. Without a terminal, `update(..., log=True)`
    prints a plain line instead (so a log file gets one line per log step).
    """

    var total: Int
    var label: String
    var unit: String
    var width: Int
    var interactive: Bool
    var t_start: Int
    var t_draw: Int
    var last_logged: Int  # last step printed as a plain line

    def __init__(out self, total: Int, label: String = "", unit: String = "it", width: Int = 28):
        self.total = max(total, 1)
        self.label = label
        self.unit = unit
        self.width = width
        self.interactive = is_terminal()
        self.t_start = Int(perf_counter_ns())
        self.t_draw = 0
        self.last_logged = -1

    def elapsed(self) -> Float64:
        return seconds_since(self.t_start)

    def _line(self, step: Int, info: String) -> String:
        var frac = min(Float64(step) / Float64(self.total), 1.0)
        var elapsed = self.elapsed()
        var rate = Float64(step) / elapsed if elapsed > 0 else 0.0
        var eta = Float64(self.total - step) / rate if rate > 0 else 0.0
        var filled = Int(frac * Float64(self.width))
        var bar = String()
        for _ in range(filled):
            bar += "━"
        var rest = String()
        var empty = "━" if colors_enabled() else "·"  # dimmed when colored
        for _ in range(self.width - filled):
            rest += empty
        var pct = String(Int(frac * 100.0))
        var line = String()
        if self.label.byte_length() > 0:
            line += bold(self.label) + " "
        line += cyan(bar) + dim(rest) + " "
        line += String(spaces(3 - pct.byte_length()), pct, "% ")
        line += String(step, "/", self.total, " ")
        line += dim("|") + String(" ", fmt_fixed(rate, 0), " ", self.unit, "/s ") + dim("|")
        line += String(" ", fmt_duration(elapsed), dim("<"), fmt_duration(eta))
        if info.byte_length() > 0:
            line += String(" ", dim("|"), " ", info)
        return line

    def update(mut self, step: Int, info: String = "", log: Bool = False):
        """Redraw (at most ~10 times a second) after `step` of `total`."""
        if self.interactive:
            var now = Int(perf_counter_ns())
            if now - self.t_draw < 100_000_000 and step < self.total:
                return
            self.t_draw = now
            print("\r\x1b[2K", self._line(step, info), sep="", end="", flush=True)
        elif log:
            print(self._line(step, info))
            self.last_logged = step

    def message(mut self, msg: String):
        """Print a line above the bar (the bar is redrawn on the next update)."""
        if self.interactive:
            print("\r\x1b[2K", msg, sep="")
            self.t_draw = 0
        else:
            print(msg)

    def finish(mut self, info: String = ""):
        """Draw the final state and end the line."""
        if self.interactive:
            print("\r\x1b[2K", self._line(self.total, info), sep="", flush=True)
        elif self.last_logged != self.total:
            print(self._line(self.total, info))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


struct Table(Movable):
    """Two-column key / value summary."""

    var title: String
    var keys: List[String]
    var values: List[String]

    def __init__(out self, title: String = ""):
        self.title = title
        self.keys = List[String]()
        self.values = List[String]()

    def add(mut self, key: String, value: String):
        self.keys.append(key)
        self.values.append(value)

    def print(self):
        var w = 0
        for k in self.keys:
            w = max(w, visible_width(k))
        var rule_w = w + 2
        for v in self.values:
            rule_w = max(rule_w, w + 2 + visible_width(v))
        rule_w = min(rule_w, 48)
        if self.title.byte_length() > 0:
            print(bold(self.title))
        var rule = String()
        for _ in range(rule_w):
            rule += "─"
        print(dim(rule))
        for i in range(len(self.keys)):
            var pad = spaces(w - visible_width(self.keys[i]) + 2)
            print(dim(self.keys[i]), pad, self.values[i], sep="")
        print(dim(rule))
