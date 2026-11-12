"""Whole programs for any method: `train` and `evaluate`.

`tools/train.mojo` and `tools/eval.mojo` only pick the method named by
`--method` and call these. A repository with its own method can do the same:

    from mojosplat.apps import train

    def main() raises:
        train[MyMethod]()
"""

from std.sys import argv, exit

from ..cli import error

from .evaluation import EvalArgs, evaluate
from .training import TrainArgs, train


def method_arg(default: String) -> String:
    """The value of `--method` on the command line (`--method x` or `--method=x`), or `default`.

    Read before the full parse, because each method has its own options."""
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--method" and i + 1 < len(args):
            return String(args[i + 1])
        if a.startswith("--method="):
            return String(a[byte=9:])
    return default


def unknown_method(name: String, available: List[String]):
    """Report a `--method` that names no method, as a usage error (exit status 2)."""
    var names = String()
    for i in range(len(available)):
        names += String(", " if i > 0 else "", available[i])
    error(String("unknown method '", name, "' (available: ", names, ")"))
    exit(2)
