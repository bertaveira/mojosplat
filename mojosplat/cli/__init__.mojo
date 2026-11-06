"""Helpers for command-line programs: options generated from config structs
(`parse_args`) and terminal output (colors, progress bar, tables).

Depends only on the standard library, so configs anywhere in the package
can carry `Arg` help annotations."""

from .args import Arg, parse_args, parse_args_from
from .term import (
    ProgressBar,
    Table,
    bold,
    colors_enabled,
    cyan,
    dim,
    done,
    error,
    fmt_count,
    fmt_duration,
    fmt_fixed,
    fmt_thousands,
    green,
    is_terminal,
    magenta,
    red,
    seconds_since,
    spaces,
    visible_width,
    warn,
    yellow,
)
