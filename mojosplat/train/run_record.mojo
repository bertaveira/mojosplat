"""`run.json`: everything needed to tell what produced a result."""

from std.reflection import source_location
from std.subprocess import run

from ..cli.args import config_items


def _git(root: String, command: String) -> String:
    """Output of `git -C <root> <command>`, or "" if git fails or is missing."""
    try:
        return String(run(String("git -C '", root, "' ", command, " 2>/dev/null")).strip())
    except:
        return ""


def mojosplat_root() -> String:
    """The MojoSplat checkout this program was built from (where `mojosplat/` lives).

    Taken from this file's path at compile time, so it is MojoSplat's own
    repository even when the program belongs to another one that uses it. The
    path is relative when the package is found through a relative `-I`; it is
    then relative to the directory the program runs from, as with `pixi run`."""
    var here = String()
    here.write(source_location().file_name())
    var marker = "/mojosplat/train/"
    var at = here.find(marker)
    return String(here[byte=:at]) if at >= 0 else String(".")


@fieldwise_init
struct GitState(Copyable, Movable):
    var commit: String  # "unknown" outside a git checkout
    var dirty: Bool  # tracked files differ from the commit


def git_state(root: String) -> GitState:
    """The checked-out commit of `root` and whether tracked files have uncommitted changes."""
    var commit = _git(root, "rev-parse HEAD")
    if commit.byte_length() == 0:
        return GitState("unknown", False)
    return GitState(commit, _git(root, "status --porcelain --untracked-files=no").byte_length() > 0)


def _quoted(text: String) -> String:
    return String('"', text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n"), '"')


def _json_value(kind: String, value: String) -> String:
    if kind == "BOOL":
        return "true" if value == "on" else "false"
    if kind == "INT" or kind == "FLOAT":
        return value
    return _quoted(value)


def write_run_record[
    T: AnyType
](
    path: String, method: String, scene: String, gpu: String, ref config: T, metrics: List[Tuple[String, Float64]]
) raises:
    """Write `run.json`: method, scene, MojoSplat's commit (and whether it had
    uncommitted changes), GPU, the resolved settings and the results."""
    var git = git_state(mojosplat_root())
    var out = String("{\n")
    out += String('  "method": ', _quoted(method), ",\n")
    out += String('  "scene": ', _quoted(scene), ",\n")
    out += String('  "commit": ', _quoted(git.commit), ",\n")
    out += String('  "dirty": ', "true" if git.dirty else "false", ",\n")
    out += String('  "gpu": ', _quoted(gpu), ",\n")
    out += '  "config": {\n'
    var items = config_items[T](config)
    for i in range(len(items)):
        out += String("    ", _quoted(items[i][0]), ": ", _json_value(items[i][1], items[i][2]))
        out += ",\n" if i + 1 < len(items) else "\n"
    out += "  },\n"
    out += '  "metrics": {\n'
    for i in range(len(metrics)):
        out += String("    ", _quoted(metrics[i][0]), ": ", metrics[i][1])
        out += ",\n" if i + 1 < len(metrics) else "\n"
    out += "  }\n}\n"
    with open(path, "w") as f:
        f.write(out)
