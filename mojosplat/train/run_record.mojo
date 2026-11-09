"""`run.json`: everything needed to tell what produced a result."""

from std.os.path import exists

from ..cli.args import config_items


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def git_commit() -> String:
    """The checked-out commit, or "unknown" outside a git repository."""
    try:
        var head = String(_read(".git/HEAD").strip())
        if not head.startswith("ref: "):
            return head
        var ref_path = String(".git/", head[byte=5:])
        if exists(ref_path):
            return String(_read(ref_path).strip())
        return "unknown"
    except:
        return "unknown"


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
    """Write `run.json`: method, scene, commit, GPU, the resolved settings and the results."""
    var out = String("{\n")
    out += String('  "method": ', _quoted(method), ",\n")
    out += String('  "scene": ', _quoted(scene), ",\n")
    out += String('  "commit": ', _quoted(git_commit()), ",\n")
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
