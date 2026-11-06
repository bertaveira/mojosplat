"""Command-line options generated from a config struct (in the spirit of tyro).

Every field of the struct becomes an option, found by compile-time
reflection, so adding a field to a config adds a flag:

    struct TrainArgs(Defaultable, Movable):
        @__annotation(Arg("COLMAP scene directory", positional=True))
        var scene: String
        @__annotation(Arg("Image folder inside the scene"))
        var images: String
        @__annotation(Arg("Training"))
        var train: TrainConfig          # nested: its fields join the options

        def __init__(out self):         # defaults
            ...

    var args = parse_args[TrainArgs]("train", "Train a model.")

Field `sh_degree` is `--sh-degree`; Bool fields are switches (`--densify` /
`--no-densify`); `--name=value` works too. Supported field types: Int,
Float32, Float64, Bool, String and SIMD[float32, 4] (`--background 1,1,1`).
A field of any other struct type is a group: its fields are options too,
listed under the group's help text in `--help`.

`Arg` annotations (`@__annotation(...)`, Mojo's field metadata) add help
text, a short flag (`short="o"` -> `-o`) or make the field positional. A
positional field whose default is empty is required.

On `--help` the help is printed and the program exits; on a usage error
the error is printed (with the closest flag, for a misspelled one) and the
program exits with status 2.
"""

from std.sys import argv, exit, stderr

from .term import bold, cyan, dim, error, spaces


struct Arg(ImplicitlyCopyable, Movable):
    """Field metadata for `parse_args`: `@__annotation(Arg("help"))`."""

    var help: StaticString
    var short: StaticString
    var positional: Bool

    def __init__(out self, help: StaticString, *, short: StaticString = "", positional: Bool = False):
        self.help = help
        self.short = short
        self.positional = positional


@fieldwise_init
struct _Option(Copyable, Movable):
    var name: String  # without dashes: "sh-degree"
    var short: String
    var kind: String  # INT, FLOAT, BOOL, STR, LIST (or GROUP for a header)
    var help: String
    var default: String
    var positional: Bool


comptime _FLOAT4 = SIMD[DType.float32, 4]


def _kind[FT: AnyType]() -> StaticString:
    """The option kind of a field type, or "" for a nested struct."""
    comptime name = reflect[FT].name()
    comptime if name == reflect[Int].name():
        return "INT"
    elif name == reflect[Float32].name() or name == reflect[Float64].name():
        return "FLOAT"
    elif name == reflect[Bool].name():
        return "BOOL"
    elif name == reflect[String].name():
        return "STR"
    elif name == reflect[_FLOAT4].name():
        return "LIST"
    else:
        return ""


def _arg_of[T: AnyType, i: Int]() -> Arg:
    """The `Arg` annotation of field `i` of `T` (empty if there is none)."""
    var result = Arg("")
    var anns = reflect[T].field_annotations[i]()
    comptime for k in range(type_of(anns).__len__()):
        comptime AT = type_of(anns).Ts[k]
        comptime if reflect[AT].name() == reflect[Arg].name():
            result = rebind[Arg](anns[k])
    return result


def _flag_name(field: StaticString) -> String:
    return String(field).replace("_", "-")


def _compact(x: Float32) -> String:
    """1.0 -> "1", 0.5 -> "0.5"."""
    return String(Int(x)) if Float32(Int(x)) == x else String(x)


def _get[FT: AnyType](ref value: FT) -> String:
    """A leaf field's value as text (for `--help` defaults)."""
    var p = Pointer(to=value)
    comptime k = _kind[FT]()
    comptime if reflect[FT].name() == reflect[Int].name():
        return String(p.unsafe_bitcast[Int]()[])
    elif reflect[FT].name() == reflect[Float32].name():
        return String(p.unsafe_bitcast[Float32]()[])
    elif reflect[FT].name() == reflect[Float64].name():
        return String(p.unsafe_bitcast[Float64]()[])
    elif k == "BOOL":
        return "on" if p.unsafe_bitcast[Bool]()[] else "off"
    elif k == "STR":
        return p.unsafe_bitcast[String]()[]
    elif k == "LIST":
        var v = p.unsafe_bitcast[_FLOAT4]()[]
        return String(_compact(v[0]), ",", _compact(v[1]), ",", _compact(v[2]))
    else:
        return ""


def _parse_float(flag: String, text: String) raises -> Float64:
    try:
        return Float64(text)
    except:
        raise Error(String("invalid value '", text, "' for ", flag, ": expected a number"))


def _set[FT: AnyType](mut value: FT, flag: String, text: String) raises:
    """Parse `text` into a leaf field."""
    var p = Pointer(to=value)
    comptime k = _kind[FT]()
    comptime if reflect[FT].name() == reflect[Int].name():
        try:
            p.unsafe_bitcast[Int]()[] = Int(text)
        except:
            raise Error(String("invalid value '", text, "' for ", flag, ": expected an integer"))
    elif reflect[FT].name() == reflect[Float32].name():
        p.unsafe_bitcast[Float32]()[] = Float32(_parse_float(flag, text))
    elif reflect[FT].name() == reflect[Float64].name():
        p.unsafe_bitcast[Float64]()[] = _parse_float(flag, text)
    elif k == "BOOL":
        var t = text.lower()
        if t == "true" or t == "1" or t == "yes" or t == "on":
            p.unsafe_bitcast[Bool]()[] = True
        elif t == "false" or t == "0" or t == "no" or t == "off":
            p.unsafe_bitcast[Bool]()[] = False
        else:
            raise Error(String("invalid value '", text, "' for ", flag, ": expected true or false"))
    elif k == "STR":
        p.unsafe_bitcast[String]()[] = text
    elif k == "LIST":
        var parts = text.split(",")
        if len(parts) < 1 or len(parts) > 4:
            raise Error(String("invalid value '", text, "' for ", flag, ": expected up to 4 numbers, e.g. 1,1,1"))
        var v = _FLOAT4(0.0)
        for j in range(len(parts)):
            v[j] = Float32(_parse_float(flag, String(parts[j])))
        p.unsafe_bitcast[_FLOAT4]()[] = v


def _collect[T: AnyType](ref cfg: T, mut opts: List[_Option]) raises:
    """Append the options of every field (recursing into groups)."""
    comptime names = reflect[T].field_names()
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_at[i].T
        comptime k = _kind[FT]()
        var a = _arg_of[T, i]()
        var name = _flag_name(materialize[names[i]]())
        comptime if k == "":
            comptime assert reflect[FT].is_struct(), "parse_args: unsupported field type"
            opts.append(_Option(name, "", "GROUP", String(a.help), "", False))
            _collect[FT](reflect[T].field_ref[i](cfg), opts)
        else:
            var default = _get[FT](reflect[T].field_ref[i](cfg))
            opts.append(_Option(name, String(a.short), String(k), String(a.help), default, a.positional))


def _assign[T: AnyType](mut cfg: T, name: String, flag: String, text: String) raises -> Bool:
    """Set the field whose option is `name`; False if there is none."""
    comptime names = reflect[T].field_names()
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_at[i].T
        comptime if _kind[FT]() == "":
            if _assign[FT](reflect[T].field_ref[i](cfg), name, flag, text):
                return True
        else:
            if _flag_name(materialize[names[i]]()) == name:
                _set[FT](reflect[T].field_ref[i](cfg), flag, text)
                return True
    return False


def _edit_distance(a: String, b: String) -> Int:
    var n = b.byte_length()
    var prev = List[Int](capacity=n + 1)
    for j in range(n + 1):
        prev.append(j)
    var ab = a.as_bytes()
    var bb = b.as_bytes()
    for i in range(1, a.byte_length() + 1):
        var cur = List[Int](capacity=n + 1)
        cur.append(i)
        for j in range(1, n + 1):
            var cost = 0 if ab[i - 1] == bb[j - 1] else 1
            cur.append(min(min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + cost))
        prev = cur^
    return prev[n]


def _metavar(o: _Option) -> String:
    if o.kind == "LIST":
        return "R,G,B" if o.name.find("background") >= 0 else "X,Y,Z"
    return o.kind


def _usage(prog: String, opts: List[_Option]) -> String:
    var s = String("usage: ", prog)
    for o in opts:
        if o.positional:
            s += String(" <", o.name, ">") if o.default.byte_length() == 0 else String(" [", o.name, "]")
    return s + " [options]"


def _print_help(prog: String, description: String, opts: List[_Option]):
    var left = List[String]()
    var w = 0
    for o in opts:
        var l: String
        if o.kind == "GROUP":
            l = ""
        elif o.positional:
            l = o.name
        elif o.kind == "BOOL":
            l = String("--", o.name, " / --no-", o.name)
        else:
            var short = String("-", o.short, ", ") if o.short.byte_length() > 0 else String()
            l = String(short, "--", o.name, " ", _metavar(o))
        left.append(l)
        w = max(w, l.byte_length())
    w = min(w, 40)

    if description.byte_length() > 0:
        print(description)
        print()
    print(bold(_usage(prog, opts)))

    var has_pos = False
    for o in opts:
        has_pos = has_pos or o.positional
    if has_pos:
        print()
        print(bold("Arguments:"))
        for i in range(len(opts)):
            if opts[i].positional:
                print("  ", cyan(left[i]), spaces(w - left[i].byte_length() + 2), opts[i].help, sep="")

    print()
    print(bold("Options:"))
    var hl = String("-h, --help")
    print("  ", cyan(hl), spaces(w - hl.byte_length() + 2), "Show this message and exit", sep="")
    for i in range(len(opts)):
        ref o = opts[i]
        if o.positional:
            continue
        if o.kind == "GROUP":
            print()
            print(bold(String(o.help if o.help.byte_length() > 0 else o.name, ":")))
            continue
        var pad = spaces(w - left[i].byte_length() + 2) if left[i].byte_length() <= w else String("\n  ", spaces(w + 2))
        var default = String()
        if o.default.byte_length() > 0:
            default = dim(String(" (default: ", o.default, ")"))
        print("  ", cyan(left[i]), pad, o.help, default, sep="")


def _find(opts: List[_Option], name: String) -> Int:
    for i in range(len(opts)):
        if opts[i].kind != "GROUP" and not opts[i].positional and opts[i].name == name:
            return i
    return -1


def _usage_error(prog: String, msg: String):
    error(msg)
    print(dim(String("Run '", prog, " --help' for the list of options.")), file=stderr)
    exit(2)


def _looks_numeric(s: String) -> Bool:
    if s.byte_length() < 2:
        return False
    var c = Int(s.as_bytes()[1])
    return (c >= ord("0") and c <= ord("9")) or c == ord(".")


def parse_args[T: Defaultable & Movable & Deinitable](prog: String, description: String = "") -> T:
    """Parse the command line into a `T` (starting from `T()`'s defaults)."""
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    try:
        return parse_args_from[T](args, prog, description)
    except e:
        _usage_error(prog, String(e))
    return T()  # unreachable


def parse_args_from[T: Defaultable & Movable & Deinitable](args: List[String], prog: String, description: String = "") raises -> T:
    """Like `parse_args` for an explicit argument list (no program name);
    raises on usage errors instead of exiting (prints help and exits on
    `--help`)."""
    var cfg = T()
    var opts = List[_Option]()
    _collect[T](cfg, opts)
    for i in range(len(opts)):
        for j in range(i):
            if opts[i].kind != "GROUP" and opts[j].kind != "GROUP" and opts[i].name == opts[j].name:
                raise Error(String("parse_args: two fields named '", opts[i].name, "'"))

    var positional = List[Int]()
    for i in range(len(opts)):
        if opts[i].positional:
            positional.append(i)
    var n_pos = 0

    var i = 0
    while i < len(args):
        var a = args[i]
        i += 1
        if a == "-h" or a == "--help":
            _print_help(prog, description, opts)
            exit(0)
        if a.startswith("-") and not _looks_numeric(a) and a != "-":
            var name: String
            var inline = String()
            var has_inline = False
            if a.startswith("--"):
                name = String(a[byte=2:])
                var eq = name.find("=")
                if eq >= 0:
                    var key = String(name[byte=:eq])
                    inline = String(name[byte = eq + 1 :])
                    name = key
                    has_inline = True
            else:
                var short = String(a[byte=1:])
                name = ""
                for o in opts:
                    if o.short == short and o.kind != "GROUP":
                        name = o.name
                if name.byte_length() == 0:
                    raise Error(String("unknown option '", a, "'"))
            var k = _find(opts, name)
            var value = inline
            if k < 0 and name.startswith("no-"):
                var base = String(name[byte=3:])
                var kb = _find(opts, base)
                if kb >= 0 and opts[kb].kind == "BOOL" and not has_inline:
                    k = kb
                    name = base
                    value = "false"
                    has_inline = True
            if k < 0:
                var best = String()
                var best_d = 4
                for o in opts:
                    if o.kind == "GROUP" or o.positional:
                        continue
                    var d = _edit_distance(name, o.name)
                    if d < best_d:
                        best_d = d
                        best = o.name
                var msg = String("unknown option '--", name, "'")
                if best.byte_length() > 0:
                    msg += String(" (did you mean '--", best, "'?)")
                raise Error(msg)
            var flag = String("--", name)
            if opts[k].kind == "BOOL":
                if not has_inline:
                    value = "true"
            elif not has_inline:
                if i >= len(args):
                    raise Error(String("missing value for ", flag, " (", _metavar(opts[k]), ")"))
                value = args[i]
                i += 1
            _ = _assign[T](cfg, name, flag, value)
        else:
            if n_pos >= len(positional):
                raise Error(String("unexpected argument '", a, "'"))
            ref o = opts[positional[n_pos]]
            _ = _assign[T](cfg, o.name, o.name, a)
            n_pos += 1

    for j in range(n_pos, len(positional)):
        ref o = opts[positional[j]]
        if o.default.byte_length() == 0:
            raise Error(String("missing argument <", o.name, ">\n", _usage(prog, opts)))
    return cfg^
