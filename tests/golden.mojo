"""Helpers to load the golden reference data in tests/data/golden/."""

from mojosplat.core.camera import Camera
from mojosplat.io.arrays import read_array

comptime GOLDEN_DIR = "tests/data/golden/"


struct GoldenScene(Movable):
    var name: String
    var n: Int
    var m: Int
    var width: Int
    var height: Int
    var tile_size: Int
    var sh_degree: Int
    var sh_rest: Int
    var camera: Camera
    var background: List[Float32]

    def __init__(out self, name: String) raises:
        var meta: String
        with open(String(GOLDEN_DIR, name, "/meta.txt"), "r") as f:
            meta = f.read()
        var n = 0
        var m = 0
        var w = 0
        var h = 0
        var ts = 16
        var sh_degree = 0
        var sh_rest = 0
        for line in meta.splitlines():
            var kv = line.split("=")
            if len(kv) != 2:
                continue
            var key = String(kv[0])
            var val = Int(String(kv[1]))
            if key == "N":
                n = val
            elif key == "M":
                m = val
            elif key == "W":
                w = val
            elif key == "H":
                h = val
            elif key == "tile_size":
                ts = val
            elif key == "sh_degree":
                sh_degree = val
            elif key == "sh_rest":
                sh_rest = val
        self.name = name
        self.n = n
        self.m = m
        self.width = w
        self.height = h
        self.tile_size = ts
        self.sh_degree = sh_degree
        self.sh_rest = sh_rest
        var c = read_array[DType.float32](String(GOLDEN_DIR, name, "/camera.f32"))
        var view = SIMD[DType.float32, 16](0.0)
        for i in range(16):
            view[i] = c[i]
        self.camera = Camera(view, c[16], c[17], c[18], c[19], w, h, c[20], c[21])
        self.background = read_array[DType.float32](String(GOLDEN_DIR, name, "/background.f32"))

    def path(self, file: String) -> String:
        return String(GOLDEN_DIR, self.name, "/", file)

    def f32(self, file: String) raises -> List[Float32]:
        return read_array[DType.float32](self.path(file))

    def i32(self, file: String) raises -> List[Int32]:
        return read_array[DType.int32](self.path(file))


def read_kv(path: String) raises -> Dict[String, String]:
    """key=value lines (as written by the tests/data generators)."""
    var out = Dict[String, String]()
    var text: String
    with open(path, "r") as f:
        text = f.read()
    for line in text.splitlines():
        var i = line.find("=")
        if i > 0:
            out[String(line[byte=0:i])] = String(line[byte = i + 1 :])
    return out^


def floats(s: String) raises -> List[Float64]:
    var out = List[Float64]()
    for p in s.split(" "):
        if p.byte_length() > 0:
            out.append(Float64(String(p)))
    return out^
