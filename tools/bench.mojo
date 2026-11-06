"""Benchmark the Rasterizer on a .splat scene.

Usage:
    pixi run bench [scene.splat] [-n 200] [-W 1280] [-H 720]

Reports full frames without per-stage syncs (median / mean / min), then a
per-stage breakdown (one sync after every stage, so stages add up to a bit
more than a frame). Camera matches the original torch benchmark: identity
rotation, t = (0, 0, 5), fx = fy = 0.9 * W.
"""

from max.gpu.host import DeviceContext
from std.sys import argv
from std.time import perf_counter_ns

from mojosplat import Camera, Rasterizer, Splats, load_splat


def summarize(label: String, mut times: List[Float64]):
    sort(times)
    var mean: Float64 = 0
    for t in times:
        mean += t
    mean /= Float64(len(times))
    var median = times[len(times) // 2]
    print(
        label, ": median", median, "ms | mean", mean, "ms | min", times[0],
        "ms |", 1000.0 / median, "FPS",
    )


def _value(flag: StringSlice, i: Int) raises -> String:
    """The command-line argument after `flag` (raises if it is missing)."""
    var args = argv()
    if i >= len(args):
        raise Error(String("missing value after ", flag))
    return String(args[i])


def main() raises:
    var args = argv()
    var path = String("examples/bicycle.splat")
    var runs = 200
    var W = 1280
    var H = 720
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "-n":
            runs = Int(_value(args[i], i + 1))
            i += 2
        elif a == "-W":
            W = Int(_value(args[i], i + 1))
            i += 2
        elif a == "-H":
            H = Int(_value(args[i], i + 1))
            i += 2
        else:
            path = a
            i += 1

    var scene = load_splat(path)
    var ctx = DeviceContext()
    var camera = Camera.translated(0.0, 0.0, 5.0, W, H, Float32(W) * 0.9, Float32(W) * 0.9)
    var image = ctx.enqueue_create_buffer[DType.float32](W * H * 3)
    var bg = SIMD[DType.float32, 4](0.0)
    print("Scene:", path, "|", scene.count, "Gaussians |", W, "x", H, "|", ctx.name())

    var splats = Splats.from_scene(ctx, scene)
    var rast = Rasterizer(ctx)
    for _ in range(20):
        rast.render(splats, camera, bg, image)
    ctx.synchronize()
    var times = List[Float64](capacity=runs)
    for _ in range(runs):
        var t0 = perf_counter_ns()
        rast.render(splats, camera, bg, image)
        ctx.synchronize()
        times.append(Float64(perf_counter_ns() - t0) / 1e6)
    print(rast.num_visible, "visible Gaussians,", rast.num_instances, "tile instances")
    summarize("full frame", times)
    rast.profile = True
    var acc = SIMD[DType.float64, 8](0)
    for _ in range(runs):
        rast.render(splats, camera, bg, image)
        acc += rast.stage_ms
    acc /= Float64(runs)
    print("stages (ms): preprocess", acc[0], "| depth sort", acc[1], "| instances", acc[2],
          "| tile sort", acc[3], "| blend", acc[4])
