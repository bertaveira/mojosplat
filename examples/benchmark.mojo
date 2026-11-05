"""Benchmark the pure-Mojo render pipeline on a .splat scene.

Usage:
    pixi run bench [scene.splat] [-n 200] [-W 1280] [-H 720]

Reports the full pipeline (no per-stage syncs) and a per-stage breakdown
(projection / binning / rasterization, with a sync after each stage).
Camera matches the original torch benchmark: identity rotation,
t = (0, 0, 5), fx = fy = 0.9 * W.
"""

from max.gpu.host import DeviceContext
from std.sys import argv
from std.time import perf_counter_ns

from mojosplat.camera import Camera
from mojosplat.io import load_splat
from mojosplat.render import GaussianBuffers, Renderer


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
            runs = Int(String(args[i + 1]))
            i += 2
        elif a == "-W":
            W = Int(String(args[i + 1]))
            i += 2
        elif a == "-H":
            H = Int(String(args[i + 1]))
            i += 2
        else:
            path = a
            i += 1

    var scene = load_splat(path)
    var ctx = DeviceContext()
    var gaussians = GaussianBuffers.from_scene(ctx, scene)
    var camera = Camera.translated(0.0, 0.0, 5.0, W, H, Float32(W) * 0.9, Float32(W) * 0.9)
    var image = ctx.enqueue_create_buffer[DType.float32](W * H * 3)
    var bg = SIMD[DType.float32, 4](0.0)
    var renderer = Renderer(ctx)

    print("Scene:", path, "|", scene.count, "Gaussians |", W, "x", H, "|", ctx.name())
    for _ in range(20):
        renderer.render(gaussians, camera, bg, image)
    ctx.synchronize()
    print("Tile intersections:", renderer.num_intersections)

    # Full pipeline
    var times = List[Float64](capacity=runs)
    for _ in range(runs):
        var t0 = perf_counter_ns()
        renderer.render(gaussians, camera, bg, image)
        ctx.synchronize()
        times.append(Float64(perf_counter_ns() - t0) / 1e6)
    sort(times)
    var mean: Float64 = 0
    for t in times:
        mean += t
    mean /= Float64(runs)
    print(
        "Full pipeline: mean", mean, "ms | median", times[runs // 2], "ms | min",
        times[0], "ms |", 1000.0 / mean, "FPS",
    )

    # Per-stage
    renderer.profile = True
    var acc = SIMD[DType.float64, 4](0)
    for _ in range(runs):
        renderer.render(gaussians, camera, bg, image)
        acc += renderer.stage_ms
    acc /= Float64(runs)
    print("Per stage (synced, mean ms): projection", acc[0], "| binning", acc[1],
          "| rasterization", acc[2], "| total", acc[3])
