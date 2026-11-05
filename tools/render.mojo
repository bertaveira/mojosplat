"""Render a .splat file to an image, entirely in Mojo (no Python).

Usage:
    pixi run render [scene.splat] [-o out.png|out.ppm] [-W 1280] [-H 720]
                    [--cam X Y Z] [--fx F] [--raw out.f32]

The camera has identity rotation (looking down +z) and world-to-camera
translation (X, Y, Z); default (0, 0, 5) with fx = fy = 0.9 * W, matching
the benchmark camera of the original torch examples.
"""

from max.gpu.host import DeviceContext
from std.sys import argv
from std.time import perf_counter_ns

from mojosplat import (
    Camera, Rasterizer, SplatBuffers, download, load_splat, write_array, write_png, write_ppm,
)


def main() raises:
    var args = argv()
    var path = String("examples/bicycle.splat")
    var out = String("render.png")
    var raw = String("")
    var W = 1280
    var H = 720
    var cam_t = SIMD[DType.float32, 4](0.0, 0.0, 5.0, 0.0)
    var fx: Float32 = -1.0
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "-o":
            out = String(args[i + 1])
            i += 2
        elif a == "--raw":
            raw = String(args[i + 1])
            i += 2
        elif a == "-W":
            W = Int(String(args[i + 1]))
            i += 2
        elif a == "-H":
            H = Int(String(args[i + 1]))
            i += 2
        elif a == "--fx":
            fx = Float32(Float64(String(args[i + 1])))
            i += 2
        elif a == "--cam":
            for k in range(3):
                cam_t[k] = Float32(Float64(String(args[i + 1 + k])))
            i += 4
        else:
            path = a
            i += 1
    if fx <= 0:
        fx = Float32(W) * 0.9

    print("Loading", path, "...")
    var t0 = perf_counter_ns()
    var scene = load_splat(path)
    print("  ", scene.count, "Gaussians in", Float64(perf_counter_ns() - t0) / 1e6, "ms")

    var ctx = DeviceContext()
    print("Device:", ctx.name())
    var splats = SplatBuffers.from_scene(ctx, scene)
    var camera = Camera.translated(cam_t[0], cam_t[1], cam_t[2], W, H, fx, fx)
    var image = ctx.enqueue_create_buffer[DType.float32](W * H * 3)
    var rasterizer = Rasterizer(ctx)

    rasterizer.render(splats, camera, SIMD[DType.float32, 4](0.0), image)  # warm-up / JIT
    ctx.synchronize()
    var t1 = perf_counter_ns()
    rasterizer.render(splats, camera, SIMD[DType.float32, 4](0.0), image)
    ctx.synchronize()
    print(
        "Rendered", W, "x", H, "in", Float64(perf_counter_ns() - t1) / 1e6, "ms:",
        rasterizer.num_visible, "visible Gaussians,", rasterizer.num_instances, "tile instances",
    )

    var pixels = download(ctx, image, W * H * 3)
    if out.endswith(".ppm"):
        write_ppm(out, pixels, W, H)
    else:
        write_png(out, pixels, W, H)
    print("Saved", out)
    if raw.byte_length() > 0:
        write_array(raw, pixels)
        print("Saved raw float32 image", raw)
