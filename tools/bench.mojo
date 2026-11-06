"""Benchmark the Rasterizer on a .splat or .ply scene.

Usage:
    pixi run bench [scene.splat] [-n 200] [-W 1280] [-H 720]

Reports full frames without per-stage syncs (median / mean / min), then a
per-stage breakdown (one sync after every stage, so stages add up to a bit
more than a frame). Camera matches the original torch benchmark: identity
rotation, t = (0, 0, 5), fx = fy = 0.9 * W.
"""

from max.gpu.host import DeviceContext
from std.time import perf_counter_ns

from mojosplat import Camera, Rasterizer, Splats, load_splat, read_ply
from mojosplat.cli import Arg, Table, bold, dim, done, fmt_fixed, fmt_thousands, parse_args


struct BenchArgs(Defaultable, Movable):
    @__annotation(Arg("Scene to render (.splat or .ply)", positional=True))
    var scene: String

    @__annotation(Arg("Timed frames", short="n"))
    var runs: Int

    @__annotation(Arg("Image width", short="W"))
    var width: Int

    @__annotation(Arg("Image height", short="H"))
    var height: Int

    def __init__(out self):
        self.scene = "examples/bicycle.splat"
        self.runs = 200
        self.width = 1280
        self.height = 720


def ms(x: Float64) -> String:
    return String(fmt_fixed(x, 3), " ms")


def main() raises:
    var args = parse_args[BenchArgs]("pixi run bench", "Benchmark the rasterizer on one scene.")
    var W = args.width
    var H = args.height
    var runs = max(args.runs, 1)
    var ctx = DeviceContext()
    print(bold("MojoSplat"), dim("·"), "bench", dim("·"), ctx.name())
    var splats = Splats.from_host(ctx, read_ply(args.scene)) if args.scene.endswith(".ply") else Splats.from_scene(
        ctx, load_splat(args.scene)
    )
    done(String("Loaded ", args.scene, ": ", fmt_thousands(splats.count), " Gaussians"))
    var camera = Camera.translated(0.0, 0.0, 5.0, W, H, Float32(W) * 0.9, Float32(W) * 0.9)
    var image = ctx.enqueue_create_buffer[DType.float32](W * H * 3)
    var bg = SIMD[DType.float32, 4](0.0)

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
    sort(times)
    var mean: Float64 = 0
    for t in times:
        mean += t
    mean /= Float64(len(times))
    var median = times[len(times) // 2]

    rast.profile = True
    var acc = SIMD[DType.float64, 8](0)
    for _ in range(runs):
        rast.render(splats, camera, bg, image)
        acc += rast.stage_ms
    acc /= Float64(runs)

    print()
    var t = Table(String("Full frame, ", W, "x", H, ", ", runs, " runs"))
    t.add("Median", String(bold(ms(median)), dim(String("  (", fmt_fixed(1000.0 / median, 0), " FPS)"))))
    t.add("Mean", ms(mean))
    t.add("Min", ms(times[0]))
    t.add("Visible", fmt_thousands(rast.num_visible))
    t.add("Tile instances", fmt_thousands(rast.num_instances))
    t.print()
    print()
    var s = Table("Stages (synced after each)")
    s.add("Preprocess", ms(acc[0]))
    s.add("Depth sort", ms(acc[1]))
    s.add("Instances", ms(acc[2]))
    s.add("Tile sort", ms(acc[3]))
    s.add("Blend", ms(acc[4]))
    s.print()
