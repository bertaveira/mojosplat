"""Render a .splat or 3DGS .ply file to an image, entirely in Mojo (no Python).

Usage:
    pixi run render [scene.splat|scene.ply] [-o out.png|out.ppm] [-W 1280] [-H 720]
                    [--cam X,Y,Z] [--fx F] [--raw out.f32]

The camera has identity rotation (looking down +z) and world-to-camera
translation (X, Y, Z); default (0, 0, 5) with fx = fy = 0.9 * W, matching
the benchmark camera of the original torch examples.
"""

from max.gpu.host import DeviceContext
from std.time import perf_counter_ns

from mojosplat import (
    Camera,
    Rasterizer,
    Splats,
    download,
    load_splat,
    read_ply,
    write_array,
    write_png,
    write_ppm,
)
from mojosplat.cli import Arg, bold, dim, done, fmt_fixed, fmt_thousands, parse_args


struct RenderArgs(Defaultable, Movable):
    @__annotation(Arg("Scene to render (.splat or .ply)", positional=True))
    var scene: String

    @__annotation(Arg("Output image (.png or .ppm)", short="o"))
    var out: String

    @__annotation(Arg("Image width", short="W"))
    var width: Int

    @__annotation(Arg("Image height", short="H"))
    var height: Int

    @__annotation(Arg("World-to-camera translation (identity rotation)"))
    var cam: SIMD[DType.float32, 4]

    @__annotation(Arg("Focal length in pixels (default: 0.9 * width)"))
    var fx: Float32

    @__annotation(Arg("Also write the float32 [H, W, 3] image to this file"))
    var raw: String

    def __init__(out self):
        self.scene = "examples/bicycle.splat"
        self.out = "render.png"
        self.width = 1280
        self.height = 720
        self.cam = SIMD[DType.float32, 4](0.0, 0.0, 5.0, 0.0)
        self.fx = 0.0
        self.raw = ""


def main() raises:
    var args = parse_args[RenderArgs]("pixi run render", "Render a .splat or 3DGS .ply scene to an image.")
    var W = args.width
    var H = args.height
    var fx = args.fx if args.fx > 0 else Float32(W) * 0.9

    var ctx = DeviceContext()
    print(bold("MojoSplat"), dim("·"), "render", dim("·"), ctx.name())
    var t0 = perf_counter_ns()
    var splats: Splats
    if args.scene.endswith(".ply"):
        splats = Splats.from_host(ctx, read_ply(args.scene))
    else:
        splats = Splats.from_scene(ctx, load_splat(args.scene))
    done(
        String(
            "Loaded ",
            args.scene,
            ": ",
            fmt_thousands(splats.count),
            " Gaussians ",
            dim(String("(", fmt_fixed(Float64(perf_counter_ns() - t0) / 1e6, 0), " ms)")),
        )
    )
    var c = args.cam
    var camera = Camera.translated(c[0], c[1], c[2], W, H, fx, fx)
    var image = ctx.enqueue_create_buffer[DType.float32](W * H * 3)
    var rasterizer = Rasterizer(ctx)

    rasterizer.render(splats, camera, SIMD[DType.float32, 4](0.0), image)  # warm-up / JIT
    ctx.synchronize()
    var t1 = perf_counter_ns()
    rasterizer.render(splats, camera, SIMD[DType.float32, 4](0.0), image)
    ctx.synchronize()
    done(
        String(
            "Rendered ",
            W,
            "x",
            H,
            " in ",
            bold(String(fmt_fixed(Float64(perf_counter_ns() - t1) / 1e6, 2), " ms")),
            dim(
                String(
                    " (",
                    fmt_thousands(rasterizer.num_visible),
                    " visible Gaussians, ",
                    fmt_thousands(rasterizer.num_instances),
                    " tile instances)",
                )
            ),
        )
    )

    var pixels = download(ctx, image, W * H * 3)
    if args.out.endswith(".ppm"):
        write_ppm(args.out, pixels, W, H)
    else:
        write_png(args.out, pixels, W, H)
    done(String("Saved ", args.out))
    if args.raw.byte_length() > 0:
        write_array[DType.float32](args.raw, pixels)
        done(String("Saved ", args.raw))
