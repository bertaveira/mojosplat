"""Evaluate a trained 3DGS model on the held-out views of a COLMAP scene.

Usage:
    pixi run eval data/mipnerf360/garden output/garden/point_cloud.ply
                  [--images images_4] [--save out_dir] [--test-every 8]

Renders every test view (every 8th image, as in training) and reports the
mean PSNR twice: on the float render clamped to [0, 1], and on the render
quantized to 8 bits (what is saved to PNG; how NeRFICG / Faster-GS compute
their metrics). With --save, writes <out_dir>/renders/<name>.png and
<out_dir>/gt/<name>.png for external metrics (SSIM, LPIPS).
"""

from max.gpu.host import DeviceContext
from std.math import log10
from std.os import makedirs
from std.os.path import exists
from std.time import perf_counter_ns

from mojosplat import Rasterizer, Splats, download, read_ply, write_png
from mojosplat.cli import Arg, ProgressBar, Table, bold, dim, done, fmt_fixed, fmt_thousands, parse_args
from mojosplat.train.dataset import load_colmap_dataset


struct EvalArgs(Defaultable, Movable):
    @__annotation(Arg("COLMAP scene directory", positional=True))
    var scene: String
    @__annotation(Arg("Trained model (.ply)", positional=True))
    var model: String
    @__annotation(Arg("Image folder inside the scene"))
    var images: String
    @__annotation(Arg("Test views are every N-th image (as in training)"))
    var test_every: Int
    @__annotation(Arg("Write <dir>/renders/*.png and <dir>/gt/*.png"))
    var save: String

    def __init__(out self):
        self.scene = ""
        self.model = ""
        self.images = "images_4"
        self.test_every = 8
        self.save = ""


def main() raises:
    var args = parse_args[EvalArgs]("pixi run eval", "Evaluate a trained model on the held-out views of a scene.")
    var ctx = DeviceContext()
    print(bold("MojoSplat"), dim("·"), "eval", dim("·"), ctx.name())
    var ds = load_colmap_dataset(ctx, args.scene, args.images, args.test_every)
    if len(ds.test) == 0:
        raise Error("the scene has no test views (--test-every 0?)")
    var splats = Splats.from_host(ctx, read_ply(args.model))
    done(String("Loaded ", args.model, ": ", fmt_thousands(splats.count), " Gaussians, ", len(ds.test), " test views"))
    var save = args.save
    if save.byte_length() > 0:
        for d in [save + "/renders", save + "/gt"]:
            if not exists(d):
                makedirs(d)

    var r = Rasterizer(ctx)
    var max_px = 1
    for v in ds.test:
        max_px = max(max_px, v.camera.width * v.camera.height)
    var image = ctx.enqueue_create_buffer[DType.float32](max_px * 3)
    var bg = SIMD[DType.float32, 4](0.0)
    var psnr_f: Float64 = 0
    var psnr_8: Float64 = 0
    var ms: Float64 = 0
    var bar = ProgressBar(len(ds.test), "eval", unit="views")
    for k in range(len(ds.test)):
        ref v = ds.test[k]
        var W = v.camera.width
        var H = v.camera.height
        var t0 = perf_counter_ns()
        r.render(splats, v.camera, bg, image, splats.view().max_sh_degree())
        ctx.synchronize()
        ms += Float64(perf_counter_ns() - t0) / 1e6
        var px = download(ctx, image, W * H * 3)
        var gt = download(ctx, v.image, W * H * 3)
        var se_f: Float64 = 0
        var se_8: Float64 = 0
        for j in range(W * H * 3):
            var c = Float64(min(max(px[j], 0.0), 1.0))
            var g = Float64(gt[j]) / 255.0
            var q = Float64(Int(c * 255.0 + 0.5)) / 255.0
            se_f += (c - g) * (c - g)
            se_8 += (q - g) * (q - g)
        var n = Float64(W * H * 3)
        psnr_f += -10.0 * log10(se_f / n)
        psnr_8 += -10.0 * log10(se_8 / n)
        if save.byte_length() > 0:
            var name = v.name.split(".")[0]
            write_png(String(save, "/renders/", name, ".png"), px, W, H)
            var gtf = List[Float32](capacity=W * H * 3)
            for p in gt:
                gtf.append(Float32(p) / 255.0)
            write_png(String(save, "/gt/", name, ".png"), gtf, W, H)
        bar.update(k + 1, String("PSNR ", fmt_fixed(psnr_8 / Float64(k + 1), 2), " dB"))
    var n = Float64(len(ds.test))
    bar.finish(String("PSNR ", fmt_fixed(psnr_8 / n, 2), " dB"))
    if save.byte_length() > 0:
        done(String("Saved renders to ", save))

    print()
    var t = Table("Test views")
    t.add("PSNR (8-bit)", String(bold(fmt_fixed(psnr_8 / n, 2)), " dB"))
    t.add("PSNR (float)", String(fmt_fixed(psnr_f / n, 2), " dB"))
    t.add("Render time", String(fmt_fixed(ms / n, 2), " ms ", dim(String("(", ds.test[0].camera.width, "x", ds.test[0].camera.height, ")"))))
    t.print()
