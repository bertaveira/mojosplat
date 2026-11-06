"""Train a 3D Gaussian Splatting model on a COLMAP scene, in pure Mojo.

Usage:
    pixi run train data/mipnerf360/garden [--iterations 30000] [--no-densify] ...
    pixi run train --help        # every option (all of TrainConfig)

Loads <scene>/sparse/0 and <scene>/<images> (every 8th image held out),
initializes Gaussians from the SfM points, trains with the 3DGS / Faster-GS
recipe, then reports the test PSNR and writes <out>/point_cloud.ply.
"""

from max.gpu.host import DeviceContext
from std.os import makedirs
from std.os.path import exists
from std.time import perf_counter_ns

from mojosplat.cli import (
    Arg, ProgressBar, Table, bold, dim, done, fmt_count, fmt_duration, fmt_fixed, fmt_thousands,
    parse_args, seconds_since,
)
from mojosplat.train.dataset import load_colmap_dataset
from mojosplat.train.trainer import TrainConfig, Trainer


struct TrainArgs(Defaultable, Movable):
    @__annotation(Arg("COLMAP scene directory (with sparse/0 and the image folder)", positional=True))
    var scene: String
    @__annotation(Arg("Image folder inside the scene"))
    var images: String
    @__annotation(Arg("Output directory (default: output/<scene name>)", short="o"))
    var out: String
    @__annotation(Arg("Hold out every N-th image for testing (0 = train on all)"))
    var test_every: Int
    @__annotation(Arg("Training"))
    var train: TrainConfig

    def __init__(out self):
        self.scene = ""
        self.images = "images_4"
        self.out = ""
        self.test_every = 8
        self.train = TrainConfig()


def main() raises:
    var args = parse_args[TrainArgs](
        "pixi run train", "Train a 3D Gaussian Splatting model on a COLMAP scene."
    )
    var parts = args.scene.rstrip("/").split("/")
    var name = String(parts[len(parts) - 1])
    var out = args.out if args.out.byte_length() > 0 else String("output/", name)
    var config = args.train.copy()

    var ctx = DeviceContext()
    print(bold("MojoSplat"), dim("·"), "train", dim("·"), ctx.name())
    var t0 = Int(perf_counter_ns())
    var ds = load_colmap_dataset(ctx, args.scene, args.images, args.test_every)
    done(String(
        "Loaded ", bold(name), ": ", len(ds.train), " train / ", len(ds.test), " test views, ",
        fmt_thousands(ds.points.count()), " SfM points ", dim(String("(", fmt_duration(seconds_since(t0)), ")")),
    ))
    var trainer = Trainer(ctx, ds^, config)
    done(String("Initialized ", fmt_thousands(trainer.splats.count), " Gaussians"))

    var bar = ProgressBar(config.iterations, "train")
    var info = String()
    for i in range(config.iterations):
        var loss = trainer.step()
        if loss >= 0.0:
            info = String(
                "loss ", fmt_fixed(loss, 4), dim(" | "), fmt_count(trainer.splats.count), " Gaussians",
                dim(" | "), "SH ", trainer.active_sh_degree(),
            )
        bar.update(i + 1, info, log=loss >= 0.0)
    var removed = trainer.finish()
    ctx.synchronize()
    var train_s = bar.elapsed()
    bar.finish(info)
    done(String(
        "Trained ", fmt_thousands(config.iterations), " iterations in ", fmt_duration(train_s),
        " ", dim(String("(", fmt_fixed(Float64(config.iterations) / train_s, 0), " it/s)")),
    ))
    if removed > 0:
        done(String("Pruned ", fmt_thousands(removed), " transparent Gaussians"))

    var psnr = trainer.evaluate(test=True) if args.test_every > 0 else -1.0
    if not exists(out):
        makedirs(out)
    var ply = out + "/point_cloud.ply"
    trainer.save_ply(ply)
    done(String("Saved ", ply))

    print()
    var table = Table("Results")
    table.add("Scene", name)
    table.add("Training time", fmt_duration(train_s))
    table.add("Gaussians", fmt_thousands(trainer.splats.count))
    if psnr >= 0.0:
        table.add("Test PSNR", String(bold(fmt_fixed(psnr, 2)), " dB"))
    table.add("Model", ply)
    table.print()
