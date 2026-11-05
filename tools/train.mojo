"""Train a 3D Gaussian Splatting model on a COLMAP scene, in pure Mojo.

Usage:
    pixi run train data/mipnerf360/garden [--iters 30000] [--images images_4]
                   [--out output/garden] [--log 100] [--sh-degree 3]

Loads <scene>/sparse/0 and <scene>/<images> (every 8th image held out),
initializes Gaussians from the SfM points, trains with the 3DGS / Faster-GS
recipe (with classic densification; --no-densify disables it), then reports
the test PSNR and writes <out>/point_cloud.ply.
"""

from max.gpu.host import DeviceContext
from std.os import makedirs
from std.os.path import exists
from std.sys import argv
from std.time import perf_counter_ns

from mojosplat.train.dataset import load_colmap_dataset
from mojosplat.train.trainer import TrainConfig, Trainer


def main() raises:
    var args = argv()
    var scene = String("data/mipnerf360/garden")
    var images = String("images_4")
    var out = String("")
    var config = TrainConfig()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--iters":
            config.iterations = Int(String(args[i + 1]))
            i += 2
        elif a == "--images":
            images = String(args[i + 1])
            i += 2
        elif a == "--out":
            out = String(args[i + 1])
            i += 2
        elif a == "--log":
            config.log_interval = Int(String(args[i + 1]))
            i += 2
        elif a == "--no-densify":
            config.densify = False
            i += 1
        elif a == "--sh-degree":
            config.sh_degree = Int(String(args[i + 1]))
            i += 2
        else:
            scene = a
            i += 1
    if out.byte_length() == 0:
        var parts = scene.rstrip("/").split("/")
        out = String("output/", parts[len(parts) - 1])

    var ctx = DeviceContext()
    print("Device:", ctx.name())
    var t0 = perf_counter_ns()
    var ds = load_colmap_dataset(ctx, scene, images)
    print(
        "Loaded", scene, ":", len(ds.train), "train /", len(ds.test), "test views,",
        ds.points.count(), "points, extent", ds.scene_extent, "in",
        Float64(perf_counter_ns() - t0) / 1e9, "s",
    )
    var trainer = Trainer(ctx, ds^, config)
    print("Initialized", trainer.splats.count, "Gaussians")

    var t_start = perf_counter_ns()
    var t_last = t_start
    for _ in range(config.iterations):
        var loss = trainer.step()
        if loss >= 0.0:
            var now = perf_counter_ns()
            var it_s = Float64(config.log_interval) / (Float64(now - t_last) / 1e9)
            t_last = now
            print(
                "iter", trainer.iteration, "| loss", loss, "|", trainer.splats.count, "Gaussians | sh",
                trainer.active_sh_degree(), "|", it_s, "it/s | elapsed", Float64(now - t_start) / 1e9, "s",
            )
    var removed = trainer.finish()
    ctx.synchronize()
    var train_s = Float64(perf_counter_ns() - t_start) / 1e9
    print("Training time:", train_s, "s for", config.iterations, "iterations")
    print("Final model:", trainer.splats.count, "Gaussians (", removed, "transparent ones removed )")
    print("Test PSNR:", trainer.evaluate(test=True), "dB")
    if not exists(out):
        makedirs(out)
    trainer.save_ply(out + "/point_cloud.ply")
    print("Saved", out + "/point_cloud.ply")
