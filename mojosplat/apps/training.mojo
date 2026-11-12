"""The `train` program for any method (`pixi run train` runs it for the method given with `--method`).

Loads <scene>/sparse/0 and <scene>/<images> (every 8th image held out),
initializes Gaussians from the SfM points, trains with the method's pipeline,
densification strategy and settings, then reports the test PSNR and writes
<out>/point_cloud.ply and <out>/run.json.
"""

from max.gpu.host import DeviceContext
from std.os import makedirs
from std.os.path import exists
from std.time import perf_counter_ns

from ..cli import (
    Arg,
    ProgressBar,
    Table,
    bold,
    dim,
    done,
    fmt_count,
    fmt_duration,
    fmt_fixed,
    fmt_thousands,
    parse_args,
    parse_args_over,
    seconds_since,
)
from ..core.camera import Camera
from ..methods import FasterGS, Method, MethodConfig
from ..render.pipelines import Standard3DGS
from ..train.dataset import load_colmap_dataset
from ..train.run_record import write_run_record
from ..train.trainer import Trainer
from ..viewer import Viewer, ViewerConfig


struct TrainArgs[C: MethodConfig](Defaultable, Movable):
    """The options of `train`: these, then the method's settings `C`, then the viewer's."""

    @__annotation(Arg("COLMAP scene directory (with sparse/0 and the image folder)", positional=True))
    var scene: String

    @__annotation(Arg("Reconstruction method (see mojosplat/methods)"))
    var method: String

    @__annotation(Arg("Output directory (default: output/<scene name>)", short="o"))
    var out: String

    @__annotation(Arg("Hold out every N-th image for testing (0 = train on all)"))
    var test_every: Int

    @__annotation(Arg("Serve the live web viewer while training (stays open afterwards)"))
    var viewer: Bool

    @__annotation(Arg("Method settings"))
    var settings: Self.C

    @__annotation(Arg("Viewer (with --viewer)"))
    var view: ViewerConfig

    def __init__(out self):
        self.scene = ""
        self.method = FasterGS.name()
        self.out = ""
        self.test_every = 8
        self.viewer = False
        self.settings = Self.C()
        self.view = ViewerConfig()


def _stats(it: Int, total: Int, loss: Float64, gaussians: Int, bar: ProgressBar) -> List[Tuple[String, String]]:
    var items = List[Tuple[String, String]]()
    var elapsed = bar.elapsed()
    items.append(("Iteration", String(fmt_thousands(it), " / ", fmt_thousands(total))))
    items.append(("Loss", fmt_fixed(loss, 4)))
    items.append(("Gaussians", fmt_thousands(gaussians)))
    items.append(("Speed", String(fmt_fixed(Float64(it) / max(elapsed, 1e-9), 0), " it/s")))
    items.append(("Elapsed", fmt_duration(elapsed)))
    return items^


comptime DESCRIPTION = "Train a 3D Gaussian Splatting model on a COLMAP scene."


def scene_name(path: String) raises -> String:
    var parts = path.rstrip("/").split("/")
    return String(parts[len(parts) - 1])


def train[M: Method](program: String = "pixi run train") raises:
    """The whole `train` program for method `M`: parse the command line, train,
    evaluate, and write `<out>/point_cloud.ply` and `<out>/run.json`.

    The method's settings for the scene are the starting point, and options on
    the command line override them. A repository with its own method only needs

        def main() raises:
            train[MyMethod]("pixi run train")
    """
    var first = parse_args[TrainArgs[M.Config]](program, DESCRIPTION)
    var base = TrainArgs[M.Config]()
    base.method = M.name()
    base.settings = M.config(scene_name(first.scene))
    var args = parse_args_over(base^, program, DESCRIPTION)
    var name = scene_name(args.scene)
    var out = args.out if args.out.byte_length() > 0 else String("output/", name)
    var config = args.settings.training()
    if args.viewer and M.Pipe.ID != Standard3DGS.ID:
        raise Error("--viewer only supports the standard pipeline for now")

    var ctx = DeviceContext()
    print(bold("MojoSplat"), dim("·"), "train", dim("·"), ctx.name())
    var t0 = Int(perf_counter_ns())
    var ds = load_colmap_dataset(ctx, args.scene, config.images, args.test_every, config.near, config.far)
    done(
        String(
            "Loaded ",
            bold(name),
            ": ",
            len(ds.train),
            " train / ",
            len(ds.test),
            " test views, ",
            fmt_thousands(ds.points.count()),
            " SfM points ",
            dim(String("(", fmt_duration(seconds_since(t0)), ")")),
        )
    )
    var trainer = Trainer[M.Pipe, M.Config.Strategy](ctx, ds^, config, args.settings.densification())
    done(String("Initialized ", fmt_thousands(trainer.splats.count), " Gaussians"))

    var viewer = Optional[Viewer]()
    if args.viewer:
        var cams = List[Camera]()
        for v in trainer.dataset.train:
            cams.append(v.camera)
        viewer = Viewer(ctx, args.view, trainer.splats, cams, name, True)

    var bar = ProgressBar(config.iterations, "train")
    var info = String()
    for i in range(config.iterations):
        var loss = trainer.step()
        if loss >= 0.0:
            info = String(
                "loss ",
                fmt_fixed(loss, 4),
                dim(" | "),
                fmt_count(trainer.splats.count),
                " Gaussians",
                dim(" | "),
                "SH ",
                trainer.active_sh_degree(),
            )
        bar.update(i + 1, info, log=loss >= 0.0)
        if viewer:
            ref v = viewer.value()
            if loss >= 0.0:
                v.set_stats(
                    _stats(i + 1, config.iterations, loss, trainer.splats.count, bar),
                    Float64(i + 1) / Float64(config.iterations),
                )
            v.update(trainer.splats, trainer.active_sh_degree(), scene_changed=True)
            if v.paused:
                var t_pause = Int(perf_counter_ns())
                bar.message(dim("Paused from the viewer"))
                v.set_stats(
                    _stats(i + 1, config.iterations, loss, trainer.splats.count, bar),
                    Float64(i + 1) / Float64(config.iterations),
                )
                v.wait_while_paused(trainer.splats, trainer.active_sh_degree())
                bar.exclude(seconds_since(t_pause))
                bar.message(dim("Resumed"))
    var removed = trainer.finish()
    ctx.synchronize()
    var train_s = bar.elapsed()
    bar.finish(info)
    done(
        String(
            "Trained ",
            fmt_thousands(config.iterations),
            " iterations in ",
            fmt_duration(train_s),
            " ",
            dim(String("(", fmt_fixed(Float64(config.iterations) / train_s, 0), " it/s)")),
        )
    )
    if removed > 0:
        done(String("Pruned ", fmt_thousands(removed), " transparent Gaussians"))

    var psnr = trainer.evaluate(test=True) if args.test_every > 0 else -1.0
    if not exists(out):
        makedirs(out)
    var ply = out + "/point_cloud.ply"
    trainer.save_ply(ply)
    done(String("Saved ", ply))
    var metrics = List[Tuple[String, Float64]]()
    metrics.append(("train_seconds", train_s))
    metrics.append(("gaussians", Float64(trainer.splats.count)))
    if psnr >= 0.0:
        metrics.append(("test_psnr", psnr))
    write_run_record(out + "/run.json", M.name(), args.scene, ctx.name(), args.settings, metrics)

    print()
    var table = Table("Results")
    table.add("Scene", name)
    table.add("Training time", fmt_duration(train_s))
    table.add("Gaussians", fmt_thousands(trainer.splats.count))
    if psnr >= 0.0:
        table.add("Test PSNR", String(bold(fmt_fixed(psnr, 2)), " dB"))
    table.add("Model", ply)
    table.print()

    if viewer:
        ref v = viewer.value()
        var items = List[Tuple[String, String]]()
        items.append(("Status", "Finished"))
        items.append(("Training time", fmt_duration(train_s)))
        items.append(("Gaussians", fmt_thousands(trainer.splats.count)))
        if psnr >= 0.0:
            items.append(("Test PSNR", String(fmt_fixed(psnr, 2), " dB")))
        v.training = False
        v.set_stats(items, 1.0)
        print()
        print(dim("Viewer still running with the final model. Ctrl+C to quit."))
        v.serve_forever(trainer.splats, config.sh_degree)
