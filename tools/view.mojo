"""Open a trained model in the web viewer.

Usage:
    pixi run view output/garden/point_cloud.ply [--scene data/mipnerf360/garden] [--port 8080]

Serves http://localhost:8080 until interrupted. With --scene, the dataset's
cameras are available in the viewer (and set the initial view and up
direction).
"""

from max.gpu.host import DeviceContext

from mojosplat import Splats, load_splat, read_ply
from mojosplat.cli import Arg, bold, dim, done, fmt_thousands, parse_args
from mojosplat.core.camera import Camera
from mojosplat.train.dataset import load_colmap_dataset
from mojosplat.viewer import Viewer, ViewerConfig


struct ViewArgs(Defaultable, Movable):
    @__annotation(Arg("Model to view (.ply or .splat)", positional=True))
    var model: String

    @__annotation(Arg("COLMAP scene: use its cameras for the initial view and 'View' buttons"))
    var scene: String

    @__annotation(Arg("Image folder inside the scene"))
    var images: String

    @__annotation(Arg("Viewer"))
    var viewer: ViewerConfig

    def __init__(out self):
        self.model = ""
        self.scene = ""
        self.images = "images_4"
        self.viewer = ViewerConfig()


def main() raises:
    var args = parse_args[ViewArgs]("pixi run view", "Open a trained model in the web viewer.")
    var ctx = DeviceContext()
    print(bold("MojoSplat"), dim("·"), "view", dim("·"), ctx.name())
    var splats: Splats
    if args.model.endswith(".ply"):
        splats = Splats.from_host(ctx, read_ply(args.model))
    else:
        splats = Splats.from_scene(ctx, load_splat(args.model))
    done(String("Loaded ", args.model, ": ", fmt_thousands(splats.count), " Gaussians"))
    var cams = List[Camera]()
    if args.scene.byte_length() > 0:
        var ds = load_colmap_dataset(ctx, args.scene, args.images, 0)
        for v in ds.train:
            cams.append(v.camera)
        done(String("Loaded ", len(cams), " cameras from ", args.scene))
    var parts = args.model.rstrip("/").split("/")
    var title = String(parts[len(parts) - 1])
    if len(parts) >= 2 and title == "point_cloud.ply":
        title = String(parts[len(parts) - 2])
    var viewer = Viewer(ctx, args.viewer, splats, cams, title, False)
    print(dim("Ctrl+C to quit"))
    viewer.serve_forever(splats, splats.view().max_sh_degree())
