"""Evaluate a trained model on the held-out views of a COLMAP scene.

Usage:
    pixi run eval data/mipnerf360/garden output/garden/point_cloud.ply
                  [--method faster-gs] [--images images_4] [--save out_dir] [--test-every 8]

The program itself is `mojosplat.apps.evaluate`; this picks the method.
"""

from mojosplat.apps import evaluate, method_arg, unknown_method
from mojosplat.methods import FasterGS


def main() raises:
    var name = method_arg(FasterGS.name())
    if name == FasterGS.name():
        evaluate[FasterGS]()
    else:
        unknown_method(name, [FasterGS.name()])
