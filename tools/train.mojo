"""Train a model on a COLMAP scene, in pure Mojo.

Usage:
    pixi run train data/mipnerf360/garden [--iterations 30000] [--no-densify] ...
    pixi run train data/mipnerf360/garden --method faster-gs --help   # every option of a method
    pixi run train data/mipnerf360/garden --viewer   # watch it live at http://localhost:8080

The program itself is `mojosplat.apps.train`; this picks the method.
"""

from mojosplat.apps import method_arg, train, unknown_method
from mojosplat.methods import BetaSplattingMethod, FasterGS


def main() raises:
    var name = method_arg(FasterGS.name())
    if name == FasterGS.name():
        train[FasterGS]()
    elif name == BetaSplattingMethod.name():
        train[BetaSplattingMethod]()
    else:
        unknown_method(name, [FasterGS.name(), BetaSplattingMethod.name()])
