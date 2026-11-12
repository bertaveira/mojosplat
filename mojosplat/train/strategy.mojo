"""Densification strategies: how a method adds, moves and removes Gaussians while training.

The trainer calls its strategy at fixed points (see trainer.mojo):

    before_step(it)        before iteration `it` renders its view
    collects_stats(it)     whether that iteration's backward accumulates the
                           densification statistics (per-Gaussian visibility
                           count and screen-space gradient norm)
    finish()               once after the last iteration (e.g. a final prune)

A strategy owns its settings as `Config`. Every field of it is a command-line
option (see cli/args.mojo), so a method only shows the options of the strategy
it uses. `ClassicDensification` (densify.mojo) is 3DGS's clone, split and prune.
"""

from max.gpu.host import DeviceContext

from ..model.splats import Splats


trait Densification(Deinitable, Movable):
    comptime Config: Copyable & Defaultable & Deinitable & Movable

    def __init__(out self, ctx: DeviceContext, config: Self.Config) raises:
        ...

    def before_step(mut self, it: Int, mut splats: Splats, scene_extent: Float32) raises:
        """Called before iteration `it` (0-based) renders its view."""
        ...

    def collects_stats(self, it: Int) -> Bool:
        """Whether the backward of iteration `it` accumulates densification statistics."""
        ...

    def finish(mut self, mut splats: Splats) raises -> Int:
        """After training. Returns the number of Gaussians removed."""
        ...
