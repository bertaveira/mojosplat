"""The antimatter15 `.splat` scene format (32 bytes per Gaussian)."""

from std.sys import size_of

from .binary import read_file


# ---------------------------------------------------------------------------
# .splat scenes
# ---------------------------------------------------------------------------


struct SplatScene(Movable):
    """Gaussians in host memory, laid out as the pipeline expects.

    scales are linear (not log), quats are (w, x, y, z), opacities in [0, 1].
    """

    var count: Int
    var means: List[Float32]  # [N, 3]
    var scales: List[Float32]  # [N, 3]
    var quats: List[Float32]  # [N, 4]
    var opacities: List[Float32]  # [N]
    var colors: List[Float32]  # [N, 3]

    def __init__(out self, count: Int):
        self.count = count
        self.means = List[Float32](capacity=count * 3)
        self.scales = List[Float32](capacity=count * 3)
        self.quats = List[Float32](capacity=count * 4)
        self.opacities = List[Float32](capacity=count)
        self.colors = List[Float32](capacity=count * 3)


def load_splat(path: String) raises -> SplatScene:
    """Load an antimatter15-style `.splat` file (32 bytes per Gaussian).

    Layout: position xyz (3 x f32), scale xyz (3 x f32, linear), RGBA (4 x u8,
    alpha = activated opacity), rotation (4 x u8, (q * 128 + 128), order w x y z).
    """
    var data = read_file(path)
    if len(data) % 32 != 0 or len(data) == 0:
        raise Error(String("invalid .splat file: ", path))
    var n = len(data) // 32
    var scene = SplatScene(n)
    var base = data.unsafe_ptr()
    for i in range(n):
        var rec = base.unsafe_offset(i * 32)
        var f = rec.unsafe_bitcast[Float32]()
        for k in range(3):
            scene.means.append(f[unsafe_offset=k])
        for k in range(3):
            scene.scales.append(f[unsafe_offset=3 + k])
        for k in range(3):
            scene.colors.append(Float32(rec[unsafe_offset=24 + k]) / 255.0)
        scene.opacities.append(Float32(rec[unsafe_offset=27]) / 255.0)
        var q0 = (Float32(rec[unsafe_offset=28]) - 128.0) / 128.0
        var q1 = (Float32(rec[unsafe_offset=29]) - 128.0) / 128.0
        var q2 = (Float32(rec[unsafe_offset=30]) - 128.0) / 128.0
        var q3 = (Float32(rec[unsafe_offset=31]) - 128.0) / 128.0
        var norm = max((q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3) ** 0.5, 1e-10)
        scene.quats.append(q0 / norm)
        scene.quats.append(q1 / norm)
        scene.quats.append(q2 / norm)
        scene.quats.append(q3 / norm)
    return scene^
