"""Python extension module exposing the Mojo renderer to PyTorch.

Torch owns the memory: every tensor is passed as its raw device address
(`tensor.data_ptr()`), so no copies are made. The Python wrapper
(`mojosplat_torch`) validates shapes/dtypes/contiguity and synchronizes the
torch stream before each call; every call here synchronizes the Mojo stream
before returning, so results are ready for torch.

Build: `pixi run -e torch build-ext`
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceContext

from mojosplat.binning import copy_i32
from mojosplat.camera import Camera
from mojosplat.common import F32Ptr, I32Ptr
from mojosplat.render import Gaussians, Renderer

comptime TILE_SIZE = 16
comptime CHANNELS = 3


def _f32(obj: PythonObject) raises -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(py=obj))


def _i32(obj: PythonObject) raises -> I32Ptr:
    return I32Ptr(unsafe_from_address=Int(py=obj))


def _camera(cam: PythonObject) raises -> Camera:
    """cam = [16 x view (row-major world-to-camera), fx, fy, cx, cy, W, H, near, far]."""
    if len(cam) != 24:
        raise Error("camera must be a sequence of 24 floats")
    var view = SIMD[DType.float32, 16](0.0)
    for i in range(16):
        view[i] = Float32(Float64(py=cam[i]))
    return Camera(
        view,
        Float32(Float64(py=cam[16])),
        Float32(Float64(py=cam[17])),
        Float32(Float64(py=cam[18])),
        Float32(Float64(py=cam[19])),
        Int(py=cam[20]),
        Int(py=cam[21]),
        Float32(Float64(py=cam[22])),
        Float32(Float64(py=cam[23])),
    )


def _background(bg: PythonObject) raises -> SIMD[DType.float32, 4]:
    var out = SIMD[DType.float32, 4](0.0)
    for i in range(min(len(bg), 4)):
        out[i] = Float32(Float64(py=bg[i]))
    return out


struct Context(Movable, Writable):
    """Holds a Mojo DeviceContext (GPU 0) and the renderer's workspaces."""

    var renderer: Renderer[TILE_SIZE, CHANNELS]

    def __init__(out self) raises:
        self.renderer = Renderer[TILE_SIZE, CHANNELS](DeviceContext())

    def write_to(self, mut writer: Some[Writer]):
        writer.write("mojosplat Context(tile_size=", TILE_SIZE, ", channels=", CHANNELS, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def py_init(out self: Context, args: PythonObject, kwargs: PythonObject) raises:
        self = Context()

    @staticmethod
    def project(
        self_: Pointer[Self, MutUnsafeAnyOrigin],
        gaussian_ptrs: PythonObject,  # (means3d, scales, quats, opacities)
        out_ptrs: PythonObject,  # (means2d, conics, depths, radii)
        camera: PythonObject,
        n: PythonObject,
    ) raises -> PythonObject:
        var g = Gaussians(
            _f32(gaussian_ptrs[0]), _f32(gaussian_ptrs[1]), _f32(gaussian_ptrs[2]),
            _f32(gaussian_ptrs[3]), _f32(gaussian_ptrs[3]), Int(py=n),
        )
        self_[].renderer.project(
            g, _camera(camera), _f32(out_ptrs[0]), _f32(out_ptrs[1]),
            _f32(out_ptrs[2]), _i32(out_ptrs[3]),
        )
        self_[].renderer.ctx.synchronize()
        return PythonObject(None)

    @staticmethod
    def bin(
        self_: Pointer[Self, MutUnsafeAnyOrigin],
        in_ptrs: PythonObject,  # (means2d, radii, depths)
        n: PythonObject,
        width: PythonObject,
        height: PythonObject,
    ) raises -> PythonObject:
        """Bin into renderer memory; returns M. Fetch with `read_bins`."""
        var m = self_[].renderer.bin(
            _f32(in_ptrs[0]), _i32(in_ptrs[1]), _f32(in_ptrs[2]),
            Int(py=n), Int(py=width), Int(py=height),
        )
        self_[].renderer.ctx.synchronize()
        return PythonObject(m)

    @staticmethod
    def read_bins(
        self_: Pointer[Self, MutUnsafeAnyOrigin],
        ids_ptr: PythonObject,  # out int32 [M]
        offsets_ptr: PythonObject,  # out int32 [num_tiles + 1]
        num_tiles: PythonObject,
    ) raises -> PythonObject:
        """Copy the last binning result into torch-owned buffers."""
        ref r = self_[].renderer
        var ctx = r.ctx
        copy_i32(ctx, ctx.stream(), r.sorted_ids(), _i32(ids_ptr), r.num_intersections)
        copy_i32(ctx, ctx.stream(), r.tile_offsets_ptr(), _i32(offsets_ptr), Int(py=num_tiles) + 1)
        ctx.synchronize()
        return PythonObject(None)

    @staticmethod
    def rasterize(
        self_: Pointer[Self, MutUnsafeAnyOrigin],
        ptrs: PythonObject,  # (means2d, conics, colors, opacities, tile_offsets, ids, image)
        background: PythonObject,
        width: PythonObject,
        height: PythonObject,
    ) raises -> PythonObject:
        self_[].renderer.rasterize(
            _f32(ptrs[0]), _f32(ptrs[1]), _f32(ptrs[2]), _f32(ptrs[3]),
            _background(background), _i32(ptrs[4]), _i32(ptrs[5]), _f32(ptrs[6]),
            Int(py=width), Int(py=height),
        )
        self_[].renderer.ctx.synchronize()
        return PythonObject(None)

    @staticmethod
    def render(
        self_: Pointer[Self, MutUnsafeAnyOrigin],
        gaussian_ptrs: PythonObject,  # (means3d, scales, quats, opacities, colors)
        image_ptr: PythonObject,
        camera: PythonObject,
        background: PythonObject,
        n: PythonObject,
    ) raises -> PythonObject:
        """Full pipeline; returns the number of tile intersections."""
        var g = Gaussians(
            _f32(gaussian_ptrs[0]), _f32(gaussian_ptrs[1]), _f32(gaussian_ptrs[2]),
            _f32(gaussian_ptrs[3]), _f32(gaussian_ptrs[4]), Int(py=n),
        )
        self_[].renderer.render(g, _camera(camera), _background(background), _f32(image_ptr))
        self_[].renderer.ctx.synchronize()
        return PythonObject(self_[].renderer.num_intersections)


@export
def PyInit__mojosplat() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojosplat")
        _ = (
            m.add_type[Context]("Context")
            .def_py_init[Context.py_init]()
            .def_method[Context.project]("project")
            .def_method[Context.bin]("bin")
            .def_method[Context.read_bins]("read_bins")
            .def_method[Context.rasterize]("rasterize")
            .def_method[Context.render]("render")
        )
        m.def_function[_tile_size]("tile_size")
        return m.finalize()
    except e:
        abort(String("failed to create _mojosplat module: ", e))


def _tile_size() raises -> PythonObject:
    return PythonObject(TILE_SIZE)
