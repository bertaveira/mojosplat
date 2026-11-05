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
from mojosplat.common import I32Ptr as _I32, U32Ptr
from mojosplat.rasterizer import ForwardState, Rasterizer, SplatParams, StateAllocator
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


struct PyAllocator(StateAllocator, Movable):
    """StateAllocator backed by a Python callable `alloc(nbytes) -> address`
    (torch.empty on the caching allocator; the Python side keeps the tensors
    alive until the backward)."""

    var callback: PythonObject

    def __init__(out self, callback: PythonObject):
        self.callback = callback

    def alloc(mut self, nbytes: Int) raises -> Int:
        return Int(py=self.callback(nbytes))


def _u32(obj: PythonObject) raises -> U32Ptr:
    return U32Ptr(unsafe_from_address=Int(py=obj))


def _splats(ptrs: PythonObject, sizes: PythonObject) raises -> SplatParams:
    return SplatParams(
        _f32(ptrs[0]), _f32(ptrs[1]), _f32(ptrs[2]), _f32(ptrs[3]),
        _f32(ptrs[4]), _f32(ptrs[5]), Int(py=sizes[0]), Int(py=sizes[1]),
    )


def _state_to_py(st: ForwardState) raises -> PythonObject:
    """ForwardState -> list of ints (sizes, then device addresses)."""
    var t = Python.list()
    for v in [
        st.n, st.n_visible, st.n_instances, st.n_tiles, st.n_bucket_blocks,
        st.width, st.height, st.grid_w, st.grid_h, st.active_bases,
    ]:
        t.append(v)
    for a in [
        Int(st.primitive_ids), Int(st.slot_of), Int(st.records), Int(st.colors),
        Int(st.instance_slots), Int(st.inst_a), Int(st.inst_b), Int(st.tile_offsets),
        Int(st.bucket_start), Int(st.bucket_tile), Int(st.snapshots), Int(st.final_T),
        Int(st.n_processed), Int(st.tile_max_n),
    ]:
        t.append(a)
    return t


def _state_from_py(t: PythonObject, cam: Camera, bg: SIMD[DType.float32, 4]) raises -> ForwardState:
    var st = ForwardState()
    st.n = Int(py=t[0])
    st.n_visible = Int(py=t[1])
    st.n_instances = Int(py=t[2])
    st.n_tiles = Int(py=t[3])
    st.n_bucket_blocks = Int(py=t[4])
    st.width = Int(py=t[5])
    st.height = Int(py=t[6])
    st.grid_w = Int(py=t[7])
    st.grid_h = Int(py=t[8])
    st.active_bases = Int(py=t[9])
    st.primitive_ids = _I32(unsafe_from_address=Int(py=t[10]))
    st.slot_of = _I32(unsafe_from_address=Int(py=t[11]))
    st.records = _f32(t[12])
    st.colors = _f32(t[13])
    st.instance_slots = _I32(unsafe_from_address=Int(py=t[14]))
    st.inst_a = _I32(unsafe_from_address=Int(py=t[15]))
    st.inst_b = _I32(unsafe_from_address=Int(py=t[16]))
    st.tile_offsets = _u32(t[17])
    st.bucket_start = _u32(t[18])
    st.bucket_tile = _u32(t[19])
    st.snapshots = _f32(t[20])
    st.final_T = _f32(t[21])
    st.n_processed = _u32(t[22])
    st.tile_max_n = _u32(t[23])
    st.cam = cam.packed()
    st.background = bg
    return st


struct Context(Movable, Writable):
    """Holds a Mojo DeviceContext (GPU 0) and the renderers' workspaces."""

    var renderer: Renderer[TILE_SIZE, CHANNELS]  # gsplat-compatible stages
    var rasterizer: Rasterizer[]  # Faster-GS style forward

    def __init__(out self) raises:
        var ctx = DeviceContext()
        self.renderer = Renderer[TILE_SIZE, CHANNELS](ctx)
        self.rasterizer = Rasterizer(ctx)

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


    @staticmethod
    def render_splats(
        self_: Pointer[Self, MutUnsafeAnyOrigin],
        ptrs: PythonObject,  # (means, log_scales, quats, opacity_logits, sh0, shN, image)
        camera: PythonObject,
        background: PythonObject,
        sizes: PythonObject,  # (N, shN bases per Gaussian, sh_degree)
    ) raises -> PythonObject:
        """Faster-GS style render from raw parameters; returns (visible, instances)."""
        var splats = SplatParams(
            _f32(ptrs[0]), _f32(ptrs[1]), _f32(ptrs[2]), _f32(ptrs[3]),
            _f32(ptrs[4]), _f32(ptrs[5]), Int(py=sizes[0]), Int(py=sizes[1]),
        )
        ref r = self_[].rasterizer
        r.render(splats, _camera(camera), _background(background), _f32(ptrs[6]), Int(py=sizes[2]))
        r.ctx.synchronize()
        return Python.tuple(r.num_visible, r.num_instances)


    @staticmethod
    def forward_train(
        self_: Pointer[Self, MutUnsafeAnyOrigin],
        ptrs: PythonObject,  # (means, log_scales, quats, opacity_logits, sh0, shN, image)
        cam_bg: PythonObject,  # (camera list, background list)
        sizes: PythonObject,  # (N, shN bases per Gaussian, sh_degree)
        alloc: PythonObject,  # alloc(nbytes) -> device address
    ) raises -> PythonObject:
        """Training forward; returns the saved state as a tuple of ints."""
        var allocator = PyAllocator(alloc)
        ref r = self_[].rasterizer
        var st = r.forward(
            _splats(ptrs, sizes), _camera(cam_bg[0]), _background(cam_bg[1]),
            _f32(ptrs[6]), Int(py=sizes[2]), allocator,
        )
        r.ctx.synchronize()
        return _state_to_py(st)

    @staticmethod
    def backward(
        self_: Pointer[Self, MutUnsafeAnyOrigin],
        ptrs: PythonObject,  # (means, log_scales, quats, opacity_logits, sh0, shN, image, grad_image)
        grad_ptrs: PythonObject,  # (6 gradient buffers, densification [2, N] or 0)
        cam_bg: PythonObject,
        state: PythonObject,
        sizes: PythonObject,  # (N, shN bases per Gaussian)
    ) raises -> PythonObject:
        """Writes every gradient buffer (culled Gaussians get 0)."""
        var st = _state_from_py(state, _camera(cam_bg[0]), _background(cam_bg[1]))
        var has_dens = Int(py=grad_ptrs[6]) != 0
        var dens = _f32(grad_ptrs[6]) if has_dens else _f32(grad_ptrs[0])
        ref r = self_[].rasterizer
        r.backward(
            _splats(ptrs, sizes), st, _f32(ptrs[6]), _f32(ptrs[7]),
            _splats(grad_ptrs, sizes), dens, has_dens,
        )
        r.ctx.synchronize()
        return PythonObject(None)


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
            .def_method[Context.render_splats]("render_splats")
            .def_method[Context.forward_train]("forward_train")
            .def_method[Context.backward]("backward")
        )
        m.def_function[_tile_size]("tile_size")
        return m.finalize()
    except e:
        abort(String("failed to create _mojosplat module: ", e))


def _tile_size() raises -> PythonObject:
    return PythonObject(TILE_SIZE)
