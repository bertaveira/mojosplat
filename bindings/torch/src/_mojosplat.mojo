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

from mojosplat.core.camera import Camera
from mojosplat.core.splats import SplatView
from mojosplat.core.types import F32Ptr, I32Ptr, U32Ptr
from mojosplat.gpu.arena import StateAllocator
from mojosplat.render.rasterizer import ForwardState, Rasterizer


def _f32(obj: PythonObject) raises -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(py=obj))


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


struct PyAllocator(Movable, StateAllocator):
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


def _splats(ptrs: PythonObject, sizes: PythonObject) raises -> SplatView:
    return SplatView(
        _f32(ptrs[0]),
        _f32(ptrs[1]),
        _f32(ptrs[2]),
        _f32(ptrs[3]),
        _f32(ptrs[4]),
        _f32(ptrs[5]),
        Int(py=sizes[0]),
        Int(py=sizes[1]),
    )


def _state_to_py(st: ForwardState) raises -> PythonObject:
    """ForwardState -> list of ints (sizes, then device addresses)."""
    var t = Python.list()
    for v in [
        st.n,
        st.n_visible,
        st.n_instances,
        st.n_tiles,
        st.n_bucket_blocks,
        st.width,
        st.height,
        st.grid_w,
        st.grid_h,
        st.active_bases,
    ]:
        t.append(v)
    for a in [
        Int(st.slot_of),
        Int(st.records),
        Int(st.colors),
        Int(st.instance_slots),
        Int(st.inst_a),
        Int(st.inst_b),
        Int(st.tile_offsets),
        Int(st.bucket_start),
        Int(st.bucket_tile),
        Int(st.snapshots),
        Int(st.final_T),
        Int(st.n_processed),
        Int(st.tile_max_n),
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
    st.slot_of = I32Ptr(unsafe_from_address=Int(py=t[10]))
    st.records = _f32(t[11])
    st.colors = _f32(t[12])
    st.instance_slots = I32Ptr(unsafe_from_address=Int(py=t[13]))
    st.inst_a = I32Ptr(unsafe_from_address=Int(py=t[14]))
    st.inst_b = I32Ptr(unsafe_from_address=Int(py=t[15]))
    st.tile_offsets = _u32(t[16])
    st.bucket_start = _u32(t[17])
    st.bucket_tile = _u32(t[18])
    st.snapshots = _f32(t[19])
    st.final_T = _f32(t[20])
    st.n_processed = _u32(t[21])
    st.tile_max_n = _u32(t[22])
    st.cam = cam.packed()
    st.background = bg
    return st


struct Context(Movable, Writable):
    """Holds a Mojo DeviceContext (GPU 0) and the rasterizer's workspaces."""

    var rasterizer: Rasterizer[]

    def __init__(out self) raises:
        self.rasterizer = Rasterizer(DeviceContext())

    def write_to(self, mut writer: Some[Writer]):
        writer.write("mojosplat Context(tile=", Rasterizer[].tile_w, "x", Rasterizer[].tile_h, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def py_init(out self: Context, args: PythonObject, kwargs: PythonObject) raises:
        self = Context()

    @staticmethod
    def render_splats(
        self_: Pointer[Self, MutUnsafeAnyOrigin],
        ptrs: PythonObject,  # (means, log_scales, quats, opacity_logits, sh0, shN, image)
        camera: PythonObject,
        background: PythonObject,
        sizes: PythonObject,  # (N, shN bases per Gaussian, sh_degree)
    ) raises -> PythonObject:
        """Faster-GS style render from raw parameters; returns (visible, instances)."""
        ref r = self_[].rasterizer
        r.render(_splats(ptrs, sizes), _camera(camera), _background(background), _f32(ptrs[6]), Int(py=sizes[2]))
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
            _splats(ptrs, sizes),
            _camera(cam_bg[0]),
            _background(cam_bg[1]),
            _f32(ptrs[6]),
            Int(py=sizes[2]),
            allocator,
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
            _splats(ptrs, sizes),
            st,
            _f32(ptrs[6]),
            _f32(ptrs[7]),
            _splats(grad_ptrs, sizes),
            dens,
            has_dens,
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
            .def_method[Context.render_splats]("render_splats")
            .def_method[Context.forward_train]("forward_train")
            .def_method[Context.backward]("backward")
        )
        return m.finalize()
    except e:
        abort(String("failed to create _mojosplat module: ", e))
