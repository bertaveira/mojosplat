"""Device memory providers for state that must outlive a call."""

from max.gpu.host import DeviceBuffer, DeviceContext

from ..core.types import dptr


trait StateAllocator:
    """Provides device memory for a training forward's saved state."""

    def alloc(mut self, nbytes: Int) raises -> Int:
        """Return the device address of >= nbytes, 256-byte aligned, that
        stays valid until the matching backward has run."""
        ...


struct DeviceArena(StateAllocator, Movable):
    """Reusable `StateAllocator` for pure-Mojo training loops.

    After `reset()`, the i-th `alloc` returns the i-th buffer (grown when
    needed), so a forward/backward per iteration does not reallocate. Only
    one `ForwardState` per arena is valid at a time.
    """

    var ctx: DeviceContext
    var buffers: List[DeviceBuffer[DType.uint8]]
    var sizes: List[Int]
    var next: Int

    def __init__(out self, ctx: DeviceContext):
        self.ctx = ctx
        self.buffers = List[DeviceBuffer[DType.uint8]]()
        self.sizes = List[Int]()
        self.next = 0

    def reset(mut self):
        self.next = 0

    def alloc(mut self, nbytes: Int) raises -> Int:
        var need = max(nbytes, 256)
        if self.next == len(self.buffers):
            self.buffers.append(self.ctx.enqueue_create_buffer[DType.uint8](need))
            self.sizes.append(need)
        elif self.sizes[self.next] < need:
            var cap = max(need, self.sizes[self.next] * 3 // 2)
            self.buffers[self.next] = self.ctx.enqueue_create_buffer[DType.uint8](cap)
            self.sizes[self.next] = cap
        var addr = Int(dptr(self.buffers[self.next]))
        self.next += 1
        return addr
