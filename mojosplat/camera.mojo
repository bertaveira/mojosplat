"""Pinhole camera description shared by all pipeline stages."""


@fieldwise_init
struct Camera(ImplicitlyCopyable, Movable):
    """World-to-camera pinhole camera (OpenCV convention: +z forward, +y down).

    `view` is the 4x4 world-to-camera matrix in row-major order; only the top
    3x4 block ([R | t]) is used.
    """

    var view: SIMD[DType.float32, 16]
    var fx: Float32
    var fy: Float32
    var cx: Float32
    var cy: Float32
    var width: Int
    var height: Int
    var near: Float32
    var far: Float32

    @staticmethod
    def from_rt(
        R: SIMD[DType.float32, 16],
        width: Int,
        height: Int,
        fx: Float32,
        fy: Float32,
        cx: Float32,
        cy: Float32,
        near: Float32 = 0.1,
        far: Float32 = 100.0,
    ) -> Camera:
        """Build from a row-major 4x4 world-to-camera matrix."""
        return Camera(R, fx, fy, cx, cy, width, height, near, far)

    @staticmethod
    def translated(
        tx: Float32,
        ty: Float32,
        tz: Float32,
        width: Int,
        height: Int,
        fx: Float32,
        fy: Float32,
        near: Float32 = 0.1,
        far: Float32 = 100.0,
    ) -> Camera:
        """Identity rotation, world-to-camera translation (tx, ty, tz); principal point at the image center."""
        var v = SIMD[DType.float32, 16](0.0)
        v[0] = 1.0
        v[5] = 1.0
        v[10] = 1.0
        v[15] = 1.0
        v[3] = tx
        v[7] = ty
        v[11] = tz
        return Camera(
            v, fx, fy, Float32(width) / 2.0, Float32(height) / 2.0, width, height, near, far
        )

    def tiles_x(self, tile_size: Int) -> Int:
        return (self.width + tile_size - 1) // tile_size

    def tiles_y(self, tile_size: Int) -> Int:
        return (self.height + tile_size - 1) // tile_size

    def packed(self) -> SIMD[DType.float32, 32]:
        """Pack into the kernel-side parameter vector (see `CAM_*` indices)."""
        var p = SIMD[DType.float32, 32](0.0)
        comptime for i in range(12):
            p[i] = self.view[i]
        var W = Float32(self.width)
        var H = Float32(self.height)
        var tan_fov_x = 0.5 * W / self.fx
        var tan_fov_y = 0.5 * H / self.fy
        p[CAM_FX] = self.fx
        p[CAM_FY] = self.fy
        p[CAM_CX] = self.cx
        p[CAM_CY] = self.cy
        p[CAM_W] = W
        p[CAM_H] = H
        # Frustum limits used to clamp the Jacobian (as in gsplat).
        p[CAM_LIM_X_POS] = (W - self.cx) / self.fx + 0.3 * tan_fov_x
        p[CAM_LIM_X_NEG] = self.cx / self.fx + 0.3 * tan_fov_x
        p[CAM_LIM_Y_POS] = (H - self.cy) / self.fy + 0.3 * tan_fov_y
        p[CAM_LIM_Y_NEG] = self.cy / self.fy + 0.3 * tan_fov_y
        p[CAM_NEAR] = self.near
        p[CAM_FAR] = self.far
        return p


# Indices into the packed camera vector. 0..11 hold the top 3x4 of the view
# matrix in row-major order.
comptime CAM_FX = 12
comptime CAM_FY = 13
comptime CAM_CX = 14
comptime CAM_CY = 15
comptime CAM_W = 16
comptime CAM_H = 17
comptime CAM_LIM_X_POS = 18
comptime CAM_LIM_X_NEG = 19
comptime CAM_LIM_Y_POS = 20
comptime CAM_LIM_Y_NEG = 21
comptime CAM_NEAR = 22
comptime CAM_FAR = 23
