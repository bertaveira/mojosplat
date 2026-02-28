import pytest
import torch
import numpy as np

from mojosplat.projection import project_gaussians
from mojosplat.utils import Camera


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device("cuda:0")


@pytest.fixture
def identity_camera(device):
    """Camera at origin, identity rotation, looking down +Z."""
    H, W = 64, 64
    R = torch.eye(3, device=device, dtype=torch.float32)
    T = torch.zeros(3, device=device, dtype=torch.float32)
    return Camera(R, T, H, W, fx=100.0, fy=100.0, cx=W / 2.0, cy=H / 2.0, near=0.1, far=100.0)


@pytest.fixture
def offset_camera(device):
    """Camera translated along Z, looking at origin."""
    H, W = 64, 64
    R = torch.eye(3, device=device, dtype=torch.float32)
    T = torch.tensor([0.0, 0.0, 5.0], device=device, dtype=torch.float32)
    return Camera(R, T, H, W, fx=100.0, fy=100.0, cx=W / 2.0, cy=H / 2.0, near=0.1, far=100.0)


def make_gaussians(device, N=10, *, seed=42):
    """Generate reproducible random Gaussians that are visible (positive depth, non-trivial rotation)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    means3d = torch.randn(N, 3, device=device, dtype=torch.float32, generator=gen) * 2.0
    means3d[:, 2] = means3d[:, 2].abs() + 1.0

    scales = torch.log(torch.rand(N, 3, device=device, dtype=torch.float32, generator=gen) * 0.3 + 0.05)

    quats = torch.randn(N, 4, device=device, dtype=torch.float32, generator=gen)
    quats = torch.nn.functional.normalize(quats, p=2, dim=-1)

    opacities = torch.sigmoid(torch.randn(N, 1, device=device, dtype=torch.float32, generator=gen))
    return means3d, scales, quats, opacities


class TestProjectionShapes:
    """Verify output shapes and dtypes from the mojo backend."""

    def test_single_gaussian(self, device, identity_camera):
        means3d = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=torch.float32)
        scales = torch.log(torch.tensor([[0.1, 0.1, 0.1]], device=device, dtype=torch.float32))
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        opacities = torch.tensor([[1.0]], device=device, dtype=torch.float32)

        means2d, conics, depths, radii = project_gaussians(
            means3d, scales, quats, opacities, identity_camera, backend="mojo"
        )

        assert means2d.shape == (1, 2), f"Expected (1,2), got {means2d.shape}"
        assert conics.shape == (1, 3), f"Expected (1,3), got {conics.shape}"
        assert depths.shape == (1,), f"Expected (1,), got {depths.shape}"
        assert radii.shape == (1, 2), f"Expected (1,2), got {radii.shape}"
        assert means2d.dtype == torch.float32
        assert radii.dtype == torch.int32

    def test_batch(self, device, identity_camera):
        N = 50
        means3d, scales, quats, opacities = make_gaussians(device, N)

        means2d, conics, depths, radii = project_gaussians(
            means3d, scales, quats, opacities, identity_camera, backend="mojo"
        )

        assert means2d.shape == (N, 2)
        assert conics.shape == (N, 3)
        assert depths.shape == (N,)
        assert radii.shape == (N, 2)


class TestMojoVsGsplat:
    """
    Compare mojo projection against gsplat (the ground-truth reference).

    Both backends receive the same inputs and should produce nearly identical
    outputs.  The torch backend uses a slightly different radius formula
    (no opacity-aware radius), so we only compare mojo vs gsplat here.
    """

    @pytest.fixture(params=[1, 10, 100, 500])
    def gaussians(self, request, device):
        return make_gaussians(device, N=request.param)

    @pytest.fixture(params=["identity", "offset"])
    def camera(self, request, device, identity_camera, offset_camera):
        return {"identity": identity_camera, "offset": offset_camera}[request.param]

    def _run_both(self, means3d, scales, quats, opacities, camera):
        mojo_out = project_gaussians(means3d, scales, quats, opacities, camera, backend="mojo")
        gsplat_out = project_gaussians(means3d, scales, quats, opacities, camera, backend="gsplat")
        return mojo_out, gsplat_out

    def test_means2d(self, device, gaussians, camera):
        means3d, scales, quats, opacities = gaussians
        (m2d_m, _, _, radii_m), (m2d_g, _, _, radii_g) = self._run_both(
            means3d, scales, quats, opacities, camera
        )

        # Only compare non-culled gaussians (radii > 0 in both backends)
        visible_m = (radii_m[:, 0] > 0) & (radii_m[:, 1] > 0)
        visible_g = (radii_g[:, 0] > 0) & (radii_g[:, 1] > 0)
        visible = visible_m & visible_g

        if visible.sum() == 0:
            pytest.skip("No visible gaussians in this configuration")

        torch.testing.assert_close(
            m2d_m[visible], m2d_g[visible],
            atol=1e-3, rtol=1e-3,
            msg="means2d mismatch between mojo and gsplat",
        )

    def test_depths(self, device, gaussians, camera):
        means3d, scales, quats, opacities = gaussians
        (_, _, dep_m, radii_m), (_, _, dep_g, radii_g) = self._run_both(
            means3d, scales, quats, opacities, camera
        )

        visible_m = (radii_m[:, 0] > 0) & (radii_m[:, 1] > 0)
        visible_g = (radii_g[:, 0] > 0) & (radii_g[:, 1] > 0)
        visible = visible_m & visible_g

        if visible.sum() == 0:
            pytest.skip("No visible gaussians in this configuration")

        torch.testing.assert_close(
            dep_m[visible], dep_g[visible],
            atol=1e-4, rtol=1e-4,
            msg="depths mismatch between mojo and gsplat",
        )

    def test_conics(self, device, gaussians, camera):
        means3d, scales, quats, opacities = gaussians
        (_, con_m, _, radii_m), (_, con_g, _, radii_g) = self._run_both(
            means3d, scales, quats, opacities, camera
        )

        visible_m = (radii_m[:, 0] > 0) & (radii_m[:, 1] > 0)
        visible_g = (radii_g[:, 0] > 0) & (radii_g[:, 1] > 0)
        visible = visible_m & visible_g

        if visible.sum() == 0:
            pytest.skip("No visible gaussians in this configuration")

        torch.testing.assert_close(
            con_m[visible], con_g[visible],
            atol=1e-2, rtol=1e-2,
            msg="conics mismatch between mojo and gsplat",
        )

    def test_radii(self, device, gaussians, camera):
        means3d, scales, quats, opacities = gaussians
        (_, _, _, radii_m), (_, _, _, radii_g) = self._run_both(
            means3d, scales, quats, opacities, camera
        )

        visible_m = (radii_m[:, 0] > 0) & (radii_m[:, 1] > 0)
        visible_g = (radii_g[:, 0] > 0) & (radii_g[:, 1] > 0)
        visible = visible_m & visible_g

        if visible.sum() == 0:
            pytest.skip("No visible gaussians in this configuration")

        torch.testing.assert_close(
            radii_m[visible].float(), radii_g[visible].float(),
            atol=1.0, rtol=0.1,
            msg="radii mismatch between mojo and gsplat",
        )

    def test_culling_agreement(self, device, gaussians, camera):
        """Both backends should cull the same gaussians."""
        means3d, scales, quats, opacities = gaussians
        (_, _, _, radii_m), (_, _, _, radii_g) = self._run_both(
            means3d, scales, quats, opacities, camera
        )

        culled_m = (radii_m[:, 0] == 0) & (radii_m[:, 1] == 0)
        culled_g = (radii_g[:, 0] == 0) & (radii_g[:, 1] == 0)

        # Mismatch count should be very small (edge-cases from float rounding)
        mismatch = (culled_m != culled_g).sum().item()
        total = means3d.shape[0]
        assert mismatch <= max(1, total // 20), (
            f"Culling disagreement: {mismatch}/{total} gaussians differ"
        )


class TestProjectionGeometry:
    """Sanity checks on geometric properties."""

    def test_on_axis_projects_to_center(self, device, offset_camera):
        """A Gaussian on the camera axis should project near the image center."""
        means3d = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        scales = torch.log(torch.tensor([[0.1, 0.1, 0.1]], device=device, dtype=torch.float32))
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        opacities = torch.tensor([[0.5]], device=device, dtype=torch.float32)

        means2d, _, depths, radii = project_gaussians(
            means3d, scales, quats, opacities, offset_camera, backend="mojo"
        )

        if (radii == 0).all():
            pytest.skip("Gaussian was culled")

        px, py = means2d[0, 0].item(), means2d[0, 1].item()
        cx, cy = offset_camera.cx, offset_camera.cy
        assert abs(px - cx) < 2.0, f"X={px} should be near cx={cx}"
        assert abs(py - cy) < 2.0, f"Y={py} should be near cy={cy}"

    def test_depth_matches_geometry(self, device, identity_camera):
        """Depth should equal the z-coordinate when camera is at origin with identity rotation."""
        z_vals = [1.0, 3.0, 10.0]
        means3d = torch.tensor([[0.0, 0.0, z] for z in z_vals], device=device, dtype=torch.float32)
        scales = torch.log(torch.tensor([[0.1, 0.1, 0.1]] * len(z_vals), device=device, dtype=torch.float32))
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(z_vals), device=device, dtype=torch.float32)
        opacities = torch.ones(len(z_vals), 1, device=device, dtype=torch.float32)

        _, _, depths, radii = project_gaussians(
            means3d, scales, quats, opacities, identity_camera, backend="mojo"
        )

        for i, z in enumerate(z_vals):
            if radii[i, 0] > 0:
                assert abs(depths[i].item() - z) < 1e-3, f"depth={depths[i].item()} != z={z}"

    def test_opacity_culling(self, device, identity_camera):
        means3d = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=torch.float32)
        scales = torch.log(torch.tensor([[0.1, 0.1, 0.1]], device=device, dtype=torch.float32))
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        low_opacity = torch.tensor([[0.001]], device=device, dtype=torch.float32)

        _, _, _, radii = project_gaussians(
            means3d, scales, quats, low_opacity, identity_camera, backend="mojo"
        )
        assert (radii == 0).all(), "Low-opacity Gaussian should be culled"

    def test_behind_camera_culled(self, device, identity_camera):
        means3d = torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=torch.float32)
        scales = torch.log(torch.tensor([[0.1, 0.1, 0.1]], device=device, dtype=torch.float32))
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        opacities = torch.tensor([[1.0]], device=device, dtype=torch.float32)

        _, _, _, radii = project_gaussians(
            means3d, scales, quats, opacities, identity_camera, backend="mojo"
        )
        assert (radii == 0).all(), "Gaussian behind camera should be culled"


class TestNonTrivialRotations:
    """
    Specifically test with non-identity quaternions.  Rotation bugs only
    manifest when the Gaussian orientation actually matters for the
    projected covariance.
    """

    def test_anisotropic_rotated_gaussian(self, device, identity_camera):
        """An elongated Gaussian rotated 45 deg should match gsplat exactly."""
        means3d = torch.tensor([[0.0, 0.0, 5.0]], device=device, dtype=torch.float32)
        scales = torch.log(torch.tensor([[0.5, 0.05, 0.05]], device=device, dtype=torch.float32))

        # 45-degree rotation around Z axis: quat = (cos(pi/8), 0, 0, sin(pi/8))
        angle = torch.tensor(torch.pi / 4)
        quats = torch.tensor([[torch.cos(angle / 2), 0.0, 0.0, torch.sin(angle / 2)]],
                             device=device, dtype=torch.float32)
        opacities = torch.tensor([[1.0]], device=device, dtype=torch.float32)

        mojo_out = project_gaussians(means3d, scales, quats, opacities, identity_camera, backend="mojo")
        gsplat_out = project_gaussians(means3d, scales, quats, opacities, identity_camera, backend="gsplat")

        m2d_m, con_m, dep_m, rad_m = mojo_out
        m2d_g, con_g, dep_g, rad_g = gsplat_out

        torch.testing.assert_close(m2d_m, m2d_g, atol=1e-3, rtol=1e-3, msg="means2d")
        torch.testing.assert_close(dep_m, dep_g, atol=1e-4, rtol=1e-4, msg="depths")
        torch.testing.assert_close(con_m, con_g, atol=1e-2, rtol=1e-2, msg="conics")
        torch.testing.assert_close(rad_m.float(), rad_g.float(), atol=1.0, rtol=0.15, msg="radii")

    def test_multiple_orientations(self, device, identity_camera):
        """Several Gaussians with different orientations should all match gsplat."""
        N = 8
        means3d = torch.zeros(N, 3, device=device, dtype=torch.float32)
        means3d[:, 2] = 5.0  # all at z=5

        scales = torch.log(torch.tensor([[0.4, 0.05, 0.05]] * N, device=device, dtype=torch.float32))

        # Rotate each Gaussian by a different angle around Z
        angles = torch.linspace(0, torch.pi, N, device=device)
        quats = torch.stack([
            torch.cos(angles / 2),
            torch.zeros_like(angles),
            torch.zeros_like(angles),
            torch.sin(angles / 2),
        ], dim=-1)
        opacities = torch.ones(N, 1, device=device, dtype=torch.float32)

        mojo_out = project_gaussians(means3d, scales, quats, opacities, identity_camera, backend="mojo")
        gsplat_out = project_gaussians(means3d, scales, quats, opacities, identity_camera, backend="gsplat")

        _, con_m, _, _ = mojo_out
        _, con_g, _, _ = gsplat_out

        torch.testing.assert_close(
            con_m, con_g, atol=1e-2, rtol=1e-2,
            msg="Conics should match across multiple orientations",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
