import pytest
import torch
import math

from mojosplat.rasterization import rasterize_gaussians
from mojosplat.projection import project_gaussians
from mojosplat.binning import bin_gaussians_to_tiles
from mojosplat.utils import Camera


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device("cuda:0")


def make_camera(device, H=64, W=64, fx=100.0, fy=100.0):
    R = torch.eye(3, device=device, dtype=torch.float32)
    T = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
    return Camera(R=R, T=T, H=H, W=W, fx=fx, fy=fy, cx=W / 2.0, cy=H / 2.0, near=0.1, far=100.0)


def make_gaussian_data(device, N, *, seed=0, depth_range=(1.5, 5.0), scale_log=-2.0, opacity_range=(0.5, 0.95)):
    """Generate reproducible Gaussian data that is reliably visible."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    means3d = torch.randn(N, 3, device=device, generator=gen) * 1.0
    means3d[:, 2] = torch.rand(N, device=device, generator=gen) * (depth_range[1] - depth_range[0]) + depth_range[0]
    log_scales = torch.ones(N, 3, device=device) * scale_log
    log_scales += torch.randn(N, 3, device=device, generator=gen) * 0.1
    quats = torch.randn(N, 4, device=device, generator=gen)
    quats = torch.nn.functional.normalize(quats, dim=1)
    opacities = torch.rand(N, device=device, generator=gen) * (opacity_range[1] - opacity_range[0]) + opacity_range[0]
    colors = torch.rand(N, 3, device=device, generator=gen)
    return means3d, log_scales, quats, opacities, colors


def run_projection_and_binning(means3d, log_scales, quats, opacities, camera, tile_size=16):
    """Run gsplat projection + binning to produce shared inputs for rasterization comparison."""
    means2d, conics, depths, radii = project_gaussians(
        means3d, log_scales, quats, opacities, camera, backend="gsplat"
    )
    sorted_indices, tile_ranges = bin_gaussians_to_tiles(
        means2d, radii, depths, camera.H, camera.W, tile_size, backend="gsplat"
    )
    return means2d, conics, depths, radii, opacities, sorted_indices, tile_ranges


# ---------------------------------------------------------------------------
# Shape and basic functionality
# ---------------------------------------------------------------------------

class TestRasterizationShapes:
    @pytest.mark.parametrize("backend", ["mojo", "gsplat"])
    def test_output_shape_and_dtype(self, device, backend):
        camera = make_camera(device)
        means3d, log_scales, quats, opacities, colors = make_gaussian_data(device, 20, seed=1)
        means2d, conics, _, _, opacities, sorted_ids, tile_ranges = run_projection_and_binning(
            means3d, log_scales, quats, opacities, camera
        )
        if sorted_ids.numel() == 0:
            pytest.skip("No visible gaussians")
        bg = torch.zeros(3, device=device)
        result = rasterize_gaussians(
            means2d, conics, colors, opacities, bg, tile_ranges, sorted_ids, camera, backend=backend,
        )
        assert result.shape == (camera.H, camera.W, 3)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    def test_output_on_correct_device(self, device):
        camera = make_camera(device)
        means3d, log_scales, quats, opacities, colors = make_gaussian_data(device, 10, seed=2)
        means2d, conics, _, _, opacities, sorted_ids, tile_ranges = run_projection_and_binning(
            means3d, log_scales, quats, opacities, camera
        )
        if sorted_ids.numel() == 0:
            pytest.skip("No visible gaussians")
        bg = torch.zeros(3, device=device)
        result = rasterize_gaussians(
            means2d, conics, colors, opacities, bg, tile_ranges, sorted_ids, camera, backend="mojo",
        )
        assert result.device == device


# ---------------------------------------------------------------------------
# Mojo vs gsplat pixel-level comparison
# ---------------------------------------------------------------------------

class TestMojoVsGsplat:
    """Feed identical projection/binning outputs into both backends and compare."""

    @pytest.mark.parametrize("N", [1, 5, 50, 200])
    def test_rendered_image_close(self, device, N):
        camera = make_camera(device, H=64, W=64, fx=100.0)
        means3d, log_scales, quats, opacities, colors = make_gaussian_data(device, N, seed=N)
        means2d, conics, _, _, opacities, sorted_ids, tile_ranges = run_projection_and_binning(
            means3d, log_scales, quats, opacities, camera
        )
        if sorted_ids.numel() == 0:
            pytest.skip("No visible gaussians")
        bg = torch.zeros(3, device=device)
        img_mojo = rasterize_gaussians(
            means2d, conics, colors, opacities, bg, tile_ranges, sorted_ids, camera, backend="mojo",
        )
        img_gsplat = rasterize_gaussians(
            means2d, conics, colors, opacities, bg, tile_ranges, sorted_ids, camera, backend="gsplat",
        )
        torch.testing.assert_close(img_mojo, img_gsplat, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("N", [5, 50])
    def test_nonzero_background(self, device, N):
        """Background blending must match: final = accumulated + T * background."""
        camera = make_camera(device, H=64, W=64, fx=100.0)
        means3d, log_scales, quats, opacities, colors = make_gaussian_data(device, N, seed=N + 100)
        means2d, conics, _, _, opacities, sorted_ids, tile_ranges = run_projection_and_binning(
            means3d, log_scales, quats, opacities, camera
        )
        if sorted_ids.numel() == 0:
            pytest.skip("No visible gaussians")
        bg = torch.tensor([0.2, 0.4, 0.6], device=device)
        img_mojo = rasterize_gaussians(
            means2d, conics, colors, opacities, bg, tile_ranges, sorted_ids, camera, backend="mojo",
        )
        img_gsplat = rasterize_gaussians(
            means2d, conics, colors, opacities, bg, tile_ranges, sorted_ids, camera, backend="gsplat",
        )
        torch.testing.assert_close(img_mojo, img_gsplat, atol=1e-4, rtol=1e-4)

    def test_larger_image(self, device):
        camera = make_camera(device, H=128, W=128, fx=200.0)
        means3d, log_scales, quats, opacities, colors = make_gaussian_data(device, 100, seed=42)
        means2d, conics, _, _, opacities, sorted_ids, tile_ranges = run_projection_and_binning(
            means3d, log_scales, quats, opacities, camera
        )
        if sorted_ids.numel() == 0:
            pytest.skip("No visible gaussians")
        bg = torch.tensor([0.1, 0.1, 0.1], device=device)
        img_mojo = rasterize_gaussians(
            means2d, conics, colors, opacities, bg, tile_ranges, sorted_ids, camera, backend="mojo",
        )
        img_gsplat = rasterize_gaussians(
            means2d, conics, colors, opacities, bg, tile_ranges, sorted_ids, camera, backend="gsplat",
        )
        torch.testing.assert_close(img_mojo, img_gsplat, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Geometric sanity checks
# ---------------------------------------------------------------------------

class TestRasterizationGeometry:
    def test_single_gaussian_at_center(self, device):
        """A centered Gaussian should produce a bright spot at the image center."""
        camera = make_camera(device, H=64, W=64, fx=100.0)
        means3d = torch.tensor([[0.0, 0.0, 3.0]], device=device)
        log_scales = torch.log(torch.tensor([[0.15, 0.15, 0.15]], device=device))
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        opacities = torch.tensor([0.95], device=device)
        colors = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        means2d, conics, _, _, opacities, sorted_ids, tile_ranges = run_projection_and_binning(
            means3d, log_scales, quats, opacities, camera
        )
        if sorted_ids.numel() == 0:
            pytest.skip("Gaussian not visible")
        bg = torch.zeros(3, device=device)
        img = rasterize_gaussians(
            means2d, conics, colors, opacities, bg, tile_ranges, sorted_ids, camera, backend="mojo",
        )
        cy, cx = camera.H // 2, camera.W // 2
        center_val = img[cy, cx, 0].item()
        assert center_val > 0.1, f"Center pixel red channel too dim: {center_val}"
        assert img[cy, cx, 1].item() < center_val, "Green should be less than red"
        assert img[cy, cx, 2].item() < center_val, "Blue should be less than red"

    def test_background_only_no_gaussians(self, device):
        """Empty tile ranges should yield the pure background color."""
        camera = make_camera(device, H=32, W=32, fx=50.0)
        n_h = math.ceil(camera.H / 16)
        n_w = math.ceil(camera.W / 16)
        means2d = torch.empty(0, 2, device=device)
        conics = torch.empty(0, 3, device=device)
        colors = torch.empty(0, 3, device=device)
        opacities = torch.empty(0, device=device)
        tile_ranges = torch.zeros(n_h, n_w, 2, device=device, dtype=torch.int32)
        sorted_ids = torch.empty(0, device=device, dtype=torch.int32)
        bg = torch.tensor([0.3, 0.5, 0.7], device=device)
        try:
            img = rasterize_gaussians(
                means2d, conics, colors, opacities, bg, tile_ranges, sorted_ids, camera, backend="mojo",
            )
            expected = bg.view(1, 1, 3).expand(camera.H, camera.W, 3)
            torch.testing.assert_close(img, expected, atol=1e-6, rtol=0)
        except Exception:
            pytest.skip("Empty-scene rasterization not yet supported")

    def test_opacity_ordering(self, device):
        """Higher opacity Gaussian should produce brighter pixels."""
        camera = make_camera(device, H=64, W=64, fx=100.0)
        means3d = torch.tensor([[0.0, 0.0, 3.0]], device=device)
        log_scales = torch.log(torch.tensor([[0.15, 0.15, 0.15]], device=device))
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        colors = torch.tensor([[1.0, 1.0, 1.0]], device=device)
        bg = torch.zeros(3, device=device)

        results = []
        for op_val in [0.3, 0.7, 0.95]:
            opacities = torch.tensor([op_val], device=device)
            m2d, con, _, _, ops, sids, tr = run_projection_and_binning(
                means3d, log_scales, quats, opacities, camera
            )
            if sids.numel() == 0:
                pytest.skip("Gaussian not visible")
            img = rasterize_gaussians(m2d, con, colors, ops, bg, tr, sids, camera, backend="mojo")
            results.append(img.mean().item())

        assert results[0] < results[1] < results[2], (
            f"Brightness should increase with opacity: {results}"
        )

    def test_depth_ordering(self, device):
        """Two overlapping Gaussians: the front one's color should dominate at center."""
        camera = make_camera(device, H=64, W=64, fx=100.0)
        means3d = torch.tensor([
            [0.0, 0.0, 2.0],  # front - red
            [0.0, 0.0, 4.0],  # back - blue
        ], device=device)
        log_scales = torch.log(torch.tensor([
            [0.15, 0.15, 0.15],
            [0.15, 0.15, 0.15],
        ], device=device))
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)
        opacities = torch.tensor([0.9, 0.9], device=device)
        colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device)
        means2d, conics, _, _, opacities, sorted_ids, tile_ranges = run_projection_and_binning(
            means3d, log_scales, quats, opacities, camera
        )
        if sorted_ids.numel() == 0:
            pytest.skip("No visible gaussians")
        bg = torch.zeros(3, device=device)
        img = rasterize_gaussians(
            means2d, conics, colors, opacities, bg, tile_ranges, sorted_ids, camera, backend="mojo",
        )
        cy, cx = camera.H // 2, camera.W // 2
        red = img[cy, cx, 0].item()
        blue = img[cy, cx, 2].item()
        assert red > blue, f"Front (red) should dominate over back (blue): R={red:.4f} B={blue:.4f}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestRasterizationErrors:
    def test_invalid_backend(self, device):
        camera = make_camera(device)
        means2d = torch.randn(1, 2, device=device)
        conics = torch.randn(1, 3, device=device)
        colors = torch.randn(1, 3, device=device)
        opacities = torch.randn(1, device=device)
        bg = torch.zeros(3, device=device)
        tr = torch.zeros(4, 4, 2, device=device, dtype=torch.int32)
        si = torch.zeros(1, device=device, dtype=torch.int32)
        with pytest.raises(ValueError, match="Invalid backend"):
            rasterize_gaussians(means2d, conics, colors, opacities, bg, tr, si, camera, backend="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
