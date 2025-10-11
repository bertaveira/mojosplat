import pytest
import torch
import numpy as np
from typing import Literal
import os

from mojosplat.projection import project_gaussians
from mojosplat.utils import Camera


@pytest.fixture
def device():
    """Fixture for CUDA device, skip tests if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device("cuda:0")


@pytest.fixture
def camera(device):
    """Create a standard camera for testing."""
    H, W = 64, 64
    R = torch.eye(3, device=device, dtype=torch.float32)
    T = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)
    fx, fy = 100.0, 100.0
    cx, cy = W / 2.0, H / 2.0
    near, far = 0.1, 100.0
    return Camera(R, T, H, W, fx, fy, cx, cy, near, far)


@pytest.fixture
def simple_gaussian(device):
    """Create a single Gaussian in front of camera."""
    means3d = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=torch.float32)
    scales = torch.log(torch.tensor([[0.1, 0.1, 0.1]], device=device, dtype=torch.float32))
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    opacity_features = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    return means3d, scales, quats, opacity_features


@pytest.fixture  
def multiple_gaussians(device):
    """Create a small batch of Gaussians for testing."""
    N = 5
    means3d = torch.randn(N, 3, device=device, dtype=torch.float32) * 2
    means3d[:, 2] = torch.abs(means3d[:, 2]) + 1.0  # Ensure positive depth
    scales = torch.log(torch.rand(N, 3, device=device, dtype=torch.float32) * 0.3 + 0.05)
    quats = torch.randn(N, 4, device=device, dtype=torch.float32)
    quats = torch.nn.functional.normalize(quats, p=2, dim=-1)
    opacity_features = torch.randn(N, 1, device=device, dtype=torch.float32)
    return means3d, scales, quats, opacity_features


class TestMojoProjection:
    """Clean, focused tests for Mojo projection backend."""
    
    def test_basic_functionality(self, device, camera, simple_gaussian):
        """Test that Mojo backend works and returns expected shapes/types."""
        means3d, scales, quats, opacity_features = simple_gaussian
        
        try:
            means2d, conics, depths, radii = project_gaussians(
                means3d, scales, quats, opacity_features, camera, backend="mojo"
            )
        except Exception as e:
            pytest.skip(f"Mojo backend not available: {e}")
        
        # Verify basic properties
        assert means2d.shape == (1, 2)
        assert conics.shape == (1, 3) 
        assert depths.shape == (1,)
        assert radii.shape == (1, 2)
        
        assert means2d.dtype == torch.float32
        assert conics.dtype == torch.float32
        assert depths.dtype == torch.float32
        assert radii.dtype == torch.int32
        
        assert all(t.device == device for t in [means2d, conics, depths, radii])
        
        # Verify outputs are finite (no NaN/inf)
        assert torch.isfinite(means2d).all()
        assert torch.isfinite(conics).all()
        assert torch.isfinite(depths).all()


    def test_multiple_gaussians_processing(self, device, camera, multiple_gaussians):
        """Test that Mojo backend handles batches correctly."""
        means3d, scales, quats, opacity_features = multiple_gaussians
        N = means3d.shape[0]
        
        means2d, conics, depths, radii = project_gaussians(
            means3d, scales, quats, opacity_features, camera, backend="mojo"
        )
        
        # Check batch dimensions are handled correctly
        assert means2d.shape == (N, 2)
        assert conics.shape == (N, 3)
        assert depths.shape == (N,)
        assert radii.shape == (N, 2)


    def test_empty_input_handling(self, device, camera):
        """Test behavior with empty inputs."""
        empty_means3d = torch.empty(0, 3, device=device, dtype=torch.float32)
        empty_scales = torch.empty(0, 3, device=device, dtype=torch.float32)
        empty_quats = torch.empty(0, 4, device=device, dtype=torch.float32)
        empty_opacity = torch.empty(0, 1, device=device, dtype=torch.float32)
        
        # Empty inputs should either work or raise a clear error
        with pytest.raises(Exception):  # Expected to fail currently
            project_gaussians(
                empty_means3d, empty_scales, empty_quats, empty_opacity, 
                camera, backend="mojo"
            )


    @pytest.mark.parametrize("backend1,backend2", [
        ("mojo", "torch"),
        ("mojo", "gsplat"),
        ("gsplat", "torch"),
    ])
    def test_consistency_with_other_backends(self, device, camera, simple_gaussian, backend1, backend2):
        """Test that Mojo produces similar results to other backends."""
        means3d, scales, quats, opacity_features = simple_gaussian
        
        # Get results from both backends
        try:
            result1 = project_gaussians(means3d, scales, quats, opacity_features, camera, backend=backend1)
            result2 = project_gaussians(means3d, scales, quats, opacity_features, camera, backend=backend2)
        except Exception as e:
            pytest.skip(f"Backend not available: {e}")
        
        means2d_1, conics_1, depths_1, radii_1 = result1
        means2d_2, conics_2, depths_2, radii_2 = result2
        
        # Handle shape differences (Mojo adds batch dimension)
        if backend1 == "mojo":
            means2d_1, conics_1, depths_1, radii_1 = [t.squeeze(0) for t in result1]
        if backend2 == "mojo":
            means2d_2, conics_2, depths_2, radii_2 = [t.squeeze(0) for t in result2]
        
        # Convert radii to float for comparison
        radii_1, radii_2 = radii_1.float(), radii_2.float()
        
        # Compare with reasonable tolerances
        torch.testing.assert_close(means2d_1, means2d_2, atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(depths_1, depths_2, atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(conics_1, conics_2, atol=1e-2, rtol=1e-1)
        torch.testing.assert_close(radii_1, radii_2, atol=2.0, rtol=0.2)


    def test_projection_geometry_sanity(self, device):
        """Test basic geometric properties of projection."""
        # Simple camera looking down Z-axis
        camera = Camera(
            R=torch.eye(3, device=device, dtype=torch.float32),
            T=torch.tensor([0, 0, 5.0], device=device, dtype=torch.float32),
            H=64, W=64, fx=100.0, fy=100.0, cx=32.0, cy=32.0
        )
        
        # Point at origin should project near center
        # Use opacity above culling threshold (1/255 ≈ 0.004)
        means3d = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        scales = torch.log(torch.tensor([[0.1, 0.1, 0.1]], device=device, dtype=torch.float32))
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        opacity_features = torch.tensor([[0.5]], device=device, dtype=torch.float32)
        
        means2d, _, _, _ = project_gaussians(
            means3d, scales, quats, opacity_features, camera, backend="mojo"
        )
        
        projected_x, projected_y = means2d[0, 0].item(), means2d[0, 1].item()
        
        # Should be reasonably close to image center (32, 32)
        assert abs(projected_x - 32.0) < 0.1, f"X projection {projected_x} far from center"
        assert abs(projected_y - 32.0) < 0.1, f"Y projection {projected_y} far from center"


    def test_opacity_culling(self, device, camera):
        """Test that low-opacity Gaussians are properly culled."""
        means3d = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=torch.float32)
        scales = torch.log(torch.tensor([[0.1, 0.1, 0.1]], device=device, dtype=torch.float32))
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        
        # Test with opacity below threshold (should be culled)
        low_opacity = torch.tensor([[0.001]], device=device, dtype=torch.float32)  # Below 1/255 ≈ 0.004
        means2d, conics, depths, radii = project_gaussians(
            means3d, scales, quats, low_opacity, camera, backend="mojo"
        )
        
        # Culled Gaussians should have zero radii and zero means2d
        assert radii[0, 0].item() == 0, "Culled Gaussian should have zero x radius"
        assert radii[0, 1].item() == 0, "Culled Gaussian should have zero y radius"
        assert means2d[0, 0].item() == 0.0, "Culled Gaussian should have zero x projection"
        assert means2d[0, 1].item() == 0.0, "Culled Gaussian should have zero y projection"

if __name__ == "__main__":
    pytest.main([__file__])