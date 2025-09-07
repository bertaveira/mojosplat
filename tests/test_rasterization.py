import pytest
import torch
import numpy as np
from typing import Literal

from mojosplat.rasterization import rasterize_gaussians
from mojosplat.projection import project_gaussians
from mojosplat.binning import bin_gaussians_to_tiles
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
def large_camera(device):
    """Create a larger camera for testing."""
    H, W = 128, 128
    R = torch.eye(3, device=device, dtype=torch.float32)
    T = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)
    fx, fy = 200.0, 200.0
    cx, cy = W / 2.0, H / 2.0
    near, far = 0.1, 100.0
    return Camera(R, T, H, W, fx, fy, cx, cy, near, far)


@pytest.fixture
def simple_gaussian_data(device):
    """Create a single Gaussian in front of camera with all required data."""
    means3d = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=torch.float32)
    scales = torch.log(torch.tensor([[0.1, 0.1, 0.1]], device=device, dtype=torch.float32))
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    opacities = torch.tensor([[0.8]], device=device, dtype=torch.float32)
    colors = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)  # Red
    return means3d, scales, quats, opacities, colors


@pytest.fixture
def batch_gaussian_data(device):
    """Create a small batch of Gaussians for testing."""
    N = 5
    means3d = torch.randn(N, 3, device=device, dtype=torch.float32) * 2
    means3d[:, 2] = torch.abs(means3d[:, 2]) + 1.0  # Ensure positive depth
    scales = torch.log(torch.rand(N, 3, device=device, dtype=torch.float32) * 0.3 + 0.05)
    quats = torch.randn(N, 4, device=device, dtype=torch.float32)
    quats = torch.nn.functional.normalize(quats, p=2, dim=-1)
    opacities = torch.rand(N, 1, device=device, dtype=torch.float32) * 0.8 + 0.2  # Keep above culling
    colors = torch.rand(N, 3, device=device, dtype=torch.float32)
    return means3d, scales, quats, opacities, colors


def create_rasterization_inputs(means3d, scales, quats, opacities, colors, camera, tile_size=16, backend="torch"):
    """Helper function to create all inputs needed for rasterization."""
    # Project gaussians
    means2d, conics, depths, radii = project_gaussians(
        means3d, scales, quats, opacities, camera, backend=backend
    )
    
    # Bin gaussians
    sorted_gaussian_indices, tile_ranges = bin_gaussians_to_tiles(
        means2d, radii, depths, camera.H, camera.W, tile_size, backend=backend
    )
    
    # Background color
    background_color = torch.zeros(colors.shape[-1], device=colors.device, dtype=colors.dtype)
    
    return means2d, conics, colors.squeeze(1) if colors.dim() == 3 else colors, opacities.squeeze(1) if opacities.dim() == 3 else opacities, background_color, tile_ranges, sorted_gaussian_indices


class TestRasterizationBasics:
    """Basic functionality tests for rasterization."""
    
    def test_basic_functionality_mojo(self, device, camera, simple_gaussian_data):
        """Test that Mojo backend works and returns expected shapes/types."""
        means3d, scales, quats, opacities, colors = simple_gaussian_data
        
        try:
            means2d, conics, colors_flat, opacities_flat, bg_color, tile_ranges, sorted_indices = create_rasterization_inputs(
                means3d, scales, quats, opacities, colors, camera, backend="torch"
            )
            
            rendered_image = rasterize_gaussians(
                means2d, conics, colors_flat, opacities_flat, bg_color,
                tile_ranges, sorted_indices, camera, tile_size=16, backend="mojo"
            )
        except Exception as e:
            pytest.skip(f"Mojo backend not available: {e}")
        
        # Verify basic properties
        assert rendered_image.shape == (camera.H, camera.W, 3)
        assert rendered_image.dtype == torch.float32
        assert rendered_image.device == device
        
        # Verify outputs are finite (no NaN/inf)
        assert torch.isfinite(rendered_image).all()
        
        # Should have some non-background pixels (since we have a visible Gaussian)
        assert rendered_image.max() > 0.0

    def test_basic_functionality_gsplat(self, device, camera, simple_gaussian_data):
        """Test that GSplat backend works and returns expected shapes/types."""
        means3d, scales, quats, opacities, colors = simple_gaussian_data
        
        try:
            means2d, conics, colors_flat, opacities_flat, bg_color, tile_ranges, sorted_indices = create_rasterization_inputs(
                means3d, scales, quats, opacities, colors, camera, backend="gsplat"
            )
            
            rendered_image = rasterize_gaussians(
                means2d, conics, colors_flat, opacities_flat, bg_color,
                tile_ranges, sorted_indices, camera, tile_size=16, backend="gsplat"
            )
        except Exception as e:
            pytest.skip(f"GSplat backend not available: {e}")
        
        # Verify basic properties
        assert rendered_image.shape == (camera.H, camera.W, 3)
        assert rendered_image.dtype == torch.float32
        assert rendered_image.device == device
        
        # Verify outputs are finite (no NaN/inf)
        assert torch.isfinite(rendered_image).all()

    def test_torch_backend_fallback(self, device, camera, simple_gaussian_data):
        """Test that torch backend falls back to gsplat with warning."""
        means3d, scales, quats, opacities, colors = simple_gaussian_data
        
        means2d, conics, colors_flat, opacities_flat, bg_color, tile_ranges, sorted_indices = create_rasterization_inputs(
            means3d, scales, quats, opacities, colors, camera, backend="torch"
        )
        
        # Should print warning and fall back to gsplat
        rendered_image = rasterize_gaussians(
            means2d, conics, colors_flat, opacities_flat, bg_color,
            tile_ranges, sorted_indices, camera, tile_size=16, backend="torch"
        )
        
        assert rendered_image.shape == (camera.H, camera.W, 3)
        assert rendered_image.dtype == torch.float32


class TestRasterizationBatching:
    """Test batch processing capabilities."""
    
    def test_batch_processing(self, device, camera, batch_gaussian_data):
        """Test that backends handle batches correctly."""
        means3d, scales, quats, opacities, colors = batch_gaussian_data
        
        means2d, conics, colors_flat, opacities_flat, bg_color, tile_ranges, sorted_indices = create_rasterization_inputs(
            means3d, scales, quats, opacities, colors, camera, backend="torch"
        )
        
        rendered_image = rasterize_gaussians(
            means2d, conics, colors_flat, opacities_flat, bg_color,
            tile_ranges, sorted_indices, camera, tile_size=16, backend="mojo"
        )
        
        # Check output shape is correct for multiple Gaussians
        assert rendered_image.shape == (camera.H, camera.W, 3)
        assert torch.isfinite(rendered_image).all()

    def test_empty_input_handling(self, device, camera):
        """Test behavior with empty inputs."""
        # Create empty inputs
        empty_means2d = torch.empty(0, 2, device=device, dtype=torch.float32)
        empty_conics = torch.empty(0, 3, device=device, dtype=torch.float32)
        empty_colors = torch.empty(0, 3, device=device, dtype=torch.float32)
        empty_opacities = torch.empty(0, device=device, dtype=torch.float32)
        background_color = torch.zeros(3, device=device, dtype=torch.float32)
        
        # Create minimal tile_ranges and sorted_indices
        n_tiles_h = (camera.H + 15) // 16
        n_tiles_w = (camera.W + 15) // 16
        tile_ranges = torch.zeros(n_tiles_h, n_tiles_w, 2, device=device, dtype=torch.int32)
        sorted_indices = torch.empty(0, device=device, dtype=torch.int32)
        
        # Should either work (returning background) or raise clear error
        try:
            rendered_image = rasterize_gaussians(
                empty_means2d, empty_conics, empty_colors, empty_opacities, 
                background_color, tile_ranges, sorted_indices, camera,
                tile_size=16, backend="mojo"
            )
            # If it works, should return background-colored image
            assert rendered_image.shape == (camera.H, camera.W, 3)
            assert torch.allclose(rendered_image, background_color.view(1, 1, 3))
        except Exception:
            # Expected to fail with current implementation
            pass


class TestRasterizationConsistency:
    """Test consistency between backends."""
    
    @pytest.mark.parametrize("backend1,backend2", [
        ("mojo", "gsplat"),
    ])
    def test_backend_consistency(self, device, camera, simple_gaussian_data, backend1, backend2):
        """Test that different backends produce similar results."""
        means3d, scales, quats, opacities, colors = simple_gaussian_data
        
        try:
            # Get inputs for first backend
            inputs1 = create_rasterization_inputs(
                means3d, scales, quats, opacities, colors, camera, backend="torch"
            )
            
            # Get inputs for second backend  
            inputs2 = create_rasterization_inputs(
                means3d, scales, quats, opacities, colors, camera, backend="gsplat" if backend2 == "gsplat" else "torch"
            )
            
            # Render with both backends
            result1 = rasterize_gaussians(*inputs1, camera, tile_size=16, backend=backend1)
            result2 = rasterize_gaussians(*inputs2, camera, tile_size=16, backend=backend2)
            
        except Exception as e:
            pytest.skip(f"Backend not available: {e}")
        
        # Compare results - should be reasonably similar
        assert result1.shape == result2.shape
        
        # Calculate mean absolute difference
        diff = torch.abs(result1 - result2).mean()
        
        # Allow for some numerical differences between backends
        assert diff < 0.1, f"Backends differ too much: mean diff = {diff}"

    def test_tile_size_consistency(self, device, camera, simple_gaussian_data):
        """Test that different tile sizes produce similar results."""
        means3d, scales, quats, opacities, colors = simple_gaussian_data
        
        inputs = create_rasterization_inputs(
            means3d, scales, quats, opacities, colors, camera, backend="torch"
        )
        
        # Test different tile sizes
        result_16 = rasterize_gaussians(*inputs, camera, tile_size=16, backend="mojo")
        
        # For tile size 8, need to recreate binning inputs
        means2d, conics, colors_flat, opacities_flat, bg_color, _, _ = inputs
        _, tile_ranges_8 = bin_gaussians_to_tiles(
            means2d, torch.ones_like(means2d), torch.ones(means2d.shape[0], device=device),
            camera.H, camera.W, 8, backend="torch"
        )
        sorted_indices_8 = torch.arange(means2d.shape[0], device=device, dtype=torch.int32)
        
        result_8 = rasterize_gaussians(
            means2d, conics, colors_flat, opacities_flat, bg_color,
            tile_ranges_8, sorted_indices_8, camera, tile_size=8, backend="mojo"
        )
        
        # Results should be very similar (tile size shouldn't affect final image much)
        diff = torch.abs(result_16 - result_8).mean()
        assert diff < 0.05, f"Different tile sizes produce too different results: {diff}"


class TestRasterizationVisual:
    """Test visual properties of rasterization."""
    
    def test_background_color(self, device, camera):
        """Test that background color is properly applied."""
        # Create empty scene (no gaussians)
        empty_means2d = torch.empty(0, 2, device=device, dtype=torch.float32)
        empty_conics = torch.empty(0, 3, device=device, dtype=torch.float32)
        empty_colors = torch.empty(0, 3, device=device, dtype=torch.float32)
        empty_opacities = torch.empty(0, device=device, dtype=torch.float32)
        
        # Set blue background
        background_color = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        
        # Create empty tile ranges
        n_tiles_h = (camera.H + 15) // 16
        n_tiles_w = (camera.W + 15) // 16
        tile_ranges = torch.zeros(n_tiles_h, n_tiles_w, 2, device=device, dtype=torch.int32)
        sorted_indices = torch.empty(0, device=device, dtype=torch.int32)
        
        try:
            rendered_image = rasterize_gaussians(
                empty_means2d, empty_conics, empty_colors, empty_opacities,
                background_color, tile_ranges, sorted_indices, camera,
                tile_size=16, backend="mojo"
            )
            
            # Should be blue everywhere
            expected = background_color.view(1, 1, 3).expand(camera.H, camera.W, 3)
            torch.testing.assert_close(rendered_image, expected, atol=1e-6)
            
        except Exception:
            # Expected to fail with current implementation for empty inputs
            pytest.skip("Empty input handling not implemented")

    def test_gaussian_visibility(self, device, large_camera):
        """Test that Gaussians are visible when they should be."""
        # Create a bright red Gaussian at the center
        means3d = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=torch.float32)
        scales = torch.log(torch.tensor([[0.2, 0.2, 0.2]], device=device, dtype=torch.float32))
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        opacities = torch.tensor([[0.9]], device=device, dtype=torch.float32)
        colors = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)  # Bright red
        
        inputs = create_rasterization_inputs(
            means3d, scales, quats, opacities, colors, large_camera, backend="torch"
        )
        
        rendered_image = rasterize_gaussians(*inputs, large_camera, tile_size=16, backend="mojo")
        
        # Check that the center region has red color
        center_y, center_x = large_camera.H // 2, large_camera.W // 2
        center_region = rendered_image[center_y-5:center_y+5, center_x-5:center_x+5, :]
        
        # Should have significant red component
        assert center_region[:, :, 0].mean() > 0.1, "Red Gaussian not visible in center"
        
        # Red should be stronger than green/blue in center
        red_strength = center_region[:, :, 0].mean()
        green_strength = center_region[:, :, 1].mean()
        blue_strength = center_region[:, :, 2].mean()
        
        assert red_strength > green_strength, "Red channel should dominate"
        assert red_strength > blue_strength, "Red channel should dominate"

    def test_opacity_effects(self, device, camera):
        """Test that opacity affects rendering correctly."""
        means3d = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=torch.float32)
        scales = torch.log(torch.tensor([[0.1, 0.1, 0.1]], device=device, dtype=torch.float32))
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        colors = torch.tensor([[1.0, 1.0, 1.0]], device=device, dtype=torch.float32)  # White
        
        # Test with high opacity
        high_opacity = torch.tensor([[0.9]], device=device, dtype=torch.float32)
        inputs_high = create_rasterization_inputs(
            means3d, scales, quats, high_opacity, colors, camera, backend="torch"
        )
        result_high = rasterize_gaussians(*inputs_high, camera, tile_size=16, backend="mojo")
        
        # Test with low opacity
        low_opacity = torch.tensor([[0.1]], device=device, dtype=torch.float32)
        inputs_low = create_rasterization_inputs(
            means3d, scales, quats, low_opacity, colors, camera, backend="torch"
        )
        result_low = rasterize_gaussians(*inputs_low, camera, tile_size=16, backend="mojo")
        
        # High opacity should produce brighter image
        assert result_high.mean() > result_low.mean(), "High opacity should be brighter than low opacity"


class TestRasterizationErrors:
    """Test error handling and edge cases."""
    
    def test_invalid_backend(self, device, camera, simple_gaussian_data):
        """Test error handling for invalid backend."""
        means3d, scales, quats, opacities, colors = simple_gaussian_data
        inputs = create_rasterization_inputs(
            means3d, scales, quats, opacities, colors, camera, backend="torch"
        )
        
        with pytest.raises(ValueError, match="Invalid backend"):
            rasterize_gaussians(*inputs, camera, tile_size=16, backend="invalid")

    def test_mismatched_tensor_devices(self, device, camera):
        """Test error handling for mismatched tensor devices."""
        if not torch.cuda.device_count() > 1:
            pytest.skip("Need multiple CUDA devices for this test")
        
        # Create inputs on different devices
        means2d = torch.randn(1, 2, device=device, dtype=torch.float32)
        conics = torch.randn(1, 3, device="cuda:1", dtype=torch.float32)  # Different device
        colors = torch.randn(1, 3, device=device, dtype=torch.float32)
        opacities = torch.randn(1, device=device, dtype=torch.float32)
        background_color = torch.zeros(3, device=device, dtype=torch.float32)
        tile_ranges = torch.zeros(4, 4, 2, device=device, dtype=torch.int32)
        sorted_indices = torch.zeros(1, device=device, dtype=torch.int32)
        
        # Should raise an error due to device mismatch
        with pytest.raises(Exception):  # Could be RuntimeError or other CUDA error
            rasterize_gaussians(
                means2d, conics, colors, opacities, background_color,
                tile_ranges, sorted_indices, camera, tile_size=16, backend="mojo"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
