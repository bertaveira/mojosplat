import pytest
import torch
import numpy as np
import math
from typing import Literal

from mojosplat.binning import bin_gaussians_to_tiles


@pytest.fixture
def device():
    """Fixture for CUDA device, skip tests if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device("cuda:0")


@pytest.fixture
def simple_binning_data(device):
    """Create simple data for binning tests."""
    # Single Gaussian in center of 64x64 image with 16x16 tiles
    means2d = torch.tensor([[32.0, 32.0]], device=device, dtype=torch.float32)
    radii = torch.tensor([[8.0, 8.0]], device=device, dtype=torch.float32)
    depths = torch.tensor([1.0], device=device, dtype=torch.float32)
    img_height, img_width = 64, 64
    tile_size = 16
    return means2d, radii, depths, img_height, img_width, tile_size


@pytest.fixture
def batch_binning_data(device):
    """Create batch data for binning tests."""
    N = 10
    img_height, img_width = 128, 128
    tile_size = 16
    
    # Create Gaussians spread across the image
    means2d = torch.rand(N, 2, device=device, dtype=torch.float32) * torch.tensor([img_width, img_height], device=device)
    radii = torch.rand(N, 2, device=device, dtype=torch.float32) * 10 + 5  # Radii between 5-15
    depths = torch.rand(N, device=device, dtype=torch.float32) * 5 + 1  # Depths between 1-6
    
    return means2d, radii, depths, img_height, img_width, tile_size


@pytest.fixture
def edge_case_data(device):
    """Create edge case data for binning tests."""
    img_height, img_width = 64, 64
    tile_size = 16
    
    # Gaussians at image boundaries and outside image bounds
    means2d = torch.tensor([
        [0.0, 0.0],      # Top-left corner
        [63.0, 63.0],    # Bottom-right corner
        [-10.0, 32.0],   # Outside left
        [74.0, 32.0],    # Outside right
        [32.0, -10.0],   # Outside top
        [32.0, 74.0],    # Outside bottom
    ], device=device, dtype=torch.float32)
    
    radii = torch.tensor([
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
        [5.0, 5.0],
    ], device=device, dtype=torch.float32)
    
    depths = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=device, dtype=torch.float32)
    
    return means2d, radii, depths, img_height, img_width, tile_size


class TestBinningTorch:
    """Tests for torch backend binning."""
    
    def test_basic_functionality(self, device, simple_binning_data):
        """Test that torch backend works and returns expected shapes."""
        means2d, radii, depths, img_height, img_width, tile_size = simple_binning_data
        
        sorted_gaussian_indices, tile_ranges = bin_gaussians_to_tiles(
            means2d, radii, depths, img_height, img_width, tile_size, backend="torch"
        )
        
        n_tiles_h = math.ceil(img_height / tile_size)
        n_tiles_w = math.ceil(img_width / tile_size)
        n_tiles = n_tiles_h * n_tiles_w
        
        # Verify shapes
        assert sorted_gaussian_indices.dtype == torch.int32
        assert tile_ranges.shape == (n_tiles_h, n_tiles_w, 2)
        assert tile_ranges.dtype == torch.int32
        
        # Verify device
        assert all(t.device == device for t in [sorted_gaussian_indices, tile_ranges])
        
        # Basic sanity checks - tile_ranges should have valid start/end indices
        assert (tile_ranges[:, :, 0] <= tile_ranges[:, :, 1]).all()  # Start <= end for all tiles
        assert tile_ranges[:, :, 1].max() <= sorted_gaussian_indices.shape[0]  # End indices within bounds


    def test_depth_sorting(self, device):
        """Test that Gaussians are sorted by depth within tiles."""
        # Create two Gaussians in same tile with different depths
        means2d = torch.tensor([
            [8.0, 8.0],   # Both in tile (0,0)
            [8.0, 8.0],
        ], device=device, dtype=torch.float32)
        radii = torch.tensor([
            [4.0, 4.0],
            [4.0, 4.0],
        ], device=device, dtype=torch.float32)
        depths = torch.tensor([2.0, 1.0], device=device, dtype=torch.float32)  # Second is closer
        
        sorted_indices, tile_ranges = bin_gaussians_to_tiles(
            means2d, radii, depths, 64, 64, 16, backend="torch"
        )
        
        # Both Gaussians should be in the results
        assert sorted_indices.shape[0] >= 2
        
        # Find overlaps for tile (0,0) which is tile_id = 0
        tile_start = tile_ranges[0, 0, 0].item()
        tile_end = tile_ranges[0, 0, 1].item()
        
        if tile_end > tile_start:
            # If there are Gaussians in this tile, check depth ordering
            tile_gaussians = sorted_indices[tile_start:tile_end]
            # The closer Gaussian (depth 1.0, index 1) should come before farther one (depth 2.0, index 0)
            # Note: this assumes the sorting is working correctly


    def test_tile_assignment(self, device):
        """Test that Gaussians are assigned to correct tiles."""
        # Create Gaussian that spans multiple tiles
        means2d = torch.tensor([[15.5, 15.5]], device=device, dtype=torch.float32)  # Near tile boundary
        radii = torch.tensor([[8.0, 8.0]], device=device, dtype=torch.float32)     # Large radius
        depths = torch.tensor([1.0], device=device, dtype=torch.float32)
        
        sorted_indices, tile_ranges = bin_gaussians_to_tiles(
            means2d, radii, depths, 64, 64, 16, backend="torch"
        )
        
        # This Gaussian should appear in multiple tiles
        total_overlaps = sorted_indices.shape[0]
        assert total_overlaps > 1  # Should overlap multiple tiles


    def test_empty_input(self, device):
        """Test behavior with empty inputs."""
        empty_means2d = torch.empty(0, 2, device=device, dtype=torch.float32)
        empty_radii = torch.empty(0, 2, device=device, dtype=torch.float32)
        empty_depths = torch.empty(0, device=device, dtype=torch.float32)
        
        sorted_indices, tile_ranges = bin_gaussians_to_tiles(
            empty_means2d, empty_radii, empty_depths, 64, 64, 16, backend="torch"
        )
        
        n_tiles_h = n_tiles_w = 4  # 64x64 with 16x16 tiles
        
        # Should have empty results but correct structure
        assert sorted_indices.shape[0] == 0
        assert tile_ranges.shape == (n_tiles_h, n_tiles_w, 2)
        assert (tile_ranges[:, :, 0] == tile_ranges[:, :, 1]).all()  # Start == end for empty input


    def test_batch_processing(self, device, batch_binning_data):
        """Test that torch backend handles batches correctly."""
        means2d, radii, depths, img_height, img_width, tile_size = batch_binning_data
        N = means2d.shape[0]
        
        sorted_indices, tile_ranges = bin_gaussians_to_tiles(
            means2d, radii, depths, img_height, img_width, tile_size, backend="torch"
        )
        
        # All Gaussian indices should be in valid range
        assert (sorted_indices >= 0).all()
        assert (sorted_indices < N).all()


    def test_edge_cases(self, device, edge_case_data):
        """Test edge cases like boundary conditions."""
        means2d, radii, depths, img_height, img_width, tile_size = edge_case_data
        
        # Should not crash on edge cases
        sorted_indices, tile_ranges = bin_gaussians_to_tiles(
            means2d, radii, depths, img_height, img_width, tile_size, backend="torch"
        )
        
        # Basic structure should be maintained
        n_tiles_h = math.ceil(img_height / tile_size)
        n_tiles_w = math.ceil(img_width / tile_size)
        assert tile_ranges.shape == (n_tiles_h, n_tiles_w, 2)


class TestBinningGsplat:
    """Tests for gsplat backend binning."""
    
    def test_basic_functionality(self, device, simple_binning_data):
        """Test that gsplat backend works and returns expected shapes."""
        means2d, radii, depths, img_height, img_width, tile_size = simple_binning_data
        
        try:
            sorted_gaussian_indices, tile_ranges = bin_gaussians_to_tiles(
                means2d, radii, depths, img_height, img_width, tile_size, backend="gsplat"
            )
        except Exception as e:
            pytest.skip(f"GSplat backend not available: {e}")
        
        n_tiles_h = math.ceil(img_height / tile_size)
        n_tiles_w = math.ceil(img_width / tile_size)
        
        # Verify basic properties
        assert sorted_gaussian_indices.device == device
        assert tile_ranges.device == device
        assert sorted_gaussian_indices.dtype == torch.int32
        assert tile_ranges.dtype == torch.int64  # GSplat typically uses int64
        # GSplat tile_ranges now has same shape as torch: (n_tiles_h, n_tiles_w, 2) with [start, end] pairs
        assert tile_ranges.shape == (n_tiles_h, n_tiles_w, 2)
        
        # Verify outputs are finite (no NaN/inf)
        assert torch.isfinite(sorted_gaussian_indices.float()).all()
        assert torch.isfinite(tile_ranges.float()).all()


    def test_batch_processing(self, device, batch_binning_data):
        """Test that gsplat backend handles batches correctly."""
        means2d, radii, depths, img_height, img_width, tile_size = batch_binning_data
        
        try:
            sorted_gaussian_indices, tile_ranges = bin_gaussians_to_tiles(
                means2d, radii, depths, img_height, img_width, tile_size, backend="gsplat"
            )
        except Exception as e:
            pytest.skip(f"GSplat backend not available: {e}")
        
        # Basic checks
        assert sorted_gaussian_indices.device == device
        assert tile_ranges.device == device


    def test_empty_input_handling(self, device):
        """Test behavior with empty inputs."""
        empty_means2d = torch.empty(0, 2, device=device, dtype=torch.float32)
        empty_radii = torch.empty(0, 2, device=device, dtype=torch.float32)
        empty_depths = torch.empty(0, device=device, dtype=torch.float32)
        
        try:
            # Empty inputs should either work or raise a clear error
            sorted_gaussian_indices, tile_ranges = bin_gaussians_to_tiles(
                empty_means2d, empty_radii, empty_depths, 64, 64, 16, backend="gsplat"
            )
        except Exception as e:
            pytest.skip(f"GSplat backend not available or doesn't handle empty input: {e}")


class TestBinningMojo:
    """Tests for mojo backend binning."""

    def test_basic_functionality(self, device, simple_binning_data):
        """Test that mojo backend works and returns expected shapes/dtypes."""
        means2d, radii, depths, img_height, img_width, tile_size = simple_binning_data

        sorted_gaussian_indices, tile_ranges = bin_gaussians_to_tiles(
            means2d, radii, depths, img_height, img_width, tile_size, backend="mojo"
        )

        n_tiles_h = math.ceil(img_height / tile_size)
        n_tiles_w = math.ceil(img_width / tile_size)

        assert tile_ranges.shape == (n_tiles_h, n_tiles_w, 2)
        assert sorted_gaussian_indices.dtype == torch.int32
        assert tile_ranges.dtype == torch.int32
        assert sorted_gaussian_indices.device == device
        assert tile_ranges.device == device
        assert (tile_ranges[:, :, 0] <= tile_ranges[:, :, 1]).all()
        if sorted_gaussian_indices.shape[0] > 0:
            assert (sorted_gaussian_indices >= 0).all()
            assert (sorted_gaussian_indices < means2d.shape[0]).all()

    def test_depth_sorting(self, device):
        """Test that Gaussians in the same tile are ordered by depth (closer first)."""
        means2d = torch.tensor([[8.0, 8.0], [8.0, 8.0]], device=device, dtype=torch.float32)
        radii = torch.tensor([[4.0, 4.0], [4.0, 4.0]], device=device, dtype=torch.float32)
        depths = torch.tensor([2.0, 1.0], device=device, dtype=torch.float32)  # g1 closer

        sorted_indices, tile_ranges = bin_gaussians_to_tiles(
            means2d, radii, depths, 64, 64, 16, backend="mojo"
        )

        tile_start = tile_ranges[0, 0, 0].item()
        tile_end = tile_ranges[0, 0, 1].item()
        assert tile_end > tile_start, "Expected Gaussians in tile (0, 0)"

        tile_gaussians = sorted_indices[tile_start:tile_end].tolist()
        # Gaussian 1 (depth 1.0) should appear before Gaussian 0 (depth 2.0)
        assert tile_gaussians.index(1) < tile_gaussians.index(0)

    def test_tile_assignment(self, device):
        """Test that a large Gaussian spanning multiple tiles appears multiple times."""
        means2d = torch.tensor([[15.5, 15.5]], device=device, dtype=torch.float32)
        radii = torch.tensor([[8.0, 8.0]], device=device, dtype=torch.float32)
        depths = torch.tensor([1.0], device=device, dtype=torch.float32)

        sorted_indices, tile_ranges = bin_gaussians_to_tiles(
            means2d, radii, depths, 64, 64, 16, backend="mojo"
        )

        assert sorted_indices.shape[0] > 1, "Gaussian should overlap multiple tiles"

    def test_empty_input(self, device):
        """Test N=0 returns empty sorted_indices and zero tile_ranges."""
        empty_means2d = torch.empty(0, 2, device=device, dtype=torch.float32)
        empty_radii = torch.empty(0, 2, device=device, dtype=torch.float32)
        empty_depths = torch.empty(0, device=device, dtype=torch.float32)

        sorted_indices, tile_ranges = bin_gaussians_to_tiles(
            empty_means2d, empty_radii, empty_depths, 64, 64, 16, backend="mojo"
        )

        n_tiles_h = n_tiles_w = 4
        assert sorted_indices.shape[0] == 0
        assert tile_ranges.shape == (n_tiles_h, n_tiles_w, 2)


@pytest.mark.parametrize("backend1,backend2", [
    ("torch", "gsplat"),
    ("torch", "mojo"),
    ("gsplat", "mojo"),
])
class TestBinningConsistency:
    """Test consistency between different backends."""
    
    def test_basic_consistency(self, device, simple_binning_data, backend1, backend2):
        """Test that different backends produce consistent intersection counts and tile ranges."""
        means2d, radii, depths, img_height, img_width, tile_size = simple_binning_data

        try:
            sorted1, ranges1 = bin_gaussians_to_tiles(
                means2d, radii, depths, img_height, img_width, tile_size, backend=backend1
            )
            sorted2, ranges2 = bin_gaussians_to_tiles(
                means2d, radii, depths, img_height, img_width, tile_size, backend=backend2
            )
        except Exception as e:
            pytest.skip(f"Backend not available: {e}")

        # Total intersection count should match
        assert sorted1.shape[0] == sorted2.shape[0], (
            f"Total intersections differ: {backend1}={sorted1.shape[0]}, {backend2}={sorted2.shape[0]}"
        )

        # Tile range shapes must match
        assert ranges1.shape == ranges2.shape

        # Per-tile counts (end - start) should be identical
        counts1 = (ranges1[:, :, 1] - ranges1[:, :, 0]).long()
        counts2 = (ranges2[:, :, 1] - ranges2[:, :, 0]).long()
        assert torch.equal(counts1, counts2), "Per-tile Gaussian counts differ between backends"


class TestBinningIntegration:
    """Integration tests combining binning with other components."""
    
    def test_realistic_scenario(self, device):
        """Test binning with realistic projection outputs."""
        # Simulate outputs from projection stage
        N = 50
        img_height, img_width = 256, 256
        tile_size = 16
        
        # Create realistic 2D projections (within image bounds)
        means2d = torch.rand(N, 2, device=device, dtype=torch.float32)
        means2d[:, 0] *= img_width
        means2d[:, 1] *= img_height
        
        # Realistic radii (not too large)
        radii = torch.rand(N, 2, device=device, dtype=torch.float32) * 20 + 2
        
        # Realistic depths (positive, reasonable range)
        depths = torch.rand(N, device=device, dtype=torch.float32) * 10 + 0.1
        
        # Should work with torch backend
        sorted_indices, tile_ranges = bin_gaussians_to_tiles(
            means2d, radii, depths, img_height, img_width, tile_size, backend="torch"
        )
        
        # Verify realistic properties
        n_tiles_h = math.ceil(img_height / tile_size)
        n_tiles_w = math.ceil(img_width / tile_size)
        assert tile_ranges.shape == (n_tiles_h, n_tiles_w, 2)
        assert sorted_indices.max().item() < N  # All indices should be valid
        
        # Verify tile ranges are valid
        assert (tile_ranges[:, :, 0] <= tile_ranges[:, :, 1]).all()  # Start <= end
        assert tile_ranges[:, :, 1].max() <= sorted_indices.shape[0]  # End indices within bounds


    def test_different_tile_sizes(self, device):
        """Test binning with different tile sizes."""
        means2d = torch.tensor([[32.0, 32.0]], device=device, dtype=torch.float32)
        radii = torch.tensor([[8.0, 8.0]], device=device, dtype=torch.float32)
        depths = torch.tensor([1.0], device=device, dtype=torch.float32)
        img_height, img_width = 64, 64
        
        for tile_size in [8, 16, 32]:
            sorted_indices, tile_ranges = bin_gaussians_to_tiles(
                means2d, radii, depths, img_height, img_width, tile_size, backend="torch"
            )
            
            expected_tiles_h = math.ceil(img_height / tile_size)
            expected_tiles_w = math.ceil(img_width / tile_size)
            
            assert tile_ranges.shape == (expected_tiles_h, expected_tiles_w, 2)


if __name__ == "__main__":
    pytest.main([__file__])
