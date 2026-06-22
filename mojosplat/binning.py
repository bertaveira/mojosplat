import torch
from torch import Tensor
import math
from pathlib import Path

from typing_extensions import Literal

from max.torch import graph_op
from max.graph import ops as graph_ops, TensorType as GraphTensorType, DeviceRef
from max.dtype import DType as MaxDType

mojo_kernels = Path(__file__).parent / "kernels"
_gpu = DeviceRef.GPU()

_isect_count_ops: dict = {}
_isect_write_ops: dict = {}


def _get_isect_count_op(tile_size: int):
    if tile_size in _isect_count_ops:
        return _isect_count_ops[tile_size]

    _N = "N"

    @graph_op(
        name=f"isect_count_ts{tile_size}",
        kernel_library=mojo_kernels,
        input_types=[
            # "Output" passed as input (written in-place via rebind)
            GraphTensorType(MaxDType.int32, (_N,), device=_gpu),      # tiles_per_gauss
            # Regular inputs
            GraphTensorType(MaxDType.float32, (_N, 2), device=_gpu),  # means2d
            GraphTensorType(MaxDType.int32,   (_N, 2), device=_gpu),  # radii
            GraphTensorType(MaxDType.int32,   (2,),    device=_gpu),  # tile_dims
        ],
        output_types=[
            GraphTensorType(MaxDType.float32, (1,), device=_gpu),  # dummy to prevent DCE
        ],
    )
    def _graph(tiles_per_gauss, means2d, radii, tile_dims):
        return graph_ops.custom(
            "isect_count",
            _gpu,
            [tiles_per_gauss, means2d, radii, tile_dims],
            out_types=[GraphTensorType(MaxDType.float32, (1,), device=_gpu)],
            parameters={"tile_size": tile_size},
        )

    _isect_count_ops[tile_size] = _graph
    return _graph


def _get_isect_write_op(tile_size: int):
    if tile_size in _isect_write_ops:
        return _isect_write_ops[tile_size]

    _N = "N"
    _M = "M"

    @graph_op(
        name=f"isect_write_ts{tile_size}",
        kernel_library=mojo_kernels,
        input_types=[
            # "Outputs" passed as inputs (written in-place via rebind)
            GraphTensorType(MaxDType.int64, (_M,), device=_gpu),      # isect_ids
            GraphTensorType(MaxDType.int32, (_M,), device=_gpu),      # flatten_ids
            # Regular inputs
            GraphTensorType(MaxDType.float32, (_N, 2), device=_gpu),  # means2d
            GraphTensorType(MaxDType.int32,   (_N, 2), device=_gpu),  # radii
            GraphTensorType(MaxDType.float32, (_N,),   device=_gpu),  # depths
            GraphTensorType(MaxDType.int32,   (_N,),   device=_gpu),  # offsets
            GraphTensorType(MaxDType.int32,   (2,),    device=_gpu),  # tile_dims
        ],
        output_types=[
            GraphTensorType(MaxDType.float32, (1,), device=_gpu),  # dummy to prevent DCE
        ],
    )
    def _graph(isect_ids, flatten_ids, means2d, radii, depths, offsets, tile_dims):
        return graph_ops.custom(
            "isect_write",
            _gpu,
            [isect_ids, flatten_ids, means2d, radii, depths, offsets, tile_dims],
            out_types=[
                GraphTensorType(MaxDType.float32, (1,), device=_gpu),
            ],
            parameters={"tile_size": tile_size},
        )

    _isect_write_ops[tile_size] = _graph
    return _graph


def bin_gaussians_to_tiles(
    means2d: Tensor,  # [N, 2]
    radii: Tensor,  # [N, 2]
    depths: Tensor,  # [N]
    img_height: int,
    img_width: int,
    tile_size: int,
    backend: Literal["torch", "gsplat", "mojo"] = "gsplat",
) -> tuple:
    """Bin Gaussians to tiles.
    
    Args:
        means2d: [N, 2]
        radii: [N, 2]
        depths: [N]
        tile_size: int
    """
    N = means2d.shape[0]
    n_tiles_h = math.ceil(img_height / tile_size)
    n_tiles_w = math.ceil(img_width / tile_size)
    n_tiles = n_tiles_h * n_tiles_w

    if backend == "torch":
        return bin_gaussians_to_tiles_torch(means2d, radii, depths, img_height, img_width, tile_size)
    elif backend == "gsplat":
        return bin_gaussians_to_tiles_gsplat(means2d, radii, depths, tile_size, n_tiles_w, n_tiles_h)
    elif backend == "mojo":
        return bin_gaussians_to_tiles_mojo(means2d, radii, depths, tile_size, n_tiles_w, n_tiles_h)
    else:
        raise ValueError(f"Invalid backend: {backend}")



def bin_gaussians_to_tiles_gsplat(
    means2d: Tensor,  # [N, 2]
    radii: Tensor,  # [N, 2]
    depths: Tensor,  # [N]
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> tuple:
    """Bin Gaussians to tiles.
    
    Args:
        means2d: [N, 2]
        radii: [N, 2]
        depths: [N]
        tile_size: int
        tile_width: int
        tile_height: int

    Returns:
        sorted_gaussian_indices: [M,]
        tile_pointers: [n_tiles+1,]
        tile_ranges: [n_tiles, 2]
    """
    from gsplat import isect_tiles, isect_offset_encode

    means2d = means2d.unsqueeze(0).contiguous()
    radii = radii.unsqueeze(0).contiguous()
    depths = depths.unsqueeze(0).contiguous()
    segmented = False
    packed = False

    I = 1  # Single image
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        segmented=segmented,
        packed=packed,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height).squeeze(0)
    
    # Convert to compatible format: sorted_gaussian_indices and tile_ranges
    sorted_gaussian_indices = flatten_ids.squeeze(0) if flatten_ids.dim() > 1 else flatten_ids
    start_offsets = isect_offsets.view(tile_height, tile_width)  # (n_tiles_h, n_tiles_w)
    end_offsets = torch.zeros_like(start_offsets)
    
    # Flatten for easier computation
    start_flat = start_offsets.view(-1)
    end_flat = end_offsets.view(-1)
    
    # End of each tile is the start of the next tile
    end_flat[:-1] = start_flat[1:]
    end_flat[-1] = sorted_gaussian_indices.shape[0]  # Last tile ends at total count
    
    tile_ranges = torch.stack([start_flat, end_flat], dim=-1)
    tile_ranges = tile_ranges.view(tile_height, tile_width, 2)

    return sorted_gaussian_indices, tile_ranges



################################################################################

def bin_gaussians_to_tiles_torch(
    means2d: torch.Tensor, # (N, 2) Pixel coordinates
    radii: torch.Tensor,   # (N, 2) Pixel radius
    depths: torch.Tensor,  # (N,) Camera-space Z depths (used for initial sorting)
    img_height: int,
    img_width: int,
    tile_size: int,
) -> tuple:
    """Assigns Gaussians to overlapping screen tiles.

    Args:
        means2d: Projected 2D means in pixel coordinates.
        radii: Estimated radius of Gaussians in pixels.
        depths: Camera-space Z depths for sorting.
        img_height: Height of the image in pixels.
        img_width: Width of the image in pixels.
        tile_size: Size of square tiles in pixels (e.g., 16).

    Returns:
        A tuple containing:
        - sorted_gaussian_indices: (M,) Tensor of Gaussian indices sorted by tile_id and then depth.
        - tile_ranges: (tile_height, tile_width, 2) Start and end pointers for each tile.
    """
    N = means2d.shape[0]
    device = means2d.device

    n_tiles_h = math.ceil(img_height / tile_size)
    n_tiles_w = math.ceil(img_width / tile_size)
    n_tiles = n_tiles_h * n_tiles_w

    # --- 1. Calculate Gaussian bounding boxes ---
    min_x = means2d[:, 0] - radii[:, 0]
    max_x = means2d[:, 0] + radii[:, 0]
    min_y = means2d[:, 1] - radii[:, 1]
    max_y = means2d[:, 1] + radii[:, 1]

    # --- 2. Determine tile overlap ranges ---
    # Clamp bounding box to image bounds
    min_x = torch.clamp(min_x, 0, img_width - 1)
    max_x = torch.clamp(max_x, 0, img_width - 1)
    min_y = torch.clamp(min_y, 0, img_height - 1)
    max_y = torch.clamp(max_y, 0, img_height - 1)

    # Convert pixel coordinates to tile coordinates
    min_tile_x = (min_x / tile_size).to(torch.int32)
    max_tile_x = (max_x / tile_size).to(torch.int32)
    min_tile_y = (min_y / tile_size).to(torch.int32)
    max_tile_y = (max_y / tile_size).to(torch.int32)

    # --- 3. Generate (gaussian_idx, tile_id, depth) pairs --- 
    # Create indices for each Gaussian
    gaussian_indices = torch.arange(N, device=device)

    # Calculate number of tiles each gaussian overlaps (approximate for allocation)
    num_tiles_per_gaussian = (max_tile_x - min_tile_x + 1) * (max_tile_y - min_tile_y + 1)
    total_overlaps_approx = num_tiles_per_gaussian.sum()

    # Allocate buffers (use approximation, might need adjustment)
    overlap_gaussian_indices = torch.empty(total_overlaps_approx, dtype=torch.int32, device=device)
    overlap_tile_ids = torch.empty(total_overlaps_approx, dtype=torch.int32, device=device)
    overlap_depths = torch.empty(total_overlaps_approx, dtype=means2d.dtype, device=device)

    # Fill buffers (this loop is slow in Python, ideally done in CUDA/Triton if it becomes bottleneck)
    current_idx = 0
    for i in range(N):
        # Get potentially unclamped tile coords
        gx_min_unclamped = min_tile_x[i]
        gx_max_unclamped = max_tile_x[i]
        gy_min_unclamped = min_tile_y[i]
        gy_max_unclamped = max_tile_y[i]
        depth_i = depths[i]

        # Clamp tile coordinates to valid range [0, n_tiles_w/h - 1]
        gx_min = torch.clamp(gx_min_unclamped, 0, n_tiles_w - 1)
        gx_max = torch.clamp(gx_max_unclamped, 0, n_tiles_w - 1)
        gy_min = torch.clamp(gy_min_unclamped, 0, n_tiles_h - 1)
        gy_max = torch.clamp(gy_max_unclamped, 0, n_tiles_h - 1)

        # Use clamped indices in the loop ranges
        # Convert tensor bounds to Python ints for range()
        gy_min_int = gy_min.item()
        gy_max_int = gy_max.item()
        gx_min_int = gx_min.item()
        gx_max_int = gx_max.item()

        for ty in range(gy_min_int, gy_max_int + 1):
            for tx in range(gx_min_int, gx_max_int + 1):
                if current_idx < total_overlaps_approx: # Basic bounds check
                    tile_id = ty * n_tiles_w + tx
                    overlap_gaussian_indices[current_idx] = i
                    overlap_tile_ids[current_idx] = tile_id
                    overlap_depths[current_idx] = depth_i
                    current_idx += 1
                else:
                    # This indicates our approximation was too small, handle error or resize
                    print(f"WARN: Exceeded estimated overlap buffer size ({total_overlaps_approx}).")
                    # For now, we'll just stop filling, leading to potentially missed Gaussians.
                    # A robust implementation would reallocate or use a better estimation.
                    break
            else: # Continue if inner loop wasn't broken
                continue
            break # Break outer loop if inner loop was broken

    # Trim buffers to actual size
    actual_overlaps = current_idx
    overlap_gaussian_indices = overlap_gaussian_indices[:actual_overlaps]
    overlap_tile_ids = overlap_tile_ids[:actual_overlaps]
    overlap_depths = overlap_depths[:actual_overlaps]

    # --- 4. Sort by tile_id, then depth --- 
    # Sort primarily by tile_id, secondarily by depth (front-to-back)
    sort_key_depths = overlap_depths
    sort_key_tiles = overlap_tile_ids

    # Argsort by depth
    perm_depth = torch.argsort(sort_key_depths)
    overlap_gaussian_indices = overlap_gaussian_indices[perm_depth]
    overlap_tile_ids = overlap_tile_ids[perm_depth]
    # overlap_depths = overlap_depths[perm_depth] # Keep depths sorted for potential future use

    # Argsort by tile_id (stable sort preserves depth order within each tile)
    perm_tile = torch.argsort(overlap_tile_ids, stable=True)
    sorted_gaussian_indices = overlap_gaussian_indices[perm_tile]
    sorted_tile_ids = overlap_tile_ids[perm_tile] # Keep this for computing pointers
    # sorted_depths = overlap_depths[perm_tile] # If needed later

    # --- 5. Compute tile pointers --- 
    # Find where the tile_id changes in the sorted list (Not strictly needed for searchsorted approach, but good for understanding)
    # tile_change = torch.cat([
    #     torch.tensor([True], device=device), # Start is always a change
    #     sorted_tile_ids[1:] != sorted_tile_ids[:-1],
    #     torch.tensor([True], device=device) # End is always a change
    # ])
    # change_indices = torch.where(tile_change)[0] # Indices where tile_id changes
    # unique_tile_ids = sorted_tile_ids[change_indices[:-1]]

    # # Populate tile_pointers: the start index for tile_id is the index where it first appears
    # tile_pointers = torch.zeros(n_tiles + 1, dtype=torch.int32, device=device)
    # tile_pointers[unique_tile_ids] = change_indices[:-1].to(torch.int32)
    # # Correct Forward Fill for empty tiles (Removed)
    # # ... (removed forward fill code) ...

    # Efficiently compute tile_pointers using searchsorted
    # tile_pointers[i] will be the index of the first element in sorted_tile_ids >= i
    tile_pointers = torch.searchsorted(
        sorted_tile_ids,
        torch.arange(n_tiles + 1, device=device), # Bins [0, 1, ..., n_tiles]
        side='left' # Find first occurrence
    ).to(torch.int32) # Ensure output is int32

    # Create tile_ranges (n_tiles, 2) for easier kernel access
    tile_ranges = torch.stack([tile_pointers[:-1], tile_pointers[1:]], dim=-1)
    tile_ranges = tile_ranges.view(n_tiles_h, n_tiles_w, 2)

    return sorted_gaussian_indices, tile_ranges

def bin_gaussians_to_tiles_mojo(
    means2d: torch.Tensor,  # (N, 2) Pixel coordinates
    radii: torch.Tensor,    # (N, 2) Pixel radius
    depths: torch.Tensor,   # (N,) Camera-space Z depths
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> tuple:
    """Mojo GPU implementation of Gaussian-to-tile binning.

    Uses two Mojo kernels (IsectCount + IsectWrite) plus Python-side prefix sum,
    sort, and tile-range extraction.
    """
    N = means2d.shape[0]
    device = means2d.device
    n_tiles = tile_width * tile_height

    if N == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.zeros(tile_height, tile_width, 2, dtype=torch.int32, device=device),
        )

    means2d_c = means2d.contiguous()
    radii_i32 = radii.to(torch.int32).contiguous()
    depths_c = depths.contiguous()
    tile_dims = torch.tensor([tile_width, tile_height], dtype=torch.int32, device=device).contiguous()
    dummy = torch.empty(1, dtype=torch.float32, device=device)

    # Step 1: Count how many tiles each Gaussian overlaps
    # Kernel writes every element (0 for culled gaussians)
    tiles_per_gauss = torch.empty(N, dtype=torch.int32, device=device)
    _get_isect_count_op(tile_size)(dummy, tiles_per_gauss, means2d_c, radii_i32, tile_dims)

    # Step 2: Exclusive prefix sum → per-Gaussian write offsets
    cum_tiles = torch.cumsum(tiles_per_gauss, dim=0).to(torch.int32)
    M = int(cum_tiles[-1].item())

    # Slice assignment avoids torch.cat + extra allocation
    offsets = torch.empty(N, dtype=torch.int32, device=device)
    offsets[0] = 0
    offsets[1:] = cum_tiles[:-1]

    if M == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.zeros(tile_height, tile_width, 2, dtype=torch.int32, device=device),
        )

    # Step 3: Write (tile_id << 32 | depth_bits) and gaussian index per intersection
    isect_ids = torch.empty(M, dtype=torch.int64, device=device)
    flatten_ids = torch.empty(M, dtype=torch.int32, device=device)
    _get_isect_write_op(tile_size)(dummy, isect_ids, flatten_ids, means2d_c, radii_i32, depths_c, offsets, tile_dims)

    # Step 4: Sort by (tile_id, depth) — torch.sort returns both sorted values and indices
    sorted_keys, sort_perm = torch.sort(isect_ids)
    sorted_gaussian_indices = flatten_ids[sort_perm]

    # Step 5: Compute per-tile [start, end) ranges — single searchsorted with n_tiles+1
    tile_ids_sorted = (sorted_keys >> 32).to(torch.int32)
    tile_offsets = torch.searchsorted(
        tile_ids_sorted,
        torch.arange(n_tiles + 1, dtype=torch.int32, device=device),
    )
    tile_ranges = torch.stack(
        [tile_offsets[:-1], tile_offsets[1:]], dim=-1
    ).view(tile_height, tile_width, 2)

    return sorted_gaussian_indices.to(torch.int32), tile_ranges.to(torch.int32)