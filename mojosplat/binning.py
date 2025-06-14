# triton_splat/binning.py
import torch
import math

def bin_gaussians_to_tiles(
    means2d: torch.Tensor, # (N, 2) Pixel coordinates
    radii: torch.Tensor,   # (N,) Pixel radius
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
        - tile_pointers: (n_tiles+1,) Tensor where tile_pointers[i] is the start index
                         in sorted_gaussian_indices for tile i, and tile_pointers[i+1] is the end index.
                         The last element is the total number of overlaps M.
        - tile_ranges: (n_tiles, 2) Start and end pointers derived from tile_pointers.
    """
    N = means2d.shape[0]
    device = means2d.device

    n_tiles_h = math.ceil(img_height / tile_size)
    n_tiles_w = math.ceil(img_width / tile_size)
    n_tiles = n_tiles_h * n_tiles_w

    # --- 1. Calculate Gaussian bounding boxes ---
    min_x = means2d[:, 0] - radii
    max_x = means2d[:, 0] + radii
    min_y = means2d[:, 1] - radii
    max_y = means2d[:, 1] + radii

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

    return sorted_gaussian_indices, tile_pointers, tile_ranges 