#!/usr/bin/env python3
"""Generate golden reference data for the pure-Mojo test-suite.

This script was run once on the last torch-based version of MojoSplat
(branch `stabilize-modular-26.2`, modular==26.2.0), whose mojo backend matches
gsplat. It writes raw little-endian binary files that the Mojo tests read back
without needing Python, torch or gsplat:

    tests/data/golden/<scene>/
        meta.txt          key=value lines (N, M, W, H, tile_size, ...)
        camera.f32        [16 view matrix (row-major)] + [fx, fy, cx, cy, near, far]
        background.f32    [3]
        means3d.f32       [N, 3]
        log_scales.f32    [N, 3]   (log-space, as stored in .splat-derived data)
        quats.f32         [N, 4]   (w, x, y, z), not normalized
        opacities.f32     [N]      (already activated, in [0, 1])
        colors.f32        [N, 3]
        means2d.f32       [N, 2]   projection outputs (only valid where radii > 0)
        conics.f32        [N, 3]
        depths.f32        [N]
        radii.i32         [N, 2]
        flatten_ids.i32   [M]      gaussian ids sorted by (tile, depth)
        tile_ranges.i32   [TH, TW, 2]
        image.f32         [H, W, 3]  mojo backend render
        image_gsplat.f32  [H, W, 3]  gsplat render (cross-check), if available

Usage (from the repo root, in the 26.2 torch environment):
    uv run python tests/data/make_golden.py
"""

import math
from pathlib import Path

import numpy as np
import torch

from mojosplat.binning import bin_gaussians_to_tiles
from mojosplat.projection import project_gaussians
from mojosplat.rasterization import rasterize_gaussians
from mojosplat.utils import Camera

OUT = Path(__file__).parent / "golden"
DEV = torch.device("cuda")


def rot_xyz(ax, ay, az):
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).float()


def make_scene(N, W, H, seed, rot=(0.0, 0.0, 0.0), extra_cull=False):
    g = torch.Generator().manual_seed(seed)
    R = rot_xyz(*rot)
    T = torch.tensor([0.1, -0.2, 0.3])
    # Sample camera-space points mostly inside the frustum, then map to world.
    z = torch.rand(N, generator=g) * 9.0 + 1.0
    x = (torch.rand(N, generator=g) - 0.5) * 1.4 * z * (W / max(W, H))
    y = (torch.rand(N, generator=g) - 0.5) * 1.4 * z * (H / max(W, H))
    pc = torch.stack([x, y, z], dim=-1)
    if extra_cull:
        # A few behind the camera / beyond far / outside the image.
        pc[: N // 50, 2] = -pc[: N // 50, 2]
        pc[N // 50 : N // 25, 0] += 50.0
    means3d = (pc - T) @ R  # world = R^T (pc - T)
    log_scales = torch.log(torch.rand(N, 3, generator=g) * 0.08 + 0.005)
    quats = torch.randn(N, 4, generator=g)
    opacities = torch.rand(N, generator=g)
    if extra_cull:
        opacities[N // 25 : N // 20] = 0.001  # below 1/255 -> culled
    colors = torch.rand(N, 3, generator=g)
    fx = fy = 0.9 * max(W, H)
    cam = Camera(R.to(DEV), T.to(DEV), H, W, fx, fy, W / 2.0 + 0.3, H / 2.0 - 0.7, 0.1, 100.0)
    return dict(
        means3d=means3d.to(DEV), log_scales=log_scales.to(DEV), quats=quats.to(DEV),
        opacities=opacities.to(DEV), colors=colors.to(DEV), camera=cam,
    )


def save(d: Path, name: str, t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if name.endswith(".f32"):
        a = a.astype("<f4")
    elif name.endswith(".i32"):
        a = a.astype("<i4")
    a.tofile(d / name)


def run(name, scene, tile_size=16):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    cam: Camera = scene["camera"]
    bg = torch.tensor([0.1, 0.2, 0.3], device=DEV)

    means2d, conics, depths, radii = project_gaussians(
        scene["means3d"], scene["log_scales"], scene["quats"], scene["opacities"], cam, backend="mojo"
    )
    flatten_ids, tile_ranges = bin_gaussians_to_tiles(
        means2d, radii, depths, cam.H, cam.W, tile_size, backend="mojo"
    )
    image = rasterize_gaussians(
        means2d, conics, scene["colors"], scene["opacities"], bg, tile_ranges, flatten_ids,
        cam, tile_size=tile_size, backend="mojo",
    )

    # gsplat cross-check (full pipeline through gsplat backends)
    image_gs = None
    try:
        m2, c2, d2, r2 = project_gaussians(
            scene["means3d"], scene["log_scales"], scene["quats"], scene["opacities"], cam, backend="gsplat"
        )
        f2, t2 = bin_gaussians_to_tiles(m2, r2, d2, cam.H, cam.W, tile_size, backend="gsplat")
        image_gs = rasterize_gaussians(
            m2, c2, scene["colors"], scene["opacities"], bg, t2, f2, cam, tile_size=tile_size, backend="gsplat"
        )
    except Exception as e:  # pragma: no cover
        print(f"  gsplat cross-check unavailable: {e}")

    N = scene["means3d"].shape[0]
    M = flatten_ids.numel()
    th, tw = tile_ranges.shape[0], tile_ranges.shape[1]
    cam_vals = torch.cat([
        cam.view_matrix.reshape(-1).cpu(),
        torch.tensor([cam.fx, cam.fy, cam.cx, cam.cy, cam.near, cam.far]),
    ])
    save(d, "camera.f32", cam_vals)
    save(d, "background.f32", bg)
    for k in ["means3d", "log_scales", "quats", "opacities", "colors"]:
        save(d, f"{k}.f32", scene[k])
    valid = radii.min(dim=-1).values > 0
    # Zero out the undefined outputs of culled gaussians so files are deterministic.
    save(d, "means2d.f32", torch.where(valid[:, None], means2d, 0.0))
    save(d, "conics.f32", torch.where(valid[:, None], conics, 0.0))
    save(d, "depths.f32", torch.where(valid, depths, 0.0))
    save(d, "radii.i32", torch.where(valid[:, None], radii, 0))
    save(d, "flatten_ids.i32", flatten_ids)
    save(d, "tile_ranges.i32", tile_ranges)
    save(d, "image.f32", image)
    if image_gs is not None:
        save(d, "image_gsplat.f32", image_gs)
        diff = (image - image_gs).abs().max().item()
    else:
        diff = float("nan")
    (d / "meta.txt").write_text(
        f"N={N}\nM={M}\nW={cam.W}\nH={cam.H}\ntile_size={tile_size}\n"
        f"tiles_w={tw}\ntiles_h={th}\nvisible={int(valid.sum())}\n"
    )
    print(f"{name}: N={N} visible={int(valid.sum())} M={M} {cam.W}x{cam.H} "
          f"max|mojo-gsplat|={diff:.2e}")


if __name__ == "__main__":
    run("small", make_scene(500, 128, 96, seed=0))
    run("medium", make_scene(20000, 333, 250, seed=1, rot=(0.2, -0.3, 0.1), extra_cull=True))
