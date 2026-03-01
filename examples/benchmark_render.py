#!/usr/bin/env python3
"""Benchmark full render pipeline and individual kernels: gsplat vs mojo.

Usage:
    uv run python examples/benchmark_render.py [path/to/scene.splat] [options]

Loads a .splat file (default: examples/bicycle.splat), warms up both backends,
then measures 1000 full-pipeline renders and reports FPS. Afterwards, each kernel
(projection, binning, rasterization) is benchmarked individually in ms.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from mojosplat.render import render_gaussians
from mojosplat.projection import project_gaussians
from mojosplat.binning import bin_gaussians_to_tiles
from mojosplat.rasterization import rasterize_gaussians
from mojosplat.utils import Camera, detect_device

TILE_SIZE = 16


def device_synchronize(device: torch.device):
    """Synchronize the given device (works across CUDA, MPS, etc.)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# .splat loader (antimatter15 binary format, 32 bytes/Gaussian)
# ---------------------------------------------------------------------------

def load_splat(path: str):
    data = Path(path).read_bytes()
    n = len(data) // 32
    if n == 0:
        raise ValueError(f"Empty or invalid .splat file: {path}")

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n, 32)

    positions = raw[:, 0:12].view(np.float32).reshape(n, 3).copy()
    scales_exp = raw[:, 12:24].view(np.float32).reshape(n, 3).copy()
    scales = np.log(np.clip(scales_exp, 1e-10, None))

    rgba = raw[:, 24:28].astype(np.float32) / 255.0
    rgb = rgba[:, :3]
    opacity = rgba[:, 3]

    rot_raw = raw[:, 28:32].astype(np.float32)
    quats = (rot_raw - 128.0) / 128.0
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(norms, 1e-10, None)

    return positions, scales, rgb, opacity, quats


def make_camera(device, H=720, W=1280):
    R = torch.eye(3, dtype=torch.float32, device=device)
    T = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32, device=device)
    return Camera(
        R=R, T=T, H=H, W=W,
        fx=float(W * 0.9), fy=float(W * 0.9),
        cx=W / 2.0, cy=H / 2.0,
    )


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def benchmark_full_pipeline(backend, means3d, scales, quats, opacities, features,
                            camera, bg, num_runs, device):
    """Time `num_runs` full render_gaussians calls; return list of per-frame seconds."""
    times = []
    for _ in range(num_runs):
        device_synchronize(device)
        t0 = time.perf_counter()
        render_gaussians(
            means3d, scales, quats, opacities, features,
            camera, background_color=bg, backend=backend,
        )
        device_synchronize(device)
        times.append(time.perf_counter() - t0)
    return times


def benchmark_projection(backend, means3d, scales, quats, opacities, camera, num_runs, device):
    times = []
    for _ in range(num_runs):
        device_synchronize(device)
        t0 = time.perf_counter()
        project_gaussians(means3d, scales, quats, opacities, camera, backend=backend)
        device_synchronize(device)
        times.append(time.perf_counter() - t0)
    return times


def benchmark_binning(backend, means2d, radii, depths, camera, num_runs, device):
    times = []
    for _ in range(num_runs):
        device_synchronize(device)
        t0 = time.perf_counter()
        bin_gaussians_to_tiles(means2d, radii, depths, camera.H, camera.W,
                               TILE_SIZE, backend=backend)
        device_synchronize(device)
        times.append(time.perf_counter() - t0)
    return times


def benchmark_rasterization(backend, means2d, conics, colors, opacities, bg,
                            tile_ranges, sorted_ids, camera, num_runs, device):
    times = []
    for _ in range(num_runs):
        device_synchronize(device)
        t0 = time.perf_counter()
        rasterize_gaussians(
            means2d, conics, colors, opacities, bg,
            tile_ranges, sorted_ids, camera, tile_size=TILE_SIZE, backend=backend,
        )
        device_synchronize(device)
        times.append(time.perf_counter() - t0)
    return times


def fmt_stats(times_s, label=""):
    """Return a formatted string with mean/std/min/max in ms and FPS."""
    arr = np.array(times_s)
    mean = arr.mean()
    std = arr.std()
    fps = 1.0 / mean if mean > 0 else float("inf")
    return (f"{label:<14}  mean {mean*1000:8.2f} ms  "
            f"std {std*1000:7.2f} ms  "
            f"min {arr.min()*1000:8.2f} ms  "
            f"max {arr.max()*1000:8.2f} ms  "
            f"({fps:7.1f} FPS)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark gsplat vs mojo full pipeline + kernels")
    parser.add_argument(
        "splat", nargs="?",
        default=str(Path(__file__).parent / "bicycle.splat"),
        help="Path to .splat file (default: examples/bicycle.splat)",
    )
    parser.add_argument("-W", "--width", type=int, default=1280)
    parser.add_argument("-H", "--height", type=int, default=720)
    parser.add_argument("-n", "--num-runs", type=int, default=1000,
                        help="Number of timed runs for full pipeline")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Warmup iterations before timing")
    parser.add_argument("--kernel-runs", type=int, default=200,
                        help="Number of timed runs per individual kernel")
    args = parser.parse_args()

    device = detect_device()
    if device.type == "cpu":
        raise RuntimeError("No GPU found. A CUDA, ROCm, or Apple Silicon GPU is required.")
    print(f"Using device: {device}")

    # -- Load scene -----------------------------------------------------------
    print(f"Loading {args.splat} ...")
    positions, scales_np, rgb_np, opacity_np, quats_np = load_splat(args.splat)
    N = len(positions)
    print(f"  {N:,} Gaussians")

    means3d  = torch.tensor(positions, dtype=torch.float32, device=device)
    scales   = torch.tensor(scales_np, dtype=torch.float32, device=device)
    quats    = torch.tensor(quats_np,  dtype=torch.float32, device=device)
    opacities = torch.tensor(opacity_np, dtype=torch.float32, device=device)
    features = torch.tensor(rgb_np, dtype=torch.float32, device=device)
    bg = torch.zeros(3, dtype=torch.float32, device=device)

    camera = make_camera(device, H=args.height, W=args.width)
    backends = ["gsplat", "mojo"]

    # -- Warmup (triggers JIT for mojo) ---------------------------------------
    for backend in backends:
        print(f"Warming up {backend} ({args.warmup} iters) ...")
        for _ in range(args.warmup):
            render_gaussians(means3d, scales, quats, opacities, features,
                             camera, background_color=bg, backend=backend)
            device_synchronize(device)
        print(f"  {backend} warm.")

    # =========================================================================
    # Full pipeline benchmark
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"FULL PIPELINE BENCHMARK  ({args.num_runs} runs, {args.width}x{args.height}, "
          f"{N:,} Gaussians)")
    print(f"{'='*80}")

    for backend in backends:
        times = benchmark_full_pipeline(
            backend, means3d, scales, quats, opacities, features,
            camera, bg, args.num_runs, device,
        )
        print(fmt_stats(times, backend))

    # =========================================================================
    # Per-kernel benchmarks
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"PER-KERNEL BENCHMARKS  ({args.kernel_runs} runs each)")
    print(f"{'='*80}")

    # Pre-compute intermediate tensors per backend for binning & rasterization.
    intermediate = {}
    with torch.no_grad():
        for backend in backends:
            means2d, conics, depths, radii = project_gaussians(
                means3d, scales, quats, opacities, camera, backend=backend)
            sorted_ids, tile_ranges = bin_gaussians_to_tiles(
                means2d, radii, depths, camera.H, camera.W, TILE_SIZE, backend=backend)
            intermediate[backend] = {
                "means2d": means2d, "conics": conics, "depths": depths,
                "radii": radii, "sorted_ids": sorted_ids, "tile_ranges": tile_ranges,
            }

    # -- Projection -----------------------------------------------------------
    print(f"\n  Projection:")
    for backend in backends:
        # warmup
        for _ in range(10):
            project_gaussians(means3d, scales, quats, opacities, camera, backend=backend)
            device_synchronize(device)
        times = benchmark_projection(backend, means3d, scales, quats, opacities,
                                     camera, args.kernel_runs, device)
        print(f"    {fmt_stats(times, backend)}")

    # -- Binning --------------------------------------------------------------
    print(f"\n  Binning:")
    for backend in backends:
        d = intermediate[backend]
        for _ in range(10):
            bin_gaussians_to_tiles(d["means2d"], d["radii"], d["depths"],
                                   camera.H, camera.W, TILE_SIZE, backend=backend)
            device_synchronize(device)
        times = benchmark_binning(backend, d["means2d"], d["radii"], d["depths"],
                                  camera, args.kernel_runs, device)
        print(f"    {fmt_stats(times, backend)}")

    # -- Rasterization --------------------------------------------------------
    print(f"\n  Rasterization:")
    for backend in backends:
        d = intermediate[backend]
        for _ in range(10):
            rasterize_gaussians(d["means2d"], d["conics"], features, opacities, bg,
                                d["tile_ranges"], d["sorted_ids"], camera,
                                tile_size=TILE_SIZE, backend=backend)
            device_synchronize(device)
        times = benchmark_rasterization(backend, d["means2d"], d["conics"],
                                        features, opacities, bg, d["tile_ranges"],
                                        d["sorted_ids"], camera, args.kernel_runs, device)
        print(f"    {fmt_stats(times, backend)}")

    print()


if __name__ == "__main__":
    main()
