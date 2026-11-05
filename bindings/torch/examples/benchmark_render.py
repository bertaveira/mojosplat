#!/usr/bin/env python3
"""Benchmark mojosplat vs gsplat through PyTorch: rendering and training step.

Usage:
    pixi run -e torch python bindings/torch/examples/benchmark_render.py [scene.splat] [options]

Loads a .splat file (default: examples/bicycle.splat) and times, for both
renderers on the same raw parameters (SH degree 0 from the .splat colors):

* a full frame (no gradients),
* a training step: forward + backward of L = sum(image * w) w.r.t. all raw
  parameters, through torch autograd.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from mojosplat_torch import Camera, load_splat, render_gsplat, render_splats


def time_calls(fn, num_runs, warmup=20):
    """Per-call seconds of fn(), with a device sync around each call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def fmt_stats(times_s, label=""):
    arr = np.array(times_s) * 1000
    return (f"{label:<10}  mean {arr.mean():7.2f} ms  median {np.median(arr):7.2f} ms  "
            f"min {arr.min():7.2f} ms  ({1000 / arr.mean():6.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark mojosplat vs gsplat (PyTorch)")
    parser.add_argument(
        "splat", nargs="?",
        default=str(Path(__file__).resolve().parents[3] / "examples" / "bicycle.splat"),
    )
    parser.add_argument("-W", "--width", type=int, default=1280)
    parser.add_argument("-H", "--height", type=int, default=720)
    parser.add_argument("-n", "--num-runs", type=int, default=500, help="Timed frames")
    parser.add_argument("--train-runs", type=int, default=100, help="Timed training steps")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    device = torch.device("cuda:0")

    print(f"Loading {args.splat} ...")
    params = load_splat(args.splat, device)
    n = params["means"].shape[0]
    W, H = args.width, args.height
    camera = Camera(
        R=torch.eye(3, device=device), T=torch.tensor([0.0, 0.0, 5.0], device=device),
        H=H, W=W, fx=0.9 * W, fy=0.9 * W, cx=W / 2.0, cy=H / 2.0,
    )
    bg = torch.zeros(3, device=device)
    renderers = {"mojo": render_splats, "gsplat": render_gsplat}
    print(f"  {n:,} Gaussians, {W}x{H}, {torch.cuda.get_device_name()}")

    print(f"\nFULL FRAME ({args.num_runs} runs)")
    for name, render in renderers.items():
        def frame():
            with torch.no_grad():
                render(**params, camera=camera, sh_degree=0, background=bg)
        print(fmt_stats(time_calls(frame, args.num_runs), name))

    print(f"\nTRAINING STEP: forward + backward ({args.train_runs} runs)")
    w = torch.rand(H, W, 3, device=device) * 2 - 1
    leaves = {k: (v.detach().clone().requires_grad_() if v is not None else None) for k, v in params.items()}
    for name, render in renderers.items():
        def step():
            img = render(**leaves, camera=camera, sh_degree=0, background=bg)
            (img * w).sum().backward()
        print(fmt_stats(time_calls(step, args.train_runs), name))


if __name__ == "__main__":
    main()
