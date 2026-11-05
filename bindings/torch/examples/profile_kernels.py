#!/usr/bin/env python3
"""Profile Mojo vs GSplat kernels to locate the 2x slowdown.

Usage:
    uv run python examples/profile_kernels.py [path/to/scene.splat]

Three output modes:
  1. CUDA-event per-kernel timing breakdown (always runs)
  2. torch.profiler trace written to /tmp/mojo_profile/ (open in Chrome chrome://tracing)
  3. Instructions for nsys / ncu commands to dig deeper
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from mojosplat_torch.projection import project_gaussians
from mojosplat_torch.binning import bin_gaussians_to_tiles
from mojosplat_torch.rasterization import rasterize_gaussians
from mojosplat_torch.utils import Camera

TILE_SIZE = 16
WARMUP = 20
TIMED_RUNS = 100


# ---------------------------------------------------------------------------
# .splat loader (antimatter15 binary format)
# ---------------------------------------------------------------------------

def load_splat(path: str):
    data = Path(path).read_bytes()
    n = len(data) // 32
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n, 32)
    positions = raw[:, 0:12].view(np.float32).reshape(n, 3).copy()
    scales_exp = raw[:, 12:24].view(np.float32).reshape(n, 3).copy()
    scales = np.log(np.clip(scales_exp, 1e-10, None))
    rgba = raw[:, 24:28].astype(np.float32) / 255.0
    opacity = rgba[:, 3]
    rgb = rgba[:, :3]
    rot_raw = raw[:, 28:32].astype(np.float32)
    quats = (rot_raw - 128.0) / 128.0
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(norms, 1e-10, None)
    return positions, scales, rgb, opacity, quats


def make_camera(device, H=720, W=1280):
    R = torch.eye(3, dtype=torch.float32, device=device)
    T = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32, device=device)
    return Camera(R=R, T=T, H=H, W=W, fx=float(W*0.9), fy=float(W*0.9), cx=W/2.0, cy=H/2.0)


# ---------------------------------------------------------------------------
# CUDA-event timing (eliminates Python-side measurement noise)
# ---------------------------------------------------------------------------

def cuda_event_time_kernel(fn, warmup=WARMUP, runs=TIMED_RUNS):
    """Run fn() with CUDA event timing. Returns (mean_ms, std_ms, min_ms) tuple."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]

    for i in range(runs):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    arr = np.array(times)
    return arr.mean(), arr.std(), arr.min()


def fmt(label, mean, std, mn):
    fps = 1000.0 / mean if mean > 0 else float("inf")
    return f"  {label:<22} mean {mean:7.3f} ms  std {std:6.3f} ms  min {mn:7.3f} ms  ({fps:7.1f} FPS)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("splat", nargs="?",
                        default=str(Path(__file__).resolve().parents[3] / "examples" / "bicycle.splat"))
    parser.add_argument("-W", "--width",  type=int, default=1280)
    parser.add_argument("-H", "--height", type=int, default=720)
    parser.add_argument("--no-torch-profiler", action="store_true",
                        help="Skip torch.profiler trace (faster)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")

    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading {args.splat}...")
    positions, scales_np, rgb_np, opacity_np, quats_np = load_splat(args.splat)
    N = len(positions)
    print(f"  {N:,} Gaussians  |  image {args.width}x{args.height}")

    means3d   = torch.tensor(positions,  dtype=torch.float32, device=device)
    scales    = torch.tensor(scales_np,  dtype=torch.float32, device=device)
    quats     = torch.tensor(quats_np,   dtype=torch.float32, device=device)
    opacities = torch.tensor(opacity_np, dtype=torch.float32, device=device)
    features  = torch.tensor(rgb_np,     dtype=torch.float32, device=device)
    bg        = torch.zeros(3, dtype=torch.float32, device=device)
    camera    = make_camera(device, H=args.height, W=args.width)

    # Precompute intermediate tensors using each backend
    print("\nWarming up / compiling kernels ...")
    for backend in ("gsplat", "mojo"):
        for _ in range(5):
            m2d, con, dep, rad = project_gaussians(means3d, scales, quats, opacities, camera, backend=backend)
            sids, tranges = bin_gaussians_to_tiles(m2d, rad, dep, camera.H, camera.W, TILE_SIZE, backend=backend)
            rasterize_gaussians(m2d, con, features, opacities, bg, tranges, sids, camera, TILE_SIZE, backend=backend)
        torch.cuda.synchronize()
        print(f"  {backend} warm.")

    # Precompute fixed intermediate tensors for isolated kernel benchmarks
    m2d_gs, con_gs, dep_gs, rad_gs = project_gaussians(means3d, scales, quats, opacities, camera, backend="gsplat")
    sids_gs, tr_gs = bin_gaussians_to_tiles(m2d_gs, rad_gs, dep_gs, camera.H, camera.W, TILE_SIZE, backend="gsplat")

    m2d_mo, con_mo, dep_mo, rad_mo = project_gaussians(means3d, scales, quats, opacities, camera, backend="mojo")
    sids_mo, tr_mo = bin_gaussians_to_tiles(m2d_mo, rad_mo, dep_mo, camera.H, camera.W, TILE_SIZE, backend="mojo")

    # Report intersection density (affects early-exit benefit)
    print(f"\n  GSplat intersections: {sids_gs.numel():,}")
    print(f"  Mojo   intersections: {sids_mo.numel():,}")
    pixels = args.width * args.height
    print(f"  Gaussians/pixel (approx): {sids_gs.numel() / pixels:.1f}")

    # =========================================================================
    # CUDA-event per-kernel timing
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"CUDA-EVENT TIMING  ({TIMED_RUNS} runs each, after {WARMUP} warmup)")
    print(f"{'='*70}")

    print("\n[Projection]")
    for backend, m3d_, s_, q_, o_ in [
        ("gsplat", means3d, scales, quats, opacities),
        ("mojo",   means3d, scales, quats, opacities),
    ]:
        m, s, mn = cuda_event_time_kernel(
            lambda b=backend: project_gaussians(m3d_, s_, q_, o_, camera, backend=b))
        print(fmt(backend, m, s, mn))

    print("\n[Binning]")
    for backend, m2d_, rad_, dep_ in [
        ("gsplat", m2d_gs, rad_gs, dep_gs),
        ("mojo",   m2d_mo, rad_mo, dep_mo),
    ]:
        m, s, mn = cuda_event_time_kernel(
            lambda b=backend, a=m2d_, r=rad_, d=dep_: bin_gaussians_to_tiles(
                a, r, d, camera.H, camera.W, TILE_SIZE, backend=b))
        print(fmt(backend, m, s, mn))

    print("\n[Rasterization]")
    for backend, m2d_, con_, sids_, tr_ in [
        ("gsplat", m2d_gs, con_gs, sids_gs, tr_gs),
        ("mojo",   m2d_mo, con_mo, sids_mo, tr_mo),
    ]:
        m, s, mn = cuda_event_time_kernel(
            lambda b=backend, a=m2d_, c=con_, si=sids_, tr=tr_: rasterize_gaussians(
                a, c, features, opacities, bg, tr, si, camera, TILE_SIZE, backend=b))
        print(fmt(backend, m, s, mn))

    # =========================================================================
    # torch.profiler trace
    # =========================================================================
    if not args.no_torch_profiler:
        trace_dir = "/tmp/mojo_profile"
        print(f"\n{'='*70}")
        print(f"torch.profiler trace → {trace_dir}/")
        print(f"{'='*70}")

        import torch.profiler as profiler

        for backend, m2d_, con_, sids_, tr_ in [
            ("gsplat", m2d_gs, con_gs, sids_gs, tr_gs),
            ("mojo",   m2d_mo, con_mo, sids_mo, tr_mo),
        ]:
            trace_path = f"{trace_dir}_{backend}"
            print(f"\nProfiling rasterization/{backend} → {trace_path}")
            with profiler.profile(
                activities=[profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
            ) as prof:
                for _ in range(10):
                    rasterize_gaussians(
                        m2d_, con_, features, opacities, bg,
                        tr_, sids_, camera, TILE_SIZE, backend=backend)
                torch.cuda.synchronize()

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))
            prof.export_chrome_trace(f"{trace_path}.json")
            print(f"  Chrome trace saved to {trace_path}.json")

    # =========================================================================
    # nsys / ncu instructions
    # =========================================================================
    script = Path(__file__).resolve()
    print(f"\n{'='*70}")
    print("DEEPER PROFILING COMMANDS")
    print(f"{'='*70}")
    print(f"""
[1] Nsight Systems — full GPU/CPU timeline:
    nsys profile --trace=cuda,nvtx --output=/tmp/mojo_trace \\
      uv run python {script} --no-torch-profiler
    nsys stats /tmp/mojo_trace.nsys-rep

[2] Nsight Compute — kernel-level metrics (run rasterization only):
    ncu --metrics \\
      sm__warps_active.avg.pct_of_peak_sustained_active,\\
      smsp__inst_executed.sum,\\
      l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\\
      sm__throughput.avg.pct_of_peak_sustained_elapsed \\
      --target-processes all --kernel-name "rasterize_to_pixels" \\
      uv run python {script} --no-torch-profiler

[3] Generate PTX from Mojo kernel (check for __expf vs exp, early-exit):
    cd mojosplat/kernels
    magic run mojo build rasterization.mojo --emit-mlir -o /tmp/rast.mlir
    # or check compiled .ptx in MAX cache:
    find ~/.cache -name "*.ptx" 2>/dev/null | head -10

Hypothesis: the main gap is the missing early-exit in Mojo.
GSplat uses __syncthreads_count(done) >= block_size to break the
batch loop when ALL pixels in a tile have converged. Mojo iterates
all batches regardless, doing barrier() + shared-mem loads for nothing.
""")


if __name__ == "__main__":
    main()
