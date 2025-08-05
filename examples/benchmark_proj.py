#!/usr/bin/env python3
"""
Benchmark script for Gaussian Splatting projection kernels.

Tests the performance of different backends (torch, gsplat, mojo) for projecting
3D Gaussians to 2D image space with varying numbers of Gaussians.
"""

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import argparse
from dataclasses import dataclass

from mojosplat.projection import project_gaussians, Camera


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    backend: str
    num_gaussians: int
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    
    
def look_at(eye, target, up):
    """Calculate view matrix for a camera looking at a target."""
    eye = eye.float()
    target = target.float()
    up = up.float()
    device = eye.device

    z_axis = torch.nn.functional.normalize(eye - target, dim=0)
    x_axis = torch.nn.functional.normalize(torch.cross(up, z_axis, dim=0), dim=0)
    y_axis = torch.nn.functional.normalize(torch.cross(z_axis, x_axis, dim=0), dim=0)

    # World-to-camera rotation
    R_t = torch.stack([x_axis, y_axis, z_axis], dim=0)
    # World-to-camera translation
    t = -torch.matmul(R_t, eye)

    view_matrix = torch.eye(4, device=device, dtype=eye.dtype)
    view_matrix[:3, :3] = R_t
    view_matrix[:3, 3] = t
    return view_matrix


def create_camera(device: torch.device, img_height: int = 1080, img_width: int = 1920) -> Camera:
    """Create a camera setup for benchmarking."""
    # Camera setup
    cam_pos = torch.tensor([0., 1.5, 5.0], device=device)
    cam_target = torch.tensor([0., 0., 0.], device=device)
    cam_up = torch.tensor([0., 1., 0.], device=device)
    view_matrix = look_at(cam_pos, cam_target, cam_up)

    # Camera intrinsics
    focal_length = 80.0
    fx = focal_length
    fy = focal_length
    cx = img_width / 2.0
    cy = img_height / 2.0
    near_plane = 0.1
    far_plane = 100.0

    # Extract R and T from the view matrix
    R = view_matrix[:3, :3]
    T = view_matrix[:3, 3]

    return Camera(
        H=img_height, W=img_width,
        fx=fx, fy=fy, cx=cx, cy=cy,
        R=R, T=T,
        near=near_plane, far=far_plane,
    )


def generate_gaussian_data(num_gaussians: int, device: torch.device) -> tuple:
    """Generate random Gaussian data for benchmarking."""
    # Means centered around origin, spread out
    means3d = torch.randn(num_gaussians, 3, device=device) * 2.0
    
    # Log-scales, mostly small with some variation
    log_scales = torch.ones(num_gaussians, 3, device=device) * -3.0
    log_scales += torch.randn(num_gaussians, 3, device=device) * 0.1
    
    # Random rotations (unit quaternions w, x, y, z)
    random_rots = torch.randn(num_gaussians, 4, device=device)
    rotations = torch.nn.functional.normalize(random_rots, dim=1)
    
    # Opacities (pre-activation)
    opacities = torch.randn(num_gaussians, 1, device=device)
    
    # Ensure correct dtypes
    means3d = means3d.float()
    log_scales = log_scales.float()
    rotations = rotations.float()
    opacities = opacities.float()
    
    return means3d, log_scales, rotations, opacities


def warmup_backend(backend: str, means3d: torch.Tensor, log_scales: torch.Tensor, 
                  rotations: torch.Tensor, opacities: torch.Tensor, camera: Camera, 
                  warmup_iterations: int = 3):
    """Warmup the backend to ensure fair benchmarking."""
    print(f"Warming up {backend} backend...")
    for _ in range(warmup_iterations):
        project_gaussians(means3d, log_scales, rotations, opacities, camera, backend=backend)
        try:
            project_gaussians(means3d, log_scales, rotations, opacities, camera, backend=backend)
            torch.cuda.synchronize()  # Ensure CUDA operations complete
        except Exception as e:
            print(f"Warmup failed for {backend}: {e}")
            return False
    return True


def benchmark_backend(backend: str, means3d: torch.Tensor, log_scales: torch.Tensor,
                     rotations: torch.Tensor, opacities: torch.Tensor, camera: Camera,
                     num_iterations: int = 10) -> tuple:
    """Benchmark a specific backend."""
    times = []
    
    for i in range(num_iterations):
        torch.cuda.synchronize()  # Ensure clean start
        start_time = time.perf_counter()
        
        try:
            means2d, conics, depths, radii = project_gaussians(
                means3d, log_scales, rotations, opacities, camera, backend=backend
            )
            torch.cuda.synchronize()  # Wait for GPU operations to complete
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
        except Exception as e:
            print(f"Error with {backend} backend: {e}")
            return None, str(e)
    
    return np.array(times), None


def run_benchmark(backends: List[str], gaussian_counts: List[int], 
                 img_height: int = 1080, img_width: int = 1920,
                 num_iterations: int = 10) -> List[BenchmarkResult]:
    """Run benchmarks across all backends and Gaussian counts."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cpu':
        print("Warning: CUDA not available, running on CPU (slow).")
        if 'mojo' in backends:
            print("Removing 'mojo' backend as it requires CUDA.")
            backends = [b for b in backends if b != 'mojo']
    
    camera = create_camera(device, img_height, img_width)
    results = []
    
    print(f"Running benchmarks on {device}")
    print(f"Image size: {img_width}x{img_height}")
    print(f"Backends: {backends}")
    print(f"Gaussian counts: {gaussian_counts}")
    print(f"Iterations per test: {num_iterations}")
    print("-" * 60)
    
    for num_gaussians in gaussian_counts:
        print(f"\nTesting with {num_gaussians:,} Gaussians:")
        
        # Generate data once per Gaussian count
        means3d, log_scales, rotations, opacities = generate_gaussian_data(num_gaussians, device)
        
        for backend in backends:
            print(f"  Backend: {backend:<10}", end="")
            
            # Warmup
            if not warmup_backend(backend, means3d, log_scales, rotations, opacities, camera):
                print("FAILED (warmup)")
                continue
            
            # Benchmark
            times, error = benchmark_backend(
                backend, means3d, log_scales, rotations, opacities, camera, num_iterations
            )
            
            if times is not None:
                result = BenchmarkResult(
                    backend=backend,
                    num_gaussians=num_gaussians,
                    mean_time=np.mean(times) * 1000,  # Convert to ms
                    std_time=np.std(times) * 1000,
                    min_time=np.min(times) * 1000,
                    max_time=np.max(times) * 1000,
                )
                results.append(result)
                print(f"Mean: {result.mean_time:.2f}ms ± {result.std_time:.2f}ms")
            else:
                print(f"FAILED ({error})")
    
    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    # Group by number of Gaussians
    by_gaussians = {}
    for result in results:
        if result.num_gaussians not in by_gaussians:
            by_gaussians[result.num_gaussians] = []
        by_gaussians[result.num_gaussians].append(result)
    
    print(f"{'Gaussians':<12} {'Backend':<10} {'Mean (ms)':<12} {'Std (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10}")
    print("-" * 80)
    
    for num_gaussians in sorted(by_gaussians.keys()):
        backend_results = by_gaussians[num_gaussians]
        for i, result in enumerate(backend_results):
            gaussians_str = f"{result.num_gaussians:,}" if i == 0 else ""
            print(f"{gaussians_str:<12} {result.backend:<10} {result.mean_time:<12.2f} "
                  f"{result.std_time:<10.2f} {result.min_time:<10.2f} {result.max_time:<10.2f}")
        if num_gaussians != max(by_gaussians.keys()):
            print("-" * 80)


def plot_results(results: List[BenchmarkResult], save_path: str = "projection_benchmark.png"):
    """Plot benchmark results."""
    if not results:
        print("No results to plot")
        return
    
    # Group by backend
    by_backend = {}
    for result in results:
        if result.backend not in by_backend:
            by_backend[result.backend] = []
        by_backend[result.backend].append(result)
    
    plt.figure(figsize=(12, 8))
    
    # Plot lines for each backend
    for backend, backend_results in by_backend.items():
        backend_results.sort(key=lambda x: x.num_gaussians)
        x_vals = [r.num_gaussians for r in backend_results]
        y_vals = [r.mean_time for r in backend_results]
        y_errs = [r.std_time for r in backend_results]
        
        plt.errorbar(x_vals, y_vals, yerr=y_errs, marker='o', label=backend, linewidth=2, markersize=6)
    
    plt.xlabel('Number of Gaussians')
    plt.ylabel('Projection Time (ms)')
    plt.title('Gaussian Splatting Projection Backend Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    # Format x-axis
    plt.gca().set_xticks([1000, 10000, 100000, 1000000])
    plt.gca().set_xticklabels(['1K', '10K', '100K', '1M'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Gaussian Splatting projection kernels")
    parser.add_argument('--backends', nargs='+', default=['torch', 'gsplat', 'mojo'],
                       choices=['torch', 'gsplat', 'mojo'],
                       help='Backends to benchmark')
    parser.add_argument('--gaussians', nargs='+', type=int, 
                       default=[1000, 5000, 10000, 50000, 100000, 500000],
                       help='Number of Gaussians to test')
    parser.add_argument('--width', type=int, default=1920, help='Image width')
    parser.add_argument('--height', type=int, default=1080, help='Image height')
    parser.add_argument('--iterations', type=int, default=100, help='Iterations per test')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting results')
    parser.add_argument('--output', default='projection_benchmark.png', help='Output plot filename')
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_benchmark(
        backends=args.backends,
        gaussian_counts=args.gaussians,
        img_height=args.height,
        img_width=args.width,
        num_iterations=args.iterations
    )
    
    # Print results
    print_results_table(results)
    
    # Plot results
    if not args.no_plot and results:
        plot_results(results, args.output)
    
    # Performance analysis
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        # Find fastest backend for each Gaussian count
        by_gaussians = {}
        for result in results:
            if result.num_gaussians not in by_gaussians:
                by_gaussians[result.num_gaussians] = []
            by_gaussians[result.num_gaussians].append(result)
        
        for num_gaussians in sorted(by_gaussians.keys()):
            backend_results = by_gaussians[num_gaussians]
            if len(backend_results) > 1:
                fastest = min(backend_results, key=lambda x: x.mean_time)
                slowest = max(backend_results, key=lambda x: x.mean_time)
                speedup = slowest.mean_time / fastest.mean_time
                print(f"{num_gaussians:,} Gaussians: {fastest.backend} fastest "
                      f"({fastest.mean_time:.2f}ms), {speedup:.1f}x faster than {slowest.backend}")


if __name__ == "__main__":
    main()
