"""Render an orbit video around a .splat scene.

Usage:
    pixi run -e torch python bindings/torch/examples/orbit_video.py [path/to/scene.splat] [options]

Produces orbit.mp4 (or --output path). Requires CUDA and ffmpeg.
"""
import argparse
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mojosplat_torch import Camera, load_splat, render_gsplat, render_splats

RENDERERS = {"mojo": render_splats, "gsplat": render_gsplat}


# ---------------------------------------------------------------------------
# Camera math — OpenCV convention (Y-down, Z-forward)
# ---------------------------------------------------------------------------

def lookat(eye: np.ndarray, center: np.ndarray):
    """World-to-camera R and T for an OpenCV camera (Y-down world)."""
    eye    = np.asarray(eye,    dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    world_down = np.array([0.0, 1.0, 0.0])   # +Y is down in this scene

    z = center - eye
    z /= np.linalg.norm(z)

    x = np.cross(world_down, z)
    if np.linalg.norm(x) < 1e-6:             # camera directly above/below
        x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)

    y = np.cross(z, x)                        # camera Y (down in image)
    y /= np.linalg.norm(y)

    R_c2w = np.stack([x, y, z], axis=1)       # columns = camera axes in world
    R_w2c = R_c2w.T
    T_w2c = -R_w2c @ eye
    return R_w2c.astype(np.float32), T_w2c.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Orbit video around a .splat scene")
    parser.add_argument("splat", nargs="?",
                        default=str(Path(__file__).resolve().parents[3] / "examples" / "bicycle.splat"))
    parser.add_argument("--output", "-o", default="orbit.mp4")
    parser.add_argument("--width",  "-W", type=int, default=800)
    parser.add_argument("--height", "-H", type=int, default=450)
    parser.add_argument("--frames", "-n", type=int, default=120,
                        help="Total frames (default 120 → 4 s at 30 fps)")
    parser.add_argument("--fps",    type=int,   default=30)
    parser.add_argument("--radius", "-r", type=float, default=5.0,
                        help="Orbit radius in scene units (default 5)")
    parser.add_argument("--elevation", "-e", type=float, default=2.5,
                        help="Camera height above center in scene units (default 2.5)")
    parser.add_argument("--center", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Orbit center (default: auto-detected near ground level)")
    parser.add_argument("--backend", "-b", default="mojo",
                        choices=list(RENDERERS))
    parser.add_argument("--background", nargs=3, type=float,
                        default=[1.0, 1.0, 1.0], metavar=("R", "G", "B"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for rendering")
    device = torch.device("cuda:0")

    # ---- Load scene --------------------------------------------------------
    print(f"Loading {args.splat} …")
    params = load_splat(args.splat, device)
    positions = params["means"].cpu().numpy()
    print(f"  {len(positions):,} Gaussians")
    bg = torch.tensor(args.background, dtype=torch.float32, device=device)
    render = RENDERERS[args.backend]

    W, H = args.width, args.height
    fx = fy = W * 0.9   # square pixels, ~62° horizontal FoV for 800 px

    if args.center is not None:
        center = np.array(args.center, dtype=np.float64)
    else:
        # The scene uses Y-down convention: sky has large negative Y, ground has
        # small positive Y. The mean is sky-biased; use the 80th-percentile of Y
        # (near ground/bicycle level) and the median of X and Z.
        center = np.array([
            np.median(positions[:, 0]),
            np.percentile(positions[:, 1], 80),
            np.median(positions[:, 2]),
        ])
    print(f"  Orbit center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

    # ---- JIT warm-up -------------------------------------------------------
    print("Warming up renderer (JIT) …")
    R0, T0 = lookat(center + np.array([args.radius, -args.elevation, 0.0]), center)
    _cam = Camera(R=torch.tensor(R0, device=device),
                  T=torch.tensor(T0, device=device),
                  H=H, W=W, fx=fx, fy=fy, cx=W/2, cy=H/2)
    with torch.no_grad():
        render(**params, camera=_cam, background=bg)
    print("  JIT ready.")

    # ---- Render frames -----------------------------------------------------
    print(f"Rendering {args.frames} frames at {W}×{H} …")
    frames = []
    for i in range(args.frames):
        theta = 2 * math.pi * i / args.frames
        eye = center + np.array([
            args.radius * math.cos(theta),
            -args.elevation,              # above center: Y-down → subtract
            args.radius * math.sin(theta),
        ])
        R_w2c, T_w2c = lookat(eye, center)
        cam = Camera(
            R=torch.tensor(R_w2c, device=device),
            T=torch.tensor(T_w2c, device=device),
            H=H, W=W, fx=fx, fy=fy, cx=W/2, cy=H/2,
        )
        with torch.no_grad():
            img = render(**params, camera=cam, background=bg)
        frames.append((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        if (i + 1) % 20 == 0 or i == args.frames - 1:
            print(f"  {i+1}/{args.frames}")

    # ---- Encode video ------------------------------------------------------
    out_path = Path(args.output)
    if out_path.suffix.lower() == ".gif":
        _save_gif(frames, out_path, args.fps)
    else:
        _save_mp4(frames, out_path, args.fps)


def _save_mp4(frames, out_path: Path, fps: int):
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(f"{tmp}/{i:04d}.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", f"{tmp}/%04d.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Saved → {out_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"ffmpeg failed ({exc}), falling back to GIF")
            _save_gif(frames, out_path.with_suffix(".gif"), fps)


def _save_gif(frames, out_path: Path, fps: int):
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(out_path, save_all=True, append_images=pil[1:],
                duration=int(1000 / fps), loop=0)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
