"""Render a .splat file (antimatter15 format) using mojosplat."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mojosplat.render import render_gaussians
from mojosplat.utils import Camera


def load_splat(path: str):
    """Load a .splat file.

    Binary layout per Gaussian (32 bytes):
      bytes  0-11: position xyz          (3 × float32)
      bytes 12-23: scale xyz             (3 × float32, stored as exp(log_scale))
      bytes 24-27: color RGBA            (4 × uint8 in [0, 255])
      bytes 28-31: rotation quaternion   (4 × uint8, packed as (q*128+128).clip(0,255))

    Colors: RGB = uint8/255, alpha = sigmoid(opacity) already applied.
    Rotation: order is (w, x, y, z), decode as (uint8 - 128) / 128, then normalize.
    """
    data = Path(path).read_bytes()
    n = len(data) // 32
    if n == 0:
        raise ValueError(f"Empty or invalid .splat file: {path}")

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n, 32)

    positions = raw[:, 0:12].view(np.float32).reshape(n, 3).copy()

    # Scales stored as exp(log_scale) — convert back to log-space
    scales_exp = raw[:, 12:24].view(np.float32).reshape(n, 3).copy()
    scales = np.log(np.clip(scales_exp, 1e-10, None))

    rgba = raw[:, 24:28].astype(np.float32) / 255.0
    rgb = rgba[:, :3]
    opacity = rgba[:, 3]  # already sigmoid-applied, in [0, 1]

    # Quaternion (w, x, y, z), packed as (q * 128 + 128).clip(0, 255)
    rot_raw = raw[:, 28:32].astype(np.float32)
    quats = (rot_raw - 128.0) / 128.0
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(norms, 1e-10, None)

    return positions, scales, rgb, opacity, quats


def main():
    parser = argparse.ArgumentParser(description="Render a .splat file with mojosplat")
    parser.add_argument(
        "splat",
        nargs="?",
        default=str(Path(__file__).parent / "bicycle.splat"),
        help="Path to .splat file (default: examples/bicycle.splat)",
    )
    parser.add_argument("--output", "-o", default="render.png", help="Output image path")
    parser.add_argument("--width", "-W", type=int, default=1200)
    parser.add_argument("--height", "-H", type=int, default=800)
    parser.add_argument(
        "--fx", type=float, default=None,
        help="Focal length x in pixels (default: width * 0.9)",
    )
    parser.add_argument(
        "--fy", type=float, default=None,
        help="Focal length y in pixels (default: height * 0.9)",
    )
    parser.add_argument(
        "--cam-pos", type=float, nargs=3, default=[0.0, 0.0, -3.0],
        metavar=("X", "Y", "Z"),
        help="Camera position in world space (default: 0 0 -3)",
    )
    parser.add_argument(
        "--background", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        metavar=("R", "G", "B"),
        help="Background color (default: white)",
    )
    parser.add_argument(
        "--backend", default="mojo", choices=["mojo", "gsplat", "torch"],
        help="Rendering backend (default: mojo)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for rendering")
    device = torch.device("cuda:0")

    print(f"Loading {args.splat} ...")
    positions, scales, rgb, opacity, quats = load_splat(args.splat)
    N = len(positions)
    print(f"  {N:,} Gaussians loaded")

    means3d   = torch.tensor(positions, dtype=torch.float32, device=device)
    scales_t  = torch.tensor(scales,    dtype=torch.float32, device=device)
    quats_t   = torch.tensor(quats,     dtype=torch.float32, device=device)
    opacity_t = torch.tensor(opacity,   dtype=torch.float32, device=device)
    features_t = torch.tensor(rgb,      dtype=torch.float32, device=device)

    W, H = args.width, args.height
    fx = args.fx if args.fx is not None else W * 0.9
    fy = args.fy if args.fy is not None else H * 0.9

    # Identity rotation = camera looking down +Z axis (OpenCV convention)
    R = torch.eye(3, dtype=torch.float32, device=device)
    T = torch.tensor(args.cam_pos, dtype=torch.float32, device=device)
    camera = Camera(R=R, T=T, H=H, W=W, fx=fx, fy=fy, cx=W / 2.0, cy=H / 2.0)

    bg = torch.tensor(args.background, dtype=torch.float32, device=device)

    print(f"Rendering {W}x{H} with '{args.backend}' backend ...")
    image = render_gaussians(
        means3d, scales_t, quats_t, opacity_t, features_t,
        camera,
        background_color=bg,
        backend=args.backend,
    )

    img_np = (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img_np).save(args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
