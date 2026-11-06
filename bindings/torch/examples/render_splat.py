"""Render a .splat file (antimatter15 format) with mojosplat (or gsplat)."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mojosplat_torch import Camera, load_splat, render_gsplat, render_splats


def main():
    parser = argparse.ArgumentParser(description="Render a .splat file with mojosplat")
    parser.add_argument(
        "splat", nargs="?",
        default=str(Path(__file__).resolve().parents[3] / "examples" / "bicycle.splat"),
        help="Path to .splat file (default: examples/bicycle.splat)",
    )
    parser.add_argument("--output", "-o", default="render.png", help="Output image path")
    parser.add_argument("--width", "-W", type=int, default=1280)
    parser.add_argument("--height", "-H", type=int, default=720)
    parser.add_argument("--fx", type=float, default=None, help="Focal length in pixels (default: 0.9 * width)")
    parser.add_argument("--cam-pos", type=float, nargs=3, default=[0.0, 0.0, 5.0], metavar=("X", "Y", "Z"),
                        help="World-to-camera translation (identity rotation; default: 0 0 5)")
    parser.add_argument("--background", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("R", "G", "B"))
    parser.add_argument("--backend", default="mojo", choices=["mojo", "gsplat"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for rendering")
    device = torch.device("cuda:0")

    print(f"Loading {args.splat} ...")
    params = load_splat(args.splat, device)
    print(f"  {params['means'].shape[0]:,} Gaussians")

    W, H = args.width, args.height
    f = args.fx if args.fx is not None else 0.9 * W
    camera = Camera(
        R=torch.eye(3, device=device), T=torch.tensor(args.cam_pos, device=device),
        H=H, W=W, fx=f, fy=f, cx=W / 2.0, cy=H / 2.0,
    )
    bg = torch.tensor(args.background, dtype=torch.float32, device=device)
    render = render_splats if args.backend == "mojo" else render_gsplat
    with torch.no_grad():
        image = render(**params, camera=camera, background=bg)

    Image.fromarray((image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
