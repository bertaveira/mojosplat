"""Interactive 3DGS viewer — open a .splat file in the browser via viser.

Usage:
    uv run python examples/viewer.py [path/to/scene.splat] [options]

Then open the printed URL in a browser. Drag to orbit, scroll to zoom.
The scene re-renders on every camera move.

First render triggers JIT compilation (~30–60 s); subsequent renders are fast.
"""
import argparse
import threading
import time
from pathlib import Path

import numpy as np
import torch
import viser
import viser.transforms as vtf

from mojosplat.render import render_gaussians
from mojosplat.utils import Camera


# ---------------------------------------------------------------------------
# .splat loader (antimatter15 binary format, 32 bytes/Gaussian)
# ---------------------------------------------------------------------------

def load_splat(path: str):
    """Load a .splat file and return tensors ready for render_gaussians.

    Binary layout per Gaussian (32 bytes):
      bytes  0-11  position xyz        (3 × float32)
      bytes 12-23  scale xyz           (3 × float32, stored as exp(log_scale))
      bytes 24-27  color RGBA          (4 × uint8)
      bytes 28-31  rotation quaternion (4 × uint8, packed as q*128+128)
    """
    data = Path(path).read_bytes()
    n = len(data) // 32
    if n == 0:
        raise ValueError(f"Empty or invalid .splat file: {path}")

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n, 32)

    positions = raw[:, 0:12].view(np.float32).reshape(n, 3).copy()

    scales_exp = raw[:, 12:24].view(np.float32).reshape(n, 3).copy()
    scales = np.log(np.clip(scales_exp, 1e-10, None))  # back to log-space

    rgba = raw[:, 24:28].astype(np.float32) / 255.0
    rgb = rgba[:, :3]
    opacity = rgba[:, 3]  # already sigmoid-applied

    rot_raw = raw[:, 28:32].astype(np.float32)
    quats = (rot_raw - 128.0) / 128.0
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(norms, 1e-10, None)  # (w, x, y, z)

    return positions, scales, rgb, opacity, quats


# ---------------------------------------------------------------------------
# Camera conversion: viser → mojosplat
# ---------------------------------------------------------------------------
# viser 1.x cameras use OpenCV convention: +X right, +Y down, +Z forward.
# mojosplat/gsplat use the same convention, so conversion is a plain inversion
# of the camera-to-world pose — no axis flip needed.


def _viser_cam_to_mojosplat(cam: viser.CameraHandle, H: int, W: int, device):
    """Convert a viser CameraHandle to a mojosplat Camera."""
    # viser gives camera-to-world pose in OpenCV convention
    R_c2w = vtf.SO3(cam.wxyz).as_matrix()               # (3, 3) float64
    pos   = np.asarray(cam.position, dtype=np.float64)  # (3,)  world coords

    # Invert to world-to-camera
    R_w2c = R_c2w.T.astype(np.float32)
    T_w2c = (-R_c2w.T @ pos).astype(np.float32)

    # Focal lengths from vertical FoV (square pixels)
    fy = H / (2.0 * np.tan(cam.fov / 2.0))
    fx = fy

    return Camera(
        R=torch.tensor(R_w2c, device=device),
        T=torch.tensor(T_w2c, device=device),
        H=H, W=W,
        fx=float(fx), fy=float(fy),
        cx=W / 2.0, cy=H / 2.0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive 3DGS viewer (viser)")
    parser.add_argument(
        "splat",
        nargs="?",
        default=str(Path(__file__).parent / "bicycle.splat"),
        help="Path to .splat file (default: examples/bicycle.splat)",
    )
    parser.add_argument("--width",   "-W", type=int, default=1280, help="Render width  (default 1280)")
    parser.add_argument("--height",  "-H", type=int, default=720,  help="Render height (default 720)")
    parser.add_argument("--port",    "-p", type=int, default=8080, help="Viser port    (default 8080)")
    parser.add_argument(
        "--backend", "-b", default="mojo", choices=["mojo", "gsplat", "torch"],
        help="Rendering backend (default: mojo)",
    )
    parser.add_argument(
        "--bg", nargs=3, type=float, default=[0.0, 0.0, 0.0],
        metavar=("R", "G", "B"),
        help="Background colour 0-1 (default: black)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for rendering")
    device = torch.device("cuda:0")

    # ---- Load scene --------------------------------------------------------
    print(f"Loading {args.splat} …")
    positions, scales, rgb, opacity, quats = load_splat(args.splat)
    N = len(positions)
    print(f"  {N:,} Gaussians")

    means3d   = torch.tensor(positions, dtype=torch.float32, device=device)
    scales_t  = torch.tensor(scales,    dtype=torch.float32, device=device)
    quats_t   = torch.tensor(quats,     dtype=torch.float32, device=device)
    opacity_t = torch.tensor(opacity,   dtype=torch.float32, device=device)
    features_t = torch.tensor(rgb,      dtype=torch.float32, device=device)
    bg = torch.tensor(args.bg,          dtype=torch.float32, device=device)

    W, H = args.width, args.height

    # ---- Warm up JIT (avoid stall on first interactive render) -------------
    print("Warming up renderer (JIT compilation) …")
    _warmup_R = torch.eye(3, dtype=torch.float32, device=device)
    _warmup_T = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32, device=device)
    _warmup_cam = Camera(R=_warmup_R, T=_warmup_T, H=H, W=W,
                         fx=float(W * 0.9), fy=float(W * 0.9),
                         cx=W / 2.0, cy=H / 2.0)
    with torch.no_grad():
        render_gaussians(means3d, scales_t, quats_t, opacity_t, features_t,
                         _warmup_cam, background_color=bg, backend=args.backend)
    print("  JIT ready.")

    # ---- Viser server ------------------------------------------------------
    server = viser.ViserServer(port=args.port, label="MojoSplat Viewer")
    # The .splat scene uses OpenCV world convention: Y is DOWN (-Y is up).
    # Setting "-y" tells viser to orbit with -Y as up, and makes it report
    # camera poses in the same Y-down frame as the scene's Gaussian positions.
    server.scene.set_up_direction("-y")

    # Per-render lock: avoid concurrent GPU renders from multiple clients
    render_lock = threading.Lock()

    @server.on_client_connect
    def on_client(client: viser.ClientHandle):
        # ---- Per-client GUI ------------------------------------------------
        with client.gui.add_folder("Renderer"):
            backend_dd = client.gui.add_dropdown(
                "Backend",
                options=["mojo", "gsplat", "torch"],
                initial_value=args.backend,
            )
            bg_picker = client.gui.add_rgb(
                "Background",
                initial_value=tuple(int(v * 255) for v in args.bg),
            )
        with client.gui.add_folder("Stats"):
            fps_handle   = client.gui.add_number("FPS", min=0, max=999, step=0.1,
                                                 initial_value=0, disabled=True)
            ms_handle    = client.gui.add_number("ms",  min=0, max=9999, step=1,
                                                 initial_value=0, disabled=True)

        # ---- Camera callback -----------------------------------------------
        @client.camera.on_update
        def _(cam: viser.CameraHandle):
            t0 = time.perf_counter()

            # Build background from GUI picker (values are 0-255 ints)
            bg_rgb = bg_picker.value  # (r, g, b) ints 0-255
            bg_t = torch.tensor(
                [c / 255.0 for c in bg_rgb], dtype=torch.float32, device=device
            )

            camera = _viser_cam_to_mojosplat(cam, H, W, device)

            with render_lock:
                with torch.no_grad():
                    img = render_gaussians(
                        means3d, scales_t, quats_t, opacity_t, features_t,
                        camera,
                        background_color=bg_t,
                        backend=backend_dd.value,
                    )

            dt = time.perf_counter() - t0
            img_np = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            client.scene.set_background_image(img_np, format="jpeg", jpeg_quality=90)
            fps_handle.value = round(1.0 / dt, 1)
            ms_handle.value  = round(dt * 1000, 0)

    print(f"\nViewer ready →  http://localhost:{args.port}\n"
          "Drag to orbit · scroll to zoom · Ctrl+drag to pan\n"
          "Press Ctrl-C to quit.\n")
    server.sleep_forever()


if __name__ == "__main__":
    main()
