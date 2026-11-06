"""Scene loading for the torch bindings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .rasterizer import rgb_to_sh0

__all__ = ["load_splat"]


def load_splat(path: str | Path, device: str | torch.device = "cuda") -> dict:
    """Load an antimatter15 `.splat` file as raw (trainable) parameters.

    32 bytes per Gaussian: position (3 x f32), linear scale (3 x f32),
    RGBA (4 x u8, alpha = activated opacity), rotation (4 x u8, (w, x, y, z)
    packed as q * 128 + 128).

    Returns the keyword arguments of `render_splats`: means, quats, scales
    (log), opacities (logits), sh0 (from RGB), shN=None.
    """
    data = Path(path).read_bytes()
    n = len(data) // 32
    if n == 0:
        raise ValueError(f"Empty or invalid .splat file: {path}")
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n, 32)
    means = raw[:, 0:12].view(np.float32).reshape(n, 3)
    scales = np.log(np.clip(raw[:, 12:24].view(np.float32).reshape(n, 3), 1e-30, None))
    rgba = raw[:, 24:28].astype(np.float32) / 255.0
    quats = (raw[:, 28:32].astype(np.float32) - 128.0) / 128.0

    def t(a):
        return torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)

    opacity = t(rgba[:, 3]).clamp(1e-6, 1 - 1e-6)
    return dict(
        means=t(means),
        quats=t(quats),
        scales=t(scales),
        opacities=torch.logit(opacity),
        sh0=rgb_to_sh0(t(rgba[:, :3])),
        shN=None,
    )
