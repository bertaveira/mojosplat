"""Thin glue between torch tensors and the Mojo extension (`_mojosplat.so`).

Torch owns all memory; tensors are handed to Mojo as raw device addresses.
Synchronization: we wait for torch's current stream before calling into
Mojo, and every Mojo call synchronizes its own stream before returning.
"""

from __future__ import annotations

import torch

try:
    from . import _mojosplat as _native
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "mojosplat_torch: the Mojo extension is not built. "
        "Run `pixi run -e torch build-ext` from the repository root."
    ) from e

TILE_SIZE: int = _native.tile_size()
CHANNELS: int = 3

_context = None


def context():
    """Process-wide Mojo context (GPU 0) holding the renderer's workspaces."""
    global _context
    if _context is None:
        _context = _native.Context()
    return _context


def check(t: torch.Tensor, name: str, dtype: torch.dtype) -> torch.Tensor:
    """Validate a tensor for the Mojo kernels; returns a contiguous view."""
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if t.device.index not in (None, 0):
        raise ValueError(f"{name}: only cuda:0 is supported by the Mojo backend")
    if t.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {t.dtype}")
    return t.contiguous()


def camera_list(camera) -> list:
    """Flatten a `Camera` into the 24-float layout the extension expects."""
    view = camera.view_matrix.detach().to("cpu", torch.float32).reshape(-1).tolist()
    return view + [
        float(camera.fx), float(camera.fy), float(camera.cx), float(camera.cy),
        int(camera.W), int(camera.H), float(camera.near), float(camera.far),
    ]


def sync_torch():
    torch.cuda.current_stream().synchronize()
