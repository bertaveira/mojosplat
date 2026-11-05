"""Tests for `render_splats` (Faster-GS style Mojo rasterizer).

* vs the float64 brute-force reference stored in tests/data/golden/splats_*
  (exact up to float32 rounding; checks the binding plumbing: camera, raw
  parameter conventions, SH layout, background);
* vs gsplat.rasterization (close, not identical: gsplat truncates Gaussians
  at 3 sigma, this rasterizer at the exact alpha = 1/255 ellipse).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from mojosplat_torch import Camera, render_splats, rgb_to_sh0

GOLDEN = Path(__file__).resolve().parents[3] / "tests" / "data" / "golden"
DEV = torch.device("cuda")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def load_golden(name):
    d = GOLDEN / name
    meta = dict(line.split("=") for line in (d / "meta.txt").read_text().split())
    meta = {k: int(v) for k, v in meta.items()}
    f = lambda n: torch.from_numpy(np.fromfile(d / n, dtype="<f4")).to(DEV)
    n, k = meta["N"], meta["sh_rest"]
    cam = f("camera.f32").cpu()
    view = cam[:16].reshape(4, 4)
    fx, fy, cx, cy, near, far = cam[16:].tolist()
    camera = Camera(
        R=view[:3, :3].to(DEV), T=view[:3, 3].to(DEV), H=meta["H"], W=meta["W"],
        fx=fx, fy=fy, cx=cx, cy=cy, near=near, far=far,
    )
    params = dict(
        means=f("means3d.f32").reshape(n, 3), quats=f("quats.f32").reshape(n, 4),
        scales=f("log_scales.f32").reshape(n, 3), opacities=f("opacity_logits.f32"),
        sh0=f("sh0.f32").reshape(n, 3),
        shN=f("shN.f32").reshape(n, k, 3) if k > 0 else None,
    )
    image = f("image.f32").reshape(meta["H"], meta["W"], 3)
    return params, camera, meta["sh_degree"], f("background.f32"), image


@pytest.mark.parametrize("name", ["splats_small", "splats_medium", "splats_dc"])
def test_matches_float64_reference(name):
    params, camera, degree, bg, ref = load_golden(name)
    img = render_splats(**params, camera=camera, sh_degree=degree, background=bg)
    assert img.shape == ref.shape
    err = (img - ref).abs()
    assert err.max().item() < 5e-3
    assert err.mean().item() < 1e-6


def random_scene(n, seed, W=320, H=240, sh_rest=15):
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.rand(n, generator=g, device=DEV) * 8 + 2
    xy = (torch.rand(n, 2, generator=g, device=DEV) - 0.5) * 1.5 * z[:, None]
    means = torch.cat([xy, z[:, None]], -1)
    params = dict(
        means=means,
        quats=torch.randn(n, 4, generator=g, device=DEV),
        scales=torch.log(torch.rand(n, 3, generator=g, device=DEV) * 0.08 + 0.01),
        opacities=torch.randn(n, generator=g, device=DEV) * 2,
        sh0=rgb_to_sh0(torch.rand(n, 3, generator=g, device=DEV)),
        shN=torch.randn(n, sh_rest, 3, generator=g, device=DEV) * 0.1 if sh_rest else None,
    )
    camera = Camera(
        R=torch.eye(3, device=DEV), T=torch.zeros(3, device=DEV), H=H, W=W,
        fx=0.9 * W, fy=0.9 * W, cx=W / 2, cy=H / 2,
    )
    return params, camera


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_close_to_gsplat(degree):
    gsplat = pytest.importorskip("gsplat")
    params, camera = random_scene(5000, seed=degree)
    bg = torch.tensor([0.2, 0.3, 0.4], device=DEV)
    img = render_splats(**params, camera=camera, sh_degree=degree, background=bg)

    colors = torch.cat([params["sh0"][:, None], params["shN"]], 1)
    ref, alpha, _ = gsplat.rasterization(
        params["means"], params["quats"], torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), colors, camera.view_matrix[None],
        camera.Ks[None], camera.W, camera.H, near_plane=camera.near,
        far_plane=camera.far, sh_degree=degree,
    )
    ref = ref[0] + (1 - alpha[0]) * bg
    err = (img - ref).abs()
    assert err.mean().item() < 2e-4, err.mean().item()
    assert err.max().item() < 5e-2, err.max().item()


def test_accepts_gsplat_param_shapes_and_reports_stats():
    params, camera = random_scene(2000, seed=7, sh_rest=8)
    params["sh0"] = params["sh0"][:, None, :]  # [N, 1, 3] as in gsplat trainers
    params["opacities"] = params["opacities"][:, None]  # [N, 1]
    img, (visible, instances) = render_splats(**params, camera=camera, return_stats=True)
    assert img.shape == (camera.H, camera.W, 3)
    assert 0 < visible <= 2000 and instances >= visible


def test_degree_zero_equals_rgb():
    """With only sh0, the color is 0.5 + C0 * sh0 = the RGB it was built from."""
    rgb = torch.tensor([[0.9, 0.2, 0.1]], device=DEV)
    params = dict(
        means=torch.tensor([[0.0, 0.0, 4.0]], device=DEV),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEV),
        scales=torch.log(torch.full((1, 3), 5.0, device=DEV)),  # ~50 px: flat at the center
        opacities=torch.tensor([10.0], device=DEV),  # sigmoid ~ 1
        sh0=rgb_to_sh0(rgb), shN=None,
    )
    camera = Camera(R=torch.eye(3, device=DEV), T=torch.zeros(3, device=DEV),
                    H=32, W=32, fx=40.0, fy=40.0, cx=16.0, cy=16.0)
    img = render_splats(**params, camera=camera)
    torch.testing.assert_close(img[16, 16], rgb[0], atol=1e-3, rtol=0)


def test_empty_scene_returns_background():
    z3 = torch.zeros(0, 3, device=DEV)
    camera = Camera(R=torch.eye(3, device=DEV), T=torch.zeros(3, device=DEV),
                    H=20, W=30, fx=30.0, fy=30.0, cx=15.0, cy=10.0)
    bg = torch.tensor([0.1, 0.2, 0.3], device=DEV)
    img = render_splats(z3, torch.zeros(0, 4, device=DEV), z3, torch.zeros(0, device=DEV),
                        z3, None, camera, background=bg)
    torch.testing.assert_close(img, bg.expand(20, 30, 3))


def test_rejects_bad_inputs():
    params, camera = random_scene(10, seed=1, sh_rest=15)
    with pytest.raises(ValueError):
        render_splats(**{**params, "shN": params["shN"][:, :5]}, camera=camera)
    with pytest.raises(ValueError):
        render_splats(**{**params, "shN": params["shN"][:, :3]}, camera=camera, sh_degree=2)
    with pytest.raises(TypeError):
        render_splats(**{**params, "means": params["means"].double()}, camera=camera)
