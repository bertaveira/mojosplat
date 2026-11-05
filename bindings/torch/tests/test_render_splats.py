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


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

PARAM_FILES = {
    "means": "grad_means3d.f32", "quats": "grad_quats.f32", "scales": "grad_log_scales.f32",
    "opacities": "grad_opacity_logits.f32", "sh0": "grad_sh0.f32", "shN": "grad_shN.f32",
}


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("name", ["splats_small", "splats_medium", "splats_dc"])
def test_gradients_match_float64_autograd(name):
    params, camera, degree, bg, _ = load_golden(name)
    d = GOLDEN / name
    params = {k: (v.clone().requires_grad_() if v is not None else None) for k, v in params.items()}
    H, W = camera.H, camera.W
    grad_image = torch.from_numpy(np.fromfile(d / "grad_image.f32", dtype="<f4")).to(DEV).reshape(H, W, 3)
    img = render_splats(**params, camera=camera, sh_degree=degree, background=bg)
    (img * grad_image).sum().backward()
    for k, f in PARAM_FILES.items():
        if params[k] is None:
            continue
        ref = torch.from_numpy(np.fromfile(d / f, dtype="<f4")).to(DEV).reshape(params[k].shape)
        assert rel(params[k].grad, ref) < 1e-4, (k, rel(params[k].grad, ref))


def test_gradients_close_to_gsplat():
    gsplat = pytest.importorskip("gsplat")
    params, camera = random_scene(3000, seed=11)
    leaves = {k: v.clone().requires_grad_() for k, v in params.items()}
    g = torch.Generator(device=DEV).manual_seed(0)
    w = torch.randn(camera.H, camera.W, 3, generator=g, device=DEV)

    img = render_splats(**leaves, camera=camera, sh_degree=3)
    (img * w).sum().backward()
    ours = {k: v.grad for k, v in leaves.items()}

    ref_leaves = {k: v.clone().requires_grad_() for k, v in params.items()}
    colors = torch.cat([ref_leaves["sh0"][:, None], ref_leaves["shN"]], 1)
    ref, _, _ = gsplat.rasterization(
        ref_leaves["means"], ref_leaves["quats"], torch.exp(ref_leaves["scales"]),
        torch.sigmoid(ref_leaves["opacities"]), colors, camera.view_matrix[None],
        camera.Ks[None], camera.W, camera.H, sh_degree=3,
    )
    (ref[0] * w).sum().backward()
    # In general not identical (gsplat truncates at 3 sigma and clamps the
    # Jacobian at a different margin); on this scene they agree to ~3e-6.
    for k in leaves:
        assert rel(ours[k], ref_leaves[k].grad) < 1e-3, (k, rel(ours[k], ref_leaves[k].grad))


def test_two_forwards_before_backward():
    """Each forward keeps its own saved state (e.g. a batch of cameras)."""
    params, cam1 = random_scene(2000, seed=3)
    cam2 = Camera(R=cam1.R, T=torch.tensor([0.3, -0.2, 0.5], device=DEV), H=cam1.H, W=cam1.W,
                  fx=cam1.fx, fy=cam1.fy, cx=cam1.cx, cy=cam1.cy)

    def grads(cams):
        leaves = {k: v.clone().requires_grad_() for k, v in params.items()}
        loss = sum((render_splats(**leaves, camera=c) ** 2).sum() for c in cams)
        loss.backward()
        return {k: v.grad for k, v in leaves.items()}

    both = grads([cam1, cam2])
    g1, g2 = grads([cam1]), grads([cam2])
    # Float atomics accumulate in a nondeterministic order: compare loosely.
    for k in both:
        torch.testing.assert_close(both[k], g1[k] + g2[k], rtol=1e-3, atol=1e-4)


def test_partial_requires_grad_and_densification():
    params, camera = random_scene(2000, seed=5, sh_rest=0)
    means = params["means"].clone().requires_grad_()
    dens = torch.zeros(2, 2000, device=DEV)
    img, (visible, _) = render_splats(**{**params, "means": means}, camera=camera,
                                      densification=dens, return_stats=True)
    img.sum().backward()
    assert means.grad is not None and means.grad.abs().sum() > 0
    assert int(dens[0].sum().item()) == visible  # one count per visible Gaussian
    assert torch.all(dens[1][dens[0] == 0] == 0) and dens[1].sum() > 0
    culled = means.grad.abs().sum(-1)[dens[0] == 0]
    assert torch.all(culled == 0)


def test_optimization_converges():
    """Fit perturbed splats back to a target image with Adam."""
    target_params, camera = random_scene(1500, seed=9, sh_rest=8)
    with torch.no_grad():
        target = render_splats(**target_params, camera=camera)
    g = torch.Generator(device=DEV).manual_seed(1)
    leaves = {
        k: (v + 0.05 * torch.randn(v.shape, generator=g, device=DEV)).requires_grad_()
        for k, v in target_params.items()
    }
    opt = torch.optim.Adam(leaves.values(), lr=5e-3)
    losses = []
    for _ in range(150):
        opt.zero_grad()
        loss = (render_splats(**leaves, camera=camera) - target).abs().mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.3 * losses[0], (losses[0], losses[-1])
