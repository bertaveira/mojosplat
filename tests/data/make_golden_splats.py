#!/usr/bin/env python3
"""Golden data for the Faster-GS style `Rasterizer` (mojosplat/rasterizer.mojo).

The reference is an independent brute-force renderer in float64: every
Gaussian is evaluated at every pixel, with no tiles, bounding boxes or
sorting tricks. It implements the rasterizer's semantics:

  * inputs are raw parameters: log scales, unnormalized (w, x, y, z)
    quaternions, opacity logits, SH coefficients (sh0 [N,3], shN [N,K,3]);
  * cull: depth outside [near, far], sigmoid(opacity) < 1/255,
    |q|^2 < 1e-8, det(cov2d + 0.3 I) < 1e-6;
  * EWA projection with the Jacobian clamped to 15% outside the image;
  * alpha = sigmoid(o) * exp(-0.5 d^T conic d), fragments with
    alpha < 1/255 skipped, front-to-back by depth, a pixel stops after the
    fragment that brings T below 1e-4; color = SH + 0.5 clamped at 0.

Writes raw little-endian binaries under tests/data/golden/<scene>/ that
tests/test_rasterizer.mojo reads back (no Python needed to run the tests):

    meta.txt          N, W, H, sh_degree, sh_rest (K)
    camera.f32        [16 view (row-major world-to-camera)] + [fx, fy, cx, cy, near, far]
    background.f32    [3]
    means3d.f32 [N,3]  log_scales.f32 [N,3]  quats.f32 [N,4]
    opacity_logits.f32 [N]  sh0.f32 [N,3]  shN.f32 [N,K,3]
    image.f32         [H, W, 3] reference render
    image_gsplat.f32  [H, W, 3] gsplat.rasterization (if available; info only)

Usage (repo root):  pixi run -e torch python tests/data/make_golden_splats.py
"""

import math
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).parent / "golden"
DEV = torch.device("cuda")
F64 = torch.float64

SH_C0 = 0.28209479177387814
SH_C1 = 0.48860251190291987
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525251999,
         -1.0925484305920792, 0.54627421529603959]
SH_C3 = [-0.59004358992664352, 2.8906114426405538, -0.45704579946446572,
         0.37317633259011546, -0.45704579946446572, 1.4453057213202769,
         -0.59004358992664352]


def eval_sh(deg, sh0, shN, dirs):
    """Standard 3DGS real SH (independent of the kernel's formulation)."""
    x, y, z = dirs.unbind(-1)
    x, y, z = x[:, None], y[:, None], z[:, None]
    res = SH_C0 * sh0
    if deg > 0:
        res = res - SH_C1 * y * shN[:, 0] + SH_C1 * z * shN[:, 1] - SH_C1 * x * shN[:, 2]
    if deg > 1:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        res = (res + SH_C2[0] * xy * shN[:, 3] + SH_C2[1] * yz * shN[:, 4]
               + SH_C2[2] * (2 * zz - xx - yy) * shN[:, 5]
               + SH_C2[3] * xz * shN[:, 6] + SH_C2[4] * (xx - yy) * shN[:, 7])
    if deg > 2:
        res = (res + SH_C3[0] * y * (3 * xx - yy) * shN[:, 8]
               + SH_C3[1] * xy * z * shN[:, 9]
               + SH_C3[2] * y * (4 * zz - xx - yy) * shN[:, 10]
               + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * shN[:, 11]
               + SH_C3[4] * x * (4 * zz - xx - yy) * shN[:, 12]
               + SH_C3[5] * z * (xx - yy) * shN[:, 13]
               + SH_C3[6] * x * (xx - 3 * yy) * shN[:, 14])
    return res


def quat_to_rot(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)


def reference_render(s, chunk=4096):
    view, (fx, fy, cx, cy, near, far), W, H = s["view"], s["intr"], s["W"], s["H"]
    means, ls, q, ol = (s[k].to(F64) for k in ["means3d", "log_scales", "quats", "opacity_logits"])
    sh0, shN, deg, bg = s["sh0"].to(F64), s["shN"].to(F64), s["sh_degree"], s["bg"].to(F64)
    R, t = view[:3, :3].to(F64), view[:3, 3].to(F64)
    pc = means @ R.T + t
    depth = pc[:, 2]
    opac = torch.sigmoid(ol)
    nq = (q * q).sum(-1)
    valid = (depth >= near) & (depth <= far) & (opac >= 1 / 255) & (nq >= 1e-8)
    Rq = quat_to_rot(torch.where(valid[:, None], q, torch.tensor([1.0, 0, 0, 0], dtype=F64, device=DEV)))
    cov3 = Rq @ torch.diag_embed(torch.exp(2 * ls)) @ Rq.transpose(1, 2)
    z = torch.where(valid, depth, torch.ones_like(depth))
    x, y = pc[:, 0] / z, pc[:, 1] / z
    xc = x.clamp((-0.15 * W - cx) / fx, (1.15 * W - cx) / fx)
    yc = y.clamp((-0.15 * H - cy) / fy, (1.15 * H - cy) / fy)
    J = torch.zeros(len(z), 2, 3, dtype=F64, device=DEV)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx / z * xc
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy / z * yc
    JW = J @ R
    cov2 = JW @ cov3 @ JW.transpose(1, 2) + 0.3 * torch.eye(2, dtype=F64, device=DEV)
    det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2
    valid &= det >= 1e-6
    det = torch.where(valid, det, torch.ones_like(det))
    ca, cb, cc = cov2[:, 1, 1] / det, -cov2[:, 0, 1] / det, cov2[:, 0, 0] / det
    mx, my = x * fx + cx, y * fy + cy
    cam_pos = -R.T @ t
    dirs = means - cam_pos
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    color = (eval_sh(deg, sh0, shN, dirs) + 0.5).clamp_min(0.0)

    idx = torch.nonzero(valid).squeeze(-1)
    idx = idx[torch.argsort(depth[idx], stable=True)]
    mx, my, ca, cb, cc, op, color = mx[idx], my[idx], ca[idx], cb[idx], cc[idx], opac[idx], color[idx]

    ys, xs = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    px = xs.reshape(-1).to(F64) + 0.5
    py = ys.reshape(-1).to(F64) + 0.5
    out = torch.empty(H * W, 3, dtype=F64, device=DEV)
    for s0 in range(0, H * W, chunk):
        dx = mx[None, :] - px[s0:s0 + chunk, None]
        dy = my[None, :] - py[s0:s0 + chunk, None]
        power = 0.5 * (ca * dx * dx + cc * dy * dy) + cb * dx * dy
        alpha = op * torch.exp(-power.clamp_min(0.0))
        alpha = torch.where(alpha >= 1 / 255, alpha, torch.zeros_like(alpha))
        T_before = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], 1), 1)
        inc = (T_before >= 1e-4).to(F64)
        w = inc * T_before * alpha
        T_final = torch.where(inc > 0, 1 - alpha, torch.ones_like(alpha)).prod(1)
        out[s0:s0 + chunk] = w @ color + T_final[:, None] * bg
    return out.reshape(H, W, 3), int(valid.sum())


def rot_xyz(ax, ay, az):
    cx_, sx = math.cos(ax), math.sin(ax)
    cy_, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = torch.tensor([[1, 0, 0], [0, cx_, -sx], [0, sx, cx_]])
    Ry = torch.tensor([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).float()


def make_scene(N, W, H, seed, sh_degree, sh_rest, rot=(0.0, 0.0, 0.0), extra_cull=False):
    g = torch.Generator().manual_seed(seed)
    R = rot_xyz(*rot)
    T = torch.tensor([0.1, -0.2, 0.3])
    z = torch.rand(N, generator=g) * 9.0 + 1.0
    x = (torch.rand(N, generator=g) - 0.5) * 1.6 * z * (W / max(W, H))
    y = (torch.rand(N, generator=g) - 0.5) * 1.6 * z * (H / max(W, H))
    pc = torch.stack([x, y, z], dim=-1)
    log_scales = torch.log(torch.rand(N, 3, generator=g) * 0.08 + 0.005)
    quats = torch.randn(N, 4, generator=g)
    opacity_logits = torch.randn(N, generator=g) * 2.0
    if extra_cull:
        k = N // 50
        pc[:k, 2] = -pc[:k, 2]  # behind the camera
        pc[k:2 * k, 0] += 50.0  # far outside the image
        opacity_logits[2 * k:3 * k] = -8.0  # sigmoid < 1/255
        quats[3 * k:4 * k] = 0.0  # degenerate rotation
        log_scales[4 * k:5 * k] = 1.5  # huge splats -> cooperative tile counting
        pc[5 * k:6 * k, 2] = 0.05  # in front of the near plane
    means3d = (pc - T) @ R  # world = R^T (pc - T)
    sh0 = (torch.rand(N, 3, generator=g) - 0.5) / SH_C0
    shN = torch.randn(N, sh_rest, 3, generator=g) * 0.15
    view = torch.eye(4)
    view[:3, :3] = R
    view[:3, 3] = T
    f = 0.9 * max(W, H)
    return dict(
        view=view.to(DEV), intr=(f, f * 1.05, W / 2.0 + 3.3, H / 2.0 - 2.7, 0.2, 60.0),
        W=W, H=H, sh_degree=sh_degree,
        means3d=means3d.to(DEV), log_scales=log_scales.to(DEV), quats=quats.to(DEV),
        opacity_logits=opacity_logits.to(DEV), sh0=sh0.to(DEV), shN=shN.to(DEV),
        bg=torch.tensor([0.1, 0.2, 0.3], device=DEV),
    )


def save(d: Path, name: str, t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    a.astype("<f4").tofile(d / name)


def gsplat_render(s):
    try:
        from gsplat import rasterization
    except Exception as e:  # pragma: no cover
        print(f"  gsplat unavailable: {e}")
        return None
    fx, fy, cx, cy, near, far = s["intr"]
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=DEV)
    colors = torch.cat([s["sh0"][:, None], s["shN"]], 1)
    img, alpha, _ = rasterization(
        s["means3d"], s["quats"], torch.exp(s["log_scales"]), torch.sigmoid(s["opacity_logits"]),
        colors, s["view"][None], K[None], s["W"], s["H"], near_plane=near, far_plane=far,
        sh_degree=s["sh_degree"],
    )
    return img[0] + (1.0 - alpha[0]) * s["bg"]


def run(name, s):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    image, visible = reference_render(s)
    fx, fy, cx, cy, near, far = s["intr"]
    save(d, "camera.f32", torch.cat([s["view"].reshape(-1).cpu(), torch.tensor([fx, fy, cx, cy, near, far])]))
    save(d, "background.f32", s["bg"])
    for k in ["means3d", "log_scales", "quats", "opacity_logits", "sh0", "shN"]:
        save(d, f"{k}.f32", s[k])
    save(d, "image.f32", image)
    msg = ""
    img_gs = gsplat_render(s)
    if img_gs is not None:
        save(d, "image_gsplat.f32", img_gs)
        msg = f" max|ref-gsplat|={(image.float() - img_gs).abs().max().item():.2e}"
    N = s["means3d"].shape[0]
    (d / "meta.txt").write_text(
        f"N={N}\nW={s['W']}\nH={s['H']}\nsh_degree={s['sh_degree']}\n"
        f"sh_rest={s['shN'].shape[1]}\nvisible={visible}\n"
    )
    print(f"{name}: N={N} visible={visible} {s['W']}x{s['H']} deg={s['sh_degree']}{msg}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    run("splats_small", make_scene(400, 128, 96, seed=0, sh_degree=3, sh_rest=15))
    run("splats_medium", make_scene(12000, 333, 250, seed=1, sh_degree=2, sh_rest=15,
                                    rot=(0.2, -0.3, 0.1), extra_cull=True))
    run("splats_dc", make_scene(3000, 200, 120, seed=2, sh_degree=0, sh_rest=0,
                                rot=(-0.1, 0.25, 0.0), extra_cull=True))
