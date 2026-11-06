#!/usr/bin/env python3
"""Golden data for the training components (tests/train).

Writes into tests/data/golden/train/:

    loss/   image.f32 [H,W,3], target.u8 [H,W,3], grad.f32 = dL/dimage,
            meta.txt (H, W, loss, psnr), for
            L = 0.8 * L1 + 0.2 * (1 - SSIM) with 3DGS's SSIM (11x11 Gaussian
            window, sigma 1.5, conv2d with zero padding), float64 autograd.
    adam/   5 steps of torch.optim.Adam (eps 1e-15, one group per attribute
            with its own lr) on random float32 parameters and gradients:
            init_<a>.f32, grads_<a>.f32 [steps, n], param_<a>.f32,
            exp_avg_<a>.f32, exp_avg_sq_<a>.f32, meta.txt
    knn/    points.f32 [N,3] and dist2.f32 [N]: mean squared distance to the
            3 nearest other points (numpy brute force, float64)
    densify/ one step of Faster-GS's adaptive_density_control (clone, split
            with given normals, prune) on random Gaussians with Adam moments:
            in_*.f32 -> out_*.f32, info.f32 [2, N], randn.f32, meta.txt

Usage (repo root):  pixi run -e torch python tests/data/make_golden_train.py
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(__file__).parent / "golden" / "train"


def save(path: Path, a, dtype="<f4"):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a).astype(dtype).tofile(path)


# 3DGS's ssim (utils/loss_utils.py), written out here.
def gaussian(window_size, sigma):
    g = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
                     dtype=torch.float64)
    return g / g.sum()


def ssim(img1, img2, window_size=11):
    channel = img1.shape[0]
    w1 = gaussian(window_size, 1.5).unsqueeze(1)
    w2 = (w1 @ w1.t()).unsqueeze(0).unsqueeze(0)
    window = w2.expand(channel, 1, window_size, window_size).contiguous()
    img1, img2 = img1[None], img2[None]
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def make_loss(d: Path):
    g = torch.Generator().manual_seed(0)
    H, W = 29, 37
    # Smooth-ish target plus noise so SSIM has structure; image near the target.
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    base = torch.stack([x / W, y / H, (x + y) / (W + H)], -1).double()
    target = ((base + 0.1 * torch.rand(H, W, 3, generator=g, dtype=torch.float64)).clamp(0, 1) * 255).round().to(torch.uint8)
    image = (target.double() / 255 + 0.15 * torch.randn(H, W, 3, generator=g, dtype=torch.float64)).clamp(-0.1, 1.1)
    image = image.float().double().requires_grad_()  # exactly representable in float32
    t = target.double() / 255
    l1 = (image - t).abs().mean()
    loss = 0.8 * l1 + 0.2 * (1 - ssim(image.permute(2, 0, 1), t.permute(2, 0, 1)))
    loss.backward()
    mse = ((image.detach().clamp(0, 1) - t) ** 2).mean()
    save(d / "image.f32", image)
    save(d / "target.u8", target, "u1")
    save(d / "grad.f32", image.grad)
    (d / "meta.txt").write_text(f"H={H}\nW={W}\nloss={loss.item():.12g}\npsnr={(-10 * torch.log10(mse)).item():.12g}\n")
    print(f"loss: {loss.item():.6f}")


ATTRS = [("means", 3), ("log_scales", 3), ("rotations", 4), ("opacity_logits", 1), ("sh0", 3), ("sh_rest", 9)]
LRS = [0.01, 0.02, 0.003, 0.05, 0.004, 0.001]


def make_adam(d: Path, n=7, steps=5):
    g = torch.Generator().manual_seed(1)
    params = [torch.randn(n * w, generator=g).requires_grad_() for _, w in ATTRS]
    grads = [torch.randn(steps, n * w, generator=g) * 0.1 for _, w in ATTRS]
    for (name, _), p in zip(ATTRS, params):
        save(d / f"init_{name}.f32", p)
    opt = torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in zip(params, LRS)], betas=(0.9, 0.999), eps=1e-15)
    for s in range(steps):
        for p, gr in zip(params, grads):
            p.grad = gr[s].clone()
        opt.step()
    for (name, _), p, gr in zip(ATTRS, params, grads):
        st = opt.state[p]
        save(d / f"grads_{name}.f32", gr)
        save(d / f"param_{name}.f32", p)
        save(d / f"exp_avg_{name}.f32", st["exp_avg"])
        save(d / f"exp_avg_sq_{name}.f32", st["exp_avg_sq"])
    (d / "meta.txt").write_text(f"N={n}\nsteps={steps}\nsh_rest=3\n" + "".join(f"lr_{a}={lr}\n" for (a, _), lr in zip(ATTRS, LRS)))


def make_knn(d: Path, n=2000):
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(n, 3)).astype(np.float32) * np.float32([1.0, 2.0, 0.5])
    pts[10] = pts[11]  # a duplicate: distance 0 must count
    p = pts.astype(np.float64)
    d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nearest = np.sort(d2, axis=1)[:, :3]
    save(d / "points.f32", pts)
    save(d / "dist2.f32", nearest.mean(1))


def make_densify(d: Path, n=200, k=3):
    """One step of Faster-GS's adaptive_density_control (Model.py), written
    out in torch float32 with given split normals."""
    g = torch.Generator().manual_seed(3)
    extent, thr, pd, min_op = 1.0, 2e-4, 0.01, 0.005
    means = torch.randn(n, 3, generator=g)
    log_scales = torch.log(torch.rand(n, 3, generator=g) * 0.03 + 0.002)
    log_scales[n // 2:] = torch.log(torch.rand(n - n // 2, 3, generator=g) * 0.007 + 0.002)  # small: cloned
    log_scales[:10] = math.log(0.2)  # too large (pruned when prune_large)
    rotations = torch.randn(n, 4, generator=g)
    rotations[10:13] = 1e-5  # degenerate
    opacities = torch.randn(n, generator=g) * 3
    opacities[13:20] = -8.0  # below min opacity
    sh0 = torch.randn(n, 3, generator=g)
    shN = torch.randn(n, k * 3, generator=g)
    counts = torch.randint(0, 6, (n,), generator=g).float()
    gsum = torch.rand(n, generator=g) * 4e-4 * counts.clamp_min(1)
    attrs = [means, log_scales, rotations, opacities[:, None], sh0, shN]
    names = ["means", "log_scales", "rotations", "opacity_logits", "sh0", "sh_rest"]
    m = [torch.randn(a.shape, generator=g) for a in attrs]
    v = [torch.rand(a.shape, generator=g) for a in attrs]

    sel = gsum >= thr * counts.clamp_min(1)
    small = log_scales.max(1).values <= math.log(pd * extent)
    dup, split = sel & small, sel & ~small
    ns = int(split.sum())
    z = torch.randn(2 * ns, 3, generator=g)
    s = log_scales[split].exp().repeat(2, 1)
    q = rotations[split].repeat(2, 1)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, zq = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + zq * zq), 2 * (x * y - w * zq), 2 * (x * zq + w * y),
        2 * (x * y + w * zq), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - w * x),
        2 * (x * zq - w * y), 2 * (y * zq + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)
    split_means = means[split].repeat(2, 1) + (R @ (s * z)[..., None])[..., 0]
    split_scales = log_scales[split].repeat(2, 1) + math.log(0.625)
    new = []
    for i, a in enumerate(attrs):
        if i == 0:
            new.append(torch.cat([a, a[dup], split_means]))
        elif i == 1:
            new.append(torch.cat([a, a[dup], split_scales]))
        else:
            new.append(torch.cat([a, a[dup], a[split].repeat(2, 1)]))
    n_new = int(dup.sum()) + 2 * ns
    new_m = [torch.cat([a, torch.zeros(n_new, a.shape[1])]) for a in m]
    new_v = [torch.cat([a, torch.zeros(n_new, a.shape[1])]) for a in v]
    prune = torch.cat([split, torch.zeros(n_new, dtype=torch.bool)])
    prune |= new[3][:, 0] < math.log(min_op / (1 - min_op))
    prune |= (new[2] ** 2).sum(1) < 1e-8
    prune |= new[1].max(1).values > math.log(0.1 * extent)  # prune_large
    keep = ~prune
    for i, nm in enumerate(names):
        save(d / f"in_{nm}.f32", attrs[i])
        save(d / f"in_m_{nm}.f32", m[i])
        save(d / f"in_v_{nm}.f32", v[i])
        save(d / f"out_{nm}.f32", new[i][keep])
        save(d / f"out_m_{nm}.f32", new_m[i][keep])
        save(d / f"out_v_{nm}.f32", new_v[i][keep])
    save(d / "info.f32", torch.cat([counts, gsum]))
    save(d / "randn.f32", z)
    (d / "meta.txt").write_text(
        f"N={n}\nK={k}\nextent={extent}\ncloned={int(dup.sum())}\nsplit={ns}\n"
        f"pruned={int(prune.sum())}\ncount={int(keep.sum())}\n"
    )
    print(f"densify: cloned {int(dup.sum())}, split {ns}, pruned {int(prune.sum())}, count {int(keep.sum())}")


if __name__ == "__main__":
    make_densify(OUT / "densify")
    make_loss(OUT / "loss")
    make_adam(OUT / "adam")
    make_knn(OUT / "knn")
    print("written to", OUT)
