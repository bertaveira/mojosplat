---
title: Methods and results
sidebar_position: 3
description: Every method against its official code on Mip-NeRF 360, for quality, training time and rendering speed.
---

# Methods and results

Each method in MojoSplat reproduces a paper and is selected with `--method`. Every method is
trained with its official code and with MojoSplat on the same GPU and the same scenes,
and both are scored with the same metric code. Differences that are still open are
listed with each method.

| Method | `--method` | Paper | Training time per scene | Rendering |
|---|---|---|---|---|
| Faster-GS | `faster-gs` (default) | [Hahlbohm et al., CVPR 2026](https://fhahlbohm.github.io/faster-gaussian-splatting) | 2m24s (1.13×) | 1119 FPS (0.88×) |
| Beta Splatting | `beta-splatting` | [Liu et al., 2025](https://arxiv.org/abs/2501.18630) | 3m31s (3.90×) | 489 FPS (4.58×) |
| 3DGS-MCMC | `3dgs-mcmc` | [Kheradmand et al., NeurIPS 2024](https://arxiv.org/abs/2404.09591) | 3m19s (7.13×) | 682 FPS (2.86×) |

MojoSplat's mean training time per scene and rendering speed on an RTX 5090, over the
seven Mip-NeRF 360 scenes below. In brackets is the speedup over the official code: its
time divided by MojoSplat's, from total training time and mean milliseconds per view.
Below 1× means MojoSplat is slower. The official times are in each method's section.

## How the comparison is run

- **GPU:** NVIDIA RTX 5090, one job at a time.
- **Scenes:** the seven Mip-NeRF 360 scenes available here: garden, bicycle and stump
  at `images_4`, and room, counter, kitchen and bonsai at `images_2`. Every 8th image is
  a test view. `flowers` and `treehill` were not available.
- **Final model only:** all numbers come from the final iteration of training. No
  checkpoint is selected on test views.
- **Same scorer:** both sides render 8-bit PNGs, which are scored against the same
  ground truth with the metric functions the official code uses. LPIPS values are only
  comparable within one method, because methods use different LPIPS implementations.
- **Repeats:** a quality gap of about 0.15 dB or more on a first run is repeated on both
  sides, with a different seed each time, before it counts as a real difference.
- **Training time** is the training loop only, from each codebase's own timer. Dataset
  loading and evaluation are excluded.
- **Rendering time** is milliseconds per view for the same model, cameras and
  resolution. Each frame is timed from a device sync to a device sync, with no image
  readback. Every cell is two alternating rounds of 11 passes over all test views, with
  the first pass of each round dropped. The two renderers' images agree at a mean of
  82-108 dB per scene for Faster-GS and Beta Splatting and 70-77 dB for 3DGS-MCMC,
  far closer than one 8-bit step (about 48 dB), so they do the same work.

## Faster-GS

[Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting) is MojoSplat's
default method. Its rasterizer and training recipe are a Mojo port of the official CUDA
code. Official code: NeRFICG at `623ad16`.

### Quality

PSNR / SSIM / LPIPS of the first run of each side. SSIM is torchmetrics with
`data_range=1.0`, and LPIPS is VGG with `normalize=True`.

| Scene | Official | MojoSplat |
|---|---|---|
| garden | 27.51 / 0.867 / 0.120 | 27.43 / 0.866 / 0.122 |
| bicycle | 25.29 / 0.766 / 0.231 | 25.26 / 0.765 / 0.233 |
| stump | 26.60 / 0.771 / 0.247 | 26.63 / 0.770 / 0.248 |
| room | 31.66 / 0.920 / 0.282 | 31.62 / 0.920 / 0.284 |
| counter | 29.12 / 0.910 / 0.254 | 29.05 / 0.908 / 0.256 |
| kitchen | 31.56 / 0.929 / 0.152 | 31.21 / 0.928 / 0.153 |
| bonsai | 32.20 / 0.942 / 0.251 | 32.05 / 0.941 / 0.253 |
| **mean** | **29.13 / 0.872 / 0.220** | **29.04 / 0.871 / 0.221** |

Kitchen, bonsai and counter were run three times on each side. The means agree:

| Scene | Official PSNR | MojoSplat PSNR |
|---|---|---|
| kitchen | 31.56, 31.45, 31.34 (mean 31.45) | 31.21, 31.43, 31.46 (mean 31.37) |
| bonsai | 32.20, 32.22, 32.25 (mean 32.22) | 32.05, 32.29, 32.36 (mean 32.23) |
| counter | 29.12, 29.12, 29.15 (mean 29.13) | 29.05, 29.10, 29.10 (mean 29.08) |

Both sides end with the same number of Gaussians to within 2.5%. For example, garden
has 4.15M with the official code and 4.20M with MojoSplat.

### Training time

| Scene | Official (s) | MojoSplat (s) | Speedup |
|---|---|---|---|
| garden | 225 | 199 | 1.13× |
| bicycle | 233 | 204 | 1.14× |
| stump | 197 | 166 | 1.19× |
| room | 115 | 105 | 1.10× |
| counter | 112 | 104 | 1.08× |
| kitchen | 147 | 137 | 1.07× |
| bonsai | 110 | 93 | 1.18× |
| **total** | **1139** | **1008** | **1.13×** |

### Rendering speed

Milliseconds per view, with frames per second in brackets.

| Scene | Official | MojoSplat | Speedup |
|---|---|---|---|
| garden | 0.96 ms (1037) | 1.08 ms (924) | 0.89× |
| bicycle | 1.13 ms (888) | 1.26 ms (795) | 0.90× |
| stump | 0.76 ms (1311) | 0.85 ms (1181) | 0.90× |
| room | 0.63 ms (1597) | 0.71 ms (1406) | 0.88× |
| counter | 0.64 ms (1553) | 0.75 ms (1339) | 0.86× |
| kitchen | 0.83 ms (1209) | 0.97 ms (1033) | 0.85× |
| bonsai | 0.55 ms (1808) | 0.64 ms (1558) | 0.86× |
| **mean** | **0.79 ms (1272)** | **0.89 ms (1119)** | **0.88×** |

The remaining gap is the radix sort. MojoSplat's sort runs separate upsweep, scan and
downsweep kernels for every pass, while the official code uses CUB's single-pass
Onesweep sort. Official timings are for its `rasterize` entry point.

## Beta Splatting

[Deformable Beta Splatting](https://github.com/RongLiu-Leo/beta-splatting) replaces the
Gaussian footprint with a Beta kernel, whose sharpness is learned per primitive, and
view-dependent color with Spherical Beta lobes. It densifies by relocation, as 3DGS-MCMC
does, up to a fixed cap per scene. Official code: `431d575`, with the settings of its
`benchmark.py`.

```bash
pixi run train data/mipnerf360/garden --method beta-splatting
```

### Quality

PSNR / SSIM / LPIPS of the first run of each side. SSIM is `fused_ssim`, and LPIPS is
`lpipsPyTorch` VGG, as in the official code. Its LPIPS values are not comparable with the
Faster-GS table above.

| Scene | Official | MojoSplat |
|---|---|---|
| garden | 27.63 / 0.869 / 0.102 | 27.61 / 0.870 / 0.102 |
| bicycle | 25.23 / 0.783 / 0.174 | 25.27 / 0.783 / 0.174 |
| stump | 26.76 / 0.794 / 0.186 | 26.77 / 0.795 / 0.186 |
| room | 32.55 / 0.933 / 0.192 | 32.59 / 0.931 / 0.193 |
| counter | 30.12 / 0.922 / 0.174 | 30.15 / 0.923 / 0.173 |
| kitchen | 32.23 / 0.932 / 0.122 | 32.22 / 0.933 / 0.121 |
| bonsai | 33.70 / 0.952 / 0.182 | 33.40 / 0.951 / 0.183 |
| **mean** | **29.75 / 0.884 / 0.162** | **29.72 / 0.884 / 0.162** |

Repeats, PSNR per run. The official numbers here are the float values it reports, which
are about 0.05 dB above its 8-bit scores.

| Scene | Official PSNR | MojoSplat PSNR |
|---|---|---|
| room | 32.60, 32.54, 32.53 (mean 32.56) | 32.59, 32.50, 32.37 (mean 32.49) |
| bonsai | 33.76, 33.75, 33.75 with seed 0; 33.62 seed 1; 33.63 seed 2 | 33.40, 33.56, 33.48 (mean 33.48) |

Two differences are still open:

- **Bonsai PSNR** is about 0.13 dB below the mean of the three official seeds, with SSIM
  and LPIPS level. All three MojoSplat runs are below the lowest official run, so it may
  be a small real difference.
- **Room SSIM** is about 0.002 lower in all three MojoSplat runs, while PSNR matches.

Both sides reach the fixed Gaussian cap: 5M for garden, 6M for bicycle, 4.5M for stump
and 1.5M for the indoor scenes.

### Training time

MojoSplat's times are from a later run of all seven scenes, with the code as merged. Its
PSNR, from MojoSplat's own evaluator, was within run-to-run noise of the runs above.

| Scene | Official (s) | MojoSplat (s) | Speedup |
|---|---|---|---|
| garden | 1098 | 243 | 4.53× |
| bicycle | 1262 | 259 | 4.86× |
| stump | 947 | 207 | 4.57× |
| room | 580 | 172 | 3.37× |
| counter | 693 | 217 | 3.19× |
| kitchen | 623 | 199 | 3.14× |
| bonsai | 564 | 180 | 3.13× |
| **total** | **5767** | **1478** | **3.90×** |

### Rendering speed

Milliseconds per view, with frames per second in brackets. Both renderers draw the
MojoSplat-trained model from the training-time run. The official timing is the same call
as its `BetaModel.render`.

| Scene | Official | MojoSplat | Speedup |
|---|---|---|---|
| garden | 11.51 ms (87) | 1.79 ms (557) | 6.41× |
| bicycle | 15.54 ms (64) | 2.26 ms (443) | 6.88× |
| stump | 12.76 ms (78) | 2.22 ms (450) | 5.75× |
| room | 5.98 ms (167) | 1.90 ms (528) | 3.15× |
| counter | 7.18 ms (139) | 2.28 ms (439) | 3.15× |
| kitchen | 6.50 ms (154) | 1.76 ms (567) | 3.68× |
| bonsai | 6.05 ms (165) | 2.10 ms (477) | 2.89× |
| **mean** | **9.36 ms (107)** | **2.04 ms (489)** | **4.58×** |

## 3DGS-MCMC

[3DGS-MCMC](https://github.com/ubc-vision/3dgs-mcmc) keeps the 3DGS renderer and replaces
its densification: Gaussians that become transparent are moved onto opaque ones, the
model grows by 5% at a time up to a fixed cap per scene, and position noise lets
Gaussians explore. MojoSplat implements it from the paper and uses the official per-scene
settings. Training starts from 100,000 random points, as the official code does by
default; `--no-random-init` starts from the SfM points. Official code: `7b4fc9f`.

```bash
pixi run train data/mipnerf360/garden --method 3dgs-mcmc
```

### Quality

PSNR / SSIM / LPIPS of the first run of each side, from random starts. SSIM and LPIPS
are the functions of the official `metrics.py` (3DGS's SSIM, `lpipsPyTorch` VGG).

| Scene | Official | MojoSplat |
|---|---|---|
| garden | 27.57 / 0.873 / 0.100 | 27.59 / 0.873 / 0.101 |
| bicycle | 25.59 / 0.795 / 0.175 | 25.59 / 0.794 / 0.176 |
| stump | 27.27 / 0.808 / 0.177 | 27.16 / 0.807 / 0.178 |
| room | 31.99 / 0.928 / 0.201 | 32.00 / 0.928 / 0.200 |
| counter | 29.11 / 0.913 / 0.192 | 29.04 / 0.913 / 0.192 |
| kitchen | 31.81 / 0.931 / 0.126 | 32.02 / 0.932 / 0.126 |
| bonsai | 32.41 / 0.945 / 0.197 | 32.50 / 0.946 / 0.196 |
| **mean** | **29.39 / 0.885 / 0.167** | **29.41 / 0.884 / 0.167** |

The two largest gaps were run again on both sides, with a different seed:

| Scene | Official PSNR | MojoSplat PSNR |
|---|---|---|
| stump | 27.27, 27.24 (mean 27.26) | 27.16, 27.24 (mean 27.20) |
| kitchen | 31.81, 31.95 (mean 31.88) | 32.02, 31.83 (mean 31.93) |

Both are run-to-run noise. Both sides reach the same Gaussian cap: 5.2M for garden,
5.9M for bicycle, 4.75M for stump, 1.5M for room, 1.2M for counter, 1.8M for kitchen
and 1.3M for bonsai.

The paper's random-start table is 0.05-0.47 dB higher on every scene than both
codebases here. The unmodified official code is as far below it as MojoSplat, so the
difference is not in the reimplementation.

### Training time

The official times are the wall time of its `train.py`, which has no timer of its own.
That includes loading the dataset and saving the model, about 16 s of a 22-minute
kitchen run, so the speedups are about 1% too high.

| Scene | Official (s) | MojoSplat (s) | Speedup |
|---|---|---|---|
| garden | 1695 | 277 | 6.13× |
| bicycle | 1832 | 293 | 6.25× |
| stump | 1580 | 256 | 6.18× |
| room | 1242 | 131 | 9.45× |
| counter | 1173 | 134 | 8.74× |
| kitchen | 1342 | 172 | 7.80× |
| bonsai | 1042 | 127 | 8.21× |
| **total** | **9906** | **1390** | **7.13×** |

### Rendering speed

Milliseconds per view, with frames per second in brackets. Both renderers draw the
MojoSplat-trained model. MojoSplat renders it with the same rasterizer as Faster-GS. The
official timing is its `render` function, which uses the original 3DGS rasterizer.
Their images agree at a mean of 70-77 dB per scene, a little less closely than the
other methods because the two rasterizers round and cull slightly differently.

| Scene | Official | MojoSplat | Speedup |
|---|---|---|---|
| garden | 4.09 ms (245) | 1.66 ms (603) | 2.47× |
| bicycle | 5.24 ms (191) | 1.88 ms (533) | 2.79× |
| stump | 5.18 ms (193) | 2.13 ms (471) | 2.44× |
| room | 3.29 ms (304) | 0.95 ms (1056) | 3.48× |
| counter | 3.94 ms (254) | 1.09 ms (920) | 3.62× |
| kitchen | 4.33 ms (231) | 1.42 ms (704) | 3.05× |
| bonsai | 3.30 ms (303) | 1.16 ms (865) | 2.85× |
| **mean** | **4.20 ms (238)** | **1.47 ms (682)** | **2.86×** |
