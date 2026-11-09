---
title: PyTorch
sidebar_position: 3
---

# PyTorch

The bindings give existing PyTorch training code a rasterizer with autograd. They
are Linux only and live in a separate pixi environment:

```bash
pixi install -e torch
pixi run -e torch build-ext
```

Run the build again after switching branches.

```python
from mojosplat_torch import Camera, render_splats

# Raw 3DGS / gsplat parameters, as float32 CUDA tensors:
# means [N, 3], quats [N, 4] (wxyz), scales [N, 3] (log),
# opacities [N] (logits), sh0 [N, 3], shN [N, K, 3] or None
image = render_splats(means, quats, scales, opacities, sh0, shN, camera)  # (H, W, 3)
image.mean().backward()
```

`render_splats` also accepts `sh_degree`, `background`, `return_stats` and
`densification` (a float32 `[2, N]` tensor that accumulates the 3DGS
densification statistics during the backward).

Current limits: one camera per call, RGB output and `cuda:0` only. Camera poses
are not differentiable: gradients through `Camera` are dropped.
