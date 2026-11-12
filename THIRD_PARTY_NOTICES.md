# Third-party notices

## Faster-GS

The rasterizer in `mojosplat/render/` (`rasterizer.mojo`, `preprocess.mojo`,
`sh.mojo`, `instances.mojo`, `blend.mojo`, `blend_backward.mojo`,
`preprocess_backward.mojo`) and the SH constants in `mojosplat/core/math.mojo`
are a Mojo port of, and derived from, the CUDA backend of Faster-GS. The
densification in `mojosplat/train/densify.mojo` (clone / split / prune,
opacity reset) and the training schedule in `mojosplat/train/trainer.mojo`
follow Faster-GS's `adaptive_density_control` and trainer callbacks:

- Project: https://github.com/nerficg-project/faster-gaussian-splatting
- Paper: F. Hahlbohm, L. Franke, M. Eisemann, M. Magnor,
  "Faster-GS: Analyzing and Improving Gaussian Splatting Optimization", CVPR 2026
- Copyright 2026 The Faster-GS Authors
- License: Apache License, Version 2.0 — https://www.apache.org/licenses/LICENSE-2.0

Changes made in the port: rewritten in Mojo; per-Gaussian outputs are
compacted into visible-order slots with a packed 32-byte record; instance
creation and tile counting use warp shuffles instead of shared memory; the
blend kernel lets each warp traverse the tile's instance list independently
(no block-wide batches/barriers); tile ranges are stored as offsets; scan and
radix sort are MojoSplat's own implementations instead of CUB; the training
forward saves its backward state from the warp-independent blend kernel; the
preprocess backward runs in Gaussian order over a slot map so all gradient
writes are coalesced and no separate zero-initialization is needed.

Faster-GS in turn credits StopThePop (Radl et al., 2024) for the exact
Gaussian/tile intersection test and the instance-range extraction, and
Splatshop (Schütz et al.) for the separate depth and tile sorts, and
Taming 3DGS (Mallick et al., 2024) for the per-Gaussian backward pass.

## Deformable Beta Splatting

The Spherical Beta color in `mojosplat/render/blocks/appearance/spherical_beta.mojo`
is ported from the official Beta Splatting code. The Beta-kernel footprint in
`mojosplat/render/blocks/response/beta.mojo` and the `beta-splatting` method's
settings follow it:

- Project: https://github.com/RongLiu-Leo/beta-splatting
- Paper: R. Liu, D. Sun, M. Chen, Y. Wang, A. Feng, "Deformable Beta Splatting", arXiv:2501.18630, 2025
- Copyright the Deformable Beta Splatting authors
- License: Apache License, Version 2.0 — https://www.apache.org/licenses/LICENSE-2.0

Changes made in the port: rewritten in Mojo as MojoSplat blocks, with the color
evaluated in MojoSplat's preprocess kernels and its gradients written by
MojoSplat's preprocess backward.

## NeRFICG

The Morton (Z-order) encoding in `mojosplat/model/morton.mojo` follows
NeRFICG's `MortonEncoding` (itself after Schütz et al.). The metric code used
for the comparison in the README (PSNR / SSIM / LPIPS) is NeRFICG's; it is not
part of this repository.

- Project: https://github.com/nerficg-project/nerficg
- Copyright (c) 2026 Moritz Kappel, Florian Hahlbohm, Timon Scholz
- License: MIT

## fused-ssim

The SSIM backward in `mojosplat/train/loss.mojo` (gradient through the local
moments, computed with separable Gaussian passes) follows fused-ssim:

- Project: https://github.com/rahul-goel/fused-ssim
- Copyright (c) 2024 Rahul Goel
- License: MIT

## MIT License text (NeRFICG, fused-ssim)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
