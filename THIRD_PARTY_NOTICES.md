# Third-party notices

## Faster-GS

The rasterizer in `mojosplat/` (`rasterizer.mojo`, `preprocess.mojo`,
`sh.mojo`, `instances.mojo`, `blend.mojo`, `backward.mojo`) is a Mojo port of,
and derived from, the CUDA backend of Faster-GS:

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
