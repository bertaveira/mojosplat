---
slug: /
title: Introduction
sidebar_position: 1
---

# MojoSplat

MojoSplat is 3D Gaussian Splatting, training and rendering, written entirely in
[Mojo](https://www.modular.com/mojo): one language from the COLMAP loader to the
GPU kernels. The same source runs on NVIDIA and Apple GPUs. It also compiles for AMD GPUs, but that
has not been tested yet.

The rasterizer and training recipe are a Mojo port of
[Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting)
(Hahlbohm et al., CVPR 2026).

## Where to go next

- **[Quickstart](getting-started/quickstart.md):** install it and train a scene.
- **[Using the Mojo library](getting-started/mojo-library.md):** render and train from your own Mojo code.
- **[PyTorch](getting-started/pytorch.md):** use the rasterizer from existing Python training code.
- **[How rendering works](how-rendering-works.mdx):** the pipeline step by step, compared with 3DGS CUDA.
- **[Architecture](architecture/overview.mdx):** how the code is organized: layers, kernels, blocks, pipelines and data.
