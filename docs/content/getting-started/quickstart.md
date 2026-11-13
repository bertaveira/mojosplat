---
title: Quickstart
sidebar_position: 1
---

# Quickstart

## Install

You need [pixi](https://pixi.sh) and an NVIDIA GPU or an Apple silicon Mac. The code
also compiles for AMD GPUs, but that has not been tested yet. There is no CUDA toolkit, compiler or Python to install: pixi puts
Mojo in the project folder the first time you run something.

```bash
git clone https://github.com/bertaveira/mojosplat.git
cd mojosplat
```

On a Mac, Mojo compiles kernels with Xcode's Metal toolchain. Check for it with
`xcrun metal --version`. If it is missing, install it:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Train a scene

A scene is a COLMAP folder containing `sparse/0/` and an image folder (`images_4/`
by default), for example from [Mip-NeRF 360](https://jonbarron.info/mipnerf360/).

```bash
pixi run train path/to/scene --viewer   # watch it train at http://localhost:8080
```

The model is written to `output/<scene name>/point_cloud.ply`, in the standard
3DGS layout. Every 8th image is held out for testing. Next to the model, `run.json`
records the method, scene, commit, GPU, settings and results.

Every setting is a command-line flag. `pixi run train --help` lists them all. The
ones you are most likely to use:

| Flag | Meaning | Default |
|---|---|---|
| `--images` | Image folder inside the scene | `images_4` |
| `-o`, `--out` | Output directory | `output/<scene name>` |
| `--test-every` | Hold out every N-th image (0 trains on all) | `8` |
| `--iterations` | Training iterations | `30000` |
| `--viewer` | Serve the live web viewer while training | off |
| `--method` | Reconstruction method: `faster-gs`, `beta-splatting` or `3dgs-mcmc` ([Methods and results](../results.md)) | `faster-gs` |
| `--no-densify` | Turn densification off | densification on |
| `--seed` | Random seed | `0` |
| `--port` | Viewer port | `8080` |
| `--host` | Address the viewer listens on | `0.0.0.0` |

## Evaluate, view and render

```bash
pixi run eval path/to/scene output/scene/point_cloud.ply
pixi run view output/scene/point_cloud.ply
pixi run render output/scene/point_cloud.ply -o render.png   # also renders .splat files
```

`eval` reports PSNR on the held-out views, and with `--save <dir>` writes the
renders and ground truth as PNGs.
