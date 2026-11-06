# MojoSplat

**3D Gaussian Splatting, training and rendering, written entirely in [Mojo](https://www.modular.com/mojo).** No Python, no CUDA C++: one language from the COLMAP loader to the GPU kernels.

https://github.com/user-attachments/assets/606288c3-1d81-4fa7-9edd-ad549de8cda0

Gaussian Splatting codebases are usually Python orchestrating hand-written CUDA kernels, tied to NVIDIA GPUs. Mojo promises CUDA-level performance in a single, Python-like language that also targets AMD and Apple GPUs. MojoSplat is a full, fast 3DGS implementation built to test that promise.

- **Train a scene with one command.** COLMAP dataset in, `.ply` out (the standard 3DGS format).
- **Fast.** Matches the quality of [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting), a state-of-the-art CUDA implementation, and trains slightly faster: Mip-NeRF360 garden in 3½ minutes on an RTX 5090. Rendering and a training step are about 2× faster than gsplat.
- **Faithful.** Images and gradients match a float64 reference implementation to about 1e-6.
- **Usable from PyTorch.** Bindings with autograd provide a fast drop-in rasterizer for existing training code.

## Quick start

Requires [pixi](https://pixi.sh) and an NVIDIA GPU on Linux. macOS on Apple silicon installs too, but running on Apple GPUs is still untested (see [Roadmap](#roadmap)). The PyTorch bindings are Linux-only.

```bash
git clone https://github.com/bertaveira/mojosplat.git && cd mojosplat
pixi install                                    # Mojo + MAX + libjpeg-turbo
```

**Train** on a COLMAP scene, e.g. from [Mip-NeRF360](https://jonbarron.info/mipnerf360/). Each scene folder holds `sparse/0/` and `images_4/`.

```bash
ln -s /path/to/datasets data
pixi run train data/mipnerf360/garden           # -> output/garden/point_cloud.ply
pixi run eval data/mipnerf360/garden output/garden/point_cloud.ply   # test-view PSNR
```

**Watch it train** in the browser with `--viewer`, then open http://localhost:8080. Orbit around the scene while it trains, pause and resume training, and jump between the dataset cameras. The viewer stays open on the final model. A trained model can be opened on its own with `pixi run view output/garden/point_cloud.ply --scene data/mipnerf360/garden`. The viewer listens on all interfaces (like viser), so a model training on a GPU box can be watched from a laptop; use `--host 127.0.0.1` to keep it local.

Every hyperparameter is a command-line option, e.g. `pixi run train data/mipnerf360/garden --iterations 7000 --sh-degree 1 --no-densify`. `pixi run train --help` lists them all. The options are generated from the config structs, so a new `TrainConfig` field is a new flag.

**Render** a trained `.ply`, or any antimatter15 `.splat` file:

```bash
pixi run render output/garden/point_cloud.ply -o render.png
curl -L --create-dirs -o examples/bicycle.splat https://huggingface.co/cakewalk/splat-data/resolve/main/bicycle.splat
pixi run render examples/bicycle.splat -o render.png -W 1920 -H 1080
```

## Performance

RTX 5090. Training on Mip-NeRF360 garden (`images_4`, 30k iterations, every 8th image held out). Faster-GS was run from its official code on the same machine and data. Both are scored with the same metric code: PSNR / SSIM / LPIPS-VGG on 8-bit test renders.

| | PSNR | SSIM | LPIPS | Gaussians | Training time |
|---|---|---|---|---|---|
| **MojoSplat** | **27.49** | 0.865 | 0.122 | 4.2M | **3m31s** |
| [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting) (CUDA + PyTorch) | 27.44 | 0.866 | 0.121 | 4.1M | 3m48s |
| 3DGS (published) | 27.41 | 0.868 | 0.103 | ~5.8M | — |

Rendering bicycle.splat (6.1M Gaussians, 1280×720), and one training step (forward + backward of all parameters), compared with gsplat through PyTorch:

| | MojoSplat | gsplat |
|---|---|---|
| Render, pure Mojo | **1.21 ms** | — |
| Render via PyTorch | **1.28 ms** | 2.59 ms |
| Training step via PyTorch | **4.00 ms** | 8.29 ms |

Reproduce with `pixi run bench` and `bindings/torch/examples/benchmark_render.py`.

## Using the library

The `mojosplat` package is plain Mojo; use it from your own programs (`pixi run mojo run -I . my_program.mojo`).

Render:

```mojo
from max.gpu.host import DeviceContext
from mojosplat import Camera, Rasterizer, Splats, download, read_ply, write_png

def main() raises:
    var ctx = DeviceContext()
    var splats = Splats.from_host(ctx, read_ply("output/garden/point_cloud.ply"))
    var camera = Camera.translated(0.0, 0.0, 5.0, 1280, 720, 1152.0, 1152.0)
    var image = ctx.enqueue_create_buffer[DType.float32](1280 * 720 * 3)
    var rasterizer = Rasterizer(ctx)  # owns and reuses its GPU buffers
    rasterizer.render(splats, camera, SIMD[DType.float32, 4](0.0), image)
    write_png("out.png", download(ctx, image, 1280 * 720 * 3), 1280, 720)
```

Train:

```mojo
from max.gpu.host import DeviceContext
from mojosplat import TrainConfig, Trainer, load_colmap_dataset

def main() raises:
    var ctx = DeviceContext()
    var config = TrainConfig()  # Faster-GS / 3DGS defaults; every field can be changed
    var trainer = Trainer(ctx, load_colmap_dataset(ctx, "data/mipnerf360/garden"), config)
    for _ in range(config.iterations):
        _ = trainer.step()
    _ = trainer.finish()
    print("test PSNR:", trainer.evaluate())
    trainer.save_ply("garden.ply")
```

The lower-level pieces can be used on their own:
- `Rasterizer.forward` / `backward` for gradients of any image loss;
- `PhotometricLoss`, `Adam` and `Densifier` in `mojosplat.train`;
- the GPU radix sort and scan in `mojosplat.gpu`.

## PyTorch

```bash
pixi install -e torch && pixi run -e torch build-ext
```

```python
from mojosplat_torch import Camera, render_splats

# Raw parameters as stored by 3DGS / gsplat trainers (float32 CUDA tensors):
# means [N,3], quats [N,4] (w,x,y,z), scales [N,3] (log), opacities [N] (logits), sh0 [N,3], shN [N,K,3]
image = render_splats(means, quats, scales, opacities, sh0, shN, camera)   # (H, W, 3)
loss = (image - target).abs().mean()
loss.backward()
```

`bindings/torch/examples/` has an interactive viewer (viser), an orbit-video renderer and the gsplat benchmark.

## How it works

MojoSplat follows the design of [Faster-GS](https://fhahlbohm.github.io/faster-gaussian-splatting) (CVPR 2026):
- **Preprocessing:** a fused pass computes projection, spherical harmonics and *exact* Gaussian–tile intersections, then compacts the visible Gaussians.
- **Sorting:** visible Gaussians are sorted by depth, then their tile instances by a 16-bit tile key, with our own GPU radix sort.
- **Blending:** tiles are composited warp by warp, and each warp skips Gaussians outside its 8×4 sub-tile.
- **Backward:** per-Gaussian rather than per-pixel. The Adam step for the spherical-harmonics coefficients recomputes their gradients on the fly instead of storing them.
- **Training:** the 3DGS recipe (L1 + SSIM loss, Adam, clone/split/prune densification) with Faster-GS's schedule and Z-order reordering of the Gaussians.

Details, design decisions and measurements are in [`docs/design.md`](docs/design.md).

| Path | Contents |
|---|---|
| `mojosplat/` | The library: `core` (types, camera), `gpu` (sort, scan), `render` (rasterizer), `model` (Gaussian storage), `io` (COLMAP, JPEG, PLY, PNG), `train` (loss, Adam, densification, trainer), `cli` (options from config structs, progress bars), `viewer` (web viewer) |
| `tools/` | `train`, `eval`, `view`, `render`, `bench` |
| `tests/` | Mojo tests against reference data (`pixi run test`) |
| `bindings/torch/` | PyTorch package `mojosplat_torch` with tests and examples |

## Roadmap

- **AMD and Apple Silicon.** Every kernel, training included, already compiles for AMD (RDNA3, RDNA4, MI300) and Apple M-series GPUs. They still need validating and tuning on real hardware.
- **Benchmarks.** All Mip-NeRF360 scenes, plus Tanks&Temples and Deep Blending, compared with Faster-GS.
- **Faster training options.** Skip optimizer updates for Gaussians not seen in a view, and 16-bit optimizer state.
- **More training methods.** MCMC densification and anti-aliasing (Mip-Splatting style filters).
- **Evaluation.** LPIPS computed in Mojo.
- **PyTorch.** Run on PyTorch's CUDA stream, and support devices other than `cuda:0`.

Contributions are welcome. Please include tests: most components are checked against a reference implementation in `tests/data/`.

## License

[MIT](LICENSE). Code ported from Faster-GS (Apache-2.0), NeRFICG (MIT) and fused-ssim (MIT) is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). MojoSplat targets Mojo 1.2 / MAX 26.7. MAX is installed by pixi under Modular's [MAX License](https://github.com/modular/modular/blob/main/Licenses/MAX_LICENSE), which is free to use and not redistributed here.

## Acknowledgments

MojoSplat's rasterizer and training recipe are a Mojo port of **[Faster-GS](https://fhahlbohm.github.io/faster-gaussian-splatting)** ([code](https://github.com/nerficg-project/faster-gaussian-splatting), Apache-2.0) by Florian Hahlbohm, Linus Franke, Martin Eisemann and Marcus Magnor. That covers:
- the forward pipeline;
- the per-Gaussian backward pass;
- the densification rules and hyperparameters.

Their paper and code were the blueprint for most of this repository. If you use MojoSplat, please cite their work:

```bibtex
@inproceedings{hahlbohm2026fastergs,
  title     = {Faster-GS: Analyzing and Improving Gaussian Splatting Optimization},
  author    = {Hahlbohm, Florian and Franke, Linus and Eisemann, Martin and Magnor, Marcus},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

Also thanks to:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al.) for the original method;
- the work Faster-GS builds on:
  - [Taming 3DGS](https://github.com/humansensinglab/taming-3dgs) (Mallick et al.) for the per-Gaussian backward;
  - [StopThePop](https://github.com/r4dl/StopThePop-Rasterization) (Radl et al.) for the exact tile test;
  - [Splatshop](https://github.com/m-schuetz/Splatshop) (Schütz et al.) for the two-stage sort;
  - [fused-ssim](https://github.com/rahul-goel/fused-ssim) (Goel et al.) for the SSIM backward;
- [gsplat](https://github.com/nerfstudio-project/gsplat) for the reference implementation we validated against;
- [Modular](https://www.modular.com/) for Mojo.
