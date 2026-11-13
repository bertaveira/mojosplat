# MojoSplat

**3D Gaussian Splatting, training and rendering, written entirely in [Mojo](https://www.modular.com/mojo).** No Python, no CUDA C++: one language from the COLMAP loader to the GPU kernels.

https://github.com/user-attachments/assets/606288c3-1d81-4fa7-9edd-ad549de8cda0

Gaussian Splatting codebases are usually Python orchestrating hand-written CUDA kernels, tied to NVIDIA GPUs. Mojo promises CUDA-level performance in a single, Python-like language that also targets AMD and Apple GPUs. MojoSplat is a full, fast 3DGS implementation built to test that promise.

- **Train a scene with one command.** COLMAP dataset in, `.ply` out (the standard 3DGS format).
- **Fast.** Matches the quality of [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting), a state-of-the-art CUDA implementation, and trains slightly faster: Mip-NeRF360 garden in under 3½ minutes on an RTX 5090. Rendering and a training step are about 2× faster than gsplat.
- **Faithful.** Images and gradients match a float64 reference implementation to about 1e-6.
- **Usable from PyTorch.** Bindings with autograd provide a fast drop-in rasterizer for existing training code.

**Documentation:** [bertaveira.github.io/mojosplat](https://bertaveira.github.io/mojosplat/) has a quickstart, an illustrated walkthrough of the renderer, and how to add a method.

## Quick start

All you need is [pixi](https://pixi.sh) and an NVIDIA GPU, AMD GPU, or an Apple silicon Mac. No CUDA toolkit, compiler or Python needed: pixi installs Mojo into the project folder on the first run. Just clone this repository `git clone https://github.com/bertaveira/mojosplat.git` and you're ready to go.

```bash
pixi run train path/to/scene --viewer        # watch it train at http://localhost:8080
```

You can evaluate, view, and render the trained scene with the following commands:
```bash
pixi run eval path/to/scene output/scene/point_cloud.ply
pixi run view output/scene/point_cloud.ply
pixi run render output/scene/point_cloud.ply -o render.png   # also renders .splat files
```

A scene is a COLMAP folder with `sparse/0/` and `images_4/`, e.g. from [Mip-NeRF360](https://jonbarron.info/mipnerf360/). Every hyperparameter is a flag: `pixi run train --help`.

On a Mac, Mojo compiles kernels with Xcode's Metal toolchain: check it with `xcrun metal --version` and, if it is missing, install it with `xcodebuild -downloadComponent MetalToolchain`.

## Performance

Mip-NeRF360 garden (`images_4`, 30k iterations). The last four columns are training time per GPU. PSNR, SSIM and LPIPS are from the RTX 5090 runs; PSNR on the RTX 3080 and Apple M1 Pro lands within 0.1 dB. Faster-GS was run from its official code on the same RTX 5090 and scored with the same metric code.

| | PSNR | SSIM | LPIPS | Gaussians | RTX 5090 | RTX 3080 (10 GB) | Apple M4 Max | Apple M1 Pro |
|---|---|---|---|---|---|---|---|---|
| **MojoSplat** | **27.45** | 0.866 | 0.122 | 4.2M | **3m19s** | **~10m30s** | 16m06s | 1h03m |
| [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting) (CUDA + PyTorch) | 27.44 | 0.866 | 0.121 | 4.1M | 3m48s | — | — | — |
| 3DGS (published) | 27.41 | 0.868 | —\* | ~5.8M | — | — | — | — |

\* The 3DGS paper's LPIPS (0.103) is not comparable: its evaluation code passes images in [0, 1] to an LPIPS network that expects [-1, 1], which lowers the score. The other rows use LPIPS-VGG on correctly scaled images.

Each method, selected with `--method`, is also trained with its official code on seven Mip-NeRF 360 scenes, on the same RTX 5090, and scored the same way. The table gives MojoSplat's mean training time per scene and rendering speed, with the speedup over the official code in brackets (below 1× is slower). Per-scene quality, timings and open differences are in the [full results](https://bertaveira.github.io/mojosplat/results).

| Method | `--method` | Training time per scene | Rendering |
|---|---|---|---|
| [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting) | `faster-gs` (default) | 2m24s (1.13×) | 1119 FPS (0.88×) |
| [Beta Splatting](https://github.com/RongLiu-Leo/beta-splatting) | `beta-splatting` | 3m31s (3.90×) | 489 FPS (4.58×) |

## Using the library

`mojosplat` is a plain Mojo package (`pixi run mojo run -I . my_program.mojo`):

```mojo
from max.gpu.host import DeviceContext
from mojosplat import Camera, Rasterizer, Splats, download, read_ply, write_png

def main() raises:
    var ctx = DeviceContext()
    var splats = Splats.from_host(ctx, read_ply("output/garden/point_cloud.ply"))
    var camera = Camera.translated(0.0, 0.0, 5.0, 1280, 720, 1152.0, 1152.0)
    var image = ctx.enqueue_create_buffer[DType.float32](1280 * 720 * 3)
    var rasterizer = Rasterizer(ctx)
    rasterizer.render(splats, camera, SIMD[DType.float32, 4](0.0), image)
    write_png("out.png", download(ctx, image, 1280 * 720 * 3), 1280, 720)
```

Training is `Trainer(ctx, load_colmap_dataset(ctx, path), TrainConfig())` and `trainer.step()`; `Rasterizer.forward` / `backward` give gradients for any image loss. The design is described in [`docs/design.md`](docs/design.md).

## PyTorch

Linux only, in a separate pixi environment:

```bash
pixi install -e torch && pixi run -e torch build-ext
```

```python
from mojosplat_torch import Camera, render_splats

# Raw 3DGS / gsplat parameters: means [N,3], quats [N,4] (wxyz), scales [N,3] (log),
# opacities [N] (logits), sh0 [N,3], shN [N,K,3]; float32 CUDA tensors
image = render_splats(means, quats, scales, opacities, sh0, shN, camera)   # (H, W, 3)
image.mean().backward()
```

## Roadmap

- Validate on AMD GPUs and tune on AMD and Apple GPUs (every kernel already compiles for both; Apple runs).
- Benchmarks on all Mip-NeRF360, Tanks&Temples and Deep Blending scenes.
- MCMC densification, anti-aliasing, faster optimizer options, LPIPS in Mojo.

Contributions are welcome; please include tests (`pixi run test`).

## License

[MIT](LICENSE). Ported code from Faster-GS (Apache-2.0), NeRFICG (MIT) and fused-ssim (MIT) is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). MAX is installed by pixi under Modular's [MAX License](https://github.com/modular/modular/blob/main/Licenses/MAX_LICENSE).

## Acknowledgments

MojoSplat's rasterizer and training recipe are a Mojo port of **[Faster-GS](https://fhahlbohm.github.io/faster-gaussian-splatting)** ([code](https://github.com/nerficg-project/faster-gaussian-splatting)) by Florian Hahlbohm, Linus Franke, Martin Eisemann and Marcus Magnor. If you use MojoSplat, please cite their work:

```bibtex
@inproceedings{hahlbohm2026fastergs,
  title     = {Faster-GS: Analyzing and Improving Gaussian Splatting Optimization},
  author    = {Hahlbohm, Florian and Franke, Linus and Eisemann, Martin and Magnor, Marcus},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

Also thanks to [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [Taming 3DGS](https://github.com/humansensinglab/taming-3dgs), [StopThePop](https://github.com/r4dl/StopThePop-Rasterization), [Splatshop](https://github.com/m-schuetz/Splatshop), [fused-ssim](https://github.com/rahul-goel/fused-ssim), [gsplat](https://github.com/nerfstudio-project/gsplat) (our reference implementation) and [Modular](https://www.modular.com/).
