# MojoSplat

MojoSplat is an experimental port of Gaussian Splatting kernels to [Mojo](https://www.modular.com/mojo), exploring the potential performance and multi-vendor support of Mojo for GPU acceleration.



https://github.com/user-attachments/assets/ae4409a3-a656-4635-bb57-f22aca2f6141



MojoSplat implements 3D Gaussian Splatting rendering, **forward and backward, entirely in Mojo**:

- **`Rasterizer`**: the design of [Faster-GS](https://fhahlbohm.github.io/faster-gaussian-splatting) (Hahlbohm et al., CVPR 2026). It takes the raw trainable parameters (log scales, quaternions, opacity logits, SH up to degree 3). The pipeline:
  - a fused preprocess pass with exact Gaussian/tile culling;
  - a two-stage sort: depth first, then a stable 16-bit tile sort;
  - per-tile blending with 8×4 sub-tile skipping;
  - a per-Gaussian backward pass (Taming-3DGS / Faster-GS) for training, with gradients for all raw parameters.

The core library (`mojosplat/`) has **no Python dependency**: you can load a `.splat` file, render it and write a PNG from a Mojo program. PyTorch support lives separately in [`bindings/torch`](bindings/torch) (in the spirit of tiny-cuda-nn), where torch tensors are handed to the Mojo kernels by pointer, without copies.

> [!WARNING]
> 1. This is NOT production ready (tested on NVIDIA; AMD RDNA3/4 and Apple Silicon compile but are untested; the pure-Mojo training loop is in progress, see `docs/design.md`).
> 2. Mojo is evolving fast. This version targets **Mojo 1.2 / MAX 26.7**.

## Repository layout

| Path | What |
|------|------|
| `mojosplat/` | The library (pure Mojo, no CLI), split by layer. |
| `mojosplat/core/` | Plain types: pointers, `Camera`, the `SplatView` view, SH constants. |
| `mojosplat/gpu/` | Generic GPU building blocks: device scan, radix sort (16/32/64-bit keys), fills, memory arenas. |
| `mojosplat/render/` | The `Rasterizer`: `preprocess`, `sh`, `instances`, `blend`, `blend_backward`, `preprocess_backward`. |
| `mojosplat/model/` | Owning storage of the Gaussians (`Splats`). |
| `mojosplat/io/` | Raw arrays, `.splat` and 3DGS `.ply` scenes, COLMAP binary models, JPEG decoding (libjpeg-turbo), PNG/PPM output. |
| `mojosplat/train/` | Training: COLMAP datasets (loss, optimizer and trainer in progress). |
| `tools/` | Command-line programs: `render.mojo`, `bench.mojo`. |
| `tests/` | Mojo tests, mirroring the package (`tests/gpu/`, `tests/render/`, `tests/io/`, `tests/train/`). Dataset tests use Mip-NeRF360 garden from `data/` (a local symlink to your datasets folder) and skip if it is missing. Golden data in `tests/data/golden` comes from an independent float64 brute-force reference and its autograd (`make_golden_splats.py`). |
| `bindings/torch/` | `mojosplat_torch` Python package: the Mojo extension (`render_splats` with autograd), a gsplat reference (`render_gsplat`), pytest suite, and Python examples (viewer, orbit video, benchmark). |
| `docs/design.md` | Architecture and roadmap. |

## Installation

Requires an NVIDIA GPU and [pixi](https://pixi.sh) (Modular's recommended way to install Mojo).

```bash
git clone https://github.com/bertaveira/mojosplat.git
cd mojosplat
pixi install            # Mojo 1.2.0 + MAX 26.7.0 (GPU runtime), no torch
```

Get a scene to play with (antimatter15 `.splat` format):

```bash
curl -L -o examples/bicycle.splat https://huggingface.co/cakewalk/splat-data/resolve/main/bicycle.splat
```

## Pure Mojo usage

```bash
pixi run render                                  # examples/bicycle.splat -> render.png
pixi run render my_scene.splat -o out.png -W 1920 -H 1080 --cam 0 0 4
pixi run render trained.ply -o out.png              # standard 3DGS .ply files work too
pixi run bench -n 200                            # full frame + per-stage timings
pixi run test                                    # Mojo test-suite
```

From your own Mojo code:

```mojo
from max.gpu.host import DeviceContext
from mojosplat import Camera, Rasterizer, Splats, download, load_splat, write_png

def main() raises:
    var ctx = DeviceContext()
    # .splat stores activated values; from_scene converts to raw parameters
    var splats = Splats.from_scene(ctx, load_splat("examples/bicycle.splat"))
    var camera = Camera.translated(0.0, 0.0, 5.0, 1280, 720, 1152.0, 1152.0)
    var image = ctx.enqueue_create_buffer[DType.float32](1280 * 720 * 3)
    var rasterizer = Rasterizer(ctx)      # owns and reuses all intermediate buffers
    rasterizer.render(splats, camera, SIMD[DType.float32, 4](0.0), image)
    write_png("out.png", download(ctx, image, 1280 * 720 * 3), 1280, 720)
```

`Splats(ctx, means, log_scales, quats, opacity_logits, sh0, shN, rest_stride)` builds a scene from raw parameters (e.g. a trained 3DGS model with `sh0` [N, 3] and `shN` [N, 15, 3]).

Gradients, for training:

```mojo
var state = rasterizer.forward(splats.view(), camera, bg, dptr(image), sh_degree)
# ... compute dL/dimage into grad_image ([H, W, 3]) ...
rasterizer.backward(splats.view(), state, dptr(image), dptr(grad_image), grads)
# grads: a SplatView of gradient buffers laid out like the parameters
```

Run it with `pixi run mojo run -I . my_program.mojo`.

## PyTorch bindings

```bash
pixi install -e torch          # adds torch, gsplat (reference), pytest
pixi run -e torch build-ext    # builds bindings/torch/mojosplat_torch/_mojosplat.so
pixi run -e torch test-torch   # pytest suite (vs float64 reference and gsplat)
```

```python
import torch
from mojosplat_torch import Camera, render_splats

# Raw parameters as stored by 3DGS / gsplat trainers, float32 CUDA tensors on cuda:0:
# means [N,3], quats [N,4] (w,x,y,z), scales [N,3] (log), opacities [N] (logits),
# sh0 [N,3] or [N,1,3], shN [N,K,3] (K = 3, 8, 15) or None
camera = Camera(R=torch.eye(3, device="cuda"), T=torch.tensor([0.0, 0.0, 5.0], device="cuda"),
                H=720, W=1280, fx=1152.0, fy=1152.0, cx=640.0, cy=360.0)
image = render_splats(means, quats, scales, opacities, sh0, shN, camera)  # (H, W, 3)
loss = (image - target).abs().mean()
loss.backward()   # gradients for means, quats, scales, opacities, sh0, shN
```

`render_splats` is a `torch.autograd.Function`: each forward keeps its own saved state in torch-allocated buffers, so several renders (e.g. a batch of cameras) can be backpropagated together. Pass `densification=torch.zeros(2, N, device="cuda")` to accumulate 3DGS densification statistics (visibility count, screen-space gradient norm) during the backward.

`load_splat(path)` returns these raw parameters from a `.splat` file, and `render_gsplat(...)` renders the same parameters with gsplat for comparison. Python examples (interactive viser viewer, orbit video, benchmark) are in `bindings/torch/examples/`, e.g. `pixi run -e torch python bindings/torch/examples/viewer.py --backend mojo`.

Current limitations: 3 color channels, `cuda:0`, no gradients w.r.t. the camera or background, and each call synchronizes the torch stream (no stream overlap yet).

## Performance

RTX 5090 (idle), bicycle.splat (6.1M Gaussians), 1280x720, camera at (0, 0, 5) with fx = 0.9 W, SH degree 0:

| Renderer | Full frame | |
| -------- | ---------- | --- |
| **mojo `Rasterizer`** (pure Mojo, `pixi run bench`) | **1.21 ms** (828 FPS) | median |
| **mojo `render_splats`** (via torch) | **1.28 ms** (784 FPS) | mean |
| gsplat `rasterization` (via torch, incl. exp/sigmoid activations) | 2.59 ms (387 FPS) | mean |
| previous gsplat-port pipeline (pure Mojo, removed) | 2.32 ms (432 FPS) | median |

Per-stage time for `Rasterizer` from `pixi run bench`, in ms with a sync after each stage:

| Stage | Time | Notes |
|---|---|---|
| preprocess | 0.34 | 1.27M visible Gaussians |
| depth sort | 0.18 | |
| instances | 0.09 | 2.46M tile instances |
| tile sort | 0.13 | |
| blend | 0.49 | |

Training step: forward + backward of all raw parameters through torch autograd. Same scene, timed with `benchmark_render.py`:

| | Forward + backward |
|---|---|
| **mojo `render_splats`** | **4.00 ms** |
| gsplat `rasterization` | 8.29 ms |

In pure Mojo the training forward is 1.42 ms and the backward 1.60 ms. Gradients match float64 autograd of the brute-force reference to ~1e-6 relative.

Where the speed comes from:
- **Blending** (1.55 → 0.49 ms):
  - Each warp only composites the Gaussians that overlap its 8×4 sub-tile.
  - Warps walk the tile list independently, so there are no block barriers. This is our change to Faster-GS's batched kernel and is worth about 20%.
- **Exact culling** gives 12% fewer tile instances.
- **Two-stage sort**: 2 passes over 16-bit tile keys instead of 6 passes over 64-bit keys.
- **Compact 32-byte records**: per-Gaussian data is written in visible order (one memory sector per gather). It fits in L2.

## Contributing

Contributions are very welcome! This is an experimental project exploring the intersection of Mojo and high-performance graphics.

Areas where help is needed:

- **PyTorch Rasterization**: Native PyTorch rasterization reference (`torch` backend currently falls back to gsplat)
- **Performance Optimization**: Analyse current implementation and improve existing Mojo kernels. For example, try to udnersdtand how the generated PTX compares with GSplat and how we can get closer or surpass its performance. 
- **Training loop**: fused Adam, densification (MCMC or 3DGS), losses (L1 + SSIM) and a COLMAP loader in Mojo, to train without Python.
- **Testing**: More comprehensive test coverage
- **Unscented Projection**: Implmeent the Unscented projection from 3DGUT as an alternative to EWA

To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

[MIT License](LICENSE). The `Rasterizer` kernels are ported from [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting) (Apache-2.0); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). MojoSplat only depends on Mojo and MAX as installed packages; MAX is distributed by Modular under the [MAX License](https://github.com/modular/modular/blob/main/Licenses/MAX_LICENSE) (free, no device restrictions). Users install it themselves through pixi, so nothing from MAX is redistributed here.

## Acknowledgments

- [Faster-GS](https://fhahlbohm.github.io/faster-gaussian-splatting) (Hahlbohm, Franke, Eisemann, Magnor) for the forward pipeline design and kernels behind `Rasterizer`
- [StopThePop](https://github.com/r4dl/StopThePop-Rasterization) (Radl et al.) for the exact Gaussian/tile test and [Splatshop](https://github.com/m-schuetz/Splatshop) (Schütz et al.) for the two-stage sort, as adopted by Faster-GS
- [GSplat](https://github.com/nerfstudio-project/gsplat) for the reference implementation
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for the original method
- [Modular](https://www.modular.com/) for the Mojo language

