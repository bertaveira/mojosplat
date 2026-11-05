# MojoSplat

MojoSplat is an experimental port of Gaussian Splatting kernels to [Mojo](https://www.modular.com/mojo), exploring the potential performance and multi-vendor support of Mojo for GPU acceleration.



https://github.com/user-attachments/assets/ae4409a3-a656-4635-bb57-f22aca2f6141



MojoSplat implements the forward pass of 3D Gaussian Splatting **entirely in Mojo**:

- **Projection**: EWA projection of 3D Gaussians to 2D conics (matches gsplat's `fully_fused_projection`)
- **Binning**: tile intersection counting, a device-wide prefix scan, and a GPU radix sort of `(tile, depth)` keys
- **Rasterization**: tile-based front-to-back alpha compositing

The core library (`mojosplat/`) has **no Python dependency**: you can load a `.splat` file, render it and write a PNG from a Mojo program. PyTorch support lives separately in [`bindings/torch`](bindings/torch) (in the spirit of tiny-cuda-nn), where torch tensors are handed to the Mojo kernels by pointer, without copies.

> [!WARNING]
> 1. This is NOT production ready (forward pass only, NVIDIA GPUs only so far).
> 2. Mojo is evolving fast. This version targets **Mojo 1.2 / MAX 26.7**.

## Repository layout

| Path | What |
|------|------|
| `mojosplat/` | Pure Mojo package: `projection`, `scan`, `sort`, `binning`, `rasterization`, `render` (`Renderer`), `camera`, `io` (`.splat` loader, PNG/PPM writer) |
| `examples/` | Mojo programs: `render_splat.mojo`, `benchmark.mojo` |
| `tests/` | Mojo test-suite, validated against golden data in `tests/data/golden` (generated from the previous, gsplat-matched pipeline) |
| `bindings/torch/` | `mojosplat_torch` Python package: Mojo extension module + torch/gsplat reference backends, pytest suite, Python examples (viewer, orbit video, benchmark) |

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
pixi run bench -n 200                            # full pipeline + per-stage timings
pixi run test                                    # Mojo test-suite
```

From your own Mojo code:

```mojo
from max.gpu.host import DeviceContext
from mojosplat.camera import Camera
from mojosplat.io import download, load_splat, write_png
from mojosplat.render import GaussianBuffers, Renderer

def main() raises:
    var ctx = DeviceContext()
    var gaussians = GaussianBuffers.from_scene(ctx, load_splat("examples/bicycle.splat"))
    var camera = Camera.translated(0.0, 0.0, 5.0, 1280, 720, 1152.0, 1152.0)
    var image = ctx.enqueue_create_buffer[DType.float32](1280 * 720 * 3)
    var renderer = Renderer(ctx)          # owns and reuses all intermediate buffers
    renderer.render(gaussians, camera, SIMD[DType.float32, 4](0.0), image)
    write_png("out.png", download(ctx, image, 1280 * 720 * 3), 1280, 720)
```

Run it with `pixi run mojo run -I . my_program.mojo`.

## PyTorch bindings

```bash
pixi install -e torch          # adds torch, gsplat (reference), pytest
pixi run -e torch build-ext    # builds bindings/torch/mojosplat_torch/_mojosplat.so
pixi run -e torch test-torch   # pytest suite: mojo vs gsplat vs torch backends
```

```python
import torch
from mojosplat_torch import Camera, render_gaussians

# All inputs: float32 CUDA tensors on cuda:0. scales in log-space, quats (w, x, y, z), opacities (N,)
camera = Camera(R=torch.eye(3, device="cuda"), T=torch.tensor([0.0, 0.0, 5.0], device="cuda"),
                H=720, W=1280, fx=1152.0, fy=1152.0, cx=640.0, cy=360.0)
image = render_gaussians(means3d, scales, quats, opacities, colors, camera, backend="mojo")  # (H, W, 3)
```

The individual stages (`project_gaussians`, `bin_gaussians_to_tiles`, `rasterize_gaussians`) are also exposed, each with `torch`, `gsplat` and `mojo` backends. Python examples (interactive viser viewer, orbit video, benchmark) are in `bindings/torch/examples/`, e.g. `pixi run -e torch python bindings/torch/examples/viewer.py`.

Current limitations of the mojo backend: `tile_size=16`, 3 color channels, `cuda:0`, and each call synchronizes the torch stream (no stream overlap yet).

## Performance

Previous version (modular 26.2, torch-driven, RTX 5090, bicycle.splat, 6.1M Gaussians, 1280x720):

| Backend    | Full pipeline       | Projection | Binning | Rasterization |
| ---------- | ------------------- | ---------- | ------- | ------------- |
| **gsplat** | 2.41 ms (414.9 FPS) | 0.43 ms    | 0.46 ms | 1.56 ms       |
| **mojo**   | 2.68 ms (373.7 FPS) | 0.49 ms    | 0.73 ms | 1.55 ms       |

The binning gap came from `torch.sort` processing all 64 key bits. The 1.2 port replaces `torch.cumsum`/`torch.sort` with a Mojo device scan and an 8-bit LSD radix sort that only sorts `32 + bit_width(num_tiles)` bits (6 passes at 1280x720). Fresh numbers for the 1.2 version will be added once measured on an idle GPU.

## Contributing

Contributions are very welcome! This is an experimental project exploring the intersection of Mojo and high-performance graphics.

Areas where help is needed:

- **PyTorch Rasterization**: Native PyTorch rasterization reference (`torch` backend currently falls back to gsplat)
- **Performance Optimization**: Analyse current implementation and improve existing Mojo kernels. For example, try to udnersdtand how the generated PTX compares with GSplat and how we can get closer or surpass its performance. 
- **Backwards pass**: implement the mojo kernels for the backwards pass. This will allow the MojoSplat to be used in training the gaussian representation.
- **Testing**: More comprehensive test coverage
- **Unscented Projection**: Implmeent the Unscented projection from 3DGUT as an alternative to EWA

To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

[MIT License](LICENSE). MojoSplat only depends on Mojo and MAX as installed packages; MAX is distributed by Modular under the [MAX License](https://github.com/modular/modular/blob/main/Licenses/MAX_LICENSE) (free, no device restrictions). Users install it themselves through pixi, so nothing from MAX is redistributed here.

## Acknowledgments

- [GSplat](https://github.com/nerfstudio-project/gsplat) for the reference implementation
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for the original method
- [Modular](https://www.modular.com/) for the Mojo language

