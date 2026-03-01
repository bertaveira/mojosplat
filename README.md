# MojoSplat

MojoSplat is an experimental port of Gaussian Splatting kernels to [Mojo](https://www.modular.com/mojo), exploring the potential performance and multi-vendor support of Mojo for GPU acceleration.

This project implements the three core kernels of 3D Gaussian Splatting:
- **Projection**: Transform 3D Gaussians to 2D image space
- **Binning**: Sort and assign Gaussians to screen tiles  
- **Rasterization**: Render Gaussians to pixels with alpha blending

You can call the render function or any of the individual kernels directly from python (using pytorch). The mojo kernels will be compiled on the fly.

## Implementation Status

| Kernel | PyTorch | GSplat | Mojo |
|--------|---------|--------|------|
| **Projection** | ✅ | ✅ | ✅ |
| **Binning** | ✅ | ✅ | ✅ |
| **Rasterization** | ❌* | ✅ | ✅ |

*PyTorch rasterization falls back to GSplat implementation

> [!WARNING]
> 1. This is NOT production ready.
> 2. Performance is inferior to the GSplat CUDA version. Maybe some day we will be capable of surpassing it.
> 3. Mojo is evolving very fast. Faster than I work on this (this is very much a side project). So thsi projects will likely not be up to date with latest Mojo all the time as each update requires a non insignificant amount of work. Particularly the Mojo interop with python/torch is a very novel thigns and the API is changing with every version.


## Installation

### Standalone Development (with uv)

For development or standalone usage, this project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
# Clone the repository
git clone https://github.com/bertaveira/mojosplat.git
cd mojosplat

# Install dependencies and activate environment
uv sync
```

### As a Dependency in Your Project

#### Using pip with GitHub
```bash
pip install git+https://github.com/bertaveira/mojosplat.git
```

#### Using uv in your project
Add to your `pyproject.toml`:
```toml
dependencies = [
    "mojosplat @ git+https://github.com/bertaveira/mojosplat.git",
    # ... your other dependencies
]
```

#### Using pip requirements.txt
Add to your `requirements.txt`:
```
git+https://github.com/bertaveira/mojosplat.git
```

#### Using conda/mamba environment.yml
```yaml
dependencies:
  - pip
  - pip:
    - git+https://github.com/bertaveira/mojosplat.git
```

## Usage

### Basic Rendering

All inputs must be CUDA tensors (`float32`). `scales` are in log-space; `quats` are `(w, x, y, z)`; `opacities` shape is `(N,)`.

```python
import torch
from mojosplat.render import render_gaussians
from mojosplat.utils import Camera

# 3D Gaussian data (e.g. from your scene or .splat file)
N = 1000
device = "cuda"
means3d = torch.randn(N, 3, device=device, dtype=torch.float32)
scales = torch.randn(N, 3, device=device, dtype=torch.float32)   # log-space
quats = torch.randn(N, 4, device=device, dtype=torch.float32)     # (w, x, y, z)
quats = quats / quats.norm(dim=1, keepdim=True)
opacities = torch.randn(N, device=device, dtype=torch.float32)   # (N,) not (N, 1)
features = torch.randn(N, 3, device=device, dtype=torch.float32) # RGB

# Camera: R (3,3) world-to-camera, T (3,) world-to-camera, H, W, fx, fy, cx, cy
R = torch.eye(3, device=device, dtype=torch.float32)
T = torch.tensor([0.0, 0.0, 5.0], device=device, dtype=torch.float32)
camera = Camera(R=R, T=T, H=720, W=1280, fx=1152.0, fy=1152.0, cx=640.0, cy=360.0)

# Render (backend: "mojo", "gsplat", or "torch")
image = render_gaussians(means3d, scales, quats, opacities, features, camera, backend="mojo")
# image shape: (H, W, C)
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific kernel tests
uv run pytest tests/test_projection_mojo.py
uv run pytest tests/test_binning.py  
uv run pytest tests/test_rasterization.py
uv run pytest tests/test_render.py

# Run with verbose output
uv run pytest -v
```

### Benchmarking

```bash
# Benchmark with a real .splat scene (e.g. bicycle)
# First download a .splat file (antimatter15 binary format):
curl -L -o examples/bicycle.splat https://huggingface.co/cakewalk/splat-data/resolve/main/bicycle.splat

# Then run the benchmark (defaults to examples/bicycle.splat if present)
uv run python examples/benchmark_render.py examples/bicycle.splat
```

### Interactive viewer

You can view a `.splat` scene in the browser with the interactive viewer (drag to orbit, scroll to zoom). Use the same `bicycle.splat` file as above:

```bash
# Ensure you have the scene file (see Benchmarking above for download URL)
uv run python examples/viewer.py examples/bicycle.splat
```

Then open the URL printed in the terminal in your browser. The first render triggers JIT compilation (~30–60 s); subsequent renders are fast.

### Performance (RTX 5090, bicycle.splat, 6.1M Gaussians, 1280×720)

Benchmark: `uv run python examples/benchmark_render.py examples/bicycle.splat` (1000 runs full pipeline, 200 runs per kernel).

| Backend | Full pipeline | Projection | Binning | Rasterization |
|---------|---------------|------------|---------|---------------|
| **gsplat** | 2.41 ms (414.7 FPS) | 0.43 ms | 0.46 ms | 1.56 ms |
| **mojo**   | 6.96 ms (143.7 FPS) | 2.15 ms | 0.87 ms | 4.03 ms |


## Contributing

Contributions are very welcome! This is an experimental project exploring the intersection of Mojo and high-performance graphics.

Areas where help is needed:
- **PyTorch Rasterization**: Native PyTorch rasterization kernel
- **Performance Optimization**: Analyse current implementation and improve existing Mojo kernels. For example, try to udnersdtand how the generated PTX compares with GSplat and how we can get closer or surpass its performance. Also measure the overhead of the python to mojo connection.
- **Backwards pass**: implement the mojo kernels for the backwards pass. This will allow the MojoSplat to be used in training the gaussian representation.
- **Testing**: More comprehensive test coverage
- **Unscented Projection**: Implmeent the Unscented projection from 3DGUT as an alternative to EWA

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

[MIT License](LICENSE)

## Acknowledgments

- [GSplat](https://github.com/nerfstudio-project/gsplat) for the reference implementation
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for the original method
- [Modular](https://www.modular.com/) for the Mojo language
