# MojoSplat

> [!NOTE]
> **Work in Progress - Experimental**

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
| **Binning** | ✅ | ✅ | ❌ (WIP) |
| **Rasterization** | ❌* | ✅ | ✅ |

*PyTorch rasterization falls back to GSplat implementation

> [!WARNING]
> 1. This is NOT production ready or even finished.
> 2. Due to Mojo compiler limitations the compiler optimizes on tensor shapes. This causes some kernels to need to be recompiled if either the number of gaussians change or the view matrix changes.
> 3. Performance is inferior to the GSplat CUDA version. Maybe some day we will be capable of surpassing it.
> 4. Mojo is evolving very fast. Faster than I work on this (this is very much a side project). So thsi projects will likely not be up to date with latest Mojo all the time as each update requires a non insignificant amount of work. Particularly the Mojo interop with python/torch is a very novel thigns and the API is changing with every version.


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

### Generate a random image
```bash
uv run render_sample.py
```

### Basic Rendering

```python
import torch
from mojosplat.render import render_gaussians
from mojosplat.projection import Camera

# Your 3D Gaussian data
means3d = torch.randn(1000, 3, device='cuda')
scales = torch.randn(1000, 3, device='cuda') 
quats = torch.randn(1000, 4, device='cuda')
opacities = torch.randn(1000, 1, device='cuda')
features = torch.randn(1000, 3, device='cuda')  # RGB colors

# Set up camera
camera = Camera(...)  # Configure your camera parameters

# Render with different backends
image_mojo = render_gaussians(means3d, scales, quats, opacities, features, camera, backend="mojo")
image_gsplat = render_gaussians(means3d, scales, quats, opacities, features, camera, backend="gsplat")  
image_torch = render_gaussians(means3d, scales, quats, opacities, features, camera, backend="torch")
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
# Benchmark projection kernels
uv run python examples/benchmark_proj.py

# Benchmark full rendering pipeline  
uv run python examples/benchmark.py
```

### Performance on RTX 2080

*Benchmarks coming soon - performance data will be added once comprehensive testing is complete.*

## Contributing

Contributions are very welcome! This is an experimental project exploring the intersection of Mojo and high-performance graphics.

Areas where help is needed:
- **Mojo Binning Kernel**: Complete the binning implementation in Mojo
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

[License information to be added]

## Acknowledgments

- [GSplat](https://github.com/nerfstudio-project/gsplat) for the reference implementation
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for the original method
- [Modular](https://www.modular.com/) for the Mojo language
