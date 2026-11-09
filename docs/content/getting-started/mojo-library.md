---
title: Using the Mojo library
sidebar_position: 2
---

# Using the Mojo library

`mojosplat` is a plain Mojo package. Run your program with the repository on the
import path:

```bash
pixi run mojo run -I . my_program.mojo
```

## Render a model

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

## Train

```mojo
var trainer = Trainer(ctx, load_colmap_dataset(ctx, path), TrainConfig())
for _ in range(30_000):
    _ = trainer.step()
```

## Gradients for your own loss

`Rasterizer.forward` renders and returns what the backward needs.
`Rasterizer.backward` takes dL/d(image) from any image loss you compute and
writes the gradients of every raw parameter:

```mojo
var state = rasterizer.forward(splats.view(), camera, background, image_ptr)
# ... compute your loss and write dL/d(image) into grad_image_ptr ...
rasterizer.backward(splats.view(), state, image_ptr, grad_image_ptr, splats.grad_view())
```

Call `splats.enable_training()` first, so the gradient buffers exist. Keep the
parameters unchanged between the two calls, because the backward recomputes the
per-Gaussian math from them.
