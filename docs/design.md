# MojoSplat design

Architecture of the library and the plan for training 3D Gaussian Splatting
models entirely in Mojo (no Python). Prior art studied:
[Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting)
(Apache-2.0, the source of the rasterizer kernels),
[LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) (GPL-3.0: ideas only, no code),
[llm.c](https://github.com/karpathy/llm.c), [tinygrad](https://github.com/tinygrad/tinygrad)
and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Layout

```
mojosplat/      the library: kernels + model, no CLI or dataset policy
  core/         plain types: pointers, Camera, SplatView view, SH constants
  gpu/          generic GPU building blocks: scan, radix sort, fills, arenas
  render/       the rasterizer: forward (inference + training) and backward
  model/        owning storage of the Gaussians
  io/           raw arrays, .splat/.ply, COLMAP, JPEG (libjpeg-turbo FFI), images
  train/        dataset (done); loss, metrics, optimizer, densification, trainer (planned)
tools/          command-line programs (render, bench; planned: train, eval)
examples/       short API demos
tests/          mirrors the package; golden data + generators in tests/data/
dev/            (planned) kernel experiments and sweeps that never ship
bindings/torch/ PyTorch extension (render_splats with autograd)
```

Dependencies point one way: `core` <- `gpu` <- `model` <- `render` <- `train`
<- `tools`. `gpu/` knows nothing about splats; `render/` is usable for
inference without any training code. Executables live in `tools/` (as in
llama.cpp), each a thin `main` around a `run(args)` function so they can later
be combined into one `mojosplat <command>` binary. Code shared only by
programs (argument parsing, progress output) goes in `tools/common/`, not in
the library.

## Rasterizer

Faster-GS's pipeline, see the module docstrings in `render/`:
preprocess (activations, EWA projection, exact tile culling, SH, compaction
into slots with a 32-byte record) -> depth sort -> instances -> stable 16-bit
tile sort -> tile ranges -> blend (warp-independent sub-tiles). The training
forward additionally saves per-bucket (color, T) snapshots; the backward is
per-Gaussian (one warp per 32 instances, pixel state passed by shuffles) plus
a coalesced preprocess backward in Gaussian order. State that must outlive a
call comes from a `StateAllocator` (`gpu/arena.mojo`).

## Training (planned)

### SplatStore: one owner for every per-Gaussian row

The central abstraction, in `model/`. It owns, for capacity C >= n Gaussians:

- the six raw parameter arrays (means, log scales, quaternions, opacity
  logits, sh0, shN),
- when training: their Adam moments (m, v) and the densification statistics.

Capacity grows by 1.5x; `n` is the live count. The row operations are
implemented **only here** and apply to every array at once:

- `append(rows)` / `gather_append(index)` for clone and split,
- `compact(keep_mask)` for pruning (stream compaction with `gpu/scan`),
- `permute(order)` for Morton (Z-order) reordering,
- `reset_moments(rows)`.

The renderer keeps taking a `SplatView` view; the Adam kernel sees the same
per-attribute layout. Densification strategies only call store operations,
which keeps the bookkeeping that LichtFeld Studio spreads over its model,
optimizer and each strategy in one place, and makes "one more per-Gaussian
array" a one-line change.

### Optimizer

Adam with per-attribute learning rates (means decay exponentially; SH rest at
1/20 of the DC rate) and bias corrections computed on the host (as llm.c,
ggml and Faster-GS do). First a standalone kernel: one 2D launch over all
attributes, per-attribute settings in a small packed argument, and the
per-element update as a separate `@always_inline` function. Later the same
function is fused into the preprocess backward (Faster-GS's biggest training
speedup after the per-Gaussian backward).

### Densification

A small interface (`post_backward(iter)`, `is_refining(iter)`), not owning the
optimizer. Classic 3DGS first, with Faster-GS's defaults: densify from
iteration 600 to 14,900 every 100 (gradient threshold 2e-4, percent dense
0.01), opacity reset every 3,000 iterations plus once at 500, Morton
reordering every 5,000 until 15,000. Pruning compacts at densification steps.
MCMC can be added later as a second strategy on the same store operations.

### Loss, data, I/O

- Loss: fused L1 + D-SSIM forward and backward (port of fused-ssim, MIT),
  0.8 L1 + 0.2 D-SSIM.
- Metrics: PSNR and SSIM in Mojo; LPIPS computed afterwards in Python until
  we have our own VGG.
- Data: COLMAP binary reader, JPEG decoding via libjpeg-turbo (TurboJPEG 3
  API through Mojo's C FFI), every 8th image held out, images preloaded to
  the GPU.
- Export: PLY in the standard 3DGS layout; checkpoints as safetensors
  (parameters) plus a separate optimizer-state file (as llm.c).

### Testing

1. Kernels: golden data from float64 references (as now).
2. Optimizer: several steps against `torch.optim.Adam` with per-group rates.
3. Store row operations: against numpy indexing on all arrays.
4. Training: a few steps' loss against a Python reference, and save ->
   resume -> continue with tolerances (float atomics make the backward
   non-bitwise-deterministic).

## Roadmap

0. Refactor into this layout, remove the gsplat-compatible pipeline. **Done.**
1. Data: COLMAP reader, JPEG decode, PLY read/write, dataset. **Done.**
2. SplatStore + Adam, fused L1+SSIM, PSNR, minimal trainer without
   densification (loss must drop on Mip-NeRF360 garden).
3. Classic densification, schedules, kNN scale initialization; full 30k run
   on garden vs published per-scene PSNR.
4. Performance: Adam fused into the backward, Morton reordering; compare with
   Faster-GS's 2m43s Mip-NeRF360 average on an RTX 5090.
5. Evaluation tool over the Mip-NeRF360 scenes; README.
