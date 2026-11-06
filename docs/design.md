# MojoSplat design

Architecture of the library and of training 3D Gaussian Splatting models
entirely in Mojo (no Python). Prior art studied:
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
  train/        dataset, loss (L1 + SSIM), PSNR, Adam, densification, trainer
tools/          command-line programs: train, eval, render, bench
tests/          mirrors the package; golden data + generators in tests/data/
bindings/torch/ PyTorch extension (render_splats with autograd)
```

Dependencies point one way: `core` <- `gpu` <- `model` <- `render` <- `train`
<- `tools`. `gpu/` knows nothing about splats; `render/` is usable for
inference without any training code. Executables live in `tools/` (as in
llama.cpp) and stay thin: argument parsing and output around library calls.

## Rasterizer

Faster-GS's pipeline, see the module docstrings in `render/`:
preprocess (activations, EWA projection, exact tile culling, SH, compaction
into slots with a 32-byte record) -> depth sort -> instances -> stable 16-bit
tile sort -> tile ranges -> blend (warp-independent sub-tiles). The training
forward additionally saves per-bucket (color, T) snapshots; the backward is
per-Gaussian (one warp per 32 instances, pixel state passed by shuffles) plus
a coalesced preprocess backward in Gaussian order. State that must outlive a
call comes from a `StateAllocator` (`gpu/arena.mojo`).

## Training

`Trainer` (`train/trainer.mojo`) runs the 3DGS recipe with Faster-GS's
defaults (`TrainConfig`); its docstring lists the order of one iteration.

### Splats: one owner for every per-Gaussian row

The central abstraction, `Splats` in `model/` (`SplatView` is its
non-owning pointer view, `HostSplats` the CPU copy for file I/O). It owns,
for capacity C >= n Gaussians, every per-Gaussian array, indexed by `Attr`:

- the six raw parameter arrays (means, log scales, quaternions, opacity
  logits, sh0, shN),
- after `enable_training()`: their gradients, Adam moments (m, v) and the
  densification statistics.

There is one row operation, `gather(index, n_out, zero_moments_from)`,
applied to every array at once: clone and split append rows (the index is
all rows followed by the copies; moments of the new rows are zeroed),
pruning keeps the surviving rows, Morton reordering permutes them. The
renderer takes a `SplatView`; the Adam kernel sees the same per-attribute
layout. This keeps the bookkeeping that LichtFeld Studio spreads over its
model, optimizer and each strategy in one place, and makes "one more
per-Gaussian array" a one-line change.

### Optimizer

Adam with PyTorch semantics and per-attribute learning rates (means decay
exponentially, scaled by the scene extent; SH rest at 1/20 of the DC rate),
bias corrections computed on the host (as llm.c, ggml and Faster-GS do). The
SH-rest update is fused with its gradient: Adam recomputes
basis(dir) * dL/dcolor elementwise, so the backward never writes it.

### Densification

`Densifier` (`train/densify.mojo`), classic 3DGS with Faster-GS's defaults:
densify from iteration 600 to 14,900 every 100 (gradient threshold 2e-4,
percent dense 0.01, prune opacity < 0.005, and Gaussians larger than 0.1 x
the scene extent after the first opacity reset), opacity reset every 3,000
iterations (Faster-GS resets once more at 500 only with a non-black
background; MojoSplat trains on black), Morton reordering every 5,000 until
15,000 and once at the end, and a final prune of opacity < 1/255. MCMC can
be added later as a second strategy on the same `gather`.

### Loss, data, I/O

- Loss: L1 + D-SSIM forward and backward with separable passes (backward as
  in fused-ssim, MIT), 0.8 L1 + 0.2 D-SSIM.
- Metrics: PSNR in Mojo (`tools/eval.mojo`); SSIM and LPIPS computed from
  its saved renders in Python until we have our own VGG.
- Data: COLMAP binary reader, JPEG decoding via libjpeg-turbo (TurboJPEG 3
  API through Mojo's C FFI), every 8th image held out, images kept on the
  GPU as 8-bit.
- Export: PLY in the standard 3DGS layout. Checkpoints with optimizer state
  are not implemented yet.

### Testing

1. Kernels: golden data from float64 references.
2. Optimizer: several steps against `torch.optim.Adam` with per-group rates.
3. Densification: one step against a torch transcription of Faster-GS's
   `adaptive_density_control`.
4. Training: a short run on garden must lower the loss and raise test PSNR
   (float atomics make the backward non-bitwise-deterministic).

## Roadmap

0. Refactor into this layout, remove the gsplat-compatible pipeline. **Done.**
1. Data: COLMAP reader, JPEG decode, PLY read/write, dataset. **Done.**
2. Splats with capacity + training state, Adam, L1+SSIM, PSNR, kNN
   initialization, trainer without densification. **Done:** garden, 30k
   iterations in 38 s, 24.8 dB test PSNR with the 138k SfM Gaussians.
3. Classic densification (clone/split/prune, opacity reset) on the Splats
   row operations; full 30k run on garden vs published per-scene PSNR.
   **Done:** one `Splats.gather` primitive (append = gather with copies,
   prune = gather of kept rows); a step matches a torch transcription of
   Faster-GS's adaptive_density_control exactly. Garden: 27.43 dB test PSNR
   (3DGS: 27.41), 4.19M Gaussians, 4m28s.
4. Performance: Adam fused into the backward, Morton reordering; compare with
   Faster-GS's 2m43s Mip-NeRF360 average on an RTX 5090.
   **Done (first pass):** garden 268 s -> 211 s. Profile of a steady-state
   step (4.19M Gaussians) went 9.9 -> 7.7 ms:
   - The SH-rest Adam recomputes basis(dir) * dL/dcolor elementwise
     (`Rasterizer.sh_rest_grad` -> `Adam.step(splats, sh_rest)`), so the
     backward never writes that gradient. Its scattered writes had made the
     preprocess backward cost 2.0 ms; it now takes 0.4 ms.
   - Morton reordering: -10-15% on the per-Gaussian kernels.

   Now: Adam ~4.1 ms (at memory bandwidth: dense Adam over 59 floats x 4.2M
   x (p, m, v)), blend backward 1.6 ms (compute-bound warp pipeline).
   Tried and rejected: restricting each blend-backward bucket to the rows
   its 32 Gaussians cover. Their union is almost always the whole 12-row
   tile, so it added instructions and registers (72 -> 87) and was slower.
   Remaining levers change the numerics (user decision): skip Adam for
   invisible Gaussians (Faster-GS option), 16-bit Adam moments (as
   LichtFeld Studio does).
5. Evaluation tool over the Mip-NeRF360 scenes; README.
