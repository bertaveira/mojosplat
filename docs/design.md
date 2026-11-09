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
  core/         plain types: pointers, Camera, SplatView view, linalg, SH constants
  gpu/          generic GPU building blocks: scan, radix sort, fills, arenas
  render/       the rasterizer: forward (inference + training) and backward
    blocks/     per-Gaussian math behind the kernels; start with contracts.mojo
  model/        owning storage of the Gaussians
  io/           raw arrays, .splat/.ply, COLMAP, JPEG (libjpeg-turbo FFI), images
  train/        dataset, loss (L1 + SSIM), PSNR, Adam, densification, trainer
  cli/          command-line options from config structs, terminal output
  viewer/       web viewer: HTTP + WebSocket server, JPEG frames
tools/          command-line programs: train, eval, render, bench
tests/          mirrors the package; golden data + generators in tests/data/
bindings/torch/ PyTorch extension (render_splats with autograd)
```

Dependencies point one way: `core` <- `gpu` <- `model` <- `render` <- `train`
<- `tools`. `gpu/` knows nothing about splats; `render/` is usable for
inference without any training code. Executables live in `tools/` (as in
llama.cpp) and stay thin: argument parsing and output around library calls.

`cli/` depends only on the standard library. `parse_args[T]()` turns the
fields of a config struct into options by compile-time reflection (in the
spirit of tyro): `sh_degree` becomes `--sh-degree`, Bools are switches,
nested structs are option groups, and `@__annotation(Arg("help"))` on a
field gives its help text. Configs such as `TrainConfig` carry these
annotations themselves, so a new field is documented and settable from the
command line with no tool changes. `@__annotation` is Mojo's field-metadata
syntax; the leading underscores suggest it may still change.

`viewer/` is a small single-threaded web server written in Mojo over libc
sockets (`poll` + MSG_DONTWAIT; no variadic C calls, which Mojo's FFI cannot
make portably on Apple arm64). It serves one embedded HTML/JS page; the page
sends its camera over a WebSocket and gets JPEG frames back (rendered with
the viewer's own `Rasterizer`, quantized on the GPU, encoded with
libjpeg-turbo). The owner calls `update()` between training steps: a frame
is rendered when a browser's camera moved (up to `--max-fps`) or a few
times a second while training changes the scene (`--refresh-fps`), and
only once the previous frame was fully sent, so slow connections get fewer
frames instead of slowing training. Measured on garden: no overhead with no
browser connected, ~2% while watching at 5 fps. lightbug_http, the main
Mojo HTTP library, targets Mojo 0.26 and has no WebSockets; what the viewer
needs (one page and the WebSocket upgrade) is ~150 lines.

## Portability

The same sources build for NVIDIA, AMD (RDNA3, RDNA4, MI300) and Apple
M-series GPUs. Warp-level code is written for any warp size: lane masks use
`LANE_MASK_DTYPE` (uint64 on wave64 MI300) and the default tile height is 16
there instead of 12. Platform differences on the host side are
`comptime if CompilationTarget.is_macos()` branches.

Rules for kernels, all from Mojo 1.2 on Metal (Apple M1 Pro):

- **Vector kernel arguments are `KernelVector[dtype, width]`, never a bare
  `SIMD`.** Mojo lowers a by-value SIMD argument to an AIR vector parameter,
  which Apple's pipeline compiler rejects at runtime. `KernelVector`
  (`core/types.mojo`) wraps the same bytes in a struct that is passed in a
  constant buffer; on NVIDIA the generated code is unchanged. Nothing on
  Linux catches a violation: it compiles, cross-compiles for `apple-m4`
  (`--emit air`) and passes every test, and fails only when run on a Mac.
- **Build SIMD values that hold device loads lane by lane**
  (`v = SIMD(0); v[k] = p[unsafe_offset=...]`, see `load_rgb` in `render/blocks/appearance/sh.mojo`).
  A variadic SIMD constructor that mixes device loads and local values
  lowers with mismatched address spaces and crashes Apple's `air-lld`.
- **Raw-pointer fills and copies use kernels on macOS**
  (`gpu/elementwise.mojo`). The Metal driver's fill and copy need its own
  buffer handles and cannot wrap an arbitrary pointer in a non-owning
  `DeviceBuffer`.
- **Keep a buffer's owner alive until the GPU is done with it.** `dptr()`
  erases the origin, so Mojo may free the buffer right after its last use on
  the host while a kernel still reads it. Some tests got away with this on
  CUDA; the macOS port fixed them. Keep owners in structs, or end with
  `_ = (buf^, ...)` after a synchronize.

`tests/gpu/test_kernel_args.mojo` exercises `KernelVector` arguments of
several widths and the lane-by-lane loads.

## Rasterizer

Faster-GS's pipeline, see the module docstrings in `render/`:
preprocess (activations, EWA projection, exact tile culling, SH, compaction
into slots with a 32-byte record) -> depth sort -> instances -> stable 16-bit
tile sort -> tile ranges -> blend (warp-independent sub-tiles). The training
forward additionally saves per-bucket (color, T) snapshots; the backward is
per-Gaussian (one warp per 32 instances, pixel state passed by shuffles) plus
a coalesced preprocess backward in Gaussian order. State that must outlive a
call comes from a `StateAllocator` (`gpu/arena.mojo`).

## Blocks and pipelines

The render code separates GPU work from the math for one Gaussian:

- **Kernels** (`render/*.mojo`) own threads, culling, memory, tile lists and
  compositing. The blend kernels stay hand-tuned.
- **Blocks** (`render/blocks/`) hold the math, forward and backward together.
  A block's constructor runs its forward and keeps what its backward needs in
  its fields, so a kernel's local variables are the tape. Everything is inlined;
  fields the backward doesn't use are removed by the compiler.
- A **pipeline** (`render/pipelines.mojo`) names one block of each kind.

`blocks/contracts.mojo` defines the three kinds:

| Trait | Question | Today |
|---|---|---|
| `Placement` | Where does a Gaussian land on the image? | `PinholeEWA` |
| `Response` / `LoadedResponse` | What does a pixel see of it? | `Gaussian2D` |
| `Appearance` | What color does it have? | `SphericalHarmonics` |

Nothing is registered or looked up at run time. Kernels take the pipeline as a
compile-time parameter (`_preprocess_kernel[..., Pipe]`), and Mojo generates one
kernel per pipeline named in the code. `Rasterizer.render`, `forward` and
`backward` take it per call (`rast.forward[SomePipeline](...)`) and default to
`Standard3DGS`; a backward using another pipeline than its forward raises.

A response or appearance owns `Splats` columns after the core ones. Kernels
receive them as `ExtraColumns`, up to four pointers, with the appearance's first.
A response can also save per-slot values besides the record (`EXTRA`) and has its
own slot gradients (`GRAD`, `grad_index`). The per-slot record is otherwise fixed
at 32 bytes, and blending at three channels.

To add a block, implement one trait, name it in a pipeline, and add it to
`tests/render/test_block_gradients.mojo`. Mojo checks generic code only when it
is instantiated, so a block that no test instantiates may not compile.

The current `Response` contract fits footprints evaluated in 2D. Evaluating along
a ray with per-hit data (3DGUT, Neural Harmonic Textures) will need that contract
extended or its own blend kernel.

## Training

`Trainer` (`train/trainer.mojo`) runs the 3DGS recipe with Faster-GS's
defaults (`TrainConfig`); its docstring lists the order of one iteration.

### Splats: one owner for every per-Gaussian row

The central abstraction, `Splats` in `model/` (`SplatView` is its
non-owning pointer view, `HostSplats` the CPU copy for file I/O). It owns,
for capacity C >= n Gaussians, every per-Gaussian array as a list of `Column`s
(name, width, trainable):

- the four core columns (means, log scales, quaternions, opacity logits), then
  those of the appearance (sh0 and shN for spherical harmonics) and the response.
  `Attr` holds the indices of the standard ones,
- after `enable_training()`, for trainable columns: their gradients and Adam
  moments (m, v), plus the densification statistics.

`add_column` adds a column. Columns beyond the standard ones are saved in PLY
files as `extra_<name>_<i>` properties.

There is one row operation, `gather(index, n_out, zero_moments_from)`,
applied to every array at once: clone and split append rows (the index is
all rows followed by the copies; moments of the new rows are zeroed),
pruning keeps the surviving rows, Morton reordering permutes them. The
renderer takes a `SplatView`; the Adam kernel sees the same per-attribute
layout. This keeps the bookkeeping that LichtFeld Studio spreads over its
model, optimizer and each strategy in one place, and makes "one more
per-Gaussian array" a one-line change. `reset_moments` zeroes the Adam moments of
arbitrary rows, and a permutation can keep the densification statistics.

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

1. Kernels: golden data from float64 references. Blocks: finite-difference
   checks of every backward, one output at a time
   (`tests/render/test_block_gradients.mojo`).
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
