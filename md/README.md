# Jules

Jules is a game-dev + ML-oriented language/runtime with:
- ECS + system runtime
- ML tensor engine (CPU + GPU hooks)
- C ABI for embedding in external engines
- lexical borrow-checking pass for reference alias safety (`&` / `&mut`)

For a consolidated list of implemented optimization techniques, see `md/OPTIMIZATIONS.md`.

## Engine / Host Integrations

### C++
Use `bindings/cpp/jules.hpp` and link against the generated `libjules`.

### C# (Unity/Godot C#)
Use `bindings/csharp/Jules.cs` (`DllImport("jules")`) and ship the native library next to the game binary.

### Python
Use `bindings/python/jules.py` (ctypes wrapper) or build with `--features python` for PyO3 bindings.

## C ABI entry points

- `jules_run_file_ffi(const char* path)`
- `jules_check_code_ffi(const char* source)`
- tensor lifecycle + shape/data APIs
- physics sample APIs

## Experimental low-byte inference

Jules now includes an INT8 inference path for linear layers:
- `Tensor::quantize_linear_int8()` (offline weight quantization)
- `Tensor::linear_int8()` (inference with 1-byte weights + per-channel scales)
- `Int8LinearWeights::effective_bytes_per_param()` to estimate memory cost
- `Tensor::softmax_last_dim()` for batched class probabilities
- `Tensor::gelu()` and `Tensor::silu()` activations for Transformer/LMM-style blocks
- `Tensor::gelu_backward()` and `Tensor::silu_backward()` gradients for training
- `LossFunctions::cross_entropy_from_logits_last_dim()` for indexed-class training
- `LossFunctions::cross_entropy_from_logits_last_dim_gradient()` for stable logits gradients

This targets roughly **~1 byte/parameter** plus small scale overhead, for
running (inference) workloads.

## Strict ML memory ceiling (core + extra)

Jules supports an explicit memory-cap model for ML workloads:

- **Core floor** `M_min`: model-critical bytes (weights, gradients, active batch activations)
- **Extra headroom** `ΔM`: runtime scratchpad (prefetch, kernel workspace, temp tensors)

The runtime enforces:

`M_usage <= M_min + ΔM`

This is exposed via FFI/bindings APIs:

- `jules_ml_memory_configure(min_bytes, extra_bytes)`
- `jules_ml_memory_acquire(bytes, pool)` with `Core` / `Extra`
- `jules_ml_memory_release(bytes, pool)`
- `jules_ml_memory_snapshot(...)`

This helps game+ML scenarios avoid OOM crashes, maintain deterministic frame-time behavior,
and control temporary overhead when using INT8 quantization/inference flows.

## Faster CPU matmul path

`Tensor::matmul()` uses a cache-aware blocked kernel for larger matrices
(with existing multi-threading), while keeping the low-overhead unrolled kernel
for smaller shapes.

The CPU fallback in `gpu_backend.rs` now mirrors this strategy for `matmul`
as well (transpose + blocked kernel + threaded row chunks), following the same
core ideas used by high-performance JIT stacks: contiguous access, tiling, and
parallel work partitioning.

## Game-dev movement helpers

Jules now ships additional game-loop-friendly built-ins in `math`:

- `math::approach(current, target, max_delta)` for deterministic scalar movement/easing
- `math::move_towards2(cx, cy, tx, ty, max_delta)` for fixed-step 2D follow behavior
- `math::angle_to(ax, ay, bx, by)` for heading/orientation calculations
- `math::rand_unit2()` for random normalized directions

## Game simulation optimization focus

Jules is now tuned for game simulation workloads around deterministic stepping and
high-frequency spatial queries. The current game-sim optimization profile includes:

- deterministic entity iteration order in `sim` worlds for stable replay/debug behavior
- uniform-grid broadphase collision filtering in `sim::step` to reduce pair checks
- sorted/stable nearest/radius entity query behavior for gameplay logic consistency
- fixed-step-friendly movement helpers (`math::approach`, `math::move_towards2`)
- command-buffered rendering API (`render::*`) for predictable frame submission
- strict ML memory cap controls (`M_min + ΔM`) for mixed game+ML runtime stability

## Rendering API with AoT-friendly command streams

Jules now includes a `render::*` API that records frame commands into a structured
command buffer:

- `render::begin_frame(width, height)`
- `render::clear(r, g, b, a)`
- `render::rect(x, y, w, h, r, g, b, a, layer)`
- `render::sprite(sprite_id, x, y, w, h, rotation_deg, layer)`
- `render::flush()` (drain queued commands for host-side execution)
- `render::stats()` (queue + frame metrics)

This keeps rendering deterministic at script-level and makes AoT/embedded engine
integration easier because command data is plain structured values.



## Standard built-in library modules

Jules now exposes a structured built-in stdlib with module namespaces (`core`, `math`,
`tensor`, `nn`, `train`, `data`, `io`, `sys`, `error`, `diag`, `collections`,
`compute`, `quant`, `model`, `debug`, `sim`, `window`) and runtime discovery via `std::modules()`.
See `md/STDLIB.md` for module details and philosophy.

## JAX backend for optimized ML training

For the highest-throughput ML training path, use the JAX bridge script:

```bash
python scripts/jules_jax_backend.py --ir model_ir.json --dataset train.npz --out artifacts/jax
```

This path uses XLA-compiled training (`jax.jit`) and exports INT8-per-channel weights
compatible with Jules low-byte inference. Full guide: `md/JAX_BACKEND.md`.

You can also choose backend directly from Jules CLI:

```bash
# Native Jules training runtime (default)
jules train my_agent.jules --ml-backend jules

# JAX training backend, reusing model syntax from the same .jules file
jules train my_agent.jules --ml-backend jax --jax-dataset train.npz --jax-out artifacts/jax
```

## Automated syntax fix flow

Use:

```bash
jules fix path/to/file.jules
```

`jules fix` applies safe automatic syntax fixes from parser diagnostics
(e.g. missing `;`, missing `)`, `]`, `}`, missing `,`, assignment/operator
replacement like `==`→`=`, return arrow insertion, block opener insertion,
and common `fun`/`func` → `fn` keyword typo recovery).

## ML calculator (`jules estimate`)

`jules estimate` computes an approximate training ETA by modeling two independent
throughput limits:

- **sim throughput** (environment stepping rate, scaled by env count + device)
- **model throughput** (forward/backward/update rate, scaled by params + batch)

Final `steps/s` is the lower of the two (the current bottleneck). Output now includes:

- `sim≈...`
- `model≈...`
- `bottleneck=sim|model`

This keeps the calculator simple while making bottlenecks explicit for tuning.
It also emits actionable suggestions, e.g.:

- model-bound: reduce params by ~30% (with projected speedup)
- sim-bound: increase env count (with projected speedup)
- memory-risk warnings when estimated usage approaches available RAM

## System-level runtime built-ins

Jules includes OS/system built-ins for low-level workflows:

- File/data: `sys::read_bytes`, `sys::write_bytes`, `sys::list_dir`
- File/path ops: `sys::copy`, `sys::rename`, `sys::metadata`, `sys::remove_path`
- Process: `sys::process_id`, `sys::sleep_ms`, `sys::exec`, `sys::exec_argv`, `sys::exec_argv_in`
- Environment: `sys::env_get`, `sys::env_set`, `sys::env_remove`
- Host info/directories: `sys::os`, `sys::arch`, `sys::temp_dir`, `sys::cwd`, `sys::set_cwd`, `sys::mkdir`, `sys::rmdir`

## Run a sample game script

```bash
cargo run --offline -- run small_game.jules
```

## Arcade game example

The repository also includes `game_arcade_showcase.jules`, a compact arcade
loop example (player/enemy movement, collisions, waves, scoring, HP).

```bash
cargo run --bin jules -- check game_arcade_showcase.jules
```

## Chess ML learning environment + benchmark

A high-throughput chess-like learning environment is available via:

```bash
cargo run --release --bin bench-chess-ml -- 50000 24
```

This runs the Jules training loop, reports steps/s, and compares against a Python baseline to verify runtime performance.


## Jules-native ML chess script

A toy ML chess training environment written directly in Jules is available at `ml_chess.jules`.

```bash
cargo run --bin jules -- run ml_chess.jules
```

To benchmark script execution vs Python baseline:

```bash
cargo run --release --bin bench-jules-ml-chess -- ml_chess.jules
```

## Jules-native ant simulation with online AI

An RL-style ant colony simulation with online linear-Q learning is available at `ant_ai_sim.jules`.

```bash
cargo run --offline --bin jules -- run ant_ai_sim.jules
```
