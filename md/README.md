# Jules

Jules is a game-dev + ML-oriented language/runtime with:
- ECS + system runtime
- ML tensor engine (CPU + GPU hooks)
- C ABI for embedding in external engines
- lexical borrow-checking pass for reference alias safety (`&` / `&mut`)

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

## Automated syntax fix flow

Use:

```bash
jules fix path/to/file.jules
```

`jules fix` applies safe automatic syntax fixes from parser diagnostics
(e.g. missing `;`, missing `)`, `]`, `}`, missing `,`, assignment/operator
replacement like `==`→`=`, return arrow insertion, block opener insertion,
and common `fun`/`func` → `fn` keyword typo recovery).

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
