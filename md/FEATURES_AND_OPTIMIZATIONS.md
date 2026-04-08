# Jules Features and Optimizations

This document consolidates the major implemented capabilities of Jules and the optimization techniques currently described in project documentation.

## Features

### Core language/runtime
- Game-development + ML-oriented language/runtime.
- ECS + system runtime.
- Lexical borrow-checking pass for alias safety (`&` / `&mut`).
- Incremental compilation to recompile only changed code paths.
- Incremental on-disk cache for `jules check` (`.jules_cache/check`) with immediate returns for unchanged, clean sources.

### Host integrations and FFI
- C ABI for embedding in external engines.
- C++ binding via `bindings/cpp/jules.hpp`.
- C# binding via `bindings/csharp/Jules.cs`.
- Python binding via `bindings/python/jules.py` and optional PyO3 support (`--features python`).
- C ABI entry points for file execution, code checking, tensor lifecycle/shape/data APIs, and physics sample APIs.

### ML and tensor capabilities
- ML tensor engine with CPU + GPU hooks.
- INT8 linear inference path:
  - Offline weight quantization (`Tensor::quantize_linear_int8`).
  - INT8 inference (`Tensor::linear_int8`).
  - Effective bytes-per-parameter estimation (`Int8LinearWeights::effective_bytes_per_param`).
- `Tensor::softmax_last_dim` for batched class probabilities.
- Transformer/LMM-style activations and gradients:
  - `Tensor::gelu`, `Tensor::silu`.
  - `Tensor::gelu_backward`, `Tensor::silu_backward`.
- Cross-entropy helpers for indexed-class training:
  - `LossFunctions::cross_entropy_from_logits_last_dim`.
  - `LossFunctions::cross_entropy_from_logits_last_dim_gradient`.
- Strict ML memory ceiling model (`M_usage <= M_min + ΔM`) with APIs for configure/acquire/release/snapshot and separate Core/Extra pools.

### Game/simulation and rendering
- Deterministic entity iteration order in `sim` worlds.
- Uniform-grid broadphase collision filtering in `sim::step`.
- Sorted/stable nearest and radius entity query behavior.
- Game-loop movement helpers:
  - `math::approach`
  - `math::move_towards2`
  - `math::angle_to`
  - `math::rand_unit2`
- AoT-friendly render command API:
  - `render::begin_frame`, `render::clear`, `render::rect`, `render::sprite`, `render::flush`, `render::stats`.

### Standard library and tooling
- Structured stdlib modules (`core`, `math`, `tensor`, `nn`, `train`, `data`, `io`, `sys`, `error`, `diag`, `collections`, `compute`, `quant`, `model`, `debug`, `sim`, `window`) and runtime discovery via `std::modules()`.
- JAX backend support for optimized ML training and INT8 artifact export.
- Backend-selectable training via `--ml-backend jules|jax`.
- Automated syntax recovery with `jules fix` (safe parser-diagnostic-based fixes).
- Training ETA estimator (`jules estimate`) with sim/model bottleneck reporting and tuning suggestions.
- System-level built-ins for file/path/process/environment/host operations.
- ECS benchmark modes: `baseline`, `soa-linear`, `fused-linear`, `chunked`, `aot-hash`.

## Optimizations

### Build/codegen
- Release profile tuning:
  - `opt-level = 3`
  - `lto = "fat"`
  - `codegen-units = 1`
  - `panic = "abort"`
  - `strip = "symbols"`
  - `incremental = false`
- Bench profile inherits release settings.
- Host-specific codegen flag in local runs: `-C target-cpu=native`.

### Phase 3 x86-64 JIT
- Slot-frequency profiling and hot-slot pinning in registers (`r8`, `r9`).
- Precomputed slot offset table for emission.
- Superinstruction/peephole fusion patterns:
  - `BinOp + Store` fusion.
  - `BinOp + JumpFalse/JumpTrue` fusion.
  - `LoadI32/LoadI64 + BinOp` immediate fusion.
  - Constant-load + conditional-jump translation-time folding.
- Immediate-form instruction selection (`add/sub/imul/cmp` with immediates).
- Strength reductions:
  - `+1/-1` -> `inc/dec`
  - `*0` -> zeroing xor
  - `*1` -> no-op
  - `*2^k` -> shift left
  - LEA forms for `*3`, `*5`, `*9`

### Interpreter/runtime
- Fast paths for common integer/bool operations.
- Specialized control-flow bytecode handling.
- Thread-local register scratch reuse (`EXEC_REGS`) to reduce realloc churn.
- Compiler-validated unchecked operand access in VM dispatch to reduce repeated bounds checks.
- VM call-site argument loading optimizations and reduced redundant builtin/method-name cloning.

### ML/tensor performance
- Cache-aware blocked CPU matmul kernels.
- Threaded row chunking for applicable CPU matmul paths.
- GPU-backend CPU fallback using blocked + transpose-friendly access.
- INT8 low-byte linear inference to reduce memory/bandwidth pressure.
- Numerically stable activation/loss helper set for training/inference.

### JAX/XLA backend
- `jax.jit`-compiled training step.
- Optax AdamW integration.
- Export path for FP32 checkpoints + Jules-compatible INT8 artifacts.

### Benchmarking/perf tooling
- `bench-interp-vs-rust` full mode and `aot-time` mode.
- `bench-chess-ml` unified baseline runs (Jules/Python/JAX/Rust).
- `--skip-strength-eval` flag to isolate timing runs.

### Known next opportunities (documented backlog)
- Div/rem guard lowering for safe zero/overflow behavior.
- Basic-block local value numbering.
- Wider constant propagation windows.
- Broader branch folding/jump-threading.
