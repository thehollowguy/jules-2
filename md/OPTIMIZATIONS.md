# Jules Optimization Catalog

This document is the single inventory of optimization techniques currently implemented in Jules.
It covers runtime, compiler/JIT, ML kernels, memory behavior, and benchmark tooling.

## 1) Build-time and codegen optimizations

- Release profile tuning (`Cargo.toml`):
  - `opt-level = 3`
  - `lto = "fat"`
  - `codegen-units = 1`
  - `panic = "abort"`
  - `strip = "symbols"`
  - `incremental = false`
- Bench profile inherits release settings for apples-to-apples perf numbers.
- Local host-specific codegen (`.cargo/config.toml`):
  - `-C target-cpu=native` for architecture-specific instruction selection in local runs.

## 2) Phase 3 machine-code JIT optimizations (x86-64)

### 2.1 Register and layout optimizations

- Slot-frequency profiling and hot-slot pinning:
  - Two hottest slots are pinned in `r8` / `r9` to reduce memory traffic.
- Precomputed slot offset table to avoid repeated offset arithmetic during emission.

### 2.2 Superinstruction / peephole fusion

- `BinOp(tmp, op, l, r)` + `Store(slot, tmp)` fusion:
  - Emits direct arithmetic/compare into destination slot.
- `BinOp(tmp, op, l, r)` + `JumpFalse(tmp, off)` fusion:
  - Eliminates temporary slot roundtrip before branch.
- `BinOp(tmp, op, l, r)` + `JumpTrue(tmp, off)` fusion:
  - Eliminates temporary slot roundtrip before branch.
- `LoadI32/LoadI64(tmp, c)` + `BinOp(dst, op, x, tmp)` fusion:
  - Uses immediate arithmetic/compare opcodes directly.
- `LoadI*/LoadBool/LoadUnit(tmp, c)` + `JumpFalse(tmp, off)`:
  - Branch resolved at translation time.
  - Emits unconditional jump only when needed; otherwise branch is removed.
- `LoadI*/LoadBool/LoadUnit(tmp, c)` + `JumpTrue(tmp, off)`:
  - Branch resolved at translation time.
  - Emits unconditional jump only when needed; otherwise branch is removed.

### 2.3 Immediate-form instruction selection

- Emits compact immediate forms for:
  - `add rax, imm32`
  - `sub rax, imm32`
  - `imul rax, imm32`
  - `cmp rax, imm32`
- Strength reduction patterns:
  - `+1/-1` -> `inc/dec`
  - `*0` -> `xor rax, rax`
  - `*1` -> no-op
  - `*2^k` -> `shl rax, k`
  - tiny superoptimizer-guided LEA forms for selected small constants:
    - `*3` -> `lea rax, [rax + rax*2]`
    - `*5` -> `lea rax, [rax + rax*4]`
    - `*9` -> `lea rax, [rax + rax*8]`

## 3) Interpreter/runtime-level optimization highlights

- Fast-path support for common integer/bool operations in hot loops.
- Specialized handling for common control-flow bytecodes.
- Thread-local register scratch reuse during JIT execution (`EXEC_REGS`) to reduce realloc churn.

## 4) ML and tensor engine optimizations

- Cache-aware blocked CPU matmul kernels (plus threaded row chunking where applicable).
- CPU fallback in GPU backend mirrors blocked + transpose-friendly access patterns.
- INT8 linear inference path:
  - offline quantization + per-output-channel scales
  - low-byte inference kernels to reduce memory/bandwidth pressure.
- Additional numerically stable loss/activation helpers for model training/inference pipelines.

## 5) JAX/XLA backend optimizations for training

- `jax.jit` training step compilation for low Python overhead.
- Optax AdamW integration with batch execution.
- Export path for fp32 checkpoint + Jules-compatible INT8 artifacts.

## 6) Benchmarking and measurement optimizations

- `bench-interp-vs-rust`:
  - Full mode (compile + runtime) and `aot-time` mode (compile-only baseline).
- `bench-chess-ml`:
  - Jules + Python + JAX + Rust baselines in one run.
  - `--skip-strength-eval` to isolate timing runs from long strength loops.

## 7) Known next machine-code opportunities

- Div/rem guard lowering (safe/defined behavior around zero and overflow traps).
- Basic-block local value numbering to remove redundant slot loads.
- Broader constant propagation across short instruction windows.
- Additional branch folding and jump-threading across larger basic-block windows.
