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

This targets roughly **~1 byte/parameter** plus small scale overhead, for
running (inference) workloads.

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
