# Jules Language Performance Guide

This guide covers how to build and run the Jules compiler/interpreter with maximum runtime performance.

## Quick Start

### Standard Optimized Build (Recommended)

```bash
# Build with all optimizations enabled
./scripts/build-optimized.sh

# Or manually
cargo build --release --features full
```

The binary will be at `target/release/jules`.

## Performance Build Options

### 1. Standard Release Build

The baseline optimized build with:
- `-O3` optimization level
- Fat LTO (Link-Time Optimization)
- Single codegen unit for better inlining
- Native CPU optimizations (AVX2, FMA, BMI2, LZCNT, POPCNT)
- Panic abort (no unwinding overhead)
- Symbol stripping for smaller binary

```bash
cargo build --release --features full
```

### 2. PGO (Profile-Guided Optimization)

PGO can provide **10-20% additional performance** by collecting runtime profiles and using them to guide optimization decisions.

```bash
# Run the automated PGO build script
./scripts/pgo-build.sh
```

This script:
1. Builds with instrumentation enabled
2. Runs benchmark workloads to collect profiles
3. Merges profile data
4. Rebuilds using the profile data

**Expected speedup:** 10-20% for interpreter workloads, 5-15% for compiler

### 3. BOLT (Binary Optimization and Layout Tool)

BOLT optimizes the binary layout for better instruction cache performance.

```bash
# Requires llvm-bolt (install via llvm-16-tools or build from source)
./scripts/bolt-optimize.sh
```

This produces `target/release/jules.bolt-optimized`.

**Expected speedup:** 5-10% for large programs

## Maximum Performance Build

For absolute maximum performance, combine all techniques:

```bash
# 1. Build with PGO
./scripts/pgo-build.sh

# 2. Optimize with BOLT
./scripts/bolt-optimize.sh

# 3. Use the final optimized binary
./target/release/jules.bolt-optimized run your_program.jules
```

**Total expected speedup:** 20-40% over standard release build

## Compiler Flags Explained

### Current Release Profile (`Cargo.toml`)

```toml
[profile.release]
opt-level = 3           # Maximum optimization level
lto = "fat"            # Cross-crate LTO
codegen-units = 1      # Single CGU for best inlining
panic = "abort"        # No unwinding overhead
strip = "symbols"      # Smaller binary
incremental = false    # Clean build for max perf
```

### Native CPU Features

The `[build]` section enables:
- `target-cpu=native`: All features of your CPU
- `target-feature=+avx2,+fma,+bmi2,+lzcnt,+popcnt`: Vector math and bit manipulation
- `force-frame-pointers=no`: Remove frame pointer overhead

## Interpreter Optimizations

The interpreter hot paths have been optimized with:
- `#[inline]` attributes on critical functions (`eval_expr`, `eval_binop`, `eval_numeric_binop`, etc.)
- Fast paths for I32 and F32 arithmetic (most common case)
- FxHashMap for 2x faster variable lookups
- Slot-based environment for O(1) variable access

## Benchmarking

### Run Built-in Benchmarks

```bash
# ECS benchmark
cargo run --release --bin bench-ecs

# Micro benchmarks
cargo run --release --bin micro-benchmark

# Chess ML benchmark
cargo run --release --bin bench-chess-ml

# Interpreter vs Rust comparison
cargo run --release --bin bench-interp-vs-rust
```

### Profile Your Program

```bash
# Using perf (Linux)
perf record --call-graph dwarf ./target/release/jules run your_program.jules
perf report

# Using cargo-flamegraph
cargo install flamegraph
cargo flamegraph --bin jules -- run your_program.jules
```

## JIT Compilation

For even faster execution, enable the Phase 3 JIT compiler:

```bash
cargo build --release --features "full,phase3-jit"
```

The JIT compiles hot loops to native x86-64 code at runtime, providing **5-50x speedup** for compute-heavy programs.

## SIMD Optimizations

Enable Phase 6 SIMD for vectorized math:

```bash
cargo build --release --features "full,phase6-simd"
```

This adds:
- AVX-512F (16 particles/iter)
- AVX2+FMA (32 particles/4x unrolled)
- AVX (16 particles/2x unrolled)
- SSE2 (8 particles/2x unrolled)

## GPU Acceleration

For ML workloads, the GPU backend can be used:

```bash
cargo build --release --features full
```

Tensor matmul operations automatically use optimized GEMM via `matrixmultiply::sgemm`.

## Performance Tips for Jules Programs

1. **Use typed variables**: `let x: i32 = 5` instead of `let x = 5`
2. **Prefer tensors over loops**: `a @ b` instead of manual matrix multiply
3. **Use `parallel for`** for independent iterations
4. **Enable JIT** for compute-heavy loops: `@jit fn compute() { ... }`
5. **Use SIMD hints**: `@simd fn process() { ... }`

## Build Profiles Summary

| Profile | Opt Level | LTO | CGU | Use Case |
|---------|-----------|-----|-----|----------|
| `dev` | 0 | off | 16 | Development |
| `release` | 3 | fat | 1 | Production |
| `max-perf` | 3 | fat | 1 | Maximum perf (same as release) |
| `bench` | 3 | fat | 1 | Benchmarks |

## Troubleshooting

### Build fails with AVX errors

Your CPU may not support AVX2. Edit `Cargo.toml` and remove `+avx2,+fma,+bmi2` from `target-feature`.

### PGO build fails

Ensure you have LLVM tools installed:
```bash
# Ubuntu
sudo apt install llvm-16 llvm-16-dev llvm-16-tools

# macOS
brew install llvm
```

### BOLT not available

BOLT requires LLVM. Install or build from source:
```bash
git clone https://github.com/llvm/llvm-project
cd llvm-project
cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS="bolt"
cmake --build build
```

## Performance Comparison

Typical speedups vs. unoptimized debug build:

| Optimization | Speedup |
|--------------|---------|
| Release build | 3-5x |
| + Native CPU | 1.1-1.3x |
| + PGO | 1.1-1.2x |
| + BOLT | 1.05-1.1x |
| + JIT (for hot loops) | 5-50x |
| **Total** | **5-10x** |

## Additional Resources

- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [PGO in Rust](https://doc.rust-lang.org/rustc/profile-guided-optimization.html)
- [LLVM BOLT](https://github.com/llvm/llvm-project/blob/main/bolt/README.md)
