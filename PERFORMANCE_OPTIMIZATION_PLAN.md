# PERFORMANCE OPTIMIZATION PLAN

## Overview
Goal: Improve Jules for game developers by focusing on low-latency embedding (FFI), ECS hot-path performance, and numeric kernels (SIMD/GPU). Follow the Jules ML optimization protocol for isolated, measurable changes.

## Phase 1 (Scaffold & Baseline)
- Add per-phase feature flags.
- Add a SIMD callsite helper and minimal implementation.
- Add an ECS microbenchmark (`bench-ecs`).

Baseline metrics to record (fill after running):
- entities: <n>
- steps: <steps>
- baseline time (s): <baseline_seconds>
- simd time (s) [if run]: <simd_seconds>
- RSS (MB): <rss>
- timestamp: <ts>

## Phase 2 (Micro-optimizations)
- Replace `phase6_simd::update_positions` with an architecture-optimized implementation (portable_simd or LLVM intrinsics).
- Reduce allocation churn in ECS query path (avoid Vec allocations per tick).
- Add property tests for numerical stability.

## Verification
- `cargo test --no-default-features`
- `cargo test --features phase6-simd`
- `cargo clippy --all-targets -- -D warnings`
- `cargo fmt --check`
- `cargo build --release`
- `cargo run --release --bin bench-ecs -- <entities> <steps> <dt>`

## Risks / Edge cases
- Ensure no aliasing or lifetime bugs when switching to in-place SIMD writes.
- Verify bit-exactness or document epsilon bounds for floating-point differences.
- Keep ABI stable for existing FFI functions.

## Notes
Update this file with measured baseline and optimized numbers for every experiment. Keep changes small and reversible.
