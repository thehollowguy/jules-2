# PERFORMANCE OPTIMIZATION PLAN

## Overview
Goal: Improve Jules for game developers by focusing on low-latency embedding (FFI), ECS hot-path performance, and numeric kernels (SIMD/GPU). Follow the Jules ML optimization protocol for isolated, measurable changes.

## Phase 1 (Scaffold & Baseline)
- Add per-phase feature flags.
- Add a SIMD callsite helper and minimal implementation.
- Add an ECS microbenchmark (`bench-ecs`).

Benchmark snapshot (recorded on 2026-04-08, `bench-ecs`):
- entities: 5000
- steps: 50
- dt: 0.016
- command: `cargo run --release --bin bench-ecs -- 5000 50 0.016 both`
- baseline: 716.6 steps/s
- soa-linear: 6754.8 steps/s
- fused-linear: 9238.8 steps/s
- chunked-fused: 12413.6 steps/s
- superoptimizer: 40377.0 steps/s
- aot-hash: 43218.8 steps/s (kernel selected by CPU feature detection: `Avx2Fma`, x16-unrolled dispatch enabled)
- rust: 93035.0 steps/s
- aot-hash prep/iterate split: prepare=0.000000s, iterate=0.001157s
- hotspot weights (aot-hash): query=0.0%, fetch=0.0%, math=100.0%, write=0.0%
- recommendation: for throughput runs use `aot-hash` (feature-selected kernel) or `superoptimizer`; keep query matching/cache validation outside the simulation loop.

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
