# Jules Optimization Backlog (Multi-Session Plan)

This plan breaks optimization into smaller sessions so we can iterate with measurable wins.

## Baseline (as of 2026-04-06)

- Benchmark command:
  - `cargo run --release --bin bench-interp-vs-rust -- 400000 8`
- Initial observed output (local):
  - Jules compile: `0.001152s total`
  - Jules runtime: `3.674230s total`
  - Rust runtime: `0.005436s total`
  - Runtime ratio: `675.90x`
- Latest observed output after current-session start work:
  - Jules compile: `0.000911s total`
  - Jules runtime: `2.510593s total`
  - Rust runtime: `0.003830s total`
  - Runtime ratio: `655.57x` (improved)

---

## Session 1 — High-Impact Interpreter Hot Paths

- [ ] **Profile full interpreter loop cost** with flamegraph/perf and identify top 5 symbols.
- [ ] **Specialize `for` range execution** to avoid generic iterator/value overhead in integer loops.
- [x] **Add arithmetic super-fast path for loop counters** (`i32/i64`) with minimal `Value` churn.
- [ ] **Reduce `Env` lookup overhead** for local variables in tight loops (slot-index cache).
- [x] **Add microbench: interpreter vs rust integer-loop baseline** and track throughput trend.

## Session 2 — Builtins Dispatch + Render Path

- [ ] Replace large string-match builtin dispatch with **interned IDs / dispatch table**.
- [ ] Remove avoidable allocations from `render::flush` conversion (reuse per-frame buffers).
- [ ] Add command-buffer typed view to avoid map/string-heavy shape in hot mode.
- [ ] Add benchmark: 10k/100k render commands per frame with stats output.

## Session 3 — Typechecker / Compile-Time Performance

- [ ] Cache path classification (`runtime path?`) per AST node during a pass.
- [ ] Minimize `String` allocations in diagnostics for non-error paths.
- [ ] Add incremental-like memoization for repeated type-check of unchanged modules.
- [ ] Add compile-time benchmark suite for:
  - tiny program
  - declaration-heavy program
  - expression-heavy program
  - module-heavy program

## Session 4 — Memory + Allocation Pressure

- [ ] Add allocation instrumentation counters (`alloc_count`, `bytes_allocated`) around eval.
- [ ] Reuse transient vectors/maps in eval helpers (scratch arena pattern).
- [ ] Audit `clone()` hotspots in interpreter and convert to borrow/copy where safe.
- [ ] Add regression gate for allocation growth on benchmark cases.

## Session 5 — Data-Oriented ECS / Sim Pipeline

- [ ] Move sim entities to SoA layout for cache-local updates.
- [ ] Reduce hash-map access in broadphase by bucket pooling and ID-index compaction.
- [ ] Add deterministic parallel step option for independent regions.
- [ ] Add benchmark matrix: entities × steps × collision density.

## Session 6 — Quality Gates + CI Performance Budgets

- [ ] Add `bench-interp-vs-rust` parser that emits machine-readable JSON.
- [ ] Add CI job to compare current vs baseline and fail on >N% regression.
- [ ] Store per-commit performance history (CSV/JSON in artifacts).
- [ ] Publish weekly trend report in `md/PERF_REPORT.md`.

---

## Immediate Next Session (Recommended)

1. Implement local-variable slot caching in interpreter loop execution.
2. Add dedicated `for range` fast executor.
3. Re-run `bench-interp-vs-rust` and compare against current `675.90x` baseline.

---

## Started in this session

- Added a **`for-in` string fast path** in the interpreter to avoid collecting a temporary iterable vector when iterating over string characters.
