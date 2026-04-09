# Deep Performance Analysis
### JIT · AoT Pipeline · Superoptimizer · SIMD · ECS
> Files: `phase3_jit.rs`, `phase6_simd.rs`, `interp.rs`, `main.rs`, `borrowck.rs`

---

## Part 1 — `likely` / `unlikely` Are No-Ops

This is the single most pervasive correctness issue in `interp.rs` and it silently invalidates every branch-prediction annotation in the file.

```rust
#[inline(always)]
fn likely(b: bool) -> bool { b }

#[inline(always)]
fn unlikely(b: bool) -> bool { b }
```

Both functions are identity transforms. They compile to nothing. Every call site that uses them — `likely(self.top < STACK_CAPACITY)`, `unlikely(self.top == 0)`, `unlikely(i + 7 < chunk.len())`, `likely(matches!(..., Value::Vec3(_)))` — gets zero prediction benefit. The file header claims "core::intrinsics::likely()/unlikely() wrappers guide CPU branch predictor" but the wrappers don't call the intrinsics. They need to be:

```rust
#[inline(always)]
fn likely(b: bool) -> bool {
    unsafe { std::intrinsics::likely(b) }
}
```

Until this is fixed, the CPU's own dynamic predictor is the only thing working, and it's being undermined in places like `unlikely(i + 7 < chunk.len())` on the hot SIMD loop — which tells the static predictor the loop body is cold when it's the exact opposite (see §8.3 below for more on that specific case).

---

## Part 2 — JIT (`phase3_jit.rs`)

### 2.1 The JIT Is Wired Up But the Entry Guard Variable `jit_hot` Is Never Defined

In `call_fn`:

```rust
if self.jit_enabled && jit_hot {
```

`jit_hot` is not a field, not a local variable defined above this line, and not imported. This code does not compile unless there is an implicit `let jit_hot = true;` somewhere in scope that was elided in the snippet — which would make the guard permanently true regardless of call counts or profiling. The `fn_call_counts` map is cleared in `set_jit_enabled` and `load_program` but is never written to anywhere visible in `call_fn`. If `jit_hot` is hardcoded `true`, then **every single function call** unconditionally compiles the function on first invocation and runs STOKE on it synchronously before returning. With `stoke_budget: 200_000` for functions ≥ 40 instructions, that is 200k MCMC proposals happening inline during the first call. For the user, this looks like the program hanging.

### 2.2 `mprotect` Syscall on Every JIT Compile

In `ExecMem::new`, the arena fast path issues `mprotect(RX)` over the page range covering the new allocation:

```rust
let base = (ptr as usize) & !(page - 1);
let plen = (((ptr as usize) + len + page - 1) & !(page - 1)) - base;
if unsafe { mprotect(base as *mut _, plen, PROT_READ | PROT_EXEC) } != 0 {
    return None;
}
```

`mprotect` is a syscall that acquires `mmap_lock` (a kernel-wide reader-writer semaphore) and walks the VMA tree. On Linux 5.x this costs ~1–3 µs per call under no contention. Calling it per-function means compiling 1000 functions costs 1–3 ms in kernel time alone before any code runs. The arena should be flipped to RX in a single `mprotect` call at the end of a compilation batch, or maintained as a dual-mapping (RW alias + RX alias) to avoid the syscall entirely.

### 2.3 `is_dead_def` Is O(n) Per Call, Called O(n) Times → O(n²) Total

```rust
fn is_dead_def(instrs: &[Instr], pc: usize, slot: u16) -> bool {
    let mut i = pc + 1;
    while i < instrs.len() {
        let n = &instrs[i];
        if is_cf_barrier(n) { return false; }
        if instr_reads_slot(n, slot)  { return false; }
        if instr_writes_slot(n, slot) { return true; }
        i += 1;
    }
    true
}
```

It's called at every fusion site (D.1, D.2, E.1–E.8) and again in the single-instruction fallback for every `BinOp`, `Move`, `Store`, `Load`, and `LoadX` instruction. In the worst case this is called 8 times per instruction and scans to the end of the function each time. A 500-instruction function hits ~2M comparisons per compile. The fix is a single pre-pass that fills a `Vec<usize>` (`last_use[slot] = pc`) using the live-interval data that `compute_live_intervals` already computes — then `is_dead_def(pc, slot)` becomes `last_use[slot] < pc`, which is O(1).

### 2.4 `Vec::insert` in the Linear-Scan Active List Is O(n) Per Interval

```rust
let pos = active.partition_point(|(e,_,_)| *e <= iv.last);
active.insert(pos, (iv.last, iv.slot, reg));
```

`Vec::insert` shifts all elements at and after `pos` one position to the right. The active list holds all currently live intervals, and in a register-pressure-heavy function this can be 8–10 entries at once. But even for small active lists, this is called once per interval and `partition_point` + `insert` together scan O(active.len()) on every allocation. For a function with 200 intervals, this is 200 × O(10) = 2000 operations, which is fine — but the `active.remove(evict_idx)` in the spill path is also O(n), and if spilling is frequent (because you have 10 physical registers and 50 live slots), the inner loop degrades. A `BinaryHeap<Reverse<(usize, u16, u8)>>` keyed on endpoint eliminates all O(n) insertions.

### 2.5 Three Independent Analysis Passes Over `instrs`

These three functions all iterate `instrs` sequentially:

1. `compute_live_intervals(instrs, slot_count)` — forward + backward scan
2. `compute_slot_hotness(instrs, slot_count)` — which itself calls:
3. `compute_loop_body_weight(instrs)` — separate backward scan for loop detection

All three are called in sequence in `translate`. That is four passes over the instruction array before a single byte of machine code is emitted. They share no data. A single merged pass could produce intervals, hotness, and loop weights simultaneously. For small functions this doesn't matter, but the STOKE superoptimizer is supposed to operate on small functions anyway — for larger ones this preamble adds unnecessary latency.

### 2.6 Branch Relaxation Leaves NOPs In-Line Instead of Compacting

When a rel32 branch is relaxed to rel8, the 3 freed bytes become NOPs:

```rust
for b in &mut buf[op_start + 2..fx.disp_pos + 4] { *b = 0x90; }
```

Three single-byte `0x90` NOPs in sequence is decoded as three separate µops on Intel (pre-Skylake decoders handle multi-byte NOPs only if they are the `0F 1F` form). On Skylake and later, `0x90` NOPs still consume a front-end slot. In branch-heavy code (loops with multiple exits), this leaves a sludge of NOP µops in the decoded instruction cache (DSB). The correct fix is a second compaction pass that rewrites offsets — or, alternatively, using `0x0F 0x1F 0x00` (3-byte NOP) which is handled in a single decoder slot.

### 2.7 Constant Propagation Table Cleared Too Aggressively at Branches

Every branch — including unconditional forward jumps to known targets — clears the entire `ConstTable`:

```rust
Instr::Jump(off) => { ... const_at.clear(); }
Instr::JumpFalse(..) => { ... const_at.clear(); }
Instr::JumpTrue(..) => { ... const_at.clear(); }
```

For the common if-else pattern, the else branch is an unconditional forward jump to a known offset. The constants that were valid before the branch are still valid at the else target if no writes occurred. Clearing them forces re-emitting `MOV rax, imm` for any constant that gets used in the else arm. This is measurable in any function with constant-folded branches.

### 2.8 Prologue Pre-Loads Every Register Slot Unconditionally

```rust
for iv in &intervals {
    if let RegLoc::Reg(r) = ra.location(iv.slot) {
        let bit = 1u32 << r;
        if loaded & bit == 0 { loaded |= bit; em.load_reg_mem(r, (iv.slot as i32) * 8); }
    }
}
```

This emits a `MOV reg, [rdi + offset]` for every register-allocated slot, keyed by unique physical register. If 6 slots are register-allocated, 6 loads appear in the prologue unconditionally — even if the function's first instruction overwrites those registers with a `LoadI64`. The correct approach is lazy load-on-first-use, which `is_dead_def`-style analysis could easily provide.

---

## Part 3 — Superoptimizer (`interp.rs`)

### 3.1 STOKE Runs Synchronously on the First Call to Every Function

From `maybe_superoptimize`, called inside `precompile_loaded_functions` and inside the `compiled_fns.entry(...).or_insert_with(...)` in `call_fn`:

```rust
fn maybe_superoptimize(compiled: &mut CompiledFn, ...) {
    let cfg = SuperoptConfig {
        run_stoke: compiled.instrs.len() >= 4,
        stoke_budget: if compiled.instrs.len() < 8 { 50_000 }
                      else if compiled.instrs.len() < 40 { 100_000 }
                      else { 200_000 },
        ...
    };
    superoptimize_fn(&mut candidate, &cfg);
}
```

For any function with ≥ 40 instructions, STOKE runs 200,000 MCMC proposals split across 4 Rayon parallel chains (50,000 each). Each proposal clones the instruction vector, mutates it, runs `eval_concrete` on all 16 test vectors, and computes `correctness_distance`. `eval_concrete` is itself an interpreter pass. 200,000 × 16 interpreter runs happen before the function returns for the first time. This is a superoptimizer, not a JIT — STOKE compile times are measured in seconds, not microseconds.

For a program that calls 50 distinct functions, `load_program` triggers 50 simultaneous STOKE runs on the Rayon thread pool. The process will appear to freeze for seconds on startup.

There is no tiered compilation here: STOKE should only run on functions that are proven hot after many thousands of real calls, in a background thread, with the bytecode VM running in the foreground until the optimized version is ready.

### 3.2 `load_program` Hashes the Entire Program via `format!("{:?}", program)`

```rust
let mut hasher = std::collections::hash_map::DefaultHasher::new();
format!("{:?}", program).hash(&mut hasher);
```

`format!("{:?}", program)` allocates a `String` containing the full debug representation of the AST — potentially megabytes for a large program — on every `load_program` call. This is used only to detect whether the program changed since last load. A structural hash derived from the source text's byte length + a cheap rolling hash of the source bytes (already available in `cmd_run` for the incremental check cache) would avoid the allocation entirely.

### 3.3 Three Independent Join-Plan Copies for the Same Pair

`vec3_plan_cache` (used by `integrate_vec3_superoptimizer` and `integrate_vec3_chunked_precomputed`), `fused_plan_cache` (used by `integrate_vec3_and_health_chunked`), and `adaptive_vec3_cache` (used by `integrate_vec3_adaptive`) all independently store `Vec<(usize, usize)>` join-plan pair lists for the same `(pos_comp, vel_comp)` join. When a component version changes:

- `vec3_plan_cache` is invalidated and rebuilds via `O(pos.dense_ids.len())` scan.
- `adaptive_vec3_cache` is invalidated separately and rebuilds via `build_vec3_join_plan` (same O(n) scan).
- `fused_plan_cache` for the 4-component case also rebuilds.

So a single `despawn()` that bumps a version counter causes three independent O(n) rebuild scans for the same underlying data. These should share a single canonical join plan, version-gated once.

### 3.4 SIMD Gather/Scatter in `simd_update_vec3_x8` Defeats the Entire Point

The 8-wide AVX2+FMA path in `simd_update_vec3_x8` does:

1. **Gather**: extract `f32` values from 8 `Value::Vec3` enum entries into staging `Buf([f32; 8])` arrays — 8 non-contiguous memory reads per component (24 total reads).
2. **Compute**: 6 `_mm256_fmadd_ps` instructions.
3. **Scatter**: write results back into 8 `Value::Vec3` enum entries — 8 non-contiguous writes per component (24 total writes).

The 6 FMA instructions take ~3 cycles on port 0 (Skylake). The 48 non-contiguous memory accesses, each potentially on a separate cache line (since `Value` is a large enum and `dense_vals` is a `Vec<Value>` not a `Vec<[f32;3]>`), take 4–12 cycles each on a cache miss. The SIMD portion is entirely hidden behind the gather/scatter latency. This path is not faster than the SoA loop in `integrate_vec3_linear_fused` which operates on contiguous `f32` arrays that auto-vectorise. It may actually be slower due to the staging buffer allocation on the stack (three `Buf([0f32; 8])` for pos + three for vel = 192 bytes of stack writes before a single FMA fires).

### 3.5 Prefetch in `simd_update_vec3_x8` Prefetches the Array Base, Not the Current Offset

```rust
_mm_prefetch(pos_ptr as *const i8, _MM_HINT_T0);
_mm_prefetch(vel_ptr as *const i8, _MM_HINT_T0);
```

`pos_ptr` and `vel_ptr` are the base pointers of `dense_vals`. On every invocation of `simd_update_vec3_x8`, the same base address is prefetched. After the first call, that cache line is already hot. Every subsequent call's prefetch is a no-op. The correct form is `pos_ptr.add(slots[7].0 + LOOKAHEAD)` — prefetching the next batch's data while computing the current batch.

In `integrate_vec3_and_health_chunked` the same mistake appears, compounded:

```rust
_mm_prefetch(pos_ptr as *const i8, _MM_HINT_T0);
_mm_prefetch(vel_ptr as *const i8, _MM_HINT_T0);
_mm_prefetch(hp_ptr  as *const i8, _MM_HINT_T0);
_mm_prefetch(dmg_ptr as *const i8, _MM_HINT_T0);
```

Four base-pointer prefetches, all cold-cache-only useful on the very first chunk iteration.

### 3.6 `unlikely` on the Hot Loop Guard in `integrate_vec3_and_health_chunked`

```rust
while unlikely(i + 7 < chunk.len()) {
```

`unlikely` signals the compiler (when it actually works — it doesn't here, see §1) that this branch is rarely taken. The loop body processes 8 entities per iteration; for a chunk of 256 entities, the branch is taken 31 times and falls through once. The hot path is `i + 7 < chunk.len()` being true. This is `likely`, not `unlikely`. Swapping these is a one-character fix that, once `unlikely` is properly wired to the intrinsic, will cause the compiler to lay out the loop body as the fall-through path and the exit as the taken branch — which is the layout the branch predictor prefers for counted loops.

### 3.7 `integrate_vec3_linear_fused` SoA Write-Back Is a Full Second Pass

After the SoA integration loop finishes (which is the fast, autovectorisable path), the results are written back into `Value::Vec3`:

```rust
for i in 0..n {
    if let Value::Vec3(ref mut p3) = pos_set.dense_vals[i] {
        unsafe {
            p3[0] = *pos_set.xs.get_unchecked(i);
            p3[1] = *pos_set.ys.get_unchecked(i);
            p3[2] = *pos_set.zs.get_unchecked(i);
        }
    }
}
```

This is a complete second O(n) pass over `dense_vals` that writes back values into a tagged enum layout. The SoA arrays and the AoS `dense_vals` are now both in cache after the first pass, so this write-back doesn't cause cold misses — but it does double the store bandwidth consumed, and it prevents the autovectoriser from treating the two loops as a single fused compute-and-store. If downstream code only reads Vec3s through `get_component` (which pattern-matches `Value::Vec3`), the write-back could be deferred lazily: mark a `soa_dirty` flag, and sync `dense_vals` only when a non-SoA consumer actually touches the component. In steady-state simulation ticks this write-back never fires.

### 3.8 `maybe_superoptimize` Is Called Twice for the Same Function on First Call

In `precompile_loaded_functions`:

```rust
let compiled = compile_fn(&closure.decl);
maybe_superoptimize(&mut compiled, ...);
self.compiled_fns.insert(name, Arc::new(compiled));
```

And again in `call_fn` via the `or_insert_with` closure:

```rust
self.compiled_fns
    .entry(name.to_owned())
    .or_insert_with(|| {
        let mut compiled = compile_fn(&closure.decl);
        maybe_superoptimize(&mut compiled, ...);
        Arc::new(compiled)
    })
```

`precompile_loaded_functions` is called at the end of `load_program`. If the function was already inserted there, `or_insert_with` won't call the closure again — so in the normal path this is fine. But `precompile_loaded_functions` early-returns if `jit_hot_threshold > 1`, while `call_fn` always compiles on first entry regardless. A function defined after the threshold cutoff will be compiled in `call_fn` but was also attempted by `precompile_loaded_functions` (which skipped it), meaning the bytecode is compiled and STOKE'd **synchronously on the first call at runtime** regardless of whether AoT precompile was enabled.

### 3.9 `total_latency` Used as STOKE's Performance Cost Is a Sum, Not a Critical Path

```rust
fn total_latency(instrs: &[Instr]) -> u32 {
    instrs.iter().map(instr_latency).sum()
}
```

STOKE minimises `total_latency` as a proxy for execution speed. But `total_latency` is the sum of all instruction latencies, which equals execution time only on a perfectly serial, single-issue, in-order pipeline. On a modern OoO superscalar CPU (which is the entire point of the JIT), independent instructions execute in parallel. The correct metric is the critical-path latency through the data-dependency DAG. STOKE will therefore accept instruction sequence A over sequence B even if B has lower critical-path latency, as long as B has a higher total sum — meaning it can make code slower on real hardware while reporting it as optimised.

---

## Part 4 — AoT Pipeline (`main.rs`)

### 4.1 All Five Compiler Passes Re-Run on Every `cmd_run` Invocation

`cmd_run` calls `pipeline.run(&mut unit)` which sequentially executes lex → parse → typeck → sema → borrowck from scratch. There is no caching at this layer for execution (only `cmd_check` has an incremental cache, and it only caches the diagnostic result, not the AST or bytecode). Running the same unchanged file twice runs all five passes twice. For a 10,000-line program with many functions, the borrow checker and type checker alone can take tens of milliseconds.

The `load_program` hash (`format!("{:?}", program)`) could in principle skip re-loading, but it's checked after all five passes have already run. The check gates `load_item` calls, not the compiler passes.

### 4.2 `detect_silent_issues` Runs Regex-Style Line Scans on Every Execution

```rust
let result = pipeline.run(&mut unit);
unit.diags.extend(detect_silent_issues(&source));
```

`detect_silent_issues` scans every line of source for float equality comparisons, magic numbers, mutable globals, and similar heuristic issues. It runs unconditionally on every `cmd_run`, even for unchanged sources. Its output is purely diagnostic and does not affect execution. It should be gated behind a lint flag or folded into the incremental check cache.

### 4.3 `Interpreter::new()` Eagerly Allocates Four `Arc<Mutex<_>>` Subsystems

```rust
gpu: Some(Box::new(JulesGpuAdapter::new())),
physics_world: Some(Arc::new(Mutex::new(PhysicsWorld::new()))),
render_state: Some(Arc::new(Mutex::new(RenderState::new()))),
input_state: Some(Arc::new(Mutex::new(InputState::new()))),
computation_graph: Some(Arc::new(Mutex::new(ComputationGraph::new()))),
```

Every interpreter instantiation — including one-shot `jules run`, REPL evaluations, and `jules check` — pays the allocation and initialisation cost of a GPU adapter, physics world, render state, input state, and computation graph. For a compiler-style pipeline that never uses any of these, this is pure overhead. These should be `None` by default, initialised lazily on first access.

### 4.4 Borrow Checker Allocates Fresh Collections Per Function

In `borrowck.rs`, `jules_borrowck` allocates a `BTreeMap` and several `HashSet`s per function declaration it analyses. For a program with 200 functions, this is 200 independent allocator round-trips. A single pre-allocated scratch struct passed by `&mut` through the analysis — or even a simple reset-and-reuse pattern — would collapse this to one allocation per borrow-check run.

---

## Part 5 — SIMD (`phase6_simd.rs`)

### 5.1 Runtime Feature Detection on Every `update_positions` Call

```rust
pub fn update_positions(positions: &mut [[f32; 3]], velocities: &[[f32; 3]], dt: f32) {
    ...
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx2") { ... }
    if is_x86_feature_detected!("fma")     && is_x86_feature_detected!("avx2") { ... }
    if is_x86_feature_detected!("avx") { ... }
    if is_x86_feature_detected!("sse2") { ... }
```

`is_x86_feature_detected!` on stable Rust caches the result in a static after the first call, so subsequent calls are a single atomic load. The 4 conditional branches are the real cost — they add 4 taken/not-taken outcomes to the branch predictor's budget on every call. In a simulation tick that calls `update_positions` once per frame this is negligible. But the ECS superoptimizer kernel (`run_vec3_superoptimizer_kernel`) also does feature detection inline per batch, and it's called potentially thousands of times per tick. Resolving dispatch to a function pointer once at startup (or at `EcsWorld` construction) eliminates all of this.

### 5.2 `load_aos4` Not Marked `#[inline(always)]`

`load_aos8` calls `load_aos4` twice. `load_aos4` is marked `#[target_feature(enable = "sse2")]` but not `#[inline(always)]`. The inliner will inline it in release mode under most circumstances, but `#[target_feature]` functions have different inlining rules — Rust requires that the caller also have the matching target feature enabled for a guaranteed inline. Since `load_aos8` is `#[target_feature(enable = "avx2")]` and AVX2 implies SSE2, the inline should happen — but without `#[inline(always)]` it is not guaranteed. If the inliner decides not to inline, you get two call/return pairs inside the hot AVX2 dispatch path, destroying the claim that load_aos8 "decomposes entirely into in-lane 128-bit operations" and adds the function call overhead on top.

### 5.3 `update_avx2_fma` Alignment Prefix Skips Prefetch for Scalar Head

The scalar prefix loop (to align to a 32-byte boundary) calls `scalar_tail`:

```rust
let align_skip = (p_pre.len() / 3).min(n);
scalar_tail(positions, velocities, dt, 0, align_skip);
let mut i = align_skip;
```

The scalar tail processes up to 7 particles without any prefetch. For most particle counts that are not a multiple of 8, the scalar tail at the *end* of the function handles the remainder at `scalar_tail(positions, velocities, dt, i, n)`. The prefetch in the main loop issues 48 elements ahead, but the scalar tail at the end has no prefetch at all. For small `n` (e.g., 9–15 particles), the entire workload is scalar and gets no prefetch benefit whatsoever despite the function's comments implying full hardware prefetch coverage.

---

## Summary by Severity

### Critical (causes seconds-level latency or completely broken behaviour)

| # | Location | Issue |
|---|----------|-------|
| C1 | `interp.rs` | `likely`/`unlikely` are identity functions — all branch hints are no-ops |
| C2 | `interp.rs` | STOKE runs synchronously (up to 200k proposals) on first call to every function |
| C3 | `interp.rs` | `jit_hot` is never defined as a local variable — call_fn may not compile as shown |
| C4 | `interp.rs` | `total_latency` is a sum not a critical-path — STOKE optimises the wrong metric |

### High (measurable regression in real workloads)

| # | Location | Issue |
|---|----------|-------|
| H1 | `phase3_jit.rs` | `mprotect` syscall per function compile; blocks `mmap_lock` |
| H2 | `phase3_jit.rs` | `is_dead_def` O(n) per instruction → O(n²) total compile time |
| H3 | `phase3_jit.rs` | Prologue pre-loads all register slots unconditionally including dead defs |
| H4 | `interp.rs` | `simd_update_vec3_x8` gather/scatter through `Value` enum defeats SIMD |
| H5 | `interp.rs` | Three independent join-plan lists; each rebuilt separately on version change |
| H6 | `interp.rs` | `load_program` allocates full `format!("{:?}", program)` string for hashing |
| H7 | `main.rs` | All 5 compiler passes re-run from scratch on every `cmd_run` |
| H8 | `main.rs` | `Interpreter::new()` eagerly initialises GPU, physics, render, input, autodiff subsystems |

### Medium (visible in profiling, fixable without redesign)

| # | Location | Issue |
|---|----------|-------|
| M1 | `phase3_jit.rs` | `Vec::insert` in active list → O(n) per interval in register allocator |
| M2 | `phase3_jit.rs` | 3 separate analysis passes (intervals, hotness, loop weight) could be 1 |
| M3 | `phase3_jit.rs` | `const_at.clear()` at every branch discards constants valid at known targets |
| M4 | `phase3_jit.rs` | Branch relaxation fills freed bytes with single-byte `0x90` NOPs |
| M5 | `interp.rs` | `maybe_superoptimize` called redundantly from both `precompile` and `call_fn` |
| M6 | `interp.rs` | Prefetch always issues from base pointer, not current batch offset |
| M7 | `interp.rs` | `unlikely` on hot loop guard in health-fused SIMD path (inverted hint) |
| M8 | `interp.rs` | SoA write-back in `integrate_vec3_linear_fused` is a full second O(n) pass |
| M9 | `interp.rs` | `integrate_step_adaptive` bypasses adaptive cache for 4-component fused path |
| M10 | `main.rs` | `detect_silent_issues` line-scans entire source on every execution |
| M11 | `borrowck.rs` | Fresh `BTreeMap`/`HashSet` allocated per function in borrow check |

### Low (polish / minor wins)

| # | Location | Issue |
|---|----------|-------|
| L1 | `phase6_simd.rs` | `load_aos4` not `#[inline(always)]` — inline into `load_aos8` not guaranteed |
| L2 | `phase6_simd.rs` | Feature detection dispatched on every `update_positions` call |
| L3 | `phase6_simd.rs` | Scalar head/tail in `update_avx2_fma` has no prefetch coverage |
