# Jules Performance Optimization(read until the end)

## Official ai Execution Protocol for Jules ML Language Optimization

**Status:** Source of truth
**Applies to:** Claude’s work on Jules ML performance, optimization, benchmarking, debugging, and validation
**Updated:** 2026-03-23

---

## 0. Purpose

This document is the **single authoritative protocol** for Claude when working on Jules ML language optimization.

The goal is simple: **maximize performance without sacrificing correctness, compatibility, or maintainability**. Humans do love turning one problem into seven, so this protocol keeps the work disciplined.

Every rule below is mandatory unless the user explicitly overrides it in a later instruction.

---

## 1. Operating Principles

Claude must follow these principles for every optimization task:

1. **Measure before optimizing.** No change is allowed without a baseline.
2. **Preserve backward compatibility.** Existing behavior, builtins, enums, and public APIs must remain stable unless the user explicitly requests a breaking change.
3. **Optimize one phase at a time.** Each phase must be isolated, implemented, benchmarked, and validated independently.
4. **Prove the speedup.** Any claimed improvement must be backed by documented benchmark evidence.
5. **Validate correctness first.** A faster bug is still a bug, just with better marketing.
6. **Explain why.** Every optimization must include a clear rationale, expected benefit, and risk assessment.
7. **Prefer the smallest effective change.** Do not add complexity unless it measurably helps.
8. **Keep the baseline runnable.** The interpreter must always remain functional and comparable to optimized paths.

---

## 2. Continuous Improvement Rule

**New rule for future revisions:** any modification to this protocol must be justified by one of the following:

* a newly observed failure mode,
* a benchmark gap,
* a correctness regression,
* a maintainability issue,
* or a real optimization opportunity.

If none of those exist, do **not** add extra rules. This prevents the protocol from ballooning into a ceremonial pile of Markdown that nobody reads.

---

## 3. Mandatory Pre-Action Requirements

Before writing, editing, or suggesting any code, Claude must complete the following in order:

### 3.1 Quote the applicable rules

Claude must explicitly quote the exact rules from this document that govern the task.

### 3.2 Build a structured reasoning log

Claude must produce a concise, structured internal plan containing:

* **Relevant rules:** all applicable rules from the sections below
* **Plan:** what phase, benchmark, workload, or micro-optimization is being targeted
* **Potential bugs and edge cases:** include 0, 1, -1, NaN, infinities, empty tensors, tiny values, huge values, alignment issues, aliasing, thread hazards, integer overflow, precision drift, and undefined behavior risks
* **Implementation steps:** file edits, feature flags, call-path changes, new tests, benchmark additions, and config changes
* **Verification plan:** unit tests, property tests, integration tests, benchmarks, profiling commands, and compatibility checks

### 3.3 Bug-check before output

No code or final technical recommendation may be output until the bug-check section is complete.

If any credible bug risk remains unresolved, Claude must stop and request clarification rather than guessing.

### 3.4 Output gate

Claude may only produce final code or final technical output after the documentable bug-check has passed.

---

## 4. Bug Check Protocol

Claude must mentally simulate and document the following before declaring success:

### 4.1 Test simulation

* `cargo test --all-features`
* `cargo test --no-default-features`
* `cargo test --features phaseN`
* `cargo clippy --all-targets -- -D warnings`
* `cargo fmt --check`

### 4.2 Benchmark simulation

* `cargo bench` on the affected benchmark(s)
* Any phase-specific microbenchmarks
* Any end-to-end neural-network workload benchmark used as a compatibility check

### 4.3 Numerical and behavioral verification

* Verify no new panics
* Verify no regressions in edge cases
* Verify no unexpected numerical drift
* Verify backward compatibility between the baseline interpreter and optimized paths
* Verify that any floating-point differences are within the documented epsilon, if epsilon-based comparison is appropriate

### 4.4 Memory and allocation sanity checks

* Peak allocation must not increase without explicit justification
* No memory leaks
* Any copy-on-write or tensor-allocation change must be validated
* Any lifetime or aliasing risk must be explicitly addressed

### 4.5 Failure handling

If any test, benchmark, or sanity check fails, Claude must:

1. revert the plan,
2. explain why it failed,
3. and choose a narrower or safer approach.

**A task is not complete unless the response explicitly includes:**

`✅ Bug Check Passed`

---

## 5. Performance Measurement Rules

### 5.1 Baseline creation

Before any optimization, Claude must record a baseline that includes:

* execution time in milliseconds,
* memory usage, preferably RSS in MB,
* CPU cycles per operation where available,
* and a timestamped benchmark result.

### 5.2 Benchmark commands

Use benchmark runs such as:

```bash
cargo build --release
./benches/run_all.sh > results/baseline_$(date +%s).txt
```

### 5.3 Compare interpreted and optimized paths in the same benchmark context

Where possible, benchmark:

* the interpreted path,
* the JIT path,
* the LLVM path,
* and any other selected optimization path

under comparable conditions.

### 5.4 Avoid measurement noise

* Do not mix setup and teardown into timed sections
* Measure only the operation being optimized
* Avoid separate unrelated runs when the same workload can be compared in one run
* Keep environment variables, CPU frequency, and workload parameters consistent

### 5.5 Report speedups clearly

* Track speedups multiplicatively relative to the baseline
* Document both predicted and actual improvements
* Never claim a speedup without numbers

---

## 6. Architecture Organization Rules

### 6.1 One module per phase

Use a clean phase separation such as:

* `src/interp.rs` for the core interpreter
* `src/phase1_dispatch.rs`
* `src/phase2_fastval.rs`
* `src/phase3_jit.rs`
* `src/phase4_llvm.rs`
* `src/phase5_cow.rs`
* `src/phase6_simd.rs`

### 6.2 Feature flags

Each phase must have its own Cargo feature flag.

Suggested defaults:

* enabled by default: `phase1`, `phase2`
* optional: `phase3-jit`, `phase4-llvm`, `phase5-cow`, `phase6-simd`

### 6.3 Unified dispatch

* Use a consistent `OptimizationLevel` enum
* All execution paths must branch through the same selection interface
* Every phase must be independently selectable or combinable, as intended by the design

### 6.4 Avoid phase entanglement

Do not mix multiple optimization phases inside one function unless that coupling is explicitly required and documented.

---

## 7. Testing and Validation Rules

After every change, Claude must verify the relevant test surface.

### 7.1 Required test commands

* `cargo test --all-features`
* `cargo test --no-default-features`
* `cargo test --features phaseN`
* `cargo clippy --all-targets`
* `cargo fmt --check`

### 7.2 Required test types

* Property-based numeric tests
* Integration tests
* End-to-end workload tests
* Compatibility tests against the baseline interpreter

### 7.3 Correctness standard

* Optimized paths must match the baseline exactly when exact comparison is valid
* For floating-point math, use a documented epsilon only when exact match is not realistic
* Any acceptable deviation must be justified and measured

### 7.4 Simulation requirement

Claude must mentally validate the test suite and benchmark outputs before reporting success.

---

## 8. Debugging and Profiling Rules

### 8.1 CPU profiling

Use cycle-level profiling when investigating hotspots:

```bash
perf record -e cycles,cache-references,cache-misses ./target/release/jules ./test_program.julius
perf report
perf stat
```

### 8.2 Trace inspection

Use trace-level logging around the relevant test or benchmark:

```bash
RUST_LOG=trace cargo test test_matmul_128 2>&1 | head -100
```

### 8.3 Memory profiling

Use allocation or heap profiling for memory-sensitive work:

```bash
valgrind --tool=massif --max-snapshots=100 ./target/release/jules ./memory_heavy_program.julius
ms_print massif.out.* | head -200
```

### 8.4 Compare before and after

For every profiling session, compare:

* hotspot location,
* branch behavior,
* cache behavior,
* allocation count,
* and any change in assembly or IR where relevant.

---

## 9. Code Quality Rules

### 9.1 Document the reason for every optimization

Every performance change must explain:

* why it exists,
* what bottleneck it targets,
* how much improvement is expected,
* what numerical or behavioral risks exist,
* and how edge cases are handled.

### 9.2 Keep logic isolated

* Prefer traits, helpers, and modules over tangled conditionals
* Keep phase-specific code contained
* Avoid “clever” shortcuts that obscure behavior

### 9.3 Avoid undocumented risk

* No undocumented `unsafe`
* No hidden assumptions about layout, aliasing, or lifetimes
* No dependency additions without review

---

## 10. Optimization Implementation Rules

For each phase or optimization task, Claude must follow this workflow:

### Step 1: Create or identify a microbenchmark

* Quote the applicable rules
* Identify the bottleneck clearly
* Tie the benchmark to a real workload where possible

### Step 2: Implement the optimization

* Include WHY comments where the code is non-obvious
* Keep the change minimal
* Preserve the baseline path

### Step 3: Measure the full program

* Re-run the relevant benchmark(s)
* Compare baseline and optimized paths
* Record predicted vs. actual results

### Step 4: Execute the Bug Check Protocol

* Confirm tests, benchmarks, and compatibility checks
* Verify edge cases and memory behavior

### Step 5: Update the optimization plan

Update `PERFORMANCE_OPTIMIZATION_PLAN.md` with:

* benchmark name,
* baseline numbers,
* optimized numbers,
* speedup,
* test results,
* and any caveats.

### Step 6: Only then report success

Only after the bug-check is complete may Claude output final code or the final implementation summary.

---

## 11. Dependency and Platform Rules

### 11.1 Confirm dependency versions

Before using or changing dependencies, verify the versions and compatibility of:

* `std`
* `cranelift`
* `inkwell`
* `portable_simd`

### 11.2 Dependency hygiene

* Do not introduce unvetted dependencies
* Prefer stable, well-understood tooling
* Justify every dependency addition with a clear need

---

## 12. Decision Trees

### 12.1 Should this function be optimized?

Optimize only if one or more are true:

* the hot path consumes more than 10% of CPU time,
* it is called frequently, for example more than 1000 times per second,
* profiling identifies it as a bottleneck,
* or the current implementation is visibly non-scalable.

### 12.2 Which optimization phase fits the bottleneck?

* **Dispatch bottlenecks:** phase1 dispatch optimizations
* **Value representation overhead:** phase2 fast value optimizations
* **Dynamic compilation opportunities:** phase3 JIT
* **IR generation and codegen bottlenecks:** phase4 LLVM
* **Allocation pressure / cloning overhead:** phase5 COW
* **Vectorizable numeric loops:** phase6 SIMD

### 12.3 How to handle regressions

1. confirm with benchmarks,
2. profile the hotspot,
3. compare the relevant assembly or IR,
4. roll back incrementally,
5. reapply only the useful part,
6. use git bisect if necessary.

---

## 13. Pre-Commit Checklist

### Phase N: [Phase Name] - Pre-Commit Checklist

#### Code Quality

* [ ] `cargo test --all-features` passes
* [ ] `cargo clippy --all-targets` passes cleanly
* [ ] `cargo fmt --check` passes
* [ ] No undocumented `unsafe` unless explicitly justified

#### Performance Validation

* [ ] Baseline benchmark recorded
* [ ] Optimized benchmark recorded
* [ ] Required speedup demonstrated or the optimization rejected
* [ ] No regressions in unrelated benchmarks

#### Bug Check Protocol

* [ ] Full test suite mentally simulated and documented
* [ ] 5+ edge-case numerical checks considered
* [ ] Backward-compatibility test passed on a real neural-network workload
* [ ] No new panics, allocation spikes, or memory leaks
* [ ] `✅ Bug Check Passed` explicitly included

#### Testing and Documentation

* [ ] Property tests green
* [ ] Integration tests green
* [ ] `PERFORMANCE_OPTIMIZATION_PLAN.md` updated
* [ ] Benchmark results documented

#### Git Hygiene

* [ ] Single logical commit
* [ ] Commit message includes benchmark names and exact speedup numbers

---

## 14. Required Output Format for Claude

When Claude is asked to optimize code under this protocol, the response should follow this order:

1. Relevant rules quoted
2. Structured plan
3. Bug and edge-case analysis
4. Implementation summary
5. Verification plan
6. `✅ Bug Check Passed`
7. Final code or final recommendation

If code is not yet safe to output, Claude must stop before step 7.

---

## 15. Large-Scale Fixes, Giant Debug Runs, and Major Improvements

When the task is unusually large, Claude must use a stricter process instead of improvising like a caffeinated raccoon.

### 15.1 Huge fixes

For large bug fixes or refactors:

* Break the work into the smallest safe sequence of patches
* Define the exact failure mode before changing code
* Identify the minimum set of files and call paths involved
* Prefer reversible steps with clear checkpoints
* Validate each patch independently before moving to the next
* Keep the baseline path intact until the replacement is proven

### 15.2 Giant debug runs

For deep debugging sessions or long-running investigations:

* Start with the narrowest failing test or reproducer available
* Add targeted logging before broad instrumentation
* Capture baseline behavior first, then compare after each change
* Reduce the search space with feature flags, smaller inputs, or phase isolation
* Escalate from unit tests to integration tests to workload runs only when needed
* Record every hypothesis, result, and eliminated cause in the working log

### 15.3 Huge improvements

For major performance improvements or architectural wins:

* Treat the change as a multi-stage program, not a single edit
* Define the target speedup and acceptable tradeoffs up front
* Benchmark each sub-change separately so the gain is attributable
* Validate both the fast path and the fallback path
* Document the numerical, memory, and compatibility impact of the improvement
* Reject any “big win” that cannot be shown in numbers

---

## 16. Final Non-Negotiables

* Never assume correctness
* Never claim a speedup without evidence
* Never skip edge cases
* Never collapse multiple phases into one unreviewed change
* Never remove the baseline interpreter path
* Never output code before the bug check passes
* Never hide uncertainty
* Always document why the change exists
* Always keep the optimization plan current

---

## 16. Revision Log

### 2026-03-23

* Created the consolidated execution protocol
* Added a continuous improvement rule to prevent pointless rule bloat
* Clarified the order of reasoning, testing, benchmarking, and reporting
* Reorganized the document for easier phase-by-phase execution




make an edit to this file to improve it every time you are done with a task

---
### Edit Log (automatically appended)

- 2026-03-29: Added phase feature flags, exposed `phase6_simd`, implemented `phase6_simd::update_positions` (portable callsite), and added `bench-ecs` microbenchmark and `PERFORMANCE_OPTIMIZATION_PLAN.md` scaffold. Follow-up: run the benchmark locally/CI and record baseline numbers in `PERFORMANCE_OPTIMIZATION_PLAN.md`.