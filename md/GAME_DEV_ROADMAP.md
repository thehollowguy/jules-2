# Game Development Roadmap for Jules

This document lists the major game-dev features the user requested and provides a minimal plan and initial scaffolding so work can proceed in small, measurable steps.

What Jules absolutely does NOT have yet (requested):

- Frame-by-frame debugger for simulations
- Visual scene editor
- Asset importer pipeline
- Shader tooling
- Networking tools
- Profiling that doesn’t lie to you
- Hot reload that actually works

For each item below there is: a short justification, a minimal-first implementation plan, verification steps, and potential risks.

1) Frame-by-frame debugger
- Why: deterministically inspect simulation state per tick; invaluable for debugging physics, AI, and gameplay.
- Minimal plan: provide a runtime API for stepping one frame, pausing, inspecting entity state, and rewinding a short history buffer. Implement as a non-invasive wrapper around the existing scheduler and ECS world.
- Verification: unit tests that step a deterministic simulation and assert inspected state; record baseline timings to ensure debugger overhead is acceptable.
- Risk: increased memory for history buffer; concurrency hazards when pausing multi-threaded systems.

2) Visual scene editor
- Why: WYSIWYG editing of entity placement, components, and prefabs accelerates level design.
- Minimal plan: provide a simple JSON/YAML scene format and a small desktop/web editor prototype that can load/save it. Start with import/export and live-preview via the runtime FFI.
- Verification: round-trip tests for scene serialization; manual smoke tests with small scenes.

3) Asset importer pipeline
- Why: convert common formats (PNG, WAV, GLTF) into runtime-optimized blobs; handle compression, packing, and metadata.
- Minimal plan: create a command-line importer that outputs engine-ready asset bundles and a simple manifest format. Add integration tests on sample assets.

4) Shader tooling
- Why: authoring, compiling, and hot-reloading shaders for rendering and compute.
- Minimal plan: integrate a minimal shader compiler wrapper (glslang / shaderc) or host a simple transpiler to SPIR-V; support hot-reload of compiled shaders.

5) Networking tools
- Why: built-in networking helpers accelerate multiplayer prototypes (replication, RPC, latency compensation).
- Minimal plan: provide a lightweight reliable UDP transport, a serialization schema for component/state sync, and sample server/client helpers.

6) Honest profiling
- Why: profiling that reports real wall-clock + CPU cycles per subsystem (ECS, physics, rendering, AI) without aggregation illusions.
- Minimal plan: add a low-overhead instrumentation API (scope timers, allocation counters) and export to `perf`/Flamegraph-compatible traces. Provide a deterministic microbenchmark harness.

7) Hot reload that works
- Why: iterative development speed-up — live code and asset reloading without full process restart.
- Minimal plan: implement a safe module-reload mechanism for script code and a file-watcher-driven asset reload pipeline. Start with limited guarantees (single-threaded reload) and expand.

Overall approach & priorities
- Phase the work: scaffolding → microbenchmarks/tests → minimal working prototype → profiling/optimization.
- Keep the baseline interpreter unchanged; gate heavy optimizations behind feature flags.
- Measure before optimizing and provide a reproducible `bench-ecs`-style harness for each feature.

Next steps (suggested immediate actions):
1. Add lightweight scaffolds and expose them from the crate (done in this commit).
2. Create micro-issues for each feature with a small reproducer and required assets.
3. Implement Frame Debugger v0 (API + tests), then iterate with profiling data.

Add measured numbers and precise verification steps into `PERFORMANCE_OPTIMIZATION_PLAN.md` as experiments are performed.
