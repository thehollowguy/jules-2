# Jules Standard Library (Built-in Modules)

Jules follows a **small-core, composable, deterministic-friendly, ML-aware** standard library design.

## Core philosophy
- Small core; add capability through modules.
- Deterministic-by-default primitives where practical.
- ML-first ergonomics (tensor, nn, train, quant, model).

## Built-in module map

- `core`: `Some`, `None`, `Ok`, `Err`, `unwrap`, `is_some`, `is_none`, `is_ok`, `is_err`
- `math`: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `tanh`, `softmax`, `random`, `random_seed`, `rand_int`, `sigmoid`, `relu`, `lerp`, `smoothstep`, `dot2`, `length2`, `distance2`, `remap`, `clamp01`
- `tensor`: `zeros`, `ones`, `sum`, `mean`, `max`, `softmax`, `normalize`
- `nn`: `relu`, `gelu`, `sigmoid`, `tanh`, `cross_entropy`, `mse`
- `train`: `optimizer::create`, `optimizer::step`
- `data`: `dataloader`, `pipeline`
- `io`: `read_file`, `write_file`, `append_file`
- `sys`: `sys::*` system/runtime APIs
- `error`: `Ok`, `Err`, `unwrap`
- `diag`: `diag::warn`, `diag::perf_hint`
- `collections`: `HashMap::new`, `len`, `range`
- `compute`: `compute::device`, `compute::parallel_map`
- `quant`: `quant::int8_export`
- `model`: `load_ir`, `save_ir`, `load_weights`, `save_weights`
- `debug`: `dbg`, `debug::tensor_shape`, `debug::disable_jit`
- `sim`: `sim::world`, `sim::spawn`, `sim::step`, `sim::reset`, `sim::get_state`, `sim::state_tensor`, `sim::apply`, `sim::entity_count`, `sim::nearest_entity`, `sim::query_radius`
- `window`: `window::create`, `window::open`, `window::clear`, `window::draw_rect`, `window::present`, `window::close`, `window::input_key_down`, `window::size`, `window::title`, `window::frames`

## Runtime discovery
Use:

```jules
let mods = std::modules();
```

This returns a map from module name to built-in function names.

## Minimal sim + window loop

```jules
let w = sim::world(0.016, 42);
let e = sim::spawn(w, {
  position: [0.0, 0.0],
  velocity: [1.0, 0.0]
});
let win = window::create(800, 600, "Jules");

while window::open(win) {
  sim::step(w, 0.016);
  let nearby = sim::query_radius(w, [0.0, 0.0], 10.0);
  let n = sim::entity_count(w);
  window::clear(win);
  window::draw_rect(win, 20.0, 20.0, 16.0, 16.0);
  window::present(win);
}
```
