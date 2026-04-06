# Jules JAX Backend (Optimized ML Training Path)

Jules now includes a JAX/XLA backend bridge for high-throughput model training and export.

## What this gives you

- JIT-compiled training step (`jax.jit`) for low Python overhead.
- `optax.adamw` optimizer support.
- Fast MLP training on CPU/GPU/TPU via XLA.
- Export to:
  - `jules_jax_fp32.npz` (full-precision checkpoint)
  - `jules_int8_weights.json` (per-output-channel INT8 + scales for Jules low-byte inference)

## 1) Prepare a Jules model IR JSON

Example `model_ir.json`:

```json
{
  "schema_version": 1,
  "model_name": "PolicyNet",
  "input_dim": 128,
  "layers": [256, 256, 10],
  "activation": "gelu",
  "task": "classification"
}
```

Minimal beginner IR is also accepted and auto-filled:

```json
{
  "input_dim": 128,
  "layers": [256, 256, 10]
}
```

This is normalized to strict schema (`schema_version=1`, inferred `model_name`,
default `activation=gelu`, default `task=classification`) before validation.

Strict IR spec is validated before training:
- required keys only: `schema_version`, `model_name`, `input_dim`, `layers`, `activation`, `task`
- `schema_version` must be `1`
- positive integer dimensions/layers only
- `activation` in `{gelu,relu,silu,tanh}`
- `task` in `{classification,regression}`

## 2) Prepare dataset NPZ

Must contain:
- `x_train`: `float32`, shape `[N, input_dim]`
- `y_train`: `int32`, shape `[N]`

## 3) Train with JAX backend

```bash
# Option A: call script directly
python scripts/jules_jax_backend.py \
  --ir model_ir.json \
  --dataset train.npz \
  --out artifacts/jax \
  --epochs 40 \
  --batch-size 256 \
  --lr 3e-4
```

```bash
# Option B: use Jules CLI backend toggle
jules train my_agent.jules --ml-backend jax --jax-dataset train.npz --jax-out artifacts/jax
```

When using `jules train ... --ml-backend jax`, Jules parses your `.jules` model syntax
and auto-generates a compatible JAX IR for dense MLP-style models, so the same source
model can be trained with either backend.

`jules train` also prints a capability matrix for models found in source, for example:

```text
PolicyNet: Jules ✅  JAX ✅
VisionNet: Jules ✅  JAX ❌ (why: unsupported layer for JAX bridge export: Conv2d { ... })
```

If Python/JAX dependencies are missing, `jules train --ml-backend jax` now runs a preflight check
and prints a direct install recipe (`pip install jax optax numpy`) before launch.

## 4) Consume artifacts in Jules

Use `jules_int8_weights.json` to populate `Int8LinearWeights`-style inference layers
(`in_dim`, `out_dim`, flat `qweights`, and `scales`) and keep `bias` in fp32.

---

This is the recommended path for large-scale training where you want Jules runtime inference,
but JAX-grade compiler/runtime acceleration during training.
