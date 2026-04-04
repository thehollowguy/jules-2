# Chess ML Learning Environment Benchmark

This repository now includes a high-throughput chess-like learning environment and benchmark path focused on **training loop throughput**.

## What was added

- `chess_ml.rs`: A compact, allocation-light environment using bitboards and a tiny policy learner.
- `bench-chess-ml` binary: Runs the Jules/Rust training loop and compares it with a Python baseline.

## Run benchmark

```bash
cargo run --release --bin bench-chess-ml -- 50000 24
```

Arguments:
- `episodes` (default: `200000`)
- `max_steps` (default: `32`)

## Output fields

- Jules engine elapsed time and steps/s
- Python baseline elapsed time and steps/s
- Calculated speedup (Jules/Python)

## Performance strategy

The environment is intentionally optimized for learning-loop throughput:

- bitboard storage (`u64`) instead of heap board objects
- fixed move buffer (`[Move; 128]`) to avoid per-step allocation
- branch-light move generation
- xorshift RNG for low overhead
- in-place linear policy updates

This gives a practical training benchmark for Jules ML/runtime work while remaining easy to iterate on.

## Jules-language chess ML script

A Jules-native training script is included at `ml_chess.jules`.

Run it directly:

```bash
cargo run --bin jules -- run ml_chess.jules
```

Benchmark that script against a Python baseline:

```bash
cargo run --release --bin bench-jules-ml-chess -- ml_chess.jules
```
