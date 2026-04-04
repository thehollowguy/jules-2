// Chess learning-environment benchmark.
// Run with: cargo run --release --bin bench-chess-ml -- <episodes> <max_steps> <batch> <envs> <device>

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use jules::chess_ml::{
    eval_policy_vs_random, train_chess_policy_batched, train_chess_policy_batched_from,
    train_chess_policy_gpu, train_chess_policy_soa,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let episodes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let max_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let batch_raw: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let envs_raw: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let batch = batch_raw.max(1).min(4096);
    let envs = envs_raw.max(1).min(4096);
    let device = args.get(5).map(String::as_str).unwrap_or("cpu");
    if batch != batch_raw || envs != envs_raw {
        println!("⚠️ safety clamp applied: batch {batch_raw}->{batch}, envs {envs_raw}->{envs}");
    }

    println!(
        "bench-chess-ml episodes={episodes} max_steps={max_steps} batch={batch} envs={envs} device={device}"
    );

    let jules = if device == "gpu" {
        train_chess_policy_gpu(episodes, envs, max_steps, batch, 0xC0FFEE).unwrap_or_else(|e| {
            println!("⚠️ GPU path unavailable ({e}), falling back to CPU SoA path.");
            train_chess_policy_soa(episodes, envs, max_steps, batch, 0xC0FFEE)
        })
    } else if envs > 1 {
        train_chess_policy_soa(episodes, envs, max_steps, batch, 0xC0FFEE)
    } else {
        train_chess_policy_batched(episodes, max_steps, batch, 0xC0FFEE)
    };
    println!(
        "Jules engine: {:.3}s, steps={}, {:.0} steps/s, win_rate={:.2}%",
        jules.elapsed.as_secs_f64(),
        jules.total_steps,
        jules.steps_per_sec,
        jules.win_rate * 100.0
    );

    match python_baseline(episodes, max_steps) {
        Ok(py) => {
            println!("Python baseline: {:.3}s, {:.0} steps/s", py.0, py.1);
            let speedup = jules.steps_per_sec / py.1.max(1e-9);
            println!("Speedup (Jules/Python): {:.2}x", speedup);
            if speedup >= 1.0 {
                println!("✅ Jules is faster than Python on this benchmark.");
            } else {
                println!("⚠️ Jules is slower than Python on this machine/run.");
            }
        }
        Err(e) => {
            println!("⚠️ Python baseline skipped: {e}");
        }
    }

    match jax_baseline(episodes, max_steps, envs.max(1)) {
        Ok((elapsed, steps_per_sec, backend)) => {
            println!(
                "JAX baseline ({backend}): {:.3}s, {:.0} steps/s",
                elapsed, steps_per_sec
            );
            let speedup = jules.steps_per_sec / steps_per_sec.max(1e-9);
            println!("Speedup (Jules/JAX): {:.2}x", speedup);
            if speedup >= 1.0 {
                println!("✅ Jules is faster than JAX on this benchmark.");
            } else {
                println!("⚠️ Jules is slower than JAX on this machine/run.");
            }
        }
        Err(e) => {
            println!("⚠️ JAX baseline skipped: {e}");
        }
    }

    let target_strength = 0.70f32;
    let mut train_seed = 0xC0FFEEu64;
    let mut total_train_episodes = 0usize;
    let mut weights = [0.03f32, 0.0, 0.0, 0.2, 0.01, -0.01, 0.0, 0.0];
    let eval_games = 1_500usize;
    let chunk = 100_000usize;
    println!(
        "Strength training: target win-rate vs random = {:.0}% (eval games={eval_games})",
        target_strength * 100.0
    );
    loop {
        let (chunk_result, new_weights) = train_chess_policy_batched_from(
            chunk,
            max_steps.max(32),
            batch.max(64),
            train_seed,
            weights,
        );
        weights = new_weights;
        total_train_episodes += chunk;
        train_seed = train_seed.wrapping_add(1);
        let eval_wr =
            eval_policy_vs_random(weights, eval_games, max_steps.max(32), train_seed ^ 0xA11CE);
        println!(
            "  episodes={} chunk_steps/s={:.0} eval_win_rate={:.2}%",
            total_train_episodes,
            chunk_result.steps_per_sec,
            eval_wr * 100.0
        );
        if eval_wr >= target_strength || total_train_episodes >= 2_000_000 {
            println!(
                "Final strength: eval_win_rate={:.2}% after {} episodes",
                eval_wr * 100.0,
                total_train_episodes
            );
            break;
        }
    }
}

fn python_baseline(episodes: usize, max_steps: usize) -> Result<(f64, f64), String> {
    let script = r#"
import time, random, sys
E = int(sys.argv[1]); M = int(sys.argv[2])
w = [0.03,0.0,0.0,0.2,0.01,-0.01,0.0,0.0]
steps = 0
t0 = time.perf_counter()
for _ in range(E):
    wp, bp = 8, 8
    wa, ba = 0.0, 0.0
    white = True
    for _ in range(M):
        f = [1.0, wp, bp, wp-bp, wa, ba, 0.0, 0.0]
        score = sum(fi*wi for fi,wi in zip(f,w)) + random.random()*0.01
        reward = 0.001 if score > 0 else -0.001
        if white:
            wa += 1.0
            if random.random() < 0.03 and bp > 0:
                bp -= 1; reward += 0.1
        else:
            ba += 1.0
            if random.random() < 0.03 and wp > 0:
                wp -= 1; reward -= 0.1
        for i in range(8):
            x = f[i]
            if x > 32: x = 32
            if x < -32: x = -32
            w[i] += 0.002 * reward * x
        white = not white
        steps += 1
elapsed = time.perf_counter() - t0
print(f"{elapsed} {steps/elapsed if elapsed>0 else 0}")
"#;

    let mut child = Command::new("python3")
        .arg("-")
        .arg(episodes.to_string())
        .arg(max_steps.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("python3 not available: {e}"))?;

    {
        let stdin = child.stdin.as_mut().ok_or("failed to open stdin")?;
        stdin
            .write_all(script.as_bytes())
            .map_err(|e| format!("failed to write python script: {e}"))?;
    }

    let out = child
        .wait_with_output()
        .map_err(|e| format!("python failed: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).to_string());
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.split_whitespace();
    let elapsed: f64 = it
        .next()
        .ok_or("missing elapsed")?
        .parse()
        .map_err(|_| "bad elapsed")?;
    let steps_per_sec: f64 = it
        .next()
        .ok_or("missing steps/s")?
        .parse()
        .map_err(|_| "bad steps/s")?;
    Ok((elapsed, steps_per_sec))
}

fn jax_baseline(
    episodes: usize,
    max_steps: usize,
    envs: usize,
) -> Result<(f64, f64, String), String> {
    let script = r#"
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys, time

try:
    import jax
    import jax.numpy as jnp
except Exception as e:
    print(f"ERR:jax_missing:{e}")
    raise SystemExit(7)

E = int(sys.argv[1]); M = int(sys.argv[2]); N = max(1, int(sys.argv[3]))

weights = jnp.array([0.03,0.0,0.0,0.2,0.01,-0.01,0.0,0.0], dtype=jnp.float32)

@jax.jit
def run_training(weights, key):
    def episode_fn(_, state):
        w, key = state
        wp = jnp.full((N,), 8.0, dtype=jnp.float32)
        bp = jnp.full((N,), 8.0, dtype=jnp.float32)
        wa = jnp.zeros((N,), dtype=jnp.float32)
        ba = jnp.zeros((N,), dtype=jnp.float32)

        def step_fn(_, carry):
            wp, bp, wa, ba, w, key = carry
            key, k1, k2 = jax.random.split(key, 3)
            f = jnp.stack([jnp.ones_like(wp), wp, bp, wp - bp, wa, ba, wa * 0.1, ba * 0.1], axis=1)
            score = jnp.sum(f * w[None, :], axis=1) + (jax.random.uniform(k1, (N,), dtype=jnp.float32) - 0.5) * 0.01
            reward = jnp.where(score > 0, 0.001, -0.001)

            hit_prob = jax.random.uniform(k2, (N,), dtype=jnp.float32)
            white_hits = (hit_prob < 0.03) & (bp > 0)
            bp = bp - white_hits.astype(jnp.float32)
            reward = reward + white_hits.astype(jnp.float32) * 0.1
            wa = wa + 1.0

            grad = jnp.mean(f * reward[:, None], axis=0)
            w = w + 0.002 * grad
            return (wp, bp, wa, ba, w, key)

        wp, bp, wa, ba, w, key = jax.lax.fori_loop(0, M, step_fn, (wp, bp, wa, ba, w, key))
        return (w, key)

    return jax.lax.fori_loop(0, E, episode_fn, (weights, key))

key = jax.random.PRNGKey(0xC0FFEE)
run_training(weights, key)[0].block_until_ready()  # warmup compile

t0 = time.perf_counter()
weights, key = run_training(weights, key)
weights.block_until_ready()
elapsed = time.perf_counter() - t0
steps = E * M * N
backend = jax.default_backend()
print(f"{elapsed} {steps/elapsed if elapsed>0 else 0} {backend}")
"#;

    let mut child = Command::new("python3")
        .arg("-")
        .arg(episodes.to_string())
        .arg(max_steps.to_string())
        .arg(envs.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("python3 not available: {e}"))?;

    child
        .stdin
        .as_mut()
        .ok_or("failed to open stdin")?
        .write_all(script.as_bytes())
        .map_err(|e| format!("failed to write JAX script: {e}"))?;

    let out = child
        .wait_with_output()
        .map_err(|e| format!("JAX run failed: {e}"))?;

    if !out.status.success() {
        let msg = format!(
            "{} {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if msg.contains("ERR:jax_missing:") {
            return Err("JAX not installed in python3 environment".to_string());
        }
        return Err(msg);
    }

    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.split_whitespace();
    let elapsed: f64 = it
        .next()
        .ok_or("missing elapsed")?
        .parse()
        .map_err(|_| "bad elapsed")?;
    let steps_per_sec: f64 = it
        .next()
        .ok_or("missing steps/s")?
        .parse()
        .map_err(|_| "bad steps/s")?;
    let backend = it.next().unwrap_or("unknown").to_string();

    Ok((elapsed, steps_per_sec, backend))
}
