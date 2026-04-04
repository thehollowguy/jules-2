// Chess learning-environment benchmark.
// Run with: cargo run --release --bin bench-chess-ml -- <episodes> <max_steps>

use std::io::Write;
use std::process::{Command, Stdio};

use jules::chess_ml::{
    train_chess_policy_batched, train_chess_policy_gpu, train_chess_policy_soa,
    train_chess_policy_soa_pipelined, train_chess_policy_soa_quantized,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let episodes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let max_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let batch: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let envs: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let device = args.get(5).map(String::as_str).unwrap_or("cpu");
    let lookahead: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(256);

    println!(
        "bench-chess-ml episodes={episodes} max_steps={max_steps} batch={batch} envs={envs} device={device} lookahead={lookahead}"
    );

    let jules = if device == "gpu" {
        train_chess_policy_gpu(episodes, envs, max_steps, batch, 0xC0FFEE)
            .expect("gpu training path failed")
    } else if device == "pipeline" {
        train_chess_policy_soa_pipelined(episodes, envs, max_steps, batch, 0xC0FFEE, lookahead)
    } else if device == "quant" {
        train_chess_policy_soa_quantized(episodes, envs, max_steps, batch, 0xC0FFEE)
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

    match jax_baseline(episodes, max_steps) {
        Ok(jx) => {
            println!("JAX baseline: {:.3}s, {:.0} steps/s", jx.0, jx.1);
            let speedup = jules.steps_per_sec / jx.1.max(1e-9);
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

fn jax_baseline(episodes: usize, max_steps: usize) -> Result<(f64, f64), String> {
    let script = r#"
import sys, time
try:
    import jax
    import jax.numpy as jnp
except Exception as e:
    print(f"ERR:missing_jax:{e}")
    raise SystemExit(7)

E = int(sys.argv[1]); M = int(sys.argv[2])

@jax.jit
def step_fn(state, _):
    wp, bp, wa, ba, white, w = state
    f = jnp.array([1.0, wp, bp, wp - bp, wa, ba, wa * 0.1, ba * 0.1], dtype=jnp.float32)
    score = jnp.dot(f, w)
    reward = jnp.where(score > 0.0, 0.001, -0.001)
    wa2 = wa + 1.0
    bp2 = jnp.maximum(0.0, bp - jnp.where(score > 0.0, 1.0, 0.0))
    w2 = w + 0.002 * reward * jnp.clip(f, -32.0, 32.0)
    return (wp, bp2, wa2, ba, 1.0 - white, w2), reward

@jax.jit
def run_fused(state, steps):
    return jax.lax.scan(step_fn, state, xs=None, length=steps)

def run():
    w = jnp.array([0.03,0.0,0.0,0.2,0.01,-0.01,0.0,0.0], dtype=jnp.float32)
    state = (8.0, 8.0, 0.0, 0.0, 1.0, w)
    total_steps = E * M
    # Compile before timing so we benchmark steady-state fused execution.
    state, _ = run_fused(state, total_steps)
    jax.block_until_ready(state[5])
    state = (8.0, 8.0, 0.0, 0.0, 1.0, w)
    t0 = time.perf_counter()
    state, _ = run_fused(state, total_steps)
    jax.block_until_ready(state[5])
    elapsed = time.perf_counter() - t0
    print(f"{elapsed} {total_steps/elapsed if elapsed>0 else 0}")

run()
"#;

    let mut child = Command::new("python3")
        .arg("-")
        .arg(episodes.to_string())
        .arg(max_steps.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("python3 not available: {e}"))?;

    {
        let stdin = child.stdin.as_mut().ok_or("failed to open stdin")?;
        stdin
            .write_all(script.as_bytes())
            .map_err(|e| format!("failed to write jax script: {e}"))?;
    }

    let out = child
        .wait_with_output()
        .map_err(|e| format!("jax process failed: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        return Err(format!(
            "jax baseline unavailable or failed: {stdout} {stderr}"
        ));
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
