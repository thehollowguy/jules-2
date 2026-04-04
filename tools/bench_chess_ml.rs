// Chess learning-environment benchmark.
// Run with: cargo run --release --bin bench-chess-ml -- <episodes> <max_steps>

use std::io::Write;
use std::process::{Command, Stdio};

use jules::chess_ml::{train_chess_policy_batched, train_chess_policy_gpu, train_chess_policy_soa};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let episodes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let max_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let batch: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let envs: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let device = args.get(5).map(String::as_str).unwrap_or("cpu");

    println!("bench-chess-ml episodes={episodes} max_steps={max_steps} batch={batch} envs={envs} device={device}");

    let jules = if device == "gpu" {
        train_chess_policy_gpu(episodes, envs, max_steps, batch, 0xC0FFEE)
            .expect("gpu training path failed")
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

    let out = child.wait_with_output().map_err(|e| format!("python failed: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).to_string());
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.split_whitespace();
    let elapsed: f64 = it.next().ok_or("missing elapsed")?.parse().map_err(|_| "bad elapsed")?;
    let steps_per_sec: f64 = it.next().ok_or("missing steps/s")?.parse().map_err(|_| "bad steps/s")?;
    Ok((elapsed, steps_per_sec))
}
