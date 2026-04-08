use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;
use std::time::SystemTime;

use jules::{CompileUnit, Pipeline, PipelineResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchMode {
    Full,
    AotTime,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    let mode = parse_mode(args.get(3).map(String::as_str));
    let seed: i64 = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(default_seed);
    let samples: usize = 10;

    println!("bench-interp-vs-rust n={n} iters={iters} mode={mode:?} seed={seed}");

    let jules_src = format!(
        r#"
fn main() {{
  let mut s = {seed};
  let mut i = 0;
  while i < {n} {{
    s = ((s * 1664525 + (i * 1013904223) + 97) % 2147483647);
    i = i + 1;
  }}
  return s;
}}
"#,
        seed = seed,
        n = n
    );

    // Jules compile benchmark (10 runs, averaged)
    let mut jules_compile_s = 0.0f64;
    let mut program = None;
    for sample in 0..samples {
        let pipeline = Pipeline::new();
        let compile_start = Instant::now();
        for i in 0..iters {
            let mut unit = CompileUnit::new(format!("<bench:{sample}:{i}>"), &jules_src);
            let result = pipeline.run(&mut unit);
            if unit.has_errors() {
                eprintln!("compile diagnostics: {}", unit.diags.len());
                std::process::exit(1);
            }
            if i + 1 == iters && sample + 1 == samples {
                match result {
                    PipelineResult::Ok(p) => program = Some(p),
                    _ => {
                        eprintln!("pipeline failed to produce executable program");
                        std::process::exit(1);
                    }
                }
            }
        }
        jules_compile_s += compile_start.elapsed().as_secs_f64();
    }
    jules_compile_s /= samples as f64;

    if mode == BenchMode::AotTime {
        let rust_compile_s = rustc_compile_baseline(n);
        println!(
            "Jules AoT compile(avg {samples}): {:.6}s total ({:.6}s/iter)",
            jules_compile_s,
            jules_compile_s / iters as f64
        );
        if let Some(rc) = rust_compile_s {
            println!("Rust AoT compile:            {:.6}s total (rustc -O)", rc);
            println!(
                "AoT compile ratio (Jules/Rust): {:.2}x",
                jules_compile_s / rc.max(1e-9)
            );
        } else {
            println!("Rust AoT compile:            skipped (rustc unavailable)");
        }
        return;
    }

    // Jules runtime benchmark (interpreter)
    let mut interp = jules::interp::Interpreter::new();
    interp.load_program(&program.expect("program should exist"));
    let check_jules = match interp.call_fn("main", vec![]) {
        Ok(jules::interp::Value::I64(v)) => v,
        Ok(jules::interp::Value::I32(v)) => v as i64,
        Ok(v) => {
            eprintln!("unexpected return value from jules kernel: {v:?}");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("runtime error: {}", e.message);
            std::process::exit(1);
        }
    };
    let check_rust = rust_kernel(n, seed);
    if check_jules != check_rust {
        eprintln!(
            "correctness mismatch: jules={check_jules}, rust={check_rust}, n={n}, seed={seed}"
        );
        std::process::exit(2);
    }

    let mut jules_runtime_s = 0.0f64;
    let mut jules_checksum = 0i64;
    for _ in 0..samples {
        let run_start = Instant::now();
        for _ in 0..iters {
            match interp.call_fn("main", vec![]) {
                Ok(jules::interp::Value::I64(v)) => {
                    jules_checksum = jules_checksum.wrapping_add(black_box(v));
                }
                Ok(jules::interp::Value::I32(v)) => {
                    jules_checksum = jules_checksum.wrapping_add(black_box(v as i64));
                }
                Ok(v) => {
                    eprintln!("unexpected return value from jules kernel: {v:?}");
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("runtime error: {}", e.message);
                    std::process::exit(1);
                }
            }
        }
        jules_runtime_s += run_start.elapsed().as_secs_f64();
    }
    jules_runtime_s /= samples as f64;

    // Rust runtime baseline
    let mut rust_runtime_s = 0.0f64;
    let mut rust_checksum = 0i64;
    for _ in 0..samples {
        let rust_runtime_start = Instant::now();
        for _ in 0..iters {
            rust_checksum = rust_checksum.wrapping_add(rust_kernel(n, seed));
        }
        rust_runtime_s += rust_runtime_start.elapsed().as_secs_f64();
    }
    rust_runtime_s /= samples as f64;

    // Rust compile baseline (single rustc -O for equivalent source)
    let rust_compile_s = rustc_compile_baseline(n);

    println!(
        "Jules compile(avg {samples}): {:.6}s total ({:.6}s/iter)",
        jules_compile_s,
        jules_compile_s / iters as f64
    );
    println!(
        "Jules runtime(avg {samples}): {:.6}s total ({:.6}s/iter) checksum={}",
        jules_runtime_s,
        jules_runtime_s / iters as f64,
        jules_checksum
    );
    println!(
        "Rust runtime(avg {samples}): {:.6}s total ({:.6}s/iter) checksum={}",
        rust_runtime_s,
        rust_runtime_s / iters as f64,
        rust_checksum
    );
    if let Some(rc) = rust_compile_s {
        println!("Rust compile:   {:.6}s total (rustc -O)", rc);
        println!(
            "Compile ratio (Jules/Rust): {:.2}x",
            jules_compile_s / rc.max(1e-9)
        );
    } else {
        println!("Rust compile:   skipped (rustc unavailable)");
    }
    println!(
        "Runtime ratio (Jules interp / Rust native): {:.2}x",
        jules_runtime_s / rust_runtime_s.max(1e-9)
    );
}

fn parse_mode(raw: Option<&str>) -> BenchMode {
    match raw {
        Some("aot") | Some("aot-time") => BenchMode::AotTime,
        _ => BenchMode::Full,
    }
}

fn rust_kernel(n: usize, seed: i64) -> i64 {
    let n = black_box(n as i32);
    let mut s: i32 = black_box(seed as i32);
    for i in 0..n {
        s = black_box(
            s.wrapping_mul(1_664_525)
                .wrapping_add(i.wrapping_mul(1_013_904_223))
                .wrapping_add(97)
                % 2_147_483_647,
        );
    }
    black_box(s as i64)
}

fn rustc_compile_baseline(n: usize) -> Option<f64> {
    let mut src_path = PathBuf::from(std::env::temp_dir());
    src_path.push(format!("jules_bench_{n}.rs"));
    let mut bin_path = PathBuf::from(std::env::temp_dir());
    bin_path.push(format!("jules_bench_{n}.bin"));

    let src = format!(
        r#"
#[inline(never)]
fn kernel(n: usize, seed: i64) -> i64 {{
    let mut s: i32 = std::hint::black_box(seed as i32);
    for i in 0..n as i32 {{
        s = std::hint::black_box(
            s.wrapping_mul(1_664_525)
                .wrapping_add(i.wrapping_mul(1_013_904_223))
                .wrapping_add(97)
                % 2_147_483_647
        );
    }}
    std::hint::black_box(s as i64)
}}
fn main() {{
    let seed = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    println!("{{}}", kernel({n}, seed));
}}
"#
    );
    if fs::write(&src_path, src).is_err() {
        return None;
    }

    let start = Instant::now();
    let status = Command::new("rustc")
        .arg("-O")
        .arg(&src_path)
        .arg("-o")
        .arg(&bin_path)
        .status()
        .ok()?;
    let elapsed = start.elapsed().as_secs_f64();

    let _ = fs::remove_file(&src_path);
    let _ = fs::remove_file(&bin_path);

    if status.success() {
        Some(elapsed)
    } else {
        None
    }
}

fn default_seed() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(1)
}
