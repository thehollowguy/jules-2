use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use jules::{CompileUnit, Pipeline, PipelineResult};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);

    println!("bench-interp-vs-rust n={n} iters={iters}");

    let jules_src = format!(
        r#"
fn main() {{
  let mut s = 0;
  for i in 0..{n} {{
    s = s + ((i * 1664525 + 1013904223) % 97);
  }}
}}
"#
    );

    // Jules compile benchmark
    let pipeline = Pipeline::new();
    let compile_start = Instant::now();
    let mut program = None;
    for i in 0..iters {
        let mut unit = CompileUnit::new(format!("<bench:{i}>"), &jules_src);
        let result = pipeline.run(&mut unit);
        if unit.has_errors() {
            eprintln!("compile diagnostics: {}", unit.diags.len());
            std::process::exit(1);
        }
        if i + 1 == iters {
            match result {
                PipelineResult::Ok(p) => program = Some(p),
                _ => {
                    eprintln!("pipeline failed to produce executable program");
                    std::process::exit(1);
                }
            }
        }
    }
    let jules_compile_s = compile_start.elapsed().as_secs_f64();

    // Jules runtime benchmark (interpreter)
    let mut interp = jules::interp::Interpreter::new();
    interp.load_program(&program.expect("program should exist"));
    let run_start = Instant::now();
    for _ in 0..iters {
        if let Err(e) = interp.call_fn("main", vec![]) {
            eprintln!("runtime error: {}", e.message);
            std::process::exit(1);
        }
    }
    let jules_runtime_s = run_start.elapsed().as_secs_f64();

    // Rust runtime baseline
    let rust_runtime_start = Instant::now();
    let mut rust_checksum = 0i64;
    for _ in 0..iters {
        rust_checksum ^= rust_kernel(n);
    }
    let rust_runtime_s = rust_runtime_start.elapsed().as_secs_f64();

    // Rust compile baseline (single rustc -O for equivalent source)
    let rust_compile_s = rustc_compile_baseline(n);

    println!(
        "Jules compile:  {:.6}s total ({:.6}s/iter)",
        jules_compile_s,
        jules_compile_s / iters as f64
    );
    println!(
        "Jules runtime:  {:.6}s total ({:.6}s/iter)",
        jules_runtime_s,
        jules_runtime_s / iters as f64
    );
    println!(
        "Rust runtime:   {:.6}s total ({:.6}s/iter) checksum={}",
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

fn rust_kernel(n: usize) -> i64 {
    let mut s: i64 = 0;
    for i in 0..n as i64 {
        s += (i * 1_664_525 + 1_013_904_223) % 97;
    }
    s
}

fn rustc_compile_baseline(n: usize) -> Option<f64> {
    let mut src_path = PathBuf::from(std::env::temp_dir());
    src_path.push(format!("jules_bench_{n}.rs"));
    let mut bin_path = PathBuf::from(std::env::temp_dir());
    bin_path.push(format!("jules_bench_{n}.bin"));

    let src = format!(
        r#"
#[inline(never)]
fn kernel(n: usize) -> i64 {{
    let mut s: i64 = 0;
    for i in 0..n as i64 {{
        s += (i * 1_664_525 + 1_013_904_223) % 97;
    }}
    s
}}
fn main() {{ let _ = kernel({n}); }}
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
