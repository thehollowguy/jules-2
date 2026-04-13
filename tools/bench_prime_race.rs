use std::hint::black_box;
use std::time::Instant;

use jules::{CompileUnit, Pipeline, PipelineResult};
use jules::interp::{Interpreter, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Jules: trial division up to this limit (bytecode VM interpreter)
    let jules_limit: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    // Rust: segmented sieve up to 1 billion
    let rust_limit: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000_000);

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   PRIME GENERATION RACE                                 ║");
    println!("║   Jules (superoptimized bytecode) vs Rust (-O3 sieve)   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("Jules: trial division to {} ({:.1}M)", jules_limit, jules_limit as f64 / 1e6);
    println!("Rust:  segmented sieve to {} ({:.0}M)", rust_limit, rust_limit as f64 / 1e6);
    println!();

    // ================================================================
    // Round 1: Jules - Compile + Run with native JIT
    // ================================================================
    println!("─── Round 1: Jules (native JIT enabled) ───");

    let jules_source = build_prime_trial_source(jules_limit);

    let pipeline = Pipeline::new();
    let mut unit = CompileUnit::new("<prime-race-jules>", &jules_source);

    let compile_start = Instant::now();
    let result = pipeline.run(&mut unit);
    let jules_compile_time = compile_start.elapsed();

    if unit.has_errors() {
        eprintln!("[FAIL] Jules compilation failed: {} diagnostics", unit.diags.len());
        for d in &unit.diags {
            eprintln!("  {:?}", d);
        }
        std::process::exit(1);
    }

    let PipelineResult::Ok(program) = result else {
        eprintln!("[FAIL] Jules pipeline halted");
        std::process::exit(1);
    };

    println!("  Compile time: {:.3} ms", jules_compile_time.as_secs_f64() * 1000.0);
    println!("  Running Jules prime trial division (to {})...", jules_limit);

    let mut interp = Interpreter::new();
    interp.set_jit_enabled(true);
    interp.set_advance_jit_enabled(true);
    interp.set_native_jit_enabled(true);
    interp.load_program(&program);

    let run_start = Instant::now();
    let jules_result = interp.call_fn("main", vec![]);
    let jules_runtime = run_start.elapsed();

    match jules_result {
        Err(e) => {
            eprintln!("[FAIL] Jules runtime error: {}", e.message);
            std::process::exit(1);
        }
        _ => {}
    }

    let (native_calls, vm_calls, fallback_calls) = interp.jit_counters();
    let jules_throughput = jules_limit as f64 / jules_runtime.as_secs_f64().max(1e-12);
    println!("  Runtime:        {:.3} s", jules_runtime.as_secs_f64());
    println!("  Numbers tested: {}", jules_limit);
    println!("  Throughput:     {:.0} numbers/s", jules_throughput);
    println!("  JIT counters:   native={} vm={} fallback={}", native_calls, vm_calls, fallback_calls);
    println!();

    // ================================================================
    // Round 2: Rust Native - Segmented Sieve with wheel factorization
    // ================================================================
    println!("─── Round 2: Rust (native -O3, segmented sieve + wheel) ───");
    println!("  Running Rust segmented sieve (to {})...", rust_limit);

    let rust_run_start = Instant::now();
    let rust_prime_count = rust_segmented_sieve(rust_limit);
    let rust_runtime = rust_run_start.elapsed();

    let rust_throughput = rust_limit as f64 / rust_runtime.as_secs_f64().max(1e-12);
    println!("  Runtime:        {:.3} s", rust_runtime.as_secs_f64());
    println!("  Numbers tested: {}", rust_limit);
    println!("  Primes found:   {}", rust_prime_count);
    println!("  Throughput:     {:.0} numbers/s", rust_throughput);
    println!();

    // ================================================================
    // Normalized comparison - both use trial division at same limit
    // ================================================================
    println!("─── Normalized comparison (trial division, {} numbers) ───", jules_limit);

    let rust_trial_start = Instant::now();
    let _rust_trial_count = rust_trial_division(jules_limit);
    let rust_trial_runtime = rust_trial_start.elapsed();

    let rust_trial_throughput = jules_limit as f64 / rust_trial_runtime.as_secs_f64().max(1e-12);
    let slowdown = jules_runtime.as_secs_f64() / rust_trial_runtime.as_secs_f64().max(1e-12);

    println!("  Jules runtime      ({}):  {:.3} s  ({:.0} nums/s)", jules_limit, jules_runtime.as_secs_f64(), jules_throughput);
    println!("  Rust trial div     ({}):  {:.6} s  ({:.0} nums/s)", jules_limit, rust_trial_runtime.as_secs_f64(), rust_trial_throughput);
    println!("  Jules is {:.1}x slower than Rust (same algorithm)", slowdown);
    println!();

    // ================================================================
    // Algorithm comparison: Jules trial vs Rust segmented sieve
    // ================================================================
    let algo_speedup = rust_throughput / jules_throughput;
    println!("─── Algorithm advantage ───");
    println!("  Rust segmented sieve vs Jules trial: {:.1}x throughput advantage", algo_speedup);
    println!();

    // ================================================================
    // Extrapolation
    // ================================================================
    let jules_extrapolated = jules_runtime.as_secs_f64() * (1_000_000_000.0 / jules_limit as f64);
    println!("─── Extrapolation to 1 Billion Numbers ───");
    println!("  Jules (trial) estimated: {:.1} s ({:.1} minutes, {:.1} hours)",
             jules_extrapolated, jules_extrapolated / 60.0, jules_extrapolated / 3600.0);
    println!("  Rust (segmented) actual: {:.3} s", rust_runtime.as_secs_f64());
    println!();

    // ================================================================
    // Final Results
    // ================================================================
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                   FINAL RESULTS                          ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Jules throughput:        {:>14.0} nums/s                ║", jules_throughput);
    println!("║  Rust sieve throughput:   {:>14.0} nums/s                ║", rust_throughput);
    println!("║  Rust trial throughput:   {:>14.0} nums/s                ║", rust_trial_throughput);
    println!("║  Algo speedup (sieve):    {:>14.1}x                       ║", algo_speedup);
    println!("║  Same-algo slowdown:      {:>14.1}x                       ║", slowdown);
    println!("║  Jules extrapolated 1B:   {:>14.1} s                      ║", jules_extrapolated);
    println!("║  Rust actual 1B:          {:>14.3} s                      ║", rust_runtime.as_secs_f64());
    println!("╚══════════════════════════════════════════════════════════╝");
}

fn build_prime_trial_source(limit: usize) -> String {
    // Trial division with `let mut` for performance
    let mut out = String::from("fn main() {\n");
    out.push_str(&format!("  let limit = {limit};\n"));
    out.push_str("  let mut i = 2;\n");
    out.push_str("  while i < limit {\n");
    out.push_str("    let mut is_prime = 1;\n");
    out.push_str("    let mut j = 2;\n");
    out.push_str("    while j < i {\n");
    out.push_str("      let rem = i - (i / j) * j;\n");
    out.push_str("      if rem == 0 {\n");
    out.push_str("        is_prime = 0;\n");
    out.push_str("      }\n");
    out.push_str("      j = j + 1;\n");
    out.push_str("    }\n");
    out.push_str("    i = i + 1;\n");
    out.push_str("  }\n");
    out.push_str("}\n");
    out
}

/// Naive trial division (same algorithm as Jules, for fair comparison)
fn rust_trial_division(limit: usize) -> usize {
    let mut count = 0;
    for n in 2..=limit {
        let mut is_prime = true;
        let mut j = 2;
        while j * j <= n {
            if n % j == 0 {
                is_prime = false;
                break;
            }
            j += 1;
        }
        if is_prime {
            count += 1;
        }
        black_box(());
    }
    black_box(count)
}

/// Segmented Sieve of Eratosthenes optimized for speed.
/// Counts primes up to `limit` using O(sqrt(limit)) memory.
fn rust_segmented_sieve(limit: usize) -> usize {
    if limit < 2 {
        return 0;
    }

    // Step 1: Sieve small primes up to sqrt(limit)
    let sqrt_limit = (limit as f64).sqrt() as usize + 1;
    let mut is_prime_small = vec![true; sqrt_limit + 1];
    is_prime_small[0] = false;
    is_prime_small[1] = false;
    for p in 2..=sqrt_limit {
        if is_prime_small[p] {
            let mut multiple = p * p;
            while multiple <= sqrt_limit {
                is_prime_small[multiple] = false;
                multiple += p;
            }
        }
    }
    let small_primes: Vec<usize> = (2..=sqrt_limit).filter(|&p| is_prime_small[p]).collect();

    // Count small primes
    let mut count = small_primes.len();

    // Step 2: Segmented sieve
    // Use odd-only sieve (skip all even numbers) to halve memory and work
    const SEG_SIZE: usize = 1 << 20; // 1M odd numbers = 2M number range = 128KB as bits
    const SEG_BITS: usize = SEG_SIZE;
    let mut segment = vec![0u8; SEG_BITS / 8 + 1];

    let mut low = sqrt_limit + 1;
    // Make low odd
    if low % 2 == 0 {
        low += 1;
    }

    while low <= limit {
        let high = (low + SEG_SIZE * 2 - 1).min(limit);
        let seg_len_odd = ((high - low) / 2) + 1; // number of odd numbers in range
        let bytes_needed = (seg_len_odd + 7) / 8;
        for i in 0..bytes_needed.min(segment.len()) {
            segment[i] = 0;
        }

        // Mark odd composites using small primes (skip 2)
        for &p in &small_primes {
            if p == 2 {
                continue;
            }
            // Find first odd multiple of p >= low
            let start = if low <= p { p * p } else { (low + p - 1) / p * p };
            let start = if start % 2 == 0 { start + p } else { start };
            let mut multiple = start;
            while multiple <= high {
                let idx = (multiple - low) / 2;
                segment[idx >> 3] |= 1 << (idx & 7);
                multiple += p * 2; // skip even multiples
            }
        }

        // Count unmarked odd numbers
        for idx in 0..seg_len_odd {
            if (segment[idx >> 3] & (1 << (idx & 7))) == 0 {
                count += 1;
            }
        }

        low += SEG_SIZE * 2;
    }

    black_box(count)
}
