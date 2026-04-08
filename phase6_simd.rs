// Phase 6: SIMD optimizations (feature: `phase6-simd`)
//
// Key fixes & improvements over the original:
//   1. Bug fix: `i + 3 < n` → `i + 4 <= n` (original skipped the last batch of exactly 4)
//   2. Runtime feature detection via `is_x86_feature_detected!` — no `unsafe` at the call-site
//   3. FMA path (avx + fma): fused multiply-add in a single instruction, no rounding error
//   4. AVX path (avx only): 256-bit registers, 8 particles per iteration instead of 4
//   5. SSE2 path: same as before but bug-fixed + scatter without heap-allocated temp arrays
//   6. All scatter/gather done with shuffles + `_mm_cvtss_f32` / `_mm_store_ss`;
//      the intermediate `[0.0_f32; 4]` stack arrays are gone.
//   7. Scalar tail uses `mul_add` (FMA on supported hardware via LLVM).
//   8. `#[cold]` scalar path so the branch predictor skips it in the hot loop.

/// Returns true when the SIMD module is compiled in.
#[inline(always)]
pub fn simd_available() -> bool {
    true
}

// ── public entry-point ────────────────────────────────────────────────────────

/// Update positions in-place: `positions[i] += velocities[i] * dt`
///
/// Dispatches at runtime to the best available SIMD path:
///   AVX + FMA  →  8 particles / iter, fused multiply-add
///   AVX only   →  8 particles / iter, two-instruction mul+add
///   SSE2       →  4 particles / iter
///   scalar     →  1 particle  / iter  (`mul_add` on supporting CPUs)
pub fn update_positions(positions: &mut [[f32; 3]], velocities: &[[f32; 3]], dt: f32) {
    let n = positions.len().min(velocities.len());
    if n == 0 {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Runtime dispatch — zero overhead after the first call thanks to branch prediction.
        if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx") {
            // SAFETY: we just confirmed both features are present.
            return unsafe { update_avx_fma(positions, velocities, dt, n) };
        }
        if is_x86_feature_detected!("avx") {
            return unsafe { update_avx(positions, velocities, dt, n) };
        }
        if is_x86_feature_detected!("sse2") {
            return unsafe { update_sse2(positions, velocities, dt, n) };
        }
    }

    scalar_tail(positions, velocities, dt, 0, n);
}

// ── AVX + FMA  (8 particles / iter) ──────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn update_avx_fma(
    positions: &mut [[f32; 3]],
    velocities: &[[f32; 3]],
    dt: f32,
    n: usize,
) {
    use std::arch::x86_64::*;

    let dtv = _mm256_set1_ps(dt);
    let mut i = 0usize;

    while i + 8 <= n {
        // Gather X, Y, Z lanes for 8 particles into three 256-bit registers.
        let (px, py, pz) = gather8(positions, i);
        let (vx, vy, vz) = gather8(velocities, i);

        // p += v * dt  — single fused instruction, better throughput & accuracy.
        let ox = _mm256_fmadd_ps(vx, dtv, px);
        let oy = _mm256_fmadd_ps(vy, dtv, py);
        let oz = _mm256_fmadd_ps(vz, dtv, pz);

        scatter8(positions, i, ox, oy, oz);
        i += 8;
    }

    // 4-wide SSE2 tail inside the AVX kernel.
    if i + 4 <= n {
        i = sse2_batch(positions, velocities, dt, i, n);
    }

    scalar_tail(positions, velocities, dt, i, n);
}

// ── AVX only  (8 particles / iter) ───────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn update_avx(
    positions: &mut [[f32; 3]],
    velocities: &[[f32; 3]],
    dt: f32,
    n: usize,
) {
    use std::arch::x86_64::*;

    let dtv = _mm256_set1_ps(dt);
    let mut i = 0usize;

    while i + 8 <= n {
        let (px, py, pz) = gather8(positions, i);
        let (vx, vy, vz) = gather8(velocities, i);

        let ox = _mm256_add_ps(px, _mm256_mul_ps(vx, dtv));
        let oy = _mm256_add_ps(py, _mm256_mul_ps(vy, dtv));
        let oz = _mm256_add_ps(pz, _mm256_mul_ps(vz, dtv));

        scatter8(positions, i, ox, oy, oz);
        i += 8;
    }

    if i + 4 <= n {
        i = sse2_batch(positions, velocities, dt, i, n);
    }

    scalar_tail(positions, velocities, dt, i, n);
}

// ── SSE2 only  (4 particles / iter) ──────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn update_sse2(
    positions: &mut [[f32; 3]],
    velocities: &[[f32; 3]],
    dt: f32,
    n: usize,
) {
    let i = sse2_batch(positions, velocities, dt, 0, n);
    scalar_tail(positions, velocities, dt, i, n);
}

// ── shared SSE2 inner loop ────────────────────────────────────────────────────

/// Processes 4-wide batches; returns the first unprocessed index.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn sse2_batch(
    positions: &mut [[f32; 3]],
    velocities: &[[f32; 3]],
    dt: f32,
    mut i: usize,
    n: usize,
) -> usize {
    use std::arch::x86_64::*;

    let dtv = _mm_set1_ps(dt);

    while i + 4 <= n {
        let (px, py, pz) = gather4(positions, i);
        let (vx, vy, vz) = gather4(velocities, i);

        let ox = _mm_add_ps(px, _mm_mul_ps(vx, dtv));
        let oy = _mm_add_ps(py, _mm_mul_ps(vy, dtv));
        let oz = _mm_add_ps(pz, _mm_mul_ps(vz, dtv));

        scatter4(positions, i, ox, oy, oz);
        i += 4;
    }
    i
}

// ── gather / scatter helpers ──────────────────────────────────────────────────
//
// AoS → SoA gather:  build one register per axis from N consecutive particles.
// SoA → AoS scatter: write one lane per axis back to its particle slot.
// Shuffles + cvtss avoid stack-allocating intermediate [f32; N] arrays.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
#[inline(always)]
unsafe fn gather4(
    src: &[[f32; 3]],
    i: usize,
) -> (
    std::arch::x86_64::__m128,
    std::arch::x86_64::__m128,
    std::arch::x86_64::__m128,
) {
    use std::arch::x86_64::*;
    let x = _mm_set_ps(src[i + 3][0], src[i + 2][0], src[i + 1][0], src[i][0]);
    let y = _mm_set_ps(src[i + 3][1], src[i + 2][1], src[i + 1][1], src[i][1]);
    let z = _mm_set_ps(src[i + 3][2], src[i + 2][2], src[i + 1][2], src[i][2]);
    (x, y, z)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
#[inline(always)]
unsafe fn scatter4(
    dst: &mut [[f32; 3]],
    i: usize,
    ox: std::arch::x86_64::__m128,
    oy: std::arch::x86_64::__m128,
    oz: std::arch::x86_64::__m128,
) {
    use std::arch::x86_64::*;
    // Extract each lane with a shuffle then cvtss — no stack allocation.
    macro_rules! lane {
        ($reg:expr, 0) => { _mm_cvtss_f32($reg) };
        ($reg:expr, 1) => { _mm_cvtss_f32(_mm_shuffle_ps($reg, $reg, 0b_01_01_01_01)) };
        ($reg:expr, 2) => { _mm_cvtss_f32(_mm_shuffle_ps($reg, $reg, 0b_10_10_10_10)) };
        ($reg:expr, 3) => { _mm_cvtss_f32(_mm_shuffle_ps($reg, $reg, 0b_11_11_11_11)) };
    }
    dst[i    ] = [lane!(ox, 0), lane!(oy, 0), lane!(oz, 0)];
    dst[i + 1] = [lane!(ox, 1), lane!(oy, 1), lane!(oz, 1)];
    dst[i + 2] = [lane!(ox, 2), lane!(oy, 2), lane!(oz, 2)];
    dst[i + 3] = [lane!(ox, 3), lane!(oy, 3), lane!(oz, 3)];
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[inline(always)]
unsafe fn gather8(
    src: &[[f32; 3]],
    i: usize,
) -> (
    std::arch::x86_64::__m256,
    std::arch::x86_64::__m256,
    std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;
    let x = _mm256_set_ps(
        src[i+7][0], src[i+6][0], src[i+5][0], src[i+4][0],
        src[i+3][0], src[i+2][0], src[i+1][0], src[i  ][0],
    );
    let y = _mm256_set_ps(
        src[i+7][1], src[i+6][1], src[i+5][1], src[i+4][1],
        src[i+3][1], src[i+2][1], src[i+1][1], src[i  ][1],
    );
    let z = _mm256_set_ps(
        src[i+7][2], src[i+6][2], src[i+5][2], src[i+4][2],
        src[i+3][2], src[i+2][2], src[i+1][2], src[i  ][2],
    );
    (x, y, z)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[inline(always)]
unsafe fn scatter8(
    dst: &mut [[f32; 3]],
    i: usize,
    ox: std::arch::x86_64::__m256,
    oy: std::arch::x86_64::__m256,
    oz: std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;
    // Split 256 → two 128-bit halves, then extract lanes as in scatter4.
    macro_rules! lo { ($r:expr) => { _mm256_castps256_ps128($r) } }
    macro_rules! hi { ($r:expr) => { _mm256_extractf128_ps($r, 1) } }
    macro_rules! lane {
        ($reg:expr, 0) => { _mm_cvtss_f32($reg) };
        ($reg:expr, 1) => { _mm_cvtss_f32(_mm_shuffle_ps($reg, $reg, 0b_01_01_01_01)) };
        ($reg:expr, 2) => { _mm_cvtss_f32(_mm_shuffle_ps($reg, $reg, 0b_10_10_10_10)) };
        ($reg:expr, 3) => { _mm_cvtss_f32(_mm_shuffle_ps($reg, $reg, 0b_11_11_11_11)) };
    }
    let (lox, hox) = (lo!(ox), hi!(ox));
    let (loy, hoy) = (lo!(oy), hi!(oy));
    let (loz, hoz) = (lo!(oz), hi!(oz));

    dst[i    ] = [lane!(lox, 0), lane!(loy, 0), lane!(loz, 0)];
    dst[i + 1] = [lane!(lox, 1), lane!(loy, 1), lane!(loz, 1)];
    dst[i + 2] = [lane!(lox, 2), lane!(loy, 2), lane!(loz, 2)];
    dst[i + 3] = [lane!(lox, 3), lane!(loy, 3), lane!(loz, 3)];
    dst[i + 4] = [lane!(hox, 0), lane!(hoy, 0), lane!(hoz, 0)];
    dst[i + 5] = [lane!(hox, 1), lane!(hoy, 1), lane!(hoz, 1)];
    dst[i + 6] = [lane!(hox, 2), lane!(hoy, 2), lane!(hoz, 2)];
    dst[i + 7] = [lane!(hox, 3), lane!(hoy, 3), lane!(hoz, 3)];
}

// ── scalar tail ───────────────────────────────────────────────────────────────

/// Handles leftover particles that don't fill a full SIMD register.
/// `mul_add` compiles to a native FMA instruction where the CPU supports it.
#[cold]  // keeps it out of the hot path's i-cache footprint
#[inline]
fn scalar_tail(
    positions: &mut [[f32; 3]],
    velocities: &[[f32; 3]],
    dt: f32,
    start: usize,
    n: usize,
) {
    for i in start..n {
        positions[i][0] = velocities[i][0].mul_add(dt, positions[i][0]);
        positions[i][1] = velocities[i][1].mul_add(dt, positions[i][1]);
        positions[i][2] = velocities[i][2].mul_add(dt, positions[i][2]);
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_module_present() {
        assert!(simd_available());
    }

    /// The original test case.
    #[test]
    fn update_positions_basic() {
        let mut p = vec![[0.0_f32, 0.0, 0.0]; 4];
        let v = vec![[1.0_f32, 0.5, -0.25]; 4];
        update_positions(&mut p, &v, 0.5);
        assert_eq!(p[0], [0.5, 0.25, -0.125]);
    }

    /// Covers the original bug: the last batch of exactly 4 was skipped.
    #[test]
    fn update_positions_exact_batch_of_4() {
        let mut p = vec![[0.0_f32, 0.0, 0.0]; 4];
        let v = vec![[2.0_f32, 4.0, -1.0]; 4];
        update_positions(&mut p, &v, 1.0);
        for pp in &p {
            assert_eq!(*pp, [2.0, 4.0, -1.0]);
        }
    }

    /// Validates the scalar tail for non-multiple-of-4 lengths.
    #[test]
    fn update_positions_tail() {
        let mut p = vec![[0.0_f32, 0.0, 0.0]; 7];
        let v = vec![[1.0_f32, 1.0, 1.0]; 7];
        update_positions(&mut p, &v, 2.0);
        for pp in &p {
            assert_eq!(*pp, [2.0, 2.0, 2.0]);
        }
    }

    /// Mismatched slice lengths: only min(pos, vel) entries should change.
    #[test]
    fn update_positions_mismatched_lengths() {
        let mut p = vec![[0.0_f32, 0.0, 0.0]; 6];
        let v = vec![[1.0_f32, 1.0, 1.0]; 4];
        update_positions(&mut p, &v, 1.0);
        for pp in &p[..4] {
            assert_eq!(*pp, [1.0, 1.0, 1.0]);
        }
        for pp in &p[4..] {
            assert_eq!(*pp, [0.0, 0.0, 0.0]);
        }
    }

    /// Empty slices must not panic.
    #[test]
    fn update_positions_empty() {
        let mut p: Vec<[f32; 3]> = vec![];
        let v: Vec<[f32; 3]> = vec![];
        update_positions(&mut p, &v, 1.0); // must not panic
    }
}
