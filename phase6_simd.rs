// Phase 6: SIMD optimizations (feature: `phase6-simd`)
// This scaffold hosts SIMD-focused micro-optimizations for numerically
// intensive game loops. Implementations must follow the Jules performance protocol.

/// Returns true when the SIMD module is compiled in.
pub fn simd_available() -> bool {
    true
}

/// Update positions in-place: positions[i] += velocities[i] * dt
/// This safe, portable implementation is intentionally simple. It provides
/// a single call-site that can later be replaced by an architecture-specific
/// vectorized implementation (Cranelift/LLVM/generic `std::simd`) behind the
/// same API, keeping the callsite stable for benchmarking.
pub fn update_positions(positions: &mut [[f32; 3]], velocities: &[[f32; 3]], dt: f32) {
    let n = positions.len().min(velocities.len());
    let mut i = 0usize;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        use std::arch::x86_64::*;
        let dtv = _mm_set1_ps(dt);
        while i + 3 < n {
            let px = _mm_set_ps(
                positions[i + 3][0],
                positions[i + 2][0],
                positions[i + 1][0],
                positions[i][0],
            );
            let py = _mm_set_ps(
                positions[i + 3][1],
                positions[i + 2][1],
                positions[i + 1][1],
                positions[i][1],
            );
            let pz = _mm_set_ps(
                positions[i + 3][2],
                positions[i + 2][2],
                positions[i + 1][2],
                positions[i][2],
            );
            let vx = _mm_set_ps(
                velocities[i + 3][0],
                velocities[i + 2][0],
                velocities[i + 1][0],
                velocities[i][0],
            );
            let vy = _mm_set_ps(
                velocities[i + 3][1],
                velocities[i + 2][1],
                velocities[i + 1][1],
                velocities[i][1],
            );
            let vz = _mm_set_ps(
                velocities[i + 3][2],
                velocities[i + 2][2],
                velocities[i + 1][2],
                velocities[i][2],
            );

            let ox = _mm_add_ps(px, _mm_mul_ps(vx, dtv));
            let oy = _mm_add_ps(py, _mm_mul_ps(vy, dtv));
            let oz = _mm_add_ps(pz, _mm_mul_ps(vz, dtv));

            let mut out_x = [0.0_f32; 4];
            let mut out_y = [0.0_f32; 4];
            let mut out_z = [0.0_f32; 4];
            _mm_storeu_ps(out_x.as_mut_ptr(), ox);
            _mm_storeu_ps(out_y.as_mut_ptr(), oy);
            _mm_storeu_ps(out_z.as_mut_ptr(), oz);

            positions[i][0] = out_x[0];
            positions[i + 1][0] = out_x[1];
            positions[i + 2][0] = out_x[2];
            positions[i + 3][0] = out_x[3];
            positions[i][1] = out_y[0];
            positions[i + 1][1] = out_y[1];
            positions[i + 2][1] = out_y[2];
            positions[i + 3][1] = out_y[3];
            positions[i][2] = out_z[0];
            positions[i + 1][2] = out_z[1];
            positions[i + 2][2] = out_z[2];
            positions[i + 3][2] = out_z[3];
            i += 4;
        }
    }

    while i < n {
        positions[i][0] = velocities[i][0].mul_add(dt, positions[i][0]);
        positions[i][1] = velocities[i][1].mul_add(dt, positions[i][1]);
        positions[i][2] = velocities[i][2].mul_add(dt, positions[i][2]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_module_present() {
        assert!(simd_available());
    }

    #[test]
    fn update_positions_basic() {
        let mut p = vec![[0.0_f32, 0.0, 0.0]; 4];
        let v = vec![[1.0_f32, 0.5, -0.25]; 4];
        update_positions(&mut p, &v, 0.5);
        assert_eq!(p[0], [0.5, 0.25, -0.125]);
    }
}
