//! Phase 4 LLVM integration hooks.
//!
//! This crate currently keeps LLVM optional behind a feature flag; this module
//! defines a small adapter surface so feature-enabled builds have a concrete
//! target to wire into.

#[derive(Clone, Debug, Default)]
pub struct LlvmPlan {
    pub function_count: usize,
    pub uses_simd: bool,
}

impl LlvmPlan {
    #[inline]
    pub fn estimated_speedup(&self) -> f32 {
        let base = 1.0 + (self.function_count as f32 * 0.01);
        if self.uses_simd {
            base * 1.15
        } else {
            base
        }
    }
}
