//! Phase 3 "JIT-like" fast paths.
//!
//! This module avoids dynamic dispatch and branchy per-action scoring by
//! selecting a specialized scoring kernel once, then reusing it in hot loops.

#[derive(Clone, Copy, Debug)]
pub enum Side {
    White,
    Black,
}

type ScoreKernel = fn(base: f32, lane: usize) -> f32;

#[derive(Clone, Copy)]
pub struct JitPolicy4 {
    kernel: ScoreKernel,
}

impl JitPolicy4 {
    #[inline]
    pub fn for_side(side: Side) -> Self {
        let kernel = match side {
            Side::White => score_white,
            Side::Black => score_black,
        };
        Self { kernel }
    }

    #[inline]
    pub fn best_action(&self, base: f32, noise4: [f32; 4]) -> u8 {
        let mut best_idx = 0usize;
        let mut best = f32::MIN;
        let mut lane = 0usize;
        while lane < 4 {
            let s = (self.kernel)(base, lane) + noise4[lane];
            if s > best {
                best = s;
                best_idx = lane;
            }
            lane += 1;
        }
        best_idx as u8
    }
}

#[inline(always)]
fn score_white(base: f32, lane: usize) -> f32 {
    const TACTICAL: [f32; 4] = [0.00, 0.02, 0.06, 0.10];
    base + TACTICAL[lane]
}

#[inline(always)]
fn score_black(base: f32, lane: usize) -> f32 {
    const TACTICAL: [f32; 4] = [0.10, 0.06, 0.02, 0.00];
    base + TACTICAL[lane]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_policy_prefers_high_lane_without_noise() {
        let p = JitPolicy4::for_side(Side::White);
        assert_eq!(p.best_action(0.0, [0.0; 4]), 3);
    }

    #[test]
    fn black_policy_prefers_low_lane_without_noise() {
        let p = JitPolicy4::for_side(Side::Black);
        assert_eq!(p.best_action(0.0, [0.0; 4]), 0);
    }
}
