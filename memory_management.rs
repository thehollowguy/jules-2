#![allow(dead_code)]

//! Memory management helpers for keeping the bytecode VM fed with data.
//! This module provides a light prefetch controller so hot loops can hint
//! upcoming instruction and slot accesses to the CPU cache hierarchy.

#[derive(Debug, Clone)]
pub struct PrefetchController {
    /// Number of instructions to prefetch ahead of current PC.
    distance: usize,
    /// Upper bound to avoid prefetching too far ahead.
    max_distance: usize,
}

impl Default for PrefetchController {
    fn default() -> Self {
        Self::new(8)
    }
}

impl PrefetchController {
    pub fn new(distance: usize) -> Self {
        Self {
            distance: distance.max(1),
            max_distance: 64,
        }
    }

    #[inline(always)]
    pub fn update_distance(&mut self, recent_branch_density: u8) {
        // Fewer branches → larger windows are useful; many branches → stay tighter.
        self.distance = if recent_branch_density <= 2 {
            12
        } else if recent_branch_density <= 5 {
            8
        } else {
            4
        }
        .min(self.max_distance);
    }

    #[inline(always)]
    pub fn prefetch_instruction<T>(&self, base_ptr: *const T, pc: usize, len: usize) {
        let next_pc = pc.saturating_add(self.distance);
        if next_pc < len {
            let ptr = unsafe { base_ptr.add(next_pc) };
            prefetch_read(ptr.cast::<u8>());
        }
    }

    #[inline(always)]
    pub fn prefetch_slot<T>(&self, slot_ptr: *const T, slot: usize, len: usize) {
        if slot < len {
            let ptr = unsafe { slot_ptr.add(slot) };
            prefetch_read(ptr.cast::<u8>());
        }
    }
}

#[inline(always)]
fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::asm;
        // Best-effort hint only; no correctness dependency.
        asm!("prfm pldl1keep, [{addr}]", addr = in(reg) ptr, options(nostack, readonly));
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}
