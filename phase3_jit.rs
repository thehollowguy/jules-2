//! Phase 3 real machine-code JIT backend (x86-64, System V).
//!
//! Optimizations active:
//!
//! A. Full linear-scan register allocation across 10 GPRs
//!    (r8-r11, rsi — caller-saved; r12-r15, rbx — callee-saved).
//!    rax/rcx = scratch accumulators.  rdx = reserved for cqo/idiv.
//!    rdi = slot-array base pointer.  All other GPRs allocated by RA.
//!    Callee-saved registers that are actually used get pushed/popped
//!    in a generated prologue/epilogue.
//!
//! B. Forward constant-propagation + compile-time BinOp folding.
//!    Any slot whose value is statically known is propagated through
//!    Move/Store/BinOp; foldable BinOps emit a single immediate load.
//!    State is conservatively cleared at every branch target.
//!
//! C. 3-instruction superinstruction fusions:
//!    • BinOp(t, Mul, x, N) + BinOp(r, Add, t, y)  →  LEA  (N ∈ {2,4,8})
//!    • BinOp(t, op1, a, b) + BinOp(r, op2, t, c)  →  two-op chain
//!      (eliminates the intermediate slot store + load of `t`)
//!
//! D. 2-instruction superinstruction fusions (all original patterns):
//!    • LoadI*(tmp, c) + JumpFalse/True(tmp, …) → compile-time branch fold
//!    • LoadI*(tmp, c) + BinOp(d, op, x, tmp)   → immediate-form arithmetic
//!    • BinOp(t, op, l, r) + Store(slot, t)      → fused compute-and-store
//!    • BinOp(t, op, l, r) + JumpFalse/True(t,…) → fused cmp+branch
//!
//! E. Optimal immediate encoding:
//!    MOV EAX, imm32 (5 B, zero-extends) when 0 ≤ v < 2³¹
//!    MOV RAX, sign-extended imm32 (7 B) when −2³¹ ≤ v < 0
//!    MOV RAX, imm64 (10 B) only when the value doesn't fit in 32 bits.
//!
//! F. All existing micro-optimisations retained:
//!    LEA for ×3/×5/×9, SHL for powers-of-two multiply,
//!    INC/DEC for ±1, TEST+Jcc, SETCC for branchless comparisons.

use std::cell::RefCell;
use std::collections::HashMap; // retained for rolling const_prop state in codegen
use std::ptr::NonNull;

use libc::{mmap, munmap, MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};

use crate::ast::BinOpKind;
use crate::interp::{CompiledFn, Instr, RuntimeError, Value};

// ─────────────────────────────────────────────────────────────────────────────
// Executable memory
// ─────────────────────────────────────────────────────────────────────────────

pub struct NativeCode {
    pub slot_count: u16,
    mem: ExecMem,
}

struct ExecMem {
    ptr: *mut u8,
    len: usize,
    arena_backed: bool,
}

struct ExecArena {
    base: NonNull<u8>,
    len: usize,
    cursor: usize,
}

impl Drop for ExecArena {
    fn drop(&mut self) {
        unsafe { munmap(self.base.as_ptr().cast(), self.len) };
    }
}

impl ExecArena {
    const DEFAULT_LEN: usize = 16 * 1024 * 1024;

    fn try_new() -> Option<Self> {
        #[cfg(target_os = "linux")]
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                Self::DEFAULT_LEN,
                PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANON | libc::MAP_HUGETLB,
                -1,
                0,
            )
        };
        #[cfg(not(target_os = "linux"))]
        let ptr = std::ptr::null_mut();

        let ptr = if ptr.is_null() || ptr == libc::MAP_FAILED {
            unsafe {
                mmap(
                    std::ptr::null_mut(),
                    Self::DEFAULT_LEN,
                    PROT_READ | PROT_WRITE | PROT_EXEC,
                    MAP_PRIVATE | MAP_ANON,
                    -1,
                    0,
                )
            }
        } else {
            ptr
        };
        if ptr.is_null() || ptr == libc::MAP_FAILED {
            return None;
        }
        Some(Self {
            base: NonNull::new(ptr.cast::<u8>())?,
            len: Self::DEFAULT_LEN,
            cursor: 0,
        })
    }

    fn alloc(&mut self, bytes: usize) -> Option<*mut u8> {
        let aligned = (bytes + 15) & !15;
        let next = self.cursor.checked_add(aligned)?;
        if next > self.len {
            return None;
        }
        let out = unsafe { self.base.as_ptr().add(self.cursor) };
        self.cursor = next;
        Some(out)
    }
}

impl Drop for ExecMem {
    fn drop(&mut self) {
        if !self.arena_backed && !self.ptr.is_null() && self.len > 0 {
            unsafe { munmap(self.ptr.cast(), self.len) };
        }
    }
}

thread_local! {
    static TLS_EXEC_ARENA: RefCell<Option<ExecArena>> = const { RefCell::new(None) };
}

impl ExecMem {
    fn new(code: &[u8]) -> Option<Self> {
        if let Some(mem) = TLS_EXEC_ARENA.with(|arena_cell| {
            let mut arena = arena_cell.borrow_mut();
            if arena.is_none() {
                *arena = ExecArena::try_new();
            }
            let arena = arena.as_mut()?;
            let ptr = arena.alloc(code.len().max(1))?;
            unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len()) };
            Some(Self {
                ptr,
                len: code.len().max(1),
                arena_backed: true,
            })
        }) {
            return Some(mem);
        }

        let len = code.len().max(1);
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if ptr.is_null() || ptr == libc::MAP_FAILED {
            return None;
        }
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), ptr.cast::<u8>(), code.len()) };
        Some(Self {
            ptr: ptr.cast::<u8>(),
            len,
        })
    }
    fn entry(&self) -> unsafe extern "C" fn(*mut i64) -> i64 {
        unsafe { std::mem::transmute(self.ptr) }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Code emitter
// ─────────────────────────────────────────────────────────────────────────────
//
// Register number convention (matches x86-64 encoding):
//   rax=0  rcx=1  rdx=2  rbx=3  rsp=4  rbp=5  rsi=6  rdi=7
//   r8=8   r9=9   r10=10 r11=11 r12=12 r13=13 r14=14 r15=15

struct Emitter {
    buf: Vec<u8>,
}

impl Emitter {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(4096),
        }
    }
    fn pos(&self) -> usize {
        self.buf.len()
    }
    fn b(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn d(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn q(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    // ── Immediate loads ──────────────────────────────────────────────────────

    /// Full 64-bit immediate into rax (10 bytes).
    fn mov_rax_imm64(&mut self, v: i64) {
        self.b(0x48);
        self.b(0xB8);
        self.q(v);
    }

    /// Optimal immediate into rax.
    ///  v ≥ 0 and fits i32 → MOV EAX, imm32 (5 B, zero-extends)
    ///  v <  0 and fits i32 → REX.W MOV RAX, sign-ext imm32 (7 B)
    ///  otherwise           → MOV RAX, imm64 (10 B)
    fn mov_rax_imm_opt(&mut self, v: i64) {
        if let Ok(v32) = i32::try_from(v) {
            if v32 >= 0 {
                self.b(0xB8);
                self.d(v32); // MOV EAX, imm32
            } else {
                self.b(0x48);
                self.b(0xC7);
                self.b(0xC0);
                self.d(v32); // MOV RAX, sx(imm32)
            }
        } else {
            self.mov_rax_imm64(v);
        }
    }

    // ── Generic register ↔ memory ────────────────────────────────────────────
    //
    // REX.W  = bit3 (always 1 for 64-bit)
    // REX.R  = bit2 (extends ModRM.reg  — destination for 8B, source for 89)
    // REX.B  = bit0 (extends ModRM.rm   — source     for 8B, dest   for 89 w/ mod=11)
    //
    // For [rdi + disp32]: ModRM = mod10 | reg_field<<3 | 7

    /// mov reg64, [rdi + disp32]
    fn load_reg_mem(&mut self, reg: u8, disp: i32) {
        self.b(0x48 | ((reg & 8) >> 1)); // REX.W | REX.R
        self.b(0x8B);
        self.b(0x87 | ((reg & 7) << 3)); // mod=10, reg, rm=7(rdi)
        self.d(disp);
    }

    /// mov [rdi + disp32], reg64
    fn store_mem_reg(&mut self, disp: i32, reg: u8) {
        self.b(0x48 | ((reg & 8) >> 1));
        self.b(0x89);
        self.b(0x87 | ((reg & 7) << 3));
        self.d(disp);
    }

    /// mov dst64, src64  — no-op when dst == src.
    fn mov_rr(&mut self, dst: u8, src: u8) {
        if dst == src {
            return;
        }
        // MOV r64, r/m64 (0x8B): REX.R extends dst, REX.B extends src
        self.b(0x48 | ((dst & 8) >> 1) | ((src & 8) >> 3));
        self.b(0x8B);
        self.b(0xC0 | ((dst & 7) << 3) | (src & 7));
    }

    // ── Arithmetic (rax / rcx) ───────────────────────────────────────────────

    fn add_rax_rcx(&mut self) {
        self.b(0x48);
        self.b(0x01);
        self.b(0xC8);
    }
    fn sub_rax_rcx(&mut self) {
        self.b(0x48);
        self.b(0x29);
        self.b(0xC8);
    }
    fn imul_rax_rcx(&mut self) {
        self.b(0x48);
        self.b(0x0F);
        self.b(0xAF);
        self.b(0xC1);
    }
    fn add_rax_imm32(&mut self, v: i32) {
        self.b(0x48);
        self.b(0x05);
        self.d(v);
    }
    fn sub_rax_imm32(&mut self, v: i32) {
        self.b(0x48);
        self.b(0x2D);
        self.d(v);
    }
    fn imul_rax_imm32(&mut self, v: i32) {
        self.b(0x48);
        self.b(0x69);
        self.b(0xC0);
        self.d(v);
    }
    fn inc_rax(&mut self) {
        self.b(0x48);
        self.b(0xFF);
        self.b(0xC0);
    }
    fn dec_rax(&mut self) {
        self.b(0x48);
        self.b(0xFF);
        self.b(0xC8);
    }
    fn neg_rax(&mut self) {
        self.b(0x48);
        self.b(0xF7);
        self.b(0xD8);
    }
    fn shl_rax_imm8(&mut self, v: u8) {
        self.b(0x48);
        self.b(0xC1);
        self.b(0xE0);
        self.b(v);
    }
    fn xor_rax_rax(&mut self) {
        self.b(0x48);
        self.b(0x31);
        self.b(0xC0);
    }

    // LEA ×N patterns on rax only (rax = rax*N via SIB with base=rax, index=rax).
    // SIB for [rax + rax*K]: scale_bits<<6 | index=rax(0)<<3 | base=rax(0)
    fn lea_rax_rax_mul3(&mut self) {
        self.b(0x48);
        self.b(0x8D);
        self.b(0x04);
        self.b(0x40);
    }
    fn lea_rax_rax_mul5(&mut self) {
        self.b(0x48);
        self.b(0x8D);
        self.b(0x04);
        self.b(0x80);
    }
    fn lea_rax_rax_mul9(&mut self) {
        self.b(0x48);
        self.b(0x8D);
        self.b(0x04);
        self.b(0xC0);
    }

    /// lea rax, [rcx + rax*scale]   scale ∈ {2,4,8}
    /// Used by Mul+Add→LEA fusion: rax = multiplicand, rcx = addend.
    /// Result: rax*scale + rcx.
    fn lea_rax_rax_scale_plus_rcx(&mut self, scale: u8) {
        // SIB: ss<<6 | index=rax(0)<<3 | base=rcx(1)
        let ss: u8 = match scale {
            4 => 2,
            8 => 3,
            _ => 1,
        };
        self.b(0x48);
        self.b(0x8D);
        self.b(0x04);
        self.b((ss << 6) | 1);
    }

    // ── Division ─────────────────────────────────────────────────────────────

    fn cqo(&mut self) {
        self.b(0x48);
        self.b(0x99);
    }
    fn idiv_rcx(&mut self) {
        self.b(0x48);
        self.b(0xF7);
        self.b(0xF9);
    }
    fn mov_rax_rdx(&mut self) {
        self.b(0x48);
        self.b(0x89);
        self.b(0xD0);
    }

    // ── Compare / branch ─────────────────────────────────────────────────────

    fn cmp_rax_rcx(&mut self) {
        self.b(0x48);
        self.b(0x39);
        self.b(0xC8);
    }
    fn cmp_rax_imm32(&mut self, v: i32) {
        self.b(0x48);
        self.b(0x3D);
        self.d(v);
    }
    fn test_rax_rax(&mut self) {
        self.b(0x48);
        self.b(0x85);
        self.b(0xC0);
    }
    fn setcc_al(&mut self, cc: u8) {
        self.b(0x0F);
        self.b(cc);
        self.b(0xC0);
    }
    fn movzx_rax_al(&mut self) {
        self.b(0x48);
        self.b(0x0F);
        self.b(0xB6);
        self.b(0xC0);
    }

    fn jmp_rel32_placeholder(&mut self) -> usize {
        self.b(0xE9);
        let p = self.pos();
        self.d(0);
        p
    }
    fn jz_rel32_placeholder(&mut self) -> usize {
        self.b(0x0F);
        self.b(0x84);
        let p = self.pos();
        self.d(0);
        p
    }
    fn jnz_rel32_placeholder(&mut self) -> usize {
        self.b(0x0F);
        self.b(0x85);
        let p = self.pos();
        self.d(0);
        p
    }

    fn ret(&mut self) {
        self.b(0xC3);
    }

    // ── Callee-saved save/restore ────────────────────────────────────────────
    //
    // Short-form PUSH/POP r64: 0x50+rd  (rd = reg & 7)
    // r8-r15 require REX.B prefix (0x41).

    fn push_reg(&mut self, reg: u8) {
        if reg >= 8 {
            self.b(0x41);
        }
        self.b(0x50 + (reg & 7));
    }
    fn pop_reg(&mut self, reg: u8) {
        if reg >= 8 {
            self.b(0x41);
        }
        self.b(0x58 + (reg & 7));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Register allocation
// ─────────────────────────────────────────────────────────────────────────────

/// Allocation pool ordered: caller-saved first (cheaper — no push/pop),
/// then callee-saved.
///
/// Excluded permanently:
///   rax(0)  — primary accumulator / return value
///   rcx(1)  — secondary scratch operand
///   rdx(2)  — clobbered by CQO/IDIV
///   rdi(7)  — slot-array base pointer (function argument)
///   rsp(4) / rbp(5) — stack management
const ALLOC_POOL: &[u8] = &[
    8, 9, 10, 11, 6, // r8-r11, rsi  (caller-saved — free)
    12, 13, 14, 15, 3, // r12-r15, rbx (callee-saved — require push/pop)
];

/// Callee-saved regs in the pool — checked inline via matches!() in linear_scan.

/// Where a bytecode slot lives at runtime.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RegLoc {
    Reg(u8),    // allocated physical register
    Spill(i32), // byte offset inside rdi[] slot array
}

struct RegAlloc {
    /// Direct Vec indexed by slot number; pre-filled with Spill defaults.
    /// O(1) lookup vs HashMap for every load/store in the hot codegen loop.
    slots: Vec<RegLoc>,
    /// Callee-saved registers actually used — must be pushed in prologue.
    used_callee_saved: Vec<u8>,
}

impl RegAlloc {
    #[inline(always)]
    fn location(&self, slot: u16) -> RegLoc {
        // Safety: slots is pre-allocated to cover all valid slot indices.
        unsafe { *self.slots.get_unchecked(slot as usize) }
    }
}

// ── Live intervals ────────────────────────────────────────────────────────────

struct LiveInterval {
    slot: u16,
    first: usize, // index of first instruction that defines or uses this slot
    last: usize,  // index of last instruction that uses this slot
}

fn compute_live_intervals(instrs: &[Instr], slot_count: usize) -> Vec<LiveInterval> {
    const UNDEF: usize = usize::MAX;
    let cap = slot_count + 1;
    let mut first_def = vec![UNDEF; cap];
    let mut last_use = vec![UNDEF; cap];

    // Ensure slot fits in our vecs; grow if needed (shouldn't happen with correct slot_count).
    macro_rules! ensure {
        ($s:expr) => {
            let s = $s as usize;
            if s >= first_def.len() {
                first_def.resize(s + 1, UNDEF);
                last_use.resize(s + 1, UNDEF);
            }
        };
    }
    macro_rules! def {
        ($s:expr, $pc:expr) => {
            ensure!($s);
            let s = $s as usize;
            if first_def[s] == UNDEF {
                first_def[s] = $pc;
            }
        };
    }
    macro_rules! use_ {
        ($s:expr, $pc:expr) => {
            ensure!($s);
            last_use[$s as usize] = $pc;
        };
    }

    for (pc, instr) in instrs.iter().enumerate() {
        match instr {
            Instr::LoadI32(d, _)
            | Instr::LoadI64(d, _)
            | Instr::LoadBool(d, _)
            | Instr::LoadUnit(d) => {
                def!(*d, pc);
            }
            Instr::Move(d, s) | Instr::Load(d, s) => {
                use_!(*s, pc);
                def!(*d, pc);
            }
            Instr::Store(slot, s) => {
                use_!(*s, pc);
                def!(*slot, pc);
            }
            Instr::BinOp(d, _, l, r) => {
                use_!(*l, pc);
                use_!(*r, pc);
                def!(*d, pc);
            }
            Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) | Instr::Return(s) => {
                use_!(*s, pc);
            }
            _ => {}
        }
    }

    // Slots that appear only as uses (never defined) are argument slots — treat as defined at 0.
    let mut intervals: Vec<LiveInterval> = Vec::new();
    for slot in 0..first_def.len() {
        let lu = last_use[slot];
        let fd = first_def[slot];
        if fd == UNDEF && lu == UNDEF {
            continue;
        }
        let first = if fd == UNDEF { 0 } else { fd };
        let last = if lu == UNDEF { first } else { lu };
        intervals.push(LiveInterval {
            slot: slot as u16,
            first,
            last,
        });
    }
    intervals.sort_unstable_by_key(|i| (i.first, i.slot));
    intervals
}

// ── Linear-scan allocator ─────────────────────────────────────────────────────

fn linear_scan(intervals: &[LiveInterval], slot_count: usize) -> RegAlloc {
    // Pre-fill slots with their default Spill locations (slot * 8 offset).
    let cap = slot_count + 1;
    let mut slots: Vec<RegLoc> = (0..cap).map(|s| RegLoc::Spill((s as i32) * 8)).collect();

    // Grow helper in case any slot exceeds slot_count (shouldn't happen normally).
    let ensure_slot = |slots: &mut Vec<RegLoc>, slot: u16| {
        let idx = slot as usize;
        if idx >= slots.len() {
            slots.resize(idx + 1, RegLoc::Spill((idx as i32) * 8));
        }
    };

    // Free list: iterate ALLOC_POOL in reverse so pop() gives caller-saved first.
    let mut free: Vec<u8> = ALLOC_POOL.iter().rev().copied().collect();
    // Active set sorted by interval end (ascending).
    let mut active: Vec<(usize, u16, u8)> = Vec::new(); // (end, slot, reg)
    let mut used_callee_saved: Vec<u8> = Vec::new();
    // Bitmask to avoid O(n) contains() on used_callee_saved (regs 0-15 fit in u16).
    let mut callee_saved_mask: u16 = 0;

    #[inline(always)]
    fn is_callee_saved(reg: u8) -> bool {
        matches!(reg, 3 | 12..=15)
    }

    for iv in intervals {
        // Expire intervals ended strictly before this one's start — no temp Vec needed.
        let mut freed_count = 0usize;
        let active_len = active.len();
        for i in 0..active_len {
            if active[i].0 < iv.first {
                free.push(active[i].2);
                freed_count += 1;
            }
        }
        if freed_count > 0 {
            // Partition: keep entries where end >= iv.first.
            active.retain(|(end, _, _)| *end >= iv.first);
        }

        let mut track_callee = |reg: u8| {
            if is_callee_saved(reg) && (callee_saved_mask & (1u16 << reg)) == 0 {
                callee_saved_mask |= 1u16 << reg;
                used_callee_saved.push(reg);
            }
        };

        if let Some(reg) = free.pop() {
            track_callee(reg);
            ensure_slot(&mut slots, iv.slot);
            slots[iv.slot as usize] = RegLoc::Reg(reg);
            let pos = active.partition_point(|(e, _, _)| *e <= iv.last);
            active.insert(pos, (iv.last, iv.slot, reg));
        } else {
            match active.last().copied() {
                Some((end, spill_slot, reg)) if end > iv.last => {
                    ensure_slot(&mut slots, spill_slot);
                    slots[spill_slot as usize] = RegLoc::Spill((spill_slot as i32) * 8);
                    active.pop();
                    track_callee(reg);
                    ensure_slot(&mut slots, iv.slot);
                    slots[iv.slot as usize] = RegLoc::Reg(reg);
                    let pos = active.partition_point(|(e, _, _)| *e <= iv.last);
                    active.insert(pos, (iv.last, iv.slot, reg));
                }
                _ => {
                    ensure_slot(&mut slots, iv.slot);
                    // Already pre-filled with Spill; nothing to do.
                }
            }
        }
    }

    RegAlloc {
        slots,
        used_callee_saved,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constant propagation
// ─────────────────────────────────────────────────────────────────────────────

fn fold_binop(op: BinOpKind, l: i64, r: i64) -> Option<i64> {
    Some(match op {
        BinOpKind::Add => l.wrapping_add(r),
        BinOpKind::Sub => l.wrapping_sub(r),
        BinOpKind::Mul => l.wrapping_mul(r),
        BinOpKind::Div => {
            if r == 0 {
                return None;
            }
            l.wrapping_div(r)
        }
        BinOpKind::Rem => {
            if r == 0 {
                return None;
            }
            l.wrapping_rem(r)
        }
        BinOpKind::Eq => i64::from(l == r),
        BinOpKind::Ne => i64::from(l != r),
        BinOpKind::Lt => i64::from(l < r),
        BinOpKind::Le => i64::from(l <= r),
        BinOpKind::Gt => i64::from(l > r),
        BinOpKind::Ge => i64::from(l >= r),
        _ => return None,
    })
}

// (Constant propagation is now maintained as a rolling inline state
//  in the main codegen loop — see `const_at` in `translate`. This
//  eliminates O(n) HashMap clones that the old pre-pass required.)

// ─────────────────────────────────────────────────────────────────────────────
// Emission helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn load_rax(em: &mut Emitter, slot: u16, ra: &RegAlloc) {
    match ra.location(slot) {
        RegLoc::Reg(r) => em.mov_rr(0, r),
        RegLoc::Spill(off) => em.load_reg_mem(0, off),
    }
}

#[inline(always)]
fn load_rcx(em: &mut Emitter, slot: u16, ra: &RegAlloc) {
    match ra.location(slot) {
        RegLoc::Reg(r) => em.mov_rr(1, r),
        RegLoc::Spill(off) => em.load_reg_mem(1, off),
    }
}

#[inline(always)]
fn store_rax(em: &mut Emitter, slot: u16, ra: &RegAlloc) {
    match ra.location(slot) {
        RegLoc::Reg(r) => em.mov_rr(r, 0),
        RegLoc::Spill(off) => em.store_mem_reg(off, 0),
    }
}

#[inline(always)]
fn instr_reads_slot(instr: &Instr, slot: u16) -> bool {
    match instr {
        Instr::Move(_, s) | Instr::Load(_, s) | Instr::Store(_, s) | Instr::Return(s) => *s == slot,
        Instr::BinOp(_, _, l, r) => *l == slot || *r == slot,
        Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => *s == slot,
        _ => false,
    }
}

#[inline(always)]
fn instr_writes_slot(instr: &Instr, slot: u16) -> bool {
    match instr {
        Instr::LoadI32(d, _)
        | Instr::LoadI64(d, _)
        | Instr::LoadBool(d, _)
        | Instr::LoadUnit(d)
        | Instr::Move(d, _)
        | Instr::Load(d, _)
        | Instr::Store(d, _)
        | Instr::BinOp(d, _, _, _) => *d == slot,
        _ => false,
    }
}

#[inline(always)]
fn is_control_flow_barrier(instr: &Instr) -> bool {
    matches!(
        instr,
        Instr::Jump(_)
            | Instr::JumpFalse(_, _)
            | Instr::JumpTrue(_, _)
            | Instr::Return(_)
            | Instr::ReturnUnit
    )
}

/// Local straight-line dead-definition check.
///
/// Returns true when a write to `slot` at `pc` is overwritten before any read
/// and before any control-flow barrier.
fn is_straight_line_dead_def(instrs: &[Instr], pc: usize, slot: u16) -> bool {
    let mut i = pc + 1;
    while i < instrs.len() {
        let next = &instrs[i];
        if is_control_flow_barrier(next) {
            return false;
        }
        if instr_reads_slot(next, slot) {
            return false;
        }
        if instr_writes_slot(next, slot) {
            return true;
        }
        i += 1;
    }
    true
}

/// Returns true when `op` is in the set we know how to emit.
#[inline(always)]
fn is_supported_binop(op: BinOpKind) -> bool {
    matches!(
        op,
        BinOpKind::Add
            | BinOpKind::Sub
            | BinOpKind::Mul
            | BinOpKind::Div
            | BinOpKind::Rem
            | BinOpKind::Eq
            | BinOpKind::Ne
            | BinOpKind::Lt
            | BinOpKind::Le
            | BinOpKind::Gt
            | BinOpKind::Ge
    )
}

/// Emit the arithmetic/comparison body: lhs already in rax, rhs in rcx.
/// Returns `false` only for ops we can't yet compile (caller must bail out
/// **before** emitting any operand loads).
fn emit_binop_rax_rcx(em: &mut Emitter, op: BinOpKind) -> bool {
    match op {
        BinOpKind::Add => em.add_rax_rcx(),
        BinOpKind::Sub => em.sub_rax_rcx(),
        BinOpKind::Mul => em.imul_rax_rcx(),
        BinOpKind::Div => {
            em.cqo();
            em.idiv_rcx();
        }
        BinOpKind::Rem => {
            em.cqo();
            em.idiv_rcx();
            em.mov_rax_rdx();
        }
        BinOpKind::Eq => {
            em.cmp_rax_rcx();
            em.setcc_al(0x94);
            em.movzx_rax_al();
        }
        BinOpKind::Ne => {
            em.cmp_rax_rcx();
            em.setcc_al(0x95);
            em.movzx_rax_al();
        }
        BinOpKind::Lt => {
            em.cmp_rax_rcx();
            em.setcc_al(0x9C);
            em.movzx_rax_al();
        }
        BinOpKind::Le => {
            em.cmp_rax_rcx();
            em.setcc_al(0x9E);
            em.movzx_rax_al();
        }
        BinOpKind::Gt => {
            em.cmp_rax_rcx();
            em.setcc_al(0x9F);
            em.movzx_rax_al();
        }
        BinOpKind::Ge => {
            em.cmp_rax_rcx();
            em.setcc_al(0x9D);
            em.movzx_rax_al();
        }
        _ => return false,
    }
    true
}

/// Emit immediate-rhs form: lhs already in rax.
/// Only called after confirming `op` and `imm` are valid for this path.
fn emit_binop_rax_imm(em: &mut Emitter, op: BinOpKind, imm: i32) {
    match op {
        BinOpKind::Add => {
            if imm == 1 {
                em.inc_rax();
            } else if imm == -1 {
                em.dec_rax();
            } else if imm != 0 {
                em.add_rax_imm32(imm);
            }
        }
        BinOpKind::Sub => {
            if imm == 1 {
                em.dec_rax();
            } else if imm == -1 {
                em.inc_rax();
            } else if imm != 0 {
                em.sub_rax_imm32(imm);
            }
        }
        BinOpKind::Mul => {
            if imm == 0 {
                em.xor_rax_rax();
            } else if imm == 1 { /* nop */
            } else if imm == -1 {
                em.neg_rax();
            } else if imm == 3 {
                em.lea_rax_rax_mul3();
            } else if imm == 5 {
                em.lea_rax_rax_mul5();
            } else if imm == 9 {
                em.lea_rax_rax_mul9();
            } else if imm > 0 && (imm as u32).is_power_of_two() {
                em.shl_rax_imm8((imm as u32).trailing_zeros() as u8);
            } else {
                em.imul_rax_imm32(imm);
            }
        }
        BinOpKind::Eq => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x94);
            em.movzx_rax_al();
        }
        BinOpKind::Ne => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x95);
            em.movzx_rax_al();
        }
        BinOpKind::Lt => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x9C);
            em.movzx_rax_al();
        }
        BinOpKind::Le => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x9E);
            em.movzx_rax_al();
        }
        BinOpKind::Gt => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x9F);
            em.movzx_rax_al();
        }
        BinOpKind::Ge => {
            em.cmp_rax_imm32(imm);
            em.setcc_al(0x9D);
            em.movzx_rax_al();
        }
        _ => {}
    }
}

/// Emit pops (reverse push order) followed by RET.
/// Every code path that returns must go through this.
fn emit_ret(em: &mut Emitter, callee_saved: &[u8]) {
    for &reg in callee_saved.iter().rev() {
        em.pop_reg(reg);
    }
    em.ret();
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

#[must_use]
pub fn is_available() -> bool {
    cfg!(target_arch = "x86_64")
}

pub fn translate(compiled: &CompiledFn) -> Option<NativeCode> {
    if !cfg!(target_arch = "x86_64") {
        return None;
    }

    // Gate: bail out early if any instruction is outside our supported set.
    for instr in &optimized_instrs {
        match instr {
            Instr::LoadI32(..)
            | Instr::LoadI64(..)
            | Instr::LoadBool(..)
            | Instr::LoadUnit(..)
            | Instr::Move(..)
            | Instr::Load(..)
            | Instr::Store(..)
            | Instr::BinOp(..)
            | Instr::Jump(..)
            | Instr::JumpFalse(..)
            | Instr::JumpTrue(..)
            | Instr::Return(..)
            | Instr::ReturnUnit
            | Instr::Nop => {}
            _ => return None,
        }
    }

    let instrs = &compiled.instrs;
    let slot_count = compiled.slot_count as usize;

    // ── Pass 1: liveness + linear-scan register allocation ───────────────
    // (const_prop state is maintained inline in the codegen loop below,
    //  eliminating the O(n) HashMap-clone pre-pass entirely.)
    let intervals = compute_live_intervals(instrs, slot_count);
    let ra = linear_scan(&intervals, slot_count);

    // ── Emission ──────────────────────────────────────────────────────────
    let mut em = Emitter::new();
    let mut pc_to_off = vec![0usize; instrs.len() + 1];
    let mut fixups: Vec<(usize, usize)> = Vec::new(); // (disp32_pos, target_pc)

    // Prologue: save callee-saved registers we actually use.
    for &reg in &ra.used_callee_saved {
        em.push_reg(reg);
    }

    // Pre-load every register-allocated slot from the slot array.
    // Argument slots receive their caller-provided values; non-argument slots
    // load zero (execute() zeroes the array) which is overwritten before first use.
    // Tracks by register so each physical register is loaded exactly once.
    {
        let mut preloaded: u32 = 0u32; // bitmask for regs 0-31
        for iv in &intervals {
            if let RegLoc::Reg(r) = ra.location(iv.slot) {
                let bit = 1u32 << r;
                if preloaded & bit == 0 {
                    preloaded |= bit;
                    let off = (iv.slot as i32) * 8;
                    em.load_reg_mem(r, off);
                }
            }
        }
    }

    // ── Main translation loop ─────────────────────────────────────────────
    // Inline constant-propagation state: avoids pre-allocating O(n) HashMaps.
    let mut const_at: HashMap<u16, i64> = HashMap::new();

    let mut pc = 0usize;
    while pc < instrs.len() {
        pc_to_off[pc] = em.pos();

        // ════════════════════════════════════════════════════════════════════
        // 3-INSTRUCTION FUSIONS
        // ════════════════════════════════════════════════════════════════════

        // ── Fusion: BinOp(t, Mul, x, N) + BinOp(r, Add, t, y)  →  LEA ──
        //
        // Emits:  lea rax, [y_reg + x_reg * N]    (N ∈ {2,4,8})
        // x must be a compile-time-unknown operand; the scalar N must be a
        // compile-time constant in const_at (so N can come from a prior LoadI).
        if pc + 1 < instrs.len() {
            if let (
                Instr::BinOp(t, BinOpKind::Mul, mul_l, mul_r),
                Instr::BinOp(r, BinOpKind::Add, add_l, add_r),
            ) = (&instrs[pc], &instrs[pc + 1])
            {
                // The Mul result `t` must be one (and only one) operand of the Add.
                // The other operand of the Add is the addend `y`.
                // `t` must not also appear as the Add's destination to avoid
                // clobbering a slot still used as the addend.
                let (addend_slot, t_consumed_by_add) = if *add_l == *t && *add_r != *t && *r != *t {
                    (Some(*add_r), true)
                } else if *add_r == *t && *add_l != *t && *r != *t {
                    (Some(*add_l), true)
                } else {
                    (None, false)
                };

                if t_consumed_by_add {
                    let addend = addend_slot.unwrap();
                    // Determine which operand of Mul is the constant scale.
                    let maybe_lea = const_at
                        .get(mul_r)
                        .copied()
                        .map(|c| (*mul_l, c))
                        .or_else(|| const_at.get(mul_l).copied().map(|c| (*mul_r, c)));

                    if let Some((base, scale_i64)) = maybe_lea {
                        if matches!(scale_i64, 2 | 4 | 8) {
                            let scale = scale_i64 as u8;
                            // rax = base (multiplicand), rcx = addend
                            load_rax(&mut em, base, &ra);
                            load_rcx(&mut em, addend, &ra);
                            em.lea_rax_rax_scale_plus_rcx(scale);
                            if !is_straight_line_dead_def(instrs, pc, *r) {
                                store_rax(&mut em, *r, &ra);
                            }
                            pc_to_off[pc + 1] = em.pos();
                            pc += 2;
                            continue;
                        }
                    }
                }
            }
        }

        // ── Fusion: BinOp(t, op1, a, b) + BinOp(r, op2, t, c) → chain ──
        //
        // Eliminates the intermediate slot write + read for `t`.
        // Requires both ops to be supported and, when `t` is the *right*
        // operand of op2, requires op2 to be commutative (so we can swap).
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op1, a, b), Instr::BinOp(r, op2, l2, r2)) =
                (&instrs[pc], &instrs[pc + 1])
            {
                // Determine whether `t` appears as lhs or (for commutative ops) rhs of op2,
                // and that `t` isn't also used as the other operand (which would alias).
                let commutative = matches!(
                    op2,
                    BinOpKind::Add | BinOpKind::Mul | BinOpKind::Eq | BinOpKind::Ne
                );
                let t_as_lhs = *l2 == *t && *r2 != *t;
                let t_as_rhs = *r2 == *t && *l2 != *t && commutative;

                if (t_as_lhs || t_as_rhs) && is_supported_binop(*op1) && is_supported_binop(*op2) {
                    let other = if t_as_lhs { *r2 } else { *l2 };
                    load_rax(&mut em, *a, &ra);
                    load_rcx(&mut em, *b, &ra);
                    // op1 is guaranteed supported — never returns false.
                    emit_binop_rax_rcx(&mut em, *op1);
                    // rax now holds the result of op1 (= the new value of `t`).
                    load_rcx(&mut em, other, &ra);
                    emit_binop_rax_rcx(&mut em, *op2);
                    if !is_straight_line_dead_def(instrs, pc, *r) {
                        store_rax(&mut em, *r, &ra);
                    }
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // 2-INSTRUCTION FUSIONS
        // ════════════════════════════════════════════════════════════════════

        // ── Fusion: Load*(tmp, c) + JumpFalse/JumpTrue → compile-time branch ──
        if pc + 1 < instrs.len() {
            let maybe_const = match &instrs[pc] {
                Instr::LoadI32(tmp, v) => Some((*tmp, *v as i64)),
                Instr::LoadI64(tmp, v) => Some((*tmp, *v)),
                Instr::LoadBool(tmp, v) => Some((*tmp, i64::from(*v))),
                Instr::LoadUnit(tmp) => Some((*tmp, 0i64)),
                _ => None,
            };
            if let Some((tmp, c)) = maybe_const {
                let mut folded = false;
                match &instrs[pc + 1] {
                    Instr::JumpFalse(cond, off) if *cond == tmp => {
                        let target = ((pc as i32) + 2 + *off) as usize;
                        if target > instrs.len() {
                            return None;
                        }
                        if c == 0 {
                            let p = em.jmp_rel32_placeholder();
                            fixups.push((p, target));
                        }
                        folded = true;
                    }
                    Instr::JumpTrue(cond, off) if *cond == tmp => {
                        let target = ((pc as i32) + 2 + *off) as usize;
                        if target > instrs.len() {
                            return None;
                        }
                        if c != 0 {
                            let p = em.jmp_rel32_placeholder();
                            fixups.push((p, target));
                        }
                        folded = true;
                    }
                    _ => {}
                }
                if folded {
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: Load*(tmp, c) + BinOp(dst, op, x, tmp) → imm arithmetic ──
        if pc + 1 < instrs.len() {
            let maybe_imm = match &instrs[pc] {
                Instr::LoadI32(tmp, v) => Some((*tmp, *v as i64)),
                Instr::LoadI64(tmp, v) => Some((*tmp, *v)),
                _ => None,
            };
            if let Some((tmp, c)) = maybe_imm {
                if let Instr::BinOp(dst, op, l, r) = &instrs[pc + 1] {
                    if let Ok(imm) = i32::try_from(c) {
                        let rhs_is_imm = *r == tmp;
                        let lhs_is_imm = *l == tmp;
                        // Can use imm form when: tmp is the rhs for all ops,
                        // or the lhs for commutative (Add/Mul) ops.
                        let can_use = match op {
                            BinOpKind::Add | BinOpKind::Mul => rhs_is_imm || lhs_is_imm,
                            BinOpKind::Sub
                            | BinOpKind::Eq
                            | BinOpKind::Ne
                            | BinOpKind::Lt
                            | BinOpKind::Le
                            | BinOpKind::Gt
                            | BinOpKind::Ge => rhs_is_imm,
                            _ => false,
                        };
                        if can_use {
                            let live_reg = if rhs_is_imm { *l } else { *r };
                            load_rax(&mut em, live_reg, &ra);
                            emit_binop_rax_imm(&mut em, *op, imm);
                            if !is_straight_line_dead_def(instrs, pc, *dst) {
                                store_rax(&mut em, *dst, &ra);
                            }
                            pc_to_off[pc + 1] = em.pos();
                            pc += 2;
                            continue;
                        }
                    }
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + Store(slot, t) ──────────────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::Store(slot, src)) =
                (&instrs[pc], &instrs[pc + 1])
            {
                if t == src && is_supported_binop(*op) {
                    load_rax(&mut em, *l, &ra);
                    load_rcx(&mut em, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op);
                    if !is_straight_line_dead_def(instrs, pc, *slot) {
                        store_rax(&mut em, *slot, &ra);
                    }
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + JumpFalse(t, off) ──────────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::JumpFalse(cond, off)) =
                (&instrs[pc], &instrs[pc + 1])
            {
                if t == cond && is_supported_binop(*op) {
                    let target = ((pc as i32) + 2 + *off) as usize;
                    if target > instrs.len() {
                        return None;
                    }
                    load_rax(&mut em, *l, &ra);
                    load_rcx(&mut em, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op);
                    em.test_rax_rax();
                    let p = em.jz_rel32_placeholder();
                    fixups.push((p, target));
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + JumpTrue(t, off) ───────────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::JumpTrue(cond, off)) =
                (&instrs[pc], &instrs[pc + 1])
            {
                if t == cond && is_supported_binop(*op) {
                    let target = ((pc as i32) + 2 + *off) as usize;
                    if target > instrs.len() {
                        return None;
                    }
                    load_rax(&mut em, *l, &ra);
                    load_rcx(&mut em, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op);
                    em.test_rax_rax();
                    let p = em.jnz_rel32_placeholder();
                    fixups.push((p, target));
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + Return(t) ──────────────────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::Return(ret)) = (&instrs[pc], &instrs[pc + 1])
            {
                if t == ret && is_supported_binop(*op) {
                    load_rax(&mut em, *l, &ra);
                    load_rcx(&mut em, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op);
                    emit_ret(&mut em, &ra.used_callee_saved);
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // CONSTANT FOLDING (single BinOp with both operands known)
        // ════════════════════════════════════════════════════════════════════

        if let Instr::BinOp(d, op, l, r) = &instrs[pc] {
            if let Some(v) = const_at
                .get(l)
                .copied()
                .zip(const_at.get(r).copied())
                .and_then(|(lv, rv)| fold_binop(*op, lv, rv))
            {
                em.mov_rax_imm_opt(v);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, v);
                pc += 1;
                continue;
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // SINGLE-INSTRUCTION FALLBACK
        // ════════════════════════════════════════════════════════════════════

        match &instrs[pc] {
            Instr::LoadI32(d, v) => {
                let cv = *v as i64;
                em.mov_rax_imm_opt(cv);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, cv);
            }
            Instr::LoadI64(d, v) => {
                em.mov_rax_imm_opt(*v);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, *v);
            }
            Instr::LoadBool(d, v) => {
                let cv = i64::from(*v);
                em.mov_rax_imm_opt(cv);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, cv);
            }
            Instr::LoadUnit(d) => {
                em.xor_rax_rax();
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                const_at.insert(*d, 0);
            }
            Instr::Move(d, s) | Instr::Load(d, s) => {
                load_rax(&mut em, *s, &ra);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                if let Some(&c) = const_at.get(s).map(|c| c) {
                    const_at.insert(*d, c);
                } else {
                    const_at.remove(d);
                }
            }
            Instr::Store(slot, s) => {
                load_rax(&mut em, *s, &ra);
                if !is_straight_line_dead_def(instrs, pc, *slot) {
                    store_rax(&mut em, *slot, &ra);
                }
                if let Some(&c) = const_at.get(s) {
                    const_at.insert(*slot, c);
                } else {
                    const_at.remove(slot);
                }
            }
            Instr::BinOp(d, op, l, r) => {
                if !is_supported_binop(*op) {
                    return None;
                }
                load_rax(&mut em, *l, &ra);
                load_rcx(&mut em, *r, &ra);
                emit_binop_rax_rcx(&mut em, *op);
                if !is_straight_line_dead_def(instrs, pc, *d) {
                    store_rax(&mut em, *d, &ra);
                }
                let folded = const_at
                    .get(l)
                    .copied()
                    .zip(const_at.get(r).copied())
                    .and_then(|(lv, rv)| fold_binop(*op, lv, rv));
                match folded {
                    Some(c) => {
                        const_at.insert(*d, c);
                    }
                    None => {
                        const_at.remove(d);
                    }
                }
            }
            Instr::Jump(off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                if target > instrs.len() {
                    return None;
                }
                let p = em.jmp_rel32_placeholder();
                fixups.push((p, target));
                const_at.clear(); // conservative: branch target may have unknown state
            }
            Instr::JumpFalse(cond, off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                if target > instrs.len() {
                    return None;
                }
                load_rax(&mut em, *cond, &ra);
                em.test_rax_rax();
                let p = em.jz_rel32_placeholder();
                fixups.push((p, target));
                const_at.clear();
            }
            Instr::JumpTrue(cond, off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                if target > instrs.len() {
                    return None;
                }
                load_rax(&mut em, *cond, &ra);
                em.test_rax_rax();
                let p = em.jnz_rel32_placeholder();
                fixups.push((p, target));
                const_at.clear();
            }
            Instr::Return(r) => {
                load_rax(&mut em, *r, &ra);
                emit_ret(&mut em, &ra.used_callee_saved);
            }
            Instr::ReturnUnit => {
                em.xor_rax_rax();
                emit_ret(&mut em, &ra.used_callee_saved);
            }
            Instr::Nop => {}
            _ => return None,
        }
        pc += 1;
    }

    // Fallthrough epilogue (reached when execution falls off the end).
    pc_to_off[instrs.len()] = em.pos();
    em.xor_rax_rax();
    emit_ret(&mut em, &ra.used_callee_saved);

    // ── Patch branch displacements ────────────────────────────────────────
    for (disp_pos, target_pc) in fixups {
        let target_off = *pc_to_off.get(target_pc)? as isize;
        let next_ip = (disp_pos + 4) as isize;
        let rel = i32::try_from(target_off - next_ip).ok()?;
        em.buf[disp_pos..disp_pos + 4].copy_from_slice(&rel.to_le_bytes());
    }

    let mem = ExecMem::new(&em.buf)?;
    Some(NativeCode {
        slot_count: compiled.slot_count,
        mem,
    })
}

pub fn execute(native: &NativeCode, args: &[Value]) -> Result<Value, RuntimeError> {
    thread_local! {
        static EXEC_REGS: RefCell<Vec<i64>> = RefCell::new(Vec::new());
    }
    let needed = native.slot_count as usize + 32;
    EXEC_REGS.with(|cell| -> Result<Value, RuntimeError> {
        let mut regs = cell.borrow_mut();
        if regs.len() < needed {
            regs.resize(needed, 0);
        } else {
            regs[..needed].fill(0);
        }
        for (i, arg) in args.iter().enumerate() {
            if i >= needed {
                break;
            }
            regs[i] = match arg {
                Value::I8(v) => *v as i64,
                Value::I16(v) => *v as i64,
                Value::I32(v) => *v as i64,
                Value::I64(v) => *v,
                Value::U8(v) => *v as i64,
                Value::U16(v) => *v as i64,
                Value::U32(v) => *v as i64,
                Value::U64(v) => *v as i64,
                Value::Bool(v) => i64::from(*v),
                _ => {
                    return Err(RuntimeError::new(
                        "native machine-code JIT supports int/bool args",
                    ))
                }
            };
        }
        let f = native.mem.entry();
        let out = unsafe { f(regs.as_mut_ptr()) };
        Ok(Value::I64(out))
    })
}
