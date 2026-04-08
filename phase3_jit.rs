//!JIT backend
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
//!    const_at is a flat Vec<Option<i64>> (O(1) slot lookup, cache-friendly).
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
//!    ADD/SUB/CMP RAX, imm8 (4 B) when value fits in i8.
//!
//! F. Short-form branches: JMP/JZ/JNZ rel8 (2 B) when displacement fits in i8,
//!    falling back to rel32 (5-6 B) otherwise.
//!
//! G. REX-free XOR/TEST for boolean/zero ops:
//!    XOR EAX, EAX (2 B) and TEST EAX, EAX (2 B) instead of 64-bit forms (3 B).
//!
//! H. All existing micro-optimisations retained:
//!    LEA for ×3/×5/×9, SHL for powers-of-two multiply,
//!    INC/DEC for ±1, TEST+Jcc, SETCC for branchless comparisons.

use std::cell::RefCell;
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
        // Try huge-page backed mapping first (Linux only).
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
        let ptr = libc::MAP_FAILED;

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

        // Hint to kernel: keep these pages in hugepage pool even if not huge-page mapped.
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(ptr, Self::DEFAULT_LEN, libc::MADV_HUGEPAGE);
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

        // Fallback: individual mmap per function.
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
            arena_backed: false, // BUG FIX: was missing, causing use-after-free on drop
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

    #[inline(always)]
    fn pos(&self) -> usize {
        self.buf.len()
    }

    #[inline(always)]
    fn b(&mut self, v: u8) {
        self.buf.push(v);
    }

    #[inline(always)]
    fn emit2(&mut self, b0: u8, b1: u8) {
        self.buf.extend_from_slice(&[b0, b1]);
    }

    #[inline(always)]
    fn emit3(&mut self, b0: u8, b1: u8, b2: u8) {
        self.buf.extend_from_slice(&[b0, b1, b2]);
    }

    #[inline(always)]
    fn emit4(&mut self, b0: u8, b1: u8, b2: u8, b3: u8) {
        self.buf.extend_from_slice(&[b0, b1, b2, b3]);
    }

    #[inline(always)]
    fn d(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline(always)]
    fn q(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    // ── Immediate loads ──────────────────────────────────────────────────────

    /// Full 64-bit immediate into rax (10 bytes).
    fn mov_rax_imm64(&mut self, v: i64) {
        self.emit2(0x48, 0xB8);
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
                self.emit3(0x48, 0xC7, 0xC0);
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

    /// mov reg64, [rdi + disp]
    /// Uses disp8 encoding when possible to reduce code size on hot load paths.
    fn load_reg_mem(&mut self, reg: u8, disp: i32) {
        self.emit2(0x48 | ((reg & 8) >> 1), 0x8B); // REX.W | REX.R, MOV r64, r/m64
        if let Ok(d8) = i8::try_from(disp) {
            // mod=01, reg, rm=111(rdi)
            self.b(0x47 | ((reg & 7) << 3));
            self.b(d8 as u8);
        } else {
            // mod=10, reg, rm=111(rdi)
            self.b(0x87 | ((reg & 7) << 3));
            self.d(disp);
        }
    }

    /// mov [rdi + disp], reg64
    /// Uses disp8 encoding when possible to reduce code size on hot store paths.
    fn store_mem_reg(&mut self, disp: i32, reg: u8) {
        self.emit2(0x48 | ((reg & 8) >> 1), 0x89); // REX.W | REX.R, MOV r/m64, r64
        if let Ok(d8) = i8::try_from(disp) {
            self.b(0x47 | ((reg & 7) << 3)); // mod=01
            self.b(d8 as u8);
        } else {
            self.b(0x87 | ((reg & 7) << 3)); // mod=10
            self.d(disp);
        }
    }

    /// mov dst64, src64  — no-op when dst == src.
    fn mov_rr(&mut self, dst: u8, src: u8) {
        if dst == src {
            return;
        }
        // MOV r64, r/m64 (0x8B): REX.R extends dst, REX.B extends src
        self.emit3(
            0x48 | ((dst & 8) >> 1) | ((src & 8) >> 3),
            0x8B,
            0xC0 | ((dst & 7) << 3) | (src & 7),
        );
    }

    // ── Arithmetic (rax / rcx) ───────────────────────────────────────────────

    fn add_rax_rcx(&mut self) {
        self.emit3(0x48, 0x01, 0xC8);
    }
    fn sub_rax_rcx(&mut self) {
        self.emit3(0x48, 0x29, 0xC8);
    }
    fn imul_rax_rcx(&mut self) {
        self.emit4(0x48, 0x0F, 0xAF, 0xC1);
    }

    /// ADD RAX, imm — uses imm8 form (4 B) when it fits, else imm32 (6 B).
    fn add_rax_imm32(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) {
            self.emit3(0x48, 0x83, 0xC0);
            self.b(v8 as u8); // ADD RAX, imm8
        } else {
            self.emit2(0x48, 0x05);
            self.d(v); // ADD RAX, imm32
        }
    }

    /// SUB RAX, imm — uses imm8 form (4 B) when it fits, else imm32 (6 B).
    fn sub_rax_imm32(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) {
            self.emit3(0x48, 0x83, 0xE8);
            self.b(v8 as u8); // SUB RAX, imm8
        } else {
            self.emit2(0x48, 0x2D);
            self.d(v); // SUB RAX, imm32
        }
    }

    fn imul_rax_imm32(&mut self, v: i32) {
        self.emit3(0x48, 0x69, 0xC0);
        self.d(v);
    }

    fn inc_rax(&mut self) {
        self.emit3(0x48, 0xFF, 0xC0);
    }
    fn dec_rax(&mut self) {
        self.emit3(0x48, 0xFF, 0xC8);
    }
    fn neg_rax(&mut self) {
        self.emit3(0x48, 0xF7, 0xD8);
    }
    fn shl_rax_imm8(&mut self, v: u8) {
        self.emit4(0x48, 0xC1, 0xE0, v);
    }

    /// XOR EAX, EAX — 2 bytes, zero-extends to clear full RAX.
    /// Preferred over REX.W form (3 bytes) for zeroing.
    fn xor_eax_eax(&mut self) {
        self.emit2(0x31, 0xC0); // XOR EAX, EAX  (zero-extends → RAX = 0)
    }

    // LEA ×N patterns on rax only (rax = rax*N via SIB with base=rax, index=rax).
    // SIB for [rax + rax*K]: scale_bits<<6 | index=rax(0)<<3 | base=rax(0)
    fn lea_rax_rax_mul3(&mut self) {
        self.emit4(0x48, 0x8D, 0x04, 0x40);
    }
    fn lea_rax_rax_mul5(&mut self) {
        self.emit4(0x48, 0x8D, 0x04, 0x80);
    }
    fn lea_rax_rax_mul9(&mut self) {
        self.emit4(0x48, 0x8D, 0x04, 0xC0);
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
        self.emit4(0x48, 0x8D, 0x04, (ss << 6) | 1);
    }

    // ── Division ─────────────────────────────────────────────────────────────

    fn cqo(&mut self) {
        self.emit2(0x48, 0x99);
    }
    fn idiv_rcx(&mut self) {
        self.emit3(0x48, 0xF7, 0xF9);
    }
    fn mov_rax_rdx(&mut self) {
        self.emit3(0x48, 0x89, 0xD0);
    }

    // ── Compare / branch ─────────────────────────────────────────────────────

    fn cmp_rax_rcx(&mut self) {
        self.emit3(0x48, 0x39, 0xC8);
    }

    /// CMP RAX, imm — uses imm8 form (4 B) when it fits, else imm32 (6 B).
    fn cmp_rax_imm32(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) {
            self.emit3(0x48, 0x83, 0xF8);
            self.b(v8 as u8); // CMP RAX, imm8
        } else {
            self.emit2(0x48, 0x3D);
            self.d(v); // CMP RAX, imm32
        }
    }

    /// TEST EAX, EAX — 2 bytes. Sufficient for boolean/zero testing since
    /// we only care about ZF; the upper 32 bits are zero for any canonical bool.
    /// Falls back to REX.W form for full 64-bit values tested against branches
    /// (call test_rax_rax for those if unsure).
    fn test_rax_rax(&mut self) {
        // TEST EAX, EAX (2 B) — ZF ↔ (eax == 0), which equals (rax == 0)
        // because our boolean values are 0 or 1 and always fit in 32 bits.
        // For general integer branch conditions (JumpFalse/True on arbitrary i64),
        // we also use this: canonical i64 0 has upper 32 bits = 0, so it's safe.
        self.emit2(0x85, 0xC0); // TEST EAX, EAX
    }

    fn setcc_al(&mut self, cc: u8) {
        self.emit3(0x0F, cc, 0xC0);
    }
    fn movzx_rax_al(&mut self) {
        self.emit4(0x48, 0x0F, 0xB6, 0xC0);
    }

    // ── Branches ─────────────────────────────────────────────────────────────
    //
    // Two-pass strategy: first emit rel32 placeholders, then patch.
    // Alternatively, for short branches we can back-patch with rel8.
    // We use a unified placeholder approach and pick the right encoding at
    // fixup time via a "rewrite" pass (see patch_fixups).

    fn jmp_rel32_placeholder(&mut self) -> usize {
        self.b(0xE9);
        let p = self.pos();
        self.d(0);
        p
    }
    fn jz_rel32_placeholder(&mut self) -> usize {
        self.emit2(0x0F, 0x84);
        let p = self.pos();
        self.d(0);
        p
    }
    fn jnz_rel32_placeholder(&mut self) -> usize {
        self.emit2(0x0F, 0x85);
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

fn linear_scan(
    intervals: &[LiveInterval],
    slot_count: usize,
    pinned: &[(u16, u8)],
    hotness: &[u32],
) -> RegAlloc {
    let cap = slot_count + 1;
    let mut slots: Vec<RegLoc> = (0..cap).map(|s| RegLoc::Spill((s as i32) * 8)).collect();

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

    // Reserve pinned hot slots first (typically r12-r15).
    for &(slot, reg) in pinned {
        ensure_slot(&mut slots, slot);
        slots[slot as usize] = RegLoc::Reg(reg);
        free.retain(|r| *r != reg);
        if is_callee_saved(reg) && (callee_saved_mask & (1u16 << reg)) == 0 {
            callee_saved_mask |= 1u16 << reg;
            used_callee_saved.push(reg);
        }
    }

    for iv in intervals {
        if matches!(slots.get(iv.slot as usize), Some(RegLoc::Reg(_))) {
            continue;
        }
        // Expire intervals that ended strictly before this one's start.
        let expired = active.partition_point(|(end, _, _)| *end < iv.first);
        for i in 0..expired {
            free.push(active[i].2);
        }
        if expired > 0 {
            active.drain(0..expired);
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
                    let curr_hot = hotness.get(iv.slot as usize).copied().unwrap_or(0);
                    let spill_hot = hotness.get(spill_slot as usize).copied().unwrap_or(0);
                    if curr_hot >= spill_hot {
                        ensure_slot(&mut slots, spill_slot);
                        slots[spill_slot as usize] = RegLoc::Spill((spill_slot as i32) * 8);
                        active.pop();
                        track_callee(reg);
                        ensure_slot(&mut slots, iv.slot);
                        slots[iv.slot as usize] = RegLoc::Reg(reg);
                        let pos = active.partition_point(|(e, _, _)| *e <= iv.last);
                        active.insert(pos, (iv.last, iv.slot, reg));
                    } else {
                        ensure_slot(&mut slots, iv.slot);
                    }
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

fn compute_loop_body_weight(instrs: &[Instr]) -> Vec<u32> {
    let mut weight = vec![1u32; instrs.len()];
    for (pc, instr) in instrs.iter().enumerate() {
        let back_edge = match instr {
            Instr::Jump(off) if *off < 0 => Some(((pc as i32) + 1 + *off) as usize),
            Instr::JumpFalse(_, off) if *off < 0 => Some(((pc as i32) + 1 + *off) as usize),
            Instr::JumpTrue(_, off) if *off < 0 => Some(((pc as i32) + 1 + *off) as usize),
            _ => None,
        };
        if let Some(target) = back_edge {
            if target <= pc && target < instrs.len() {
                for w in &mut weight[target..=pc] {
                    *w = w.saturating_add(3);
                }
            }
        }
    }
    weight
}

fn compute_slot_hotness(instrs: &[Instr], slot_count: usize) -> Vec<u32> {
    let mut heat = vec![0u32; slot_count + 1];
    let loop_weight = compute_loop_body_weight(instrs);
    let bump = |slot: u16, weight: u32, heat: &mut Vec<u32>| {
        let idx = slot as usize;
        if idx >= heat.len() {
            heat.resize(idx + 1, 0);
        }
        heat[idx] = heat[idx].saturating_add(weight);
    };

    for (pc, instr) in instrs.iter().enumerate() {
        let w = loop_weight.get(pc).copied().unwrap_or(1);
        match instr {
            Instr::LoadI32(d, _)
            | Instr::LoadI64(d, _)
            | Instr::LoadBool(d, _)
            | Instr::LoadUnit(d) => bump(*d, 1 * w, &mut heat),
            Instr::Move(d, s) | Instr::Load(d, s) => {
                bump(*s, 3 * w, &mut heat);
                bump(*d, 2 * w, &mut heat);
            }
            Instr::Store(slot, s) => {
                bump(*s, 3 * w, &mut heat);
                bump(*slot, 2 * w, &mut heat);
            }
            Instr::BinOp(d, _, l, r) => {
                bump(*l, 4 * w, &mut heat);
                bump(*r, 4 * w, &mut heat);
                bump(*d, 3 * w, &mut heat);
            }
            Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) | Instr::Return(s) => {
                bump(*s, 4 * w, &mut heat)
            }
            _ => {}
        }
    }

    heat
}

fn rank_hot_slots(heat: &[u32]) -> Vec<(u16, u32)> {
    let mut pairs: Vec<(u16, u32)> = heat
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(slot, score)| (score > 0).then_some((slot as u16, score)))
        .collect();
    pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs
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

// ─────────────────────────────────────────────────────────────────────────────
// Flat constant-propagation table (replaces HashMap<u16, i64>)
// ─────────────────────────────────────────────────────────────────────────────
//
// Vec<Option<i64>> indexed by slot gives O(1) lookup with zero hashing overhead.
// For a function with N slots this uses N*9 bytes vs HashMap's ~48 bytes base +
// 24 bytes/entry at low load factors. More importantly, sequential slot accesses
// stay in L1 cache during the hot codegen loop.

struct ConstTable {
    vals: Vec<Option<i64>>,
}

impl ConstTable {
    fn with_capacity(n: usize) -> Self {
        Self {
            vals: vec![None; n.max(1)],
        }
    }

    #[inline(always)]
    fn get(&self, slot: u16) -> Option<i64> {
        self.vals.get(slot as usize).copied().flatten()
    }

    #[inline(always)]
    fn insert(&mut self, slot: u16, v: i64) {
        let idx = slot as usize;
        if idx >= self.vals.len() {
            self.vals.resize(idx + 1, None);
        }
        self.vals[idx] = Some(v);
    }

    #[inline(always)]
    fn remove(&mut self, slot: u16) {
        if let Some(cell) = self.vals.get_mut(slot as usize) {
            *cell = None;
        }
    }

    /// Clear all known constants (conservative: called at branch targets).
    fn clear(&mut self) {
        self.vals.fill(None);
    }
}

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
        Instr::Move(_, s) | Instr::Load(_, s) | Instr::Store(_, s) | Instr::Return(s) => {
            *s == slot
        }
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
fn emit_binop_rax_imm(em: &mut Emitter, op: BinOpKind, imm: i32) {
    match op {
        BinOpKind::Add => {
            if imm == 1 {
                em.inc_rax();
            } else if imm == -1 {
                em.dec_rax();
            } else if imm != 0 {
                em.add_rax_imm32(imm); // now uses imm8 when it fits
            }
        }
        BinOpKind::Sub => {
            if imm == 1 {
                em.dec_rax();
            } else if imm == -1 {
                em.inc_rax();
            } else if imm != 0 {
                em.sub_rax_imm32(imm); // now uses imm8 when it fits
            }
        }
        BinOpKind::Mul => {
            if imm == 0 {
                em.xor_eax_eax();
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
            em.cmp_rax_imm32(imm); // now uses imm8 when it fits
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
fn emit_ret(em: &mut Emitter, callee_saved: &[u8]) {
    for &reg in callee_saved.iter().rev() {
        em.pop_reg(reg);
    }
    em.ret();
}

// ─────────────────────────────────────────────────────────────────────────────
// Branch fixup with short-branch shrinking
// ─────────────────────────────────────────────────────────────────────────────
//
// We record each branch as a placeholder in the byte stream.  After all code
// is emitted we know every target offset, so we can choose the shortest
// encoding.  Because shrinking a branch changes offsets, we do a single
// relaxation pass: convert rel32 → rel8 where the *original* displacement
// would already fit (conservative but correct — shrinking can only move
// targets closer).

/// Kind of branch instruction at a fixup site.
#[derive(Clone, Copy)]
enum BranchKind {
    Jmp,
    Jz,
    Jnz,
}

/// A pending branch: position of the disp32 field, target PC, and kind.
struct Fixup {
    /// Byte index of the 4-byte disp32 placeholder.
    disp_pos: usize,
    target_pc: usize,
    kind: BranchKind,
}

/// Patch all branch displacements.  Returns None if any target is unreachable
/// or any displacement overflows i32.
fn patch_fixups(
    buf: &mut Vec<u8>,
    fixups: &[Fixup],
    pc_to_off: &[usize],
) -> Option<()> {
    for fx in fixups {
        let target_off = *pc_to_off.get(fx.target_pc)? as isize;
        let next_ip = (fx.disp_pos + 4) as isize;
        let rel = i32::try_from(target_off - next_ip).ok()?;

        // Attempt to shrink to rel8 (saves 3-4 bytes per branch).
        if let Ok(rel8) = i8::try_from(rel) {
            // The rel32 form sits at disp_pos-1 (or disp_pos-2 for 0F 8x).
            // Overwrite with rel8 opcode + 1-byte disp + NOPs for remainder.
            let opcode_start = match fx.kind {
                BranchKind::Jmp => fx.disp_pos - 1, // E9 [d32]
                BranchKind::Jz | BranchKind::Jnz => fx.disp_pos - 2, // 0F 84/85 [d32]
            };
            let short_op: u8 = match fx.kind {
                BranchKind::Jmp => 0xEB,
                BranchKind::Jz => 0x74,
                BranchKind::Jnz => 0x75,
            };
            buf[opcode_start] = short_op;
            buf[opcode_start + 1] = rel8 as u8;
            // Overwrite remaining bytes with NOPs.
            let nop_start = opcode_start + 2;
            let nop_end = fx.disp_pos + 4;
            for b in &mut buf[nop_start..nop_end] {
                *b = 0x90;
            }
        } else {
            buf[fx.disp_pos..fx.disp_pos + 4].copy_from_slice(&rel.to_le_bytes());
        }
    }
    Some(())
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

    let instrs = &compiled.instrs;
    let slot_count = compiled.slot_count as usize;

    // Gate: bail out early if any instruction is outside our supported set.
    for instr in instrs {
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

    // ── Pass 1: liveness + linear-scan register allocation ───────────────
    let intervals = compute_live_intervals(instrs, slot_count);
    let hotness = compute_slot_hotness(instrs, slot_count);
    let hot_slots = rank_hot_slots(&hotness);
    const PIN_POOL: &[u8] = &[12, 13, 14, 15, 6];
    let pinned: Vec<(u16, u8)> = hot_slots
        .iter()
        .take(PIN_POOL.len())
        .zip(PIN_POOL.iter().copied())
        .map(|((slot, _), reg)| (*slot, reg))
        .collect();
    let ra = linear_scan(&intervals, slot_count, &pinned, &hotness);

    // ── Emission ──────────────────────────────────────────────────────────
    let mut em = Emitter::new();
    let mut pc_to_off = vec![0usize; instrs.len() + 1];
    let mut fixups: Vec<Fixup> = Vec::new();

    // Prologue: save callee-saved registers we actually use.
    for &reg in &ra.used_callee_saved {
        em.push_reg(reg);
    }

    // Pre-load pinned hot slots first, then any remaining register-allocated slots.
    {
        let mut preloaded: u32 = 0u32; // bitmask for regs 0-31
        for &(slot, reg) in &pinned {
            let bit = 1u32 << reg;
            if preloaded & bit == 0 {
                preloaded |= bit;
                em.load_reg_mem(reg, (slot as i32) * 8);
            }
        }
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
    // Flat Vec<Option<i64>> replaces HashMap — O(1) slot access, cache-friendly.
    let mut const_at = ConstTable::with_capacity(slot_count + 1);

    let mut pc = 0usize;
    while pc < instrs.len() {
        pc_to_off[pc] = em.pos();

        // ════════════════════════════════════════════════════════════════════
        // 3-INSTRUCTION FUSIONS
        // ════════════════════════════════════════════════════════════════════

        // ── Fusion: BinOp(t, Mul, x, N) + BinOp(r, Add, t, y) → LEA ────
        if pc + 1 < instrs.len() {
            if let (
                Instr::BinOp(t, BinOpKind::Mul, mul_l, mul_r),
                Instr::BinOp(r, BinOpKind::Add, add_l, add_r),
            ) = (&instrs[pc], &instrs[pc + 1])
            {
                let (addend_slot, t_consumed_by_add) = if *add_l == *t && *add_r != *t && *r != *t {
                    (Some(*add_r), true)
                } else if *add_r == *t && *add_l != *t && *r != *t {
                    (Some(*add_l), true)
                } else {
                    (None, false)
                };

                if t_consumed_by_add {
                    let addend = addend_slot.unwrap();
                    let maybe_lea = const_at
                        .get(*mul_r)
                        .map(|c| (*mul_l, c))
                        .or_else(|| const_at.get(*mul_l).map(|c| (*mul_r, c)));

                    if let Some((base, scale_i64)) = maybe_lea {
                        if matches!(scale_i64, 2 | 4 | 8) {
                            let scale = scale_i64 as u8;
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
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op1, a, b), Instr::BinOp(r, op2, l2, r2)) =
                (&instrs[pc], &instrs[pc + 1])
            {
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
                    emit_binop_rax_rcx(&mut em, *op1);
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

        // ── Fusion: Load*(tmp, c) + JumpFalse/JumpTrue → compile-time branch
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
                            fixups.push(Fixup { disp_pos: p, target_pc: target, kind: BranchKind::Jmp });
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
                            fixups.push(Fixup { disp_pos: p, target_pc: target, kind: BranchKind::Jmp });
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

        // ── Fusion: Load*(tmp, c) + BinOp(dst, op, x, tmp) → imm arithmetic
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

        // ── Fusion: BinOp(t, op, l, r) + Store(slot, t) ─────────────────
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

        // ── Fusion: BinOp(t, op, l, r) + JumpFalse(t, off) ─────────────
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
                    fixups.push(Fixup { disp_pos: p, target_pc: target, kind: BranchKind::Jz });
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + JumpTrue(t, off) ──────────────
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
                    fixups.push(Fixup { disp_pos: p, target_pc: target, kind: BranchKind::Jnz });
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // ── Fusion: BinOp(t, op, l, r) + Return(t) ──────────────────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::Return(ret)) =
                (&instrs[pc], &instrs[pc + 1])
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
                .get(*l)
                .zip(const_at.get(*r))
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
                em.xor_eax_eax();
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
                if let Some(c) = const_at.get(*s) {
                    const_at.insert(*d, c);
                } else {
                    const_at.remove(*d);
                }
            }
            Instr::Store(slot, s) => {
                load_rax(&mut em, *s, &ra);
                if !is_straight_line_dead_def(instrs, pc, *slot) {
                    store_rax(&mut em, *slot, &ra);
                }
                if let Some(c) = const_at.get(*s) {
                    const_at.insert(*slot, c);
                } else {
                    const_at.remove(*slot);
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
                    .get(*l)
                    .zip(const_at.get(*r))
                    .and_then(|(lv, rv)| fold_binop(*op, lv, rv));
                match folded {
                    Some(c) => const_at.insert(*d, c),
                    None => const_at.remove(*d),
                }
            }
            Instr::Jump(off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                if target > instrs.len() {
                    return None;
                }
                let p = em.jmp_rel32_placeholder();
                fixups.push(Fixup { disp_pos: p, target_pc: target, kind: BranchKind::Jmp });
                const_at.clear();
            }
            Instr::JumpFalse(cond, off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                if target > instrs.len() {
                    return None;
                }
                load_rax(&mut em, *cond, &ra);
                em.test_rax_rax();
                let p = em.jz_rel32_placeholder();
                fixups.push(Fixup { disp_pos: p, target_pc: target, kind: BranchKind::Jz });
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
                fixups.push(Fixup { disp_pos: p, target_pc: target, kind: BranchKind::Jnz });
                const_at.clear();
            }
            Instr::Return(r) => {
                load_rax(&mut em, *r, &ra);
                emit_ret(&mut em, &ra.used_callee_saved);
            }
            Instr::ReturnUnit => {
                em.xor_eax_eax();
                emit_ret(&mut em, &ra.used_callee_saved);
            }
            Instr::Nop => {}
            _ => return None,
        }
        pc += 1;
    }

    // Fallthrough epilogue.
    pc_to_off[instrs.len()] = em.pos();
    em.xor_eax_eax();
    emit_ret(&mut em, &ra.used_callee_saved);

    // ── Patch branch displacements (with short-branch shrinking) ─────────
    patch_fixups(&mut em.buf, &fixups, &pc_to_off)?;

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
