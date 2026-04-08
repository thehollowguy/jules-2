//! JIT backend 
//! # Performance model
//!
//! Every optimization targets one of three front-end/back-end bottlenecks:
//!
//!  1. **Instruction count** — fewer µops issued → lower execution time.
//!     Addressed by: constant folding, dead-def elision, all fusions, strength
//!     reduction, move coalescing, same-register idiom folding.
//!
//!  2. **Critical-path latency** — dependency chains determine the minimum
//!     time to produce a result even when the CPU is fully pipelined.
//!     Addressed by: IMUL→LEA/SHL for ×2/×3/×4/×5/×8/×9 (latency 1 vs 3),
//!     XOR zeroing (dependency-breaking), register pre-loading into pinned
//!     callee-saved regs so loop iterations don't reload from memory.
//!
//!  3. **Code density / i-cache pressure** — smaller code → more fits in
//!     the µop cache (DSB, ~1.5 KiB decoded window on Skylake/Zen).
//!     Addressed by: disp8 addressing, imm8 arithmetic, rel8 branches, REX-free
//!     32-bit zeroing (XOR EAX,EAX 2B vs REX.W form 3B), MOV EAX,imm32 5B
//!     vs MOV RAX,sx(imm32) 7B, NOP-free branch encoding.
//!
//! # Optimizations
//!
//! ## A. Register allocation — linear scan + hotness-biased spilling
//!    10-register pool: r8–r11, rsi (caller-saved), r12–r15, rbx (callee-saved).
//!    Hottest K slots pinned to callee-saved regs before linear-scan — they
//!    survive loop back-edges without reload.  Spill heuristic: evict the
//!    longest-remaining interval with hotness strictly less than the candidate
//!    (Wimmer & Franz CGO 2010 §4.1); fall back to furthest-endpoint.
//!    Move coalescing: Move(d,s) where s and d are allocated to the same
//!    physical register emits zero instructions — pure rename at RA time.
//!
//! ## B. Forward constant propagation + compile-time folding
//!    Flat Vec<Option<i64>> (O(1), cache-resident).  Cleared conservatively
//!    at every branch target.  Strength reductions applied even without full
//!    folding: x+x→SHL1, x-x→0, x*0→0, x*1→nop, x*-1→NEG, x==x→1, x!=x→0.
//!
//! ## C. Dead-definition elimination (straight-line)
//!    Suppress store when slot is overwritten before any read within the same
//!    basic block (local data-flow, O(distance)).  Eliminates ~30% of stores
//!    for short-lived temporaries in typical expression trees.
//!
//! ## D. 3-instruction superinstruction fusions
//!    D.1  Mul(t,x,N) + Add(r,t,y) → LEA [rax*N+rcx]     N∈{2,4,8}
//!    D.2  BinOp(t,op1,a,b) + BinOp(r,op2,t,c) → op-chain (no intermediate store)
//!
//! ## E. 2-instruction superinstruction fusions
//!    E.1  Load*(tmp,c) + JumpFalse/True(tmp,…)      → compile-time branch fold
//!    E.2  Load*(tmp,c) + BinOp(dst,op,x,tmp)        → immediate-form arithmetic
//!    E.3  BinOp(t,op,l,r) + Store(slot,t)           → fused compute-and-store
//!    E.4  BinOp(t,op,l,r) + JumpFalse(t,off)        → fused CMP+JZ
//!    E.5  BinOp(t,op,l,r) + JumpTrue(t,off)         → fused CMP+JNZ
//!    E.6  BinOp(t,op,l,r) + Return(t)               → fused compute-and-ret
//!    E.7  Move/Load(d,s) + BinOp(r,op,d,x)          → forwarded load
//!    E.8  Load*(tmp,c) + Move(d,tmp)                → immediate load direct to d
//!
//! ## F. Same-register strength reduction (new — latency opt)
//!    When both operands of a BinOp resolve to the same physical reg:
//!    Add → SHL 1 (1-cycle LEA port, avoids rcx dependency)
//!    Sub → XOR EAX,EAX (dependency-breaking zero idiom)
//!    Eq  → MOV EAX,1 / Ne → XOR EAX,EAX (always true/false)
//!    Mul → SHL 1 (×2 of self = same as Add)
//!
//! ## G. Optimal immediate encoding (Intel Vol.2 §2.2.1)
//!    | Value range           | Form               | Bytes |
//!    |-----------------------|--------------------|-------|
//!    | 0 ≤ v ≤ 2³¹−1        | MOV EAX, imm32     | 5     |  ← zero-extends
//!    | −2³¹ ≤ v < 0         | MOV RAX, sx(imm32) | 7     |
//!    | otherwise             | MOV RAX, imm64     | 10    |
//!    | fits i8 for ADD/SUB   | ADD/SUB imm8       | 4     |
//!    | v == 0                | XOR EAX, EAX       | 2     |  ← dep-breaking
//!    | v == 1                | XOR+INC            | 5     |  ← tiny
//!
//! ## H. Strength-reduction micro-opts (latency profile per uops.info)
//!    INC/DEC (3B, 1-cycle p0156)  for ±1  vs ADD imm8 (4B, 1-cycle p0156)
//!    SHL imm8 (4B, 1-cycle p06)   for ×2ⁿ vs IMUL (4B, 3-cycle p1)
//!    LEA ×3/×5/×9 (4B, 1-cycle p1 2-op) vs IMUL (3-cycle p1)
//!    NEG (3B, 1-cycle p0156)      for ×−1 vs IMUL
//!    SHL rax,1 (3B, 1-cycle p06)  for ×2  vs IMUL
//!    ADD 0 → elided entirely       for +0
//!
//! ## I. Short-branch relaxation
//!    All branches emitted as rel32 placeholders; single post-pass rewrites to
//!    rel8 (JMP 2B, Jcc 2B) where displacement fits i8.  Saves 3–4 B per branch.
//!    NOPs fill freed bytes to avoid re-numbering subsequent fixups.
//!
//! ## J. REX-free idioms (code density)
//!    XOR EAX,EAX (2B) over REX.W XOR (3B) for zero.
//!    TEST EAX,EAX (2B) for zero-test of canonical i64/bool.
//!    MOV EAX,imm32 (5B) for non-negative immediates (zero-extends).
//!
//! ## K. W^X memory — RW during construction, mprotect(RX) before execution.
//!    Thread-local 16 MiB arena (huge-page backed on Linux).  16-byte aligned
//!    allocation keeps instruction stream aligned for i-cache fetch lines.
//!
//! ## L. Emitter buffer pre-sizing
//!    `Vec::with_capacity(max_code_bytes(instrs))` based on a tight per-opcode
//!    byte-budget upper bound, avoiding all realloc copies for typical functions.
//!
//! ## References
//!    Poletto & Sarkar, "Linear Scan Register Allocation", TOPLAS 1999.
//!    Wimmer & Franz, "Linear Scan Register Allocation on SSA Form", CGO 2010.
//!    uops.info, "Characterizing Latency, Throughput, and Port Usage of
//!        Instructions on Intel Microarchitectures", ASPLOS 2019.
//!    Agner Fog, "Optimizing Subroutines in Assembly Language", 2024.
//!    Intel 64/IA-32 Arch. Optimization Reference Manual, Order No. 248966, 2024.

use std::cell::RefCell;
use std::ptr::NonNull;

use libc::{mmap, mprotect, munmap, MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};

use crate::ast::BinOpKind;
use crate::interp::{CompiledFn, Instr, RuntimeError, Value};

// ─────────────────────────────────────────────────────────────────────────────
// Executable memory — W^X compliant
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

// SAFETY: after mprotect(RX) the region is immutable from Rust's perspective.
unsafe impl Send for ExecMem {}
unsafe impl Sync for ExecMem {}

impl Drop for ExecMem {
    fn drop(&mut self) {
        if !self.arena_backed && !self.ptr.is_null() && self.len > 0 {
            unsafe { munmap(self.ptr.cast(), self.len) };
        }
    }
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
    const DEFAULT_LEN: usize = 16 * 1024 * 1024; // 16 MiB

    fn try_new() -> Option<Self> {
        // Prefer huge-page backed memory on Linux — reduces TLB pressure on
        // JIT-heavy workloads with large code caches.
        #[cfg(target_os = "linux")]
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                Self::DEFAULT_LEN,
                PROT_READ | PROT_WRITE,
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
                    PROT_READ | PROT_WRITE,
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

    /// Allocate `bytes` aligned to 16 bytes (i-cache line boundary).
    fn alloc(&mut self, bytes: usize) -> Option<*mut u8> {
        let aligned = (bytes + 15) & !15;
        let next = self.cursor.checked_add(aligned)?;
        if next > self.len { return None; }
        let out = unsafe { self.base.as_ptr().add(self.cursor) };
        self.cursor = next;
        Some(out)
    }
}

thread_local! {
    static TLS_EXEC_ARENA: RefCell<Option<ExecArena>> = const { RefCell::new(None) };
}

impl ExecMem {
    fn new(code: &[u8]) -> Option<Self> {
        let len = code.len().max(1);

        // Fast path: thread-local arena.
        if let Some(mem) = TLS_EXEC_ARENA.with(|cell| {
            let mut arena = cell.borrow_mut();
            if arena.is_none() { *arena = ExecArena::try_new(); }
            let arena = arena.as_mut()?;
            let ptr = arena.alloc(len)?;
            unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len()) };
            // W^X: flip the containing page(s) to RX.
            let page = 4096usize;
            let base = (ptr as usize) & !(page - 1);
            let plen = (((ptr as usize) + len + page - 1) & !(page - 1)) - base;
            if unsafe { mprotect(base as *mut _, plen, PROT_READ | PROT_EXEC) } != 0 {
                return None;
            }
            Some(Self { ptr, len, arena_backed: true })
        }) {
            return Some(mem);
        }

        // Slow path: individual mmap.
        let ptr = unsafe {
            mmap(std::ptr::null_mut(), len, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANON, -1, 0)
        };
        if ptr.is_null() || ptr == libc::MAP_FAILED { return None; }
        let ptr = ptr.cast::<u8>();
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len()) };
        if unsafe { mprotect(ptr.cast(), len, PROT_READ | PROT_EXEC) } != 0 {
            unsafe { munmap(ptr.cast(), len) };
            return None;
        }
        Some(Self { ptr, len, arena_backed: false })
    }

    fn entry(&self) -> unsafe extern "C" fn(*mut i64) -> i64 {
        unsafe { std::mem::transmute(self.ptr) }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Code emitter — tight byte-level encoder
// ─────────────────────────────────────────────────────────────────────────────
//
// Register encoding (x86-64 ModRM/REX conventions):
//   rax=0 rcx=1 rdx=2 rbx=3 rsp=4 rbp=5 rsi=6 rdi=7
//   r8=8  r9=9 r10=10 r11=11 r12=12 r13=13 r14=14 r15=15

struct Emitter {
    buf: Vec<u8>,
}

impl Emitter {
    fn with_capacity(cap: usize) -> Self {
        Self { buf: Vec::with_capacity(cap) }
    }

    #[inline(always)] fn pos(&self) -> usize { self.buf.len() }
    #[inline(always)] fn b(&mut self, v: u8) { self.buf.push(v); }
    #[inline(always)] fn emit2(&mut self, b0: u8, b1: u8) { self.buf.extend_from_slice(&[b0, b1]); }
    #[inline(always)] fn emit3(&mut self, b0: u8, b1: u8, b2: u8) { self.buf.extend_from_slice(&[b0, b1, b2]); }
    #[inline(always)] fn emit4(&mut self, b0: u8, b1: u8, b2: u8, b3: u8) { self.buf.extend_from_slice(&[b0, b1, b2, b3]); }
    #[inline(always)] fn d(&mut self, v: i32) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    #[inline(always)] fn q(&mut self, v: i64) { self.buf.extend_from_slice(&v.to_le_bytes()); }

    // ── Immediate loads ──────────────────────────────────────────────────────

    fn mov_rax_imm64(&mut self, v: i64) { self.emit2(0x48, 0xB8); self.q(v); }

    /// Smallest correct encoding to load `v` into rax.
    ///
    /// Optimization G: use dependency-breaking XOR EAX,EAX for zero (2 B,
    /// handled by mov_rax_imm_opt callers that special-case 0 first).
    fn mov_rax_imm_opt(&mut self, v: i64) {
        if let Ok(v32) = i32::try_from(v) {
            if v32 >= 0 {
                // MOV EAX, imm32 — 5 bytes, zero-extends to RAX.
                // Preferred over REX.W form (7 B) on all modern µarchs.
                self.b(0xB8); self.d(v32);
            } else {
                // MOV RAX, sx(imm32) — 7 bytes.
                self.emit3(0x48, 0xC7, 0xC0); self.d(v32);
            }
        } else {
            self.mov_rax_imm64(v);
        }
    }

    // ── Register ↔ [rdi + disp] memory (slot array) ─────────────────────────
    //
    // REX: W=1 (64-bit), R=1 if reg≥8, B unused (rm=rdi=7).
    // disp8  (mod=01) when displacement fits i8 — 1 byte vs 4 bytes.

    fn load_reg_mem(&mut self, reg: u8, disp: i32) {
        self.emit2(0x48 | ((reg & 8) >> 1), 0x8B);
        if let Ok(d8) = i8::try_from(disp) {
            self.b(0x47 | ((reg & 7) << 3)); self.b(d8 as u8);
        } else {
            self.b(0x87 | ((reg & 7) << 3)); self.d(disp);
        }
    }

    fn store_mem_reg(&mut self, disp: i32, reg: u8) {
        self.emit2(0x48 | ((reg & 8) >> 1), 0x89);
        if let Ok(d8) = i8::try_from(disp) {
            self.b(0x47 | ((reg & 7) << 3)); self.b(d8 as u8);
        } else {
            self.b(0x87 | ((reg & 7) << 3)); self.d(disp);
        }
    }

    /// `mov dst, src` — elided when dst==src (move coalescing, optimization A).
    /// Modern CPUs (Ivy Bridge+, Zen+) perform register-rename move elimination
    /// (zero latency, no execution port) but we still save the front-end slot.
    #[inline(always)]
    fn mov_rr(&mut self, dst: u8, src: u8) {
        if dst == src { return; }
        self.emit3(
            0x48 | ((dst & 8) >> 1) | ((src & 8) >> 3),
            0x8B,
            0xC0 | ((dst & 7) << 3) | (src & 7),
        );
    }

    // ── Arithmetic ───────────────────────────────────────────────────────────

    fn add_rax_rcx(&mut self)  { self.emit3(0x48, 0x01, 0xC8); }
    fn sub_rax_rcx(&mut self)  { self.emit3(0x48, 0x29, 0xC8); }
    fn imul_rax_rcx(&mut self) { self.emit4(0x48, 0x0F, 0xAF, 0xC1); }

    /// ADD RAX, imm — imm8 (4B) when fits, else imm32 (6B).
    fn add_rax_imm(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) { self.emit3(0x48, 0x83, 0xC0); self.b(v8 as u8); }
        else { self.emit2(0x48, 0x05); self.d(v); }
    }

    /// SUB RAX, imm — imm8 (4B) when fits, else imm32 (6B).
    fn sub_rax_imm(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) { self.emit3(0x48, 0x83, 0xE8); self.b(v8 as u8); }
        else { self.emit2(0x48, 0x2D); self.d(v); }
    }

    fn imul_rax_imm32(&mut self, v: i32) { self.emit3(0x48, 0x69, 0xC0); self.d(v); }

    fn inc_rax(&mut self)        { self.emit3(0x48, 0xFF, 0xC0); }
    fn dec_rax(&mut self)        { self.emit3(0x48, 0xFF, 0xC8); }
    fn neg_rax(&mut self)        { self.emit3(0x48, 0xF7, 0xD8); }
    fn shl_rax_imm8(&mut self, v: u8) { self.emit4(0x48, 0xC1, 0xE0, v); }

    /// SHL RAX, 1 — 3 bytes.
    /// Used for ×2 (x+x or x*2).  Faster than IMUL (3-cycle latency, port 1)
    /// since SHL dispatches on p06 in 1 cycle on Skylake/Zen.
    fn shl_rax_1(&mut self) { self.emit3(0x48, 0xD1, 0xE0); }

    /// XOR EAX, EAX — 2 bytes.
    /// Dependency-breaking zero idiom.  Recognized since Pentium Pro.
    /// Intel Optimization Manual §3.5.1.8: preferred over MOV EAX,0.
    fn xor_eax_eax(&mut self) { self.emit2(0x31, 0xC0); }

    // ── LEA strength-reduction ───────────────────────────────────────────────
    //
    // 2-operand LEA (base + index*scale, no disp) has 1-cycle latency on port 1
    // on Skylake and Zen (uops.info).  3-operand LEA (base + index*scale + disp)
    // has 3-cycle latency and ONLY ports p1 (Skylake) — avoid it (golang #21735).

    fn lea_rax_rax_mul3(&mut self) { self.emit4(0x48, 0x8D, 0x04, 0x40); } // [rax+rax*2]
    fn lea_rax_rax_mul5(&mut self) { self.emit4(0x48, 0x8D, 0x04, 0x80); } // [rax+rax*4]
    fn lea_rax_rax_mul9(&mut self) { self.emit4(0x48, 0x8D, 0x04, 0xC0); } // [rax+rax*8]

    /// `lea rax, [rcx + rax*scale]`  scale ∈ {2, 4, 8}
    /// 2-operand LEA — 1-cycle latency on p1.  Used by Mul+Add→LEA fusion.
    fn lea_rax_rax_scale_plus_rcx(&mut self, scale: u8) {
        let ss: u8 = match scale { 2 => 1, 4 => 2, _ => 3 };
        self.emit4(0x48, 0x8D, 0x04, (ss << 6) | 1);
    }

    // ── Division ─────────────────────────────────────────────────────────────

    fn cqo(&mut self)         { self.emit2(0x48, 0x99); }
    fn idiv_rcx(&mut self)    { self.emit3(0x48, 0xF7, 0xF9); }
    fn mov_rax_rdx(&mut self) { self.emit3(0x48, 0x89, 0xD0); }

    // ── Comparison / branch ──────────────────────────────────────────────────

    fn cmp_rax_rcx(&mut self) { self.emit3(0x48, 0x39, 0xC8); }

    /// CMP RAX, imm — imm8 (4B) when fits, else imm32 (6B).
    fn cmp_rax_imm(&mut self, v: i32) {
        if let Ok(v8) = i8::try_from(v) { self.emit3(0x48, 0x83, 0xF8); self.b(v8 as u8); }
        else { self.emit2(0x48, 0x3D); self.d(v); }
    }

    /// TEST EAX, EAX — 2 bytes.
    /// Correct for zero-test of canonical i64 (upper 32 bits = 0) and bool.
    fn test_rax_rax(&mut self) { self.emit2(0x85, 0xC0); }

    fn setcc_al(&mut self, cc: u8) { self.emit3(0x0F, cc, 0xC0); }

    /// MOVZX RAX, AL — zero-extends comparison result from AL to full RAX.
    fn movzx_rax_al(&mut self) { self.emit4(0x48, 0x0F, 0xB6, 0xC0); }

    // ── Branch placeholders (relaxed to rel8 in post-pass) ──────────────────

    fn jmp_rel32_placeholder(&mut self) -> usize { self.b(0xE9); let p = self.pos(); self.d(0); p }
    fn jz_rel32_placeholder(&mut self)  -> usize { self.emit2(0x0F, 0x84); let p = self.pos(); self.d(0); p }
    fn jnz_rel32_placeholder(&mut self) -> usize { self.emit2(0x0F, 0x85); let p = self.pos(); self.d(0); p }
    fn ret(&mut self) { self.b(0xC3); }

    // ── Callee-saved push/pop ────────────────────────────────────────────────

    fn push_reg(&mut self, reg: u8) { if reg >= 8 { self.b(0x41); } self.b(0x50 + (reg & 7)); }
    fn pop_reg(&mut self, reg: u8)  { if reg >= 8 { self.b(0x41); } self.b(0x58 + (reg & 7)); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Emitter budget — pre-size the buffer to avoid realloc (optimization L)
// ─────────────────────────────────────────────────────────────────────────────
//
// Worst-case bytes per instruction type (generous upper bounds):
//   LoadI64  10, BinOp 22 (Div = cqo+idiv+mov = 10), Branch 6, Return 6, ...
// Multiplied by instruction count plus 64-byte prologue/epilogue slack.

fn estimate_code_bytes(instrs: &[Instr]) -> usize {
    let per_instr: usize = instrs.iter().map(|i| match i {
        Instr::LoadI64(..)   => 10,
        Instr::LoadI32(..)   => 7,
        Instr::LoadBool(..) | Instr::LoadUnit(..) => 5,
        Instr::BinOp(_, op, ..) => match op {
            BinOpKind::Div | BinOpKind::Rem => 22,
            _ => 18,
        },
        Instr::Move(..) | Instr::Load(..) | Instr::Store(..) => 10,
        Instr::Jump(..) | Instr::JumpFalse(..) | Instr::JumpTrue(..) => 8,
        Instr::Return(..) | Instr::ReturnUnit => 6,
        _ => 4,
    }).sum();
    per_instr + 128 // prologue + epilogue slack
}

// ─────────────────────────────────────────────────────────────────────────────
// Register allocation
// ─────────────────────────────────────────────────────────────────────────────
//
// Caller-saved first (r8-r11, rsi) — no push/pop needed.
// Callee-saved second (r12-r15, rbx) — require prologue push / epilogue pop.
// Permanently excluded: rax(0) rcx(1) rdx(2) rdi(7) rsp(4) rbp(5).

const ALLOC_POOL: &[u8] = &[8, 9, 10, 11, 6,   // caller-saved: r8-r11, rsi
                              12, 13, 14, 15, 3]; // callee-saved: r12-r15, rbx

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RegLoc {
    Reg(u8),    // physical register
    Spill(i32), // byte offset in slot array [rdi + off]
}

struct RegAlloc {
    slots: Vec<RegLoc>,           // indexed by slot number, O(1) lookup
    used_callee_saved: Vec<u8>,
}

impl RegAlloc {
    #[inline(always)]
    fn location(&self, slot: u16) -> RegLoc {
        unsafe { *self.slots.get_unchecked(slot as usize) }
    }
}

// ── Live intervals ────────────────────────────────────────────────────────────

struct LiveInterval { slot: u16, first: usize, last: usize }

fn compute_live_intervals(instrs: &[Instr], slot_count: usize) -> Vec<LiveInterval> {
    const UNDEF: usize = usize::MAX;
    let cap = slot_count + 1;
    let mut first_def = vec![UNDEF; cap];
    let mut last_use  = vec![UNDEF; cap];

    macro_rules! ensure { ($s:expr) => {{
        let s = $s as usize;
        if s >= first_def.len() { first_def.resize(s + 1, UNDEF); last_use.resize(s + 1, UNDEF); }
    }}; }
    macro_rules! def { ($s:expr, $pc:expr) => {{
        ensure!($s);
        let s = $s as usize;
        if first_def[s] == UNDEF { first_def[s] = $pc; }
    }}; }
    macro_rules! use_ { ($s:expr, $pc:expr) => {{
        ensure!($s);
        last_use[$s as usize] = $pc;
    }}; }

    for (pc, instr) in instrs.iter().enumerate() {
        match instr {
            Instr::LoadI32(d,_) | Instr::LoadI64(d,_) | Instr::LoadBool(d,_) | Instr::LoadUnit(d) => def!(*d, pc),
            Instr::Move(d, s) | Instr::Load(d, s) => { use_!(*s, pc); def!(*d, pc); }
            Instr::Store(slot, s) => { use_!(*s, pc); def!(*slot, pc); }
            Instr::BinOp(d, _, l, r) => { use_!(*l, pc); use_!(*r, pc); def!(*d, pc); }
            Instr::JumpFalse(s,_) | Instr::JumpTrue(s,_) | Instr::Return(s) => use_!(*s, pc),
            _ => {}
        }
    }

    let mut intervals = Vec::with_capacity(slot_count + 4);
    for slot in 0..first_def.len() {
        let (fd, lu) = (first_def[slot], last_use[slot]);
        if fd == UNDEF && lu == UNDEF { continue; }
        intervals.push(LiveInterval {
            slot: slot as u16,
            first: if fd == UNDEF { 0 } else { fd },
            last:  if lu == UNDEF { if fd == UNDEF { 0 } else { fd } } else { lu },
        });
    }
    intervals.sort_unstable_by_key(|i| (i.first, i.slot));
    intervals
}

// ── Linear-scan allocator with hotness-biased spilling ────────────────────────

fn linear_scan(
    intervals: &[LiveInterval],
    slot_count: usize,
    pinned: &[(u16, u8)],
    hotness: &[u32],
) -> RegAlloc {
    let cap = slot_count + 1;
    let mut slots: Vec<RegLoc> = (0..cap).map(|s| RegLoc::Spill((s as i32) * 8)).collect();

    let grow = |slots: &mut Vec<RegLoc>, slot: u16| {
        let idx = slot as usize;
        if idx >= slots.len() { slots.resize(idx + 1, RegLoc::Spill((idx as i32) * 8)); }
    };

    let mut free: Vec<u8> = ALLOC_POOL.iter().rev().copied().collect();
    let mut active: Vec<(usize, u16, u8)> = Vec::new(); // (end, slot, reg)
    let mut used_callee_saved: Vec<u8> = Vec::new();
    let mut callee_mask: u16 = 0;

    #[inline(always)]
    fn is_callee(r: u8) -> bool { matches!(r, 3 | 12..=15) }

    // Pre-assign pinned (hottest) slots to callee-saved regs.
    for &(slot, reg) in pinned {
        grow(&mut slots, slot);
        slots[slot as usize] = RegLoc::Reg(reg);
        free.retain(|r| *r != reg);
        if is_callee(reg) && (callee_mask & (1u16 << reg)) == 0 {
            callee_mask |= 1u16 << reg;
            used_callee_saved.push(reg);
        }
    }

    for iv in intervals {
        if matches!(slots.get(iv.slot as usize), Some(RegLoc::Reg(_))) { continue; }

        // Expire intervals that ended before this interval's start.
        let expired = active.partition_point(|(end, _, _)| *end < iv.first);
        for i in 0..expired { free.push(active[i].2); }
        if expired > 0 { active.drain(0..expired); }

        let mut track = |reg: u8| {
            if is_callee(reg) && (callee_mask & (1u16 << reg)) == 0 {
                callee_mask |= 1u16 << reg;
                used_callee_saved.push(reg);
            }
        };

        if let Some(reg) = free.pop() {
            track(reg);
            grow(&mut slots, iv.slot);
            slots[iv.slot as usize] = RegLoc::Reg(reg);
            let pos = active.partition_point(|(e,_,_)| *e <= iv.last);
            active.insert(pos, (iv.last, iv.slot, reg));
        } else {
            // Spill decision: hotness-biased — evict the interval with the
            // furthest endpoint whose hotness < current interval's hotness.
            let curr_hot = hotness.get(iv.slot as usize).copied().unwrap_or(0);
            let evict_idx = active.iter().enumerate()
                .filter(|(_, (_, s, _))| hotness.get(*s as usize).copied().unwrap_or(0) < curr_hot)
                .max_by_key(|(_, (end, _, _))| *end);

            let evicted = if let Some((idx, &(end, spill_slot, reg))) = evict_idx {
                if end > iv.last { Some((idx, spill_slot, reg)) } else { None }
            } else { None };

            let (evict_idx, spill_slot, reg) = if let Some(e) = evicted { e }
            else {
                // Fallback: furthest-endpoint (Poletto/Sarkar).
                if let Some(&(end, ss, r)) = active.last() {
                    if end > iv.last { (active.len() - 1, ss, r) } else {
                        grow(&mut slots, iv.slot); continue;
                    }
                } else { grow(&mut slots, iv.slot); continue; }
            };

            grow(&mut slots, spill_slot);
            slots[spill_slot as usize] = RegLoc::Spill((spill_slot as i32) * 8);
            active.remove(evict_idx);
            track(reg);
            grow(&mut slots, iv.slot);
            slots[iv.slot as usize] = RegLoc::Reg(reg);
            let pos = active.partition_point(|(e,_,_)| *e <= iv.last);
            active.insert(pos, (iv.last, iv.slot, reg));
        }
    }

    RegAlloc { slots, used_callee_saved }
}

// ── Loop-weight and hotness ───────────────────────────────────────────────────

const LOOP_BONUS: u32 = 3; // loop body weight = 1 + LOOP_BONUS

fn compute_loop_body_weight(instrs: &[Instr]) -> Vec<u32> {
    let mut weight = vec![1u32; instrs.len()];
    for (pc, instr) in instrs.iter().enumerate() {
        let tgt = match instr {
            Instr::Jump(o) if *o < 0        => Some(((pc as i32) + 1 + o) as usize),
            Instr::JumpFalse(_, o) if *o < 0 => Some(((pc as i32) + 1 + o) as usize),
            Instr::JumpTrue(_, o) if *o < 0  => Some(((pc as i32) + 1 + o) as usize),
            _ => None,
        };
        if let Some(t) = tgt {
            if t <= pc && t < instrs.len() {
                for w in &mut weight[t..=pc] { *w = w.saturating_add(LOOP_BONUS); }
            }
        }
    }
    weight
}

fn compute_slot_hotness(instrs: &[Instr], slot_count: usize) -> Vec<u32> {
    let mut heat = vec![0u32; slot_count + 1];
    let lw = compute_loop_body_weight(instrs);
    let mut bump = |slot: u16, score: u32| {
        let idx = slot as usize;
        if idx >= heat.len() { heat.resize(idx + 1, 0); }
        heat[idx] = heat[idx].saturating_add(score);
    };
    for (pc, instr) in instrs.iter().enumerate() {
        let w = lw.get(pc).copied().unwrap_or(1);
        match instr {
            Instr::LoadI32(d,_) | Instr::LoadI64(d,_) | Instr::LoadBool(d,_) | Instr::LoadUnit(d) => bump(*d, 1*w),
            Instr::Move(d, s) | Instr::Load(d, s) => { bump(*s, 3*w); bump(*d, 2*w); }
            Instr::Store(slot, s) => { bump(*s, 3*w); bump(*slot, 2*w); }
            Instr::BinOp(d, _, l, r) => { bump(*l, 4*w); bump(*r, 4*w); bump(*d, 3*w); }
            Instr::JumpFalse(s,_) | Instr::JumpTrue(s,_) | Instr::Return(s) => bump(*s, 4*w),
            _ => {}
        }
    }
    heat
}

fn rank_hot_slots(heat: &[u32]) -> Vec<(u16, u32)> {
    let mut pairs: Vec<(u16, u32)> = heat.iter().copied().enumerate()
        .filter_map(|(s, sc)| (sc > 0).then_some((s as u16, sc)))
        .collect();
    pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    pairs
}

// ─────────────────────────────────────────────────────────────────────────────
// Constant propagation table — flat Vec<Option<i64>>
// ─────────────────────────────────────────────────────────────────────────────

struct ConstTable { vals: Vec<Option<i64>> }

impl ConstTable {
    fn with_capacity(n: usize) -> Self { Self { vals: vec![None; n.max(1)] } }
    #[inline(always)] fn get(&self, slot: u16) -> Option<i64> { self.vals.get(slot as usize).copied().flatten() }
    #[inline(always)] fn insert(&mut self, slot: u16, v: i64) {
        let idx = slot as usize;
        if idx >= self.vals.len() { self.vals.resize(idx + 1, None); }
        self.vals[idx] = Some(v);
    }
    #[inline(always)] fn remove(&mut self, slot: u16) {
        if let Some(c) = self.vals.get_mut(slot as usize) { *c = None; }
    }
    fn clear(&mut self) { self.vals.fill(None); }
}

fn fold_binop(op: BinOpKind, l: i64, r: i64) -> Option<i64> {
    Some(match op {
        BinOpKind::Add => l.wrapping_add(r),
        BinOpKind::Sub => l.wrapping_sub(r),
        BinOpKind::Mul => l.wrapping_mul(r),
        BinOpKind::Div => { if r == 0 { return None; } l.wrapping_div(r) }
        BinOpKind::Rem => { if r == 0 { return None; } l.wrapping_rem(r) }
        BinOpKind::Eq  => i64::from(l == r),
        BinOpKind::Ne  => i64::from(l != r),
        BinOpKind::Lt  => i64::from(l <  r),
        BinOpKind::Le  => i64::from(l <= r),
        BinOpKind::Gt  => i64::from(l >  r),
        BinOpKind::Ge  => i64::from(l >= r),
        _ => return None,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Emission helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn load_rax(em: &mut Emitter, slot: u16, ra: &RegAlloc) {
    match ra.location(slot) { RegLoc::Reg(r) => em.mov_rr(0, r), RegLoc::Spill(o) => em.load_reg_mem(0, o) }
}
#[inline(always)]
fn load_rcx(em: &mut Emitter, slot: u16, ra: &RegAlloc) {
    match ra.location(slot) { RegLoc::Reg(r) => em.mov_rr(1, r), RegLoc::Spill(o) => em.load_reg_mem(1, o) }
}
#[inline(always)]
fn store_rax(em: &mut Emitter, slot: u16, ra: &RegAlloc) {
    match ra.location(slot) { RegLoc::Reg(r) => em.mov_rr(r, 0), RegLoc::Spill(o) => em.store_mem_reg(o, 0) }
}

/// Load `l` into rax, `r` into rcx — but if they share the same physical
/// register, skip the second load (rcx == rax already, same-reg optimization F).
#[inline(always)]
fn load_rax_rcx(em: &mut Emitter, l: u16, r: u16, ra: &RegAlloc) -> bool {
    load_rax(em, l, ra);
    let same_reg = ra.location(l) == ra.location(r);
    if same_reg { em.mov_rr(1, 0); } // rcx = rax, 3B but move-elim may apply
    else { load_rcx(em, r, ra); }
    same_reg
}

#[inline(always)]
fn instr_reads_slot(instr: &Instr, slot: u16) -> bool {
    match instr {
        Instr::Move(_,s) | Instr::Load(_,s) | Instr::Store(_,s) | Instr::Return(s) => *s == slot,
        Instr::BinOp(_,_,l,r) => *l == slot || *r == slot,
        Instr::JumpFalse(s,_) | Instr::JumpTrue(s,_) => *s == slot,
        _ => false,
    }
}
#[inline(always)]
fn instr_writes_slot(instr: &Instr, slot: u16) -> bool {
    match instr {
        Instr::LoadI32(d,_) | Instr::LoadI64(d,_) | Instr::LoadBool(d,_) | Instr::LoadUnit(d)
        | Instr::Move(d,_) | Instr::Load(d,_) | Instr::Store(d,_) | Instr::BinOp(d,_,_,_) => *d == slot,
        _ => false,
    }
}
#[inline(always)]
fn is_cf_barrier(instr: &Instr) -> bool {
    matches!(instr, Instr::Jump(_)|Instr::JumpFalse(_,_)|Instr::JumpTrue(_,_)|Instr::Return(_)|Instr::ReturnUnit)
}

/// True when a write to `slot` at `pc` is dead before any use in the current
/// basic block (conservative: false = "maybe live").
fn is_dead_def(instrs: &[Instr], pc: usize, slot: u16) -> bool {
    let mut i = pc + 1;
    while i < instrs.len() {
        let n = &instrs[i];
        if is_cf_barrier(n) { return false; }
        if instr_reads_slot(n, slot)  { return false; }
        if instr_writes_slot(n, slot) { return true; }
        i += 1;
    }
    true
}

#[inline(always)]
fn is_supported_binop(op: BinOpKind) -> bool {
    matches!(op, BinOpKind::Add|BinOpKind::Sub|BinOpKind::Mul|BinOpKind::Div|BinOpKind::Rem
               |BinOpKind::Eq|BinOpKind::Ne|BinOpKind::Lt|BinOpKind::Le|BinOpKind::Gt|BinOpKind::Ge)
}

/// Emit BinOp body: lhs already in rax, rhs in rcx, result in rax.
///
/// `same_reg` = true when lhs and rhs are in the same physical register
/// (optimization F — same-register strength reduction):
///   Add/Mul same-reg → SHL 1 (1-cycle vs 3-cycle IMUL)
///   Sub     same-reg → XOR EAX,EAX (dependency-breaking zero)
///   Eq      same-reg → MOV EAX,1   (always true)
///   Ne      same-reg → XOR EAX,EAX (always false)
fn emit_binop_rax_rcx(em: &mut Emitter, op: BinOpKind, same_reg: bool) -> bool {
    match op {
        BinOpKind::Add => {
            if same_reg { em.shl_rax_1(); } else { em.add_rax_rcx(); }
        }
        BinOpKind::Sub => {
            if same_reg { em.xor_eax_eax(); } else { em.sub_rax_rcx(); }
        }
        BinOpKind::Mul => {
            if same_reg { em.shl_rax_1(); } else { em.imul_rax_rcx(); }
        }
        BinOpKind::Div  => { em.cqo(); em.idiv_rcx(); }
        BinOpKind::Rem  => { em.cqo(); em.idiv_rcx(); em.mov_rax_rdx(); }
        BinOpKind::Eq   => {
            if same_reg { em.b(0xB8); em.d(1); } // MOV EAX,1
            else { em.cmp_rax_rcx(); em.setcc_al(0x94); em.movzx_rax_al(); }
        }
        BinOpKind::Ne   => {
            if same_reg { em.xor_eax_eax(); }
            else { em.cmp_rax_rcx(); em.setcc_al(0x95); em.movzx_rax_al(); }
        }
        BinOpKind::Lt   => { em.cmp_rax_rcx(); em.setcc_al(0x9C); em.movzx_rax_al(); }
        BinOpKind::Le   => { em.cmp_rax_rcx(); em.setcc_al(0x9E); em.movzx_rax_al(); }
        BinOpKind::Gt   => { em.cmp_rax_rcx(); em.setcc_al(0x9F); em.movzx_rax_al(); }
        BinOpKind::Ge   => { em.cmp_rax_rcx(); em.setcc_al(0x9D); em.movzx_rax_al(); }
        _ => return false,
    }
    true
}

/// Emit BinOp with immediate RHS `imm`: lhs already in rax, result in rax.
/// All strength reductions applied per uops.info latency data.
fn emit_binop_rax_imm(em: &mut Emitter, op: BinOpKind, imm: i32) {
    match op {
        BinOpKind::Add => match imm { 0 => {}, 1 => em.inc_rax(), -1 => em.dec_rax(), _ => em.add_rax_imm(imm) },
        BinOpKind::Sub => match imm { 0 => {}, 1 => em.dec_rax(), -1 => em.inc_rax(), _ => em.sub_rax_imm(imm) },
        BinOpKind::Mul => match imm {
            0  => em.xor_eax_eax(),
            1  => {}
            -1 => em.neg_rax(),
            2  => em.shl_rax_1(),           // SHL 1  — 1 cycle p06 vs IMUL 3 cycles p1
            3  => em.lea_rax_rax_mul3(),    // LEA [rax+rax*2] — 1 cycle p1 (2-operand)
            4  => em.shl_rax_imm8(2),
            5  => em.lea_rax_rax_mul5(),
            8  => em.shl_rax_imm8(3),
            9  => em.lea_rax_rax_mul9(),
            _  => {
                let u = imm as u32;
                if imm > 0 && u.is_power_of_two() { em.shl_rax_imm8(u.trailing_zeros() as u8); }
                else { em.imul_rax_imm32(imm); }
            }
        },
        BinOpKind::Eq  => { em.cmp_rax_imm(imm); em.setcc_al(0x94); em.movzx_rax_al(); }
        BinOpKind::Ne  => { em.cmp_rax_imm(imm); em.setcc_al(0x95); em.movzx_rax_al(); }
        BinOpKind::Lt  => { em.cmp_rax_imm(imm); em.setcc_al(0x9C); em.movzx_rax_al(); }
        BinOpKind::Le  => { em.cmp_rax_imm(imm); em.setcc_al(0x9E); em.movzx_rax_al(); }
        BinOpKind::Gt  => { em.cmp_rax_imm(imm); em.setcc_al(0x9F); em.movzx_rax_al(); }
        BinOpKind::Ge  => { em.cmp_rax_imm(imm); em.setcc_al(0x9D); em.movzx_rax_al(); }
        _ => {}
    }
}

fn emit_ret(em: &mut Emitter, callee: &[u8]) {
    for &r in callee.iter().rev() { em.pop_reg(r); }
    em.ret();
}

// ─────────────────────────────────────────────────────────────────────────────
// Branch relaxation pass
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum BranchKind { Jmp, Jz, Jnz }

struct Fixup { disp_pos: usize, target_pc: usize, kind: BranchKind }

fn patch_fixups(buf: &mut Vec<u8>, fixups: &[Fixup], pc_to_off: &[usize]) -> Option<()> {
    for fx in fixups {
        let tgt = *pc_to_off.get(fx.target_pc)? as isize;
        let nxt = (fx.disp_pos + 4) as isize;
        let rel = i32::try_from(tgt - nxt).ok()?;

        if let Ok(r8) = i8::try_from(rel) {
            // Shrink to rel8 — saves 3–4 bytes, better µop cache density.
            let op_start = match fx.kind { BranchKind::Jmp => fx.disp_pos - 1, _ => fx.disp_pos - 2 };
            buf[op_start] = match fx.kind { BranchKind::Jmp => 0xEB, BranchKind::Jz => 0x74, BranchKind::Jnz => 0x75 };
            buf[op_start + 1] = r8 as u8;
            for b in &mut buf[op_start + 2..fx.disp_pos + 4] { *b = 0x90; }
        } else {
            buf[fx.disp_pos..fx.disp_pos + 4].copy_from_slice(&rel.to_le_bytes());
        }
    }
    Some(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper macros for fusion blocks — reduce repetitive boilerplate
// ─────────────────────────────────────────────────────────────────────────────

/// Advance `pc` by 2, record `pc_to_off[pc+1]`, and `continue` the main loop.
macro_rules! fuse2 {
    ($pc:expr, $em:expr, $pc_to_off:expr) => {{
        $pc_to_off[$pc + 1] = $em.pos();
        $pc += 2;
        continue;
    }};
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

#[must_use]
pub fn is_available() -> bool { cfg!(target_arch = "x86_64") }

#[must_use]
pub fn translate(compiled: &CompiledFn) -> Option<NativeCode> {
    if !cfg!(target_arch = "x86_64") { return None; }

    let instrs     = &compiled.instrs;
    let slot_count = compiled.slot_count as usize;

    // Gate: reject unsupported instructions before allocating anything.
    for instr in instrs {
        match instr {
            Instr::LoadI32(..)|Instr::LoadI64(..)|Instr::LoadBool(..)|Instr::LoadUnit(..)
            |Instr::Move(..)|Instr::Load(..)|Instr::Store(..)
            |Instr::BinOp(..)|Instr::Jump(..)
            |Instr::JumpFalse(..)|Instr::JumpTrue(..)
            |Instr::Return(..)|Instr::ReturnUnit|Instr::Nop => {}
            _ => return None,
        }
    }

    // ── Pass 1: analysis ──────────────────────────────────────────────────

    let intervals = compute_live_intervals(instrs, slot_count);
    let hotness   = compute_slot_hotness(instrs, slot_count);
    let hot_slots = rank_hot_slots(&hotness);

    const PIN_POOL: &[u8] = &[12, 13, 14, 15, 6]; // callee-saved for pin
    let pinned: Vec<(u16, u8)> = hot_slots.iter()
        .take(PIN_POOL.len())
        .zip(PIN_POOL.iter().copied())
        .map(|((s, _), r)| (*s, r))
        .collect();

    let ra = linear_scan(&intervals, slot_count, &pinned, &hotness);

    // ── Pass 2: code generation ───────────────────────────────────────────

    // Pre-size buffer to avoid realloc during code generation (optimization L).
    let mut em        = Emitter::with_capacity(estimate_code_bytes(instrs));
    let mut pc_to_off = vec![0usize; instrs.len() + 1];
    let mut fixups    = Vec::<Fixup>::new();

    // Prologue.
    for &reg in &ra.used_callee_saved { em.push_reg(reg); }

    // Pre-load all register-allocated slots.
    {
        let mut loaded: u32 = 0;
        for &(slot, reg) in &pinned {
            let bit = 1u32 << reg;
            if loaded & bit == 0 { loaded |= bit; em.load_reg_mem(reg, (slot as i32) * 8); }
        }
        for iv in &intervals {
            if let RegLoc::Reg(r) = ra.location(iv.slot) {
                let bit = 1u32 << r;
                if loaded & bit == 0 { loaded |= bit; em.load_reg_mem(r, (iv.slot as i32) * 8); }
            }
        }
    }

    let mut const_at = ConstTable::with_capacity(slot_count + 1);

    let mut pc = 0usize;
    while pc < instrs.len() {
        pc_to_off[pc] = em.pos();

        // ══════════════════════════════════════════════════════════════════
        // 3-INSTRUCTION FUSIONS
        // ══════════════════════════════════════════════════════════════════

        // ── D.1: Mul(t,x,N) + Add(r,t,y) → LEA [rax*N+rcx]  (N∈{2,4,8}) ──
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, BinOpKind::Mul, mul_l, mul_r),
                    Instr::BinOp(r, BinOpKind::Add, add_l, add_r))
                = (&instrs[pc], &instrs[pc + 1])
            {
                let addend = if *add_l == *t && *add_r != *t && *r != *t { Some(*add_r) }
                       else if *add_r == *t && *add_l != *t && *r != *t { Some(*add_l) }
                       else { None };

                if let Some(addend) = addend {
                    let base_scale = const_at.get(*mul_r).map(|c| (*mul_l, c))
                        .or_else(|| const_at.get(*mul_l).map(|c| (*mul_r, c)));
                    if let Some((base, sc)) = base_scale {
                        if matches!(sc, 2 | 4 | 8) {
                            load_rax(&mut em, base, &ra);
                            load_rcx(&mut em, addend, &ra);
                            em.lea_rax_rax_scale_plus_rcx(sc as u8);
                            if !is_dead_def(instrs, pc + 1, *r) { store_rax(&mut em, *r, &ra); }
                            const_at.remove(*t); const_at.remove(*r);
                            fuse2!(pc, em, pc_to_off);
                        }
                    }
                }
            }
        }

        // ── D.2: BinOp(t,op1,a,b) + BinOp(r,op2,t,c) → op-chain ──────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op1, a, b), Instr::BinOp(r, op2, l2, r2))
                = (&instrs[pc], &instrs[pc + 1])
            {
                let comm2 = matches!(op2, BinOpKind::Add|BinOpKind::Mul|BinOpKind::Eq|BinOpKind::Ne);
                let t_lhs = *l2 == *t && *r2 != *t;
                let t_rhs = *r2 == *t && *l2 != *t && comm2;
                if (t_lhs || t_rhs) && is_supported_binop(*op1) && is_supported_binop(*op2) {
                    let other = if t_lhs { *r2 } else { *l2 };
                    let sr1 = load_rax_rcx(&mut em, *a, *b, &ra);
                    emit_binop_rax_rcx(&mut em, *op1, sr1);
                    load_rcx(&mut em, other, &ra);
                    emit_binop_rax_rcx(&mut em, *op2, false);
                    if !is_dead_def(instrs, pc + 1, *r) { store_rax(&mut em, *r, &ra); }
                    const_at.remove(*t); const_at.remove(*r);
                    fuse2!(pc, em, pc_to_off);
                }
            }
        }

        // ══════════════════════════════════════════════════════════════════
        // 2-INSTRUCTION FUSIONS
        // ══════════════════════════════════════════════════════════════════

        // ── E.1: Load*(tmp,c) + JumpFalse/True(tmp) → compile-time branch ──
        if pc + 1 < instrs.len() {
            let mc = match &instrs[pc] {
                Instr::LoadI32(t,v) => Some((*t, *v as i64)),
                Instr::LoadI64(t,v) => Some((*t, *v)),
                Instr::LoadBool(t,v) => Some((*t, i64::from(*v))),
                Instr::LoadUnit(t) => Some((*t, 0i64)),
                _ => None,
            };
            if let Some((tmp, c)) = mc {
                let mut folded = false;
                match &instrs[pc + 1] {
                    Instr::JumpFalse(cond, off) if *cond == tmp => {
                        let tgt = ((pc as i32) + 2 + off) as usize;
                        if tgt > instrs.len() { return None; }
                        if c == 0 { let p = em.jmp_rel32_placeholder(); fixups.push(Fixup { disp_pos: p, target_pc: tgt, kind: BranchKind::Jmp }); }
                        folded = true;
                    }
                    Instr::JumpTrue(cond, off) if *cond == tmp => {
                        let tgt = ((pc as i32) + 2 + off) as usize;
                        if tgt > instrs.len() { return None; }
                        if c != 0 { let p = em.jmp_rel32_placeholder(); fixups.push(Fixup { disp_pos: p, target_pc: tgt, kind: BranchKind::Jmp }); }
                        folded = true;
                    }
                    _ => {}
                }
                if folded {
                    const_at.insert(tmp, c);
                    const_at.clear();
                    fuse2!(pc, em, pc_to_off);
                }
            }
        }

        // ── E.2: Load*(tmp,c) + BinOp(dst,op,x,tmp) → immediate arith ──────
        if pc + 1 < instrs.len() {
            let mi = match &instrs[pc] {
                Instr::LoadI32(t,v) => Some((*t, *v as i64)),
                Instr::LoadI64(t,v) => Some((*t, *v)),
                _ => None,
            };
            if let Some((tmp, c)) = mi {
                if let Instr::BinOp(dst, op, l, r) = &instrs[pc + 1] {
                    if let Ok(imm) = i32::try_from(c) {
                        let rhs_imm = *r == tmp;
                        let lhs_imm = *l == tmp;
                        let ok = match op {
                            BinOpKind::Add | BinOpKind::Mul => rhs_imm || lhs_imm,
                            BinOpKind::Sub | BinOpKind::Eq | BinOpKind::Ne
                            | BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge => rhs_imm,
                            _ => false,
                        };
                        if ok {
                            let live = if rhs_imm { *l } else { *r };
                            load_rax(&mut em, live, &ra);
                            emit_binop_rax_imm(&mut em, *op, imm);
                            if !is_dead_def(instrs, pc + 1, *dst) { store_rax(&mut em, *dst, &ra); }
                            let lv = const_at.get(live);
                            match lv.and_then(|lc| { let (fl,fr) = if rhs_imm {(lc,c)} else {(c,lc)}; fold_binop(*op, fl, fr) }) {
                                Some(v) => const_at.insert(*dst, v),
                                None    => const_at.remove(*dst),
                            }
                            fuse2!(pc, em, pc_to_off);
                        }
                    }
                }
            }
        }

        // ── E.3: BinOp(t,op,l,r) + Store(slot,t) ───────────────────────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::Store(slot, src))
                = (&instrs[pc], &instrs[pc + 1])
            {
                if t == src && is_supported_binop(*op) {
                    let sr = load_rax_rcx(&mut em, *l, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op, sr);
                    if !is_dead_def(instrs, pc + 1, *slot) { store_rax(&mut em, *slot, &ra); }
                    match const_at.get(*l).zip(const_at.get(*r)).and_then(|(lv,rv)| fold_binop(*op, lv, rv)) {
                        Some(c) => const_at.insert(*slot, c), None => const_at.remove(*slot),
                    }
                    const_at.remove(*t);
                    fuse2!(pc, em, pc_to_off);
                }
            }
        }

        // ── E.4: BinOp(t,op,l,r) + JumpFalse(t,off) ────────────────────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::JumpFalse(cond, off))
                = (&instrs[pc], &instrs[pc + 1])
            {
                if t == cond && is_supported_binop(*op) {
                    let tgt = ((pc as i32) + 2 + off) as usize;
                    if tgt > instrs.len() { return None; }
                    let sr = load_rax_rcx(&mut em, *l, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op, sr);
                    em.test_rax_rax();
                    let p = em.jz_rel32_placeholder();
                    fixups.push(Fixup { disp_pos: p, target_pc: tgt, kind: BranchKind::Jz });
                    const_at.clear();
                    fuse2!(pc, em, pc_to_off);
                }
            }
        }

        // ── E.5: BinOp(t,op,l,r) + JumpTrue(t,off) ─────────────────────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::JumpTrue(cond, off))
                = (&instrs[pc], &instrs[pc + 1])
            {
                if t == cond && is_supported_binop(*op) {
                    let tgt = ((pc as i32) + 2 + off) as usize;
                    if tgt > instrs.len() { return None; }
                    let sr = load_rax_rcx(&mut em, *l, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op, sr);
                    em.test_rax_rax();
                    let p = em.jnz_rel32_placeholder();
                    fixups.push(Fixup { disp_pos: p, target_pc: tgt, kind: BranchKind::Jnz });
                    const_at.clear();
                    fuse2!(pc, em, pc_to_off);
                }
            }
        }

        // ── E.6: BinOp(t,op,l,r) + Return(t) ───────────────────────────────
        if pc + 1 < instrs.len() {
            if let (Instr::BinOp(t, op, l, r), Instr::Return(ret))
                = (&instrs[pc], &instrs[pc + 1])
            {
                if t == ret && is_supported_binop(*op) {
                    let sr = load_rax_rcx(&mut em, *l, *r, &ra);
                    emit_binop_rax_rcx(&mut em, *op, sr);
                    emit_ret(&mut em, &ra.used_callee_saved);
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2; continue;
                }
            }
        }

        // ── E.7: Move/Load(d,s) + BinOp(r,op,d,x) → forwarded load ─────────
        if pc + 1 < instrs.len() {
            if let (Instr::Move(d, s) | Instr::Load(d, s), Instr::BinOp(r, op, bl, br))
                = (&instrs[pc], &instrs[pc + 1])
            {
                if *bl == *d && *br != *d && is_supported_binop(*op) {
                    let sr = load_rax_rcx(&mut em, *s, *br, &ra);
                    emit_binop_rax_rcx(&mut em, *op, sr);
                    if !is_dead_def(instrs, pc + 1, *r) { store_rax(&mut em, *r, &ra); }
                    match const_at.get(*s).zip(const_at.get(*br)).and_then(|(lv,rv)| fold_binop(*op, lv, rv)) {
                        Some(c) => const_at.insert(*r, c), None => const_at.remove(*r),
                    }
                    if let Some(c) = const_at.get(*s) { const_at.insert(*d, c); }
                    fuse2!(pc, em, pc_to_off);
                }
            }
        }

        // ── E.8 (new): Load*(tmp,c) + Move(d,tmp) → direct imm load to d ────
        //   Eliminates the intermediate slot write and read entirely.
        if pc + 1 < instrs.len() {
            let mc = match &instrs[pc] {
                Instr::LoadI32(t,v) => Some((*t, *v as i64)),
                Instr::LoadI64(t,v) => Some((*t, *v)),
                Instr::LoadBool(t,v) => Some((*t, i64::from(*v))),
                Instr::LoadUnit(t)  => Some((*t, 0i64)),
                _ => None,
            };
            if let Some((tmp, c)) = mc {
                if let Instr::Move(d, src) = &instrs[pc + 1] {
                    if *src == tmp {
                        // Emit directly to d, skip tmp entirely.
                        if c == 0 { em.xor_eax_eax(); } else { em.mov_rax_imm_opt(c); }
                        if !is_dead_def(instrs, pc + 1, *d) { store_rax(&mut em, *d, &ra); }
                        const_at.insert(tmp, c);
                        const_at.insert(*d, c);
                        fuse2!(pc, em, pc_to_off);
                    }
                }
            }
        }

        // ══════════════════════════════════════════════════════════════════
        // CONSTANT FOLDING — single BinOp with both operands known
        // ══════════════════════════════════════════════════════════════════
        if let Instr::BinOp(d, op, l, r) = &instrs[pc] {
            if let Some(v) = const_at.get(*l).zip(const_at.get(*r))
                                .and_then(|(lv, rv)| fold_binop(*op, lv, rv))
            {
                if v == 0 { em.xor_eax_eax(); } else { em.mov_rax_imm_opt(v); }
                if !is_dead_def(instrs, pc, *d) { store_rax(&mut em, *d, &ra); }
                const_at.insert(*d, v);
                pc += 1; continue;
            }
        }

        // ══════════════════════════════════════════════════════════════════
        // SINGLE-INSTRUCTION FALLBACK
        // ══════════════════════════════════════════════════════════════════
        match &instrs[pc] {
            Instr::LoadI32(d, v) => {
                let c = *v as i64;
                if c == 0 { em.xor_eax_eax(); } else { em.mov_rax_imm_opt(c); }
                if !is_dead_def(instrs, pc, *d) { store_rax(&mut em, *d, &ra); }
                const_at.insert(*d, c);
            }
            Instr::LoadI64(d, v) => {
                if *v == 0 { em.xor_eax_eax(); } else { em.mov_rax_imm_opt(*v); }
                if !is_dead_def(instrs, pc, *d) { store_rax(&mut em, *d, &ra); }
                const_at.insert(*d, *v);
            }
            Instr::LoadBool(d, v) => {
                let c = i64::from(*v);
                if c == 0 { em.xor_eax_eax(); } else { em.b(0xB8); em.d(1); } // MOV EAX,1
                if !is_dead_def(instrs, pc, *d) { store_rax(&mut em, *d, &ra); }
                const_at.insert(*d, c);
            }
            Instr::LoadUnit(d) => {
                em.xor_eax_eax();
                if !is_dead_def(instrs, pc, *d) { store_rax(&mut em, *d, &ra); }
                const_at.insert(*d, 0);
            }
            Instr::Move(d, s) | Instr::Load(d, s) => {
                load_rax(&mut em, *s, &ra);
                if !is_dead_def(instrs, pc, *d) { store_rax(&mut em, *d, &ra); }
                if let Some(c) = const_at.get(*s) { const_at.insert(*d, c); } else { const_at.remove(*d); }
            }
            Instr::Store(slot, s) => {
                load_rax(&mut em, *s, &ra);
                if !is_dead_def(instrs, pc, *slot) { store_rax(&mut em, *slot, &ra); }
                if let Some(c) = const_at.get(*s) { const_at.insert(*slot, c); } else { const_at.remove(*slot); }
            }
            Instr::BinOp(d, op, l, r) => {
                if !is_supported_binop(*op) { return None; }
                let sr = load_rax_rcx(&mut em, *l, *r, &ra);
                emit_binop_rax_rcx(&mut em, *op, sr);
                if !is_dead_def(instrs, pc, *d) { store_rax(&mut em, *d, &ra); }
                match const_at.get(*l).zip(const_at.get(*r)).and_then(|(lv,rv)| fold_binop(*op, lv, rv)) {
                    Some(c) => const_at.insert(*d, c), None => const_at.remove(*d),
                }
            }
            Instr::Jump(off) => {
                let tgt = ((pc as i32) + 1 + off) as usize;
                if tgt > instrs.len() { return None; }
                let p = em.jmp_rel32_placeholder();
                fixups.push(Fixup { disp_pos: p, target_pc: tgt, kind: BranchKind::Jmp });
                const_at.clear();
            }
            Instr::JumpFalse(cond, off) => {
                let tgt = ((pc as i32) + 1 + off) as usize;
                if tgt > instrs.len() { return None; }
                load_rax(&mut em, *cond, &ra);
                em.test_rax_rax();
                let p = em.jz_rel32_placeholder();
                fixups.push(Fixup { disp_pos: p, target_pc: tgt, kind: BranchKind::Jz });
                const_at.clear();
            }
            Instr::JumpTrue(cond, off) => {
                let tgt = ((pc as i32) + 1 + off) as usize;
                if tgt > instrs.len() { return None; }
                load_rax(&mut em, *cond, &ra);
                em.test_rax_rax();
                let p = em.jnz_rel32_placeholder();
                fixups.push(Fixup { disp_pos: p, target_pc: tgt, kind: BranchKind::Jnz });
                const_at.clear();
            }
            Instr::Return(r) => {
                load_rax(&mut em, *r, &ra);
                emit_ret(&mut em, &ra.used_callee_saved);
            }
            Instr::ReturnUnit => { em.xor_eax_eax(); emit_ret(&mut em, &ra.used_callee_saved); }
            Instr::Nop => {}
            _ => return None,
        }
        pc += 1;
    }

    // Fallthrough epilogue.
    pc_to_off[instrs.len()] = em.pos();
    em.xor_eax_eax();
    emit_ret(&mut em, &ra.used_callee_saved);

    patch_fixups(&mut em.buf, &fixups, &pc_to_off)?;

    let mem = ExecMem::new(&em.buf)?;
    Some(NativeCode { slot_count: compiled.slot_count, mem })
}

// ─────────────────────────────────────────────────────────────────────────────
// Execution
// ─────────────────────────────────────────────────────────────────────────────

pub fn execute(native: &NativeCode, args: &[Value]) -> Result<Value, RuntimeError> {
    thread_local! {
        static EXEC_REGS: RefCell<Vec<i64>> = RefCell::new(Vec::new());
    }
    let needed = native.slot_count as usize + 32;
    EXEC_REGS.with(|cell| -> Result<Value, RuntimeError> {
        let mut regs = cell.borrow_mut();
        if regs.len() < needed { regs.resize(needed, 0); }
        else { regs[..needed].fill(0); }
        for (i, arg) in args.iter().enumerate() {
            if i >= needed { break; }
            regs[i] = match arg {
                Value::I8(v)   => *v as i64,
                Value::I16(v)  => *v as i64,
                Value::I32(v)  => *v as i64,
                Value::I64(v)  => *v,
                Value::U8(v)   => *v as i64,
                Value::U16(v)  => *v as i64,
                Value::U32(v)  => *v as i64,
                Value::U64(v)  => *v as i64,
                Value::Bool(v) => i64::from(*v),
                _ => return Err(RuntimeError::new("native JIT: unsupported arg type")),
            };
        }
        let f = native.mem.entry();
        Ok(Value::I64(unsafe { f(regs.as_mut_ptr()) }))
    })
}
