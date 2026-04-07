//! Phase 3 real machine-code JIT backend (x86-64, System V).
//! Generates compact native code for a subset of Jules VM bytecode.
//!
//! Next realistic optimization steps:
//! - micro-kernels / superinstructions for common bytecode sequences,
//! - improved register reuse/allocation heuristics,
//! - optional alternate backends (e.g. LLVM IR) for broader near-native parity.

use std::cell::RefCell;
use std::collections::HashMap;

use libc::{mmap, munmap, MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};

use crate::ast::BinOpKind;
use crate::interp::{CompiledFn, Instr, RuntimeError, Value};

pub struct NativeCode {
    pub slot_count: u16,
    mem: ExecMem,
}

struct ExecMem {
    ptr: *mut u8,
    len: usize,
}

impl Drop for ExecMem {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe { munmap(self.ptr.cast(), self.len) };
        }
    }
}

impl ExecMem {
    fn new(code: &[u8]) -> Option<Self> {
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

struct Emitter {
    buf: Vec<u8>,
}

impl Emitter {
    fn new() -> Self {
        Self { buf: Vec::with_capacity(4096) }
    }
    fn pos(&self) -> usize { self.buf.len() }
    fn b(&mut self, v: u8) { self.buf.push(v); }
    fn d(&mut self, v: i32) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn q(&mut self, v: i64) { self.buf.extend_from_slice(&v.to_le_bytes()); }

    fn mov_rax_imm64(&mut self, v: i64) { self.b(0x48); self.b(0xB8); self.q(v); }
    fn mov_rax_m_rdi_off(&mut self, off: i32) { self.b(0x48); self.b(0x8B); self.b(0x87); self.d(off); }
    fn mov_rcx_m_rdi_off(&mut self, off: i32) { self.b(0x48); self.b(0x8B); self.b(0x8F); self.d(off); }
    fn mov_m_rdi_off_rax(&mut self, off: i32) { self.b(0x48); self.b(0x89); self.b(0x87); self.d(off); }
    fn mov_r8_m_rdi_off(&mut self, off: i32) { self.b(0x4C); self.b(0x8B); self.b(0x87); self.d(off); }
    fn mov_r9_m_rdi_off(&mut self, off: i32) { self.b(0x4C); self.b(0x8B); self.b(0x8F); self.d(off); }
    fn mov_rax_r8(&mut self) { self.b(0x4C); self.b(0x89); self.b(0xC0); }
    fn mov_rax_r9(&mut self) { self.b(0x4C); self.b(0x89); self.b(0xC8); }
    fn mov_r8_rax(&mut self) { self.b(0x49); self.b(0x89); self.b(0xC0); }
    fn mov_r9_rax(&mut self) { self.b(0x49); self.b(0x89); self.b(0xC1); }

    fn add_rax_rcx(&mut self) { self.b(0x48); self.b(0x01); self.b(0xC8); }
    fn sub_rax_rcx(&mut self) { self.b(0x48); self.b(0x29); self.b(0xC8); }
    fn imul_rax_rcx(&mut self) { self.b(0x48); self.b(0x0F); self.b(0xAF); self.b(0xC1); }
    fn add_rax_imm32(&mut self, v: i32) { self.b(0x48); self.b(0x05); self.d(v); }
    fn sub_rax_imm32(&mut self, v: i32) { self.b(0x48); self.b(0x2D); self.d(v); }
    fn imul_rax_imm32(&mut self, v: i32) { self.b(0x48); self.b(0x69); self.b(0xC0); self.d(v); }
    fn inc_rax(&mut self) { self.b(0x48); self.b(0xFF); self.b(0xC0); }
    fn dec_rax(&mut self) { self.b(0x48); self.b(0xFF); self.b(0xC8); }
    fn shl_rax_imm8(&mut self, v: u8) { self.b(0x48); self.b(0xC1); self.b(0xE0); self.b(v); }
    fn xor_rax_rax(&mut self) { self.b(0x48); self.b(0x31); self.b(0xC0); }
    fn lea_rax_rax_mul3(&mut self) { self.b(0x48); self.b(0x8D); self.b(0x04); self.b(0x40); } // rax = rax + rax*2
    fn lea_rax_rax_mul5(&mut self) { self.b(0x48); self.b(0x8D); self.b(0x04); self.b(0x80); } // rax = rax + rax*4
    fn lea_rax_rax_mul9(&mut self) { self.b(0x48); self.b(0x8D); self.b(0x04); self.b(0xC0); } // rax = rax + rax*8

    fn cmp_rax_rcx(&mut self) { self.b(0x48); self.b(0x39); self.b(0xC8); }
    fn cmp_rax_imm32(&mut self, v: i32) { self.b(0x48); self.b(0x3D); self.d(v); }
    fn setcc_al(&mut self, cc: u8) { self.b(0x0F); self.b(cc); self.b(0xC0); }
    fn movzx_rax_al(&mut self) { self.b(0x48); self.b(0x0F); self.b(0xB6); self.b(0xC0); }

    fn cqo(&mut self) { self.b(0x48); self.b(0x99); }
    fn idiv_rcx(&mut self) { self.b(0x48); self.b(0xF7); self.b(0xF9); }
    fn mov_rax_rdx(&mut self) { self.b(0x48); self.b(0x89); self.b(0xD0); }

    fn test_rax_rax(&mut self) { self.b(0x48); self.b(0x85); self.b(0xC0); }

    fn jmp_rel32_placeholder(&mut self) -> usize { self.b(0xE9); let p=self.pos(); self.d(0); p }
    fn jz_rel32_placeholder(&mut self) -> usize { self.b(0x0F); self.b(0x84); let p=self.pos(); self.d(0); p }
    fn jnz_rel32_placeholder(&mut self) -> usize { self.b(0x0F); self.b(0x85); let p=self.pos(); self.d(0); p }

    fn ret(&mut self) { self.b(0xC3); }
}

#[must_use]
pub fn is_available() -> bool { cfg!(target_arch = "x86_64") }

pub fn translate(compiled: &CompiledFn) -> Option<NativeCode> {
    if !cfg!(target_arch = "x86_64") {
        return None;
    }

    for instr in &compiled.instrs {
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

    // Lightweight profile-guided register pinning:
    // keep the two hottest slots resident in r8/r9 across the function.
    let mut slot_freq: HashMap<u16, usize> = HashMap::new();
    for instr in &compiled.instrs {
        match instr {
            Instr::Load(d, s) | Instr::Move(d, s) => {
                *slot_freq.entry(*d).or_default() += 1;
                *slot_freq.entry(*s).or_default() += 1;
            }
            Instr::Store(s, r) | Instr::ArrayGet(s, r, _) => {
                *slot_freq.entry(*s).or_default() += 1;
                *slot_freq.entry(*r).or_default() += 1;
            }
            Instr::BinOp(d, _, l, r) => {
                *slot_freq.entry(*d).or_default() += 1;
                *slot_freq.entry(*l).or_default() += 1;
                *slot_freq.entry(*r).or_default() += 1;
            }
            Instr::JumpFalse(s, _) | Instr::Return(s) => {
                *slot_freq.entry(*s).or_default() += 1;
            }
            _ => {}
        }
    }
    let mut hot_slots: Vec<(u16, usize)> = slot_freq.into_iter().collect();
    hot_slots.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    let pinned_slot0 = hot_slots.first().map(|(s, _)| *s);
    let pinned_slot1 = hot_slots.get(1).map(|(s, _)| *s);
    let slot_offsets: Vec<i32> = (0..=compiled.slot_count)
        .map(|s| (s as i32) * 8)
        .collect();

    let mut em = Emitter::new();
    let mut pc_to_off = vec![0usize; compiled.instrs.len() + 1];
    let mut fixups: Vec<(usize, usize)> = Vec::new(); // (disp_pos, target_pc)

    // Preload pinned slot.
    if let Some(s) = pinned_slot0 {
        let off = *slot_offsets.get(s as usize).unwrap_or(&((s as i32) * 8));
        em.mov_r8_m_rdi_off(off);
    }
    if let Some(s) = pinned_slot1 {
        let off = *slot_offsets.get(s as usize).unwrap_or(&((s as i32) * 8));
        em.mov_r9_m_rdi_off(off);
    }

    let mut pc = 0usize;
    while pc < compiled.instrs.len() {
        pc_to_off[pc] = em.pos();
        let slot_off = |s: u16| -> i32 {
            *slot_offsets.get(s as usize).unwrap_or(&((s as i32) * 8))
        };
        let load_slot_rax = |em: &mut Emitter, s: u16| {
            if Some(s) == pinned_slot0 {
                em.mov_rax_r8();
            } else if Some(s) == pinned_slot1 {
                em.mov_rax_r9();
            } else {
                em.mov_rax_m_rdi_off(slot_off(s));
            }
        };
        let load_slot_rcx = |em: &mut Emitter, s: u16| {
            if Some(s) == pinned_slot0 {
                // mov rcx, r8
                em.b(0x4C); em.b(0x89); em.b(0xC1);
            } else if Some(s) == pinned_slot1 {
                // mov rcx, r9
                em.b(0x4C); em.b(0x89); em.b(0xC9);
            } else {
                em.mov_rcx_m_rdi_off(slot_off(s));
            }
        };
        let store_rax_slot = |em: &mut Emitter, s: u16| {
            if Some(s) == pinned_slot0 {
                em.mov_r8_rax();
            } else if Some(s) == pinned_slot1 {
                em.mov_r9_rax();
            } else {
                em.mov_m_rdi_off_rax(slot_off(s));
            }
        };

        // Superinstruction fusion:
        //   LoadI32/LoadI64(tmp, c) ; BinOp(mid, op, x, tmp) ; JumpFalse/JumpTrue(mid, off)
        // =>
        //   compute with immediate form directly in rax and branch, without temporary slot store/load.
        if pc + 2 < compiled.instrs.len() {
            let maybe_imm = match &compiled.instrs[pc] {
                Instr::LoadI32(tmp, v) => Some((*tmp, *v as i64)),
                Instr::LoadI64(tmp, v) => Some((*tmp, *v)),
                _ => None,
            };
            if let Some((tmp, c)) = maybe_imm {
                if let (Instr::BinOp(mid, op, l, r), branch) =
                    (&compiled.instrs[pc + 1], &compiled.instrs[pc + 2])
                {
                    let (is_jump_true, cond, off) = match branch {
                        Instr::JumpFalse(cond, off) => (false, *cond, *off),
                        Instr::JumpTrue(cond, off) => (true, *cond, *off),
                        _ => (false, u16::MAX, 0),
                    };
                    if cond == *mid {
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
                                let target = ((pc as i32) + 3 + off) as usize;
                                if target > compiled.instrs.len() {
                                    return None;
                                }
                                let live_reg = if rhs_is_imm { *l } else { *r };
                                load_slot_rax(&mut em, live_reg);
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
                                    BinOpKind::Mul => {
                                        if imm == 0 {
                                            em.xor_rax_rax();
                                        } else if imm == 1 {
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
                                    BinOpKind::Sub => {
                                        if imm == 1 {
                                            em.dec_rax();
                                        } else if imm == -1 {
                                            em.inc_rax();
                                        } else if imm != 0 {
                                            em.sub_rax_imm32(imm);
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
                                em.test_rax_rax();
                                let disp_pos = if is_jump_true {
                                    em.jnz_rel32_placeholder()
                                } else {
                                    em.jz_rel32_placeholder()
                                };
                                fixups.push((disp_pos, target));
                                pc_to_off[pc + 1] = em.pos();
                                pc_to_off[pc + 2] = em.pos();
                                pc += 3;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // Superinstruction fusion:
        //   LoadI*/LoadBool/LoadUnit(tmp, c) ; JumpFalse/JumpTrue(tmp, off)
        // =>
        //   resolve branch at translation time, eliminating runtime condition checks.
        if pc + 1 < compiled.instrs.len() {
            let maybe_const = match &compiled.instrs[pc] {
                Instr::LoadI32(tmp, v) => Some((*tmp, *v as i64)),
                Instr::LoadI64(tmp, v) => Some((*tmp, *v)),
                Instr::LoadBool(tmp, v) => Some((*tmp, i64::from(*v))),
                Instr::LoadUnit(tmp) => Some((*tmp, 0)),
                _ => None,
            };
            if let Some((tmp, c)) = maybe_const {
                match &compiled.instrs[pc + 1] {
                    Instr::JumpFalse(cond, off) if *cond == tmp => {
                        let target = ((pc as i32) + 2 + *off) as usize;
                        if target > compiled.instrs.len() {
                            return None;
                        }
                        if c == 0 {
                            let disp_pos = em.jmp_rel32_placeholder();
                            fixups.push((disp_pos, target));
                        }
                    }
                    Instr::JumpTrue(cond, off) if *cond == tmp => {
                        let target = ((pc as i32) + 2 + *off) as usize;
                        if target > compiled.instrs.len() {
                            return None;
                        }
                        if c != 0 {
                            let disp_pos = em.jmp_rel32_placeholder();
                            fixups.push((disp_pos, target));
                        }
                    }
                    _ => {}
                }
                let folded_branch = matches!(
                    &compiled.instrs[pc + 1],
                    Instr::JumpFalse(cond, _) if *cond == tmp
                ) || matches!(
                    &compiled.instrs[pc + 1],
                    Instr::JumpTrue(cond, _) if *cond == tmp
                );
                if folded_branch {
                    // Preserve old-pc targetability for branches.
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // Superinstruction fusion:
        //   LoadI32/LoadI64(tmp, c) ; BinOp(dst, op, x, tmp)
        // =>
        //   use immediate arithmetic/compare forms to avoid extra load/move.
        if pc + 1 < compiled.instrs.len() {
            let maybe_imm = match &compiled.instrs[pc] {
                Instr::LoadI32(tmp, v) => Some((*tmp, *v as i64)),
                Instr::LoadI64(tmp, v) => Some((*tmp, *v)),
                _ => None,
            };
            if let Some((tmp, c)) = maybe_imm {
                if let Instr::BinOp(dst, op, l, r) = &compiled.instrs[pc + 1] {
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
                            load_slot_rax(&mut em, live_reg);
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
                                BinOpKind::Mul => {
                                    if imm == 0 {
                                        em.xor_rax_rax();
                                    } else if imm == 1 {
                                        // no-op
                                    } else if imm == 3 {
                                        // Superoptimizer-selected sequence (shorter than imul imm32).
                                        em.lea_rax_rax_mul3();
                                    } else if imm == 5 {
                                        // Superoptimizer-selected sequence (shorter than imul imm32).
                                        em.lea_rax_rax_mul5();
                                    } else if imm == 9 {
                                        // Superoptimizer-selected sequence (shorter than imul imm32).
                                        em.lea_rax_rax_mul9();
                                    } else if imm > 0 && (imm as u32).is_power_of_two() {
                                        em.shl_rax_imm8((imm as u32).trailing_zeros() as u8);
                                    } else {
                                        em.imul_rax_imm32(imm);
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
                            store_rax_slot(&mut em, *dst);
                            // Preserve old-pc targetability for branches.
                            pc_to_off[pc + 1] = em.pos();
                            pc += 2;
                            continue;
                        }
                    }
                }
            }
        }

        // Superinstruction fusion:
        //   BinOp(tmp, op, l, r) ; Store(slot, tmp)
        // =>
        //   single fused arithmetic/comparison op materialized directly into slot.
        if pc + 1 < compiled.instrs.len() {
            if let (Instr::BinOp(tmp, op, l, r), Instr::Store(slot, src)) =
                (&compiled.instrs[pc], &compiled.instrs[pc + 1])
            {
                if tmp == src {
                    load_slot_rax(&mut em, *l);
                    load_slot_rcx(&mut em, *r);
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
                        _ => {}
                    }
                    store_rax_slot(&mut em, *slot);
                    // Preserve old-pc targetability for branches.
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }
        // Superinstruction fusion:
        //   BinOp(tmp, op, l, r) ; JumpFalse(tmp, off)
        // =>
        //   fused arithmetic/comparison + branch without temporary store/load.
        if pc + 1 < compiled.instrs.len() {
            if let (Instr::BinOp(tmp, op, l, r), Instr::JumpFalse(cond, off)) =
                (&compiled.instrs[pc], &compiled.instrs[pc + 1])
            {
                if tmp == cond {
                    let target = ((pc as i32) + 2 + *off) as usize;
                    if target > compiled.instrs.len() {
                        return None;
                    }
                    load_slot_rax(&mut em, *l);
                    load_slot_rcx(&mut em, *r);
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
                        _ => return None,
                    }
                    em.test_rax_rax();
                    let disp_pos = em.jz_rel32_placeholder();
                    fixups.push((disp_pos, target));
                    // Preserve old-pc targetability for branches.
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        // Superinstruction fusion:
        //   BinOp(tmp, op, l, r) ; JumpTrue(tmp, off)
        // =>
        //   fused arithmetic/comparison + branch without temporary store/load.
        if pc + 1 < compiled.instrs.len() {
            if let (Instr::BinOp(tmp, op, l, r), Instr::JumpTrue(cond, off)) =
                (&compiled.instrs[pc], &compiled.instrs[pc + 1])
            {
                if tmp == cond {
                    let target = ((pc as i32) + 2 + *off) as usize;
                    if target > compiled.instrs.len() {
                        return None;
                    }
                    load_slot_rax(&mut em, *l);
                    load_slot_rcx(&mut em, *r);
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
                        _ => return None,
                    }
                    em.test_rax_rax();
                    let disp_pos = em.jnz_rel32_placeholder();
                    fixups.push((disp_pos, target));
                    // Preserve old-pc targetability for branches.
                    pc_to_off[pc + 1] = em.pos();
                    pc += 2;
                    continue;
                }
            }
        }

        match &compiled.instrs[pc] {
            Instr::LoadI32(d, v) => {
                em.mov_rax_imm64(*v as i64);
                store_rax_slot(&mut em, *d);
            }
            Instr::LoadI64(d, v) => {
                em.mov_rax_imm64(*v);
                store_rax_slot(&mut em, *d);
            }
            Instr::LoadBool(d, v) => {
                em.mov_rax_imm64(i64::from(*v));
                store_rax_slot(&mut em, *d);
            }
            Instr::LoadUnit(d) => {
                em.mov_rax_imm64(0);
                store_rax_slot(&mut em, *d);
            }
            Instr::Move(d, s) | Instr::Load(d, s) => {
                load_slot_rax(&mut em, *s);
                store_rax_slot(&mut em, *d);
            }
            Instr::Store(slot, s) => {
                load_slot_rax(&mut em, *s);
                store_rax_slot(&mut em, *slot);
            }
            Instr::BinOp(d, op, l, r) => {
                load_slot_rax(&mut em, *l);
                load_slot_rcx(&mut em, *r);
                match op {
                    BinOpKind::Add => em.add_rax_rcx(),
                    BinOpKind::Sub => em.sub_rax_rcx(),
                    BinOpKind::Mul => em.imul_rax_rcx(),
                    BinOpKind::Div => { em.cqo(); em.idiv_rcx(); }
                    BinOpKind::Rem => { em.cqo(); em.idiv_rcx(); em.mov_rax_rdx(); }
                    BinOpKind::Eq => { em.cmp_rax_rcx(); em.setcc_al(0x94); em.movzx_rax_al(); }
                    BinOpKind::Ne => { em.cmp_rax_rcx(); em.setcc_al(0x95); em.movzx_rax_al(); }
                    BinOpKind::Lt => { em.cmp_rax_rcx(); em.setcc_al(0x9C); em.movzx_rax_al(); }
                    BinOpKind::Le => { em.cmp_rax_rcx(); em.setcc_al(0x9E); em.movzx_rax_al(); }
                    BinOpKind::Gt => { em.cmp_rax_rcx(); em.setcc_al(0x9F); em.movzx_rax_al(); }
                    BinOpKind::Ge => { em.cmp_rax_rcx(); em.setcc_al(0x9D); em.movzx_rax_al(); }
                    _ => return None,
                }
                store_rax_slot(&mut em, *d);
            }
            Instr::Jump(off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                if target > compiled.instrs.len() { return None; }
                let disp_pos = em.jmp_rel32_placeholder();
                fixups.push((disp_pos, target));
            }
            Instr::JumpFalse(cond, off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                if target > compiled.instrs.len() { return None; }
                load_slot_rax(&mut em, *cond);
                em.test_rax_rax();
                let disp_pos = em.jz_rel32_placeholder();
                fixups.push((disp_pos, target));
            }
            Instr::JumpTrue(cond, off) => {
                let target = ((pc as i32) + 1 + *off) as usize;
                if target > compiled.instrs.len() { return None; }
                load_slot_rax(&mut em, *cond);
                em.test_rax_rax();
                let disp_pos = em.jnz_rel32_placeholder();
                fixups.push((disp_pos, target));
            }
            Instr::Return(r) => {
                load_slot_rax(&mut em, *r);
                em.ret();
            }
            Instr::ReturnUnit => {
                em.mov_rax_imm64(0);
                em.ret();
            }
            Instr::Nop => {}
            _ => return None,
        }
        pc += 1;
    }

    pc_to_off[compiled.instrs.len()] = em.pos();
    em.mov_rax_imm64(0);
    em.ret();

    for (disp_pos, target_pc) in fixups {
        let target_off = *pc_to_off.get(target_pc)? as isize;
        let next_ip = (disp_pos + 4) as isize;
        let rel = target_off - next_ip;
        let rel = i32::try_from(rel).ok()?;
        em.buf[disp_pos..disp_pos + 4].copy_from_slice(&rel.to_le_bytes());
    }

    let mem = ExecMem::new(&em.buf)?;
    Some(NativeCode { slot_count: compiled.slot_count, mem })
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
