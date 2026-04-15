// jules/src/tracing_jit.rs
// TRACING JIT COMPILER

#![allow(dead_code)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem;
use std::ptr;

use crate::interp::{Instr, RuntimeError, Value};

// §1  PLATFORM-SPECIFIC MEMORY ALLOCATION 
#[cfg(target_os = "linux")]
const MAP_ANONYMOUS: i32 = 0x20;
#[cfg(target_os = "macos")]
const MAP_ANONYMOUS: i32 = 0x1000;

const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const PROT_EXEC: i32 = 4;
const MAP_PRIVATE: i32 = 0x02;

#[cfg(unix)]
extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
    fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

// §2  TRACE DATA STRUCTURES
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ValueType {
    I64 = 0, F64 = 1, Bool = 2, Unit = 3, Tensor = 4, Unknown = 255,
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::I64(_) | Value::I32(_) | Value::I8(_) | Value::I16(_) |
            Value::U8(_) | Value::U16(_) | Value::U32(_) | Value::U64(_) => ValueType::I64,
            Value::F64(_) | Value::F32(_) => ValueType::F64,
            Value::Bool(_) => ValueType::Bool,
            Value::Unit => ValueType::Unit,
            Value::Tensor(_) => ValueType::Tensor,
            _ => ValueType::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Guard {
    pub slot: u16,
    pub expected_type: ValueType,
}

#[derive(Debug, Clone)]
pub struct TraceInstruction {
    pub original_pc: usize,
    pub instruction: Instr,
    pub guard: Option<Guard>,
}

#[derive(Debug, Clone)]
pub struct SideExit {
    pub buffer_offset: usize, // Byte offset in JIT code where guard/branch diverges
    pub fallback_pc: usize,   // Interpreter PC to resume on failure
    pub is_loop_exit: bool,
}

#[derive(Debug, Clone)]
pub struct PatchSite {
    pub buffer_offset: usize, // Where rel32 displacement starts
    pub target_label: usize,  // Logical label ID
}

#[derive(Debug, Clone)]
pub struct Trace {
    pub id: u32,
    pub entry_pc: usize,
    pub instructions: Vec<TraceInstruction>,
    pub guards: Vec<Guard>,
    pub side_exits: Vec<SideExit>,
    pub execution_count: u64,
    pub next_label_id: usize,
}

// =============================================================================
// §3  TRACE RECORDER
// =============================================================================
pub struct TraceRecorder {
    current_trace: Option<Trace>,
    next_trace_id: u32,
    traces: Vec<Trace>,
    trace_selection: HashMap<u64, u32>, // entry_pc -> trace_id
}

impl TraceRecorder {
    pub fn new() -> Self {
        Self {
            current_trace: None,
            next_trace_id: 0,
            traces: Vec::new(),
            trace_selection: HashMap::new(),
        }
    }

    pub fn start_recording(&mut self, entry_pc: usize) {
        self.current_trace = Some(Trace {
            id: self.next_trace_id,
            entry_pc,
            instructions: Vec::with_capacity(256),
            guards: Vec::with_capacity(64),
            side_exits: Vec::with_capacity(16),
            execution_count: 0,
            next_label_id: 1,
        });
        self.next_trace_id += 1;
    }

    pub fn record_instruction(&mut self, instr: &Instr, pc: usize) {
        if let Some(ref mut trace) = self.current_trace {
            trace.instructions.push(TraceInstruction {
                original_pc: pc,
                instruction: instr.clone(),
                guard: None,
            });
        }
    }

    pub fn record_guard(&mut self, slot: u16, expected_type: ValueType) {
        if let Some(ref mut trace) = self.current_trace {
            let guard = Guard { slot, expected_type };
            trace.guards.push(guard);
            if let Some(last) = trace.instructions.last_mut() {
                last.guard = Some(guard);
            }
        }
    }

    pub fn record_side_exit(&mut self, fallback_pc: usize, is_loop: bool) {
        if let Some(ref mut trace) = self.current_trace {
            trace.side_exits.push(SideExit {
                buffer_offset: 0, // Patched during compilation
                fallback_pc,
                is_loop_exit: is_loop,
            });
        }
    }

    pub fn finish_recording(&mut self) -> Option<u32> {
        if let Some(mut trace) = self.current_trace.take() {
            let id = trace.id;
            let pc = trace.entry_pc;
            self.traces.push(trace);
            self.trace_selection.insert(pc as u64, id);
            Some(id)
        } else { None }
    }

    pub fn find_trace(&self, entry_pc: usize) -> Option<u32> {
        self.trace_selection.get(&(entry_pc as u64)).copied()
    }

    pub fn get_trace(&self, id: u32) -> Option<&Trace> { self.traces.get(id as usize) }
    pub fn get_trace_mut(&mut self, id: u32) -> Option<&mut Trace> { self.traces.get_mut(id as usize) }
}

// §4  EXECUTABLE MEMORY (Zero Dep)
pub struct ExecutableMemory {
    ptr: *mut u8,
    len: usize,
}

impl ExecutableMemory {
    #[cfg(unix)]
    pub fn new(code: &[u8]) -> Result<Self, String> {
        let len = code.len().max(1);
        let ptr = unsafe {
            mmap(ptr::null_mut(), len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
        };
        if ptr.is_null() || ptr as usize == usize::MAX {
            return Err("mmap failed".into());
        }

        unsafe { ptr::copy_nonoverlapping(code.as_ptr(), ptr as *mut u8, code.len()); }

        if unsafe { mprotect(ptr, len, PROT_READ | PROT_EXEC) } != 0 {
            unsafe { munmap(ptr, len) };
            return Err("mprotect failed".into());
        }

        Ok(Self { ptr: ptr as *mut u8, len })
    }

    pub fn entry_point(&self) -> *mut u8 { self.ptr }
}

impl Drop for ExecutableMemory {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            mprotect(self.ptr as *mut _, self.len, PROT_READ | PROT_WRITE);
            munmap(self.ptr as *mut _, self.len);
        }
    }
}

// §5  NATIVE CODE GENERATOR (x86-64)
#[derive(Debug, Clone, Copy)]
enum Reg { RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI }

pub struct NativeCodeGenerator {
    code: Vec<u8>,
    labels: HashMap<usize, usize>, // label_id -> buffer_offset
    patch_sites: Vec<PatchSite>,
    deopt_stub_offset: usize,
}

impl NativeCodeGenerator {
    pub fn new() -> Self {
        Self { code: Vec::with_capacity(4096), labels: HashMap::new(), patch_sites: Vec::new(), deopt_stub_offset: 0 }
    }

    pub fn compile_trace(&mut self, trace: &Trace, deopt_trampoline: usize) -> Result<CompiledTrace, String> {
        self.code.clear();
        self.labels.clear();
        self.patch_sites.clear();

        // 1. ABI-Compliant Prologue
        self.push_rbp();
        self.mov_rbp_rsp();
        self.and_rsp_16_aligned();
        self.push_callee_saved();

        // 2. Emit Guards (enforced in machine code)
        self.deopt_stub_offset = self.code.len();
        self.emit_guard_checks(&trace.guards, deopt_trampoline)?;

        // 3. Emit Instructions
        for instr in &trace.instructions {
            self.emit_instruction(&instr.instruction)?;
        }

        // 4. Emit Epilogue & Deopt Stub
        let exit_label = self.next_label();
        self.emit_ret();
        self.labels.insert(exit_label, self.code.len());
        self.emit_deopt_stub(deopt_trampoline)?;

        // 5. Backpatch Jumps
        self.backpatch_jumps()?;

        // 6. Allocate & Return
        let exec_mem = ExecutableMemory::new(&self.code)?;
        Ok(CompiledTrace {
            trace_id: trace.id,
            entry_point: exec_mem.entry_point(),
            memory: exec_mem,
            guard_count: trace.guards.len(),
            instruction_count: trace.instructions.len(),
            side_exit_table: trace.side_exits.iter().map(|se| (se.buffer_offset, se.fallback_pc)).collect(),
        })
    }

    // --- Guard Emission ---
    fn emit_guard_checks(&mut self, guards: &[Guard], deopt_addr: usize) -> Result<(), String> {
        for guard in guards {
            // rsi points to type_tags array
            self.movzx_al_rsi_off(guard.slot);
            self.cmp_al_imm8(guard.expected_type as u8);
            
            // jne deopt_trampoline
            let deopt_label = self.next_label();
            self.jne_label(deopt_label);
            self.labels.insert(deopt_label, deopt_addr); // Patch to trampoline address
        }
        Ok(())
    }

    // --- Instruction Emission ---
    fn emit_instruction(&mut self, instr: &Instr) -> Result<(), String> {
        match instr {
            Instr::LoadI32(dst, val) => {
                self.mov_eax_imm32(*val as i32);
                self.store_rax_to_slot(*dst);
            }
            Instr::LoadI64(dst, val) => {
                self.mov_rax_imm64(*val);
                self.store_rax_to_slot(*dst);
            }
            Instr::Add(dst, lhs, rhs) => {
                self.load_slot_to_reg(*lhs, Reg::RAX);
                self.load_slot_to_reg(*rhs, Reg::RCX);
                self.add_rax_rcx();
                self.store_rax_to_slot(*dst);
            }
            Instr::Sub(dst, lhs, rhs) => {
                self.load_slot_to_reg(*lhs, Reg::RAX);
                self.load_slot_to_reg(*rhs, Reg::RCX);
                self.sub_rax_rcx();
                self.store_rax_to_slot(*dst);
            }
            Instr::Mul(dst, lhs, rhs) => {
                self.load_slot_to_reg(*lhs, Reg::RAX);
                self.load_slot_to_reg(*rhs, Reg::RCX);
                self.imul_rax_rcx();
                self.store_rax_to_slot(*dst);
            }
            Instr::JumpFalse(cond, target_pc) => {
                self.load_slot_to_reg(*cond, Reg::RAX);
                self.test_rax_rax();
                
                let jump_label = self.next_label();
                self.jz_label(jump_label);
                
                // Record patch site for later resolution
                self.patch_sites.push(PatchSite {
                    buffer_offset: self.code.len() - 4,
                    target_label: jump_label,
                });
                // Store side-exit mapping
                if let Some(trace) = self.trace_context.as_ref() {
                    if let Some(last_exit) = trace.side_exits.last() {
                        // In a real compiler, target_pc maps to a label or side-exit stub
                        self.labels.insert(jump_label, target_pc); // Simplified mapping
                    }
                }
            }
            Instr::Return(slot) => {
                self.load_slot_to_reg(*slot, Reg::RAX);
                // Fall through to epilogue
            }
            _ => return Err(format!("Unsupported instruction: {:?}", instr)),
        }
        Ok(())
    }

    // --- x86-64 Encoding Primitives ---
    fn push_rbp(&mut self) { self.b(0x55); }
    fn mov_rbp_rsp(&mut self) { self.bb(0x48, 0x89, 0xE5); }
    fn and_rsp_16_aligned(&mut self) { self.bbbb(0x48, 0x83, 0xE4, 0xF0); }
    fn push_callee_saved(&mut self) { self.bb(0x41, 0x54); self.bb(0x41, 0x55); } // r12, r13
    fn pop_callee_saved(&mut self) { self.bb(0x41, 0x5D); self.bb(0x41, 0x5C); }
    fn emit_ret(&mut self) { self.pop_callee_saved(); self.mov_rsp_rbp(); self.pop_rbp(); self.b(0xC3); }
    fn mov_rsp_rbp(&mut self) { self.bb(0x48, 0x89, 0xEC); }
    fn pop_rbp(&mut self) { self.b(0x5D); }

    fn load_slot_to_reg(&mut self, slot: u16, reg: Reg) {
        let reg_code = reg as u8;
        // mov reg64, [rdi + slot*8] -> REX.W + 8B + ModR/M + disp32
        self.bb(0x48, 0x8B);
        self.modrm(2, reg_code, 7); // mod=2, reg, rm=7(rdi)
        self.i32((slot as i32) * 8);
    }

    fn store_rax_to_slot(&mut self, slot: u16) {
        // mov [rdi + slot*8], rax -> REX.W + 89 + ModR/M + disp32
        self.bb(0x48, 0x89);
        self.modrm(2, 0, 7);
        self.i32((slot as i32) * 8);
    }

    fn movzx_al_rsi_off(&mut self, off: u16) {
        // movzx eax, byte [rsi + off] -> 0F B6 46 xx
        self.b(0x0F); self.b(0xB6);
        self.modrm(0, 0, 6); // rm=6(rsi)
        self.b(off as u8);
    }

    fn cmp_al_imm8(&mut self, imm: u8) { self.bb(0x3C, imm); }
    fn test_rax_rax(&mut self) { self.bbb(0x48, 0x85, 0xC0); }
    fn add_rax_rcx(&mut self) { self.bbb(0x48, 0x01, 0xC8); }
    fn sub_rax_rcx(&mut self) { self.bbb(0x48, 0x29, 0xC8); }
    fn imul_rax_rcx(&mut self) { self.bbbb(0x48, 0x0F, 0xAF, 0xC1); }
    fn mov_eax_imm32(&mut self, val: i32) { self.b(0xB8); self.i32(val); }
    fn mov_rax_imm64(&mut self, val: i64) { self.bb(0x48, 0xB8); self.i64(val); }

    fn jne_label(&mut self, label: usize) { self.rel32_jump(0x0F, 0x85, label); }
    fn jz_label(&mut self, label: usize) { self.rel32_jump(0x0F, 0x84, label); }

    fn rel32_jump(&mut self, prefix: u8, opcode: u8, label: usize) {
        self.b(prefix); self.b(opcode);
        self.i32(0); // Placeholder
        self.patch_sites.push(PatchSite { buffer_offset: self.code.len() - 4, target_label: label });
    }

    fn emit_deopt_stub(&mut self, deopt_trampoline: usize) -> Result<(), String> {
        // Simple deopt: call trampoline with failure code, return -1
        // mov eax, -1; ret
        // In production: save registers, snapshot state, call deopt_handler
        self.b(0xB8); self.i32(-1);
        self.b(0xC3);
        Ok(())
    }

    // --- Backpatching ---
    fn backpatch_jumps(&mut self) -> Result<(), String> {
        for patch in &self.patch_sites {
            if let Some(&target) = self.labels.get(&patch.target_label) {
                let current_ip = patch.buffer_offset + 4;
                let rel = (target as isize - current_ip as isize) as i32;
                let buf = &mut self.code;
                buf[patch.buffer_offset..patch.buffer_offset + 4].copy_from_slice(&rel.to_le_bytes());
            } else {
                return Err(format!("Unresolved label: {}", patch.target_label));
            }
        }
        Ok(())
    }

    // --- Helpers ---
    fn b(&mut self, v: u8) { self.code.push(v); }
    fn bb(&mut self, a: u8, b: u8) { self.code.extend_from_slice(&[a, b]); }
    fn bbb(&mut self, a: u8, b: u8, c: u8) { self.code.extend_from_slice(&[a, b, c]); }
    fn bbbb(&mut self, a: u8, b: u8, c: u8, d: u8) { self.code.extend_from_slice(&[a, b, c, d]); }
    fn i32(&mut self, v: i32) { self.code.extend_from_slice(&v.to_le_bytes()); }
    fn i64(&mut self, v: i64) { self.code.extend_from_slice(&v.to_le_bytes()); }
    fn modrm(&mut self, mode: u8, reg: u8, rm: u8) { self.b((mode << 6) | ((reg & 7) << 3) | (rm & 7)); }
    fn next_label(&mut self) -> usize { let l = self.code.len(); self.labels.insert(self.labels.len() + 1, l); self.labels.len() }
    // Note: trace_context is temporarily unused in this isolated file, kept for architectural completeness
    trace_context: Option<Trace>,
}

// =============================================================================
// §6  COMPILED TRACE & DEOPT
// =============================================================================
#[derive(Clone)]
pub struct CompiledTrace {
    pub trace_id: u32,
    pub entry_point: *mut u8,
    pub memory: ExecutableMemory,
    pub guard_count: usize,
    pub instruction_count: usize,
    pub side_exit_table: Vec<(usize, usize)>,
}

impl CompiledTrace {
    /// fn(slots: *mut i64, types: *const u8) -> i64
    pub unsafe fn execute(&self, slots: *mut i64, types: *const u8) -> i64 {
        let func: unsafe extern "C" fn(*mut i64, *const u8) -> i64 = mem::transmute(self.entry_point);
        func(slots, types)
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_deopt_trampoline() -> i64 { -1 }

// §7  TRACING JIT INTEGRATION
pub struct TracingJIT {
    pub recorder: TraceRecorder,
    pub codegen: NativeCodeGenerator,
    pub trace_trigger: u64,
    pub compile_trigger: u64,
    pub traces_recorded: u64,
    pub traces_compiled: u64,
    pub deoptimizations: u64,
}

impl TracingJIT {
    pub fn new() -> Self {
        Self {
            recorder: TraceRecorder::new(),
            codegen: NativeCodeGenerator::new(),
            trace_trigger: 100,
            compile_trigger: 10,
            traces_recorded: 0,
            traces_compiled: 0,
            deoptimizations: 0,
        }
    }

    pub fn should_start_tracing(&self, count: u64) -> bool { count == self.trace_trigger }
    pub fn should_compile(&self, trace: &Trace) -> bool { trace.execution_count >= self.compile_trigger }

    pub fn execute_with_jit(
        &mut self,
        entry_pc: usize,
        slots: &mut [i64],
        types: &mut [u8],
        instructions: &[Instr],
    ) -> Result<Value, RuntimeError> {
        if let Some(trace_id) = self.recorder.find_trace(entry_pc) {
            if let Some(trace) = self.recorder.get_trace(trace_id) {
                if self.should_compile(trace) && !trace.instructions.is_empty() {
                    match self.codegen.compile_trace(trace, jit_deopt_trampoline as usize) {
                        Ok(compiled) => {
                            self.traces_compiled += 1;
                            let res = unsafe { compiled.execute(slots.as_mut_ptr(), types.as_ptr()) };
                            if res >= 0 {
                                return Ok(Value::I64(res));
                            }
                            self.deoptimizations += 1; // Guard failed or deopt triggered
                        }
                        Err(_) => { self.deoptimizations += 1; }
                    }
                }
                if let Some(t) = self.recorder.get_trace_mut(trace_id) { t.execution_count += 1; }
            }
        }

        // Fallback to interpreter (or start recording)
        if self.should_start_tracing(100) {
            self.recorder.start_recording(entry_pc);
            self.traces_recorded += 1;
        }
        
        Err(RuntimeError::new("Interpreter fallback: JIT trace hot but not yet compiled, or guard failed"))
    }
}
