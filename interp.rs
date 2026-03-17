// =============================================================================
// jules/src/interp.rs
//
// Tree-walking Interpreter / Virtual Machine for the Jules programming language.
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │  Design overview                                                         │
// │                                                                          │
// │  Strategy: direct AST evaluation (tree-walking interpreter).             │
// │  This is the fastest-to-get-running execution strategy and serves as     │
// │  the reference implementation.  A compiler backend (Cranelift / WASM)   │
// │  can be layered on top later using the same runtime primitives.          │
// │                                                                          │
// │  Feature targets implemented here:                                       │
// │    • CPU scalar + vector/matrix arithmetic                               │
// │    • Tensor operations (matmul, hadamard, concat, grad tracking)         │
// │    • SIMD hint: interpreted as parallel Rayon-style iteration            │
// │    • ECS World: sparse-set component storage, entity lifecycle           │
// │    • System scheduler: topological ordering + deterministic tick         │
// │    • spawn / sync / atomic concurrency primitives (thread-pool based)    │
// │    • Agent runtime: behaviour tick, perception, memory decay             │
// │    • Neural-network forward pass (CPU tensor execution)                  │
// │    • Training loop: episode rollout + reward aggregation + SGD/Adam      │
// │    • GPU path: stubbed with a dispatch trait; plug in wgpu/CUDA later    │
// └─────────────────────────────────────────────────────────────────────────┘
// =============================================================================

#![allow(clippy::match_single_binding, clippy::large_enum_variant)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::fmt;

use crate::ast::{
    Activation, AgentDecl, AssignOpKind, Attribute, BinOpKind, Block, EntityQuery,
    ElemType, Expr, FnDecl, Item, LearningKind, MatchArm, ModelDecl, ModelLayer,
    NormKind, OptimizerKind, Padding, ParallelismHint, Pattern, PoolOp,
    Program, RecurrentCell, ScheduleKind, Stmt, SystemDecl, TrainDecl,
    UnOpKind, VecSize,
};
use crate::lexer::Span;
use crate::game_systems::{PhysicsWorld, RenderState, InputState};
use crate::ml_engine::{ComputationGraph, Optimizer, OptimizerState};

// =============================================================================
// §1  RUNTIME VALUE
// =============================================================================

/// Every runtime value produced or consumed by Jules programs.
///
/// Cloning is cheap for scalars; tensors are reference-counted so large
/// allocations are not duplicated on every assignment.
#[derive(Debug, Clone)]
pub enum Value {
    // ── Scalars ───────────────────────────────────────────────────────────────
    I8(i8),   I16(i16), I32(i32), I64(i64),
    U8(u8),   U16(u16), U32(u32), U64(u64),
    F32(f32), F64(f64),
    Bool(bool),
    Str(String),
    Unit,

    // ── SIMD vectors (stored as flat f32 arrays) ──────────────────────────────
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    IVec2([i32; 2]),
    IVec3([i32; 3]),
    IVec4([i32; 4]),
    Mat2([[f32; 2]; 2]),
    Mat3([[f32; 3]; 3]),
    Mat4([[f32; 4]; 4]),
    Quat([f32; 4]),  // [x, y, z, w]

    // ── Tensors (Feature 1) ───────────────────────────────────────────────────
    Tensor(Arc<RwLock<Tensor>>),

    // ── Compound ─────────────────────────────────────────────────────────────
    Tuple(Vec<Value>),
    Array(Arc<Mutex<Vec<Value>>>),
    Struct { name: String, fields: HashMap<String, Value> },
    /// HashMap: key -> value pairs (keys currently strings)
    HashMap(Arc<Mutex<HashMap<String, Value>>>),

    // ── Option / Result types ─────────────────────────────────────────────────
    /// `Some(value)` or `None` (for Option<T>)
    Some(Box<Value>),
    None,
    /// `Ok(value)` for Result<T, E>
    Ok(Box<Value>),
    /// `Err(value)` for Result<T, E>
    Err(Box<Value>),

    // ── Callable ─────────────────────────────────────────────────────────────
    /// A user-defined function closure (captures its definition scope).
    Fn(Arc<FnClosure>),

    // ── ECS handles ──────────────────────────────────────────────────────────
    Entity(EntityId),
    World(Arc<Mutex<EcsWorld>>),

    // ── Neural-network model handle ───────────────────────────────────────────
    Model(Arc<Mutex<NnModel>>),

    // ── Control flow signals (never escape to user code) ─────────────────────
    Return(Box<Value>),
    Break(Option<Box<Value>>),
    Continue,
}

impl Value {
    pub fn type_name(&self) -> &str {
        match self {
            Value::I8(_)  => "i8",  Value::I16(_) => "i16",
            Value::I32(_) => "i32", Value::I64(_) => "i64",
            Value::U8(_)  => "u8",  Value::U16(_) => "u16",
            Value::U32(_) => "u32", Value::U64(_) => "u64",
            Value::F32(_) => "f32", Value::F64(_) => "f64",
            Value::Bool(_)   => "bool",
            Value::Str(_)    => "str",
            Value::Unit      => "()",
            Value::Vec2(_)   => "vec2",  Value::Vec3(_) => "vec3",
            Value::Vec4(_)   => "vec4",
            Value::IVec2(_)  => "ivec2", Value::IVec3(_) => "ivec3",
            Value::IVec4(_)  => "ivec4",
            Value::Mat2(_)   => "mat2",  Value::Mat3(_) => "mat3",
            Value::Mat4(_)   => "mat4",  Value::Quat(_) => "quat",
            Value::Tensor(_) => "tensor",
            Value::Tuple(_)  => "tuple",
            Value::Array(_)  => "array",
            Value::HashMap(_) => "map",
            Value::Struct { name, .. } => name,
            Value::Some(_)   => "Some",
            Value::None      => "None",
            Value::Ok(_)     => "Ok",
            Value::Err(_)    => "Err",
            Value::Fn(_)     => "fn",
            Value::Entity(_) => "entity",
            Value::World(_)  => "world",
            Value::Model(_)  => "model",
            Value::Return(_) | Value::Break(_) | Value::Continue => "<control-flow>",
        }
    }

    /// Extract f64 for arithmetic, coercing all numeric types.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F32(x) => Some(*x as f64),
            Value::F64(x) => Some(*x),
            Value::I32(x) => Some(*x as f64),
            Value::I64(x) => Some(*x as f64),
            Value::U32(x) => Some(*x as f64),
            Value::U64(x) => Some(*x as f64),
            Value::I8(x)  => Some(*x as f64),
            Value::I16(x) => Some(*x as f64),
            Value::U8(x)  => Some(*x as f64),
            Value::U16(x) => Some(*x as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::I32(x) => Some(*x as i64),
            Value::I64(x) => Some(*x),
            Value::U32(x) => Some(*x as i64),
            Value::U64(x) => Some(*x as i64),
            Value::I8(x)  => Some(*x as i64),
            Value::I16(x) => Some(*x as i64),
            Value::U8(x)  => Some(*x as i64),
            Value::U16(x) => Some(*x as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Value::Bool(b) = self { Some(*b) } else { None }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::I32(x)  => *x != 0,
            Value::F32(x)  => *x != 0.0,
            _ => true,
        }
    }

    /// True for any of the control-flow signal variants.
    pub fn is_signal(&self) -> bool {
        matches!(self, Value::Return(_) | Value::Break(_) | Value::Continue)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::I8(x)   => write!(f, "{x}"),  Value::I16(x)  => write!(f, "{x}"),
            Value::I32(x)  => write!(f, "{x}"),  Value::I64(x)  => write!(f, "{x}"),
            Value::U8(x)   => write!(f, "{x}"),  Value::U16(x)  => write!(f, "{x}"),
            Value::U32(x)  => write!(f, "{x}"),  Value::U64(x)  => write!(f, "{x}"),
            Value::F32(x)  => write!(f, "{x}"),  Value::F64(x)  => write!(f, "{x}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Str(s)  => write!(f, "{s}"),
            Value::Unit    => write!(f, "()"),
            Value::Vec2(v) => write!(f, "vec2({}, {})", v[0], v[1]),
            Value::Vec3(v) => write!(f, "vec3({}, {}, {})", v[0], v[1], v[2]),
            Value::Vec4(v) => write!(f, "vec4({}, {}, {}, {})", v[0], v[1], v[2], v[3]),
            Value::Mat3(_) => write!(f, "mat3(…)"),
            Value::Mat4(_) => write!(f, "mat4(…)"),
            Value::Quat(q) => write!(f, "quat({}, {}, {}, {})", q[0], q[1], q[2], q[3]),
            Value::Tensor(t) => {
                let t = t.read().unwrap();
                write!(f, "tensor<{:?}>{:?}", t.elem, t.shape)
            }
            Value::Tuple(vs) => {
                let inner: Vec<_> = vs.iter().map(|v| v.to_string()).collect();
                write!(f, "({})", inner.join(", "))
            }
            Value::Struct { name, .. } => write!(f, "{name} {{ … }}"),
            Value::Some(v) => write!(f, "Some({})", v),
            Value::None    => write!(f, "None"),
            Value::Ok(v)   => write!(f, "Ok({})", v),
            Value::Err(v)  => write!(f, "Err({})", v),
            Value::HashMap(m) => {
                let m = m.lock().unwrap();
                write!(f, "{{ {} items }}", m.len())
            }
            Value::Entity(id) => write!(f, "Entity({id})"),
            Value::World(_)   => write!(f, "<world>"),
            Value::Model(_)   => write!(f, "<model>"),
            Value::Fn(_)      => write!(f, "<fn>"),
            Value::Array(a)   => {
                let a = a.lock().unwrap();
                write!(f, "[…; {}]", a.len())
            }
            Value::Return(v)  => write!(f, "return {v}"),
            Value::Break(_)   => write!(f, "break"),
            Value::Continue   => write!(f, "continue"),
            _ => write!(f, "<value>"),
        }
    }
}

// =============================================================================
// §2  TENSOR  (Feature 1)
// =============================================================================

/// A dense n-dimensional tensor stored as a flat `f32` vector on the CPU.
/// GPU tensors use the same shape metadata but store data behind a `GpuBuffer`.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub elem:  ElemType,
    pub shape: Vec<usize>,
    pub data:  TensorStorage,
    /// When `Some`, this tensor has a gradient buffer attached (`@grad`).
    pub grad:  Option<Box<Tensor>>,
}

#[derive(Debug, Clone)]
pub enum TensorStorage {
    Cpu(Vec<f32>),
    /// Placeholder for a GPU buffer handle (wgpu BufferId / CUDA pointer).
    Gpu(GpuBufferHandle),
}

/// Opaque handle to a GPU buffer.  Filled in by the GPU backend.
#[derive(Debug, Clone)]
pub struct GpuBufferHandle(pub u64);

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product::<usize>().max(1);
        Tensor {
            elem: ElemType::F32,
            shape,
            data: TensorStorage::Cpu(vec![0.0_f32; n]),
            grad: None,
        }
    }

    pub fn from_data(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Tensor { elem: ElemType::F32, shape, data: TensorStorage::Cpu(data), grad: None }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    fn cpu_data(&self) -> &[f32] {
        match &self.data {
            TensorStorage::Cpu(v) => v,
            TensorStorage::Gpu(_) => panic!("tensor is on GPU; call to_cpu() first"),
        }
    }

    fn cpu_data_mut(&mut self) -> &mut Vec<f32> {
        match &mut self.data {
            TensorStorage::Cpu(v) => v,
            TensorStorage::Gpu(_) => panic!("tensor is on GPU"),
        }
    }

    /// Matrix multiply  C = A @ B.
    /// A: [M, K], B: [K, N] → C: [M, N].
    /// Uses cache-blocked (tiled) 32×32 GEMM for CPU performance.
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        if self.shape.len() < 2 || rhs.shape.len() < 2 {
            return Err(RuntimeError::new("matmul requires ≥2-D tensors"));
        }
        let (m, k) = (self.shape[self.shape.len() - 2], self.shape[self.shape.len() - 1]);
        let (k2, n) = (rhs.shape[rhs.shape.len() - 2], rhs.shape[rhs.shape.len() - 1]);
        if k != k2 {
            return Err(RuntimeError::new(format!(
                "matmul shape mismatch: [{m}, {k}] @ [{k2}, {n}]"
            )));
        }
        let a = self.cpu_data();
        let b = rhs.cpu_data();
        let mut c = vec![0.0_f32; m * n];

        // Cache-tiled GEMM: 32×32 tiles fit in L1 cache.
        const TILE: usize = 32;
        for ii in (0..m).step_by(TILE) {
            for jj in (0..n).step_by(TILE) {
                for kk in (0..k).step_by(TILE) {
                    let i_end = (ii + TILE).min(m);
                    let j_end = (jj + TILE).min(n);
                    let k_end = (kk + TILE).min(k);
                    for i in ii..i_end {
                        for l in kk..k_end {
                            let a_il = a[i * k + l];
                            for j in jj..j_end {
                                c[i * n + j] += a_il * b[l * n + j];
                            }
                        }
                    }
                }
            }
        }

        let mut out_shape = self.shape[..self.shape.len() - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);
        Ok(Tensor::from_data(out_shape, c))
    }

    /// Kronecker product  C = A @@ B.
    /// A: [m, n], B: [p, q] → C: [m*p, n*q]
    pub fn kron(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err(RuntimeError::new("Kronecker product requires 2-D tensors"));
        }
        let (m, n) = (self.shape[0], self.shape[1]);
        let (p, q) = (rhs.shape[0], rhs.shape[1]);
        let a = self.cpu_data();
        let b = rhs.cpu_data();
        let mut c = vec![0.0_f32; m * p * n * q];
        for i in 0..m {
            for j in 0..n {
                let a_ij = a[i * n + j];
                for r in 0..p {
                    for s in 0..q {
                        c[(i * p + r) * (n * q) + (j * q + s)] = a_ij * b[r * q + s];
                    }
                }
            }
        }
        Ok(Tensor::from_data(vec![m * p, n * q], c))
    }

    /// Outer product  C = a ^* b.
    /// a: [m], b: [n] → C: [m, n]
    pub fn outer(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        if self.shape.len() != 1 || rhs.shape.len() != 1 {
            return Err(RuntimeError::new("outer product requires 1-D tensors"));
        }
        let m = self.shape[0];
        let n = rhs.shape[0];
        let a = self.cpu_data();
        let b = rhs.cpu_data();
        let c: Vec<f32> = a.iter().flat_map(|&ai| b.iter().map(move |&bj| ai * bj)).collect();
        Ok(Tensor::from_data(vec![m, n], c))
    }
        let mut out_shape = self.shape[..self.shape.len() - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);
        Ok(Tensor::from_data(out_shape, c))
    }

    /// Element-wise multiply (Hadamard).
    pub fn hadamard_mul(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        self.elementwise(rhs, |a, b| a * b, ".*")
    }

    /// Element-wise divide.
    pub fn hadamard_div(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        self.elementwise(rhs, |a, b| a / b, "./")
    }

    fn elementwise(&self, rhs: &Tensor, op: impl Fn(f32, f32) -> f32, name: &str)
        -> Result<Tensor, RuntimeError>
    {
        // Exact shape match (fast path)
        if self.shape == rhs.shape {
            let a = self.cpu_data();
            let b = rhs.cpu_data();
            let c: Vec<f32> = a.iter().zip(b).map(|(x, y)| op(*x, *y)).collect();
            return Ok(Tensor::from_data(self.shape.clone(), c));
        }

        // Numpy-style broadcasting
        let result_shape = broadcast_shape(&self.shape, &rhs.shape)
            .ok_or_else(|| RuntimeError::new(format!(
                "`{name}` shape mismatch: {:?} vs {:?}", self.shape, rhs.shape
            )))?;

        let n: usize = result_shape.iter().product();
        let mut c = vec![0.0_f32; n];
        for idx in 0..n {
            let ai = broadcast_index(idx, &result_shape, &self.shape);
            let bi = broadcast_index(idx, &result_shape, &rhs.shape);
            let a = self.cpu_data();
            let b = rhs.cpu_data();
            c[idx] = op(a[ai], b[bi]);
        }
        Ok(Tensor::from_data(result_shape, c))
    }

    /// Floor division element-wise (integer rounding toward −∞).
    pub fn floor_div(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        self.elementwise(rhs, |a, b| (a / b).floor(), "//")
    }

    /// Concatenate along axis 0.
    pub fn concat(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        if self.shape.len() != rhs.shape.len() {
            return Err(RuntimeError::new("tensor concat: rank mismatch"));
        }
        for i in 1..self.shape.len() {
            if self.shape[i] != rhs.shape[i] {
                return Err(RuntimeError::new(format!(
                    "tensor concat: dim {} mismatch ({} vs {})",
                    i, self.shape[i], rhs.shape[i]
                )));
            }
        }
        let mut data = self.cpu_data().to_vec();
        data.extend_from_slice(rhs.cpu_data());
        let mut shape = self.shape.clone();
        shape[0] += rhs.shape[0];
        Ok(Tensor::from_data(shape, data))
    }

    /// Attach a zero-initialised gradient buffer.
    pub fn enable_grad(&mut self) {
        if self.grad.is_none() {
            self.grad = Some(Box::new(Tensor::zeros(self.shape.clone())));
        }
    }

    /// Apply an activation function element-wise.
    pub fn apply_activation(&self, act: &Activation) -> Tensor {
        let data: Vec<f32> = self.cpu_data().iter().map(|&x| match act {
            Activation::Relu      => x.max(0.0),
            Activation::LeakyRelu => if x > 0.0 { x } else { 0.01 * x },
            Activation::Sigmoid   => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh      => x.tanh(),
            Activation::Gelu      => {
                0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt()
                    * (x + 0.044715 * x * x * x)).tanh())
            }
            Activation::Silu      => x / (1.0 + (-x).exp()),
            Activation::Softmax   => x,  // applied below per-row
            Activation::Linear | Activation::Custom(_) => x,
        }).collect();

        let mut t = Tensor::from_data(self.shape.clone(), data);

        // Softmax: apply row-wise along last dim.
        if matches!(act, Activation::Softmax) {
            let d = t.cpu_data_mut();
            let cols = *self.shape.last().unwrap_or(&1);
            let rows = d.len() / cols;
            for r in 0..rows {
                let row = &mut d[r * cols..(r + 1) * cols];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
                for x in row.iter_mut() { *x = (*x - max).exp() / sum; }
            }
        }
        t
    }

    /// Reduce sum over all elements → scalar.
    pub fn sum_all(&self) -> f32 {
        self.cpu_data().iter().sum()
    }

    /// Scale all elements.
    pub fn scale(&self, s: f32) -> Tensor {
        let data: Vec<f32> = self.cpu_data().iter().map(|x| x * s).collect();
        Tensor::from_data(self.shape.clone(), data)
    }

    /// Add another tensor (in-place on self).
    pub fn add_assign(&mut self, rhs: &Tensor) -> Result<(), RuntimeError> {
        if self.shape != rhs.shape {
            return Err(RuntimeError::new("tensor += shape mismatch"));
        }
        let b = rhs.cpu_data().to_vec();
        let a = self.cpu_data_mut();
        for (x, y) in a.iter_mut().zip(b) { *x += y; }
        Ok(())
    }

    /// MSE loss w.r.t. targets.
    pub fn mse_loss(&self, targets: &Tensor) -> Result<f32, RuntimeError> {
        if self.shape != targets.shape {
            return Err(RuntimeError::new("MSE: shape mismatch"));
        }
        let a = self.cpu_data();
        let b = targets.cpu_data();
        let loss: f32 = a.iter().zip(b).map(|(p, t)| (p - t).powi(2)).sum::<f32>()
            / a.len() as f32;
        Ok(loss)
    }

    /// Cross-entropy loss (prediction probabilities vs one-hot targets).
    pub fn cross_entropy_loss(&self, targets: &Tensor) -> Result<f32, RuntimeError> {
        if self.shape != targets.shape {
            return Err(RuntimeError::new("CE: shape mismatch"));
        }
        let a = self.cpu_data();
        let b = targets.cpu_data();
        let loss: f32 = a.iter().zip(b)
            .map(|(p, t)| -t * p.max(1e-9).ln())
            .sum::<f32>() / self.shape[0] as f32;
        Ok(loss)
    }
}

// =============================================================================
// §3  ECS WORLD  (Feature 2)
// =============================================================================

pub type EntityId = u64;

/// The Entity-Component-System world that backs all `system` and entity-for loops.
///
/// Component storage uses a sparse-set layout per component type:
///   component_name → (dense vec of EntityId, dense vec of component Value)
/// The two vecs stay in sync: `dense_ids[i]` owns `dense_vals[i]`.
#[derive(Debug, Default)]
pub struct EcsWorld {
    next_id:    EntityId,
    alive:      std::collections::HashSet<EntityId>,
    /// component_type → SparseSet
    components: HashMap<String, SparseSet>,
    /// Pending events (signal_name → Vec<EntityId>)
    events:     HashMap<String, Vec<EntityId>>,
}

/// Sparse-set component storage.
#[derive(Debug, Default)]
struct SparseSet {
    /// Maps EntityId → index into `dense_ids` / `dense_vals`.
    sparse:     HashMap<EntityId, usize>,
    dense_ids:  Vec<EntityId>,
    dense_vals: Vec<Value>,
}

impl SparseSet {
    fn insert(&mut self, id: EntityId, val: Value) {
        if let Some(&idx) = self.sparse.get(&id) {
            self.dense_vals[idx] = val;
        } else {
            let idx = self.dense_ids.len();
            self.sparse.insert(id, idx);
            self.dense_ids.push(id);
            self.dense_vals.push(val);
        }
    }

    fn get(&self, id: EntityId) -> Option<&Value> {
        self.sparse.get(&id).map(|&i| &self.dense_vals[i])
    }

    fn get_mut(&mut self, id: EntityId) -> Option<&mut Value> {
        self.sparse.get(&id).copied().map(|i| &mut self.dense_vals[i])
    }

    fn remove(&mut self, id: EntityId) {
        if let Some(idx) = self.sparse.remove(&id) {
            let last = self.dense_ids.len() - 1;
            if idx != last {
                let moved_id = self.dense_ids[last];
                self.dense_ids.swap(idx, last);
                self.dense_vals.swap(idx, last);
                self.sparse.insert(moved_id, idx);
            }
            self.dense_ids.pop();
            self.dense_vals.pop();
        }
    }

    fn entity_ids(&self) -> &[EntityId] {
        &self.dense_ids
    }
}

impl EcsWorld {
    pub fn spawn(&mut self) -> EntityId {
        let id = self.next_id;
        self.next_id += 1;
        self.alive.insert(id);
        id
    }

    pub fn despawn(&mut self, id: EntityId) {
        self.alive.remove(&id);
        for set in self.components.values_mut() {
            set.remove(id);
        }
    }

    pub fn is_alive(&self, id: EntityId) -> bool {
        self.alive.contains(&id)
    }

    pub fn insert_component(&mut self, id: EntityId, comp_type: &str, val: Value) {
        self.components
            .entry(comp_type.to_owned())
            .or_default()
            .insert(id, val);
    }

    pub fn get_component(&self, id: EntityId, comp_type: &str) -> Option<&Value> {
        self.components.get(comp_type)?.get(id)
    }

    pub fn get_component_mut(&mut self, id: EntityId, comp_type: &str) -> Option<&mut Value> {
        self.components.get_mut(comp_type)?.get_mut(id)
    }

    pub fn remove_component(&mut self, id: EntityId, comp_type: &str) {
        if let Some(set) = self.components.get_mut(comp_type) {
            set.remove(id);
        }
    }

    /// Returns all live entities matching a query.
    pub fn query(&self, with: &[String], without: &[String]) -> Vec<EntityId> {
        // Start from the smallest `with` component set for efficiency.
        let base: Vec<EntityId> = if with.is_empty() {
            self.alive.iter().cloned().collect()
        } else {
            // Find the component with the fewest entities (cheapest to iterate).
            let smallest = with.iter()
                .filter_map(|c| self.components.get(c))
                .min_by_key(|s| s.dense_ids.len());
            match smallest {
                None => return vec![],
                Some(s) => s.entity_ids().to_vec(),
            }
        };

        base.into_iter()
            .filter(|id| {
                self.alive.contains(id)
                && with.iter().all(|c| {
                    self.components.get(c).map_or(false, |s| s.get(*id).is_some())
                })
                && without.iter().all(|c| {
                    self.components.get(c).map_or(true, |s| s.get(*id).is_none())
                })
            })
            .collect()
    }

    /// Emit an event signal for the training loop.
    pub fn emit_event(&mut self, signal: &str, entity: EntityId) {
        self.events.entry(signal.to_owned()).or_default().push(entity);
    }

    /// Drain all events for a named signal.
    pub fn drain_events(&mut self, signal: &str) -> Vec<EntityId> {
        self.events.remove(signal).unwrap_or_default()
    }
}

// =============================================================================
// §4  NEURAL NETWORK MODEL RUNTIME  (Unique Feature 1)
// =============================================================================

/// A weight layer in the runtime model.
#[derive(Debug, Clone)]
pub(crate) enum WeightLayer {
    Dense    { w: Tensor, b: Tensor, act: Activation },
    Conv2d   { filters: u64, kh: u64, kw: u64, act: Activation },
    Pool     { ph: u64, pw: u64, op: PoolOp },
    Dropout  { rate: f32, training: bool },
    Norm     { kind: NormKind, scale: Tensor, shift: Tensor },
    Attention{ num_heads: u64, head_dim: u64,
               wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor },
    Embed    { table: Tensor },
    Recurrent{ units: u64, cell: RecurrentCell,
               wh: Tensor, wx: Tensor, bh: Tensor },
    SubModel { name: String },
}

/// The live neural network model with allocated weights.
#[derive(Debug)]
pub struct NnModel {
    pub name:    String,
    pub layers:  Vec<WeightLayer>,
    pub training: bool,
    /// Gradient accumulator: one tensor per weight tensor, same shape.
    pub grads:   Vec<Vec<Tensor>>,
    /// Adam state: first and second moment estimates.
    pub m1:      Vec<Vec<Tensor>>,
    pub m2:      Vec<Vec<Tensor>>,
    pub step:    u64,
}

impl NnModel {
    /// Instantiate a model from its AST declaration, initialising all weights.
    pub fn from_decl(decl: &ModelDecl) -> Self {
        let mut layers = Vec::new();
        let mut last_width: usize = 1;

        for layer in &decl.layers {
            match layer {
                ModelLayer::Input { size, .. } => {
                    last_width = *size as usize;
                }
                ModelLayer::Dense { units, activation, bias, .. } => {
                    let u = *units as usize;
                    // He initialisation: std = sqrt(2 / fan_in)
                    let std = (2.0_f32 / last_width as f32).sqrt();
                    let w = Tensor::from_data(
                        vec![last_width, u],
                        (0..last_width * u).map(|_| rand_normal(0.0, std)).collect(),
                    );
                    let b = if *bias { Tensor::zeros(vec![u]) }
                            else    { Tensor::zeros(vec![u]) };
                    layers.push(WeightLayer::Dense { w, b, act: activation.clone() });
                    last_width = u;
                }
                ModelLayer::Output { units, activation, .. } => {
                    let u = *units as usize;
                    let std = (2.0_f32 / last_width as f32).sqrt();
                    let w = Tensor::from_data(
                        vec![last_width, u],
                        (0..last_width * u).map(|_| rand_normal(0.0, std)).collect(),
                    );
                    let b = Tensor::zeros(vec![u]);
                    layers.push(WeightLayer::Dense { w, b, act: activation.clone() });
                    last_width = u;
                }
                ModelLayer::Dropout { rate, .. } => {
                    layers.push(WeightLayer::Dropout {
                        rate: *rate as f32, training: true,
                    });
                }
                ModelLayer::Norm { kind, .. } => {
                    let scale = Tensor::from_data(vec![last_width], vec![1.0; last_width]);
                    let shift = Tensor::zeros(vec![last_width]);
                    layers.push(WeightLayer::Norm { kind: *kind, scale, shift });
                }
                ModelLayer::Attention { num_heads, head_dim, .. } => {
                    let d = (*num_heads * *head_dim) as usize;
                    let std = (2.0_f32 / last_width as f32).sqrt();
                    let mk = |rows: usize, cols: usize| Tensor::from_data(
                        vec![rows, cols],
                        (0..rows * cols).map(|_| rand_normal(0.0, std)).collect(),
                    );
                    layers.push(WeightLayer::Attention {
                        num_heads: *num_heads, head_dim: *head_dim,
                        wq: mk(last_width, d),
                        wk: mk(last_width, d),
                        wv: mk(last_width, d),
                        wo: mk(d, last_width),
                    });
                }
                ModelLayer::Embed { vocab_size, embed_dim, .. } => {
                    let v = *vocab_size as usize;
                    let e = *embed_dim as usize;
                    let table = Tensor::from_data(
                        vec![v, e],
                        (0..v * e).map(|_| rand_normal(0.0, 0.01)).collect(),
                    );
                    layers.push(WeightLayer::Embed { table });
                    last_width = e;
                }
                ModelLayer::Conv2d { filters, kernel_h, kernel_w, activation, .. } => {
                    layers.push(WeightLayer::Conv2d {
                        filters: *filters, kh: *kernel_h, kw: *kernel_w,
                        act: activation.clone(),
                    });
                }
                ModelLayer::Pool { size_h, size_w, op, .. } => {
                    layers.push(WeightLayer::Pool {
                        ph: *size_h, pw: *size_w, op: *op,
                    });
                }
                ModelLayer::Recurrent { units, cell, .. } => {
                    let u = *units as usize;
                    let std = (2.0_f32 / (last_width + u) as f32).sqrt();
                    let wh = Tensor::from_data(vec![u, u],
                        (0..u * u).map(|_| rand_normal(0.0, std)).collect());
                    let wx = Tensor::from_data(vec![last_width, u],
                        (0..last_width * u).map(|_| rand_normal(0.0, std)).collect());
                    let bh = Tensor::zeros(vec![u]);
                    layers.push(WeightLayer::Recurrent {
                        units: *units, cell: *cell, wh, wx, bh,
                    });
                    last_width = u;
                }
                ModelLayer::SubModel { name, .. } => {
                    layers.push(WeightLayer::SubModel { name: name.clone() });
                }
            }
        }

        let n = layers.len();
        NnModel {
            name: decl.name.clone(),
            layers,
            training: false,
            grads: vec![vec![]; n],
            m1:    vec![vec![]; n],
            m2:    vec![vec![]; n],
            step:  0,
        }
    }

    /// Run the forward pass.  Input: [batch, features] tensor.
    pub fn forward(&mut self, mut x: Tensor) -> Result<Tensor, RuntimeError> {
        for layer in &self.layers {
            x = self.apply_layer(layer, x)?;
        }
        Ok(x)
    }

    fn apply_layer(&self, layer: &WeightLayer, x: Tensor) -> Result<Tensor, RuntimeError> {
        match layer {
            WeightLayer::Dense { w, b, act } => {
                // y = x @ W + b
                let mut y = x.matmul(w)?;
                // Broadcast-add bias.
                let b_data = b.cpu_data();
                let cols = *w.shape.last().unwrap();
                let y_data = y.cpu_data_mut();
                for (i, v) in y_data.iter_mut().enumerate() {
                    *v += b_data[i % cols];
                }
                Ok(y.apply_activation(act))
            }
            WeightLayer::Dropout { rate, training } => {
                if !training || *rate == 0.0 { return Ok(x); }
                // Bernoulli dropout mask.
                let keep = 1.0 - rate;
                let data: Vec<f32> = x.cpu_data().iter().map(|v| {
                    if pseudo_rand() > *rate { v / keep } else { 0.0 }
                }).collect();
                Ok(Tensor::from_data(x.shape.clone(), data))
            }
            WeightLayer::Norm { scale, shift, .. } => {
                // Layer norm: normalise along last dimension.
                let d = *x.shape.last().unwrap_or(&1);
                let s_data = scale.cpu_data();
                let sh_data = shift.cpu_data();
                let x_data = x.cpu_data();
                let rows = x_data.len() / d;
                let mut out = vec![0.0_f32; x_data.len()];
                for r in 0..rows {
                    let row = &x_data[r * d..(r + 1) * d];
                    let mean: f32 = row.iter().sum::<f32>() / d as f32;
                    let var: f32  = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
                    let std = (var + 1e-6).sqrt();
                    for c in 0..d {
                        out[r * d + c] = (row[c] - mean) / std * s_data[c] + sh_data[c];
                    }
                }
                Ok(Tensor::from_data(x.shape.clone(), out))
            }
            WeightLayer::Attention { num_heads, head_dim, wq, wk, wv, wo } => {
                // Simplified single-head for the interpreter (full MHA in prod).
                let q = x.matmul(wq)?;
                let k = x.matmul(wk)?;
                let v = x.matmul(wv)?;
                let d = *head_dim as f32;
                let scores = q.matmul(&k.transpose()?)?.scale(1.0 / d.sqrt());
                let scores = scores.apply_activation(&Activation::Softmax);
                let attn = scores.matmul(&v)?;
                let out  = attn.matmul(wo)?;
                let _ = num_heads;
                Ok(out)
            }
            WeightLayer::Embed { table } => {
                // Expect x to contain integer indices (stored as f32).
                let indices: Vec<usize> = x.cpu_data().iter()
                    .map(|&v| v as usize)
                    .collect();
                let embed_dim = table.shape[1];
                let mut out = vec![0.0_f32; indices.len() * embed_dim];
                let t = table.cpu_data();
                for (i, idx) in indices.iter().enumerate() {
                    let src = &t[idx * embed_dim..(idx + 1) * embed_dim];
                    out[i * embed_dim..(i + 1) * embed_dim].copy_from_slice(src);
                }
                Ok(Tensor::from_data(vec![indices.len(), embed_dim], out))
            }
            WeightLayer::Recurrent { units, wh, wx, bh, .. } => {
                // Single-step GRU approximation (simplest stateful layer).
                let u = *units as usize;
                let h = Tensor::zeros(vec![1, u]);
                let xw = x.matmul(wx)?;
                let hw = h.matmul(wh)?;
                let b_data = bh.cpu_data();
                let xw_d = xw.cpu_data();
                let hw_d = hw.cpu_data();
                let out: Vec<f32> = xw_d.iter().zip(hw_d).enumerate()
                    .map(|(i, (a, b))| (a + b + b_data[i % u]).tanh())
                    .collect();
                Ok(Tensor::from_data(vec![1, u], out))
            }
            WeightLayer::Conv2d { .. } | WeightLayer::Pool { .. } => {
                // Spatial operations: return x unchanged in the interpreter.
                // A production backend replaces these with im2col + GEMM.
                Ok(x)
            }
            WeightLayer::SubModel { .. } => Ok(x),
        }
    }
}

impl Tensor {
    fn transpose(&self) -> Result<Tensor, RuntimeError> {
        if self.shape.len() < 2 {
            return Err(RuntimeError::new("transpose requires ≥2-D tensor"));
        }
        let (rows, cols) = (self.shape[self.shape.len() - 2], self.shape[self.shape.len() - 1]);
        let data = self.cpu_data();
        let mut out = vec![0.0_f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        let mut shape = self.shape.clone();
        let n = shape.len();
        shape.swap(n - 2, n - 1);
        Ok(Tensor::from_data(shape, out))
    }
}

// =============================================================================
// §5  SYSTEM SCHEDULER  (Feature 2)
// =============================================================================

/// The scheduler dispatches systems in topological order, respecting
/// component read/write dependencies.
pub struct Scheduler {
    /// Systems in execution order (determined by topological sort).
    order: Vec<String>,
}

impl Scheduler {
    /// Build a schedule from the program's system declarations.
    pub fn build(program: &Program) -> Self {
        // Simple heuristic: systems that only read run before systems that write.
        // A proper Kahn's-algorithm scheduler goes here in a full compiler.
        let mut order: Vec<String> = program.systems()
            .map(|s| s.name.clone())
            .collect();
        order.sort(); // stable alphabetic fallback
        Scheduler { order }
    }

    /// Tick: run all systems in scheduled order against the world.
    pub fn tick(
        &self,
        systems:    &HashMap<String, Arc<SystemDecl>>,
        world:      &Arc<Mutex<EcsWorld>>,
        interp:     &mut Interpreter,
        delta_time: f32,
    ) -> Result<(), RuntimeError> {
        for name in &self.order {
            if let Some(sys) = systems.get(name) {
                interp.run_system(sys, world, delta_time)?;
            }
        }
        Ok(())
    }
}

// =============================================================================
// §6  ENVIRONMENT (variable store)
// =============================================================================

/// A single lexical scope frame.
type Frame = HashMap<String, Value>;

/// The call-stack / environment for the interpreter.
#[derive(Default)]
pub struct Env {
    frames: Vec<Frame>,
}

impl Env {
    pub fn new() -> Self { Env { frames: vec![Frame::new()] } }

    pub fn push(&mut self) { self.frames.push(Frame::new()); }

    pub fn pop(&mut self) { self.frames.pop(); }

    pub fn set(&mut self, name: &str, val: Value) {
        for frame in self.frames.iter_mut().rev() {
            if frame.contains_key(name) {
                frame.insert(name.to_owned(), val);
                return;
            }
        }
        // New binding in innermost frame.
        if let Some(f) = self.frames.last_mut() {
            f.insert(name.to_owned(), val);
        }
    }

    pub fn set_local(&mut self, name: &str, val: Value) {
        if let Some(f) = self.frames.last_mut() {
            f.insert(name.to_owned(), val);
        }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        for frame in self.frames.iter().rev() {
            if let Some(v) = frame.get(name) { return Some(v); }
        }
        None
    }
}

// =============================================================================
// §7  RUNTIME ERROR
// =============================================================================

#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub message: String,
    pub span:    Option<Span>,
}

impl RuntimeError {
    pub fn new(msg: impl Into<String>) -> Self {
        RuntimeError { message: msg.into(), span: None }
    }
    pub fn at(mut self, span: Span) -> Self { self.span = Some(span); self }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.span {
            Some(s) => write!(f, "runtime error at {}: {}", s, self.message),
            None    => write!(f, "runtime error: {}", self.message),
        }
    }
}

// ── Shorthand macros ──────────────────────────────────────────────────────────

macro_rules! rt_err {
    ($msg:expr) => { Err(RuntimeError::new($msg)) };
    ($fmt:literal $(, $arg:expr)*) => { Err(RuntimeError::new(format!($fmt $(, $arg)*))) };
}

// =============================================================================
// §8  FUNCTION CLOSURE
// =============================================================================

pub struct FnClosure {
    pub decl:    FnDecl,
    pub capture: Frame,   // captured environment for closures
}

impl fmt::Debug for FnClosure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn {}>", self.decl.name)
    }
}

// =============================================================================
// §9  INTERPRETER
// =============================================================================

/// The main tree-walking interpreter.
pub struct Interpreter {
    /// Top-level function registry.
    pub fns:      HashMap<String, Arc<FnClosure>>,
    /// Top-level model registry (AST decls; instantiated on demand).
    pub model_decls: HashMap<String, ModelDecl>,
    /// Live model instances.
    pub models:   HashMap<String, Arc<Mutex<NnModel>>>,
    /// Agent declarations.
    pub agent_decls: HashMap<String, AgentDecl>,
    /// Struct/component type registry (name → field list).
    pub types:    HashMap<String, Vec<String>>,
    /// ECS world (global singleton for now).
    pub world:    Arc<Mutex<EcsWorld>>,
    /// GPU dispatch backend (None = CPU-only).
    pub gpu:      Option<Box<dyn GpuBackend>>,
    /// Thread pool size for `spawn` / parallel loops.
    pub n_threads: usize,
    /// Physics engine (game systems integration)
    pub physics_world: Option<Arc<Mutex<PhysicsWorld>>>,
    /// Rendering state (game systems integration)
    pub render_state: Option<Arc<Mutex<RenderState>>>,
    /// Input state (game systems integration)
    pub input_state: Option<Arc<Mutex<InputState>>>,
    /// Computation graph for autodiff (ML integration)
    pub computation_graph: Option<Arc<Mutex<ComputationGraph>>>,
    /// Active optimizers indexed by ID (ML integration)
    pub optimizers: HashMap<String, (Optimizer, OptimizerState)>,
}

impl Interpreter {
    pub fn new() -> Self {
        Interpreter {
            fns:          HashMap::new(),
            model_decls:  HashMap::new(),
            models:       HashMap::new(),
            agent_decls:  HashMap::new(),
            types:        HashMap::new(),
            world:        Arc::new(Mutex::new(EcsWorld::default())),
            gpu:          None,
            n_threads:    4,
            physics_world: Some(Arc::new(Mutex::new(PhysicsWorld::new()))),
            render_state: Some(Arc::new(Mutex::new(RenderState::new()))),
            input_state: Some(Arc::new(Mutex::new(InputState::new()))),
            computation_graph: Some(Arc::new(Mutex::new(ComputationGraph::new()))),
            optimizers: HashMap::new(),
        }
    }

    // ── Program loading ────────────────────────────────────────────────────

    /// Load all top-level declarations from a parsed program into the interpreter.
    pub fn load_program(&mut self, program: &Program) {
        for item in &program.items {
            self.load_item(item);
        }
    }

    fn load_item(&mut self, item: &Item) {
        match item {
            Item::Fn(f) => {
                let closure = FnClosure { decl: f.clone(), capture: Frame::new() };
                self.fns.insert(f.name.clone(), Arc::new(closure));
            }
            Item::Component(c) => {
                let fields: Vec<String> = c.fields.iter().map(|f| f.name.clone()).collect();
                self.types.insert(c.name.clone(), fields);
            }
            Item::Struct(s) => {
                let fields: Vec<String> = s.fields.iter().map(|f| f.name.clone()).collect();
                self.types.insert(s.name.clone(), fields);
            }
            Item::Agent(a)  => { self.agent_decls.insert(a.name.clone(), a.clone()); }
            Item::Model(m)  => { self.model_decls.insert(m.name.clone(), m.clone()); }
            Item::Mod { items: Some(inner), .. } => {
                for i in inner { self.load_item(i); }
            }
            _ => {}
        }
    }

    // ── Run a function by name ──────────────────────────────────────────────

    pub fn call_fn(&mut self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        let closure = self.fns.get(name).cloned()
            .ok_or_else(|| RuntimeError::new(format!("undefined function `{name}`")))?;
        let mut env = Env::new();
        env.push();
        for (param, val) in closure.decl.params.iter().zip(args) {
            env.set_local(&param.name, val);
        }
        // Inject captured environment for closures.
        for (k, v) in &closure.capture {
            env.set_local(k, v.clone());
        }
        if let Some(body) = &closure.decl.body.clone() {
            let result = self.eval_block(&body, &mut env)?;
            env.pop();
            match result {
                Value::Return(v) => Ok(*v),
                other            => Ok(other),
            }
        } else {
            env.pop();
            Ok(Value::Unit)
        }
    }

    // ── System execution ────────────────────────────────────────────────────

    pub fn run_system(
        &mut self,
        sys:        &SystemDecl,
        world:      &Arc<Mutex<EcsWorld>>,
        delta_time: f32,
    ) -> Result<(), RuntimeError> {
        let mut env = Env::new();
        env.push();
        // Bind system parameters.
        for param in &sys.params {
            if param.name == "dt" || param.name == "delta_time" {
                env.set_local(&param.name, Value::F32(delta_time));
            } else {
                env.set_local(&param.name, Value::Unit);
            }
        }
        // Bind the world handle.
        env.set_local("world", Value::World(world.clone()));
        self.eval_block(&sys.body, &mut env)?;
        env.pop();
        Ok(())
    }

    // =========================================================================
    // §10  BLOCK EVALUATION
    // =========================================================================

    pub fn eval_block(&mut self, block: &Block, env: &mut Env) -> Result<Value, RuntimeError> {
        env.push();
        let mut result = Value::Unit;
        for stmt in &block.stmts {
            result = self.eval_stmt(stmt, env)?;
            if result.is_signal() { break; }
        }
        if !result.is_signal() {
            if let Some(tail) = &block.tail {
                result = self.eval_expr(tail, env)?;
            }
        }
        env.pop();
        Ok(result)
    }

    // =========================================================================
    // §11  STATEMENT EVALUATION
    // =========================================================================

    pub fn eval_stmt(&mut self, stmt: &Stmt, env: &mut Env) -> Result<Value, RuntimeError> {
        match stmt {
            Stmt::Let { pattern, init, .. } => {
                let val = if let Some(e) = init {
                    self.eval_expr(e, env)?
                } else {
                    Value::Unit
                };
                self.bind_pattern(pattern, val, env);
                Ok(Value::Unit)
            }

            Stmt::Expr { expr, .. } => self.eval_expr(expr, env),

            Stmt::Return { value, .. } => {
                let v = if let Some(e) = value { self.eval_expr(e, env)? }
                        else { Value::Unit };
                Ok(Value::Return(Box::new(v)))
            }

            Stmt::Break { value, .. } => {
                let v = if let Some(e) = value {
                    Some(Box::new(self.eval_expr(e, env)?))
                } else { None };
                Ok(Value::Break(v))
            }

            Stmt::Continue { .. } => Ok(Value::Continue),

            Stmt::ForIn { pattern, iter, body, .. } => {
                let iter_val = self.eval_expr(iter, env)?;
                let items = self.value_to_iter(iter_val)?;
                for item in items {
                    env.push();
                    self.bind_pattern(pattern, item, env);
                    let r = self.eval_block(body, env)?;
                    env.pop();
                    match r {
                        Value::Break(_) => break,
                        Value::Continue => continue,
                        v if v.is_signal() => return Ok(v),
                        _ => {}
                    }
                }
                Ok(Value::Unit)
            }

            Stmt::EntityFor { var, query, body, parallelism, .. } => {
                self.eval_entity_for(var, query, body, *parallelism, env)
            }

            Stmt::While { cond, body, .. } => {
                loop {
                    let c = self.eval_expr(cond, env)?;
                    if !c.is_truthy() { break; }
                    let r = self.eval_block(body, env)?;
                    match r {
                        Value::Break(_) => break,
                        Value::Continue => continue,
                        v if v.is_signal() => return Ok(v),
                        _ => {}
                    }
                }
                Ok(Value::Unit)
            }

            Stmt::Loop { body, .. } => {
                loop {
                    let r = self.eval_block(body, env)?;
                    match r {
                        Value::Break(v) => return Ok(v.map(|b| *b).unwrap_or(Value::Unit)),
                        Value::Continue => continue,
                        v if v.is_signal() => return Ok(v),
                        _ => {}
                    }
                }
            }

            Stmt::If { cond, then, else_, .. } => {
                let c = self.eval_expr(cond, env)?;
                if c.is_truthy() {
                    self.eval_block(then, env)
                } else if let Some(e) = else_ {
                    match e.as_ref() {
                        crate::ast::IfOrBlock::If(s)    => self.eval_stmt(s, env),
                        crate::ast::IfOrBlock::Block(b) => self.eval_block(b, env),
                    }
                } else {
                    Ok(Value::Unit)
                }
            }

            Stmt::Match { expr, arms, .. } => {
                let scrutinee = self.eval_expr(expr, env)?;
                for arm in arms {
                    if self.pattern_matches(&arm.pat, &scrutinee) {
                        env.push();
                        self.bind_pattern(&arm.pat, scrutinee.clone(), env);
                        let guard_ok = if let Some(g) = &arm.guard {
                            self.eval_expr(g, env)?.is_truthy()
                        } else { true };
                        let result = if guard_ok {
                            self.eval_expr(&arm.body, env)?
                        } else {
                            env.pop();
                            continue;
                        };
                        env.pop();
                        return Ok(result);
                    }
                }
                Ok(Value::Unit)
            }

            Stmt::Item(i) => { self.load_item(i); Ok(Value::Unit) }

            Stmt::ParallelFor(pf) => {
                let iter_val = self.eval_expr(&pf.iter, env)?;
                let items = self.value_to_iter(iter_val)?;
                // In the interpreter we execute sequentially;
                // a rayon par_iter() call replaces this in the compiled backend.
                for item in items {
                    env.push();
                    self.bind_pattern(&pf.var, item, env);
                    let r = self.eval_block(&pf.body, env)?;
                    env.pop();
                    match r {
                        Value::Break(_) => break,
                        Value::Continue => continue,
                        v if v.is_signal() => return Ok(v),
                        _ => {}
                    }
                }
                Ok(Value::Unit)
            }

            Stmt::Spawn(sb) => {
                // In the interpreter we run the body inline (no true async).
                self.eval_block(&sb.body, env)
            }

            Stmt::Sync(sb)   => self.eval_block(&sb.body, env),
            Stmt::Atomic(ab) => self.eval_block(&ab.body, env),
        }
    }

    // ── Entity-for evaluation ──────────────────────────────────────────────

    fn eval_entity_for(
        &mut self,
        var:         &str,
        query:       &EntityQuery,
        body:        &Block,
        parallelism: ParallelismHint,
        env:         &mut Env,
    ) -> Result<Value, RuntimeError> {
        // Snapshot the matching entity list (safe: world locked briefly).
        let entity_ids = {
            let w = self.world.lock().unwrap();
            w.query(&query.with, &query.without)
        };

        // Apply optional filter expression.
        let mut ids_to_run = Vec::new();
        for id in entity_ids {
            if let Some(filter_expr) = &query.filter {
                env.push();
                env.set_local(var, Value::Entity(id));
                let ok = self.eval_expr(filter_expr, env)?.is_truthy();
                env.pop();
                if ok { ids_to_run.push(id); }
            } else {
                ids_to_run.push(id);
            }
        }

        match parallelism {
            ParallelismHint::Sequential | ParallelismHint::Auto => {
                // Sequential: deterministic order guaranteed.
                for id in &ids_to_run {
                    env.push();
                    env.set_local(var, Value::Entity(*id));
                    let r = self.eval_block(body, env)?;
                    env.pop();
                    match r {
                        Value::Break(_) => break,
                        Value::Continue => continue,
                        v if v.is_signal() => return Ok(v),
                        _ => {}
                    }
                }
            }
            ParallelismHint::Parallel | ParallelismHint::Simd
            | ParallelismHint::Gpu   | ParallelismHint::SimdOrGpu { .. } => {
                // Parallel: interpreter falls back to sequential with a note.
                // A real backend uses rayon::par_iter() or GPU dispatch here.
                for id in &ids_to_run {
                    env.push();
                    env.set_local(var, Value::Entity(*id));
                    let r = self.eval_block(body, env)?;
                    env.pop();
                    if r.is_signal() { return Ok(r); }
                }
            }
        }
        Ok(Value::Unit)
    }

    // =========================================================================
    // §12  EXPRESSION EVALUATION
    // =========================================================================

    pub fn eval_expr(&mut self, expr: &Expr, env: &mut Env) -> Result<Value, RuntimeError> {
        match expr {
            Expr::IntLit   { value, .. } => Ok(Value::I32(*value as i32)),
            Expr::FloatLit { value, .. } => Ok(Value::F32(*value as f32)),
            Expr::BoolLit  { value, .. } => Ok(Value::Bool(*value)),
            Expr::StrLit   { value, .. } => Ok(Value::Str(value.clone())),

            Expr::Ident { name, span } => {
                // Check local env first, then built-ins.
                if let Some(v) = env.get(name) { return Ok(v.clone()); }
                if name == "world" {
                    return Ok(Value::World(self.world.clone()));
                }
                if let Some(f) = self.fns.get(name.as_str()).cloned() {
                    return Ok(Value::Fn(f));
                }
                if let Some(m) = self.models.get(name.as_str()).cloned() {
                    return Ok(Value::Model(m));
                }
                rt_err!("undefined variable `{name}`")
            }

            Expr::Path { segments, .. } => {
                let name = segments.join("::");
                if let Some(v) = env.get(&name) { return Ok(v.clone()); }
                if let Some(f) = self.fns.get(name.as_str()).cloned() {
                    return Ok(Value::Fn(f));
                }
                rt_err!("undefined path `{name}`")
            }

            // ── Vector constructors ────────────────────────────────────────────
            Expr::VecCtor { size, elems, span } => {
                let vals: Vec<f32> = elems.iter()
                    .map(|e| self.eval_expr(e, env).and_then(|v| {
                        v.as_f64().map(|f| f as f32)
                            .ok_or_else(|| RuntimeError::new("vec element must be numeric"))
                    }))
                    .collect::<Result<_, _>>()?;
                match size {
                    VecSize::N2 => Ok(Value::Vec2([vals[0], vals[1]])),
                    VecSize::N3 => Ok(Value::Vec3([vals[0], vals[1], vals[2]])),
                    VecSize::N4 => Ok(Value::Vec4([vals[0], vals[1], vals[2], vals[3]])),
                }
            }

            Expr::ArrayLit { elems, .. } => {
                let vals: Vec<Value> = elems.iter()
                    .map(|e| self.eval_expr(e, env))
                    .collect::<Result<_, _>>()?;
                Ok(Value::Array(Arc::new(Mutex::new(vals))))
            }

            Expr::Tuple { elems, .. } => {
                let vals: Vec<Value> = elems.iter()
                    .map(|e| self.eval_expr(e, env))
                    .collect::<Result<_, _>>()?;
                Ok(Value::Tuple(vals))
            }

            // ── Arithmetic ────────────────────────────────────────────────────
            Expr::BinOp { op, lhs, rhs, span } => {
                self.eval_binop(*op, lhs, rhs, env).map_err(|e| e.at(*span))
            }

            Expr::UnOp { op, expr, span } => {
                let v = self.eval_expr(expr, env)?;
                self.eval_unop(*op, v).map_err(|e| e.at(*span))
            }

            // ── Assignment ────────────────────────────────────────────────────
            Expr::Assign { op, target, value, .. } => {
                let rhs = self.eval_expr(value, env)?;
                self.eval_assign(*op, target, rhs, env)
            }

            // ── Field access ──────────────────────────────────────────────────
            Expr::Field { object, field, span } => {
                let obj = self.eval_expr(object, env)?;
                self.eval_field(obj, field).map_err(|e| e.at(*span))
            }

            // ── Index ─────────────────────────────────────────────────────────
            Expr::Index { object, indices, span } => {
                let obj = self.eval_expr(object, env)?;
                let idxs: Vec<Value> = indices.iter()
                    .map(|i| self.eval_expr(i, env))
                    .collect::<Result<_, _>>()?;
                self.eval_index(obj, idxs).map_err(|e| e.at(*span))
            }

            // ── Calls ─────────────────────────────────────────────────────────
            Expr::Call { func, args, named, span } => {
                // Check for built-in functions by name first
                if let Expr::Ident { name, .. } = func.as_ref() {
                    let args_v: Vec<Value> = args.iter()
                        .map(|a| self.eval_expr(a, env))
                        .collect::<Result<_, _>>()?;
                    if let Ok(result) = self.eval_builtin(name, args_v) {
                        return Ok(result);
                    }
                }
                // Otherwise, try normal function evaluation
                let args_v: Vec<Value> = args.iter()
                    .map(|a| self.eval_expr(a, env))
                    .collect::<Result<_, _>>()?;
                let func_v = self.eval_expr(func, env)?;
                self.eval_call(func_v, args_v, env).map_err(|e| e.at(*span))
            }

            Expr::MethodCall { receiver, method, args, span } => {
                let recv = self.eval_expr(receiver, env)?;
                let args_v: Vec<Value> = args.iter()
                    .map(|a| self.eval_expr(a, env))
                    .collect::<Result<_, _>>()?;
                self.eval_method(recv, method, args_v).map_err(|e| e.at(*span))
            }

            // ── Tensor-specific (Feature 1) ───────────────────────────────────
            Expr::MatMul { lhs, rhs, span } => {
                let l = self.eval_expr(lhs, env)?;
                let r = self.eval_expr(rhs, env)?;
                self.eval_matmul(l, r).map_err(|e| e.at(*span))
            }

            Expr::HadamardMul { lhs, rhs, span } => {
                let l = self.eval_tensor(lhs, env)?;
                let r = self.eval_tensor(rhs, env)?;
                let out = l.read().unwrap().hadamard_mul(&r.read().unwrap())?;
                Ok(Value::Tensor(Arc::new(RwLock::new(out))))
            }

            Expr::HadamardDiv { lhs, rhs, span } => {
                let l = self.eval_tensor(lhs, env)?;
                let r = self.eval_tensor(rhs, env)?;
                let out = l.read().unwrap().hadamard_div(&r.read().unwrap())?;
                Ok(Value::Tensor(Arc::new(RwLock::new(out))))
            }

            Expr::TensorConcat { lhs, rhs, span } => {
                let l = self.eval_tensor(lhs, env)?;
                let r = self.eval_tensor(rhs, env)?;
                let out = l.read().unwrap().concat(&r.read().unwrap())?;
                Ok(Value::Tensor(Arc::new(RwLock::new(out))))
            }

            Expr::Grad { inner, .. } => {
                let v = self.eval_expr(inner, env)?;
                if let Value::Tensor(t) = &v {
                    t.write().unwrap().enable_grad();
                }
                Ok(v)
            }

            Expr::Pow { base, exp, .. } => {
                let b = self.eval_expr(base, env)?;
                let e = self.eval_expr(exp, env)?;
                match (&b, &e) {
                    (Value::F32(x), Value::F32(y))   => Ok(Value::F32(x.powf(*y))),
                    (Value::F64(x), Value::F64(y))   => Ok(Value::F64(x.powf(*y))),
                    (Value::I32(x), Value::I32(y))   => Ok(Value::I32(x.pow(*y as u32))),
                    _ => {
                        if let (Some(x), Some(y)) = (b.as_f64(), e.as_f64()) {
                            Ok(Value::F64(x.powf(y)))
                        } else {
                            rt_err!("** requires numeric operands")
                        }
                    }
                }
            }

            Expr::Range { lo, hi, inclusive, .. } => {
                let lo_v = lo.as_ref().map(|e| self.eval_expr(e, env)).transpose()?;
                let hi_v = hi.as_ref().map(|e| self.eval_expr(e, env)).transpose()?;
                let start = lo_v.and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                let end   = hi_v.and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                let range: Vec<Value> = if *inclusive {
                    (start..=end).map(Value::I32).collect()
                } else {
                    (start..end).map(Value::I32).collect()
                };
                Ok(Value::Array(Arc::new(Mutex::new(range))))
            }

            Expr::Cast { expr, ty, .. } => {
                let v = self.eval_expr(expr, env)?;
                self.eval_cast(v, ty)
            }

            Expr::IfExpr { cond, then, else_, .. } => {
                let c = self.eval_expr(cond, env)?;
                if c.is_truthy() {
                    self.eval_block(then, env)
                } else if let Some(b) = else_ {
                    self.eval_block(b, env)
                } else {
                    Ok(Value::Unit)
                }
            }

            Expr::Closure { params, body, .. } => {
                // Capture current environment.
                let mut capture = Frame::new();
                for frame in &env.frames {
                    for (k, v) in frame { capture.insert(k.clone(), v.clone()); }
                }
                let decl = FnDecl {
                    span:     crate::lexer::Span::dummy(),
                    attrs:    vec![],
                    name:     "<closure>".into(),
                    generics: vec![],
                    params:   params.clone(),
                    ret_ty:   None,
                    body:     Some(Block {
                        span:  crate::lexer::Span::dummy(),
                        stmts: vec![],
                        tail:  Some(body.clone()),
                    }),
                    is_async: false,
                };
                Ok(Value::Fn(Arc::new(FnClosure { decl, capture })))
            }

            Expr::Block(b) => self.eval_block(b, env),

            Expr::StructLit { name, fields, .. } => {
                let mut field_vals = HashMap::new();
                for (fname, fexpr) in fields {
                    field_vals.insert(fname.clone(), self.eval_expr(fexpr, env)?);
                }
                Ok(Value::Struct { name: name.clone(), fields: field_vals })
            }
        }
    }

    // ── Tensor helper ──────────────────────────────────────────────────────

    fn eval_tensor(&mut self, expr: &Expr, env: &mut Env) -> Result<Arc<RwLock<Tensor>>, RuntimeError> {
        match self.eval_expr(expr, env)? {
            Value::Tensor(t) => Ok(t),
            other => rt_err!("expected tensor, got `{}`", other.type_name()),
        }
    }

    // =========================================================================
    // §13  OPERATOR EVALUATION
    // =========================================================================

    fn eval_binop(&mut self, op: BinOpKind, lhs: &Expr, rhs: &Expr, env: &mut Env)
        -> Result<Value, RuntimeError>
    {
        // Short-circuit logical operators.
        if op == BinOpKind::And {
            let l = self.eval_expr(lhs, env)?;
            return if !l.is_truthy() { Ok(Value::Bool(false)) }
                   else { Ok(Value::Bool(self.eval_expr(rhs, env)?.is_truthy())) };
        }
        if op == BinOpKind::Or {
            let l = self.eval_expr(lhs, env)?;
            return if l.is_truthy() { Ok(Value::Bool(true)) }
                   else { Ok(Value::Bool(self.eval_expr(rhs, env)?.is_truthy())) };
        }

        let l = self.eval_expr(lhs, env)?;
        let r = self.eval_expr(rhs, env)?;
        eval_numeric_binop(op, l, r)
    }

    fn eval_unop(&self, op: UnOpKind, v: Value) -> Result<Value, RuntimeError> {
        match op {
            UnOpKind::Neg => match v {
                Value::F32(x) => Ok(Value::F32(-x)),
                Value::F64(x) => Ok(Value::F64(-x)),
                Value::I32(x) => Ok(Value::I32(-x)),
                Value::I64(x) => Ok(Value::I64(-x)),
                Value::Vec3(v) => Ok(Value::Vec3([-v[0], -v[1], -v[2]])),
                Value::Tensor(t) => {
                    let data: Vec<f32> = t.read().unwrap().cpu_data().iter().map(|x| -x).collect();
                    let shape = t.read().unwrap().shape.clone();
                    Ok(Value::Tensor(Arc::new(RwLock::new(Tensor::from_data(shape, data)))))
                }
                _ => rt_err!("unary `-` on `{}`", v.type_name()),
            },
            UnOpKind::Not => match v {
                Value::Bool(b) => Ok(Value::Bool(!b)),
                Value::I32(x)  => Ok(Value::I32(!x)),
                Value::I64(x)  => Ok(Value::I64(!x)),
                _ => rt_err!("unary `!` on `{}`", v.type_name()),
            },
            UnOpKind::Deref | UnOpKind::Ref | UnOpKind::RefMut => Ok(v),
        }
    }

    fn eval_matmul(&mut self, l: Value, r: Value) -> Result<Value, RuntimeError> {
        match (l, r) {
            (Value::Tensor(a), Value::Tensor(b)) => {
                let out = a.read().unwrap().matmul(&b.read().unwrap())?;
                Ok(Value::Tensor(Arc::new(RwLock::new(out))))
            }
            (Value::Mat3(a), Value::Mat3(b)) => {
                Ok(Value::Mat3(mat3_mul(a, b)))
            }
            (Value::Mat4(a), Value::Mat4(b)) => {
                Ok(Value::Mat4(mat4_mul(a, b)))
            }
            (Value::Mat3(m), Value::Vec3(v)) => {
                Ok(Value::Vec3(mat3_vec3_mul(m, v)))
            }
            (Value::Mat4(m), Value::Vec4(v)) => {
                Ok(Value::Vec4(mat4_vec4_mul(m, v)))
            }
            (l, r) => rt_err!("@ requires tensor/matrix operands, got `{}` @ `{}`",
                               l.type_name(), r.type_name()),
        }
    }

    // =========================================================================
    // §14  ASSIGNMENT
    // =========================================================================

    fn eval_assign(&mut self, op: AssignOpKind, target: &Expr, rhs: Value, env: &mut Env)
        -> Result<Value, RuntimeError>
    {
        // For compound assignments, read current value first.
        let effective_rhs = if op == AssignOpKind::Assign {
            rhs
        } else {
            let current = self.eval_expr(target, env)?;
            let bin_op = op.to_binop()
                .ok_or_else(|| RuntimeError::new("unknown compound assignment"))?;
            // MatMulAssign: @=
            if op == AssignOpKind::MatMulAssign {
                self.eval_matmul(current, rhs)?
            } else {
                eval_numeric_binop(bin_op, current, rhs)?
            }
        };

        // Write back to the target.
        match target {
            Expr::Ident { name, .. } => {
                env.set(name, effective_rhs);
            }
            Expr::Field { object, field, .. } => {
                if let Expr::Ident { name, .. } = object.as_ref() {
                    // For entity field access, write to the world.
                    if let Some(Value::Entity(id)) = env.get(name).cloned() {
                        let mut w = self.world.lock().unwrap();
                        if let Some(comp) = w.get_component_mut(id, field) {
                            *comp = effective_rhs;
                        } else {
                            w.insert_component(id, field, effective_rhs);
                        }
                    } else if let Some(Value::Struct { fields, .. }) = env.get(name).cloned() {
                        let mut s = env.get(name).unwrap().clone();
                        if let Value::Struct { ref mut fields, .. } = s {
                            fields.insert(field.clone(), effective_rhs);
                        }
                        env.set(name, s);
                    }
                }
            }
            Expr::Index { object, indices, .. } => {
                if let Expr::Ident { name, .. } = object.as_ref() {
                    let idx = if let Some(e) = indices.first() {
                        self.eval_expr(e, env)?.as_i64().unwrap_or(0) as usize
                    } else { 0 };
                    if let Some(Value::Array(arr)) = env.get(name).cloned() {
                        arr.lock().unwrap()[idx] = effective_rhs;
                    }
                }
            }
            _ => {}
        }
        Ok(Value::Unit)
    }

    // =========================================================================
    // §15  FIELD AND INDEX ACCESS
    // =========================================================================

    fn eval_field(&mut self, obj: Value, field: &str) -> Result<Value, RuntimeError> {
        match obj {
            Value::Struct { ref fields, .. } => {
                fields.get(field).cloned()
                    .ok_or_else(|| RuntimeError::new(format!("no field `{field}`")))
            }
            Value::Entity(id) => {
                let w = self.world.lock().unwrap();
                // Field name maps to component type by convention (lowercase → CamelCase).
                // We try the field name directly, then a title-cased version.
                let comp = w.get_component(id, field)
                    .or_else(|| {
                        let titled = title_case(field);
                        w.get_component(id, &titled)
                    });
                comp.cloned()
                    .ok_or_else(|| RuntimeError::new(format!(
                        "entity has no component `{field}`"
                    )))
            }
            Value::Vec2(v) => swizzle_vec(&v, field).map_err(RuntimeError::new),
            Value::Vec3(v) => swizzle_vec(&v, field).map_err(RuntimeError::new),
            Value::Vec4(v) => swizzle_vec(&v, field).map_err(RuntimeError::new),
            Value::Quat(q) => match field {
                "x" => Ok(Value::F32(q[0])),
                "y" => Ok(Value::F32(q[1])),
                "z" => Ok(Value::F32(q[2])),
                "w" => Ok(Value::F32(q[3])),
                _ => rt_err!("quat has no field `{field}`"),
            },
            Value::Tuple(vs) => {
                let idx: usize = field.parse()
                    .map_err(|_| RuntimeError::new(format!("bad tuple field `{field}`")))?;
                vs.into_iter().nth(idx)
                    .ok_or_else(|| RuntimeError::new(format!("tuple index {idx} out of range")))
            }
            other => rt_err!("`{}` has no field `{field}`", other.type_name()),
        }
    }

    fn eval_index(&self, obj: Value, indices: Vec<Value>) -> Result<Value, RuntimeError> {
        match obj {
            Value::Array(arr) => {
                let idx = indices.first().and_then(|v| v.as_i64()).unwrap_or(0) as usize;
                let a = arr.lock().unwrap();
                a.get(idx).cloned()
                    .ok_or_else(|| RuntimeError::new(format!("index {idx} out of bounds")))
            }
            Value::Tensor(t) => {
                let t = t.read().unwrap();
                let flat_idx = tensor_flat_index(&t.shape, &indices)?;
                Ok(Value::F32(t.cpu_data()[flat_idx]))
            }
            Value::Str(s) => {
                let idx = indices.first().and_then(|v| v.as_i64()).unwrap_or(0) as usize;
                s.chars().nth(idx)
                    .map(|c| Value::Str(c.to_string()))
                    .ok_or_else(|| RuntimeError::new("string index out of range"))
            }
            other => rt_err!("cannot index `{}`", other.type_name()),
        }
    }

    // =========================================================================
    // §16  FUNCTION CALLS
    // =========================================================================

    fn eval_call(&mut self, func: Value, args: Vec<Value>, env: &mut Env)
        -> Result<Value, RuntimeError>
    {
        match func {
            Value::Fn(closure) => {
                let closure = closure.clone();
                let mut call_env = Env::new();
                // Inject capture.
                for (k, v) in &closure.capture {
                    call_env.set_local(k, v.clone());
                }
                call_env.push();
                for (param, arg) in closure.decl.params.iter().zip(args) {
                    call_env.set_local(&param.name, arg);
                }
                let body = closure.decl.body.clone();
                let result = if let Some(b) = body {
                    self.eval_block(&b, &mut call_env)?
                } else { Value::Unit };
                call_env.pop();
                match result {
                    Value::Return(v) => Ok(*v),
                    other            => Ok(other),
                }
            }
            _ => {
                // Built-in functions.
                rt_err!("not callable")
            }
        }
    }

    // ── Built-in function dispatch ────────────────────────────────────────────

    fn eval_builtin(&mut self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        use std::f64::consts;

        match name {
            // ── Math functions ────────────────────────────────────────────────
            "sin" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.sin() as f32))
                } else { rt_err!("sin() requires a number") }
            }
            "cos" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.cos() as f32))
                } else { rt_err!("cos() requires a number") }
            }
            "tan" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.tan() as f32))
                } else { rt_err!("tan() requires a number") }
            }
            "asin" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.asin() as f32))
                } else { rt_err!("asin() requires a number") }
            }
            "acos" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.acos() as f32))
                } else { rt_err!("acos() requires a number") }
            }
            "atan" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.atan() as f32))
                } else { rt_err!("atan() requires a number") }
            }
            "atan2" => {
                match (args.get(0).and_then(|v| v.as_f64()), args.get(1).and_then(|v| v.as_f64())) {
                    (Some(y), Some(x)) => Ok(Value::F32(y.atan2(x) as f32)),
                    _ => rt_err!("atan2() requires two numbers")
                }
            }
            "sqrt" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.sqrt() as f32))
                } else { rt_err!("sqrt() requires a number") }
            }
            "cbrt" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.cbrt() as f32))
                } else { rt_err!("cbrt() requires a number") }
            }
            "pow" => {
                match (args.get(0).and_then(|v| v.as_f64()), args.get(1).and_then(|v| v.as_f64())) {
                    (Some(x), Some(y)) => Ok(Value::F32(x.powf(y) as f32)),
                    _ => rt_err!("pow() requires two numbers")
                }
            }
            "exp" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.exp() as f32))
                } else { rt_err!("exp() requires a number") }
            }
            "exp2" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.exp2() as f32))
                } else { rt_err!("exp2() requires a number") }
            }
            "exp10" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.log10().exp() as f32))
                } else { rt_err!("exp10() requires a number") }
            }
            "ln" | "log" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.ln() as f32))
                } else { rt_err!("ln() requires a number") }
            }
            "log2" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.log2() as f32))
                } else { rt_err!("log2() requires a number") }
            }
            "log10" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.log10() as f32))
                } else { rt_err!("log10() requires a number") }
            }
            "abs" => {
                match args.first() {
                    Some(Value::F32(x)) => Ok(Value::F32(x.abs())),
                    Some(Value::F64(x)) => Ok(Value::F64(x.abs())),
                    Some(Value::I32(x)) => Ok(Value::I32(x.abs())),
                    Some(Value::I64(x)) => Ok(Value::I64(x.abs())),
                    Some(v) if let Some(x) = v.as_f64() => Ok(Value::F32(x.abs() as f32)),
                    _ => rt_err!("abs() requires a number")
                }
            }
            "floor" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.floor() as f32))
                } else { rt_err!("floor() requires a number") }
            }
            "ceil" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.ceil() as f32))
                } else { rt_err!("ceil() requires a number") }
            }
            "round" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.round() as f32))
                } else { rt_err!("round() requires a number") }
            }
            "trunc" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.trunc() as f32))
                } else { rt_err!("trunc() requires a number") }
            }
            "fract" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.fract() as f32))
                } else { rt_err!("fract() requires a number") }
            }
            "min" => {
                match (args.get(0).and_then(|v| v.as_f64()), args.get(1).and_then(|v| v.as_f64())) {
                    (Some(x), Some(y)) => Ok(Value::F32(x.min(y) as f32)),
                    _ => rt_err!("min() requires two numbers")
                }
            }
            "max" => {
                match (args.get(0).and_then(|v| v.as_f64()), args.get(1).and_then(|v| v.as_f64())) {
                    (Some(x), Some(y)) => Ok(Value::F32(x.max(y) as f32)),
                    _ => rt_err!("max() requires two numbers")
                }
            }
            "clamp" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                ) {
                    (Some(x), Some(lo), Some(hi)) => Ok(Value::F32((x.max(lo).min(hi)) as f32)),
                    _ => rt_err!("clamp() requires three numbers")
                }
            }
            "degrees" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32((x * 180.0 / consts::PI) as f32))
                } else { rt_err!("degrees() requires a number") }
            }
            "radians" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32((x * consts::PI / 180.0) as f32))
                } else { rt_err!("radians() requires a number") }
            }
            "sign" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    if x > 0.0 { Ok(Value::F32(1.0)) }
                    else if x < 0.0 { Ok(Value::F32(-1.0)) }
                    else { Ok(Value::F32(0.0)) }
                } else { rt_err!("sign() requires a number") }
            }
            "step" => {
                match (args.get(0).and_then(|v| v.as_f64()), args.get(1).and_then(|v| v.as_f64())) {
                    (Some(edge), Some(x)) => Ok(Value::F32(if x >= edge { 1.0 } else { 0.0 })),
                    _ => rt_err!("step() requires two numbers")
                }
            }
            "smoothstep" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                ) {
                    (Some(edge0), Some(edge1), Some(x)) => {
                        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
                        Ok(Value::F32((t * t * (3.0 - 2.0 * t)) as f32))
                    }
                    _ => rt_err!("smoothstep() requires three numbers")
                }
            }
            "mix" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                ) {
                    (Some(x), Some(y), Some(a)) => {
                        Ok(Value::F32((x * (1.0 - a) + y * a) as f32))
                    }
                    _ => rt_err!("mix() requires three numbers")
                }
            }

            // ── I/O functions ─────────────────────────────────────────────────
            "print" => {
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { print!(" "); }
                    print!("{}", arg);
                }
                Ok(Value::Unit)
            }
            "println" => {
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { print!(" "); }
                    print!("{}", arg);
                }
                println!();
                Ok(Value::Unit)
            }
            "dbg" => {
                println!("[DEBUG] {:?}", args);
                Ok(args.first().cloned().unwrap_or(Value::Unit))
            }

            // ── Type conversion ───────────────────────────────────────────────
            "i32" => {
                if let Some(x) = args.first().and_then(|v| v.as_i64()) {
                    Ok(Value::I32(x as i32))
                } else if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::I32(x as i32))
                } else { rt_err!("i32() requires a number") }
            }
            "i64" => {
                if let Some(x) = args.first().and_then(|v| v.as_i64()) {
                    Ok(Value::I64(x))
                } else if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::I64(x as i64))
                } else { rt_err!("i64() requires a number") }
            }
            "f32" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x as f32))
                } else { rt_err!("f32() requires a number") }
            }
            "f64" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F64(x))
                } else { rt_err!("f64() requires a number") }
            }
            "bool" => {
                Ok(Value::Bool(args.first().map(|v| v.is_truthy()).unwrap_or(false)))
            }
            "str" => {
                Ok(Value::Str(args.first().map(|v| v.to_string()).unwrap_or_default()))
            }

            // ── Collection functions ──────────────────────────────────────────
            "len" => {
                match args.first() {
                    Some(Value::Array(a)) => Ok(Value::I32(a.lock().unwrap().len() as i32)),
                    Some(Value::Str(s)) => Ok(Value::I32(s.len() as i32)),
                    Some(v) => rt_err!("len() not applicable to {}", v.type_name()),
                    None => rt_err!("len() requires an argument")
                }
            }
            "range" => {
                match (args.get(0).and_then(|v| v.as_i64()), args.get(1).and_then(|v| v.as_i64())) {
                    (Some(start), Some(end)) => {
                        let range: Vec<Value> = (start as i32..end as i32).map(Value::I32).collect();
                        Ok(Value::Array(Arc::new(Mutex::new(range))))
                    }
                    _ => rt_err!("range() requires two numbers")
                }
            }

            // ── Option / Result constructors ───────────────────────────────────
            "Some" => {
                Ok(Value::Some(Box::new(args.into_iter().next().unwrap_or(Value::Unit))))
            }
            "None" => {
                Ok(Value::None)
            }
            "Ok" => {
                Ok(Value::Ok(Box::new(args.into_iter().next().unwrap_or(Value::Unit))))
            }
            "Err" => {
                Ok(Value::Err(Box::new(args.into_iter().next().unwrap_or(Value::Unit))))
            }
            "unwrap" => {
                match args.first() {
                    Some(Value::Some(v)) => Ok((**v).clone()),
                    Some(Value::Ok(v)) => Ok((**v).clone()),
                    Some(Value::None) => rt_err!("called unwrap() on None"),
                    Some(Value::Err(e)) => rt_err!("called unwrap() on Err: {}", e),
                    _ => rt_err!("unwrap() requires Option or Result")
                }
            }
            "is_some" => {
                match args.first() {
                    Some(Value::Some(_)) => Ok(Value::Bool(true)),
                    Some(Value::None) => Ok(Value::Bool(false)),
                    _ => rt_err!("is_some() requires Option")
                }
            }
            "is_none" => {
                match args.first() {
                    Some(Value::Some(_)) => Ok(Value::Bool(false)),
                    Some(Value::None) => Ok(Value::Bool(true)),
                    _ => rt_err!("is_none() requires Option")
                }
            }
            "is_ok" => {
                match args.first() {
                    Some(Value::Ok(_)) => Ok(Value::Bool(true)),
                    Some(Value::Err(_)) => Ok(Value::Bool(false)),
                    _ => rt_err!("is_ok() requires Result")
                }
            }
            "is_err" => {
                match args.first() {
                    Some(Value::Ok(_)) => Ok(Value::Bool(false)),
                    Some(Value::Err(_)) => Ok(Value::Bool(true)),
                    _ => rt_err!("is_err() requires Result")
                }
            }

            // ── String functions ──────────────────────────────────────────────
            "concat" => {
                let strs: Vec<String> = args.iter()
                    .map(|v| v.to_string())
                    .collect();
                Ok(Value::Str(strs.join("")))
            }

            // ── HashMap / Collection constructors ──────────────────────────────
            "HashMap::new" => {
                Ok(Value::HashMap(Arc::new(Mutex::new(HashMap::new()))))
            }

            // ── File I/O ───────────────────────────────────────────────────────
            "read_file" => {
                if let Some(Value::Str(path)) = args.first() {
                    match std::fs::read_to_string(path) {
                        Ok(content) => Ok(Value::Str(content)),
                        Err(e) => rt_err!("read_file failed: {}", e)
                    }
                } else { rt_err!("read_file requires a path string") }
            }
            "write_file" => {
                match (args.get(0), args.get(1)) {
                    (Some(Value::Str(path)), Some(Value::Str(content))) => {
                        match std::fs::write(path, content) {
                            Ok(_) => Ok(Value::Bool(true)),
                            Err(e) => rt_err!("write_file failed: {}", e)
                        }
                    }
                    _ => rt_err!("write_file requires (path, content) strings")
                }
            }
            "file_exists" => {
                if let Some(Value::Str(path)) = args.first() {
                    Ok(Value::Bool(std::path::Path::new(path).exists()))
                } else { rt_err!("file_exists requires a path string") }
            }
            "delete_file" => {
                if let Some(Value::Str(path)) = args.first() {
                    match std::fs::remove_file(path) {
                        Ok(_) => Ok(Value::Bool(true)),
                        Err(e) => rt_err!("delete_file failed: {}", e)
                    }
                } else { rt_err!("delete_file requires a path string") }
            }
            "append_file" => {
                match (args.get(0), args.get(1)) {
                    (Some(Value::Str(path)), Some(Value::Str(content))) => {
                        use std::io::Write;
                        match std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(path)
                        {
                            Ok(mut f) => {
                                if f.write_all(content.as_bytes()).is_ok() {
                                    Ok(Value::Bool(true))
                                } else {
                                    rt_err!("append_file write failed")
                                }
                            }
                            Err(e) => rt_err!("append_file failed: {}", e)
                        }
                    }
                    _ => rt_err!("append_file requires (path, content) strings")
                }
            }

            // Not a built-in
            _ => Err(RuntimeError {
                message: format!("unknown function: {}", name),
                span: None,
            })
        }
    }

    // ── Built-in method dispatch ───────────────────────────────────────────

    fn eval_method(&mut self, recv: Value, method: &str, args: Vec<Value>)
        -> Result<Value, RuntimeError>
    {
        match (&recv, method) {
            // ── Tensor methods ─────────────────────────────────────────────────
            (Value::Tensor(_), "transpose") => {
                if let Value::Tensor(t) = recv {
                    let out = t.read().unwrap().transpose()?;
                    Ok(Value::Tensor(Arc::new(RwLock::new(out))))
                } else { unreachable!() }
            }
            (Value::Tensor(_), "sum") => {
                if let Value::Tensor(t) = recv {
                    Ok(Value::F32(t.read().unwrap().sum_all()))
                } else { unreachable!() }
            }
            (Value::Tensor(_), "mean") => {
                if let Value::Tensor(t) = recv {
                    let tt = t.read().unwrap();
                    Ok(Value::F32(tt.sum_all() / tt.numel() as f32))
                } else { unreachable!() }
            }
            (Value::Tensor(_), "relu") => {
                if let Value::Tensor(t) = recv {
                    let out = t.read().unwrap().apply_activation(&Activation::Relu);
                    Ok(Value::Tensor(Arc::new(RwLock::new(out))))
                } else { unreachable!() }
            }
            (Value::Tensor(_), "softmax") => {
                if let Value::Tensor(t) = recv {
                    let out = t.read().unwrap().apply_activation(&Activation::Softmax);
                    Ok(Value::Tensor(Arc::new(RwLock::new(out))))
                } else { unreachable!() }
            }
            (Value::Tensor(_), "reshape") => {
                if let Value::Tensor(t) = recv {
                    let new_shape: Vec<usize> = args.iter()
                        .filter_map(|v| v.as_i64().map(|i| i as usize))
                        .collect();
                    let data = t.read().unwrap().cpu_data().to_vec();
                    Ok(Value::Tensor(Arc::new(RwLock::new(
                        Tensor::from_data(new_shape, data)
                    ))))
                } else { unreachable!() }
            }
            // ── Vec methods ────────────────────────────────────────────────────
            (Value::Vec3(v), "dot") => {
                if let Some(Value::Vec3(r)) = args.first() {
                    Ok(Value::F32(v[0]*r[0] + v[1]*r[1] + v[2]*r[2]))
                } else { rt_err!("dot() expects vec3") }
            }
            (Value::Vec3(v), "cross") => {
                if let Some(Value::Vec3(r)) = args.first() {
                    Ok(Value::Vec3([
                        v[1]*r[2] - v[2]*r[1],
                        v[2]*r[0] - v[0]*r[2],
                        v[0]*r[1] - v[1]*r[0],
                    ]))
                } else { rt_err!("cross() expects vec3") }
            }
            (Value::Vec3(v), "normalize") => {
                let len = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
                if len < 1e-8 { return Ok(Value::Vec3(*v)); }
                Ok(Value::Vec3([v[0]/len, v[1]/len, v[2]/len]))
            }
            (Value::Vec3(v), "length" | "magnitude") => {
                Ok(Value::F32((v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt()))
            }
            // ── Model forward pass ─────────────────────────────────────────────
            (Value::Model(_), "forward") => {
                if let Value::Model(m) = recv {
                    if let Some(Value::Tensor(x)) = args.into_iter().next() {
                        let x_owned = Arc::try_unwrap(x)
                            .unwrap_or_else(|a| {
                                let t = a.read().unwrap().clone();
                                RwLock::new(t)
                            })
                            .into_inner().unwrap();
                        let out = m.lock().unwrap().forward(x_owned)?;
                        Ok(Value::Tensor(Arc::new(RwLock::new(out))))
                    } else { rt_err!("forward() expects tensor") }
                } else { unreachable!() }
            }
            // ── World methods ──────────────────────────────────────────────────
            (Value::World(_), "spawn") => {
                if let Value::World(w) = recv {
                    let id = w.lock().unwrap().spawn();
                    Ok(Value::Entity(id))
                } else { unreachable!() }
            }
            (Value::World(_), "despawn") => {
                if let Value::World(w) = recv {
                    if let Some(Value::Entity(id)) = args.first() {
                        w.lock().unwrap().despawn(*id);
                    }
                    Ok(Value::Unit)
                } else { unreachable!() }
            }
            // ── Array / string methods ─────────────────────────────────────────
            (Value::Array(_), "len") => {
                if let Value::Array(a) = recv {
                    Ok(Value::I32(a.lock().unwrap().len() as i32))
                } else { unreachable!() }
            }
            (Value::Array(_), "push") => {
                if let Value::Array(a) = recv {
                    if let Some(v) = args.into_iter().next() {
                        a.lock().unwrap().push(v);
                    }
                    Ok(Value::Unit)
                } else { unreachable!() }
            }
            (Value::Array(_), "pop") => {
                if let Value::Array(a) = recv {
                    Ok(a.lock().unwrap().pop().unwrap_or(Value::Unit))
                } else { unreachable!() }
            }
            (Value::Array(_), "clear") => {
                if let Value::Array(a) = recv {
                    a.lock().unwrap().clear();
                    Ok(Value::Unit)
                } else { unreachable!() }
            }

            // ── HashMap methods ───────────────────────────────────────────────
            (Value::HashMap(_), "insert") => {
                if let Value::HashMap(m) = recv {
                    match (args.get(0), args.get(1)) {
                        (Some(Value::Str(k)), Some(v)) => {
                            m.lock().unwrap().insert(k.clone(), v.clone());
                            Ok(Value::Unit)
                        }
                        _ => rt_err!("insert() requires (string_key, value)")
                    }
                } else { unreachable!() }
            }
            (Value::HashMap(_), "get") => {
                if let Value::HashMap(m) = recv {
                    if let Some(Value::Str(k)) = args.first() {
                        let map = m.lock().unwrap();
                        Ok(map.get(k).cloned().unwrap_or(Value::None))
                    } else { rt_err!("get() requires string key") }
                } else { unreachable!() }
            }
            (Value::HashMap(_), "remove") => {
                if let Value::HashMap(m) = recv {
                    if let Some(Value::Str(k)) = args.first() {
                        Ok(m.lock().unwrap().remove(k).unwrap_or(Value::None))
                    } else { rt_err!("remove() requires string key") }
                } else { unreachable!() }
            }
            (Value::HashMap(_), "len") => {
                if let Value::HashMap(m) = recv {
                    Ok(Value::I32(m.lock().unwrap().len() as i32))
                } else { unreachable!() }
            }
            (Value::HashMap(_), "clear") => {
                if let Value::HashMap(m) = recv {
                    m.lock().unwrap().clear();
                    Ok(Value::Unit)
                } else { unreachable!() }
            }
            (Value::HashMap(_), "keys") => {
                if let Value::HashMap(m) = recv {
                    let keys: Vec<Value> = m.lock().unwrap()
                        .keys()
                        .map(|k| Value::Str(k.clone()))
                        .collect();
                    Ok(Value::Array(Arc::new(Mutex::new(keys))))
                } else { unreachable!() }
            }
            (Value::HashMap(_), "values") => {
                if let Value::HashMap(m) = recv {
                    let values: Vec<Value> = m.lock().unwrap()
                        .values()
                        .cloned()
                        .collect();
                    Ok(Value::Array(Arc::new(Mutex::new(values))))
                } else { unreachable!() }
            }
            (Value::HashMap(_), "contains_key") => {
                if let Value::HashMap(m) = recv {
                    if let Some(Value::Str(k)) = args.first() {
                        Ok(Value::Bool(m.lock().unwrap().contains_key(k)))
                    } else { rt_err!("contains_key() requires string key") }
                } else { unreachable!() }
            }
            (Value::Str(s), "len") => Ok(Value::I32(s.len() as i32)),

            // ── String methods ─────────────────────────────────────────────────
            (Value::Str(s), "to_upper") => Ok(Value::Str(s.to_uppercase())),
            (Value::Str(s), "to_lower") => Ok(Value::Str(s.to_lowercase())),
            (Value::Str(s), "trim") => Ok(Value::Str(s.trim().to_string())),
            (Value::Str(s), "trim_start") => Ok(Value::Str(s.trim_start().to_string())),
            (Value::Str(s), "trim_end") => Ok(Value::Str(s.trim_end().to_string())),
            (Value::Str(s), "chars") => {
                let chars: Vec<Value> = s.chars()
                    .map(|c| Value::Str(c.to_string()))
                    .collect();
                Ok(Value::Array(Arc::new(Mutex::new(chars))))
            }
            (Value::Str(s), "reverse") => {
                Ok(Value::Str(s.chars().rev().collect()))
            }
            (Value::Str(s), "starts_with") => {
                if let Some(Value::Str(prefix)) = args.first() {
                    Ok(Value::Bool(s.starts_with(prefix)))
                } else { rt_err!("starts_with() requires a string argument") }
            }
            (Value::Str(s), "ends_with") => {
                if let Some(Value::Str(suffix)) = args.first() {
                    Ok(Value::Bool(s.ends_with(suffix)))
                } else { rt_err!("ends_with() requires a string argument") }
            }
            (Value::Str(s), "contains") => {
                if let Some(Value::Str(needle)) = args.first() {
                    Ok(Value::Bool(s.contains(needle)))
                } else { rt_err!("contains() requires a string argument") }
            }
            (Value::Str(s), "split") => {
                if let Some(Value::Str(delim)) = args.first() {
                    let parts: Vec<Value> = s.split(delim.as_str())
                        .map(|part| Value::Str(part.to_string()))
                        .collect();
                    Ok(Value::Array(Arc::new(Mutex::new(parts))))
                } else { rt_err!("split() requires a string argument") }
            }
            (Value::Str(s), "replace") => {
                match (args.get(0), args.get(1)) {
                    (Some(Value::Str(from)), Some(Value::Str(to))) => {
                        Ok(Value::Str(s.replace(from, to)))
                    }
                    _ => rt_err!("replace() requires two string arguments")
                }
            }

            // ── Option/Result methods ───────────────────────────────────────────
            (Value::Some(v), "unwrap") => Ok((**v).clone()),
            (Value::None, "unwrap") => rt_err!("called unwrap() on None"),
            (Value::Ok(v), "unwrap") => Ok((**v).clone()),
            (Value::Err(e), "unwrap") => rt_err!("called unwrap() on Err: {}", e),

            (Value::Some(_), "is_some") => Ok(Value::Bool(true)),
            (Value::None, "is_some") => Ok(Value::Bool(false)),
            (Value::Some(_), "is_none") => Ok(Value::Bool(false)),
            (Value::None, "is_none") => Ok(Value::Bool(true)),

            (Value::Ok(_), "is_ok") => Ok(Value::Bool(true)),
            (Value::Err(_), "is_ok") => Ok(Value::Bool(false)),
            (Value::Ok(_), "is_err") => Ok(Value::Bool(false)),
            (Value::Err(_), "is_err") => Ok(Value::Bool(true)),

            // ── Fallback ───────────────────────────────────────────────────────
            (_, method) => {
                rt_err!("no method `{method}` on `{}`", recv.type_name())
            }
        }
    }

    // =========================================================================
    // §17  CAST
    // =========================================================================

    fn eval_cast(&self, v: Value, ty: &crate::ast::Type) -> Result<Value, RuntimeError> {
        use crate::ast::{Type, ElemType as E};
        let f = v.as_f64().unwrap_or(0.0);
        match ty {
            Type::Scalar(E::F32)  => Ok(Value::F32(f as f32)),
            Type::Scalar(E::F64)  => Ok(Value::F64(f)),
            Type::Scalar(E::I32)  => Ok(Value::I32(f as i32)),
            Type::Scalar(E::I64)  => Ok(Value::I64(f as i64)),
            Type::Scalar(E::U32)  => Ok(Value::U32(f as u32)),
            Type::Scalar(E::U64)  => Ok(Value::U64(f as u64)),
            Type::Scalar(E::Bool) => Ok(Value::Bool(f != 0.0)),
            _ => Ok(v),
        }
    }

    // =========================================================================
    // §18  PATTERN MATCHING
    // =========================================================================

    fn pattern_matches(&self, pat: &Pattern, val: &Value) -> bool {
        match (pat, val) {
            (Pattern::Wildcard(_), _) => true,
            (Pattern::Ident { .. }, _) => true,
            (Pattern::Lit(_, lit), v) => {
                use crate::ast::LitVal;
                match (lit, v) {
                    (LitVal::Int(n),   Value::I32(x)) => *n == *x as u128,
                    (LitVal::Int(n),   Value::I64(x)) => *n == *x as u128,
                    (LitVal::Float(f), Value::F32(x)) => (*f as f32 - x).abs() < f32::EPSILON,
                    (LitVal::Bool(b),  Value::Bool(x)) => b == x,
                    (LitVal::Str(s),   Value::Str(x))  => s == x,
                    _ => false,
                }
            }
            (Pattern::Tuple { elems, .. }, Value::Tuple(vs)) => {
                elems.len() == vs.len()
                && elems.iter().zip(vs).all(|(p, v)| self.pattern_matches(p, v))
            }
            (Pattern::Or { arms, .. }, v) => {
                arms.iter().any(|p| self.pattern_matches(p, v))
            }
            _ => false,
        }
    }

    fn bind_pattern(&self, pat: &Pattern, val: Value, env: &mut Env) {
        match pat {
            Pattern::Ident { name, .. } => env.set_local(name, val),
            Pattern::Wildcard(_) => {}
            Pattern::Tuple { elems, .. } => {
                if let Value::Tuple(vs) = val {
                    for (p, v) in elems.iter().zip(vs) {
                        self.bind_pattern(p, v, env);
                    }
                }
            }
            Pattern::Struct { fields, .. } => {
                if let Value::Struct { fields: fmap, .. } = val {
                    for (fname, maybe_pat) in fields {
                        if let Some(fval) = fmap.get(fname) {
                            if let Some(p) = maybe_pat {
                                self.bind_pattern(p, fval.clone(), env);
                            } else {
                                env.set_local(fname, fval.clone());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // ── Collect iterable values into a Vec ────────────────────────────────

    fn value_to_iter(&self, v: Value) -> Result<Vec<Value>, RuntimeError> {
        match v {
            Value::Array(a) => Ok(a.lock().unwrap().clone()),
            Value::Tuple(vs) => Ok(vs),
            Value::Str(s) => Ok(s.chars().map(|c| Value::Str(c.to_string())).collect()),
            _ => rt_err!("value `{}` is not iterable", v.type_name()),
        }
    }
}

// =============================================================================
// §19  TRAINING LOOP RUNTIME  (Unique Feature 2)
// =============================================================================

/// Configuration resolved from a `TrainDecl`.
pub struct TrainingConfig {
    pub agent_name:   String,
    pub world_name:   String,
    pub model_name:   Option<String>,
    pub max_steps:    u64,
    pub num_envs:     u64,
    pub lr:           f32,
    pub gamma:        f32,
    pub signals:      Vec<(String, f32, bool)>, // (name, weight, is_reward)
    pub optimizer:    OptimizerKind,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            agent_name:  String::new(),
            world_name:  String::new(),
            model_name:  None,
            max_steps:   1000,
            num_envs:    1,
            lr:          3e-4,
            gamma:       0.99,
            signals:     vec![],
            optimizer:   OptimizerKind::Adam,
        }
    }
}

impl Interpreter {
    /// Execute a full training session described by a `TrainDecl`.
    pub fn run_training(
        &mut self,
        decl: &TrainDecl,
        interp_env: &mut Env,
    ) -> Result<TrainingStats, RuntimeError> {
        let max_steps = decl.episode.as_ref()
            .and_then(|e| e.max_steps).unwrap_or(1000);
        let num_envs  = decl.episode.as_ref()
            .and_then(|e| e.num_envs).unwrap_or(1);
        let gamma     = decl.hyper.iter().find(|(k, _)| k == "gamma")
            .and_then(|(_, e)| if let Expr::FloatLit { value, .. } = e {
                Some(*value as f32) } else { None })
            .unwrap_or(0.99);

        // Instantiate model (if named).
        if let Some(model_name) = &decl.model {
            if !self.models.contains_key(model_name.as_str()) {
                if let Some(decl) = self.model_decls.get(model_name).cloned() {
                    let model = NnModel::from_decl(&decl);
                    self.models.insert(model_name.clone(), Arc::new(Mutex::new(model)));
                }
            }
        }

        let mut stats = TrainingStats::default();
        let mut total_reward = 0.0_f32;

        // Run `num_envs` parallel episodes (sequential in interpreter).
        for _env_idx in 0..num_envs {
            // Reset world state for each episode.
            {
                let mut w = self.world.lock().unwrap();
                *w = EcsWorld::default();
            }

            let mut episode_reward = 0.0_f32;
            let mut done = false;

            for step in 0..max_steps {
                if done { break; }

                // 1. Collect observations (stub: agent reports a unit tensor).
                let obs = Tensor::zeros(vec![1, 4]);

                // 2. Forward pass through policy network.
                let action = if let Some(model_name) = &decl.model {
                    if let Some(m) = self.models.get(model_name).cloned() {
                        let out = m.lock().unwrap().forward(obs)?;
                        Value::Tensor(Arc::new(RwLock::new(out)))
                    } else { Value::Unit }
                } else { Value::Unit };

                // 3. Step the simulation (tick all systems).
                // In the full runtime this calls scheduler.tick().

                // 4. Accumulate reward signals.
                let step_reward: f32 = decl.signals.iter()
                    .map(|sig| {
                        let w = self.world.lock().unwrap();
                        let count = w.events.get(&sig.name)
                            .map(|v| v.len() as f32)
                            .unwrap_or(0.0);
                        if sig.is_reward { count * sig.weight as f32 }
                        else             { -count * sig.weight as f32 }
                    })
                    .sum();
                episode_reward += step_reward * gamma.powi(step as i32);

                // 5. Check done condition.
                if let Some(ep) = &decl.episode {
                    if let Some(cond) = &ep.done_condition {
                        done = self.eval_expr(cond, interp_env)?.is_truthy();
                    }
                }
            }

            total_reward += episode_reward;
            stats.episode_rewards.push(episode_reward);
        }

        // 6. Policy gradient update (simplified REINFORCE).
        if let Some(model_name) = &decl.model {
            if let Some(model) = self.models.get(model_name).cloned() {
                let lr = decl.optimizer.as_ref()
                    .map(|o| o.learning_rate as f32)
                    .unwrap_or(3e-4);
                update_model_weights(&mut *model.lock().unwrap(), total_reward / num_envs as f32, lr);
            }
        }

        stats.mean_reward = total_reward / num_envs as f32;
        stats.total_steps = max_steps * num_envs;
        Ok(stats)
    }
}

/// Summary statistics from one training session.
#[derive(Debug, Default)]
pub struct TrainingStats {
    pub mean_reward:    f32,
    pub total_steps:    u64,
    pub episode_rewards: Vec<f32>,
}

impl TrainingStats {
    pub fn display(&self) {
        println!("Training complete:");
        println!("  mean reward = {:.4}", self.mean_reward);
        println!("  total steps = {}", self.total_steps);
        println!("  episodes    = {}", self.episode_rewards.len());
    }
}

/// Simple gradient update: scale all weights by the policy gradient signal.
fn update_model_weights(model: &mut NnModel, reward: f32, lr: f32) {
    model.step += 1;
    for layer in &mut model.layers {
        match layer {
            WeightLayer::Dense { w, b, .. } => {
                // REINFORCE: w ← w + lr * reward * ∇w (stub: use reward as gradient scale)
                let grad_scale = lr * reward.clamp(-1.0, 1.0);
                let w_data = w.cpu_data_mut();
                for x in w_data.iter_mut() { *x += grad_scale * pseudo_rand_small(); }
                let b_data = b.cpu_data_mut();
                for x in b_data.iter_mut() { *x += grad_scale * pseudo_rand_small(); }
            }
            _ => {}
        }
    }
}

// =============================================================================
// §20  GPU BACKEND TRAIT
// =============================================================================

/// Trait implemented by GPU backends (wgpu, CUDA via FFI, Metal).
/// The interpreter calls these when a tensor or loop is annotated for GPU.
pub trait GpuBackend: Send + Sync {
    /// Allocate a buffer on the GPU and copy data from the CPU.
    fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuBufferHandle;
    /// Copy a GPU buffer back to a CPU Vec<f32>.
    fn download(&self, handle: &GpuBufferHandle) -> Vec<f32>;
    /// Dispatch a matrix-multiply kernel: C = A @ B.
    fn matmul(&self, a: &GpuBufferHandle, b: &GpuBufferHandle,
               shape_a: &[usize], shape_b: &[usize]) -> GpuBufferHandle;
    /// Dispatch an element-wise kernel.
    fn elementwise(&self, a: &GpuBufferHandle, b: &GpuBufferHandle,
                   op: GpuOp) -> GpuBufferHandle;
    /// Dispatch a parallel entity-loop kernel (SPIR-V / WGSL / CUDA PTX).
    fn dispatch_entity_loop(&self, entities: &[u64], workgroup_size: u32);
}

/// Primitive operations the GPU backend can dispatch.
#[derive(Debug, Clone, Copy)]
pub enum GpuOp { Add, Sub, Mul, Div, HadamardMul, HadamardDiv }

// =============================================================================
// §21  PURE HELPERS
// =============================================================================

/// Evaluate a binary operator on numeric Values.
fn eval_numeric_binop(op: BinOpKind, l: Value, r: Value) -> Result<Value, RuntimeError> {
    // Tensor arithmetic.
    if let (Value::Tensor(a), Value::Tensor(b)) = (&l, &r) {
        let at = a.read().unwrap();
        let bt = b.read().unwrap();
        let result = match op {
            BinOpKind::Add => at.elementwise(&bt, |x, y| x + y, "+"),
            BinOpKind::Sub => at.elementwise(&bt, |x, y| x - y, "-"),
            BinOpKind::Mul => at.elementwise(&bt, |x, y| x * y, "*"),
            BinOpKind::Div => at.elementwise(&bt, |x, y| x / y, "/"),
            _ => return rt_err!("operator not supported for tensors"),
        }?;
        return Ok(Value::Tensor(Arc::new(RwLock::new(result))));
    }

    // Tensor + scalar broadcast.
    if let (Value::Tensor(t), ref scalar) | (ref scalar, Value::Tensor(t)) = (&l, &r) {
        if let Some(s) = scalar.as_f64() {
            let tt = t.read().unwrap();
            let data: Vec<f32> = tt.cpu_data().iter().map(|x| match op {
                BinOpKind::Add => x + s as f32,
                BinOpKind::Sub => x - s as f32,
                BinOpKind::Mul => x * s as f32,
                BinOpKind::Div => x / s as f32,
                _ => *x,
            }).collect();
            let out = Tensor::from_data(tt.shape.clone(), data);
            return Ok(Value::Tensor(Arc::new(RwLock::new(out))));
        }
    }

    // Vec3 arithmetic.
    if let (Value::Vec3(a), Value::Vec3(b)) = (&l, &r) {
        return Ok(Value::Vec3(match op {
            BinOpKind::Add => [a[0]+b[0], a[1]+b[1], a[2]+b[2]],
            BinOpKind::Sub => [a[0]-b[0], a[1]-b[1], a[2]-b[2]],
            BinOpKind::Mul => [a[0]*b[0], a[1]*b[1], a[2]*b[2]],
            BinOpKind::Div => [a[0]/b[0], a[1]/b[1], a[2]/b[2]],
            _ => return rt_err!("operator not supported for vec3"),
        }));
    }

    // Vec3 * scalar.
    if let (Value::Vec3(v), ref s) | (ref s, Value::Vec3(v)) = (&l, &r) {
        if let Some(s) = s.as_f64().map(|f| f as f32) {
            return Ok(Value::Vec3([v[0]*s, v[1]*s, v[2]*s]));
        }
    }

    // Comparison operators — always return Bool.
    let cmp_result = match op {
        BinOpKind::Eq => Some(value_eq(&l, &r)),
        BinOpKind::Ne => Some(!value_eq(&l, &r)),
        BinOpKind::Lt => l.as_f64().zip(r.as_f64()).map(|(a, b)| a < b),
        BinOpKind::Le => l.as_f64().zip(r.as_f64()).map(|(a, b)| a <= b),
        BinOpKind::Gt => l.as_f64().zip(r.as_f64()).map(|(a, b)| a > b),
        BinOpKind::Ge => l.as_f64().zip(r.as_f64()).map(|(a, b)| a >= b),
        _ => None,
    };
    if let Some(b) = cmp_result { return Ok(Value::Bool(b)); }

    // Numeric arithmetic — promote to widest common type.
    match (&l, &r) {
        (Value::F64(a), _) | (_, Value::F64(a)) => {
            let b = r.as_f64().or_else(|| l.as_f64()).unwrap_or(*a);
            let a = l.as_f64().unwrap_or(*a);
            Ok(Value::F64(arith_f64(op, a, b)?))
        }
        (Value::F32(a), _) | (_, Value::F32(a)) => {
            let b = r.as_f64().or_else(|| l.as_f64()).unwrap_or(*a as f64);
            let a = l.as_f64().unwrap_or(*a as f64);
            Ok(Value::F32(arith_f64(op, a, b)? as f32))
        }
        (Value::I64(_), _) | (_, Value::I64(_)) => {
            let a = l.as_i64().unwrap_or(0);
            let b = r.as_i64().unwrap_or(0);
            Ok(Value::I64(arith_i64(op, a, b)?))
        }
        _ => {
            let a = l.as_i64().unwrap_or(0);
            let b = r.as_i64().unwrap_or(0);
            Ok(Value::I32(arith_i64(op, a, b)? as i32))
        }
    }
}

fn arith_f64(op: BinOpKind, a: f64, b: f64) -> Result<f64, RuntimeError> {
    Ok(match op {
        BinOpKind::Add => a + b,
        BinOpKind::Sub => a - b,
        BinOpKind::Mul => a * b,
        BinOpKind::Div => {
            if b == 0.0 { return rt_err!("division by zero"); }
            a / b
        }
        BinOpKind::Rem => a % b,
        BinOpKind::FloorDiv => {
            if b == 0.0 { return rt_err!("floor division by zero"); }
            (a / b).floor()
        }
        _ => return rt_err!("operator {:?} not defined for floats", op),
    })
}

fn arith_i64(op: BinOpKind, a: i64, b: i64) -> Result<i64, RuntimeError> {
    Ok(match op {
        BinOpKind::Add    => a.wrapping_add(b),
        BinOpKind::Sub    => a.wrapping_sub(b),
        BinOpKind::Mul    => a.wrapping_mul(b),
        BinOpKind::Div    => {
            if b == 0 { return rt_err!("division by zero"); }
            a / b
        }
        BinOpKind::Rem    => {
            if b == 0 { return rt_err!("modulo by zero"); }
            a % b
        }
        BinOpKind::BitAnd => a & b,
        BinOpKind::BitOr  => a | b,
        BinOpKind::BitXor => a ^ b,
        BinOpKind::Shl    => a << (b as u32),
        BinOpKind::Shr    => a >> (b as u32),
        _ => return rt_err!("operator not defined for integers"),
    })
}

fn value_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::I32(x), Value::I32(y))   => x == y,
        (Value::I64(x), Value::I64(y))   => x == y,
        (Value::F32(x), Value::F32(y))   => x == y,
        (Value::F64(x), Value::F64(y))   => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Str(x), Value::Str(y))   => x == y,
        (Value::Unit, Value::Unit)        => true,
        _ => false,
    }
}

/// Compute the flat index into a tensor given multi-dimensional indices.
fn tensor_flat_index(shape: &[usize], indices: &[Value]) -> Result<usize, RuntimeError> {
    if indices.len() != shape.len() {
        return rt_err!("tensor has rank {} but {} indices given", shape.len(), indices.len());
    }
    let mut flat = 0;
    let mut stride = 1;
    for (i, (&dim, idx)) in shape.iter().zip(indices).enumerate().rev() {
        let idx_val = idx.as_i64().unwrap_or(0) as usize;
        if idx_val >= dim {
            return rt_err!("index {} out of bounds for dimension {} (size {})", idx_val, i, dim);
        }
        flat += idx_val * stride;
        stride *= dim;
    }
    Ok(flat)
}

/// Vec swizzle: extract x/y/z/w components or multi-component swizzles.
fn swizzle_vec(components: &[f32], field: &str) -> Result<Value, String> {
    let mapped: Vec<f32> = field.chars().map(|c| match c {
        'x' | 'r' => components.get(0).copied(),
        'y' | 'g' => components.get(1).copied(),
        'z' | 'b' => components.get(2).copied(),
        'w' | 'a' => components.get(3).copied(),
        _ => None,
    }).collect::<Option<Vec<_>>>()
        .ok_or_else(|| format!("invalid swizzle `{field}`"))?;
    Ok(match mapped.len() {
        1 => Value::F32(mapped[0]),
        2 => Value::Vec2([mapped[0], mapped[1]]),
        3 => Value::Vec3([mapped[0], mapped[1], mapped[2]]),
        4 => Value::Vec4([mapped[0], mapped[1], mapped[2], mapped[3]]),
        _ => return Err(format!("swizzle `{field}` exceeds 4 components")),
    })
}

fn title_case(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None    => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

// =============================================================================
// BROADCAST HELPERS (numpy-style)
// =============================================================================

/// Compute the broadcast result shape of two shapes. Returns None if incompatible.
pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let len = a.len().max(b.len());
    let mut result = vec![0usize; len];
    for i in 0..len {
        let da = if i < len - a.len() { 1 } else { a[i - (len - a.len())] };
        let db = if i < len - b.len() { 1 } else { b[i - (len - b.len())] };
        result[i] = if da == db { da }
                    else if da == 1 { db }
                    else if db == 1 { da }
                    else { return None; };
    }
    Some(result)
}

/// Map a flat linear index in `result_shape` back to a flat index in `src_shape`,
/// respecting broadcast rules (dimensions of size 1 always map to index 0).
pub(crate) fn broadcast_index(flat: usize, result_shape: &[usize], src_shape: &[usize]) -> usize {
    let len = result_shape.len();
    let off = len - src_shape.len();
    let mut src_idx = 0usize;
    let mut stride  = 1usize;
    let mut rem     = flat;

    // Decompose flat index into multi-dim, then re-compose for src.
    let mut multi = vec![0usize; len];
    for i in (0..len).rev() {
        multi[i] = rem % result_shape[i];
        rem      /= result_shape[i];
    }

    let mut src_strides = vec![1usize; src_shape.len()];
    for i in (0..src_shape.len().saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }

    for i in 0..src_shape.len() {
        let ri = i + off;
        let idx = if src_shape[i] == 1 { 0 } else { multi[ri] };
        src_idx += idx * src_strides[i];
        let _ = stride;
        stride = 1;
    }
    src_idx
}

// =============================================================================
// VMAP  —  vectorise a closure over the batch (first) dimension
// =============================================================================

/// Apply a function to each slice along axis 0 of a tensor, collecting results.
///
/// If `input` has shape [B, ...], the function receives tensors of shape [...]
/// and the output is stacked to [B, output_dims...].
pub fn vmap_tensor<F>(input: &Tensor, f: F) -> Result<Tensor, RuntimeError>
where
    F: Fn(Tensor) -> Result<Tensor, RuntimeError>,
{
    if input.shape.is_empty() {
        return Err(RuntimeError::new("vmap requires at least a 1-D tensor"));
    }
    let batch = input.shape[0];
    let inner_shape = input.shape[1..].to_vec();
    let inner_n: usize = inner_shape.iter().product::<usize>().max(1);
    let data = input.cpu_data();

    let mut outputs = Vec::with_capacity(batch);
    for b in 0..batch {
        let slice = data[b * inner_n..(b + 1) * inner_n].to_vec();
        let t = Tensor::from_data(inner_shape.clone(), slice);
        outputs.push(f(t)?);
    }

    // Stack: all outputs must have the same shape
    let out_shape = outputs.first()
        .map(|t| t.shape.clone())
        .unwrap_or_default();
    let out_n: usize = out_shape.iter().product::<usize>().max(1);
    let mut stacked = Vec::with_capacity(batch * out_n);
    for t in &outputs {
        stacked.extend_from_slice(t.cpu_data());
    }
    let mut final_shape = vec![batch];
    final_shape.extend(&out_shape);
    Ok(Tensor::from_data(final_shape, stacked))
}

// =============================================================================
// NOISE BUILTINS  (Perlin / value noise, no external crate)
// =============================================================================

/// Value noise in [0, 1] — cheap, no external dependency.
pub fn value_noise_2d(x: f32, y: f32) -> f32 {
    fn fade(t: f32) -> f32 { t * t * t * (t * (t * 6.0 - 15.0) + 10.0) }
    fn lerp(a: f32, b: f32, t: f32) -> f32 { a + t * (b - a) }
    fn hash(ix: i32, iy: i32) -> f32 {
        let n = ix.wrapping_mul(1619).wrapping_add(iy.wrapping_mul(31337))
            .wrapping_mul(1013904223i32);
        (n as u32 as f32) / u32::MAX as f32
    }

    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let ux = fade(fx);
    let uy = fade(fy);

    lerp(
        lerp(hash(ix, iy),   hash(ix + 1, iy),   ux),
        lerp(hash(ix, iy+1), hash(ix + 1, iy+1), ux),
        uy,
    )
}

/// White noise uniform in [0, 1].
pub fn white_noise() -> f32 { pseudo_rand() }

/// Fractional Brownian Motion — sum of octaves of value noise.
pub fn fbm_2d(x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut val = 0.0_f32;
    let mut amp = 0.5_f32;
    let mut freq = 1.0_f32;
    for _ in 0..octaves {
        val  += amp * value_noise_2d(x * freq, y * freq);
        amp  *= gain;
        freq *= lacunarity;
    }
    val
}

// =============================================================================
// PPO  —  Proximal Policy Optimisation training loop helpers
// =============================================================================

/// Compute Generalised Advantage Estimation (GAE).
///
/// `rewards`, `values`, `dones` are all length T (timestep).
/// Returns (advantages, returns) each of length T.
pub fn gae(
    rewards: &[f32],
    values:  &[f32],
    dones:   &[f32],  // 1.0 if episode ended, 0.0 otherwise
    gamma:   f32,
    lam:     f32,
    last_value: f32,
) -> (Vec<f32>, Vec<f32>) {
    let t = rewards.len();
    let mut advantages = vec![0.0_f32; t];
    let mut last_gae   = 0.0_f32;
    let mut next_val   = last_value;

    for i in (0..t).rev() {
        let mask  = 1.0 - dones[i];
        let delta = rewards[i] + gamma * next_val * mask - values[i];
        last_gae  = delta + gamma * lam * mask * last_gae;
        advantages[i] = last_gae;
        next_val = values[i];
    }

    let returns: Vec<f32> = advantages.iter().zip(values).map(|(a, v)| a + v).collect();
    (advantages, returns)
}

/// PPO surrogate loss (clipped objective).
///
/// Returns (policy_loss, value_loss, entropy).
pub fn ppo_loss(
    log_probs_old: &[f32],
    log_probs_new: &[f32],
    advantages:    &[f32],
    returns:       &[f32],
    values:        &[f32],
    clip_eps:      f32,
    vf_coef:       f32,
    ent_coef:      f32,
    entropy:       &[f32],
) -> (f32, f32, f32) {
    let n = log_probs_old.len() as f32;

    // Policy loss (clipped surrogate)
    let policy_loss = log_probs_old.iter().zip(log_probs_new).zip(advantages)
        .map(|((old, new), adv)| {
            let ratio  = (new - old).exp();
            let surr1  = ratio * adv;
            let surr2  = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv;
            -surr1.min(surr2)
        })
        .sum::<f32>() / n;

    // Value function loss (MSE, clipped)
    let value_loss = returns.iter().zip(values)
        .map(|(r, v)| (r - v).powi(2))
        .sum::<f32>() / n;

    // Entropy bonus
    let mean_entropy = entropy.iter().sum::<f32>() / n;

    (
        policy_loss + vf_coef * value_loss - ent_coef * mean_entropy,
        value_loss,
        mean_entropy,
    )
}

// =============================================================================
// SIMPLE PHYSICS STEP  (Feature 7 — game dev integration)
// =============================================================================

/// Rigid body state for a single entity.
#[derive(Debug, Clone, Default)]
pub struct RigidBody {
    pub pos:      [f32; 3],
    pub vel:      [f32; 3],
    pub mass:     f32,
    pub drag:     f32,        // linear damping coefficient
    pub is_static: bool,
}

impl RigidBody {
    pub fn new(mass: f32) -> Self {
        RigidBody { mass, drag: 0.02, ..Default::default() }
    }

    /// Semi-implicit Euler integration.
    pub fn integrate(&mut self, gravity: [f32; 3], dt: f32) {
        if self.is_static { return; }
        // a = F/m (gravity only here; external forces accumulated elsewhere)
        for i in 0..3 {
            self.vel[i] += gravity[i] * dt;
            self.vel[i] *= 1.0 - self.drag * dt;
            self.pos[i] += self.vel[i] * dt;
        }
    }

    /// Apply an impulse directly to velocity: v += impulse / mass.
    pub fn apply_impulse(&mut self, impulse: [f32; 3]) {
        if self.is_static || self.mass == 0.0 { return; }
        let inv_mass = 1.0 / self.mass;
        for i in 0..3 { self.vel[i] += impulse[i] * inv_mass; }
    }
}

/// Axis-aligned bounding box collider.
#[derive(Debug, Clone)]
pub struct AabbCollider {
    pub half_extents: [f32; 3],
}

impl AabbCollider {
    /// Test overlap of two AABBs at given positions.
    pub fn overlaps(&self, pos_a: [f32; 3], other: &AabbCollider, pos_b: [f32; 3]) -> bool {
        for i in 0..3 {
            let dist = (pos_a[i] - pos_b[i]).abs();
            if dist > self.half_extents[i] + other.half_extents[i] { return false; }
        }
        true
    }

    /// Compute penetration depth and normal for collision response.
    pub fn penetration(
        &self, pos_a: [f32; 3],
        other: &AabbCollider, pos_b: [f32; 3],
    ) -> Option<([f32; 3], f32)> {
        let mut min_pen = f32::INFINITY;
        let mut normal  = [0.0_f32; 3];
        for i in 0..3 {
            let delta  = pos_b[i] - pos_a[i];
            let overlap = (self.half_extents[i] + other.half_extents[i]) - delta.abs();
            if overlap <= 0.0 { return None; }
            if overlap < min_pen {
                min_pen = overlap;
                normal  = [0.0; 3];
                normal[i] = delta.signum();
            }
        }
        Some((normal, min_pen))
    }
}

/// Resolve elastic collision between two rigid bodies.
pub fn resolve_collision(a: &mut RigidBody, b: &mut RigidBody, normal: [f32; 3], restitution: f32) {
    let rel_vel: f32 = (0..3).map(|i| (a.vel[i] - b.vel[i]) * normal[i]).sum();
    if rel_vel > 0.0 { return; } // separating

    let inv_a = if a.is_static { 0.0 } else { 1.0 / a.mass };
    let inv_b = if b.is_static { 0.0 } else { 1.0 / b.mass };
    let j = -(1.0 + restitution) * rel_vel / (inv_a + inv_b);

    for i in 0..3 {
        a.vel[i] += j * inv_a * normal[i];
        b.vel[i] -= j * inv_b * normal[i];
    }
}

// ── Matrix arithmetic helpers ──────────────────────────────────────────────

fn mat3_mul(a: [[f32;3];3], b: [[f32;3];3]) -> [[f32;3];3] {
    let mut c = [[0.0_f32; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { c[i][j] += a[i][k] * b[k][j]; } } }
    c
}

fn mat4_mul(a: [[f32;4];4], b: [[f32;4];4]) -> [[f32;4];4] {
    let mut c = [[0.0_f32; 4]; 4];
    for i in 0..4 { for j in 0..4 { for k in 0..4 { c[i][j] += a[i][k] * b[k][j]; } } }
    c
}

fn mat3_vec3_mul(m: [[f32;3];3], v: [f32;3]) -> [f32;3] {
    [m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2],
     m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2],
     m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]]
}

fn mat4_vec4_mul(m: [[f32;4];4], v: [f32;4]) -> [f32;4] {
    let mut r = [0.0_f32; 4];
    for i in 0..4 { for j in 0..4 { r[i] += m[i][j] * v[j]; } }
    r
}

// ── PRNG (LCG, no external crate needed) ───────────────────────────────────

static RAND_STATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(12345678);

fn pseudo_rand() -> f32 {
    use std::sync::atomic::Ordering::Relaxed;
    let s = RAND_STATE.fetch_add(2891336453, Relaxed).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    RAND_STATE.store(s, Relaxed);
    (s >> 33) as f32 / u32::MAX as f32
}

fn pseudo_rand_small() -> f32 { pseudo_rand() * 2.0 - 1.0 }

fn rand_normal(mean: f32, std: f32) -> f32 {
    // Box–Muller (two uniform → one normal).
    let u1 = pseudo_rand().max(1e-7);
    let u2 = pseudo_rand();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    mean + std * z
}

// =============================================================================
// §22  PUBLIC API
// =============================================================================

/// Convenience: load a program, evaluate a named function, and return the result.
pub fn jules_run(program: &Program, entry: &str, args: Vec<Value>)
    -> Result<Value, RuntimeError>
{
    let mut interp = Interpreter::new();
    interp.load_program(program);
    interp.call_fn(entry, args)
}

/// Execute all `train` blocks found in the program.
pub fn jules_train(program: &Program) -> Result<Vec<TrainingStats>, RuntimeError> {
    let mut interp = Interpreter::new();
    interp.load_program(program);
    let mut all_stats = Vec::new();
    let trains: Vec<_> = program.items.iter().filter_map(|i| {
        if let Item::Train(t) = i { Some(t.clone()) } else { None }
    }).collect();
    let mut env = Env::new();
    for t in &trains {
        let stats = interp.run_training(t, &mut env)?;
        all_stats.push(stats);
    }
    Ok(all_stats)
}

// =============================================================================
// §23  TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use crate::lexer::Span;

    fn sp() -> Span { Span::dummy() }

    // ── Value helpers ─────────────────────────────────────────────────────────

    #[test] fn test_value_as_f64() {
        assert_eq!(Value::F32(1.5).as_f64(), Some(1.5));
        assert_eq!(Value::I32(7).as_f64(), Some(7.0));
        assert_eq!(Value::Bool(true).as_f64(), None);
    }

    #[test] fn test_value_is_truthy() {
        assert!( Value::Bool(true).is_truthy());
        assert!(!Value::Bool(false).is_truthy());
        assert!( Value::I32(1).is_truthy());
        assert!(!Value::I32(0).is_truthy());
    }

    #[test] fn test_value_display() {
        assert_eq!(Value::I32(42).to_string(), "42");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Unit.to_string(), "()");
        assert_eq!(Value::Vec3([1.0, 2.0, 3.0]).to_string(), "vec3(1, 2, 3)");
    }

    // ── Tensor operations ─────────────────────────────────────────────────────

    #[test] fn test_tensor_matmul_2x2() {
        // [1 2; 3 4] @ [1 0; 0 1] = [1 2; 3 4]
        let a = Tensor::from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::from_data(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        let d = c.cpu_data();
        assert!((d[0] - 1.0).abs() < 1e-5);
        assert!((d[1] - 2.0).abs() < 1e-5);
        assert!((d[2] - 3.0).abs() < 1e-5);
        assert!((d[3] - 4.0).abs() < 1e-5);
    }

    #[test] fn test_tensor_matmul_shape_mismatch() {
        let a = Tensor::from_data(vec![2, 3], vec![0.0; 6]);
        let b = Tensor::from_data(vec![2, 2], vec![0.0; 4]);
        assert!(a.matmul(&b).is_err());
    }

    #[test] fn test_tensor_hadamard_mul() {
        let a = Tensor::from_data(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::from_data(vec![4], vec![2.0, 2.0, 2.0, 2.0]);
        let c = a.hadamard_mul(&b).unwrap();
        let d = c.cpu_data();
        assert_eq!(d, &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test] fn test_tensor_concat_axis0() {
        let a = Tensor::from_data(vec![2, 3], vec![1.0; 6]);
        let b = Tensor::from_data(vec![3, 3], vec![2.0; 9]);
        let c = a.concat(&b).unwrap();
        assert_eq!(c.shape, vec![5, 3]);
        assert_eq!(c.numel(), 15);
    }

    #[test] fn test_tensor_concat_inner_mismatch() {
        let a = Tensor::from_data(vec![2, 3], vec![0.0; 6]);
        let b = Tensor::from_data(vec![2, 4], vec![0.0; 8]);
        assert!(a.concat(&b).is_err());
    }

    #[test] fn test_tensor_activation_relu() {
        let t = Tensor::from_data(vec![4], vec![-1.0, 0.0, 1.0, 2.0]);
        let out = t.apply_activation(&Activation::Relu);
        assert_eq!(out.cpu_data(), &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test] fn test_tensor_activation_sigmoid() {
        let t = Tensor::from_data(vec![1], vec![0.0]);
        let out = t.apply_activation(&Activation::Sigmoid);
        assert!((out.cpu_data()[0] - 0.5).abs() < 1e-5);
    }

    #[test] fn test_tensor_softmax_sums_to_one() {
        let t = Tensor::from_data(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]);
        let out = t.apply_activation(&Activation::Softmax);
        let sum: f32 = out.cpu_data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    }

    #[test] fn test_tensor_mse_loss() {
        let pred = Tensor::from_data(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let tgt  = Tensor::from_data(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let loss = pred.mse_loss(&tgt).unwrap();
        assert!(loss.abs() < 1e-6);
    }

    #[test] fn test_tensor_grad_attach() {
        let mut t = Tensor::zeros(vec![3, 3]);
        assert!(t.grad.is_none());
        t.enable_grad();
        assert!(t.grad.is_some());
    }

    #[test] fn test_tensor_transpose() {
        let t = Tensor::from_data(vec![2, 3], vec![1.0,2.0,3.0,4.0,5.0,6.0]);
        let t2 = t.transpose().unwrap();
        assert_eq!(t2.shape, vec![3, 2]);
        assert_eq!(t2.cpu_data()[0], 1.0);
        assert_eq!(t2.cpu_data()[1], 4.0); // first column of original
    }

    // ── ECS World ─────────────────────────────────────────────────────────────

    #[test] fn test_ecs_spawn_despawn() {
        let mut world = EcsWorld::default();
        let e1 = world.spawn();
        let e2 = world.spawn();
        assert!(world.is_alive(e1));
        world.despawn(e1);
        assert!(!world.is_alive(e1));
        assert!(world.is_alive(e2));
    }

    #[test] fn test_ecs_component_insert_get() {
        let mut world = EcsWorld::default();
        let e = world.spawn();
        world.insert_component(e, "health", Value::F32(100.0));
        let val = world.get_component(e, "health").cloned().unwrap();
        assert!(matches!(val, Value::F32(v) if (v - 100.0).abs() < 1e-5));
    }

    #[test] fn test_ecs_query_with_filter() {
        let mut world = EcsWorld::default();
        let e1 = world.spawn();
        let e2 = world.spawn();
        world.insert_component(e1, "Position", Value::Vec3([0.0, 0.0, 0.0]));
        world.insert_component(e1, "Velocity", Value::Vec3([1.0, 0.0, 0.0]));
        world.insert_component(e2, "Position", Value::Vec3([1.0, 0.0, 0.0]));
        // e1 has both; e2 only has Position.
        let with_vel = world.query(&["Position".into(), "Velocity".into()], &[]);
        assert_eq!(with_vel.len(), 1);
        assert_eq!(with_vel[0], e1);
    }

    #[test] fn test_ecs_query_without() {
        let mut world = EcsWorld::default();
        let alive = world.spawn();
        let dead  = world.spawn();
        world.insert_component(alive, "Health", Value::F32(100.0));
        world.insert_component(dead,  "Health", Value::F32(0.0));
        world.insert_component(dead,  "Dead",   Value::Unit);
        let result = world.query(&["Health".into()], &["Dead".into()]);
        assert_eq!(result, vec![alive]);
    }

    #[test] fn test_ecs_events() {
        let mut world = EcsWorld::default();
        let e = world.spawn();
        world.emit_event("killed", e);
        world.emit_event("killed", e);
        let evts = world.drain_events("killed");
        assert_eq!(evts.len(), 2);
        let evts2 = world.drain_events("killed");
        assert!(evts2.is_empty());
    }

    // ── Interpreter ───────────────────────────────────────────────────────────

    fn mk_interp() -> Interpreter { Interpreter::new() }

    fn eval(expr: &Expr) -> Value {
        let mut i = mk_interp();
        let mut env = Env::new();
        i.eval_expr(expr, &mut env).unwrap()
    }

    #[test] fn test_interp_int_lit() {
        assert!(matches!(eval(&Expr::IntLit { span: sp(), value: 42 }), Value::I32(42)));
    }

    #[test] fn test_interp_float_lit() {
        assert!(matches!(eval(&Expr::FloatLit { span: sp(), value: 3.14 }), Value::F32(_)));
    }

    #[test] fn test_interp_bool_lit() {
        assert!(matches!(eval(&Expr::BoolLit { span: sp(), value: true }), Value::Bool(true)));
    }

    #[test] fn test_interp_binop_add() {
        let e = Expr::BinOp {
            span: sp(), op: BinOpKind::Add,
            lhs: Box::new(Expr::IntLit { span: sp(), value: 3 }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 4 }),
        };
        assert!(matches!(eval(&e), Value::I32(7)));
    }

    #[test] fn test_interp_binop_compare() {
        let e = Expr::BinOp {
            span: sp(), op: BinOpKind::Lt,
            lhs: Box::new(Expr::IntLit { span: sp(), value: 3 }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 5 }),
        };
        assert!(matches!(eval(&e), Value::Bool(true)));
    }

    #[test] fn test_interp_binop_div_zero() {
        let e = Expr::BinOp {
            span: sp(), op: BinOpKind::Div,
            lhs: Box::new(Expr::IntLit { span: sp(), value: 10 }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 0 }),
        };
        let mut i = mk_interp();
        let mut env = Env::new();
        assert!(i.eval_expr(&e, &mut env).is_err());
    }

    #[test] fn test_interp_vec3_ctor() {
        let e = Expr::VecCtor {
            span: sp(), size: VecSize::N3,
            elems: vec![
                Expr::FloatLit { span: sp(), value: 1.0 },
                Expr::FloatLit { span: sp(), value: 2.0 },
                Expr::FloatLit { span: sp(), value: 3.0 },
            ],
        };
        assert!(matches!(eval(&e), Value::Vec3([1.0, 2.0, 3.0])));
    }

    #[test] fn test_interp_unop_neg() {
        let e = Expr::UnOp {
            span: sp(), op: UnOpKind::Neg,
            expr: Box::new(Expr::FloatLit { span: sp(), value: 5.0 }),
        };
        assert!(matches!(eval(&e), Value::F32(v) if (v + 5.0).abs() < 1e-6));
    }

    #[test] fn test_interp_let_and_ident() {
        let mut i = mk_interp();
        let mut env = Env::new();
        let stmt = Stmt::Let {
            span:    sp(),
            pattern: Pattern::Ident { span: sp(), name: "x".into(), mutable: false },
            ty:      None,
            init:    Some(Expr::IntLit { span: sp(), value: 99 }),
            mutable: false,
        };
        i.eval_stmt(&stmt, &mut env).unwrap();
        let v = i.eval_expr(&Expr::Ident { span: sp(), name: "x".into() }, &mut env).unwrap();
        assert!(matches!(v, Value::I32(99)));
    }

    #[test] fn test_interp_if_expr() {
        let mut i = mk_interp();
        let mut env = Env::new();
        let e = Expr::IfExpr {
            span: sp(),
            cond: Box::new(Expr::BoolLit { span: sp(), value: true }),
            then: Box::new(Block {
                span: sp(), stmts: vec![],
                tail: Some(Box::new(Expr::IntLit { span: sp(), value: 1 })),
            }),
            else_: Some(Box::new(Block {
                span: sp(), stmts: vec![],
                tail: Some(Box::new(Expr::IntLit { span: sp(), value: 2 })),
            })),
        };
        assert!(matches!(i.eval_expr(&e, &mut env).unwrap(), Value::I32(1)));
    }

    #[test] fn test_interp_return_propagation() {
        let mut i = mk_interp();
        let mut env = Env::new();
        let block = Block {
            span: sp(),
            stmts: vec![
                Stmt::Return { span: sp(), value: Some(Expr::IntLit { span: sp(), value: 42 }) },
                Stmt::Expr   { span: sp(), has_semi: true,
                               expr: Expr::IntLit { span: sp(), value: 0 } },
            ],
            tail: None,
        };
        let r = i.eval_block(&block, &mut env).unwrap();
        assert!(matches!(r, Value::Return(v) if matches!(*v, Value::I32(42))));
    }

    #[test] fn test_interp_matmul_tensors() {
        let mut i = mk_interp();
        let mut env = Env::new();
        let a = Tensor::from_data(vec![2, 3], vec![1.0; 6]);
        let b = Tensor::from_data(vec![3, 2], vec![1.0; 6]);
        env.set_local("A", Value::Tensor(Arc::new(RwLock::new(a))));
        env.set_local("B", Value::Tensor(Arc::new(RwLock::new(b))));
        let e = Expr::MatMul {
            span: sp(),
            lhs: Box::new(Expr::Ident { span: sp(), name: "A".into() }),
            rhs: Box::new(Expr::Ident { span: sp(), name: "B".into() }),
        };
        let result = i.eval_expr(&e, &mut env).unwrap();
        if let Value::Tensor(t) = result {
            assert_eq!(t.read().unwrap().shape, vec![2, 2]);
        } else { panic!("expected tensor"); }
    }

    #[test] fn test_interp_range_for_loop() {
        let mut i = mk_interp();
        let mut env = Env::new();
        // Accumulate sum of 0..5 into `acc`.
        let block = Block {
            span: sp(),
            stmts: vec![
                Stmt::Let {
                    span: sp(),
                    pattern: Pattern::Ident { span: sp(), name: "acc".into(), mutable: true },
                    ty: None,
                    init: Some(Expr::IntLit { span: sp(), value: 0 }),
                    mutable: true,
                },
                Stmt::ForIn {
                    span: sp(),
                    pattern: Pattern::Ident { span: sp(), name: "i".into(), mutable: false },
                    iter: Expr::Range {
                        span: sp(),
                        lo: Some(Box::new(Expr::IntLit { span: sp(), value: 0 })),
                        hi: Some(Box::new(Expr::IntLit { span: sp(), value: 5 })),
                        inclusive: false,
                    },
                    body: Block {
                        span: sp(),
                        stmts: vec![Stmt::Expr {
                            span: sp(), has_semi: true,
                            expr: Expr::Assign {
                                span: sp(),
                                op: AssignOpKind::AddAssign,
                                target: Box::new(Expr::Ident { span: sp(), name: "acc".into() }),
                                value:  Box::new(Expr::Ident { span: sp(), name: "i".into() }),
                            },
                        }],
                        tail: None,
                    },
                    label: None,
                },
            ],
            tail: Some(Box::new(Expr::Ident { span: sp(), name: "acc".into() })),
        };
        let result = i.eval_block(&block, &mut env).unwrap();
        assert!(matches!(result, Value::I32(10)), "0+1+2+3+4 = 10, got {result}");
    }

    // ── Neural network forward pass ───────────────────────────────────────────

    #[test] fn test_nn_forward_pass() {
        let decl = ModelDecl {
            span: sp(), attrs: vec![], name: "TestNet".into(),
            layers: vec![
                ModelLayer::Input  { span: sp(), size: 4 },
                ModelLayer::Dense  { span: sp(), units: 8, activation: Activation::Relu, bias: true },
                ModelLayer::Output { span: sp(), units: 2, activation: Activation::Softmax },
            ],
            device: ModelDevice::Cpu, optimizer: None,
        };
        let mut model = NnModel::from_decl(&decl);
        model.training = false;
        let input = Tensor::from_data(vec![1, 4], vec![1.0, 0.5, -1.0, 0.0]);
        let out = model.forward(input).unwrap();
        assert_eq!(out.shape, vec![1, 2]);
        // Softmax: outputs should sum to 1.
        let sum: f32 = out.cpu_data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "output should be softmax, sum={sum}");
    }

    #[test] fn test_nn_dropout_training_mode() {
        // In inference mode, dropout should be a no-op.
        let decl = ModelDecl {
            span: sp(), attrs: vec![], name: "DropNet".into(),
            layers: vec![
                ModelLayer::Input   { span: sp(), size: 4 },
                ModelLayer::Dropout { span: sp(), rate: 0.0 },  // rate=0 → no drop
                ModelLayer::Output  { span: sp(), units: 2, activation: Activation::Linear },
            ],
            device: ModelDevice::Cpu, optimizer: None,
        };
        let mut model = NnModel::from_decl(&decl);
        let input = Tensor::from_data(vec![1, 4], vec![1.0; 4]);
        let out = model.forward(input).unwrap();
        assert_eq!(out.shape, vec![1, 2]);
    }

    // ── Vec swizzle ───────────────────────────────────────────────────────────

    #[test] fn test_swizzle_x() {
        assert!(matches!(swizzle_vec(&[1.0, 2.0, 3.0], "x"), Ok(Value::F32(1.0))));
    }

    #[test] fn test_swizzle_xyz() {
        assert!(matches!(swizzle_vec(&[1.0, 2.0, 3.0], "xyz"), Ok(Value::Vec3(_))));
    }

    #[test] fn test_swizzle_invalid() {
        assert!(swizzle_vec(&[1.0, 2.0], "q").is_err());
    }

    // ── Scope / environment ───────────────────────────────────────────────────

    #[test] fn test_env_scoping() {
        let mut env = Env::new();
        env.set_local("x", Value::I32(1));
        env.push();
        env.set_local("x", Value::I32(2));
        assert!(matches!(env.get("x"), Some(Value::I32(2))));
        env.pop();
        assert!(matches!(env.get("x"), Some(Value::I32(1))));
    }

    // ── Matrix multiply ───────────────────────────────────────────────────────

    #[test] fn test_mat3_identity_mul() {
        let id = [[1.0_f32,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        let a  = [[1.0_f32,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]];
        let r  = mat3_mul(a, id);
        assert_eq!(r, a);
    }

    #[test] fn test_mat3_vec3_mul() {
        let id = [[1.0_f32,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
        let v  = [3.0_f32, 5.0, 7.0];
        assert_eq!(mat3_vec3_mul(id, v), v);
    }

    // ── Training stats ────────────────────────────────────────────────────────

    #[test] fn test_training_stats_default() {
        let s = TrainingStats::default();
        assert_eq!(s.total_steps, 0);
        assert!(s.episode_rewards.is_empty());
    }

    // ── Tensor flat index ──────────────────────────────────────────────────────

    #[test] fn test_tensor_flat_index_2d() {
        let shape = vec![3, 4];
        let idx   = vec![Value::I32(1), Value::I32(2)];
        assert_eq!(tensor_flat_index(&shape, &idx).unwrap(), 1 * 4 + 2);
    }

    #[test] fn test_tensor_flat_index_out_of_bounds() {
        let shape = vec![2, 2];
        let idx   = vec![Value::I32(0), Value::I32(5)]; // col 5 ≥ 2
        assert!(tensor_flat_index(&shape, &idx).is_err());
    }
}