#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// GPU buffer handle (opaque to users)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuBufferHandle {
    pub id: u64,
}

#[repr(C)]
pub enum GpuOp {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
}

pub enum GpuMemoryType {
    Float32,
    Float64,
    Int32,
    Int64,
}

#[derive(Clone)]
pub struct GpuBuffer {
    pub id: u64,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub device: String, // "cuda", "metal", "wgpu", "cpu"
}

/// Main trait for GPU backends
pub trait GpuBackendImpl: Send + Sync {
    /// Upload data to GPU
    fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuBufferHandle;

    /// Download data from GPU
    fn download(&self, handle: &GpuBufferHandle) -> Vec<f32>;

    /// Matrix multiplication: C = A @ B
    fn matmul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        out: &GpuBufferHandle,
    ) -> Result<(), String>;

    /// Element-wise operation
    fn elementwise(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        op: GpuOp,
        out: &GpuBufferHandle,
    ) -> Result<(), String>;

    /// Convolution operation (for neural networks)
    fn conv2d(
        &self,
        input: &GpuBufferHandle,
        kernel: &GpuBufferHandle,
        out: &GpuBufferHandle,
        stride: u32,
        padding: u32,
    ) -> Result<(), String>;

    /// Pool operation (max or avg)
    fn pool(
        &self,
        input: &GpuBufferHandle,
        out: &GpuBufferHandle,
        pool_size: u32,
        is_max: bool,
    ) -> Result<(), String>;

    /// Activation function
    fn activation(
        &self,
        input: &GpuBufferHandle,
        out: &GpuBufferHandle,
        activation: &str, // "relu", "sigmoid", "tanh", "softmax"
    ) -> Result<(), String>;

    /// Get backend name
    fn backend_name(&self) -> &'static str;

    /// Check if backend is available
    fn is_available(&self) -> bool;
}

// =============================================================================
// CPU Backend (CPU fallback for development/testing)
// =============================================================================

pub struct CpuBackend {
    buffers: Arc<Mutex<HashMap<u64, GpuBuffer>>>,
    next_id: Arc<Mutex<u64>>,
}

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend {
            buffers: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(1)),
        }
    }
}

impl GpuBackendImpl for CpuBackend {
    fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuBufferHandle {
        let mut buffers = self.buffers.lock().unwrap();
        let mut next_id = self.next_id.lock().unwrap();

        let id = *next_id;
        *next_id += 1;

        buffers.insert(
            id,
            GpuBuffer {
                id,
                data: data.to_vec(),
                shape,
                device: "cpu".to_string(),
            },
        );

        GpuBufferHandle { id }
    }

    fn download(&self, handle: &GpuBufferHandle) -> Vec<f32> {
        let buffers = self.buffers.lock().unwrap();
        buffers
            .get(&handle.id)
            .map(|b| b.data.clone())
            .unwrap_or_default()
    }

    fn matmul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        let mut buffers = self.buffers.lock().unwrap();
        let buf_a = buffers.get(&a.id).cloned().ok_or("Buffer A not found")?;
        let buf_b = buffers.get(&b.id).cloned().ok_or("Buffer B not found")?;

        // Simple CPU matmul
        if buf_a.shape.len() < 2 || buf_b.shape.len() < 2 {
            return Err("Matmul requires 2D+ tensors".into());
        }

        let m = buf_a.shape[buf_a.shape.len() - 2];
        let k = buf_a.shape[buf_a.shape.len() - 1];
        let n = buf_b.shape[buf_b.shape.len() - 1];

        if buf_b.shape[buf_b.shape.len() - 2] != k {
            return Err("Dimension mismatch in matmul".into());
        }

        let mut out_data = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let a_val = buf_a.data[i * k + p];
                for j in 0..n {
                    out_data[i * n + j] += a_val * buf_b.data[p * n + j];
                }
            }
        }

        let out_buf = buffers.get_mut(&out.id).ok_or("Output buffer not found")?;
        out_buf.data = out_data;
        out_buf.shape = vec![m, n];

        Ok(())
    }

    fn elementwise(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        _op: GpuOp,
        _out: &GpuBufferHandle,
    ) -> Result<(), String> {
        let _buffers = self.buffers.lock().unwrap();

        if _buffers.contains_key(&a.id) && _buffers.contains_key(&b.id) {
            Ok(())
        } else {
            Err("Buffer not found".into())
        }
    }

    fn conv2d(
        &self,
        _input: &GpuBufferHandle,
        _kernel: &GpuBufferHandle,
        _out: &GpuBufferHandle,
        _stride: u32,
        _padding: u32,
    ) -> Result<(), String> {
        Ok(())
    }

    fn pool(
        &self,
        _input: &GpuBufferHandle,
        _out: &GpuBufferHandle,
        _pool_size: u32,
        _is_max: bool,
    ) -> Result<(), String> {
        Ok(())
    }

    fn activation(
        &self,
        _input: &GpuBufferHandle,
        _out: &GpuBufferHandle,
        _activation: &str,
    ) -> Result<(), String> {
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }
}

// =============================================================================
// WGPU Backend (WebGPU - cross-platform, Wasm-compatible)
// =============================================================================

pub struct WgpuBackend {
    // In a real implementation, this would contain actual wgpu device, queue, etc.
    buffers: Arc<Mutex<HashMap<u64, Vec<u8>>>>,
    next_id: Arc<Mutex<u64>>,
}

impl WgpuBackend {
    pub fn new() -> Result<Self, String> {
        // In real implementation: pollster::block_on(Self::new_async())
        Ok(WgpuBackend {
            buffers: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(1)),
        })
    }
}

impl GpuBackendImpl for WgpuBackend {
    fn upload(&self, data: &[f32], _shape: Vec<usize>) -> GpuBufferHandle {
        let mut buffers = self.buffers.lock().unwrap();
        let mut next_id = self.next_id.lock().unwrap();

        let id = *next_id;
        *next_id += 1;

        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        buffers.insert(id, bytes);

        GpuBufferHandle { id }
    }

    fn download(&self, handle: &GpuBufferHandle) -> Vec<f32> {
        let buffers = self.buffers.lock().unwrap();
        buffers
            .get(&handle.id)
            .map(|bytes| {
                bytes
                    .chunks(4)
                    .map(|chunk| {
                        let mut arr = [0u8; 4];
                        arr.copy_from_slice(chunk);
                        f32::from_le_bytes(arr)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn matmul(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _out: &GpuBufferHandle,
    ) -> Result<(), String> {
        // In real implementation: launch compute shader
        Ok(())
    }

    fn elementwise(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _op: GpuOp,
        _out: &GpuBufferHandle,
    ) -> Result<(), String> {
        // In real implementation: launch compute shader
        Ok(())
    }

    fn conv2d(
        &self,
        _input: &GpuBufferHandle,
        _kernel: &GpuBufferHandle,
        _out: &GpuBufferHandle,
        _stride: u32,
        _padding: u32,
    ) -> Result<(), String> {
        // In real implementation: launch conv compute shader
        Ok(())
    }

    fn pool(
        &self,
        _input: &GpuBufferHandle,
        _out: &GpuBufferHandle,
        _pool_size: u32,
        _is_max: bool,
    ) -> Result<(), String> {
        // In real implementation: launch pool compute shader
        Ok(())
    }

    fn activation(
        &self,
        _input: &GpuBufferHandle,
        _out: &GpuBufferHandle,
        _activation: &str,
    ) -> Result<(), String> {
        // In real implementation: launch activation compute shader
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "wgpu"
    }

    fn is_available(&self) -> bool {
        // In real implementation: check if GPU is available
        true
    }
}

// =============================================================================
// Multi-Backend Selector
// =============================================================================

pub enum GpuBackend {
    Cpu(Arc<CpuBackend>),
    Wgpu(Arc<WgpuBackend>),
}

impl GpuBackend {
    /// Auto-select best available GPU backend
    pub fn auto_select() -> Self {
        // Try WGPU first, fall back to CPU
        match WgpuBackend::new() {
            Ok(backend) => GpuBackend::Wgpu(Arc::new(backend)),
            Err(_) => GpuBackend::Cpu(Arc::new(CpuBackend::new())),
        }
    }

    /// Force CPU backend
    pub fn cpu() -> Self {
        GpuBackend::Cpu(Arc::new(CpuBackend::new()))
    }

    /// Get backend implementation trait object
    pub fn as_impl(&self) -> &dyn GpuBackendImpl {
        match self {
            GpuBackend::Cpu(backend) => backend.as_ref(),
            GpuBackend::Wgpu(backend) => backend.as_ref(),
        }
    }

    pub fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuBufferHandle {
        self.as_impl().upload(data, shape)
    }

    pub fn download(&self, handle: &GpuBufferHandle) -> Vec<f32> {
        self.as_impl().download(handle)
    }

    pub fn matmul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        self.as_impl().matmul(a, b, out)
    }

    pub fn elementwise(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        op: GpuOp,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        self.as_impl().elementwise(a, b, op, out)
    }

    pub fn backend_name(&self) -> &'static str {
        self.as_impl().backend_name()
    }

    pub fn is_available(&self) -> bool {
        self.as_impl().is_available()
    }
}

// =============================================================================
// GPU Memory Manager (handles allocation and cleanup)
// =============================================================================

pub struct GpuMemoryManager {
    backend: GpuBackend,
    allocated: Arc<Mutex<HashMap<u64, GpuBuffer>>>,
    current_bytes: Arc<Mutex<usize>>,
    config: Arc<Mutex<GpuMemoryConfig>>,
}

#[derive(Debug, Clone, Copy)]
pub struct GpuMemoryConfig {
    /// Memory reserved for runtime + model base footprint.
    pub base_reserved_bytes: usize,
    /// Additional bytes allowed above `base_reserved_bytes`.
    pub max_extra_bytes: usize,
}

impl GpuMemoryConfig {
    pub fn total_budget_bytes(self) -> usize {
        self.base_reserved_bytes
            .saturating_add(self.max_extra_bytes)
    }
}

impl GpuMemoryManager {
    pub fn new(backend: GpuBackend) -> Self {
        Self::new_with_config(
            backend,
            GpuMemoryConfig {
                base_reserved_bytes: 0,
                max_extra_bytes: usize::MAX / 2,
            },
        )
    }

    pub fn new_with_config(backend: GpuBackend, config: GpuMemoryConfig) -> Self {
        GpuMemoryManager {
            backend,
            allocated: Arc::new(Mutex::new(HashMap::new())),
            current_bytes: Arc::new(Mutex::new(0)),
            config: Arc::new(Mutex::new(config)),
        }
    }

    pub fn set_base_reserved_bytes(&self, bytes: usize) {
        let mut cfg = self.config.lock().unwrap();
        cfg.base_reserved_bytes = bytes;
    }

    pub fn set_max_extra_bytes(&self, bytes: usize) {
        let mut cfg = self.config.lock().unwrap();
        cfg.max_extra_bytes = bytes;
    }

    pub fn memory_budget_bytes(&self) -> usize {
        self.config.lock().unwrap().total_budget_bytes()
    }

    pub fn used_bytes(&self) -> usize {
        *self.current_bytes.lock().unwrap()
    }

    pub fn allocate(&self, shape: Vec<usize>, init_val: f32) -> Result<GpuBufferHandle, String> {
        let numel: usize = shape.iter().product();
        let data = vec![init_val; numel];
        self.allocate_from_data(shape, data)
    }

    pub fn allocate_from_data(
        &self,
        shape: Vec<usize>,
        data: Vec<f32>,
    ) -> Result<GpuBufferHandle, String> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err("Data length does not match shape".into());
        }
        let requested_bytes = numel.saturating_mul(std::mem::size_of::<f32>());
        let budget_bytes = self.memory_budget_bytes();
        let mut used = self.current_bytes.lock().unwrap();
        if used.saturating_add(requested_bytes) > budget_bytes {
            return Err(format!(
                "GPU memory budget exceeded: requested={} used={} budget={}",
                requested_bytes, *used, budget_bytes
            ));
        }

        let shape_clone = shape.clone();
        let handle = self.backend.upload(&data, shape);
        self.allocated.lock().unwrap().insert(
            handle.id,
            GpuBuffer {
                id: handle.id,
                data,
                shape: shape_clone,
                device: self.backend.backend_name().to_string(),
            },
        );
        *used += requested_bytes;
        Ok(handle)
    }

    pub fn free(&self, handle: &GpuBufferHandle) {
        let removed = self.allocated.lock().unwrap().remove(&handle.id);
        if let Some(buffer) = removed {
            let mut used = self.current_bytes.lock().unwrap();
            let bytes = buffer.data.len().saturating_mul(std::mem::size_of::<f32>());
            *used = used.saturating_sub(bytes);
        }
    }

    pub fn get_stats(&self) -> (usize, usize) {
        let allocated = self.allocated.lock().unwrap();
        let count = allocated.len();
        let total_elements: usize = allocated.values().map(|b| b.data.len()).sum();
        (count, total_elements)
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.backend_name()
    }

    pub fn matmul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        self.backend.matmul(a, b, out)
    }

    pub fn download(&self, handle: &GpuBufferHandle) -> Vec<f32> {
        self.backend.download(handle)
    }
}

// =============================================================================
// Kernel implementations (compute shaders for common operations)
// =============================================================================

pub struct GpuKernels;

impl GpuKernels {
    /// WGSL (WebGPU Shading Language) for matrix multiplication
    pub const MATMUL_KERNEL: &'static str = r#"
@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_out: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    var sum = 0.0;
    for (var k = 0u; k < 256u; k = k + 1u) {
        sum += matrix_a[row * 256u + k] * matrix_b[k * 256u + col];
    }

    matrix_out[row * 256u + col] = sum;
}
    "#;

    /// WGSL for ReLU activation
    pub const RELU_KERNEL: &'static str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output[idx] = max(0.0, input[idx]);
}
    "#;

    /// WGSL for element-wise addition
    pub const ADD_KERNEL: &'static str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    out[idx] = a[idx] + b[idx];
}
    "#;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend() {
        let backend = CpuBackend::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let handle = backend.upload(&data, vec![2, 2]);
        let downloaded = backend.download(&handle);
        assert_eq!(downloaded, data);
    }

    #[test]
    fn test_auto_select() {
        let backend = GpuBackend::auto_select();
        assert!(backend.is_available());
        assert!(!backend.backend_name().is_empty());
    }

    #[test]
    fn test_memory_manager() {
        let backend = GpuBackend::cpu();
        let manager = GpuMemoryManager::new(backend);
        let handle = manager.allocate(vec![10, 10], 0.0).unwrap();
        let (count, total) = manager.get_stats();
        assert!(count > 0);
        assert_eq!(total, 100);
        assert!(manager.used_bytes() >= 400);
        manager.free(&handle);
    }

    #[test]
    fn test_memory_budget_limit() {
        let backend = GpuBackend::cpu();
        let manager = GpuMemoryManager::new_with_config(
            backend,
            GpuMemoryConfig {
                base_reserved_bytes: 128,
                max_extra_bytes: 128,
            },
        );
        let too_large = manager.allocate(vec![100], 0.0);
        assert!(too_large.is_err());
    }
}
