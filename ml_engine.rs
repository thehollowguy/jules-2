// =========================================================================
// Automatic Differentiation (Autodiff) Engine for Jules
// Enables full backpropagation through neural networks
// =========================================================================

use crate::gpu_backend::GpuMemoryManager;
use std::collections::HashMap;
use std::thread;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct ComputationGraph {
    /// Each node represents an operation
    pub nodes: HashMap<u64, ComputeNode>,
    /// Node ID counter
    pub next_id: u64,
}

#[derive(Debug, Clone)]
pub struct ComputeNode {
    pub id: u64,
    pub op: Operation,
    pub inputs: Vec<u64>, // IDs of input nodes
    pub value: Tensor,
    pub gradient: Option<Tensor>,
    pub requires_grad: bool,
}

#[derive(Debug, Clone)]
pub enum Operation {
    Input,                             // Input leaf node
    Constant,                          // Constant value
    Add,                               // Element-wise addition
    Sub,                               // Element-wise subtraction
    Mul,                               // Element-wise multiplication
    Div,                               // Element-wise division
    MatMul,                            // Matrix multiplication
    ReLU,                              // ReLU activation
    Sigmoid,                           // Sigmoid activation
    Tanh,                              // Tanh activation
    Softmax,                           // Softmax activation
    Sum,                               // Sum all elements
    Mean,                              // Mean of all elements
    Reshape { new_shape: Vec<usize> }, // Reshape tensor
    Transpose { axes: Vec<usize> },    // Transpose axes
}

// Simple pseudo-random based on input for deterministic initialization
fn _as_pseudo_rand(seed: usize) -> usize {
    let mut x = seed.wrapping_mul(2654435761);
    x = ((x >> 16) ^ x).wrapping_mul(0x7feb352d);
    (x >> 15) ^ x
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Experimental INT8 weight format for inference-only execution.
/// Stores one signed byte per weight plus per-output scales.
#[derive(Debug, Clone)]
pub struct Int8LinearWeights {
    pub in_dim: usize,
    pub out_dim: usize,
    pub qweights: Vec<i8>,
    pub scales: Vec<f32>,
}

impl Int8LinearWeights {
    /// Effective bytes-per-parameter including scale overhead.
    pub fn effective_bytes_per_param(&self) -> f32 {
        let weight_bytes = self.qweights.len() as f32;
        let scale_bytes = (self.scales.len() * std::mem::size_of::<f32>()) as f32;
        (weight_bytes + scale_bytes) / self.qweights.len().max(1) as f32
    }
}

impl Tensor {
    const PARALLEL_MATMUL_MIN_OPS: usize = 2_000_000;

    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Tensor {
            shape,
            data: vec![0.0; numel],
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Tensor {
            shape,
            data: vec![1.0; numel],
        }
    }

    /// Xavier (Glorot) initialization - good for sigmoid/tanh
    pub fn xavier(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let limit = (6.0
            / (shape.get(0).copied().unwrap_or(1) + shape.get(1).copied().unwrap_or(1)) as f32)
            .sqrt();
        let data = (0..numel)
            .map(|_| {
                // Pseudo-random number between -limit and limit
                let r = ((numel as f32 * (_as_pseudo_rand(numel) as f32)).sin() * limit.abs());
                if (numel ^ _as_pseudo_rand(numel * 2)) & 1 == 0 {
                    r
                } else {
                    -r
                }
            })
            .collect();
        Tensor { shape, data }
    }

    /// He initialization - good for ReLU
    pub fn he(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let fan_in = shape.get(0).copied().unwrap_or(1) as f32;
        let std = (2.0 / fan_in).sqrt();
        let data = (0..numel)
            .map(|i| {
                let r = ((i as f32 * std * 3.14159).sin() * std);
                r.max(-3.0 * std).min(3.0 * std)
            })
            .collect();
        Tensor { shape, data }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in addition");
        let len = self.data.len();
        let mut data = vec![0.0; len];
        for i in 0..len {
            data[i] = self.data[i] + other.data[i];
        }
        Tensor {
            shape: self.shape.clone(),
            data,
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in multiplication");
        let len = self.data.len();
        let mut data = vec![0.0; len];
        for i in 0..len {
            data[i] = self.data[i] * other.data[i];
        }
        Tensor {
            shape: self.shape.clone(),
            data,
        }
    }

    pub fn relu(&self) -> Tensor {
        let mut data = vec![0.0; self.data.len()];
        for (i, x) in self.data.iter().enumerate() {
            data[i] = x.max(0.0);
        }
        Tensor {
            shape: self.shape.clone(),
            data,
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        let mut data = vec![0.0; self.data.len()];
        for (i, x) in self.data.iter().enumerate() {
            data[i] = 1.0 / (1.0 + (-x).exp());
        }
        Tensor {
            shape: self.shape.clone(),
            data,
        }
    }

    pub fn softmax(&self) -> Tensor {
        // Numerically stable softmax
        let max = self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f32> = self.data.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();

        let data = exp_vals.iter().map(|x| x / sum).collect();

        Tensor {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Numerically stable softmax over the last dimension.
    /// Useful for batched logits with shape `[batch, classes]` and higher-rank tensors.
    pub fn softmax_last_dim(&self) -> Result<Tensor, String> {
        let last_dim = *self
            .shape
            .last()
            .ok_or_else(|| "softmax_last_dim expects rank >= 1".to_string())?;
        if last_dim == 0 {
            return Err("softmax_last_dim last dimension must be > 0".into());
        }

        let rows = self.numel() / last_dim;
        let mut out = vec![0.0f32; self.numel()];

        for r in 0..rows {
            let base = r * last_dim;
            let row = &self.data[base..base + last_dim];
            let row_max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0f32;
            for i in 0..last_dim {
                let e = (row[i] - row_max).exp();
                out[base + i] = e;
                exp_sum += e;
            }
            let inv_sum = 1.0 / exp_sum.max(1e-12);
            for i in 0..last_dim {
                out[base + i] *= inv_sum;
            }
        }

        Ok(Tensor {
            shape: self.shape.clone(),
            data: out,
        })
    }

    /// GELU activation (tanh approximation used by many Transformer models).
    pub fn gelu(&self) -> Tensor {
        let k = (2.0 / std::f32::consts::PI).sqrt();
        let data = self
            .data
            .iter()
            .map(|x| {
                let cubic = 0.044715 * x * x * x;
                0.5 * x * (1.0 + (k * (x + cubic)).tanh())
            })
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Backward pass for GELU (tanh approximation).
    /// Takes `upstream_grad` of the same shape as `self`.
    pub fn gelu_backward(&self, upstream_grad: &Tensor) -> Result<Tensor, String> {
        if self.shape != upstream_grad.shape {
            return Err("gelu_backward shape mismatch".into());
        }
        let k = (2.0 / std::f32::consts::PI).sqrt();
        let data = self
            .data
            .iter()
            .zip(&upstream_grad.data)
            .map(|(x, g)| {
                let x2 = x * x;
                let x3 = x2 * x;
                let inner = k * (x + 0.044715 * x3);
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner * tanh_inner;
                let d_inner = k * (1.0 + 3.0 * 0.044715 * x2);
                let d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner;
                g * d_gelu
            })
            .collect();
        Ok(Tensor {
            shape: self.shape.clone(),
            data,
        })
    }

    /// SiLU / Swish activation: `x * sigmoid(x)`.
    pub fn silu(&self) -> Tensor {
        let data = self
            .data
            .iter()
            .map(|x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                x * sig
            })
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Backward pass for SiLU / Swish.
    /// Takes `upstream_grad` of the same shape as `self`.
    pub fn silu_backward(&self, upstream_grad: &Tensor) -> Result<Tensor, String> {
        if self.shape != upstream_grad.shape {
            return Err("silu_backward shape mismatch".into());
        }
        let data = self
            .data
            .iter()
            .zip(&upstream_grad.data)
            .map(|(x, g)| {
                let sig = 1.0 / (1.0 + (-x).exp());
                let d_silu = sig + x * sig * (1.0 - sig);
                g * d_silu
            })
            .collect();
        Ok(Tensor {
            shape: self.shape.clone(),
            data,
        })
    }

    /// LayerNorm over the last dimension.
    /// `gamma`/`beta` are optional affine params of shape `[last_dim]`.
    pub fn layer_norm_last_dim(
        &self,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor, String> {
        let last_dim = *self
            .shape
            .last()
            .ok_or_else(|| "layer_norm_last_dim expects rank >= 1".to_string())?;
        if last_dim == 0 {
            return Err("layer_norm_last_dim last dimension must be > 0".into());
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err("layer_norm_last_dim eps must be finite and > 0".into());
        }

        if let Some(g) = gamma {
            if g.shape != vec![last_dim] {
                return Err(format!(
                    "layer_norm_last_dim gamma must be shape [{}]",
                    last_dim
                ));
            }
        }
        if let Some(b) = beta {
            if b.shape != vec![last_dim] {
                return Err(format!(
                    "layer_norm_last_dim beta must be shape [{}]",
                    last_dim
                ));
            }
        }

        let rows = self.data.len() / last_dim;
        let mut out = vec![0.0f32; self.data.len()];
        for r in 0..rows {
            let base = r * last_dim;
            let row = &self.data[base..base + last_dim];
            let mean = row.iter().sum::<f32>() / last_dim as f32;
            let var = row
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .sum::<f32>()
                / last_dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..last_dim {
                let mut v = (row[i] - mean) * inv_std;
                if let Some(g) = gamma {
                    v *= g.data[i];
                }
                if let Some(b) = beta {
                    v += b.data[i];
                }
                out[base + i] = v;
            }
        }
        Ok(Tensor {
            shape: self.shape.clone(),
            data: out,
        })
    }

    /// RMSNorm over the last dimension.
    /// `weight` is optional scale param of shape `[last_dim]`.
    pub fn rms_norm_last_dim(&self, weight: Option<&Tensor>, eps: f32) -> Result<Tensor, String> {
        let last_dim = *self
            .shape
            .last()
            .ok_or_else(|| "rms_norm_last_dim expects rank >= 1".to_string())?;
        if last_dim == 0 {
            return Err("rms_norm_last_dim last dimension must be > 0".into());
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err("rms_norm_last_dim eps must be finite and > 0".into());
        }
        if let Some(w) = weight {
            if w.shape != vec![last_dim] {
                return Err(format!(
                    "rms_norm_last_dim weight must be shape [{}]",
                    last_dim
                ));
            }
        }

        let rows = self.data.len() / last_dim;
        let mut out = vec![0.0f32; self.data.len()];
        for r in 0..rows {
            let base = r * last_dim;
            let row = &self.data[base..base + last_dim];
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / last_dim as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..last_dim {
                let mut v = row[i] * inv_rms;
                if let Some(w) = weight {
                    v *= w.data[i];
                }
                out[base + i] = v;
            }
        }
        Ok(Tensor {
            shape: self.shape.clone(),
            data: out,
        })
    }

    /// Native scaled dot-product attention for tensors shaped:
    /// - q: [batch, q_len, d]
    /// - k: [batch, kv_len, d]
    /// - v: [batch, kv_len, v_dim]
    pub fn scaled_dot_product_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
    ) -> Result<Tensor, String> {
        if q.shape.len() != 3 || k.shape.len() != 3 || v.shape.len() != 3 {
            return Err("scaled_dot_product_attention expects rank-3 q,k,v tensors".into());
        }
        let (bq, q_len, d) = (q.shape[0], q.shape[1], q.shape[2]);
        let (bk, kv_len, kd) = (k.shape[0], k.shape[1], k.shape[2]);
        let (bv, vv_len, v_dim) = (v.shape[0], v.shape[1], v.shape[2]);
        if bq != bk || bq != bv {
            return Err("scaled_dot_product_attention batch mismatch".into());
        }
        if kd != d {
            return Err("scaled_dot_product_attention q/k hidden size mismatch".into());
        }
        if vv_len != kv_len {
            return Err("scaled_dot_product_attention k/v sequence mismatch".into());
        }
        if d == 0 || v_dim == 0 {
            return Err("scaled_dot_product_attention dimensions must be > 0".into());
        }

        let mut out = vec![0.0f32; bq * q_len * v_dim];
        let scale = 1.0 / (d as f32).sqrt();

        for b in 0..bq {
            for t in 0..q_len {
                let q_base = (b * q_len + t) * d;
                let q_row = &q.data[q_base..q_base + d];

                let mut scores = vec![f32::NEG_INFINITY; kv_len];
                let mut max_score = f32::NEG_INFINITY;
                for s in 0..kv_len {
                    if causal && s > t {
                        continue;
                    }
                    let k_base = (b * kv_len + s) * d;
                    let k_row = &k.data[k_base..k_base + d];
                    let mut dot = 0.0f32;
                    for i in 0..d {
                        dot += q_row[i] * k_row[i];
                    }
                    let score = dot * scale;
                    scores[s] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                let mut denom = 0.0f32;
                for s in 0..kv_len {
                    if scores[s].is_finite() {
                        scores[s] = (scores[s] - max_score).exp();
                        denom += scores[s];
                    } else {
                        scores[s] = 0.0;
                    }
                }
                let denom = denom.max(1e-12);

                let out_base = (b * q_len + t) * v_dim;
                for s in 0..kv_len {
                    let w = scores[s] / denom;
                    if w == 0.0 {
                        continue;
                    }
                    let v_base = (b * kv_len + s) * v_dim;
                    for c in 0..v_dim {
                        out[out_base + c] += w * v.data[v_base + c];
                    }
                }
            }
        }

        Ok(Tensor {
            shape: vec![bq, q_len, v_dim],
            data: out,
        })
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.numel() as f32
    }

    /// Compute gradient for ReLU
    pub fn relu_grad(&self, upstream_grad: &Tensor) -> Tensor {
        let len = upstream_grad.data.len();
        let mut data = vec![0.0; len];
        for i in 0..len {
            data[i] = if self.data[i] > 0.0 {
                upstream_grad.data[i]
            } else {
                0.0
            };
        }
        Tensor {
            shape: upstream_grad.shape.clone(),
            data,
        }
    }

    /// Compute gradient for Sigmoid
    pub fn sigmoid_grad(&self, upstream_grad: &Tensor) -> Tensor {
        let len = upstream_grad.data.len();
        let mut data = vec![0.0; len];
        for i in 0..len {
            let x = self.data[i];
            data[i] = upstream_grad.data[i] * x * (1.0 - x);
        }
        Tensor {
            shape: upstream_grad.shape.clone(),
            data,
        }
    }

    #[inline]
    fn use_native_jules_kernel(m: usize, k: usize, n: usize) -> bool {
        // Native Jules heuristic: route larger or "long-and-skinny" GEMMs
        // to the in-tree Jules kernel sooner than scalar fallback kernels.
        let ops = m.saturating_mul(k).saturating_mul(n);
        ops >= 131_072 || (k >= 256 && (m >= 16 || n >= 16))
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Left must be 2D");
        assert_eq!(other.shape.len(), 2, "Right must be 2D");
        assert_eq!(
            self.shape[1], other.shape[0],
            "Incompatible shapes for matmul"
        );

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let other_t = transpose_2d(&other.data, k, n);
        let mut result = vec![0.0; m * n];

        let ops = m.saturating_mul(k).saturating_mul(n);
        let threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        if threads > 1 && ops >= Self::PARALLEL_MATMUL_MIN_OPS {
            let rows_per_chunk = m.div_ceil(threads);
            thread::scope(|scope| {
                for (chunk_idx, result_chunk) in result.chunks_mut(rows_per_chunk * n).enumerate() {
                    let row_start = chunk_idx * rows_per_chunk;
                    let row_end = (row_start + rows_per_chunk).min(m);
                    let a_data = &self.data;
                    let bt_data = &other_t;

                    scope.spawn(move || {
                        for row in row_start..row_end {
                            let local_row = row - row_start;
                            let out_row = &mut result_chunk[local_row * n..(local_row + 1) * n];
                            let a_row = &a_data[row * k..(row + 1) * k];
                            for col in 0..n {
                                let b_row = &bt_data[col * k..(col + 1) * k];
                                out_row[col] = dot_unrolled_8(a_row, b_row);
                            }
                        }
                    });
                }
            });
        } else {
            for i in 0..m {
                let a_row = &self.data[i * k..(i + 1) * k];
                for j in 0..n {
                    let b_row = &other_t[j * k..(j + 1) * k];
                    result[i * n + j] = dot_unrolled_8(a_row, b_row);
                }
            }
        }

        Tensor {
            shape: vec![m, n],
            data: result,
        }
    }

    /// Stress helper for runtime tuning. Allocates large buffers and reports
    /// total wall-clock time for repeated matrix multiplies.
    pub fn benchmark_matmul_ms(dim: usize, iterations: usize) -> u128 {
        let a = Tensor::he(vec![dim, dim]);
        let b = Tensor::xavier(vec![dim, dim]);
        let start = Instant::now();
        let mut out = Tensor::zeros(vec![dim, dim]);
        for _ in 0..iterations {
            out = a.matmul(&b);
        }
        let _ = out.data.first().copied().unwrap_or_default();
        start.elapsed().as_millis()
    }

    /// GPU matmul path with explicit memory budget control.
    /// Callers can set base + max extra bytes on `GpuMemoryManager` before running.
    pub fn matmul_gpu(&self, other: &Tensor, memory: &GpuMemoryManager) -> Result<Tensor, String> {
        assert_eq!(self.shape.len(), 2, "Left must be 2D");
        assert_eq!(other.shape.len(), 2, "Right must be 2D");
        assert_eq!(
            self.shape[1], other.shape[0],
            "Incompatible shapes for matmul"
        );

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let a = memory.allocate_from_data(self.shape.clone(), self.data.clone())?;
        let b = memory.allocate_from_data(other.shape.clone(), other.data.clone())?;
        let out = memory.allocate(vec![m, n], 0.0)?;

        memory.matmul(&a, &b, &out)?;
        let result = memory.download(&out);

        memory.free(&a);
        memory.free(&b);
        memory.free(&out);

        if result.len() != m * n {
            return Err("GPU matmul returned incorrect output size".into());
        }

        Ok(Tensor {
            shape: vec![m, n],
            data: result,
        })
    }

    /// Compute gradient for matmul: dL/dA = dL/dC @ B^T
    pub fn matmul_grad_a(&self, upstream_grad: &Tensor, b: &Tensor) -> Tensor {
        let b_t_shape = vec![b.shape[1], b.shape[0]];
        let b_t_data = transpose_2d(&b.data, b.shape[0], b.shape[1]);

        let b_t = Tensor {
            shape: b_t_shape,
            data: b_t_data,
        };

        upstream_grad.matmul(&b_t)
    }

    /// Compute gradient for matmul: dL/dB = A^T @ dL/dC
    pub fn matmul_grad_b(&self, a: &Tensor, upstream_grad: &Tensor) -> Tensor {
        let a_t_shape = vec![a.shape[1], a.shape[0]];
        let a_t_data = transpose_2d(&a.data, a.shape[0], a.shape[1]);

        let a_t = Tensor {
            shape: a_t_shape,
            data: a_t_data,
        };

        a_t.matmul(upstream_grad)
    }

    /// Clone the gradient structure
    pub fn clone_shape(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }

    /// Clip gradients by value to prevent exploding gradients
    pub fn clip_by_value(&self, min: f32, max: f32) -> Tensor {
        let mut data = vec![0.0; self.data.len()];
        for (i, x) in self.data.iter().enumerate() {
            data[i] = x.max(min).min(max);
        }
        Tensor {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Clip gradients by global norm
    pub fn clip_by_norm(&self, max_norm: f32) -> Tensor {
        let norm = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm <= max_norm || norm < 1e-8 {
            self.clone_shape()
        } else {
            Tensor {
                shape: self.shape.clone(),
                data: self.data.iter().map(|x| x * (max_norm / norm)).collect(),
            }
        }
    }

    /// Normalize tensor to zero mean and unit variance
    pub fn normalize(&self) -> Tensor {
        let mean = self.mean();
        let variance =
            self.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.numel().max(1) as f32;
        let std = variance.sqrt() + 1e-8;

        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| (x - mean) / std).collect(),
        }
    }

    /// Quantize a `[in_dim, out_dim]` weight tensor to signed int8.
    /// This is for inference (running) only, not training.
    pub fn quantize_linear_int8(&self) -> Result<Int8LinearWeights, String> {
        if self.shape.len() != 2 {
            return Err("quantize_linear_int8 expects a 2D weight tensor".into());
        }
        let in_dim = self.shape[0];
        let out_dim = self.shape[1];
        let mut qweights = vec![0i8; in_dim * out_dim];
        let mut scales = vec![1.0f32; out_dim];

        for o in 0..out_dim {
            let mut max_abs = 0.0f32;
            for i in 0..in_dim {
                max_abs = max_abs.max(self.data[i * out_dim + o].abs());
            }
            let scale = if max_abs < 1e-8 { 1.0 } else { max_abs / 127.0 };
            scales[o] = scale;
            for i in 0..in_dim {
                let w = self.data[i * out_dim + o] / scale;
                qweights[i * out_dim + o] = w.round().clamp(-127.0, 127.0) as i8;
            }
        }

        Ok(Int8LinearWeights {
            in_dim,
            out_dim,
            qweights,
            scales,
        })
    }

    /// Run a linear projection with INT8 weights:
    /// input `[batch, in_dim]` x weights `[in_dim, out_dim]` -> `[batch, out_dim]`.
    pub fn linear_int8(
        &self,
        weights: &Int8LinearWeights,
        bias: Option<&Tensor>,
    ) -> Result<Tensor, String> {
        if self.shape.len() != 2 {
            return Err("linear_int8 expects a 2D input tensor".into());
        }
        let batch = self.shape[0];
        let in_dim = self.shape[1];
        if in_dim != weights.in_dim {
            return Err("linear_int8 input dimension mismatch".into());
        }
        if let Some(b) = bias {
            if b.shape.len() != 1 || b.shape[0] != weights.out_dim {
                return Err("linear_int8 bias must be shape [out_dim]".into());
            }
        }

        let mut out = vec![0.0f32; batch * weights.out_dim];
        for b in 0..batch {
            for o in 0..weights.out_dim {
                let mut acc = 0.0f32;
                let scale = weights.scales[o];
                for i in 0..in_dim {
                    let x = self.data[b * in_dim + i];
                    let qw = weights.qweights[i * weights.out_dim + o] as f32 * scale;
                    acc += x * qw;
                }
                if let Some(bias_t) = bias {
                    acc += bias_t.data[o];
                }
                out[b * weights.out_dim + o] = acc;
            }
        }

        Ok(Tensor {
            shape: vec![batch, weights.out_dim],
            data: out,
        })
    }
}

fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

fn matmul_blocked_rows(
    a_data: &[f32],
    bt_data: &[f32],
    row_start: usize,
    row_end: usize,
    k: usize,
    n: usize,
    out_chunk: &mut [f32],
    out_row_base: usize,
) {
    const BK: usize = 64;
    const BN: usize = 64;

    for row in row_start..row_end {
        let out_row_local = row - out_row_base;
        let out_row = &mut out_chunk[out_row_local * n..(out_row_local + 1) * n];
        let a_row = &a_data[row * k..(row + 1) * k];

        for col_block in (0..n).step_by(BN) {
            let col_end = (col_block + BN).min(n);
            for col in col_block..col_end {
                let b_row = &bt_data[col * k..(col + 1) * k];
                let mut acc = 0.0f32;
                for kk_block in (0..k).step_by(BK) {
                    let kk_end = (kk_block + BK).min(k);
                    acc += dot_unrolled_8(&a_row[kk_block..kk_end], &b_row[kk_block..kk_end]);
                }
                out_row[col] = acc;
            }
        }
    }
}

fn dot_unrolled_8(lhs: &[f32], rhs: &[f32]) -> f32 {
    let len = lhs.len();
    let chunks = len / 8;
    let mut i = 0;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;

    for _ in 0..chunks {
        s0 += lhs[i] * rhs[i];
        s1 += lhs[i + 1] * rhs[i + 1];
        s2 += lhs[i + 2] * rhs[i + 2];
        s3 += lhs[i + 3] * rhs[i + 3];
        s4 += lhs[i + 4] * rhs[i + 4];
        s5 += lhs[i + 5] * rhs[i + 5];
        s6 += lhs[i + 6] * rhs[i + 6];
        s7 += lhs[i + 7] * rhs[i + 7];
        i += 8;
    }

    let mut tail = 0.0f32;
    while i < len {
        tail += lhs[i] * rhs[i];
        i += 1;
    }

    s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + tail
}

impl ComputationGraph {
    pub fn new() -> Self {
        ComputationGraph {
            nodes: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn add_input(&mut self, tensor: Tensor) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let node = ComputeNode {
            id,
            op: Operation::Input,
            inputs: Vec::new(),
            value: tensor,
            gradient: None,
            requires_grad: true,
        };

        self.nodes.insert(id, node);
        id
    }

    pub fn add_operation(&mut self, op: Operation, inputs: Vec<u64>, result: Tensor) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let requires_grad = inputs.iter().any(|&inp_id| {
            self.nodes
                .get(&inp_id)
                .map(|n| n.requires_grad)
                .unwrap_or(false)
        });

        let node = ComputeNode {
            id,
            op,
            inputs,
            value: result,
            gradient: None,
            requires_grad,
        };

        self.nodes.insert(id, node);
        id
    }

    /// Backward pass: compute gradients through the graph
    pub fn backward(&mut self, output_id: u64) {
        // Initialize gradient of output to ones
        if let Some(output_node) = self.nodes.get_mut(&output_id) {
            output_node.gradient = Some(Tensor::ones(output_node.value.shape.clone()));
        }

        // Topological sort (simple DFS-based approach)
        let mut visited = std::collections::HashSet::new();
        let mut order = Vec::new();
        self.topo_sort(output_id, &mut visited, &mut order);
        order.reverse();

        // Backward pass through the graph
        for &node_id in &order {
            let (op, inputs, upstream_grad) = match self.nodes.get(&node_id) {
                Some(node) => {
                    let Some(upstream_grad) = node.gradient.clone() else {
                        continue;
                    };
                    (node.op.clone(), node.inputs.clone(), upstream_grad)
                }
                None => continue,
            };

            match op {
                Operation::Add => {
                    // Gradient flows equally to both inputs
                    if let Some(input_id) = inputs.get(0) {
                        if let Some(input) = self.nodes.get_mut(input_id) {
                            input.gradient = Some(match &input.gradient {
                                Some(g) => g.add(&upstream_grad),
                                None => upstream_grad.clone(),
                            });
                        }
                    }
                    if let Some(input_id) = inputs.get(1) {
                        if let Some(input) = self.nodes.get_mut(input_id) {
                            input.gradient = Some(match &input.gradient {
                                Some(g) => g.add(&upstream_grad),
                                None => upstream_grad.clone(),
                            });
                        }
                    }
                }
                Operation::Mul => {
                    // Gradient: dL/dX = dL/dY * Y
                    if let (Some(&a_id), Some(&b_id)) = (inputs.get(0), inputs.get(1)) {
                        let a_val = &self.nodes.get(&a_id).unwrap().value.clone();
                        let b_val = &self.nodes.get(&b_id).unwrap().value.clone();

                        let grad_a = upstream_grad.mul(b_val);
                        let grad_b = upstream_grad.mul(a_val);

                        if let Some(a) = self.nodes.get_mut(&a_id) {
                            a.gradient = Some(match &a.gradient {
                                Some(g) => g.add(&grad_a),
                                None => grad_a,
                            });
                        }
                        if let Some(b) = self.nodes.get_mut(&b_id) {
                            b.gradient = Some(match &b.gradient {
                                Some(g) => g.add(&grad_b),
                                None => grad_b,
                            });
                        }
                    }
                }
                Operation::MatMul => {
                    if let (Some(&a_id), Some(&b_id)) = (inputs.get(0), inputs.get(1)) {
                        let a_val = &self.nodes.get(&a_id).unwrap().value.clone();
                        let b_val = &self.nodes.get(&b_id).unwrap().value.clone();

                        let grad_a = a_val.matmul_grad_a(&upstream_grad, b_val);
                        let grad_b = a_val.matmul_grad_b(a_val, &upstream_grad);

                        if let Some(a) = self.nodes.get_mut(&a_id) {
                            a.gradient = Some(match &a.gradient {
                                Some(g) => g.add(&grad_a),
                                None => grad_a,
                            });
                        }
                        if let Some(b) = self.nodes.get_mut(&b_id) {
                            b.gradient = Some(match &b.gradient {
                                Some(g) => g.add(&grad_b),
                                None => grad_b,
                            });
                        }
                    }
                }
                Operation::ReLU => {
                    if let Some(&input_id) = inputs.get(0) {
                        let input_val = &self.nodes.get(&input_id).unwrap().value.clone();
                        let grad_input = input_val.relu_grad(&upstream_grad);

                        if let Some(input) = self.nodes.get_mut(&input_id) {
                            input.gradient = Some(match &input.gradient {
                                Some(g) => g.add(&grad_input),
                                None => grad_input,
                            });
                        }
                    }
                }
                _ => {} // Other operations handled similarly
            };
        }
    }

    fn topo_sort(
        &self,
        node_id: u64,
        visited: &mut std::collections::HashSet<u64>,
        order: &mut Vec<u64>,
    ) {
        if visited.contains(&node_id) {
            return;
        }
        visited.insert(node_id);

        if let Some(node) = self.nodes.get(&node_id) {
            for &input_id in &node.inputs {
                self.topo_sort(input_id, visited, order);
            }
        }

        order.push(node_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_linear_is_close_to_fp32() {
        let w = Tensor {
            shape: vec![3, 2],
            data: vec![0.5, -0.2, 1.2, 0.7, -0.4, 0.1],
        };
        let x = Tensor {
            shape: vec![2, 3],
            data: vec![1.0, 0.5, -1.0, 0.2, -0.1, 0.3],
        };
        let b = Tensor {
            shape: vec![2],
            data: vec![0.05, -0.03],
        };

        let q = w.quantize_linear_int8().expect("quantize");
        let y_int8 = x.linear_int8(&q, Some(&b)).expect("int8 linear");
        let y_fp32 = x.matmul(&w).add(&Tensor {
            shape: vec![2, 2],
            data: vec![0.05, -0.03, 0.05, -0.03],
        });

        for (a, b) in y_int8.data.iter().zip(y_fp32.data.iter()) {
            assert!((a - b).abs() < 0.05, "int8 error too high: {a} vs {b}");
        }

        assert!(q.effective_bytes_per_param() <= 1.8);
    }
}

// =========================================================================
// Advanced Optimizers with Learning Rate Scheduling
// =========================================================================

#[derive(Debug, Clone)]
pub struct OptimizerState {
    pub momentum: HashMap<u64, Vec<f32>>, // For momentum-based optimizers
    pub velocity_m: HashMap<u64, Vec<f32>>, // First moment (Adam)
    pub velocity_v: HashMap<u64, Vec<f32>>, // Second moment (Adam)
    pub step_count: u64,
}

impl OptimizerState {
    pub fn new() -> Self {
        OptimizerState {
            momentum: HashMap::new(),
            velocity_m: HashMap::new(),
            velocity_v: HashMap::new(),
            step_count: 0,
        }
    }
}

pub enum Optimizer {
    SGD {
        learning_rate: f32,
        momentum: f32,
    },
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
    AdamW {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },
    RMSprop {
        learning_rate: f32,
        rho: f32,
        epsilon: f32,
    },
}

impl Optimizer {
    pub fn update_weights(
        &self,
        weights: &mut Vec<f32>,
        gradients: &[f32],
        state: &mut OptimizerState,
    ) {
        match self {
            Optimizer::SGD {
                learning_rate,
                momentum,
            } => {
                Self::sgd_step(weights, gradients, *learning_rate, *momentum, state);
            }
            Optimizer::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            } => {
                Self::adam_step(
                    weights,
                    gradients,
                    *learning_rate,
                    *beta1,
                    *beta2,
                    *epsilon,
                    state,
                );
            }
            Optimizer::AdamW {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            } => {
                Self::adamw_step(
                    weights,
                    gradients,
                    *learning_rate,
                    *beta1,
                    *beta2,
                    *epsilon,
                    *weight_decay,
                    state,
                );
            }
            Optimizer::RMSprop {
                learning_rate,
                rho,
                epsilon,
            } => {
                Self::rmsprop_step(weights, gradients, *learning_rate, *rho, *epsilon, state);
            }
        }
    }

    fn sgd_step(
        weights: &mut [f32],
        grads: &[f32],
        lr: f32,
        momentum: f32,
        state: &mut OptimizerState,
    ) {
        let momentum_vec = state
            .momentum
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);

        // Ensure momentum buffer matches weight size
        if momentum_vec.len() != weights.len() {
            momentum_vec.resize(weights.len(), 0.0);
        }

        for i in 0..weights.len() {
            let grad = if i < grads.len() { grads[i] } else { 0.0 };
            momentum_vec[i] = momentum * momentum_vec[i] - lr * grad;
            weights[i] += momentum_vec[i];
        }
    }

    fn adam_step(
        weights: &mut [f32],
        grads: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        state: &mut OptimizerState,
    ) {
        state.step_count += 1;
        let step = state.step_count as f32;
        let bias_correction1 = 1.0 - beta1.powf(step);
        let bias_correction2 = 1.0 - beta2.powf(step);

        let m = state
            .velocity_m
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);
        let v = state
            .velocity_v
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);

        for i in 0..weights.len() {
            if i < grads.len() {
                m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];

                let m_hat = m[i] / bias_correction1;
                let v_hat = v[i] / bias_correction2;

                weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
    }

    fn adamw_step(
        weights: &mut [f32],
        grads: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        state: &mut OptimizerState,
    ) {
        // Same as Adam but with L2 weight decay
        Self::adam_step(weights, grads, lr, beta1, beta2, eps, state);

        // Apply weight decay
        for w in weights {
            *w *= 1.0 - wd * lr;
        }
    }

    fn rmsprop_step(
        weights: &mut [f32],
        grads: &[f32],
        lr: f32,
        rho: f32,
        eps: f32,
        state: &mut OptimizerState,
    ) {
        let v = state
            .velocity_v
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);

        for i in 0..weights.len() {
            if i < grads.len() {
                v[i] = rho * v[i] + (1.0 - rho) * grads[i] * grads[i];
                weights[i] -= lr * grads[i] / (v[i].sqrt() + eps);
            }
        }
    }
}

// Learning rate scheduler
pub struct LearningRateScheduler {
    initial_lr: f32,
    schedule: ScheduleType,
    step: u64,
}

pub enum ScheduleType {
    Constant,
    Linear { final_lr: f32, total_steps: u64 },
    Exponential { decay_rate: f32 },
    StepDecay { step_size: u64, gamma: f32 },
    CosineAnnealing { total_steps: u64 },
}

impl LearningRateScheduler {
    pub fn new(initial_lr: f32, schedule: ScheduleType) -> Self {
        LearningRateScheduler {
            initial_lr,
            schedule,
            step: 0,
        }
    }

    pub fn get_lr(&self) -> f32 {
        match &self.schedule {
            ScheduleType::Constant => self.initial_lr,
            ScheduleType::Linear {
                final_lr,
                total_steps,
            } => {
                let progress = (self.step as f32) / (*total_steps as f32);
                self.initial_lr + (final_lr - self.initial_lr) * progress
            }
            ScheduleType::Exponential { decay_rate } => {
                self.initial_lr * decay_rate.powf(self.step as f32)
            }
            ScheduleType::StepDecay { step_size, gamma } => {
                let decay_steps = self.step / step_size;
                self.initial_lr * gamma.powf(decay_steps as f32)
            }
            ScheduleType::CosineAnnealing { total_steps } => {
                let progress = (self.step as f32) / (*total_steps as f32);
                let pi = std::f32::consts::PI;
                self.initial_lr * 0.5 * (1.0 + (pi * progress).cos())
            }
        }
    }

    pub fn step(&mut self) {
        self.step += 1;
    }
}

// Implemented optimizer suite without external dependencies
// SGD, Adam, AdamW, RMSprop all use simple indexing patterns

// =========================================================================
// Loss Functions Module - Improved numerical stability
// =========================================================================

pub struct LossFunctions;

impl LossFunctions {
    /// Mean Squared Error loss with numerical stability
    pub fn mse(predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.shape, targets.shape, "Shape mismatch in MSE");
        predictions
            .data
            .iter()
            .zip(&targets.data)
            .map(|(p, t)| {
                let diff = p - t;
                diff * diff
            })
            .sum::<f32>()
            / predictions.numel().max(1) as f32
    }

    /// Cross-Entropy loss with numerical stability (log-sum-exp trick)
    pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(
            logits.shape, targets.shape,
            "Shape mismatch in Cross-Entropy"
        );

        // Numerically stable cross-entropy
        let max_logit = logits.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.data.iter().map(|x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        logits
            .data
            .iter()
            .zip(&targets.data)
            .enumerate()
            .map(|(i, (logit, target))| {
                let prob = exp_logits[i] / sum_exp;
                -target * (prob.max(1e-8).ln())
            })
            .sum::<f32>()
            / logits.numel().max(1) as f32
    }

    /// Binary Cross-Entropy with numerical stability
    pub fn binary_cross_entropy(predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.shape, targets.shape, "Shape mismatch in BCE");
        predictions
            .data
            .iter()
            .zip(&targets.data)
            .map(|(p, t)| {
                let p = p.max(1e-8).min(1.0 - 1e-8);
                -t * p.ln() - (1.0 - t) * (1.0 - p).ln()
            })
            .sum::<f32>()
            / predictions.numel().max(1) as f32
    }

    /// Cross-entropy from logits over the last dim.
    /// `targets` must contain class indices with the same prefix shape as `logits`.
    ///
    /// Example:
    /// - logits: `[batch, classes]`
    /// - targets: `[batch]` (each value in `[0, classes)`)
    pub fn cross_entropy_from_logits_last_dim(
        logits: &Tensor,
        targets: &Tensor,
    ) -> Result<f32, String> {
        let classes = *logits.shape.last().ok_or_else(|| {
            "cross_entropy_from_logits_last_dim expects logits rank >= 1".to_string()
        })?;
        if classes == 0 {
            return Err("cross_entropy_from_logits_last_dim classes must be > 0".into());
        }

        let sample_count = logits.numel() / classes;
        if targets.numel() != sample_count {
            return Err(format!(
                "targets numel {} must equal logits sample count {}",
                targets.numel(),
                sample_count
            ));
        }

        let mut loss = 0.0f32;
        for s in 0..sample_count {
            let base = s * classes;
            let row = &logits.data[base..base + classes];
            let max_logit = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let mut sum_exp = 0.0f32;
            for i in 0..classes {
                sum_exp += (row[i] - max_logit).exp();
            }
            let log_denom = max_logit + sum_exp.max(1e-12).ln();

            let target = targets.data[s];
            let class_idx = target as usize;
            if (class_idx as f32 - target).abs() > 1e-6 || class_idx >= classes {
                return Err(format!(
                    "target value {} at index {} is not a valid class index in [0, {})",
                    target, s, classes
                ));
            }
            loss += log_denom - row[class_idx];
        }

        Ok(loss / sample_count.max(1) as f32)
    }

    /// Gradient of cross-entropy with logits over the last dimension.
    /// Returns dL/dlogits with the same shape as `logits`.
    pub fn cross_entropy_from_logits_last_dim_gradient(
        logits: &Tensor,
        targets: &Tensor,
    ) -> Result<Tensor, String> {
        let classes = *logits.shape.last().ok_or_else(|| {
            "cross_entropy_from_logits_last_dim_gradient expects logits rank >= 1".to_string()
        })?;
        if classes == 0 {
            return Err("cross_entropy_from_logits_last_dim_gradient classes must be > 0".into());
        }

        let sample_count = logits.numel() / classes;
        if targets.numel() != sample_count {
            return Err(format!(
                "targets numel {} must equal logits sample count {}",
                targets.numel(),
                sample_count
            ));
        }

        let mut grad = vec![0.0f32; logits.numel()];
        let inv_n = 1.0 / sample_count.max(1) as f32;

        for s in 0..sample_count {
            let base = s * classes;
            let row = &logits.data[base..base + classes];
            let row_max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let mut sum_exp = 0.0f32;
            for i in 0..classes {
                sum_exp += (row[i] - row_max).exp();
            }
            let inv_sum = 1.0 / sum_exp.max(1e-12);

            let target = targets.data[s];
            let class_idx = target as usize;
            if (class_idx as f32 - target).abs() > 1e-6 || class_idx >= classes {
                return Err(format!(
                    "target value {} at index {} is not a valid class index in [0, {})",
                    target, s, classes
                ));
            }

            for i in 0..classes {
                let p = (row[i] - row_max).exp() * inv_sum;
                grad[base + i] = if i == class_idx { (p - 1.0) * inv_n } else { p * inv_n };
            }
        }

        Ok(Tensor {
            shape: logits.shape.clone(),
            data: grad,
        })
    }

    /// Compute MSE gradient
    pub fn mse_gradient(predictions: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(predictions.shape, targets.shape);
        Tensor {
            shape: predictions.shape.clone(),
            data: predictions
                .data
                .iter()
                .zip(&targets.data)
                .map(|(p, t)| 2.0 * (p - t) / predictions.numel().max(1) as f32)
                .collect(),
        }
    }

    /// Compute gradient of predictions for softmax cross-entropy
    pub fn softmax_cross_entropy_gradient(logits: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(logits.shape, targets.shape);

        // Softmax probabilities with stability
        let max_logit = logits.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.data.iter().map(|x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        Tensor {
            shape: logits.shape.clone(),
            data: exp_logits
                .iter()
                .zip(&targets.data)
                .map(|(prob_unnorm, target)| {
                    let prob = prob_unnorm / sum_exp;
                    (prob - target) / logits.numel().max(1) as f32
                })
                .collect(),
        }
    }
}

// =========================================================================
// Regularization Utilities
// =========================================================================

pub struct Regularization;

impl Regularization {
    /// L2 regularization penalty
    pub fn l2_penalty(weights: &Tensor, lambda: f32) -> f32 {
        lambda * weights.data.iter().map(|w| w * w).sum::<f32>()
    }

    /// L1 regularization penalty
    pub fn l1_penalty(weights: &Tensor, lambda: f32) -> f32 {
        lambda * weights.data.iter().map(|w| w.abs()).sum::<f32>()
    }

    /// Compute L2 gradient
    pub fn l2_gradient(weights: &Tensor, lambda: f32) -> Tensor {
        Tensor {
            shape: weights.shape.clone(),
            data: weights.data.iter().map(|w| 2.0 * lambda * w).collect(),
        }
    }

    /// Compute L1 gradient (subgradient)
    pub fn l1_gradient(weights: &Tensor, lambda: f32) -> Tensor {
        Tensor {
            shape: weights.shape.clone(),
            data: weights.data.iter().map(|w| lambda * w.signum()).collect(),
        }
    }
}

#[cfg(test)]
mod tests_ml_ext {
    use super::{LossFunctions, Tensor};

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn softmax_last_dim_is_row_normalized() {
        let logits = Tensor {
            shape: vec![2, 3],
            data: vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
        };
        let probs = logits.softmax_last_dim().expect("softmax_last_dim should succeed");
        assert_eq!(probs.shape, vec![2, 3]);

        let row0: f32 = probs.data[0..3].iter().sum();
        let row1: f32 = probs.data[3..6].iter().sum();
        assert!(approx(row0, 1.0, 1e-5));
        assert!(approx(row1, 1.0, 1e-5));
        assert!(probs.data[2] > probs.data[1] && probs.data[1] > probs.data[0]);
    }

    #[test]
    fn gelu_and_silu_backward_match_numerical_gradient() {
        let x = Tensor {
            shape: vec![3],
            data: vec![-1.25, 0.25, 1.75],
        };
        let up = Tensor {
            shape: vec![3],
            data: vec![1.0, 1.0, 1.0],
        };

        let gelu_grad = x.gelu_backward(&up).expect("gelu_backward");
        let silu_grad = x.silu_backward(&up).expect("silu_backward");

        let h = 1e-3f32;
        for i in 0..x.data.len() {
            let mut plus = x.clone();
            plus.data[i] += h;
            let mut minus = x.clone();
            minus.data[i] -= h;

            let num_gelu = (plus.gelu().data[i] - minus.gelu().data[i]) / (2.0 * h);
            let num_silu = (plus.silu().data[i] - minus.silu().data[i]) / (2.0 * h);

            assert!(approx(gelu_grad.data[i], num_gelu, 2e-3));
            assert!(approx(silu_grad.data[i], num_silu, 2e-3));
        }
    }

    #[test]
    fn cross_entropy_gradient_sums_to_zero_per_sample() {
        let logits = Tensor {
            shape: vec![2, 4],
            data: vec![1.0, 0.0, -1.0, 2.0, 0.5, -0.5, 0.0, 1.0],
        };
        let targets = Tensor {
            shape: vec![2],
            data: vec![3.0, 0.0],
        };
        let grad = LossFunctions::cross_entropy_from_logits_last_dim_gradient(&logits, &targets)
            .expect("gradient should succeed");
        assert_eq!(grad.shape, logits.shape);

        let row0: f32 = grad.data[0..4].iter().sum();
        let row1: f32 = grad.data[4..8].iter().sum();
        assert!(approx(row0, 0.0, 1e-5));
        assert!(approx(row1, 0.0, 1e-5));
    }
}

// Implemented optimizer suite without external dependencies
// SGD, Adam, AdamW, RMSprop all use simple indexing patterns
