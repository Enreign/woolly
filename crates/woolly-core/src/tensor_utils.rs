//! Tensor utility functions for woolly-core
//! 
//! This module provides helper functions for common tensor operations
//! used throughout the transformer model implementation.

use crate::{CoreError, Result};
use crate::model::memory_pool::TensorMemoryPool;
use crate::blas_matmul::{matmul_blas, is_blas_available};
use woolly_tensor::{Shape, ops::matmul::{Gemm, MatMulConfig}};
use std::sync::OnceLock;

/// Check if SIMD is enabled (cached for performance)
fn is_simd_enabled() -> bool {
    static SIMD_ENABLED: OnceLock<bool> = OnceLock::new();
    *SIMD_ENABLED.get_or_init(|| {
        let simd_enabled = std::env::var("WOOLLY_DISABLE_SIMD")
            .map(|v| v != "1" && v.to_lowercase() != "true")
            .unwrap_or(true);
        eprintln!("[WOOLLY INIT] SIMD enabled: {} (WOOLLY_DISABLE_SIMD={})", 
            simd_enabled, 
            std::env::var("WOOLLY_DISABLE_SIMD").unwrap_or("not set".to_string()));
        simd_enabled
    })
}

/// Simple tensor-like structure for now (will be replaced with real tensors later)
#[derive(Debug, Clone)]
pub struct SimpleTensor {
    pub data: Vec<f32>,
    pub shape: Shape,
}

impl SimpleTensor {
    pub fn new(data: Vec<f32>, shape: Shape) -> Result<Self> {
        if data.len() != shape.numel() {
            return Err(CoreError::tensor(
                "TENSOR_DATA_SIZE_MISMATCH",
                format!("Data length {} doesn't match shape elements {}", 
                    data.len(), shape.numel()),
                "SimpleTensor creation",
                "Ensure data length matches shape total elements"
            ));
        }
        Ok(Self { data, shape })
    }
    
    pub fn zeros(shape: Shape) -> Self {
        let data = vec![0.0; shape.numel()];
        Self { data, shape }
    }
    
    pub fn ones(shape: Shape) -> Self {
        let data = vec![1.0; shape.numel()];
        Self { data, shape }
    }
    
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }
    
    pub fn transpose(&self, axes: &[usize]) -> Result<Self> {
        if axes.len() != self.shape.ndim() {
            return Err(CoreError::tensor(
                "TENSOR_INVALID_TRANSPOSE_AXES",
                "Invalid transpose axes",
                "tensor transpose operation",
                "Provide valid axis indices for tensor dimensions"
            ));
        }
        
        // For now, just implement 2D transpose
        if self.shape.ndim() == 2 && axes == &[1, 0] {
            let rows = self.shape.as_slice()[0];
            let cols = self.shape.as_slice()[1];
            let mut transposed_data = vec![0.0; self.data.len()];
            
            for i in 0..rows {
                for j in 0..cols {
                    let old_idx = i * cols + j;
                    let new_idx = j * rows + i;
                    transposed_data[new_idx] = self.data[old_idx];
                }
            }
            
            let new_shape = Shape::matrix(cols, rows);
            Ok(Self { data: transposed_data, shape: new_shape })
        } else {
            Err(CoreError::tensor(
                "TENSOR_UNSUPPORTED_TRANSPOSE",
                "Only 2D transpose currently supported",
                "tensor transpose operation",
                "Use 2D tensors for transpose operations"
            ))
        }
    }
}

/// Create a tensor from a slice of f32 values
pub fn tensor_from_slice(data: &[f32], shape: Shape) -> Result<SimpleTensor> {
    SimpleTensor::new(data.to_vec(), shape)
}

/// Create a zero tensor with the given shape
pub fn zeros_tensor(shape: Shape) -> Result<SimpleTensor> {
    Ok(SimpleTensor::zeros(shape))
}

/// Create a ones tensor with the given shape
pub fn ones_tensor(shape: Shape) -> Result<SimpleTensor> {
    Ok(SimpleTensor::ones(shape))
}

/// Simple non-SIMD matrix multiplication for fallback
fn matmul_simple(a: &SimpleTensor, b: &SimpleTensor) -> Result<SimpleTensor> {
    // Validate shapes for matrix multiplication
    if a.shape.ndim() != 2 || b.shape.ndim() != 2 {
        return Err(CoreError::tensor(
            "TENSOR_MATMUL_NON_2D",
            "Matrix multiplication requires 2D tensors",
            "simple matrix multiplication",
            "Ensure both tensors are 2-dimensional"
        ));
    }
    
    let m = a.shape.as_slice()[0];
    let k = a.shape.as_slice()[1];
    let k2 = b.shape.as_slice()[0];
    let n = b.shape.as_slice()[1];
    
    if k != k2 {
        return Err(CoreError::tensor(
            "TENSOR_MATMUL_DIM_MISMATCH",
            format!("Matrix dimensions don't match for multiplication: {} != {}", k, k2),
            "simple matrix multiplication",
            "Ensure inner dimensions match for matrix multiplication"
        ));
    }
    
    // Simple naive matrix multiplication
    let mut result = vec![0.0f32; m * n];
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a.data[i * k + l] * b.data[l * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    
    tensor_from_slice(&result, Shape::matrix(m, n))
}

/// Perform matrix multiplication using tensor operations with memory pool
pub fn matmul_with_pool(a: &SimpleTensor, b: &SimpleTensor, pool: &mut TensorMemoryPool) -> Result<SimpleTensor> {
    // Validate shapes for matrix multiplication
    if a.shape.ndim() != 2 || b.shape.ndim() != 2 {
        return Err(CoreError::tensor(
            "TENSOR_MATMUL_NON_2D",
            "Matrix multiplication requires 2D tensors",
            "matrix multiplication operation",
            "Ensure both tensors are 2-dimensional"
        ));
    }
    
    let m = a.shape.as_slice()[0];
    let k = a.shape.as_slice()[1];
    let k2 = b.shape.as_slice()[0];
    let n = b.shape.as_slice()[1];
    
    if k != k2 {
        return Err(CoreError::tensor(
            "TENSOR_MATMUL_DIM_MISMATCH",
            format!("Matrix dimensions don't match for multiplication: {} != {}", k, k2),
            "matrix multiplication operation",
            "Ensure inner dimensions match for matrix multiplication"
        ));
    }
    
    // Check cache first
    let cache_key = (m, n);
    if let Some(cached_result) = pool.get_matmul_cache(cache_key) {
        if cached_result.len() == m * n {
            return tensor_from_slice(&cached_result, Shape::matrix(m, n));
        }
    }
    
    // Check if SIMD is disabled
    if !is_simd_enabled() {
        // Use simple non-SIMD implementation
        eprintln!("\n[WOOLLY] SIMD DISABLED: Using simple matrix multiplication (WOOLLY_DISABLE_SIMD=1)");
        eprintln!("[WOOLLY] Matrix dimensions: {}x{} * {}x{} = {}x{}", m, k, k2, n, m, n);
        return matmul_simple(a, b);
    }
    
    // Use buffer from pool
    let mut result_data = pool.get_buffer(m * n);
    
    // Use the optimized matmul from woolly-tensor
    Gemm::compute(
        &a.data,
        &b.data,
        &mut result_data,
        &a.shape,
        &b.shape,
        &MatMulConfig::default(),
    ).map_err(|e| CoreError::tensor(
        "TENSOR_MATMUL_OPERATION_FAILED",
        format!("Matrix multiplication failed: {}", e),
        "matrix multiplication operation",
        "Check tensor dimensions and backend implementation"
    ))?;
    
    // Cache the result for small matrices
    if m * n < 16384 { // Cache results < 64KB
        pool.cache_matmul_result(cache_key, result_data.clone());
    }
    
    // Create result tensor
    let result_shape = Shape::matrix(m, n);
    let result = tensor_from_slice(&result_data, result_shape)?;
    
    // Return buffer to pool
    pool.return_buffer(result_data);
    
    Ok(result)
}

/// Perform matrix multiplication using tensor operations (fallback)
pub fn matmul(a: &SimpleTensor, b: &SimpleTensor) -> Result<SimpleTensor> {
    // Validate shapes for matrix multiplication
    if a.shape.ndim() != 2 || b.shape.ndim() != 2 {
        return Err(CoreError::tensor(
            "TENSOR_MATMUL_NON_2D",
            "Matrix multiplication requires 2D tensors",
            "matrix multiplication operation",
            "Ensure both tensors are 2-dimensional"
        ));
    }
    
    let m = a.shape.as_slice()[0];
    let k = a.shape.as_slice()[1];
    let k2 = b.shape.as_slice()[0];
    let n = b.shape.as_slice()[1];
    
    if k != k2 {
        return Err(CoreError::tensor(
            "TENSOR_MATMUL_DIM_MISMATCH",
            format!("Matrix dimensions don't match for multiplication: {} != {}", k, k2),
            "matrix multiplication operation",
            "Ensure inner dimensions match for matrix multiplication"
        ));
    }
    
    // Try BLAS first for maximum performance on macOS
    if is_blas_available() {
        if let Some(result) = matmul_blas(a, b) {
            eprintln!("🚀 Using Accelerate BLAS for matrix multiplication ({}x{} @ {}x{})", m, k, k2, n);
            return Ok(result);
        } else {
            eprintln!("⚠️ BLAS available but failed for shape ({}x{} @ {}x{})", m, k, k2, n);
        }
    } else {
        eprintln!("⚠️ BLAS not available");
    }
    
    // Check if SIMD is disabled
    if !is_simd_enabled() {
        // Use simple non-SIMD implementation
        return matmul_simple(a, b);
    }
    
    // Compute matrix multiplication (without pool optimization)
    let mut result_data = vec![0.0f32; m * n];
    
    // Use the optimized matmul from woolly-tensor
    Gemm::compute(
        &a.data,
        &b.data,
        &mut result_data,
        &a.shape,
        &b.shape,
        &MatMulConfig::default(),
    ).map_err(|e| CoreError::tensor(
        "TENSOR_MATMUL_OPERATION_FAILED",
        format!("Matrix multiplication failed: {}", e),
        "matrix multiplication operation",
        "Check tensor dimensions and backend implementation"
    ))?;
    
    // Create result tensor
    let result_shape = Shape::matrix(m, n);
    tensor_from_slice(&result_data, result_shape)
}

/// Apply softmax along the last dimension
pub fn softmax(input: &SimpleTensor) -> Result<SimpleTensor> {
    // For now, handle 2D case (batch_size, vocab_size)
    if input.shape.ndim() != 2 {
        return Err(CoreError::tensor(
            "TENSOR_SOFTMAX_NON_2D",
            "Softmax currently only supports 2D tensors",
            "softmax operation",
            "Use 2D tensors for softmax operations"
        ));
    }
    
    let batch_size = input.shape.as_slice()[0];
    let vocab_size = input.shape.as_slice()[1];
    let mut result = vec![0.0f32; input.data.len()];
    
    for b in 0..batch_size {
        let start_idx = b * vocab_size;
        let end_idx = start_idx + vocab_size;
        let row = &input.data[start_idx..end_idx];
        
        // Find max for numerical stability
        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exp and sum
        let mut sum = 0.0f32;
        for (i, &val) in row.iter().enumerate() {
            let exp_val = (val - max_val).exp();
            result[start_idx + i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for i in start_idx..end_idx {
            result[i] /= sum;
        }
    }
    
    tensor_from_slice(&result, input.shape.clone())
}

/// Apply layer normalization with memory pool
pub fn layer_norm_with_pool(
    input: &SimpleTensor,
    weight: &SimpleTensor,
    bias: Option<&SimpleTensor>,
    eps: f32,
    pool: &mut TensorMemoryPool,
) -> Result<SimpleTensor> {
    let weight_data = &weight.data;
    let bias_data = bias.map(|b| &b.data);
    
    let shape = input.shape.clone();
    let hidden_size = shape.as_slice()[shape.ndim() - 1];
    let total_elements = input.data.len();
    let num_tokens = total_elements / hidden_size;
    
    // Use buffer from pool
    let mut result = pool.get_buffer(total_elements);
    
    for t in 0..num_tokens {
        let start_idx = t * hidden_size;
        let end_idx = start_idx + hidden_size;
        let token_data = &input.data[start_idx..end_idx];
        
        // Compute mean
        let mean = token_data.iter().sum::<f32>() / hidden_size as f32;
        
        // Compute variance
        let variance = token_data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / hidden_size as f32;
        
        let std_dev = (variance + eps).sqrt();
        
        // Normalize and scale
        for (i, &val) in token_data.iter().enumerate() {
            let normalized = (val - mean) / std_dev;
            let scaled = normalized * weight_data[i];
            let final_val = if let Some(ref bias) = bias_data {
                scaled + bias[i]
            } else {
                scaled
            };
            result[start_idx + i] = final_val;
        }
    }
    
    let tensor_result = tensor_from_slice(&result, shape)?;
    
    // Return buffer to pool
    pool.return_buffer(result);
    
    Ok(tensor_result)
}

/// Apply layer normalization (fallback)
pub fn layer_norm(
    input: &SimpleTensor,
    weight: &SimpleTensor,
    bias: Option<&SimpleTensor>,
    eps: f32,
) -> Result<SimpleTensor> {
    let weight_data = &weight.data;
    let bias_data = bias.map(|b| &b.data);
    
    let shape = input.shape.clone();
    let hidden_size = shape.as_slice()[shape.ndim() - 1];
    let total_elements = input.data.len();
    let num_tokens = total_elements / hidden_size;
    
    let mut result = vec![0.0f32; total_elements];
    
    for t in 0..num_tokens {
        let start_idx = t * hidden_size;
        let end_idx = start_idx + hidden_size;
        let token_data = &input.data[start_idx..end_idx];
        
        // Compute mean
        let mean = token_data.iter().sum::<f32>() / hidden_size as f32;
        
        // Compute variance
        let variance = token_data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / hidden_size as f32;
        
        let std_dev = (variance + eps).sqrt();
        
        // Normalize and scale
        for (i, &val) in token_data.iter().enumerate() {
            let normalized = (val - mean) / std_dev;
            let scaled = normalized * weight_data[i];
            let final_val = if let Some(ref bias) = bias_data {
                scaled + bias[i]
            } else {
                scaled
            };
            result[start_idx + i] = final_val;
        }
    }
    
    tensor_from_slice(&result, shape)
}

/// Apply RMS normalization
pub fn rms_norm(
    input: &SimpleTensor,
    weight: &SimpleTensor,
    eps: f32,
) -> Result<SimpleTensor> {
    let weight_data = &weight.data;
    
    let shape = input.shape.clone();
    let hidden_size = shape.as_slice()[shape.ndim() - 1];
    let total_elements = input.data.len();
    let num_tokens = total_elements / hidden_size;
    
    let mut result = vec![0.0f32; total_elements];
    
    for t in 0..num_tokens {
        let start_idx = t * hidden_size;
        let end_idx = start_idx + hidden_size;
        let token_data = &input.data[start_idx..end_idx];
        
        // Compute RMS
        let sum_squares = token_data.iter()
            .map(|&x| x * x)
            .sum::<f32>();
        let rms = (sum_squares / hidden_size as f32 + eps).sqrt();
        
        // Normalize and scale
        for (i, &val) in token_data.iter().enumerate() {
            let normalized = val / rms;
            let scaled = normalized * weight_data[i];
            result[start_idx + i] = scaled;
        }
    }
    
    tensor_from_slice(&result, shape)
}

/// Element-wise add two tensors
pub fn add_tensors(a: &SimpleTensor, b: &SimpleTensor) -> Result<SimpleTensor> {
    if a.shape.as_slice() != b.shape.as_slice() {
        return Err(CoreError::tensor(
            "TENSOR_ADD_SHAPE_MISMATCH",
            "Tensor shapes must match for addition",
            "tensor addition operation",
            "Ensure tensors have identical shapes for element-wise addition"
        ));
    }
    
    let result: Vec<f32> = a.data.iter()
        .zip(b.data.iter())
        .map(|(x, y)| x + y)
        .collect();
    
    tensor_from_slice(&result, a.shape.clone())
}

/// Apply SiLU activation function (x * sigmoid(x))
pub fn silu(input: &SimpleTensor) -> Result<SimpleTensor> {
    let result: Vec<f32> = input.data.iter()
        .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
        .collect();
    
    tensor_from_slice(&result, input.shape.clone())
}

/// Apply SwiGLU activation function (SiLU(gate) * up)
/// Takes gate and up projections and returns element-wise product
pub fn swiglu(gate: &SimpleTensor, up: &SimpleTensor) -> Result<SimpleTensor> {
    if gate.shape.as_slice() != up.shape.as_slice() {
        return Err(CoreError::tensor(
            "SWIGLU_SHAPE_MISMATCH",
            "Gate and up tensors must have identical shapes",
            "SwiGLU activation",
            "Ensure gate and up projections have the same dimensions"
        ));
    }
    
    let result: Vec<f32> = gate.data.iter()
        .zip(up.data.iter())
        .map(|(&g, &u)| {
            // SiLU(gate) * up
            let silu_gate = g * (1.0 / (1.0 + (-g).exp()));
            silu_gate * u
        })
        .collect();
    
    tensor_from_slice(&result, gate.shape.clone())
}

/// Apply GELU activation function
pub fn gelu(input: &SimpleTensor) -> Result<SimpleTensor> {
    let sqrt_2_over_pi = (std::f32::consts::PI / 2.0).sqrt();
    let result: Vec<f32> = input.data.iter()
        .map(|&x| {
            let tanh_input = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
            0.5 * x * (1.0 + tanh_input.tanh())
        })
        .collect();
    
    tensor_from_slice(&result, input.shape.clone())
}

/// Convert token embeddings to tensor
pub fn embedding_lookup(
    input_ids: &[u32],
    embedding_weights: &SimpleTensor,
) -> Result<SimpleTensor> {
    let weights_shape = &embedding_weights.shape;
    
    if weights_shape.ndim() != 2 {
        return Err(CoreError::tensor(
            "TENSOR_EMBEDDING_NON_2D",
            "Embedding weights must be 2D",
            "embedding lookup operation",
            "Ensure embedding weights tensor is 2-dimensional"
        ));
    }
    
    let vocab_size = weights_shape.as_slice()[0];
    let hidden_size = weights_shape.as_slice()[1];
    let seq_len = input_ids.len();
    
    let mut result = vec![0.0f32; seq_len * hidden_size];
    
    for (i, &token_id) in input_ids.iter().enumerate() {
        if token_id as usize >= vocab_size {
            return Err(CoreError::tensor(
                "TENSOR_EMBEDDING_TOKEN_OUT_OF_RANGE",
                format!("Token ID {} exceeds vocabulary size {}", token_id, vocab_size),
                "embedding lookup operation",
                "Ensure token IDs are within vocabulary range"
            ));
        }
        
        let token_idx = token_id as usize;
        let start_idx = token_idx * hidden_size;
        let end_idx = start_idx + hidden_size;
        let token_embedding = &embedding_weights.data[start_idx..end_idx];
        
        let result_start = i * hidden_size;
        let result_end = result_start + hidden_size;
        result[result_start..result_end].copy_from_slice(token_embedding);
    }
    
    let result_shape = Shape::matrix(seq_len, hidden_size);
    tensor_from_slice(&result, result_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = Shape::matrix(2, 2);
        let tensor = tensor_from_slice(&data, shape).unwrap();
        
        assert_eq!(tensor.shape().as_slice(), &[2, 2]);
        assert_eq!(tensor.data.len(), 4);
    }
    
    #[test]
    fn test_matmul() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        
        let a = tensor_from_slice(&a_data, Shape::matrix(2, 2)).unwrap();
        let b = tensor_from_slice(&b_data, Shape::matrix(2, 2)).unwrap();
        
        let result = matmul(&a, &b).unwrap();
        let result_data = result.to_vec();
        
        // Expected: [[19, 22], [43, 50]]
        assert_eq!(result_data, vec![19.0, 22.0, 43.0, 50.0]);
    }
    
    #[test]
    fn test_softmax() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = tensor_from_slice(&data, Shape::matrix(2, 3)).unwrap();
        
        let result = softmax(&tensor).unwrap();
        let result_data = result.to_vec();
        
        // Check that each row sums to 1
        let row1_sum = result_data[0] + result_data[1] + result_data[2];
        let row2_sum = result_data[3] + result_data[4] + result_data[5];
        
        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((row2_sum - 1.0).abs() < 1e-6);
    }
}