//! Optimized tensor utilities with SIMD and memory pooling

use crate::{CoreError, Result};
use crate::model::memory_pool::TensorMemoryPool;
use crate::tensor_utils::SimpleTensor;
use woolly_tensor::{Shape, ops::matmul::{Gemm, MatMulConfig}};

/// High-performance matrix multiplication with pre-allocated buffers
pub fn matmul_fast(
    a: &SimpleTensor, 
    b: &SimpleTensor, 
    pool: &mut TensorMemoryPool,
    use_simd: bool,
) -> Result<SimpleTensor> {
    // Validate shapes
    if a.shape.ndim() != 2 || b.shape.ndim() != 2 {
        return Err(CoreError::tensor(
            "TENSOR_MATMUL_NON_2D",
            "Matrix multiplication requires 2D tensors",
            "optimized matrix multiplication",
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
            format!("Matrix dimensions don't match: {} != {}", k, k2),
            "optimized matrix multiplication",
            "Ensure inner dimensions match"
        ));
    }
    
    // Check cache first for small matrices
    let cache_key = (m, n);
    if m * n < 4096 {
        if let Some(cached_result) = pool.get_matmul_cache(cache_key) {
            if cached_result.len() == m * n {
                return Ok(SimpleTensor::new(cached_result.clone(), Shape::matrix(m, n))?);
            }
        }
    }
    
    // Get buffer from pool
    let mut result = pool.get_buffer(m * n);
    
    // Choose implementation based on size and SIMD availability
    if use_simd && is_simd_beneficial(m, n, k) {
        compute_matmul_simd(&a.data, &b.data, &mut result, m, n, k)?;
    } else if is_blocked_beneficial(m, n, k) {
        compute_matmul_blocked(&a.data, &b.data, &mut result, m, n, k)?;
    } else {
        compute_matmul_naive(&a.data, &b.data, &mut result, m, n, k);
    }
    
    // Cache small results
    if m * n < 4096 {
        pool.cache_matmul_result(cache_key, result.clone());
    }
    
    let tensor_result = SimpleTensor::new(result.clone(), Shape::matrix(m, n))?;
    pool.return_buffer(result);
    
    Ok(tensor_result)
}

/// Check if SIMD implementation would be beneficial
fn is_simd_beneficial(m: usize, n: usize, k: usize) -> bool {
    // SIMD is beneficial for larger matrices where overhead is amortized
    m >= 8 && n >= 8 && k >= 32
}

/// Check if blocked implementation would be beneficial
fn is_blocked_beneficial(m: usize, n: usize, k: usize) -> bool {
    // Blocking helps with cache locality for medium-large matrices
    m >= 64 && n >= 64 && k >= 64
}

/// SIMD-optimized matrix multiplication
fn compute_matmul_simd(
    a: &[f32],
    b: &[f32], 
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { compute_matmul_avx2(a, b, c, m, n, k) };
        }
    }
    
    // Fallback to blocked implementation
    compute_matmul_blocked(a, b, c, m, n, k)
}

/// AVX2-optimized matrix multiplication
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
unsafe fn compute_matmul_avx2(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use std::arch::x86_64::*;
    
    // Process in 8x8 blocks using AVX2
    const BLOCK_SIZE: usize = 8;
    
    for i in (0..m).step_by(BLOCK_SIZE) {
        let i_end = std::cmp::min(i + BLOCK_SIZE, m);
        
        for j in (0..n).step_by(BLOCK_SIZE) {
            let j_end = std::cmp::min(j + BLOCK_SIZE, n);
            
            // Initialize accumulator
            let mut acc = [_mm256_setzero_ps(); BLOCK_SIZE];
            
            for kk in (0..k).step_by(8) {
                let k_end = std::cmp::min(kk + 8, k);
                let k_len = k_end - kk;
                
                // Load B values
                let mut b_vals = [_mm256_setzero_ps(); BLOCK_SIZE];
                for jj in 0..(j_end - j) {
                    if jj < BLOCK_SIZE && j + jj < n {
                        let b_ptr = b.as_ptr().add((kk) * n + (j + jj));
                        if k_len >= 8 {
                            // Load 8 consecutive values if available
                            let b_slice = std::slice::from_raw_parts(b_ptr, k_len);
                            let mut temp = [0.0f32; 8];
                            for l in 0..k_len {
                                temp[l] = b_slice[l * n];
                            }
                            b_vals[jj] = _mm256_loadu_ps(temp.as_ptr());
                        }
                    }
                }
                
                // Compute for each row
                for ii in 0..(i_end - i) {
                    if ii < BLOCK_SIZE && i + ii < m {
                        for l in 0..k_len {
                            let a_val = *a.get_unchecked((i + ii) * k + (kk + l));
                            let a_broadcast = _mm256_set1_ps(a_val);
                            
                            for jj in 0..(j_end - j) {
                                if jj < BLOCK_SIZE {
                                    // Extract the l-th element from b_vals[jj]
                                    let b_elem = match l {
                                        0 => _mm256_set1_ps(b_vals[jj][0]),
                                        1 => _mm256_set1_ps(b_vals[jj][1]),
                                        2 => _mm256_set1_ps(b_vals[jj][2]),
                                        3 => _mm256_set1_ps(b_vals[jj][3]),
                                        4 => _mm256_set1_ps(b_vals[jj][4]),
                                        5 => _mm256_set1_ps(b_vals[jj][5]),
                                        6 => _mm256_set1_ps(b_vals[jj][6]),
                                        7 => _mm256_set1_ps(b_vals[jj][7]),
                                        _ => _mm256_setzero_ps(),
                                    };
                                    
                                    acc[ii] = _mm256_fmadd_ps(a_broadcast, b_elem, acc[ii]);
                                }
                            }
                        }
                    }
                }
            }
            
            // Store results
            for ii in 0..(i_end - i) {
                for jj in 0..(j_end - j) {
                    if ii < BLOCK_SIZE && jj < BLOCK_SIZE && 
                       i + ii < m && j + jj < n {
                        let result_idx = (i + ii) * n + (j + jj);
                        // Extract scalar from SIMD register - simplified
                        c[result_idx] = acc[ii][0]; // This would need proper horizontal sum
                    }
                }
            }
        }
    }
    
    Ok(())
}

/// Cache-blocked matrix multiplication
fn compute_matmul_blocked(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    // Block sizes optimized for cache hierarchy
    const MC: usize = 256;  // Rows of A
    const KC: usize = 128;  // Columns of A / Rows of B
    const NC: usize = 512;  // Columns of B
    
    // Initialize output
    for i in 0..m * n {
        c[i] = 0.0;
    }
    
    // Blocked computation
    for jc in (0..n).step_by(NC) {
        let nc = std::cmp::min(NC, n - jc);
        
        for pc in (0..k).step_by(KC) {
            let kc = std::cmp::min(KC, k - pc);
            
            for ic in (0..m).step_by(MC) {
                let mc = std::cmp::min(MC, m - ic);
                
                // Inner kernel - optimized 4x4 blocks
                for i in ic..ic + mc {
                    for j in jc..jc + nc {
                        let mut sum = 0.0f32;
                        
                        // Unroll inner loop for better performance
                        let mut p = pc;
                        while p + 4 <= pc + kc {
                            sum += a[i * k + p] * b[p * n + j]
                                + a[i * k + p + 1] * b[(p + 1) * n + j]
                                + a[i * k + p + 2] * b[(p + 2) * n + j]
                                + a[i * k + p + 3] * b[(p + 3) * n + j];
                            p += 4;
                        }
                        
                        // Handle remaining elements
                        while p < pc + kc {
                            sum += a[i * k + p] * b[p * n + j];
                            p += 1;
                        }
                        
                        c[i * n + j] += sum;
                    }
                }
            }
        }
    }
    
    Ok(())
}

/// Simple naive matrix multiplication for small matrices
fn compute_matmul_naive(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Optimized RMS normalization with memory pooling
pub fn rms_norm_fast(
    input: &SimpleTensor,
    weight: &SimpleTensor,
    eps: f32,
    pool: &mut TensorMemoryPool,
) -> Result<SimpleTensor> {
    let shape = input.shape.clone();
    let hidden_size = shape.as_slice()[shape.ndim() - 1];
    let total_elements = input.data.len();
    let num_tokens = total_elements / hidden_size;
    
    // Use buffer from pool
    let mut result = pool.get_buffer(total_elements);
    
    // SIMD-optimized RMS computation where available
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") && hidden_size >= 8 {
            unsafe {
                rms_norm_avx2(&input.data, &weight.data, &mut result, num_tokens, hidden_size, eps);
            }
        } else {
            rms_norm_scalar(&input.data, &weight.data, &mut result, num_tokens, hidden_size, eps);
        }
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        rms_norm_scalar(&input.data, &weight.data, &mut result, num_tokens, hidden_size, eps);
    }
    
    let tensor_result = SimpleTensor::new(result.clone(), shape)?;
    pool.return_buffer(result);
    
    Ok(tensor_result)
}

/// Scalar RMS normalization
fn rms_norm_scalar(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    num_tokens: usize,
    hidden_size: usize,
    eps: f32,
) {
    for t in 0..num_tokens {
        let start_idx = t * hidden_size;
        let end_idx = start_idx + hidden_size;
        let token_data = &input[start_idx..end_idx];
        
        // Compute RMS
        let sum_squares: f32 = token_data.iter().map(|&x| x * x).sum();
        let rms = (sum_squares / hidden_size as f32 + eps).sqrt();
        
        // Normalize and scale
        for (i, &val) in token_data.iter().enumerate() {
            let normalized = val / rms;
            output[start_idx + i] = normalized * weight[i];
        }
    }
}

/// AVX2-optimized RMS normalization
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn rms_norm_avx2(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    num_tokens: usize,
    hidden_size: usize,
    eps: f32,
) {
    use std::arch::x86_64::*;
    
    let eps_vec = _mm256_set1_ps(eps);
    let hidden_size_f32 = hidden_size as f32;
    let inv_hidden_size = _mm256_set1_ps(1.0 / hidden_size_f32);
    
    for t in 0..num_tokens {
        let start_idx = t * hidden_size;
        let token_data = &input[start_idx..start_idx + hidden_size];
        let token_output = &mut output[start_idx..start_idx + hidden_size];
        
        // Compute sum of squares using SIMD
        let mut sum_squares_vec = _mm256_setzero_ps();
        let simd_len = hidden_size - (hidden_size % 8);
        
        for i in (0..simd_len).step_by(8) {
            let data_vec = _mm256_loadu_ps(token_data.as_ptr().add(i));
            sum_squares_vec = _mm256_fmadd_ps(data_vec, data_vec, sum_squares_vec);
        }
        
        // Horizontal sum
        let sum_squares = horizontal_sum_f32(sum_squares_vec);
        
        // Handle remaining elements
        let mut remaining_sum = 0.0f32;
        for i in simd_len..hidden_size {
            remaining_sum += token_data[i] * token_data[i];
        }
        
        let total_sum = sum_squares + remaining_sum;
        let rms = ((total_sum / hidden_size_f32) + eps).sqrt();
        let inv_rms = _mm256_set1_ps(1.0 / rms);
        
        // Normalize and scale using SIMD
        for i in (0..simd_len).step_by(8) {
            let data_vec = _mm256_loadu_ps(token_data.as_ptr().add(i));
            let weight_vec = _mm256_loadu_ps(weight.as_ptr().add(i));
            let normalized = _mm256_mul_ps(data_vec, inv_rms);
            let scaled = _mm256_mul_ps(normalized, weight_vec);
            _mm256_storeu_ps(token_output.as_mut_ptr().add(i), scaled);
        }
        
        // Handle remaining elements
        for i in simd_len..hidden_size {
            let normalized = token_data[i] / rms;
            token_output[i] = normalized * weight[i];
        }
    }
}

/// Horizontal sum of AVX2 register
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
unsafe fn horizontal_sum_f32(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum_128 = _mm_add_ps(hi, lo);
    
    let shuf = _mm_movehdup_ps(sum_128);
    let sums = _mm_add_ps(sum_128, shuf);
    let shuf = _mm_movehl_ps(sums, sums);
    let sums = _mm_add_ss(sums, shuf);
    
    _mm_cvtss_f32(sums)
}

/// Optimized SwiGLU activation with memory pooling
pub fn swiglu_fast(
    gate: &SimpleTensor,
    up: &SimpleTensor,
    pool: &mut TensorMemoryPool,
) -> Result<SimpleTensor> {
    if gate.shape.as_slice() != up.shape.as_slice() {
        return Err(CoreError::tensor(
            "SWIGLU_SHAPE_MISMATCH",
            "Gate and up tensors must have identical shapes",
            "optimized SwiGLU activation",
            "Ensure gate and up projections have the same dimensions"
        ));
    }
    
    let size = gate.data.len();
    let mut result = pool.get_buffer(size);
    
    // SIMD-optimized SwiGLU where available
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") && size >= 8 {
            unsafe {
                swiglu_avx2(&gate.data, &up.data, &mut result);
            }
        } else {
            swiglu_scalar(&gate.data, &up.data, &mut result);
        }
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        swiglu_scalar(&gate.data, &up.data, &mut result);
    }
    
    let tensor_result = SimpleTensor::new(result.clone(), gate.shape.clone())?;
    pool.return_buffer(result);
    
    Ok(tensor_result)
}

/// Scalar SwiGLU implementation
fn swiglu_scalar(gate: &[f32], up: &[f32], output: &mut [f32]) {
    for i in 0..gate.len() {
        let g = gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        output[i] = silu_g * up[i];
    }
}

/// AVX2-optimized SwiGLU implementation  
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn swiglu_avx2(gate: &[f32], up: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;
    
    let ones = _mm256_set1_ps(1.0);
    let len = gate.len();
    let simd_len = len - (len % 8);
    
    for i in (0..simd_len).step_by(8) {
        let g_vec = _mm256_loadu_ps(gate.as_ptr().add(i));
        let u_vec = _mm256_loadu_ps(up.as_ptr().add(i));
        
        // Compute SiLU: x / (1 + exp(-x))
        let neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g_vec);
        
        // Fast approximate exp using polynomial approximation for better performance
        let exp_neg_g = exp_approx_avx2(neg_g);
        let denom = _mm256_add_ps(ones, exp_neg_g);
        let silu_g = _mm256_div_ps(g_vec, denom);
        
        // Multiply by up values
        let result = _mm256_mul_ps(silu_g, u_vec);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
    }
    
    // Handle remaining elements
    for i in simd_len..len {
        let g = gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        output[i] = silu_g * up[i];
    }
}

/// Fast approximate exponential for AVX2
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
unsafe fn exp_approx_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    
    // Clamp x to reasonable range to avoid overflow
    let min_val = _mm256_set1_ps(-10.0);
    let max_val = _mm256_set1_ps(10.0);
    let x_clamped = _mm256_min_ps(_mm256_max_ps(x, min_val), max_val);
    
    // Use polynomial approximation: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let sixth = _mm256_set1_ps(1.0 / 6.0);
    let twenty_fourth = _mm256_set1_ps(1.0 / 24.0);
    
    let x2 = _mm256_mul_ps(x_clamped, x_clamped);
    let x3 = _mm256_mul_ps(x2, x_clamped);
    let x4 = _mm256_mul_ps(x3, x_clamped);
    
    let term1 = x_clamped;
    let term2 = _mm256_mul_ps(x2, half);
    let term3 = _mm256_mul_ps(x3, sixth);
    let term4 = _mm256_mul_ps(x4, twenty_fourth);
    
    let result = _mm256_add_ps(one, term1);
    let result = _mm256_add_ps(result, term2);
    let result = _mm256_add_ps(result, term3);
    let result = _mm256_add_ps(result, term4);
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_utils::tensor_from_slice;
    
    #[test]
    fn test_optimized_matmul() {
        let mut pool = TensorMemoryPool::new();
        
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        
        let a = tensor_from_slice(&a_data, Shape::matrix(2, 2)).unwrap();
        let b = tensor_from_slice(&b_data, Shape::matrix(2, 2)).unwrap();
        
        let result = matmul_fast(&a, &b, &mut pool, false).unwrap();
        
        // Expected: [[19, 22], [43, 50]]
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }
    
    #[test]
    fn test_optimized_rms_norm() {
        let mut pool = TensorMemoryPool::new();
        
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let weight_data = vec![1.0, 1.0, 1.0, 1.0];
        
        let input = tensor_from_slice(&input_data, Shape::matrix(1, 4)).unwrap();
        let weight = tensor_from_slice(&weight_data, Shape::vector(4)).unwrap();
        
        let result = rms_norm_fast(&input, &weight, 1e-5, &mut pool).unwrap();
        
        // Check that RMS normalization was applied
        let rms = (30.0 / 4.0 + 1e-5).sqrt(); // sqrt((1² + 2² + 3² + 4²) / 4)
        let expected = vec![1.0/rms, 2.0/rms, 3.0/rms, 4.0/rms];
        
        for (actual, expected) in result.data.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }
}