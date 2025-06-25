//! Tensor operations module

pub mod unary;
pub mod binary;
pub mod reduce;
pub mod matmul;
pub mod simd;

// Re-export commonly used operations
pub use unary::*;
pub use binary::*;
pub use reduce::*;
pub use matmul::*;
pub use simd::*;

use crate::Result;
use crate::backend::TensorBackend;

/// Neural network operations optimized for LLM inference
pub mod neural {
    use crate::backend::Result;
    
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    use std::arch::x86_64::*;
    
    /// Optimized softmax implementation
    pub struct Softmax;
    
    impl Softmax {
        /// Compute softmax along the last dimension with SIMD optimization
        pub fn apply_f32(input: &[f32], output: &mut [f32], batch_size: usize, dim: usize) -> Result<()> {
            assert_eq!(input.len(), batch_size * dim);
            assert_eq!(output.len(), input.len());
            
            for batch in 0..batch_size {
                let start_idx = batch * dim;
                let input_slice = &input[start_idx..start_idx + dim];
                let output_slice = &mut output[start_idx..start_idx + dim];
                
                Self::softmax_1d(input_slice, output_slice)?;
            }
            
            Ok(())
        }
        
        /// Optimized 1D softmax with numerical stability
        fn softmax_1d(input: &[f32], output: &mut [f32]) -> Result<()> {
            let _len = input.len();
            
            // Find maximum for numerical stability
            let max_val = Self::find_max_simd(input);
            
            // Compute exp(x - max) and sum
            let sum;
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                if is_x86_feature_detected!("avx2") && _len >= 8 {
                    sum = unsafe { Self::exp_and_sum_avx2(input, output, max_val) };
                } else {
                    sum = Self::exp_and_sum_scalar(input, output, max_val);
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
            {
                sum = Self::exp_and_sum_scalar(input, output, max_val);
            }
            
            // Normalize by sum
            let inv_sum = 1.0 / sum;
            
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                if is_x86_feature_detected!("avx2") && _len >= 8 {
                    unsafe { Self::normalize_avx2(output, inv_sum) };
                } else {
                    Self::normalize_scalar(output, inv_sum);
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
            {
                Self::normalize_scalar(output, inv_sum);
            }
            
            Ok(())
        }
        
        /// Find maximum value using SIMD
        fn find_max_simd(input: &[f32]) -> f32 {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                if is_x86_feature_detected!("avx2") && input.len() >= 8 {
                    return unsafe { Self::find_max_avx2(input) };
                }
            }
            
            input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        }
        
        /// AVX2 maximum finding
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "avx2")]
        unsafe fn find_max_avx2(input: &[f32]) -> f32 {
            let len = input.len();
            let simd_len = len - (len % 8);
            let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
            
            for i in (0..simd_len).step_by(8) {
                let vals = _mm256_loadu_ps(input.as_ptr().add(i));
                max_vec = _mm256_max_ps(max_vec, vals);
            }
            
            // Horizontal max
            let hi = _mm256_extractf128_ps(max_vec, 1);
            let lo = _mm256_castps256_ps128(max_vec);
            let max_128 = _mm_max_ps(hi, lo);
            
            let shuf = _mm_movehdup_ps(max_128);
            let maxs = _mm_max_ps(max_128, shuf);
            let shuf = _mm_movehl_ps(maxs, maxs);
            let maxs = _mm_max_ss(maxs, shuf);
            
            let mut result = _mm_cvtss_f32(maxs);
            
            // Handle remaining elements
            for i in simd_len..len {
                result = result.max(input[i]);
            }
            
            result
        }
        
        /// AVX2 exp and sum computation
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "avx2")]
        unsafe fn exp_and_sum_avx2(input: &[f32], output: &mut [f32], max_val: f32) -> f32 {
            let len = input.len();
            let simd_len = len - (len % 8);
            let max_vec = _mm256_set1_ps(max_val);
            let mut sum_vec = _mm256_setzero_ps();
            
            for i in (0..simd_len).step_by(8) {
                let vals = _mm256_loadu_ps(input.as_ptr().add(i));
                let shifted = _mm256_sub_ps(vals, max_vec);
                let exp_vals = Self::exp_approx_avx2(shifted);
                
                _mm256_storeu_ps(output.as_mut_ptr().add(i), exp_vals);
                sum_vec = _mm256_add_ps(sum_vec, exp_vals);
            }
            
            // Horizontal sum
            let hi = _mm256_extractf128_ps(sum_vec, 1);
            let lo = _mm256_castps256_ps128(sum_vec);
            let sum_128 = _mm_add_ps(hi, lo);
            
            let shuf = _mm_movehdup_ps(sum_128);
            let sums = _mm_add_ps(sum_128, shuf);
            let shuf = _mm_movehl_ps(sums, sums);
            let sums = _mm_add_ss(sums, shuf);
            
            let mut result = _mm_cvtss_f32(sums);
            
            // Handle remaining elements
            for i in simd_len..len {
                let exp_val = (input[i] - max_val).exp();
                output[i] = exp_val;
                result += exp_val;
            }
            
            result
        }
        
        /// Fast exp approximation using AVX2
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "avx2")]
        unsafe fn exp_approx_avx2(x: __m256) -> __m256 {
            // Fast exp approximation using polynomial
            // More accurate versions would use lookup tables
            let one = _mm256_set1_ps(1.0);
            let x2 = _mm256_mul_ps(x, x);
            let x3 = _mm256_mul_ps(x2, x);
            let x4 = _mm256_mul_ps(x3, x);
            
            // 1 + x + x^2/2 + x^3/6 + x^4/24 (Taylor series approximation)
            let term2 = _mm256_mul_ps(x2, _mm256_set1_ps(0.5));
            let term3 = _mm256_mul_ps(x3, _mm256_set1_ps(1.0/6.0));
            let term4 = _mm256_mul_ps(x4, _mm256_set1_ps(1.0/24.0));
            
            let result = _mm256_add_ps(one, x);
            let result = _mm256_add_ps(result, term2);
            let result = _mm256_add_ps(result, term3);
            _mm256_add_ps(result, term4)
        }
        
        /// Scalar exp and sum computation
        fn exp_and_sum_scalar(input: &[f32], output: &mut [f32], max_val: f32) -> f32 {
            let mut sum = 0.0f32;
            for (i, o) in input.iter().zip(output.iter_mut()) {
                let exp_val = (i - max_val).exp();
                *o = exp_val;
                sum += exp_val;
            }
            sum
        }
        
        /// AVX2 normalization
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "avx2")]
        unsafe fn normalize_avx2(output: &mut [f32], inv_sum: f32) {
            let len = output.len();
            let simd_len = len - (len % 8);
            let inv_sum_vec = _mm256_set1_ps(inv_sum);
            
            for i in (0..simd_len).step_by(8) {
                let vals = _mm256_loadu_ps(output.as_ptr().add(i));
                let normalized = _mm256_mul_ps(vals, inv_sum_vec);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), normalized);
            }
            
            // Handle remaining elements
            for i in simd_len..len {
                output[i] *= inv_sum;
            }
        }
        
        /// Scalar normalization
        fn normalize_scalar(output: &mut [f32], inv_sum: f32) {
            for o in output.iter_mut() {
                *o *= inv_sum;
            }
        }
    }
    
    /// RMS Normalization (Root Mean Square Layer Normalization)
    pub struct RMSNorm;
    
    impl RMSNorm {
        /// Apply RMS normalization with SIMD optimization
        pub fn apply_f32(
            input: &[f32],
            weight: &[f32],
            output: &mut [f32],
            batch_size: usize,
            hidden_size: usize,
            eps: f32,
        ) -> Result<()> {
            assert_eq!(input.len(), batch_size * hidden_size);
            assert_eq!(weight.len(), hidden_size);
            assert_eq!(output.len(), input.len());
            
            for batch in 0..batch_size {
                let start_idx = batch * hidden_size;
                let input_slice = &input[start_idx..start_idx + hidden_size];
                let output_slice = &mut output[start_idx..start_idx + hidden_size];
                
                Self::rms_norm_1d(input_slice, weight, output_slice, eps)?;
            }
            
            Ok(())
        }
        
        /// 1D RMS normalization
        fn rms_norm_1d(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) -> Result<()> {
            let len = input.len();
            
            // Compute RMS
            let sum_sq = Self::compute_sum_squares_simd(input);
            let rms = (sum_sq / len as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;
            
            // Normalize and apply weight
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                if is_x86_feature_detected!("avx2") && _len >= 8 {
                    unsafe { Self::normalize_and_scale_avx2(input, weight, output, inv_rms) };
                } else {
                    Self::normalize_and_scale_scalar(input, weight, output, inv_rms);
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
            {
                Self::normalize_and_scale_scalar(input, weight, output, inv_rms);
            }
            
            Ok(())
        }
        
        /// Compute sum of squares with SIMD
        fn compute_sum_squares_simd(input: &[f32]) -> f32 {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                if is_x86_feature_detected!("avx2") && input.len() >= 8 {
                    return unsafe { Self::sum_squares_avx2(input) };
                }
            }
            
            input.iter().map(|&x| x * x).sum()
        }
        
        /// AVX2 sum of squares
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "avx2")]
        unsafe fn sum_squares_avx2(input: &[f32]) -> f32 {
            let len = input.len();
            let simd_len = len - (len % 8);
            let mut sum_vec = _mm256_setzero_ps();
            
            for i in (0..simd_len).step_by(8) {
                let vals = _mm256_loadu_ps(input.as_ptr().add(i));
                let squares = _mm256_mul_ps(vals, vals);
                sum_vec = _mm256_add_ps(sum_vec, squares);
            }
            
            // Horizontal sum
            let hi = _mm256_extractf128_ps(sum_vec, 1);
            let lo = _mm256_castps256_ps128(sum_vec);
            let sum_128 = _mm_add_ps(hi, lo);
            
            let shuf = _mm_movehdup_ps(sum_128);
            let sums = _mm_add_ps(sum_128, shuf);
            let shuf = _mm_movehl_ps(sums, sums);
            let sums = _mm_add_ss(sums, shuf);
            
            let mut result = _mm_cvtss_f32(sums);
            
            // Handle remaining elements
            for i in simd_len..len {
                result += input[i] * input[i];
            }
            
            result
        }
        
        /// AVX2 normalize and scale
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "avx2")]
        unsafe fn normalize_and_scale_avx2(input: &[f32], weight: &[f32], output: &mut [f32], inv_rms: f32) {
            let len = input.len();
            let simd_len = len - (len % 8);
            let inv_rms_vec = _mm256_set1_ps(inv_rms);
            
            for i in (0..simd_len).step_by(8) {
                let input_vals = _mm256_loadu_ps(input.as_ptr().add(i));
                let weight_vals = _mm256_loadu_ps(weight.as_ptr().add(i));
                
                let normalized = _mm256_mul_ps(input_vals, inv_rms_vec);
                let result = _mm256_mul_ps(normalized, weight_vals);
                
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            
            // Handle remaining elements
            for i in simd_len..len {
                output[i] = input[i] * inv_rms * weight[i];
            }
        }
        
        /// Scalar normalize and scale
        fn normalize_and_scale_scalar(input: &[f32], weight: &[f32], output: &mut [f32], inv_rms: f32) {
            for ((i, w), o) in input.iter().zip(weight.iter()).zip(output.iter_mut()) {
                *o = i * inv_rms * w;
            }
        }
    }
    
    /// GELU activation function (Gaussian Error Linear Unit)
    pub struct GELU;
    
    impl GELU {
        /// Apply GELU activation with SIMD optimization
        pub fn apply_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
            assert_eq!(input.len(), output.len());
            
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                if is_x86_feature_detected!("avx2") && input.len() >= 8 {
                    unsafe { Self::gelu_avx2(input, output) };
                } else {
                    Self::gelu_scalar(input, output);
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
            {
                Self::gelu_scalar(input, output);
            }
            
            Ok(())
        }
        
        /// AVX2 GELU implementation
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "avx2")]
        unsafe fn gelu_avx2(input: &[f32], output: &mut [f32]) {
            let len = input.len();
            let simd_len = len - (len % 8);
            
            // Constants for GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            let half = _mm256_set1_ps(0.5);
            let one = _mm256_set1_ps(1.0);
            let gelu_coeff = _mm256_set1_ps(0.7978845608); // sqrt(2/π)
            let gelu_bias = _mm256_set1_ps(0.044715);
            
            for i in (0..simd_len).step_by(8) {
                let x = _mm256_loadu_ps(input.as_ptr().add(i));
                
                // Compute x^3
                let x2 = _mm256_mul_ps(x, x);
                let x3 = _mm256_mul_ps(x2, x);
                
                // Compute the argument to tanh: sqrt(2/π) * (x + 0.044715 * x^3)
                let bias_term = _mm256_mul_ps(gelu_bias, x3);
                let sum_term = _mm256_add_ps(x, bias_term);
                let tanh_arg = _mm256_mul_ps(gelu_coeff, sum_term);
                
                // Fast tanh approximation
                let tanh_val = Self::tanh_approx_avx2(tanh_arg);
                
                // GELU = 0.5 * x * (1 + tanh(...))
                let one_plus_tanh = _mm256_add_ps(one, tanh_val);
                let half_x = _mm256_mul_ps(half, x);
                let result = _mm256_mul_ps(half_x, one_plus_tanh);
                
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            
            // Handle remaining elements
            for i in simd_len..len {
                output[i] = Self::gelu_scalar_single(input[i]);
            }
        }
        
        /// Fast tanh approximation
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "avx2")]
        unsafe fn tanh_approx_avx2(x: __m256) -> __m256 {
            // Clamp to avoid overflow
            let bound = _mm256_set1_ps(3.0);
            let x_clamped = _mm256_max_ps(_mm256_min_ps(x, bound), _mm256_sub_ps(_mm256_setzero_ps(), bound));
            
            // Polynomial approximation for tanh
            let x2 = _mm256_mul_ps(x_clamped, x_clamped);
            let x3 = _mm256_mul_ps(x2, x_clamped);
            
            // tanh(x) ≈ x - x^3/3 + 2*x^5/15 for small x
            let term1 = x_clamped;
            let term2 = _mm256_mul_ps(x3, _mm256_set1_ps(-1.0/3.0));
            
            _mm256_add_ps(term1, term2)
        }
        
        /// Scalar GELU implementation
        fn gelu_scalar(input: &[f32], output: &mut [f32]) {
            for (i, o) in input.iter().zip(output.iter_mut()) {
                *o = Self::gelu_scalar_single(*i);
            }
        }
        
        /// Single-value GELU computation
        fn gelu_scalar_single(x: f32) -> f32 {
            const GELU_COEFF: f32 = 0.7978845608; // sqrt(2/π)
            const GELU_BIAS: f32 = 0.044715;
            
            let tanh_arg = GELU_COEFF * (x + GELU_BIAS * x * x * x);
            0.5 * x * (1.0 + tanh_arg.tanh())
        }
    }
    
    /// Rotary Position Embedding (RoPE)
    pub struct RoPE;
    
    impl RoPE {
        /// Apply RoPE to query and key tensors
        pub fn apply_f32(
            input: &[f32],        // Input tensor [batch, seq_len, n_heads, head_dim]
            output: &mut [f32],   // Output tensor
            positions: &[usize],  // Position indices
            base: f32,           // Base frequency
            batch_size: usize,
            seq_len: usize,
            n_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            assert_eq!(input.len(), batch_size * seq_len * n_heads * head_dim);
            assert_eq!(output.len(), input.len());
            assert_eq!(positions.len(), seq_len);
            assert_eq!(head_dim % 2, 0); // Head dimension must be even
            
            let half_head_dim = head_dim / 2;
            
            // Precompute frequency constants
            let mut freqs = vec![0.0f32; half_head_dim];
            for i in 0..half_head_dim {
                freqs[i] = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            }
            
            for batch in 0..batch_size {
                for seq in 0..seq_len {
                    let pos = positions[seq] as f32;
                    
                    for head in 0..n_heads {
                        let base_idx = ((batch * seq_len + seq) * n_heads + head) * head_dim;
                        
                        for i in 0..half_head_dim {
                            let freq = freqs[i];
                            let angle = pos * freq;
                            let cos_val = angle.cos();
                            let sin_val = angle.sin();
                            
                            let idx1 = base_idx + i;
                            let idx2 = base_idx + i + half_head_dim;
                            
                            let x1 = input[idx1];
                            let x2 = input[idx2];
                            
                            // Apply rotation matrix
                            output[idx1] = x1 * cos_val - x2 * sin_val;
                            output[idx2] = x1 * sin_val + x2 * cos_val;
                        }
                    }
                }
            }
            
            Ok(())
        }
    }
}

// Re-export neural network operations
pub use neural::*;

// Placeholder functions for operations referenced in lib.rs

/// Apply softmax function to a tensor along the specified axis
/// 
/// # Arguments
/// * `tensor` - Input tensor
/// * `axis` - Axis along which to apply softmax
pub fn softmax<B: TensorBackend, T>(_tensor: &crate::tensor::Tensor<B, T>, _axis: i32) -> Result<crate::tensor::Tensor<B, T>>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    todo!("Softmax integration with tensor abstraction not yet implemented")
}

/// Apply Gaussian Error Linear Unit (GELU) activation function
/// 
/// # Arguments
/// * `tensor` - Input tensor
pub fn gelu<B: TensorBackend, T>(_tensor: &crate::tensor::Tensor<B, T>) -> Result<crate::tensor::Tensor<B, T>>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    todo!("GELU integration with tensor abstraction not yet implemented")
}

/// Apply layer normalization
/// 
/// # Arguments
/// * `tensor` - Input tensor
/// * `weight` - Scaling weights
/// * `bias` - Optional bias term
/// * `eps` - Small constant for numerical stability
pub fn layer_norm<B: TensorBackend, T>(
    _tensor: &crate::tensor::Tensor<B, T>,
    _weight: &crate::tensor::Tensor<B, T>,
    _bias: Option<&crate::tensor::Tensor<B, T>>,
    _eps: f32,
) -> Result<crate::tensor::Tensor<B, T>>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    todo!("LayerNorm integration with tensor abstraction not yet implemented")
}

/// Apply Rotary Position Embedding (RoPE)
/// 
/// # Arguments
/// * `tensor` - Input tensor
/// * `positions` - Position indices
/// * `base` - Base frequency for rotary embeddings
pub fn rope<B: TensorBackend, T>(
    _tensor: &crate::tensor::Tensor<B, T>,
    _positions: &crate::tensor::Tensor<B, T>,
    _base: f32,
) -> Result<crate::tensor::Tensor<B, T>>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    todo!("RoPE integration with tensor abstraction not yet implemented")
}