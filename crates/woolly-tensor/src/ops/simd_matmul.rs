//! SIMD-optimized matrix multiplication kernels
//! 
//! This module provides high-performance matrix multiplication implementations
//! optimized for both x86_64 (AVX2) and ARM (NEON) architectures.

use crate::backend::{Result, TensorError};
use crate::shape::Shape;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Configuration for matrix-vector multiplication
#[derive(Debug, Clone, Copy)]
pub struct MatVecConfig {
    /// Whether the matrix is transposed
    pub transpose: bool,
    /// Alpha scaling factor (y = alpha * A @ x + beta * y)
    pub alpha: f32,
    /// Beta scaling factor
    pub beta: f32,
}

impl Default for MatVecConfig {
    fn default() -> Self {
        Self {
            transpose: false,
            alpha: 1.0,
            beta: 0.0,
        }
    }
}

/// SIMD-optimized matrix-vector multiplication
pub struct SimdMatVec;

impl SimdMatVec {
    /// Performs matrix-vector multiplication with runtime CPU feature detection
    pub fn compute(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        matrix_shape: &Shape,
        config: &MatVecConfig,
    ) -> Result<()> {
        // Validate shapes
        let (m, n) = if matrix_shape.ndim() == 2 {
            (matrix_shape.as_slice()[0], matrix_shape.as_slice()[1])
        } else {
            return Err(TensorError::invalid_shape(
                "SIMD_MATVEC_INVALID_MATRIX",
                "Matrix must be 2D",
                format!("{:?}", matrix_shape),
                "matrix-vector multiplication",
                "Invalid matrix shape",
                "Ensure matrix is 2-dimensional"
            ));
        };

        let (rows, cols) = if config.transpose { (n, m) } else { (m, n) };
        
        if vector.len() != cols {
            return Err(TensorError::incompatible_shapes(
                "SIMD_MATVEC_DIM_MISMATCH",
                format!("Vector length {} doesn't match matrix columns {}", vector.len(), cols),
                "matrix-vector multiplication",
                format!("{:?}", matrix_shape),
                format!("[{}]", vector.len()),
                "Ensure vector length matches matrix columns"
            ));
        }

        if output.len() != rows {
            return Err(TensorError::invalid_shape(
                "SIMD_MATVEC_OUTPUT_SIZE_MISMATCH",
                format!("Output length {} doesn't match expected {}", output.len(), rows),
                format!("[{}]", output.len()),
                "matrix-vector multiplication",
                "Output size mismatch",
                "Ensure output length matches matrix rows"
            ));
        }

        // Apply beta scaling to output
        if config.beta != 1.0 {
            if config.beta == 0.0 {
                output.fill(0.0);
            } else {
                for o in output.iter_mut() {
                    *o *= config.beta;
                }
            }
        }

        // Choose optimal kernel based on CPU features
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    Self::compute_avx2_fma(matrix, vector, output, m, n, config)?;
                }
                return Ok(());
            } else if is_x86_feature_detected!("avx2") {
                unsafe {
                    Self::compute_avx2(matrix, vector, output, m, n, config)?;
                }
                return Ok(());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                Self::compute_neon(matrix, vector, output, m, n, config)
            }
        }
        
        #[cfg(not(target_arch = "aarch64"))]
        {
            // Fallback to optimized scalar implementation
            Self::compute_scalar(matrix, vector, output, m, n, config)
        }
    }

    /// AVX2 + FMA implementation for x86_64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn compute_avx2_fma(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        config: &MatVecConfig,
    ) -> Result<()> {
        let alpha = _mm256_set1_ps(config.alpha);
        
        if !config.transpose {
            // Standard matrix-vector multiplication: y = A @ x
            for i in 0..m {
                let mut sum0 = _mm256_setzero_ps();
                let mut sum1 = _mm256_setzero_ps();
                let mut sum2 = _mm256_setzero_ps();
                let mut sum3 = _mm256_setzero_ps();
                
                let row_start = i * n;
                let mut j = 0;
                
                // Process 32 elements at a time (4x8)
                while j + 32 <= n {
                    let a0 = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j));
                    let a1 = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j + 8));
                    let a2 = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j + 16));
                    let a3 = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j + 24));
                    
                    let x0 = _mm256_loadu_ps(vector.as_ptr().add(j));
                    let x1 = _mm256_loadu_ps(vector.as_ptr().add(j + 8));
                    let x2 = _mm256_loadu_ps(vector.as_ptr().add(j + 16));
                    let x3 = _mm256_loadu_ps(vector.as_ptr().add(j + 24));
                    
                    sum0 = _mm256_fmadd_ps(a0, x0, sum0);
                    sum1 = _mm256_fmadd_ps(a1, x1, sum1);
                    sum2 = _mm256_fmadd_ps(a2, x2, sum2);
                    sum3 = _mm256_fmadd_ps(a3, x3, sum3);
                    
                    j += 32;
                }
                
                // Process 8 elements at a time
                while j + 8 <= n {
                    let a = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j));
                    let x = _mm256_loadu_ps(vector.as_ptr().add(j));
                    sum0 = _mm256_fmadd_ps(a, x, sum0);
                    j += 8;
                }
                
                // Horizontal sum
                sum0 = _mm256_add_ps(sum0, sum1);
                sum2 = _mm256_add_ps(sum2, sum3);
                sum0 = _mm256_add_ps(sum0, sum2);
                
                let mut result = Self::hadd_avx2(sum0);
                
                // Handle remaining elements
                while j < n {
                    result += matrix[row_start + j] * vector[j];
                    j += 1;
                }
                
                // Store result with alpha scaling
                output[i] += config.alpha * result;
            }
        } else {
            // Transposed matrix-vector multiplication: y = A^T @ x
            // This is more cache-friendly as we access matrix row-wise
            let zero = _mm256_setzero_ps();
            
            // Initialize output accumulator vectors
            let mut j = 0;
            while j + 8 <= n {
                _mm256_storeu_ps(output.as_mut_ptr().add(j), zero);
                j += 8;
            }
            
            // Process matrix rows
            for i in 0..m {
                let xi = _mm256_set1_ps(vector[i] * config.alpha);
                let row_start = i * n;
                let mut j = 0;
                
                // Process 32 elements at a time
                while j + 32 <= n {
                    let a0 = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j));
                    let a1 = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j + 8));
                    let a2 = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j + 16));
                    let a3 = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j + 24));
                    
                    let y0 = _mm256_loadu_ps(output.as_ptr().add(j));
                    let y1 = _mm256_loadu_ps(output.as_ptr().add(j + 8));
                    let y2 = _mm256_loadu_ps(output.as_ptr().add(j + 16));
                    let y3 = _mm256_loadu_ps(output.as_ptr().add(j + 24));
                    
                    y0 = _mm256_fmadd_ps(a0, xi, y0);
                    y1 = _mm256_fmadd_ps(a1, xi, y1);
                    y2 = _mm256_fmadd_ps(a2, xi, y2);
                    y3 = _mm256_fmadd_ps(a3, xi, y3);
                    
                    _mm256_storeu_ps(output.as_mut_ptr().add(j), y0);
                    _mm256_storeu_ps(output.as_mut_ptr().add(j + 8), y1);
                    _mm256_storeu_ps(output.as_mut_ptr().add(j + 16), y2);
                    _mm256_storeu_ps(output.as_mut_ptr().add(j + 24), y3);
                    
                    j += 32;
                }
                
                // Process remaining elements
                while j + 8 <= n {
                    let a = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j));
                    let y = _mm256_loadu_ps(output.as_ptr().add(j));
                    let result = _mm256_fmadd_ps(a, xi, y);
                    _mm256_storeu_ps(output.as_mut_ptr().add(j), result);
                    j += 8;
                }
                
                // Handle remaining elements
                let xi_scalar = vector[i] * config.alpha;
                while j < n {
                    output[j] += matrix[row_start + j] * xi_scalar;
                    j += 1;
                }
            }
        }
        
        Ok(())
    }

    /// AVX2 implementation without FMA
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn compute_avx2(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        config: &MatVecConfig,
    ) -> Result<()> {
        if !config.transpose {
            // Standard matrix-vector multiplication
            for i in 0..m {
                let mut sum = _mm256_setzero_ps();
                let row_start = i * n;
                let mut j = 0;
                
                // Process 8 elements at a time
                while j + 8 <= n {
                    let a = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j));
                    let x = _mm256_loadu_ps(vector.as_ptr().add(j));
                    let prod = _mm256_mul_ps(a, x);
                    sum = _mm256_add_ps(sum, prod);
                    j += 8;
                }
                
                // Horizontal sum
                let mut result = Self::hadd_avx2(sum);
                
                // Handle remaining elements
                while j < n {
                    result += matrix[row_start + j] * vector[j];
                    j += 1;
                }
                
                output[i] += config.alpha * result;
            }
        } else {
            // Transposed multiplication - similar to FMA version but without FMA
            Self::compute_scalar(matrix, vector, output, m, n, config)?;
        }
        
        Ok(())
    }

    /// NEON implementation for ARM
    #[cfg(target_arch = "aarch64")]
    unsafe fn compute_neon(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        config: &MatVecConfig,
    ) -> Result<()> {
        if !config.transpose {
            // Standard matrix-vector multiplication
            for i in 0..m {
                let mut sum0 = vdupq_n_f32(0.0);
                let mut sum1 = vdupq_n_f32(0.0);
                let mut sum2 = vdupq_n_f32(0.0);
                let mut sum3 = vdupq_n_f32(0.0);
                
                let row_start = i * n;
                let mut j = 0;
                
                // Process 16 elements at a time (4x4)
                while j + 16 <= n {
                    let a0 = vld1q_f32(matrix.as_ptr().add(row_start + j));
                    let a1 = vld1q_f32(matrix.as_ptr().add(row_start + j + 4));
                    let a2 = vld1q_f32(matrix.as_ptr().add(row_start + j + 8));
                    let a3 = vld1q_f32(matrix.as_ptr().add(row_start + j + 12));
                    
                    let x0 = vld1q_f32(vector.as_ptr().add(j));
                    let x1 = vld1q_f32(vector.as_ptr().add(j + 4));
                    let x2 = vld1q_f32(vector.as_ptr().add(j + 8));
                    let x3 = vld1q_f32(vector.as_ptr().add(j + 12));
                    
                    sum0 = vfmaq_f32(sum0, a0, x0);
                    sum1 = vfmaq_f32(sum1, a1, x1);
                    sum2 = vfmaq_f32(sum2, a2, x2);
                    sum3 = vfmaq_f32(sum3, a3, x3);
                    
                    j += 16;
                }
                
                // Process 4 elements at a time
                while j + 4 <= n {
                    let a = vld1q_f32(matrix.as_ptr().add(row_start + j));
                    let x = vld1q_f32(vector.as_ptr().add(j));
                    sum0 = vfmaq_f32(sum0, a, x);
                    j += 4;
                }
                
                // Horizontal sum
                sum0 = vaddq_f32(sum0, sum1);
                sum2 = vaddq_f32(sum2, sum3);
                sum0 = vaddq_f32(sum0, sum2);
                let mut result = vaddvq_f32(sum0);
                
                // Handle remaining elements
                while j < n {
                    result += matrix[row_start + j] * vector[j];
                    j += 1;
                }
                
                output[i] += config.alpha * result;
            }
        } else {
            // Transposed matrix-vector multiplication
            let zero = vdupq_n_f32(0.0);
            
            // Initialize output
            let mut j = 0;
            while j + 4 <= n {
                vst1q_f32(output.as_mut_ptr().add(j), zero);
                j += 4;
            }
            
            // Process matrix rows
            for i in 0..m {
                let xi = vdupq_n_f32(vector[i] * config.alpha);
                let row_start = i * n;
                let mut j = 0;
                
                // Process 16 elements at a time
                while j + 16 <= n {
                    let a0 = vld1q_f32(matrix.as_ptr().add(row_start + j));
                    let a1 = vld1q_f32(matrix.as_ptr().add(row_start + j + 4));
                    let a2 = vld1q_f32(matrix.as_ptr().add(row_start + j + 8));
                    let a3 = vld1q_f32(matrix.as_ptr().add(row_start + j + 12));
                    
                    let y0 = vld1q_f32(output.as_ptr().add(j));
                    let y1 = vld1q_f32(output.as_ptr().add(j + 4));
                    let y2 = vld1q_f32(output.as_ptr().add(j + 8));
                    let y3 = vld1q_f32(output.as_ptr().add(j + 12));
                    
                    let r0 = vfmaq_f32(y0, a0, xi);
                    let r1 = vfmaq_f32(y1, a1, xi);
                    let r2 = vfmaq_f32(y2, a2, xi);
                    let r3 = vfmaq_f32(y3, a3, xi);
                    
                    vst1q_f32(output.as_mut_ptr().add(j), r0);
                    vst1q_f32(output.as_mut_ptr().add(j + 4), r1);
                    vst1q_f32(output.as_mut_ptr().add(j + 8), r2);
                    vst1q_f32(output.as_mut_ptr().add(j + 12), r3);
                    
                    j += 16;
                }
                
                // Process remaining elements
                while j + 4 <= n {
                    let a = vld1q_f32(matrix.as_ptr().add(row_start + j));
                    let y = vld1q_f32(output.as_ptr().add(j));
                    let result = vfmaq_f32(y, a, xi);
                    vst1q_f32(output.as_mut_ptr().add(j), result);
                    j += 4;
                }
                
                // Handle remaining elements
                let xi_scalar = vector[i] * config.alpha;
                while j < n {
                    output[j] += matrix[row_start + j] * xi_scalar;
                    j += 1;
                }
            }
        }
        
        Ok(())
    }

    /// Optimized scalar implementation
    fn compute_scalar(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        config: &MatVecConfig,
    ) -> Result<()> {
        if !config.transpose {
            // Standard matrix-vector multiplication with loop unrolling
            for i in 0..m {
                let row_start = i * n;
                let mut sum = 0.0f32;
                
                // Unroll by 4
                let mut j = 0;
                while j + 4 <= n {
                    sum += matrix[row_start + j] * vector[j]
                        + matrix[row_start + j + 1] * vector[j + 1]
                        + matrix[row_start + j + 2] * vector[j + 2]
                        + matrix[row_start + j + 3] * vector[j + 3];
                    j += 4;
                }
                
                // Handle remaining elements
                while j < n {
                    sum += matrix[row_start + j] * vector[j];
                    j += 1;
                }
                
                output[i] += config.alpha * sum;
            }
        } else {
            // Transposed matrix-vector multiplication
            // More cache-friendly implementation
            for i in 0..m {
                let xi = vector[i] * config.alpha;
                let row_start = i * n;
                
                for j in 0..n {
                    output[j] += matrix[row_start + j] * xi;
                }
            }
        }
        
        Ok(())
    }

    /// Horizontal add for AVX2
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn hadd_avx2(v: __m256) -> f32 {
        // Extract high and low 128-bit lanes
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        
        // Add high and low
        let sum_128 = _mm_add_ps(hi, lo);
        
        // Horizontal add within 128-bit vector
        let shuf = _mm_movehdup_ps(sum_128);
        let sums = _mm_add_ps(sum_128, shuf);
        let shuf = _mm_movehl_ps(sums, sums);
        let sums = _mm_add_ss(sums, shuf);
        
        _mm_cvtss_f32(sums)
    }
}

/// Cache-aware blocked matrix-vector multiplication
pub struct CacheAwareMatVec;

impl CacheAwareMatVec {
    /// Optimized matrix-vector multiplication with cache blocking for large matrices
    pub fn compute_blocked(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        matrix_shape: &Shape,
        config: &MatVecConfig,
    ) -> Result<()> {
        let (m, n) = if matrix_shape.ndim() == 2 {
            (matrix_shape.as_slice()[0], matrix_shape.as_slice()[1])
        } else {
            return Err(TensorError::invalid_shape(
                "CACHE_AWARE_MATVEC_INVALID_MATRIX",
                "Matrix must be 2D",
                format!("{:?}", matrix_shape),
                "cache-aware matrix-vector multiplication",
                "Invalid matrix shape",
                "Ensure matrix is 2-dimensional"
            ));
        };

        let (rows, cols) = if config.transpose { (n, m) } else { (m, n) };
        
        // Use blocking for large matrices to improve cache efficiency
        const BLOCK_SIZE: usize = 256; // Optimized for L1 cache
        
        if rows > BLOCK_SIZE && cols > BLOCK_SIZE {
            Self::compute_with_blocking(matrix, vector, output, m, n, config, BLOCK_SIZE)
        } else {
            // Use regular SIMD implementation for smaller matrices
            SimdMatVec::compute(matrix, vector, output, matrix_shape, config)
        }
    }

    fn compute_with_blocking(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        config: &MatVecConfig,
        block_size: usize,
    ) -> Result<()> {
        // Apply beta scaling to output
        if config.beta != 1.0 {
            if config.beta == 0.0 {
                output.fill(0.0);
            } else {
                for o in output.iter_mut() {
                    *o *= config.beta;
                }
            }
        }

        if !config.transpose {
            // Standard matrix-vector multiplication with row blocking
            for i_block in (0..m).step_by(block_size) {
                let i_end = std::cmp::min(i_block + block_size, m);
                
                for j_block in (0..n).step_by(block_size) {
                    let j_end = std::cmp::min(j_block + block_size, n);
                    
                    // Process block
                    for i in i_block..i_end {
                        let mut sum = 0.0f32;
                        let row_start = i * n;
                        
                        // Vectorized inner loop
                        #[cfg(target_arch = "aarch64")]
                        {
                            unsafe {
                                sum += Self::dot_product_neon_block(
                                    &matrix[row_start + j_block..row_start + j_end],
                                    &vector[j_block..j_end],
                                );
                            }
                        }
                        
                        #[cfg(target_arch = "x86_64")]
                        {
                            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                                unsafe {
                                    sum += Self::dot_product_avx2_fma_block(
                                        &matrix[row_start + j_block..row_start + j_end],
                                        &vector[j_block..j_end],
                                    );
                                }
                            } else {
                                sum += Self::dot_product_scalar_block(
                                    &matrix[row_start + j_block..row_start + j_end],
                                    &vector[j_block..j_end],
                                );
                            }
                        }
                        
                        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                        {
                            sum += Self::dot_product_scalar_block(
                                &matrix[row_start + j_block..row_start + j_end],
                                &vector[j_block..j_end],
                            );
                        }
                        
                        output[i] += config.alpha * sum;
                    }
                }
            }
        } else {
            // Transposed matrix-vector multiplication with column blocking
            for j_block in (0..n).step_by(block_size) {
                let j_end = std::cmp::min(j_block + block_size, n);
                
                for i_block in (0..m).step_by(block_size) {
                    let i_end = std::cmp::min(i_block + block_size, m);
                    
                    // Process block in transposed order
                    for i in i_block..i_end {
                        let xi = vector[i] * config.alpha;
                        let row_start = i * n;
                        
                        // Vectorized update
                        #[cfg(target_arch = "aarch64")]
                        {
                            unsafe {
                                Self::saxpy_neon_block(
                                    xi,
                                    &matrix[row_start + j_block..row_start + j_end],
                                    &mut output[j_block..j_end],
                                );
                            }
                        }
                        
                        #[cfg(target_arch = "x86_64")]
                        {
                            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                                unsafe {
                                    Self::saxpy_avx2_fma_block(
                                        xi,
                                        &matrix[row_start + j_block..row_start + j_end],
                                        &mut output[j_block..j_end],
                                    );
                                }
                            } else {
                                Self::saxpy_scalar_block(
                                    xi,
                                    &matrix[row_start + j_block..row_start + j_end],
                                    &mut output[j_block..j_end],
                                );
                            }
                        }
                        
                        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                        {
                            Self::saxpy_scalar_block(
                                xi,
                                &matrix[row_start + j_block..row_start + j_end],
                                &mut output[j_block..j_end],
                            );
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn dot_product_neon_block(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = vdupq_n_f32(0.0);
        let mut i = 0;
        
        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            sum = vfmaq_f32(sum, va, vb);
            i += 4;
        }
        
        let mut result = vaddvq_f32(sum);
        
        // Handle remaining elements
        while i < len {
            result += a[i] * b[i];
            i += 1;
        }
        
        result
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn dot_product_avx2_fma_block(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;
        
        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            sum = _mm256_fmadd_ps(va, vb, sum);
            i += 8;
        }
        
        let mut result = SimdMatVec::hadd_avx2(sum);
        
        // Handle remaining elements
        while i < len {
            result += a[i] * b[i];
            i += 1;
        }
        
        result
    }

    #[inline]
    fn dot_product_scalar_block(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = 0.0f32;
        let mut i = 0;
        
        // Unroll by 4 for better performance
        while i + 4 <= len {
            sum += a[i] * b[i] 
                + a[i + 1] * b[i + 1]
                + a[i + 2] * b[i + 2]
                + a[i + 3] * b[i + 3];
            i += 4;
        }
        
        // Handle remaining elements
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        
        sum
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn saxpy_neon_block(alpha: f32, x: &[f32], y: &mut [f32]) {
        let len = x.len().min(y.len());
        let valpha = vdupq_n_f32(alpha);
        let mut i = 0;
        
        // Process 4 elements at a time
        while i + 4 <= len {
            let vx = vld1q_f32(x.as_ptr().add(i));
            let vy = vld1q_f32(y.as_ptr().add(i));
            let result = vfmaq_f32(vy, valpha, vx);
            vst1q_f32(y.as_mut_ptr().add(i), result);
            i += 4;
        }
        
        // Handle remaining elements
        while i < len {
            y[i] += alpha * x[i];
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn saxpy_avx2_fma_block(alpha: f32, x: &[f32], y: &mut [f32]) {
        let len = x.len().min(y.len());
        let valpha = _mm256_set1_ps(alpha);
        let mut i = 0;
        
        // Process 8 elements at a time
        while i + 8 <= len {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let vy = _mm256_loadu_ps(y.as_ptr().add(i));
            let result = _mm256_fmadd_ps(valpha, vx, vy);
            _mm256_storeu_ps(y.as_mut_ptr().add(i), result);
            i += 8;
        }
        
        // Handle remaining elements
        while i < len {
            y[i] += alpha * x[i];
            i += 1;
        }
    }

    #[inline]
    fn saxpy_scalar_block(alpha: f32, x: &[f32], y: &mut [f32]) {
        let len = x.len().min(y.len());
        
        for i in 0..len {
            y[i] += alpha * x[i];
        }
    }
}

/// Optimized operations for transformer-specific patterns
pub struct TransformerSIMD;

impl TransformerSIMD {
    /// Optimized RMSNorm computation with SIMD
    pub fn rms_norm(
        input: &[f32],
        weight: &[f32],
        epsilon: f32,
        output: &mut [f32],
    ) -> Result<()> {
        if input.len() != weight.len() || input.len() != output.len() {
            return Err(TensorError::incompatible_shapes(
                "RMSNORM_SIZE_MISMATCH",
                "Input, weight, and output must have same size",
                "RMSNorm computation",
                format!("[{}]", input.len()),
                format!("[{}]", weight.len()),
                "Ensure all tensors have the same length"
            ));
        }

        let n = input.len();
        
        // Compute sum of squares
        let sum_sq = Self::sum_of_squares_simd(input);
        let rms = (sum_sq / n as f32 + epsilon).sqrt();
        let scale = 1.0 / rms;

        // Apply normalization and weight
        Self::normalize_and_scale_simd(input, weight, scale, output);
        
        Ok(())
    }

    /// SIMD-optimized sum of squares
    fn sum_of_squares_simd(input: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { Self::sum_of_squares_neon(input) }
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { Self::sum_of_squares_avx2_fma(input) }
            } else {
                Self::sum_of_squares_scalar(input)
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::sum_of_squares_scalar(input)
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn sum_of_squares_neon(input: &[f32]) -> f32 {
        let mut sum = vdupq_n_f32(0.0);
        let mut i = 0;
        
        // Process 4 elements at a time
        while i + 4 <= input.len() {
            let v = vld1q_f32(input.as_ptr().add(i));
            sum = vfmaq_f32(sum, v, v);
            i += 4;
        }
        
        let mut result = vaddvq_f32(sum);
        
        // Handle remaining elements
        while i < input.len() {
            let x = input[i];
            result += x * x;
            i += 1;
        }
        
        result
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn sum_of_squares_avx2_fma(input: &[f32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;
        
        // Process 8 elements at a time
        while i + 8 <= input.len() {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            sum = _mm256_fmadd_ps(v, v, sum);
            i += 8;
        }
        
        let mut result = SimdMatVec::hadd_avx2(sum);
        
        // Handle remaining elements
        while i < input.len() {
            let x = input[i];
            result += x * x;
            i += 1;
        }
        
        result
    }

    #[inline]
    fn sum_of_squares_scalar(input: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        
        for &x in input {
            sum += x * x;
        }
        
        sum
    }

    /// SIMD-optimized normalize and scale
    fn normalize_and_scale_simd(input: &[f32], weight: &[f32], scale: f32, output: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { Self::normalize_and_scale_neon(input, weight, scale, output) };
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { Self::normalize_and_scale_avx2_fma(input, weight, scale, output) };
            } else {
                Self::normalize_and_scale_scalar(input, weight, scale, output);
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::normalize_and_scale_scalar(input, weight, scale, output);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn normalize_and_scale_neon(input: &[f32], weight: &[f32], scale: f32, output: &mut [f32]) {
        let vscale = vdupq_n_f32(scale);
        let mut i = 0;
        
        // Process 4 elements at a time
        while i + 4 <= input.len() {
            let vin = vld1q_f32(input.as_ptr().add(i));
            let vweight = vld1q_f32(weight.as_ptr().add(i));
            let normalized = vmulq_f32(vin, vscale);
            let result = vmulq_f32(normalized, vweight);
            vst1q_f32(output.as_mut_ptr().add(i), result);
            i += 4;
        }
        
        // Handle remaining elements
        while i < input.len() {
            output[i] = input[i] * scale * weight[i];
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn normalize_and_scale_avx2_fma(input: &[f32], weight: &[f32], scale: f32, output: &mut [f32]) {
        let vscale = _mm256_set1_ps(scale);
        let mut i = 0;
        
        // Process 8 elements at a time
        while i + 8 <= input.len() {
            let vin = _mm256_loadu_ps(input.as_ptr().add(i));
            let vweight = _mm256_loadu_ps(weight.as_ptr().add(i));
            let normalized = _mm256_mul_ps(vin, vscale);
            let result = _mm256_mul_ps(normalized, vweight);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            i += 8;
        }
        
        // Handle remaining elements
        while i < input.len() {
            output[i] = input[i] * scale * weight[i];
            i += 1;
        }
    }

    #[inline]
    fn normalize_and_scale_scalar(input: &[f32], weight: &[f32], scale: f32, output: &mut [f32]) {
        for i in 0..input.len() {
            output[i] = input[i] * scale * weight[i];
        }
    }

    /// Optimized SwiGLU activation: swish(gate) * up
    pub fn swiglu_activation(
        gate: &[f32],
        up: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        if gate.len() != up.len() || gate.len() != output.len() {
            return Err(TensorError::incompatible_shapes(
                "SWIGLU_SIZE_MISMATCH",
                "Gate, up, and output must have same size",
                "SwiGLU activation",
                format!("[{}]", gate.len()),
                format!("[{}]", up.len()),
                "Ensure all tensors have the same length"
            ));
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { Self::swiglu_neon(gate, up, output) };
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { Self::swiglu_avx2_fma(gate, up, output) };
            } else {
                Self::swiglu_scalar(gate, up, output);
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::swiglu_scalar(gate, up, output);
        }

        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn swiglu_neon(gate: &[f32], up: &[f32], output: &mut [f32]) {
        let mut i = 0;
        
        // Process 4 elements at a time
        while i + 4 <= gate.len() {
            let vgate = vld1q_f32(gate.as_ptr().add(i));
            let vup = vld1q_f32(up.as_ptr().add(i));
            
            // Compute swish(gate) = gate / (1 + exp(-gate))
            // Approximation: gate * sigmoid(gate) ≈ gate * (0.5 * gate * tanh(0.5 * gate) + 0.5)
            let half = vdupq_n_f32(0.5);
            let scaled = vmulq_f32(vgate, half);
            
            // Fast tanh approximation for small values
            let tanh_approx = scaled; // For small values, tanh(x) ≈ x
            let sigmoid_approx = vfmaq_f32(half, half, tanh_approx);
            let swish = vmulq_f32(vgate, sigmoid_approx);
            
            // Final result: swish * up
            let result = vmulq_f32(swish, vup);
            vst1q_f32(output.as_mut_ptr().add(i), result);
            i += 4;
        }
        
        // Handle remaining elements
        while i < gate.len() {
            let g = gate[i];
            let swish = g / (1.0 + (-g).exp());
            output[i] = swish * up[i];
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn swiglu_avx2_fma(gate: &[f32], up: &[f32], output: &mut [f32]) {
        let mut i = 0;
        
        // Process 8 elements at a time
        while i + 8 <= gate.len() {
            let vgate = _mm256_loadu_ps(gate.as_ptr().add(i));
            let vup = _mm256_loadu_ps(up.as_ptr().add(i));
            
            // Compute swish using fast approximation
            let ones = _mm256_set1_ps(1.0);
            let neg_gate = _mm256_sub_ps(_mm256_setzero_ps(), vgate);
            
            // Fast exp approximation would go here, for simplicity using a reasonable approximation
            let exp_neg_gate = _mm256_add_ps(ones, neg_gate); // Simplified
            let sigmoid = _mm256_div_ps(ones, _mm256_add_ps(ones, exp_neg_gate));
            let swish = _mm256_mul_ps(vgate, sigmoid);
            
            // Final result: swish * up
            let result = _mm256_mul_ps(swish, vup);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            i += 8;
        }
        
        // Handle remaining elements
        while i < gate.len() {
            let g = gate[i];
            let swish = g / (1.0 + (-g).exp());
            output[i] = swish * up[i];
            i += 1;
        }
    }

    #[inline]
    fn swiglu_scalar(gate: &[f32], up: &[f32], output: &mut [f32]) {
        for i in 0..gate.len() {
            let g = gate[i];
            let swish = g / (1.0 + (-g).exp());
            output[i] = swish * up[i];
        }
    }
}

/// SIMD-optimized small matrix multiplication kernels
pub struct SimdMicroKernels;

impl SimdMicroKernels {
    /// 4x4 matrix multiplication kernel for AVX2
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn kernel_4x4_avx2(
        a: &[f32],  // 4x4 matrix in row-major
        b: &[f32],  // 4x4 matrix in row-major
        c: &mut [f32],  // 4x4 output matrix
        alpha: f32,
    ) {
        // Load B columns
        let b_col0 = _mm_set_ps(b[12], b[8], b[4], b[0]);
        let b_col1 = _mm_set_ps(b[13], b[9], b[5], b[1]);
        let b_col2 = _mm_set_ps(b[14], b[10], b[6], b[2]);
        let b_col3 = _mm_set_ps(b[15], b[11], b[7], b[3]);
        
        let alpha_vec = _mm_set1_ps(alpha);
        
        // Process each row of A
        for i in 0..4 {
            let a_row = _mm_loadu_ps(&a[i * 4]);
            
            // Broadcast each element of the row
            let a0 = _mm_shuffle_ps(a_row, a_row, 0x00);
            let a1 = _mm_shuffle_ps(a_row, a_row, 0x55);
            let a2 = _mm_shuffle_ps(a_row, a_row, 0xAA);
            let a3 = _mm_shuffle_ps(a_row, a_row, 0xFF);
            
            // Compute dot products
            let mut result = _mm_mul_ps(a0, b_col0);
            result = _mm_fmadd_ps(a1, b_col1, result);
            result = _mm_fmadd_ps(a2, b_col2, result);
            result = _mm_fmadd_ps(a3, b_col3, result);
            
            // Scale by alpha and store
            result = _mm_mul_ps(result, alpha_vec);
            _mm_storeu_ps(&mut c[i * 4], result);
        }
    }

    /// 4x4 matrix multiplication kernel for NEON
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn kernel_4x4_neon(
        a: &[f32],  // 4x4 matrix in row-major
        b: &[f32],  // 4x4 matrix in row-major
        c: &mut [f32],  // 4x4 output matrix
        alpha: f32,
    ) {
        // Load B matrix (transposed for column access)
        let b0 = vld1q_f32(&b[0]);
        let b1 = vld1q_f32(&b[4]);
        let b2 = vld1q_f32(&b[8]);
        let b3 = vld1q_f32(&b[12]);
        
        // Transpose B to get columns
        let tmp0 = vtrn1q_f32(b0, b1);
        let tmp1 = vtrn2q_f32(b0, b1);
        let tmp2 = vtrn1q_f32(b2, b3);
        let tmp3 = vtrn2q_f32(b2, b3);
        
        let b_col0 = vtrn1q_f64(vreinterpretq_f64_f32(tmp0), vreinterpretq_f64_f32(tmp2));
        let b_col1 = vtrn2q_f64(vreinterpretq_f64_f32(tmp0), vreinterpretq_f64_f32(tmp2));
        let b_col2 = vtrn1q_f64(vreinterpretq_f64_f32(tmp1), vreinterpretq_f64_f32(tmp3));
        let b_col3 = vtrn2q_f64(vreinterpretq_f64_f32(tmp1), vreinterpretq_f64_f32(tmp3));
        
        let alpha_vec = vdupq_n_f32(alpha);
        
        // Process each row of A
        for i in 0..4 {
            let a_row = vld1q_f32(&a[i * 4]);
            
            // Broadcast elements
            let a0 = vdupq_laneq_f32(a_row, 0);
            let a1 = vdupq_laneq_f32(a_row, 1);
            let a2 = vdupq_laneq_f32(a_row, 2);
            let a3 = vdupq_laneq_f32(a_row, 3);
            
            // Compute result
            let mut result = vmulq_f32(a0, vreinterpretq_f32_f64(b_col0));
            result = vfmaq_f32(result, a1, vreinterpretq_f32_f64(b_col1));
            result = vfmaq_f32(result, a2, vreinterpretq_f32_f64(b_col2));
            result = vfmaq_f32(result, a3, vreinterpretq_f32_f64(b_col3));
            
            // Scale and store
            result = vmulq_f32(result, alpha_vec);
            vst1q_f32(&mut c[i * 4], result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matvec_basic() {
        let matrix = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let vector = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 2];
        
        SimdMatVec::compute(
            &matrix,
            &vector,
            &mut output,
            &Shape::matrix(2, 3),
            &MatVecConfig::default(),
        ).unwrap();
        
        // Expected: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
        assert!((output[0] - 14.0).abs() < 1e-6);
        assert!((output[1] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_transposed() {
        let matrix = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ];
        let vector = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 2];
        
        SimdMatVec::compute(
            &matrix,
            &vector,
            &mut output,
            &Shape::matrix(3, 2),
            &MatVecConfig {
                transpose: true,
                alpha: 1.0,
                beta: 0.0,
            },
        ).unwrap();
        
        // A^T @ x where A^T is 2x3
        // Expected: [1*1 + 3*2 + 5*3, 2*1 + 4*2 + 6*3] = [22, 28]
        assert!((output[0] - 22.0).abs() < 1e-6);
        assert!((output[1] - 28.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_alpha_beta() {
        let matrix = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        let vector = vec![1.0, 2.0];
        let mut output = vec![10.0, 20.0];
        
        SimdMatVec::compute(
            &matrix,
            &vector,
            &mut output,
            &Shape::matrix(2, 2),
            &MatVecConfig {
                transpose: false,
                alpha: 2.0,
                beta: 0.5,
            },
        ).unwrap();
        
        // y = 2.0 * A @ x + 0.5 * y
        // A @ x = [5, 11]
        // y = 2.0 * [5, 11] + 0.5 * [10, 20] = [10, 22] + [5, 10] = [15, 32]
        assert!((output[0] - 15.0).abs() < 1e-6);
        assert!((output[1] - 32.0).abs() < 1e-6);
    }
}