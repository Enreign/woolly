//! Optimized SIMD implementation with memory pooling and cached CPU detection
//! 
//! This module provides highly optimized SIMD operations that:
//! - Use memory pools to eliminate allocations
//! - Cache CPU feature detection
//! - Apply size thresholds for SIMD usage
//! - Reuse buffers across operations

use crate::backend::{Result, TensorError};
use crate::shape::Shape;
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Minimum size threshold for SIMD operations (in elements)
/// Below this threshold, scalar operations are more efficient due to overhead
const SIMD_MIN_SIZE: usize = 256;

/// Cached CPU feature detection
static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
struct CpuFeatures {
    #[cfg(target_arch = "x86_64")]
    has_avx2: bool,
    #[cfg(target_arch = "x86_64")]
    has_fma: bool,
    #[cfg(target_arch = "x86_64")]
    has_avx512: bool,
    #[cfg(target_arch = "aarch64")]
    has_neon: bool,
}

impl CpuFeatures {
    fn detect() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            has_avx2: is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            has_fma: is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            has_avx512: is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "aarch64")]
            has_neon: true, // NEON is always available on aarch64
        }
    }
    
    #[inline(always)]
    fn get() -> &'static Self {
        CPU_FEATURES.get_or_init(Self::detect)
    }
}

/// Thread-local buffer pool for SIMD operations
thread_local! {
    static SIMD_BUFFER_POOL: std::cell::RefCell<SimdBufferPool> = std::cell::RefCell::new(SimdBufferPool::new());
}

/// Buffer pool for SIMD operations to avoid allocations
struct SimdBufferPool {
    /// Small buffers (< 4KB)
    small_buffers: Vec<Vec<f32>>,
    /// Medium buffers (4KB - 64KB)
    medium_buffers: Vec<Vec<f32>>,
    /// Large buffers (> 64KB)
    large_buffers: Vec<Vec<f32>>,
}

impl SimdBufferPool {
    fn new() -> Self {
        Self {
            small_buffers: Vec::with_capacity(16),
            medium_buffers: Vec::with_capacity(8),
            large_buffers: Vec::with_capacity(4),
        }
    }
    
    /// Get a buffer with at least the requested capacity
    fn get_buffer(&mut self, size: usize) -> Vec<f32> {
        let pool = if size < 1024 {
            &mut self.small_buffers
        } else if size < 16384 {
            &mut self.medium_buffers
        } else {
            &mut self.large_buffers
        };
        
        // Try to find a suitable buffer
        if let Some(pos) = pool.iter().position(|buf| buf.capacity() >= size) {
            let mut buffer = pool.swap_remove(pos);
            buffer.clear();
            buffer.resize(size, 0.0);
            return buffer;
        }
        
        // Allocate new buffer with some extra capacity
        let capacity = if size < 1024 {
            (size + 255) & !255  // Round up to 256
        } else if size < 16384 {
            (size + 1023) & !1023  // Round up to 1024
        } else {
            (size + 4095) & !4095  // Round up to 4096
        };
        
        vec![0.0; capacity]
    }
    
    /// Return a buffer to the pool
    fn return_buffer(&mut self, mut buffer: Vec<f32>) {
        let capacity = buffer.capacity();
        buffer.clear();
        
        let pool = if capacity < 1024 {
            &mut self.small_buffers
        } else if capacity < 16384 {
            &mut self.medium_buffers
        } else {
            &mut self.large_buffers
        };
        
        // Only keep a limited number of buffers
        let max_buffers = if capacity < 1024 { 16 } else if capacity < 16384 { 8 } else { 4 };
        
        if pool.len() < max_buffers {
            pool.push(buffer);
        }
    }
}

/// Optimized matrix-vector multiplication configuration
#[derive(Debug, Clone, Copy)]
pub struct OptimizedMatVecConfig {
    pub transpose: bool,
    pub alpha: f32,
    pub beta: f32,
    /// Use SIMD for operations larger than this threshold
    pub simd_threshold: usize,
}

impl Default for OptimizedMatVecConfig {
    fn default() -> Self {
        Self {
            transpose: false,
            alpha: 1.0,
            beta: 0.0,
            simd_threshold: SIMD_MIN_SIZE,
        }
    }
}

/// Optimized SIMD matrix-vector multiplication
pub struct OptimizedSimdMatVec;

impl OptimizedSimdMatVec {
    /// Performs optimized matrix-vector multiplication with pooled buffers
    pub fn compute_pooled(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        matrix_shape: &Shape,
        config: &OptimizedMatVecConfig,
    ) -> Result<()> {
        let (m, n) = if matrix_shape.ndim() == 2 {
            (matrix_shape.as_slice()[0], matrix_shape.as_slice()[1])
        } else {
            return Err(TensorError::invalid_shape(
                "OPTIMIZED_MATVEC_INVALID_MATRIX",
                "Matrix must be 2D",
                format!("{:?}", matrix_shape),
                "optimized matrix-vector multiplication",
                "Invalid matrix shape",
                "Ensure matrix is 2-dimensional"
            ));
        };

        let (rows, cols) = if config.transpose { (n, m) } else { (m, n) };
        
        // Validate dimensions
        if vector.len() != cols {
            return Err(TensorError::incompatible_shapes(
                "OPTIMIZED_MATVEC_DIM_MISMATCH",
                format!("Vector length {} doesn't match matrix columns {}", vector.len(), cols),
                "optimized matrix-vector multiplication",
                format!("{:?}", matrix_shape),
                format!("[{}]", vector.len()),
                "Ensure vector length matches matrix columns"
            ));
        }

        if output.len() != rows {
            return Err(TensorError::invalid_shape(
                "OPTIMIZED_MATVEC_OUTPUT_SIZE_MISMATCH",
                format!("Output length {} doesn't match expected {}", output.len(), rows),
                format!("[{}]", output.len()),
                "optimized matrix-vector multiplication",
                "Output size mismatch",
                "Ensure output length matches matrix rows"
            ));
        }

        // Apply beta scaling
        if config.beta != 1.0 {
            if config.beta == 0.0 {
                output.fill(0.0);
            } else {
                for o in output.iter_mut() {
                    *o *= config.beta;
                }
            }
        }

        // Check if operation is large enough for SIMD
        let total_ops = rows * cols;
        if total_ops < config.simd_threshold {
            // Use scalar implementation for small operations
            return Self::compute_scalar(matrix, vector, output, m, n, config);
        }

        // Get cached CPU features
        let features = CpuFeatures::get();

        // Choose optimal kernel based on cached features
        #[cfg(target_arch = "x86_64")]
        {
            if features.has_avx2 && features.has_fma {
                unsafe {
                    return Self::compute_avx2_fma_optimized(matrix, vector, output, m, n, config);
                }
            } else if features.has_avx2 {
                unsafe {
                    return Self::compute_avx2_optimized(matrix, vector, output, m, n, config);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if features.has_neon {
                unsafe {
                    return Self::compute_neon_optimized(matrix, vector, output, m, n, config);
                }
            }
        }

        // Fallback to scalar
        Self::compute_scalar(matrix, vector, output, m, n, config)
    }

    /// Optimized AVX2 + FMA implementation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn compute_avx2_fma_optimized(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        config: &OptimizedMatVecConfig,
    ) -> Result<()> {
        let alpha = _mm256_set1_ps(config.alpha);
        
        if !config.transpose {
            // Standard matrix-vector multiplication with unrolling
            for i in 0..m {
                let mut sum0 = _mm256_setzero_ps();
                let mut sum1 = _mm256_setzero_ps();
                let mut sum2 = _mm256_setzero_ps();
                let mut sum3 = _mm256_setzero_ps();
                
                let row_start = i * n;
                let mut j = 0;
                
                // Process 32 elements at a time
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
                
                // Process remaining 8-element chunks
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
                
                output[i] += config.alpha * result;
            }
        } else {
            // Transposed multiplication with better cache usage
            let zero = _mm256_setzero_ps();
            
            // Initialize output with SIMD
            let mut j = 0;
            while j + 8 <= n {
                _mm256_storeu_ps(output.as_mut_ptr().add(j), zero);
                j += 8;
            }
            while j < n {
                output[j] = 0.0;
                j += 1;
            }
            
            // Process rows in blocks for better cache usage
            const ROW_BLOCK: usize = 4;
            for i_block in (0..m).step_by(ROW_BLOCK) {
                let i_end = (i_block + ROW_BLOCK).min(m);
                
                for i in i_block..i_end {
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
                        
                        let r0 = _mm256_fmadd_ps(a0, xi, y0);
                        let r1 = _mm256_fmadd_ps(a1, xi, y1);
                        let r2 = _mm256_fmadd_ps(a2, xi, y2);
                        let r3 = _mm256_fmadd_ps(a3, xi, y3);
                        
                        _mm256_storeu_ps(output.as_mut_ptr().add(j), r0);
                        _mm256_storeu_ps(output.as_mut_ptr().add(j + 8), r1);
                        _mm256_storeu_ps(output.as_mut_ptr().add(j + 16), r2);
                        _mm256_storeu_ps(output.as_mut_ptr().add(j + 24), r3);
                        
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
                    
                    let xi_scalar = vector[i] * config.alpha;
                    while j < n {
                        output[j] += matrix[row_start + j] * xi_scalar;
                        j += 1;
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Optimized AVX2 implementation without FMA
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn compute_avx2_optimized(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        config: &OptimizedMatVecConfig,
    ) -> Result<()> {
        let alpha = _mm256_set1_ps(config.alpha);
        
        if !config.transpose {
            for i in 0..m {
                let mut sum0 = _mm256_setzero_ps();
                let mut sum1 = _mm256_setzero_ps();
                
                let row_start = i * n;
                let mut j = 0;
                
                // Process 16 elements at a time
                while j + 16 <= n {
                    let a0 = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j));
                    let a1 = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j + 8));
                    
                    let x0 = _mm256_loadu_ps(vector.as_ptr().add(j));
                    let x1 = _mm256_loadu_ps(vector.as_ptr().add(j + 8));
                    
                    sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a0, x0));
                    sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(a1, x1));
                    
                    j += 16;
                }
                
                // Process remaining 8 elements
                while j + 8 <= n {
                    let a = _mm256_loadu_ps(matrix.as_ptr().add(row_start + j));
                    let x = _mm256_loadu_ps(vector.as_ptr().add(j));
                    sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a, x));
                    j += 8;
                }
                
                // Horizontal sum
                sum0 = _mm256_add_ps(sum0, sum1);
                let mut result = Self::hadd_avx2(sum0);
                
                // Handle remaining elements
                while j < n {
                    result += matrix[row_start + j] * vector[j];
                    j += 1;
                }
                
                let scaled = _mm_mul_ss(_mm_set_ss(result), _mm_set_ss(config.alpha));
                output[i] += _mm_cvtss_f32(scaled);
            }
        } else {
            // Delegate to scalar for transposed case without FMA
            return Self::compute_scalar(matrix, vector, output, m, n, config);
        }
        
        Ok(())
    }

    /// Optimized NEON implementation
    #[cfg(target_arch = "aarch64")]
    unsafe fn compute_neon_optimized(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        config: &OptimizedMatVecConfig,
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
                
                // Process 16 elements at a time
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
                
                // Process remaining 4 elements at a time
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
            // Transposed multiplication
            let zero = vdupq_n_f32(0.0);
            
            // Initialize output
            let mut j = 0;
            while j + 4 <= n {
                vst1q_f32(output.as_mut_ptr().add(j), zero);
                j += 4;
            }
            while j < n {
                output[j] = 0.0;
                j += 1;
            }
            
            // Process in blocks for cache efficiency
            const ROW_BLOCK: usize = 8;
            for i_block in (0..m).step_by(ROW_BLOCK) {
                let i_end = (i_block + ROW_BLOCK).min(m);
                
                for i in i_block..i_end {
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
                    
                    let xi_scalar = vector[i] * config.alpha;
                    while j < n {
                        output[j] += matrix[row_start + j] * xi_scalar;
                        j += 1;
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Optimized scalar implementation with loop unrolling
    fn compute_scalar(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        m: usize,
        n: usize,
        config: &OptimizedMatVecConfig,
    ) -> Result<()> {
        if !config.transpose {
            // Standard matrix-vector multiplication
            for i in 0..m {
                let row_start = i * n;
                let mut sum = 0.0f32;
                
                // Unroll by 8 for better performance
                let mut j = 0;
                while j + 8 <= n {
                    sum += matrix[row_start + j] * vector[j]
                        + matrix[row_start + j + 1] * vector[j + 1]
                        + matrix[row_start + j + 2] * vector[j + 2]
                        + matrix[row_start + j + 3] * vector[j + 3]
                        + matrix[row_start + j + 4] * vector[j + 4]
                        + matrix[row_start + j + 5] * vector[j + 5]
                        + matrix[row_start + j + 6] * vector[j + 6]
                        + matrix[row_start + j + 7] * vector[j + 7];
                    j += 8;
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
            // Initialize output to zero if beta is 0
            if config.beta == 0.0 {
                output.fill(0.0);
            }
            
            // Process in blocks for better cache usage
            const BLOCK_SIZE: usize = 64;
            for i_block in (0..m).step_by(BLOCK_SIZE) {
                let i_end = (i_block + BLOCK_SIZE).min(m);
                
                for i in i_block..i_end {
                    let xi = vector[i] * config.alpha;
                    let row_start = i * n;
                    
                    // Unroll by 8
                    let mut j = 0;
                    while j + 8 <= n {
                        output[j] += matrix[row_start + j] * xi;
                        output[j + 1] += matrix[row_start + j + 1] * xi;
                        output[j + 2] += matrix[row_start + j + 2] * xi;
                        output[j + 3] += matrix[row_start + j + 3] * xi;
                        output[j + 4] += matrix[row_start + j + 4] * xi;
                        output[j + 5] += matrix[row_start + j + 5] * xi;
                        output[j + 6] += matrix[row_start + j + 6] * xi;
                        output[j + 7] += matrix[row_start + j + 7] * xi;
                        j += 8;
                    }
                    
                    // Handle remaining
                    while j < n {
                        output[j] += matrix[row_start + j] * xi;
                        j += 1;
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Horizontal add for AVX2
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
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

/// Public API for optimized SIMD operations with memory pooling
pub struct SimdOpsOptimized;

impl SimdOpsOptimized {
    /// Matrix-vector multiplication with automatic memory pooling
    pub fn matvec(
        matrix: &[f32],
        vector: &[f32],
        matrix_shape: &Shape,
        transpose: bool,
    ) -> Result<Vec<f32>> {
        let (m, n) = if matrix_shape.ndim() == 2 {
            (matrix_shape.as_slice()[0], matrix_shape.as_slice()[1])
        } else {
            return Err(TensorError::invalid_shape(
                "SIMD_OPS_INVALID_MATRIX",
                "Matrix must be 2D",
                format!("{:?}", matrix_shape),
                "SIMD operations",
                "Invalid matrix shape",
                "Ensure matrix is 2-dimensional"
            ));
        };

        let (rows, _cols) = if transpose { (n, m) } else { (m, n) };
        
        // Get buffer from pool
        let mut output = SIMD_BUFFER_POOL.with(|pool| {
            pool.borrow_mut().get_buffer(rows)
        });
        
        let config = OptimizedMatVecConfig {
            transpose,
            alpha: 1.0,
            beta: 0.0,
            simd_threshold: SIMD_MIN_SIZE,
        };
        
        OptimizedSimdMatVec::compute_pooled(
            matrix,
            vector,
            &mut output,
            matrix_shape,
            &config,
        )?;
        
        Ok(output)
    }

    /// Matrix-vector multiplication with preallocated output
    pub fn matvec_into(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        matrix_shape: &Shape,
        transpose: bool,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let config = OptimizedMatVecConfig {
            transpose,
            alpha,
            beta,
            simd_threshold: SIMD_MIN_SIZE,
        };
        
        OptimizedSimdMatVec::compute_pooled(
            matrix,
            vector,
            output,
            matrix_shape,
            &config,
        )
    }

    /// Check if SIMD operations should be used for given size
    #[inline(always)]
    pub fn should_use_simd(size: usize) -> bool {
        size >= SIMD_MIN_SIZE
    }

    /// Get cached CPU features
    #[inline(always)]
    pub fn cpu_features() -> &'static CpuFeatures {
        CpuFeatures::get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_matvec() {
        let matrix = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let vector = vec![1.0, 2.0, 3.0];
        
        let result = SimdOpsOptimized::matvec(
            &matrix,
            &vector,
            &Shape::matrix(2, 3),
            false,
        ).unwrap();
        
        assert_eq!(result.len(), 2);
        assert!((result[0] - 14.0).abs() < 1e-6);
        assert!((result[1] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_threshold() {
        // Small operation should not use SIMD
        assert!(!SimdOpsOptimized::should_use_simd(100));
        
        // Large operation should use SIMD
        assert!(SimdOpsOptimized::should_use_simd(1000));
    }

    #[test]
    fn test_cpu_features_cached() {
        // Call twice to ensure caching works
        let features1 = SimdOpsOptimized::cpu_features();
        let features2 = SimdOpsOptimized::cpu_features();
        
        // Should return the same instance
        assert!(std::ptr::eq(features1, features2));
    }
}