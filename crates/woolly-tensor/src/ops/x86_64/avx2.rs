//! AVX2 SIMD implementations for various operations

use crate::ops::{SimdOps, Vectorizable, BinaryVectorizable};
use crate::backend::Result;
use std::arch::x86_64::*;

/// AVX2 implementation for f32
#[derive(Debug, Clone, Copy)]
pub struct Avx2F32;

#[cfg(target_feature = "avx2")]
impl SimdOps for Avx2F32 {
    type Scalar = f32;
    type Vector = __m256;
    
    const LANES: usize = 8;
    const ALIGN: usize = 32;
    
    #[inline]
    unsafe fn load(ptr: *const Self::Scalar) -> Self::Vector {
        _mm256_loadu_ps(ptr)
    }
    
    #[inline]
    unsafe fn load_aligned(ptr: *const Self::Scalar) -> Self::Vector {
        _mm256_load_ps(ptr)
    }
    
    #[inline]
    unsafe fn store(ptr: *mut Self::Scalar, vec: Self::Vector) {
        _mm256_storeu_ps(ptr, vec)
    }
    
    #[inline]
    unsafe fn store_aligned(ptr: *mut Self::Scalar, vec: Self::Vector) {
        _mm256_store_ps(ptr, vec)
    }
    
    #[inline]
    fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { _mm256_add_ps(a, b) }
    }
    
    #[inline]
    fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { _mm256_sub_ps(a, b) }
    }
    
    #[inline]
    fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { _mm256_mul_ps(a, b) }
    }
    
    #[inline]
    fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { _mm256_div_ps(a, b) }
    }
    
    #[inline]
    fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        #[cfg(target_feature = "fma")]
        unsafe {
            _mm256_fmadd_ps(a, b, c)
        }
        #[cfg(not(target_feature = "fma"))]
        unsafe {
            _mm256_add_ps(_mm256_mul_ps(a, b), c)
        }
    }
    
    #[inline]
    fn hadd(vec: Self::Vector) -> Self::Scalar {
        unsafe {
            // Horizontal add across all 8 lanes
            let hi = _mm256_extractf128_ps(vec, 1);
            let lo = _mm256_castps256_ps128(vec);
            let sum = _mm_add_ps(hi, lo);
            
            // Further reduction within 128-bit
            let shuf = _mm_movehdup_ps(sum);
            let sums = _mm_add_ps(sum, shuf);
            let shuf = _mm_movehl_ps(sums, sums);
            let sums = _mm_add_ss(sums, shuf);
            _mm_cvtss_f32(sums)
        }
    }
    
    #[inline]
    fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { _mm256_max_ps(a, b) }
    }
    
    #[inline]
    fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { _mm256_min_ps(a, b) }
    }
    
    #[inline]
    fn sqrt(vec: Self::Vector) -> Self::Vector {
        unsafe { _mm256_sqrt_ps(vec) }
    }
    
    #[inline]
    fn reciprocal(vec: Self::Vector) -> Self::Vector {
        unsafe { _mm256_rcp_ps(vec) }
    }
    
    #[inline]
    fn splat(value: Self::Scalar) -> Self::Vector {
        unsafe { _mm256_set1_ps(value) }
    }
    
    #[inline]
    fn zero() -> Self::Vector {
        unsafe { _mm256_setzero_ps() }
    }
}

/// AVX2-accelerated ReLU
pub struct Avx2ReLU;

impl Vectorizable for Avx2ReLU {
    fn apply_simd<S: SimdOps>(input: &[S::Scalar], output: &mut [S::Scalar]) -> Result<()> {
        if std::mem::size_of::<S::Scalar>() != std::mem::size_of::<f32>() {
            return Self::apply_scalar(input, output);
        }
        
        #[cfg(target_feature = "avx2")]
        unsafe {
            apply_relu_avx2_f32(
                std::slice::from_raw_parts(input.as_ptr() as *const f32, input.len()),
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, output.len()),
            )?;
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            Self::apply_scalar(input, output)?;
        }
        
        Ok(())
    }
    
    fn apply_scalar<T>(input: &[T], output: &mut [T]) -> Result<()> {
        // This is a generic fallback, but we need specific implementations
        unimplemented!("Generic scalar ReLU not implemented")
    }
}

/// Scalar ReLU for f32
pub fn apply_relu_scalar_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
    assert_eq!(input.len(), output.len());
    
    for (i, o) in input.iter().zip(output.iter_mut()) {
        *o = i.max(0.0);
    }
    
    Ok(())
}

/// AVX2 ReLU for f32
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn apply_relu_avx2_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
    assert_eq!(input.len(), output.len());
    
    let len = input.len();
    let simd_len = len - (len % 8);
    let zero = _mm256_setzero_ps();
    
    // Process 8 elements at a time
    for i in (0..simd_len).step_by(8) {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let result = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
    }
    
    // Handle remaining elements
    for i in simd_len..len {
        output[i] = input[i].max(0.0);
    }
    
    Ok(())
}

/// AVX2-accelerated element-wise operations
pub struct Avx2ElementWise;

impl BinaryVectorizable for Avx2ElementWise {
    fn apply_simd<S: SimdOps>(a: &[S::Scalar], b: &[S::Scalar], output: &mut [S::Scalar]) -> Result<()> {
        if std::mem::size_of::<S::Scalar>() != std::mem::size_of::<f32>() {
            return Self::apply_scalar(a, b, output);
        }
        
        #[cfg(target_feature = "avx2")]
        unsafe {
            apply_add_avx2_f32(
                std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()),
                std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()),
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, output.len()),
            )?;
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            Self::apply_scalar(a, b, output)?;
        }
        
        Ok(())
    }
    
    fn apply_scalar<T>(a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
        unimplemented!("Generic scalar binary operations not implemented")
    }
}

/// AVX2 element-wise addition for f32
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
unsafe fn apply_add_avx2_f32(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    
    let len = a.len();
    let simd_len = len - (len % 8);
    
    // Process 8 elements at a time
    for i in (0..simd_len).step_by(8) {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let result = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
    }
    
    // Handle remaining elements
    for i in simd_len..len {
        output[i] = a[i] + b[i];
    }
    
    Ok(())
}

/// AVX2 dot product for f32 vectors
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_avx2_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    let len = a.len();
    let simd_len = len - (len % 8);
    let mut sum = _mm256_setzero_ps();
    
    // Process 8 elements at a time with FMA if available
    for i in (0..simd_len).step_by(8) {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        
        #[cfg(target_feature = "fma")]
        {
            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        }
        #[cfg(not(target_feature = "fma"))]
        {
            let prod = _mm256_mul_ps(a_vec, b_vec);
            sum = _mm256_add_ps(sum, prod);
        }
    }
    
    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum_128 = _mm_add_ps(hi, lo);
    
    let shuf = _mm_movehdup_ps(sum_128);
    let sums = _mm_add_ps(sum_128, shuf);
    let shuf = _mm_movehl_ps(sums, sums);
    let sums = _mm_add_ss(sums, shuf);
    
    let mut result = _mm_cvtss_f32(sums);
    
    // Handle remaining elements
    for i in simd_len..len {
        result += a[i] * b[i];
    }
    
    result
}

/// AVX2-accelerated matrix multiplication kernel for small blocks
pub struct Avx2MatMul;

impl Avx2MatMul {
    /// Optimized 8x8 matrix multiplication kernel using AVX2
    #[cfg(target_feature = "avx2")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn kernel_8x8_f32(
        a: &[f32], // 8xK matrix in row-major order
        b: &[f32], // Kx8 matrix in row-major order
        c: &mut [f32], // 8x8 output matrix
        k: usize,
        lda: usize, // Leading dimension of A
        ldb: usize, // Leading dimension of B
        ldc: usize, // Leading dimension of C
    ) {
        // Initialize accumulator registers
        let mut c00 = _mm256_setzero_ps();
        let mut c01 = _mm256_setzero_ps();
        let mut c10 = _mm256_setzero_ps();
        let mut c11 = _mm256_setzero_ps();
        let mut c20 = _mm256_setzero_ps();
        let mut c21 = _mm256_setzero_ps();
        let mut c30 = _mm256_setzero_ps();
        let mut c31 = _mm256_setzero_ps();
        let mut c40 = _mm256_setzero_ps();
        let mut c41 = _mm256_setzero_ps();
        let mut c50 = _mm256_setzero_ps();
        let mut c51 = _mm256_setzero_ps();
        let mut c60 = _mm256_setzero_ps();
        let mut c61 = _mm256_setzero_ps();
        let mut c70 = _mm256_setzero_ps();
        let mut c71 = _mm256_setzero_ps();
        
        // Main computation loop
        for i in 0..k {
            // Load B values
            let b0 = _mm256_loadu_ps(b.as_ptr().add(i * ldb));
            
            // Load A values and compute
            let a0 = _mm256_broadcast_ss(a.as_ptr().add(0 * lda + i));
            let a1 = _mm256_broadcast_ss(a.as_ptr().add(1 * lda + i));
            let a2 = _mm256_broadcast_ss(a.as_ptr().add(2 * lda + i));
            let a3 = _mm256_broadcast_ss(a.as_ptr().add(3 * lda + i));
            let a4 = _mm256_broadcast_ss(a.as_ptr().add(4 * lda + i));
            let a5 = _mm256_broadcast_ss(a.as_ptr().add(5 * lda + i));
            let a6 = _mm256_broadcast_ss(a.as_ptr().add(6 * lda + i));
            let a7 = _mm256_broadcast_ss(a.as_ptr().add(7 * lda + i));
            
            #[cfg(target_feature = "fma")]
            {
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c20 = _mm256_fmadd_ps(a2, b0, c20);
                c30 = _mm256_fmadd_ps(a3, b0, c30);
                c40 = _mm256_fmadd_ps(a4, b0, c40);
                c50 = _mm256_fmadd_ps(a5, b0, c50);
                c60 = _mm256_fmadd_ps(a6, b0, c60);
                c70 = _mm256_fmadd_ps(a7, b0, c70);
            }
            #[cfg(not(target_feature = "fma"))]
            {
                c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
                c10 = _mm256_add_ps(c10, _mm256_mul_ps(a1, b0));
                c20 = _mm256_add_ps(c20, _mm256_mul_ps(a2, b0));
                c30 = _mm256_add_ps(c30, _mm256_mul_ps(a3, b0));
                c40 = _mm256_add_ps(c40, _mm256_mul_ps(a4, b0));
                c50 = _mm256_add_ps(c50, _mm256_mul_ps(a5, b0));
                c60 = _mm256_add_ps(c60, _mm256_mul_ps(a6, b0));
                c70 = _mm256_add_ps(c70, _mm256_mul_ps(a7, b0));
            }
        }
        
        // Store results
        _mm256_storeu_ps(c.as_mut_ptr().add(0 * ldc), c00);
        _mm256_storeu_ps(c.as_mut_ptr().add(1 * ldc), c10);
        _mm256_storeu_ps(c.as_mut_ptr().add(2 * ldc), c20);
        _mm256_storeu_ps(c.as_mut_ptr().add(3 * ldc), c30);
        _mm256_storeu_ps(c.as_mut_ptr().add(4 * ldc), c40);
        _mm256_storeu_ps(c.as_mut_ptr().add(5 * ldc), c50);
        _mm256_storeu_ps(c.as_mut_ptr().add(6 * ldc), c60);
        _mm256_storeu_ps(c.as_mut_ptr().add(7 * ldc), c70);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_avx2_dot_product() {
        #[cfg(target_feature = "avx2")]
        {
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
            
            let result = unsafe { dot_product_avx2_f32(&a, &b) };
            let expected = 165.0; // 1*9 + 2*8 + ... + 9*1
            
            assert!((result - expected).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_avx2_relu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -6.0];
        let mut output = vec![0.0; 9];
        
        #[cfg(target_feature = "avx2")]
        unsafe {
            apply_relu_avx2_f32(&input, &mut output).unwrap();
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            apply_relu_scalar_f32(&input, &mut output).unwrap();
        }
        
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0];
        for (actual, &expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}