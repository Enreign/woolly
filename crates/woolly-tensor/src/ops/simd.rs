//! SIMD utilities for CPU operations

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use std::arch::aarch64::*;

use crate::backend::Result;

/// Trait for SIMD operations on vector types
pub trait SimdOps: Copy + Clone + Send + Sync {
    /// The scalar type (e.g., f32, i32)
    type Scalar: Copy + Send + Sync;
    /// The SIMD vector type (e.g., __m256, float32x4_t)
    type Vector: Copy + Send + Sync;
    
    /// Loads unaligned data from memory
    unsafe fn load(ptr: *const Self::Scalar) -> Self::Vector;
    
    /// Loads aligned data from memory
    unsafe fn load_aligned(ptr: *const Self::Scalar) -> Self::Vector;
    
    /// Stores data to memory (unaligned)
    unsafe fn store(ptr: *mut Self::Scalar, vec: Self::Vector);
    
    /// Stores data to memory (aligned)
    unsafe fn store_aligned(ptr: *mut Self::Scalar, vec: Self::Vector);
    
    /// Vector addition
    fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    
    /// Vector subtraction
    fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    
    /// Vector multiplication
    fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    
    /// Vector division
    fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    
    /// Fused multiply-add: a * b + c
    fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;
    
    /// Horizontal addition (sum all elements)
    fn hadd(vec: Self::Vector) -> Self::Scalar;
    
    /// Element-wise maximum
    fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    
    /// Element-wise minimum
    fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    
    /// Square root
    fn sqrt(vec: Self::Vector) -> Self::Vector;
    
    /// Reciprocal approximation
    fn reciprocal(vec: Self::Vector) -> Self::Vector;
    
    /// Broadcast scalar to vector
    fn splat(value: Self::Scalar) -> Self::Vector;
    
    /// Zero vector
    fn zero() -> Self::Vector;
    
    /// Returns the number of elements in a vector
    const LANES: usize;
    
    /// Returns the alignment requirement in bytes
    const ALIGN: usize;
}

/// Trait for operations that can be vectorized
pub trait Vectorizable {
    /// Apply operation using SIMD when possible
    fn apply_simd<S: SimdOps>(input: &[S::Scalar], output: &mut [S::Scalar]) -> Result<()>;
    
    /// Apply operation using scalar fallback
    fn apply_scalar<T>(input: &[T], output: &mut [T]) -> Result<()>;
}

/// Trait for binary operations that can be vectorized  
pub trait BinaryVectorizable {
    /// Apply binary operation using SIMD when possible
    fn apply_simd<S: SimdOps>(a: &[S::Scalar], b: &[S::Scalar], output: &mut [S::Scalar]) -> Result<()>;
    
    /// Apply binary operation using scalar fallback
    fn apply_scalar<T>(a: &[T], b: &[T], output: &mut [T]) -> Result<()>;
}

/// SIMD-accelerated operations for f32
pub struct SimdF32;

impl SimdF32 {
    /// Adds two f32 slices using SIMD when available
    #[inline]
    pub fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { Self::add_avx2(a, b, out) }
                return;
            } else if is_x86_feature_detected!("sse2") {
                unsafe { Self::add_sse2(a, b, out) }
                return;
            }
        }
        
        // Fallback to scalar
        Self::add_scalar(a, b, out);
    }
    
    /// Scalar fallback for addition
    #[inline]
    fn add_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
    }
    
    /// SSE2 implementation of addition
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2")]
    unsafe fn add_sse2(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = a.len();
        let simd_len = len - (len % 4);
        
        // Process 4 elements at a time
        for i in (0..simd_len).step_by(4) {
            let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
            let result = _mm_add_ps(a_vec, b_vec);
            _mm_storeu_ps(out.as_mut_ptr().add(i), result);
        }
        
        // Handle remaining elements
        for i in simd_len..len {
            out[i] = a[i] + b[i];
        }
    }
    
    /// AVX2 implementation of addition
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = a.len();
        let simd_len = len - (len % 8);
        
        // Process 8 elements at a time
        for i in (0..simd_len).step_by(8) {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            let result = _mm256_add_ps(a_vec, b_vec);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), result);
        }
        
        // Handle remaining elements
        for i in simd_len..len {
            out[i] = a[i] + b[i];
        }
    }
    
    /// Multiplies two f32 slices using SIMD when available
    #[inline]
    pub fn mul(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { Self::mul_avx2(a, b, out) }
                return;
            } else if is_x86_feature_detected!("sse2") {
                unsafe { Self::mul_sse2(a, b, out) }
                return;
            }
        }
        
        // Fallback to scalar
        Self::mul_scalar(a, b, out);
    }
    
    /// Scalar fallback for multiplication
    #[inline]
    fn mul_scalar(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] * b[i];
        }
    }
    
    /// SSE2 implementation of multiplication
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2")]
    unsafe fn mul_sse2(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = a.len();
        let simd_len = len - (len % 4);
        
        for i in (0..simd_len).step_by(4) {
            let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
            let result = _mm_mul_ps(a_vec, b_vec);
            _mm_storeu_ps(out.as_mut_ptr().add(i), result);
        }
        
        for i in simd_len..len {
            out[i] = a[i] * b[i];
        }
    }
    
    /// AVX2 implementation of multiplication
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = a.len();
        let simd_len = len - (len % 8);
        
        for i in (0..simd_len).step_by(8) {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            let result = _mm256_mul_ps(a_vec, b_vec);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), result);
        }
        
        for i in simd_len..len {
            out[i] = a[i] * b[i];
        }
    }
    
    /// Applies ReLU using SIMD when available
    #[inline]
    pub fn relu(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { Self::relu_avx2(input, output) }
                return;
            } else if is_x86_feature_detected!("sse2") {
                unsafe { Self::relu_sse2(input, output) }
                return;
            }
        }
        
        // Fallback to scalar
        Self::relu_scalar(input, output);
    }
    
    /// Scalar fallback for ReLU
    #[inline]
    fn relu_scalar(input: &[f32], output: &mut [f32]) {
        for i in 0..input.len() {
            output[i] = input[i].max(0.0);
        }
    }
    
    /// SSE2 implementation of ReLU
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2")]
    unsafe fn relu_sse2(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len - (len % 4);
        let zero = _mm_setzero_ps();
        
        for i in (0..simd_len).step_by(4) {
            let x = _mm_loadu_ps(input.as_ptr().add(i));
            let result = _mm_max_ps(x, zero);
            _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        }
        
        for i in simd_len..len {
            output[i] = input[i].max(0.0);
        }
    }
    
    /// AVX2 implementation of ReLU
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2")]
    unsafe fn relu_avx2(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let simd_len = len - (len % 8);
        let zero = _mm256_setzero_ps();
        
        for i in (0..simd_len).step_by(8) {
            let x = _mm256_loadu_ps(input.as_ptr().add(i));
            let result = _mm256_max_ps(x, zero);
            _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        }
        
        for i in simd_len..len {
            output[i] = input[i].max(0.0);
        }
    }
    
    /// Computes dot product using SIMD when available
    #[inline]
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::dot_product_avx2(a, b) };
            } else if is_x86_feature_detected!("sse2") {
                return unsafe { Self::dot_product_sse2(a, b) };
            }
        }
        
        // Fallback to scalar
        Self::dot_product_scalar(a, b)
    }
    
    /// Scalar fallback for dot product
    #[inline]
    fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }
    
    /// SSE2 implementation of dot product
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2")]
    unsafe fn dot_product_sse2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let simd_len = len - (len % 4);
        let mut sum = _mm_setzero_ps();
        
        for i in (0..simd_len).step_by(4) {
            let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
            let prod = _mm_mul_ps(a_vec, b_vec);
            sum = _mm_add_ps(sum, prod);
        }
        
        // Horizontal sum
        let sum_array = std::mem::transmute::<__m128, [f32; 4]>(sum);
        let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        // Handle remaining elements
        for i in simd_len..len {
            result += a[i] * b[i];
        }
        
        result
    }
    
    /// AVX2 implementation of dot product
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let simd_len = len - (len % 8);
        let mut sum = _mm256_setzero_ps();
        
        for i in (0..simd_len).step_by(8) {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            let prod = _mm256_mul_ps(a_vec, b_vec);
            sum = _mm256_add_ps(sum, prod);
        }
        
        // Horizontal sum
        let sum_high = _mm256_extractf128_ps(sum, 1);
        let sum_low = _mm256_castps256_ps128(sum);
        let sum_128 = _mm_add_ps(sum_low, sum_high);
        
        let sum_array = std::mem::transmute::<__m128, [f32; 4]>(sum_128);
        let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        // Handle remaining elements
        for i in simd_len..len {
            result += a[i] * b[i];
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut out = vec![0.0; 9];
        
        SimdF32::add(&a, &b, &mut out);
        
        for i in 0..9 {
            assert_eq!(out[i], 10.0);
        }
    }
    
    #[test]
    fn test_simd_mul() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let mut out = vec![0.0; 5];
        
        SimdF32::mul(&a, &b, &mut out);
        
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }
    
    #[test]
    fn test_simd_relu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];
        
        SimdF32::relu(&input, &mut output);
        
        assert_eq!(output, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }
    
    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        
        let result = SimdF32::dot_product(&a, &b);
        
        assert_eq!(result, 20.0);
    }
}