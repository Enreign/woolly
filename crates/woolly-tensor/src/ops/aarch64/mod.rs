//! AArch64 NEON SIMD implementations

use crate::ops::SimdOps;

/// NEON implementation for f32
#[derive(Debug, Clone, Copy)]
pub struct NeonF32;

#[cfg(target_arch = "aarch64")]
impl SimdOps for NeonF32 {
    type Scalar = f32;
    type Vector = std::arch::aarch64::float32x4_t;
    
    const LANES: usize = 4;
    const ALIGN: usize = 16;
    
    #[inline]
    unsafe fn load(ptr: *const Self::Scalar) -> Self::Vector {
        std::arch::aarch64::vld1q_f32(ptr)
    }
    
    #[inline]
    unsafe fn load_aligned(ptr: *const Self::Scalar) -> Self::Vector {
        // NEON doesn't distinguish between aligned and unaligned loads
        std::arch::aarch64::vld1q_f32(ptr)
    }
    
    #[inline]
    unsafe fn store(ptr: *mut Self::Scalar, vec: Self::Vector) {
        std::arch::aarch64::vst1q_f32(ptr, vec)
    }
    
    #[inline]
    unsafe fn store_aligned(ptr: *mut Self::Scalar, vec: Self::Vector) {
        std::arch::aarch64::vst1q_f32(ptr, vec)
    }
    
    #[inline]
    fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vaddq_f32(a, b) }
    }
    
    #[inline]
    fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vsubq_f32(a, b) }
    }
    
    #[inline]
    fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vmulq_f32(a, b) }
    }
    
    #[inline]
    fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vdivq_f32(a, b) }
    }
    
    #[inline]
    fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vfmaq_f32(c, a, b) }
    }
    
    #[inline]
    fn hadd(vec: Self::Vector) -> Self::Scalar {
        unsafe {
            let sum = std::arch::aarch64::vaddvq_f32(vec);
            sum
        }
    }
    
    #[inline]
    fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vmaxq_f32(a, b) }
    }
    
    #[inline]
    fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vminq_f32(a, b) }
    }
    
    #[inline]
    fn sqrt(vec: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vsqrtq_f32(vec) }
    }
    
    #[inline]
    fn reciprocal(vec: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vrecpeq_f32(vec) }
    }
    
    #[inline]
    fn splat(value: Self::Scalar) -> Self::Vector {
        unsafe { std::arch::aarch64::vdupq_n_f32(value) }
    }
    
    #[inline]
    fn zero() -> Self::Vector {
        Self::splat(0.0)
    }
}

/// NEON implementation for f16
#[derive(Debug, Clone, Copy)]
pub struct NeonF16;

// F16 implementation would go here when half crate support is added

/// NEON implementation for i8
#[derive(Debug, Clone, Copy)]
pub struct NeonI8;

// I8 implementation would go here