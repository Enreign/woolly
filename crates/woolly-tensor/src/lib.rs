//! Woolly Tensor: High-performance tensor operations library
//!
//! This crate provides efficient tensor operations with support for multiple backends
//! including CPU (with SIMD acceleration), CUDA, and Metal.
//!
//! # Features
//!
//! - **Multiple Backends**: CPU, CUDA, and Metal support
//! - **SIMD Acceleration**: Optimized operations using AVX2/AVX-512 on x86_64 and NEON on ARM
//! - **Quantization**: Support for various quantization schemes including llama.cpp compatible formats
//! - **Zero-copy Views**: Efficient tensor slicing and broadcasting
//! - **Type Safety**: Strong typing with compile-time shape checking where possible
//!
//! # Example
//!
//! ```rust,no_run
//! use woolly_tensor::{Shape, CpuBackend, DType};
//! use woolly_tensor::ops::*;
//!
//! // Basic tensor operations using the CPU backend
//! let backend = CpuBackend::new();
//! let shape = Shape::matrix(2, 3);
//! 
//! // Example of using tensor operations directly
//! let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
//! let mut result = vec![0.0; 6];
//! 
//! // Perform element-wise addition
//! Add::apply_f32(&a, &b, &mut result)?;
//! 
//! // Apply ReLU activation
//! let input = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
//! let mut output = vec![0.0; 5];
//! ReLU::apply_f32(&input, &mut output)?;
//! 
//! # Ok::<(), woolly_tensor::TensorError>(())
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::new_without_default)]

pub mod backend;
pub mod tensor;
pub mod shape;
pub mod quantization;
pub mod ops;
pub mod validation;
pub mod memory_pool;

// Re-export main types
pub use backend::{
    TensorBackend, TensorStorage, TensorError, Result, DType, Device,
    CpuBackend,
};

#[cfg(feature = "cuda")]
pub use backend::CudaBackend;

#[cfg(feature = "metal")]
pub use backend::MetalBackend;

pub use tensor::{Tensor, TensorBuilder};
pub use shape::{Shape, Strides};
pub use quantization::{
    QuantizationScheme, Quantizer, Dequantizer,
    Int8Quantizer, Q4_0Quantizer, Q4_1Quantizer, Q8_0Quantizer,
};
pub use memory_pool::{
    MemoryPool, MemoryPoolConfig, MemoryPoolStats, PooledTensorStorage,
    global_pool, init_global_pool,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        Tensor, TensorBuilder, Shape, Strides,
        TensorBackend, DType, Device, Result,
        CpuBackend,
        validation::TensorValidator,
        memory_pool::{MemoryPool, MemoryPoolConfig, global_pool},
    };
    
    #[cfg(feature = "cuda")]
    pub use crate::CudaBackend;
    
    #[cfg(feature = "metal")]
    pub use crate::MetalBackend;
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_tensor_creation() {
        let backend = CpuBackend::new();
        let shape = Shape::vector(10);
        
        // This would fail without proper backend implementation
        // For now, we're just testing that the types compile correctly
        let _dtype = DType::F32;
        let _device = Device::Cpu;
    }
    
    #[test]
    fn test_shape_operations() {
        let shape1 = Shape::matrix(3, 4);
        assert_eq!(shape1.ndim(), 2);
        assert_eq!(shape1.numel(), 12);
        
        let shape2 = Shape::from_slice(&[2, 3, 4]);
        assert_eq!(shape2.ndim(), 3);
        assert_eq!(shape2.numel(), 24);
    }
    
    #[test]
    fn test_dtype_properties() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::I64.size_in_bytes(), 8);
        
        assert!(DType::F32.is_float());
        assert!(!DType::I32.is_float());
    }
}