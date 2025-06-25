//! MLX tensor operations
//!
//! This module provides implementations for various tensor operations using MLX.
//! Operations are organized into categories for better maintainability.

pub mod element_wise;
pub mod matmul;
pub mod quantization;
pub mod reduction;
pub mod shape_ops;

use crate::backend::MLXBackend;
use crate::error::Result;
use crate::storage::MLXStorage;

/// Trait for MLX operations that can be performed on tensors
pub trait MLXOperation {
    type Input;
    type Output;
    
    /// Execute the operation
    fn execute(&self, backend: &MLXBackend, input: Self::Input) -> Result<Self::Output>;
}

/// Matrix multiplication operation
pub struct MatMulOp;

/// Element-wise operation
pub struct ElementWiseOp;

/// Reduction operation
pub struct ReductionOp;

/// Quantization operation
pub struct QuantizationOp;

/// Shape manipulation operation
pub struct ShapeOp;

/// Common utilities for MLX operations
pub mod utils {
    use crate::backend::MLXBackend;
    use crate::device::Device;
    use crate::error::{MLXError, Result};
    use crate::storage::MLXStorage;
    use woolly_tensor::shape::Shape;
    use tracing::trace;
    
    /// Check if tensors are on compatible devices
    pub fn check_device_compatibility<T>(
        lhs: &MLXStorage<T>,
        rhs: &MLXStorage<T>,
    ) -> Result<Device>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let lhs_device = match lhs.device() {
            woolly_tensor::backend::Device::Metal => Device::GPU,
            woolly_tensor::backend::Device::Cpu => Device::CPU,
            _ => return Err(MLXError::InvalidDevice("Unsupported device".to_string())),
        };
        
        let rhs_device = match rhs.device() {
            woolly_tensor::backend::Device::Metal => Device::GPU,
            woolly_tensor::backend::Device::Cpu => Device::CPU,
            _ => return Err(MLXError::InvalidDevice("Unsupported device".to_string())),
        };
        
        // For unified memory, both devices can work together
        // Prefer GPU if available
        if lhs_device == Device::GPU || rhs_device == Device::GPU {
            Ok(Device::GPU)
        } else {
            Ok(Device::CPU)
        }
    }
    
    /// Validate tensor shapes for element-wise operations
    pub fn validate_elementwise_shapes(
        lhs_shape: &Shape,
        rhs_shape: &Shape,
    ) -> Result<Shape> {
        if lhs_shape.as_slice() == rhs_shape.as_slice() {
            Ok(lhs_shape.clone())
        } else if lhs_shape.is_broadcast_compatible(rhs_shape) {
            lhs_shape.broadcast_shape(rhs_shape)
                .map_err(|e| MLXError::ShapeMismatch {
                    expected: lhs_shape.as_slice().to_vec(),
                    actual: rhs_shape.as_slice().to_vec(),
                })
        } else {
            Err(MLXError::ShapeMismatch {
                expected: lhs_shape.as_slice().to_vec(),
                actual: rhs_shape.as_slice().to_vec(),
            })
        }
    }
    
    /// Calculate output shape for matrix multiplication
    pub fn matmul_output_shape(lhs_shape: &Shape, rhs_shape: &Shape) -> Result<Shape> {
        if lhs_shape.ndim() < 2 || rhs_shape.ndim() < 2 {
            return Err(MLXError::ArrayOperationFailed(
                "Matrix multiplication requires at least 2D tensors".to_string()
            ));
        }
        
        let lhs_rows = lhs_shape[lhs_shape.ndim() - 2];
        let lhs_cols = lhs_shape[lhs_shape.ndim() - 1];
        let rhs_rows = rhs_shape[rhs_shape.ndim() - 2];
        let rhs_cols = rhs_shape[rhs_shape.ndim() - 1];
        
        if lhs_cols != rhs_rows {
            return Err(MLXError::ShapeMismatch {
                expected: vec![lhs_rows, lhs_cols],
                actual: vec![rhs_rows, rhs_cols],
            });
        }
        
        // Handle batch dimensions
        let mut output_shape = Vec::new();
        
        // Add batch dimensions (broadcast if necessary)
        let lhs_batch_dims = &lhs_shape.as_slice()[..lhs_shape.ndim() - 2];
        let rhs_batch_dims = &rhs_shape.as_slice()[..rhs_shape.ndim() - 2];
        
        let max_batch_dims = lhs_batch_dims.len().max(rhs_batch_dims.len());
        for i in 0..max_batch_dims {
            let lhs_dim = if i < lhs_batch_dims.len() {
                lhs_batch_dims[lhs_batch_dims.len() - 1 - i]
            } else {
                1
            };
            let rhs_dim = if i < rhs_batch_dims.len() {
                rhs_batch_dims[rhs_batch_dims.len() - 1 - i]
            } else {
                1
            };
            
            if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
                return Err(MLXError::ShapeMismatch {
                    expected: lhs_shape.as_slice().to_vec(),
                    actual: rhs_shape.as_slice().to_vec(),
                });
            }
            
            output_shape.push(lhs_dim.max(rhs_dim));
        }
        
        // Reverse batch dimensions (we built them backwards)
        output_shape.reverse();
        
        // Add matrix dimensions
        output_shape.push(lhs_rows);
        output_shape.push(rhs_cols);
        
        Ok(Shape::from_slice(&output_shape))
    }
    
    /// Calculate output shape for reduction operations
    pub fn reduction_output_shape(
        input_shape: &Shape,
        axes: &[usize],
        keep_dims: bool,
    ) -> Result<Shape> {
        let mut output_dims = Vec::new();
        
        for (dim_idx, &dim_size) in input_shape.as_slice().iter().enumerate() {
            if axes.contains(&dim_idx) {
                if keep_dims {
                    output_dims.push(1);
                }
                // Otherwise, dimension is removed
            } else {
                output_dims.push(dim_size);
            }
        }
        
        // Handle scalar result
        if output_dims.is_empty() {
            output_dims.push(1); // Scalar as 1-element tensor
        }
        
        Ok(Shape::from_slice(&output_dims))
    }
    
    /// Choose optimal device for operation based on tensor sizes and types
    pub fn choose_optimal_device(
        tensors: &[&Shape],
        operation_type: &str,
    ) -> Device {
        let total_elements: usize = tensors.iter().map(|s| s.numel()).sum();
        
        match operation_type {
            "matmul" | "conv" => {
                // Always prefer GPU for compute-intensive operations
                Device::GPU
            }
            "element_wise" => {
                // Use GPU for large tensors, CPU for small ones
                if total_elements > 10_000 {
                    Device::GPU
                } else {
                    Device::CPU
                }
            }
            "reduction" => {
                // GPU is usually better for reductions
                if total_elements > 1_000 {
                    Device::GPU
                } else {
                    Device::CPU
                }
            }
            "copy" | "slice" | "reshape" => {
                // Memory operations benefit from unified memory
                Device::CPU
            }
            _ => {
                // Default to GPU for unknown operations
                Device::GPU
            }
        }
    }
    
    /// Synchronize operation if needed
    pub fn maybe_synchronize(backend: &MLXBackend, sync: bool) -> Result<()> {
        if sync {
            backend.synchronize()?;
        }
        Ok(())
    }
    
    /// Log operation performance
    pub fn log_operation_perf(
        operation: &str,
        input_shapes: &[&Shape],
        output_shape: &Shape,
        duration_ms: f64,
    ) {
        trace!(
            "MLX operation '{}' completed: inputs={:?} -> output={:?} in {:.2}ms",
            operation,
            input_shapes,
            output_shape,
            duration_ms
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::utils::*;
    use woolly_tensor::shape::Shape;
    
    #[test]
    fn test_elementwise_shape_validation() {
        let shape1 = Shape::from_slice(&[2, 3]);
        let shape2 = Shape::from_slice(&[2, 3]);
        
        match validate_elementwise_shapes(&shape1, &shape2) {
            Ok(result_shape) => {
                assert_eq!(result_shape.as_slice(), &[2, 3]);
                println!("Element-wise shape validation passed");
            }
            Err(e) => {
                panic!("Element-wise shape validation failed: {}", e);
            }
        }
    }
    
    #[test]
    fn test_matmul_output_shape() {
        let lhs_shape = Shape::from_slice(&[2, 3]);
        let rhs_shape = Shape::from_slice(&[3, 4]);
        
        match matmul_output_shape(&lhs_shape, &rhs_shape) {
            Ok(result_shape) => {
                assert_eq!(result_shape.as_slice(), &[2, 4]);
                println!("Matrix multiplication shape calculation passed");
            }
            Err(e) => {
                panic!("Matrix multiplication shape calculation failed: {}", e);
            }
        }
    }
    
    #[test]
    fn test_reduction_output_shape() {
        let input_shape = Shape::from_slice(&[2, 3, 4]);
        let axes = vec![1]; // Reduce along second dimension
        
        match reduction_output_shape(&input_shape, &axes, false) {
            Ok(result_shape) => {
                assert_eq!(result_shape.as_slice(), &[2, 4]);
                println!("Reduction shape calculation passed");
            }
            Err(e) => {
                panic!("Reduction shape calculation failed: {}", e);
            }
        }
        
        // Test with keep_dims
        match reduction_output_shape(&input_shape, &axes, true) {
            Ok(result_shape) => {
                assert_eq!(result_shape.as_slice(), &[2, 1, 4]);
                println!("Reduction shape calculation with keep_dims passed");
            }
            Err(e) => {
                panic!("Reduction shape calculation with keep_dims failed: {}", e);
            }
        }
    }
    
    #[test]
    fn test_device_selection() {
        let small_shape = Shape::from_slice(&[10, 10]);
        let large_shape = Shape::from_slice(&[1000, 1000]);
        
        // Small tensor should prefer CPU for element-wise ops
        let device = choose_optimal_device(&[&small_shape], "element_wise");
        assert_eq!(device, Device::CPU);
        
        // Large tensor should prefer GPU for element-wise ops
        let device = choose_optimal_device(&[&large_shape], "element_wise");
        assert_eq!(device, Device::GPU);
        
        // Matrix multiplication should always prefer GPU
        let device = choose_optimal_device(&[&small_shape, &small_shape], "matmul");
        assert_eq!(device, Device::GPU);
        
        println!("Device selection tests passed");
    }
}