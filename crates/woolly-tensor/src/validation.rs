//! Tensor validation utilities
//!
//! This module provides comprehensive validation for tensor operations,
//! dimensions, shapes, and data types to ensure robust error handling.

use crate::{Shape, DType, TensorError, Result};

/// Tensor validation utilities
pub struct TensorValidator;

impl TensorValidator {
    /// Validate tensor shapes for binary operations
    pub fn validate_binary_op_shapes(
        left_shape: &Shape,
        right_shape: &Shape,
        operation: &str,
    ) -> Result<()> {
        // Check if shapes are compatible for element-wise operations
        if !Self::are_shapes_broadcastable(left_shape, right_shape) {
            return Err(TensorError::incompatible_shapes(
                "TENSOR_BINARY_OP_INCOMPATIBLE",
                format!("Shapes are not compatible for {}", operation),
                operation,
                format!("{:?}", left_shape.dims()),
                format!("{:?}", right_shape.dims()),
                "Use tensors with compatible shapes or apply broadcasting rules"
            ));
        }
        Ok(())
    }
    
    /// Check if two shapes are broadcastable
    pub fn are_shapes_broadcastable(left: &Shape, right: &Shape) -> bool {
        let left_dims = left.dims();
        let right_dims = right.dims();
        
        // Pad with 1s from the left to make shapes same length
        let max_len = left_dims.len().max(right_dims.len());
        let mut left_padded = vec![1; max_len - left_dims.len()];
        left_padded.extend_from_slice(left_dims);
        let mut right_padded = vec![1; max_len - right_dims.len()];
        right_padded.extend_from_slice(right_dims);
        
        // Check broadcasting rules: dimensions must be equal or one of them must be 1
        for (l, r) in left_padded.iter().zip(right_padded.iter()) {
            if *l != *r && *l != 1 && *r != 1 {
                return false;
            }
        }
        true
    }
    
    /// Validate matrix multiplication shapes
    pub fn validate_matmul_shapes(
        left_shape: &Shape,
        right_shape: &Shape,
    ) -> Result<Shape> {
        let left_dims = left_shape.dims();
        let right_dims = right_shape.dims();
        
        // Both tensors must be at least 2D
        if left_dims.len() < 2 {
            return Err(TensorError::invalid_shape(
                "MATMUL_LEFT_TOO_FEW_DIMS",
                "Left tensor must have at least 2 dimensions for matrix multiplication",
                format!("{:?}", left_dims),
                "matrix multiplication",
                format!("Has {} dimensions, needs at least 2", left_dims.len()),
                "Use a tensor with shape [M, K] or batch dimensions + [M, K]"
            ));
        }
        
        if right_dims.len() < 2 {
            return Err(TensorError::invalid_shape(
                "MATMUL_RIGHT_TOO_FEW_DIMS",
                "Right tensor must have at least 2 dimensions for matrix multiplication",
                format!("{:?}", right_dims),
                "matrix multiplication",
                format!("Has {} dimensions, needs at least 2", right_dims.len()),
                "Use a tensor with shape [K, N] or batch dimensions + [K, N]"
            ));
        }
        
        // Extract matrix dimensions (last 2 dimensions)
        let left_rows = left_dims[left_dims.len() - 2];
        let left_cols = left_dims[left_dims.len() - 1];
        let right_rows = right_dims[right_dims.len() - 2];
        let right_cols = right_dims[right_dims.len() - 1];
        
        // Inner dimensions must match
        if left_cols != right_rows {
            return Err(TensorError::incompatible_shapes(
                "MATMUL_INNER_DIM_MISMATCH",
                "Inner dimensions must match for matrix multiplication",
                "matrix multiplication",
                format!("[..., {}, {}]", left_rows, left_cols),
                format!("[..., {}, {}]", right_rows, right_cols),
                format!("Ensure left tensor's last dimension ({}) equals right tensor's second-to-last dimension ({})", left_cols, right_rows)
            ));
        }
        
        // Check batch dimensions if present
        if left_dims.len() > 2 || right_dims.len() > 2 {
            let left_batch = &left_dims[..left_dims.len().saturating_sub(2)];
            let right_batch = &right_dims[..right_dims.len().saturating_sub(2)];
            
            // Batch dimensions must be broadcastable
            let left_batch_shape = Shape::from_slice(left_batch);
            let right_batch_shape = Shape::from_slice(right_batch);
            
            if !Self::are_shapes_broadcastable(&left_batch_shape, &right_batch_shape) {
                return Err(TensorError::incompatible_shapes(
                    "MATMUL_BATCH_DIM_MISMATCH",
                    "Batch dimensions are not broadcastable for matrix multiplication",
                    "matrix multiplication",
                    format!("{:?}", left_dims),
                    format!("{:?}", right_dims),
                    "Ensure batch dimensions follow broadcasting rules"
                ));
            }
        }
        
        // Calculate output shape
        let max_batch_len = left_dims.len().max(right_dims.len()) - 2;
        let mut output_dims = Vec::with_capacity(max_batch_len + 2);
        
        // Broadcast batch dimensions
        for i in 0..max_batch_len {
            let left_idx = left_dims.len().saturating_sub(max_batch_len + 2) + i;
            let right_idx = right_dims.len().saturating_sub(max_batch_len + 2) + i;
            
            let left_dim = if left_idx < left_dims.len() { left_dims[left_idx] } else { 1 };
            let right_dim = if right_idx < right_dims.len() { right_dims[right_idx] } else { 1 };
            
            output_dims.push(left_dim.max(right_dim));
        }
        
        // Add matrix dimensions
        output_dims.push(left_rows);
        output_dims.push(right_cols);
        
        Ok(Shape::from_slice(&output_dims))
    }
    
    /// Validate reduction operation
    pub fn validate_reduction(
        input_shape: &Shape,
        axes: &[usize],
        keep_dims: bool,
    ) -> Result<Shape> {
        let input_dims = input_shape.dims();
        let ndim = input_dims.len();
        
        // Check that all axes are valid
        for &axis in axes {
            if axis >= ndim {
                return Err(TensorError::out_of_bounds(
                    "REDUCTION_AXIS_OUT_OF_BOUNDS",
                    format!("Reduction axis {} is out of bounds", axis),
                    axis,
                    0,
                    ndim,
                    "reduction operation",
                    format!("Use axis values between 0 and {} (inclusive)", ndim - 1)
                ));
            }
        }
        
        // Check for duplicate axes
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();
        sorted_axes.dedup();
        if sorted_axes.len() != axes.len() {
            return Err(TensorError::invalid_shape(
                "REDUCTION_DUPLICATE_AXES",
                "Duplicate axes found in reduction operation",
                format!("{:?}", axes),
                "reduction operation",
                "Contains duplicate axis values",
                "Remove duplicate axes from the reduction"
            ));
        }
        
        // Calculate output shape
        let mut output_dims = Vec::new();
        for (i, &dim) in input_dims.iter().enumerate() {
            if axes.contains(&i) {
                if keep_dims {
                    output_dims.push(1);
                }
                // Otherwise, dimension is removed
            } else {
                output_dims.push(dim);
            }
        }
        
        // Ensure at least 1D output
        if output_dims.is_empty() {
            output_dims.push(1);
        }
        
        Ok(Shape::from_slice(&output_dims))
    }
    
    /// Validate reshape operation
    pub fn validate_reshape(
        input_shape: &Shape,
        target_shape: &Shape,
    ) -> Result<()> {
        let input_numel = input_shape.numel();
        let target_numel = target_shape.numel();
        
        if input_numel != target_numel {
            return Err(TensorError::invalid_shape(
                "RESHAPE_SIZE_MISMATCH",
                "Total number of elements must remain the same during reshape",
                format!("{:?} -> {:?}", input_shape.dims(), target_shape.dims()),
                "reshape operation",
                format!("Input has {} elements, target has {}", input_numel, target_numel),
                "Ensure the product of target dimensions equals the original tensor size"
            ));
        }
        
        // Check for valid dimensions (no zeros except as placeholders)
        for (i, &dim) in target_shape.dims().iter().enumerate() {
            if dim == 0 {
                return Err(TensorError::invalid_shape(
                    "RESHAPE_ZERO_DIMENSION",
                    "Reshape target cannot contain zero dimensions",
                    format!("{:?}", target_shape.dims()),
                    "reshape operation",
                    format!("Dimension {} is zero", i),
                    "Use positive integers for all target dimensions"
                ));
            }
        }
        
        Ok(())
    }
    
    /// Validate transpose operation
    pub fn validate_transpose(
        input_shape: &Shape,
        axes: &[usize],
    ) -> Result<Shape> {
        let input_dims = input_shape.dims();
        let ndim = input_dims.len();
        
        // Axes must have the same length as input dimensions
        if axes.len() != ndim {
            return Err(TensorError::invalid_shape(
                "TRANSPOSE_AXES_LENGTH_MISMATCH",
                "Number of transpose axes must match tensor dimensions",
                format!("{:?}", input_dims),
                "transpose operation",
                format!("Tensor has {} dimensions but {} axes provided", ndim, axes.len()),
                format!("Provide exactly {} axes for the transpose", ndim)
            ));
        }
        
        // All axes must be unique and within bounds
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();
        
        for (i, &axis) in sorted_axes.iter().enumerate() {
            if axis >= ndim {
                return Err(TensorError::out_of_bounds(
                    "TRANSPOSE_AXIS_OUT_OF_BOUNDS",
                    format!("Transpose axis {} is out of bounds", axis),
                    axis,
                    0,
                    ndim,
                    "transpose operation",
                    format!("Use axis values between 0 and {} (inclusive)", ndim - 1)
                ));
            }
            
            if i > 0 && axis == sorted_axes[i - 1] {
                return Err(TensorError::invalid_shape(
                    "TRANSPOSE_DUPLICATE_AXES",
                    "Duplicate axes found in transpose operation",
                    format!("{:?}", axes),
                    "transpose operation",
                    format!("Axis {} appears multiple times", axis),
                    "Each axis should appear exactly once in the transpose"
                ));
            }
        }
        
        // Check that we have all axes 0..ndim-1
        for i in 0..ndim {
            if !axes.contains(&i) {
                return Err(TensorError::invalid_shape(
                    "TRANSPOSE_MISSING_AXIS",
                    "Missing axis in transpose operation",
                    format!("{:?}", axes),
                    "transpose operation",
                    format!("Axis {} is missing", i),
                    format!("Include all axes 0 through {} in the transpose", ndim - 1)
                ));
            }
        }
        
        // Calculate output shape
        let mut output_dims = vec![0; ndim];
        for (i, &axis) in axes.iter().enumerate() {
            output_dims[i] = input_dims[axis];
        }
        
        Ok(Shape::from_slice(&output_dims))
    }
    
    /// Validate data type compatibility
    pub fn validate_dtype_compatibility(
        left_dtype: DType,
        right_dtype: DType,
        operation: &str,
    ) -> Result<DType> {
        match (left_dtype, right_dtype) {
            // Same types are always compatible
            (l, r) if l == r => Ok(l),
            
            // Float operations
            (DType::F64, DType::F32) | (DType::F32, DType::F64) => Ok(DType::F64),
            (DType::F64, DType::F16) | (DType::F16, DType::F64) => Ok(DType::F64),
            (DType::F32, DType::F16) | (DType::F16, DType::F32) => Ok(DType::F32),
            
            // Mixed float-int operations (promote to float)
            (DType::F64, DType::I64) | (DType::I64, DType::F64) => Ok(DType::F64),
            (DType::F64, DType::I32) | (DType::I32, DType::F64) => Ok(DType::F64),
            (DType::F32, DType::I64) | (DType::I64, DType::F32) => Ok(DType::F32),
            (DType::F32, DType::I32) | (DType::I32, DType::F32) => Ok(DType::F32),
            (DType::F16, DType::I32) | (DType::I32, DType::F16) => Ok(DType::F32),
            
            // Integer operations
            (DType::I64, DType::I32) | (DType::I32, DType::I64) => Ok(DType::I64),
            
            // Quantized types - generally not compatible with others
            (DType::Quantized(_), _) | (_, DType::Quantized(_)) => {
                Err(TensorError::DataTypeError {
                    code: "DTYPE_QUANTIZED_INCOMPATIBLE",
                    message: "Quantized tensors cannot be used in mixed-type operations".to_string(),
                    from_dtype: format!("{:?}", left_dtype),
                    to_dtype: format!("{:?}", right_dtype),
                    operation: operation.to_string(),
                    suggestion: "Dequantize tensors before performing operations or use quantization-aware operations".to_string(),
                })
            }
            
            // Unsupported combinations
            _ => {
                Err(TensorError::DataTypeError {
                    code: "DTYPE_INCOMPATIBLE",
                    message: format!("Data types {:?} and {:?} are not compatible for {}", left_dtype, right_dtype, operation),
                    from_dtype: format!("{:?}", left_dtype),
                    to_dtype: format!("{:?}", right_dtype),
                    operation: operation.to_string(),
                    suggestion: "Convert tensors to compatible data types before performing the operation".to_string(),
                })
            }
        }
    }
    
    /// Validate tensor indexing bounds
    pub fn validate_indexing(
        shape: &Shape,
        indices: &[usize],
    ) -> Result<()> {
        let dims = shape.dims();
        
        if indices.len() > dims.len() {
            return Err(TensorError::invalid_shape(
                "INDEX_TOO_MANY_DIMS",
                "Too many indices for tensor indexing",
                format!("{:?}", dims),
                "tensor indexing",
                format!("Tensor has {} dimensions but {} indices provided", dims.len(), indices.len()),
                format!("Provide at most {} indices", dims.len())
            ));
        }
        
        for (dim_idx, &index) in indices.iter().enumerate() {
            let dim_size = dims[dim_idx];
            if index >= dim_size {
                return Err(TensorError::out_of_bounds(
                    "INDEX_OUT_OF_BOUNDS",
                    format!("Index {} is out of bounds for dimension {}", index, dim_idx),
                    index,
                    dim_idx,
                    dim_size,
                    "tensor indexing",
                    format!("Use indices 0 to {} for dimension {}", dim_size - 1, dim_idx)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Validate slice bounds
    pub fn validate_slice(
        shape: &Shape,
        dim: usize,
        start: usize,
        end: usize,
    ) -> Result<()> {
        let dims = shape.dims();
        
        if dim >= dims.len() {
            return Err(TensorError::out_of_bounds(
                "SLICE_DIM_OUT_OF_BOUNDS",
                format!("Slice dimension {} is out of bounds", dim),
                dim,
                0,
                dims.len(),
                "tensor slicing",
                format!("Use dimension indices 0 to {}", dims.len() - 1)
            ));
        }
        
        let dim_size = dims[dim];
        
        if start >= dim_size {
            return Err(TensorError::out_of_bounds(
                "SLICE_START_OUT_OF_BOUNDS",
                format!("Slice start {} is out of bounds for dimension {}", start, dim),
                start,
                dim,
                dim_size,
                "tensor slicing",
                format!("Use start indices 0 to {} for dimension {}", dim_size - 1, dim)
            ));
        }
        
        if end > dim_size {
            return Err(TensorError::out_of_bounds(
                "SLICE_END_OUT_OF_BOUNDS",
                format!("Slice end {} is out of bounds for dimension {}", end, dim),
                end,
                dim,
                dim_size,
                "tensor slicing",
                format!("Use end indices 1 to {} for dimension {}", dim_size, dim)
            ));
        }
        
        if start >= end {
            return Err(TensorError::invalid_shape(
                "SLICE_INVALID_RANGE",
                "Slice start must be less than end",
                format!("dim {}: start={}, end={}", dim, start, end),
                "tensor slicing",
                format!("Start ({}) >= end ({})", start, end),
                "Ensure start index is less than end index"
            ));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validate_binary_op_shapes_compatible() {
        let shape1 = Shape::from_slice(&[2, 3, 4]);
        let shape2 = Shape::from_slice(&[2, 3, 4]);
        
        let result = TensorValidator::validate_binary_op_shapes(&shape1, &shape2, "addition");
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_binary_op_shapes_incompatible() {
        let shape1 = Shape::from_slice(&[2, 3, 4]);
        let shape2 = Shape::from_slice(&[2, 5, 4]);
        
        let result = TensorValidator::validate_binary_op_shapes(&shape1, &shape2, "addition");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), "TENSOR_BINARY_OP_INCOMPATIBLE");
    }
    
    #[test]
    fn test_validate_matmul_shapes_valid() {
        let left = Shape::from_slice(&[2, 3]);
        let right = Shape::from_slice(&[3, 4]);
        
        let result = TensorValidator::validate_matmul_shapes(&left, &right);
        assert!(result.is_ok());
        let output_shape = result.unwrap();
        assert_eq!(output_shape.dims(), &[2, 4]);
    }
    
    #[test]
    fn test_validate_matmul_shapes_incompatible() {
        let left = Shape::from_slice(&[2, 3]);
        let right = Shape::from_slice(&[4, 5]);  // 3 != 4
        
        let result = TensorValidator::validate_matmul_shapes(&left, &right);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), "MATMUL_INNER_DIM_MISMATCH");
    }
    
    #[test]
    fn test_validate_reshape_valid() {
        let input = Shape::from_slice(&[2, 3, 4]);  // 24 elements
        let target = Shape::from_slice(&[6, 4]);    // 24 elements
        
        let result = TensorValidator::validate_reshape(&input, &target);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_reshape_invalid() {
        let input = Shape::from_slice(&[2, 3, 4]);  // 24 elements
        let target = Shape::from_slice(&[5, 4]);    // 20 elements
        
        let result = TensorValidator::validate_reshape(&input, &target);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), "RESHAPE_SIZE_MISMATCH");
    }
    
    #[test]
    fn test_validate_transpose_valid() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        let axes = [2, 0, 1];  // Transpose dimensions
        
        let result = TensorValidator::validate_transpose(&shape, &axes);
        assert!(result.is_ok());
        let output_shape = result.unwrap();
        assert_eq!(output_shape.dims(), &[4, 2, 3]);
    }
    
    #[test]
    fn test_validate_transpose_duplicate_axes() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        let axes = [0, 1, 1];  // Duplicate axis
        
        let result = TensorValidator::validate_transpose(&shape, &axes);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), "TRANSPOSE_DUPLICATE_AXES");
    }
    
    #[test]
    fn test_validate_dtype_compatibility() {
        let result = TensorValidator::validate_dtype_compatibility(
            DType::F32,
            DType::F64,
            "addition"
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), DType::F64);
    }
    
    #[test]
    fn test_validate_indexing_valid() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        let indices = [1, 2, 0];
        
        let result = TensorValidator::validate_indexing(&shape, &indices);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_indexing_out_of_bounds() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        let indices = [1, 5, 0];  // 5 >= 3
        
        let result = TensorValidator::validate_indexing(&shape, &indices);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), "INDEX_OUT_OF_BOUNDS");
    }
}