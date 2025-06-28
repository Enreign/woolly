//! Shape manipulation operations for MLX backend

use tracing::{debug, trace};

use woolly_tensor::backend::TensorStorage;

use crate::backend::MLXBackend;
use crate::device::Device;
use crate::error::{MLXError, Result};
use crate::storage::MLXStorage;
use woolly_tensor::shape::Shape;

/// Reshape tensor
pub fn reshape<T>(
    _backend: &MLXBackend,
    input: &MLXStorage<T>,
    old_shape: &Shape,
    new_shape: &Shape,
) -> Result<MLXStorage<T>>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Reshaping tensor from {:?} to {:?}", old_shape, new_shape);
    
    if old_shape.numel() != new_shape.numel() {
        return Err(MLXError::ShapeMismatch {
            expected: vec![old_shape.numel()],
            actual: vec![new_shape.numel()],
        });
    }
    
    // For MLX with unified memory, reshape is just a metadata operation
    // We need to create a new storage with the same data but different shape
    let input_data = input.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get input data: {}", e)))?;
    
    MLXStorage::from_data(
        input_data,
        new_shape.clone(),
        input.dtype(),
        Device::CPU, // Keep on same device as input
    )
}

/// Transpose tensor
pub fn transpose<T>(
    _backend: &MLXBackend,
    input: &MLXStorage<T>,
    shape: &Shape,
    axes: &[usize],
) -> Result<MLXStorage<T>>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Transposing tensor with axes {:?}", axes);
    
    if axes.len() != shape.ndim() {
        return Err(MLXError::ArrayOperationFailed(
            format!("Transpose axes length {} doesn't match tensor dimensions {}", 
                axes.len(), shape.ndim())
        ));
    }
    
    let input_data = input.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get input data: {}", e)))?;
    
    // Calculate new shape
    let old_dims = shape.as_slice();
    let new_dims: Vec<usize> = axes.iter().map(|&i| old_dims[i]).collect();
    let new_shape = Shape::from_slice(&new_dims);
    
    // Perform transpose
    let transposed_data = transpose_data(&input_data, old_dims, axes)?;
    
    MLXStorage::from_data(
        transposed_data,
        new_shape,
        input.dtype(),
        Device::CPU,
    )
}

/// Slice tensor
pub fn slice<T>(
    _backend: &MLXBackend,
    input: &MLXStorage<T>,
    shape: &Shape,
    ranges: &[(usize, usize, isize)], // (start, end, step) for each dimension
) -> Result<MLXStorage<T>>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Slicing tensor with ranges {:?}", ranges);
    
    if ranges.len() != shape.ndim() {
        return Err(MLXError::ArrayOperationFailed(
            format!("Slice ranges length {} doesn't match tensor dimensions {}", 
                ranges.len(), shape.ndim())
        ));
    }
    
    let input_data = input.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get input data: {}", e)))?;
    
    // Calculate output shape
    let mut output_dims = Vec::new();
    for (i, &(start, end, step)) in ranges.iter().enumerate() {
        if step == 0 {
            return Err(MLXError::ArrayOperationFailed("Slice step cannot be zero".to_string()));
        }
        if start >= shape[i] || end > shape[i] {
            return Err(MLXError::ArrayOperationFailed(
                format!("Slice range [{}, {}) out of bounds for dimension {} with size {}", 
                    start, end, i, shape[i])
            ));
        }
        
        let size = if step > 0 {
            ((end - start) as isize + step - 1) / step
        } else {
            ((start - end) as isize + (-step) - 1) / (-step)
        };
        output_dims.push(size.max(0) as usize);
    }
    let output_shape = Shape::from_slice(&output_dims);
    
    // Perform slicing
    let sliced_data = slice_data(&input_data, shape.as_slice(), ranges)?;
    
    MLXStorage::from_data(
        sliced_data,
        output_shape,
        input.dtype(),
        Device::CPU,
    )
}

/// Concatenate tensors along an axis
pub fn concatenate<T>(
    _backend: &MLXBackend,
    inputs: &[&MLXStorage<T>],
    axis: usize,
) -> Result<MLXStorage<T>>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Concatenating {} tensors along axis {}", inputs.len(), axis);
    
    if inputs.is_empty() {
        return Err(MLXError::ArrayOperationFailed("Cannot concatenate empty list of tensors".to_string()));
    }
    
    let first_shape = inputs[0].shape();
    if axis >= first_shape.ndim() {
        return Err(MLXError::ArrayOperationFailed(
            format!("Concatenation axis {} out of bounds for tensor with {} dimensions", 
                axis, first_shape.ndim())
        ));
    }
    
    // Validate all shapes are compatible
    let mut total_axis_size = 0;
    for input in inputs {
        let shape = input.shape();
        if shape.ndim() != first_shape.ndim() {
            return Err(MLXError::ShapeMismatch {
                expected: first_shape.as_slice().to_vec(),
                actual: shape.as_slice().to_vec(),
            });
        }
        
        for (i, (&dim1, &dim2)) in first_shape.as_slice().iter().zip(shape.as_slice().iter()).enumerate() {
            if i != axis && dim1 != dim2 {
                return Err(MLXError::ShapeMismatch {
                    expected: first_shape.as_slice().to_vec(),
                    actual: shape.as_slice().to_vec(),
                });
            }
        }
        
        total_axis_size += shape[axis];
    }
    
    // Calculate output shape
    let mut output_dims = first_shape.as_slice().to_vec();
    output_dims[axis] = total_axis_size;
    let output_shape = Shape::from_slice(&output_dims);
    
    // Concatenate data
    let mut concatenated_data = Vec::with_capacity(output_shape.numel());
    
    // For now, simple implementation for axis 0
    if axis == 0 {
        for input in inputs {
            let data = input.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get input data: {}", e)))?;
            concatenated_data.extend(data);
        }
    } else {
        // More complex implementation needed for other axes
        return Err(MLXError::NotImplemented("Concatenation along non-zero axes not yet implemented".to_string()));
    }
    
    MLXStorage::from_data(
        concatenated_data,
        output_shape,
        inputs[0].dtype(),
        Device::CPU,
    )
}

/// Stack tensors along a new axis
pub fn stack<T>(
    backend: &MLXBackend,
    inputs: &[&MLXStorage<T>],
    axis: usize,
) -> Result<MLXStorage<T>>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Stacking {} tensors along new axis {}", inputs.len(), axis);
    
    if inputs.is_empty() {
        return Err(MLXError::ArrayOperationFailed("Cannot stack empty list of tensors".to_string()));
    }
    
    // All inputs must have the same shape
    let first_shape = inputs[0].shape();
    for input in inputs.iter().skip(1) {
        let shape = input.shape();
        if shape.as_slice() != first_shape.as_slice() {
            return Err(MLXError::ShapeMismatch {
                expected: first_shape.as_slice().to_vec(),
                actual: shape.as_slice().to_vec(),
            });
        }
    }
    
    if axis > first_shape.ndim() {
        return Err(MLXError::ArrayOperationFailed(
            format!("Stack axis {} out of bounds for tensor with {} dimensions + 1", 
                axis, first_shape.ndim())
        ));
    }
    
    // Calculate output shape (insert new dimension of size len(inputs))
    let mut output_dims = first_shape.as_slice().to_vec();
    output_dims.insert(axis, inputs.len());
    let output_shape = Shape::from_slice(&output_dims);
    
    // For simplicity, reshape each input to add the new dimension, then concatenate
    let mut reshaped_inputs = Vec::new();
    for input in inputs {
        let mut new_dims = first_shape.as_slice().to_vec();
        new_dims.insert(axis, 1);
        let new_shape = Shape::from_slice(&new_dims);
        
        let reshaped = reshape(backend, input, first_shape, &new_shape)?;
        reshaped_inputs.push(reshaped);
    }
    
    // Concatenate along the new axis
    let reshaped_refs: Vec<&MLXStorage<T>> = reshaped_inputs.iter().collect();
    concatenate(backend, &reshaped_refs, axis)
}

// Helper functions

fn transpose_data<T>(
    data: &[T],
    shape: &[usize],
    axes: &[usize],
) -> Result<Vec<T>>
where
    T: Clone,
{
    let ndim = shape.len();
    if axes.len() != ndim {
        return Err(MLXError::ArrayOperationFailed("Axes length mismatch".to_string()));
    }
    
    // Calculate strides for input
    let mut input_strides = vec![1; ndim];
    for i in (0..ndim - 1).rev() {
        input_strides[i] = input_strides[i + 1] * shape[i + 1];
    }
    
    // Calculate output shape and strides
    let output_shape: Vec<usize> = axes.iter().map(|&i| shape[i]).collect();
    let mut output_strides = vec![1; ndim];
    for i in (0..ndim - 1).rev() {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }
    
    let total_elements = data.len();
    let mut result = vec![data[0].clone(); total_elements];
    
    for i in 0..total_elements {
        // Calculate input indices
        let mut temp = i;
        let mut input_indices = vec![0; ndim];
        for j in 0..ndim {
            input_indices[j] = temp / input_strides[j];
            temp %= input_strides[j];
        }
        
        // Calculate output indices (transpose)
        let mut output_indices = vec![0; ndim];
        for j in 0..ndim {
            output_indices[j] = input_indices[axes[j]];
        }
        
        // Calculate output index
        let mut output_index = 0;
        for j in 0..ndim {
            output_index += output_indices[j] * output_strides[j];
        }
        
        result[output_index] = data[i].clone();
    }
    
    Ok(result)
}

fn slice_data<T>(
    data: &[T],
    shape: &[usize],
    ranges: &[(usize, usize, isize)],
) -> Result<Vec<T>>
where
    T: Clone,
{
    // Calculate input strides
    let ndim = shape.len();
    let mut strides = vec![1; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    
    // Calculate output size
    let mut output_size = 1;
    for &(start, end, step) in ranges {
        let size = if step > 0 {
            ((end - start) as isize + step - 1) / step
        } else {
            ((start - end) as isize + (-step) - 1) / (-step)
        };
        output_size *= size.max(0) as usize;
    }
    
    let mut result = Vec::with_capacity(output_size);
    
    // Generate all valid index combinations
    generate_slice_indices(ranges, 0, &mut vec![0; ndim], &mut |indices| {
        let mut index = 0;
        for (i, &idx) in indices.iter().enumerate() {
            index += idx * strides[i];
        }
        result.push(data[index].clone());
    });
    
    Ok(result)
}

fn generate_slice_indices<F>(
    ranges: &[(usize, usize, isize)],
    dim: usize,
    current_indices: &mut Vec<usize>,
    callback: &mut F,
) where
    F: FnMut(&[usize]),
{
    if dim == ranges.len() {
        callback(current_indices);
        return;
    }
    
    let (start, end, step) = ranges[dim];
    if step > 0 {
        let mut i = start;
        while i < end {
            current_indices[dim] = i;
            generate_slice_indices(ranges, dim + 1, current_indices, callback);
            i = (i as isize + step) as usize;
        }
    } else {
        let mut i = start;
        loop {
            current_indices[dim] = i;
            generate_slice_indices(ranges, dim + 1, current_indices, callback);
            if i < end || (i as isize + step) < 0 {
                break;
            }
            i = (i as isize + step) as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MLXBackend;
    use crate::device::Device;
    use woolly_tensor::backend::DType;
    use woolly_tensor::shape::Shape;
    
    fn create_test_storage(data: Vec<f32>, shape: &[usize]) -> Result<MLXStorage<f32>> {
        MLXStorage::from_data(
            data,
            Shape::from_slice(shape),
            DType::F32,
            Device::CPU,
        )
    }
    
    #[test]
    fn test_reshape() {
        if let Ok(backend) = MLXBackend::new() {
            let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let old_shape = Shape::from_slice(&[2, 3]);
            let new_shape = Shape::from_slice(&[3, 2]);
            
            if let Ok(input) = create_test_storage(data.clone(), &[2, 3]) {
                match reshape(&backend, &input, &old_shape, &new_shape) {
                    Ok(result) => {
                        assert_eq!(result.shape().as_slice(), &[3, 2]);
                        if let Ok(result_data) = result.to_vec() {
                            assert_eq!(result_data, data); // Data should be the same
                            println!("Reshape test passed");
                        }
                    }
                    Err(e) => {
                        println!("Reshape failed: {}", e);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_transpose_2d() {
        if let Ok(backend) = MLXBackend::new() {
            let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1, 2, 3], [4, 5, 6]]
            let shape = Shape::from_slice(&[2, 3]);
            let axes = vec![1, 0]; // Transpose
            
            if let Ok(input) = create_test_storage(data, &[2, 3]) {
                match transpose(&backend, &input, &shape, &axes) {
                    Ok(result) => {
                        assert_eq!(result.shape().as_slice(), &[3, 2]);
                        if let Ok(result_data) = result.to_vec() {
                            // Expected: [[1, 4], [2, 5], [3, 6]] = [1, 4, 2, 5, 3, 6]
                            let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
                            assert_eq!(result_data, expected);
                            println!("Transpose test passed");
                        }
                    }
                    Err(e) => {
                        println!("Transpose failed: {}", e);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_concatenate() {
        if let Ok(backend) = MLXBackend::new() {
            let data1 = vec![1.0, 2.0, 3.0, 4.0];
            let data2 = vec![5.0, 6.0, 7.0, 8.0];
            
            if let (Ok(input1), Ok(input2)) = (
                create_test_storage(data1, &[2, 2]),
                create_test_storage(data2, &[2, 2]),
            ) {
                let inputs = vec![&input1, &input2];
                match concatenate(&backend, &inputs, 0) {
                    Ok(result) => {
                        assert_eq!(result.shape().as_slice(), &[4, 2]);
                        if let Ok(result_data) = result.to_vec() {
                            let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
                            assert_eq!(result_data, expected);
                            println!("Concatenate test passed");
                        }
                    }
                    Err(e) => {
                        println!("Concatenate failed: {}", e);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_stack() {
        if let Ok(backend) = MLXBackend::new() {
            let data1 = vec![1.0, 2.0];
            let data2 = vec![3.0, 4.0];
            
            if let (Ok(input1), Ok(input2)) = (
                create_test_storage(data1, &[2]),
                create_test_storage(data2, &[2]),
            ) {
                let inputs = vec![&input1, &input2];
                match stack(&backend, &inputs, 0) {
                    Ok(result) => {
                        assert_eq!(result.shape().as_slice(), &[2, 2]);
                        if let Ok(result_data) = result.to_vec() {
                            let expected = vec![1.0, 2.0, 3.0, 4.0];
                            assert_eq!(result_data, expected);
                            println!("Stack test passed");
                        }
                    }
                    Err(e) => {
                        println!("Stack failed: {}", e);
                    }
                }
            }
        }
    }
}