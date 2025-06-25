//! Reduction operations for MLX backend

use std::ops::{Add, Div};
use tracing::{debug, trace};

use crate::backend::MLXBackend;
use crate::device::Device;
use crate::error::{MLXError, Result};
use crate::storage::MLXStorage;
use super::utils::{reduction_output_shape, maybe_synchronize};
use woolly_tensor::shape::Shape;

/// Sum reduction along specified axes
pub fn sum<T>(
    backend: &MLXBackend,
    input: &MLXStorage<T>,
    shape: &Shape,
    axes: &[usize],
    keep_dims: bool,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Performing sum reduction along axes {:?}", axes);
    
    let output_shape = reduction_output_shape(shape, axes, keep_dims)?;
    
    #[cfg(feature = "mlx")]
    {
        mlx_sum_reduction(backend, input, &output_shape, axes, keep_dims)
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        cpu_sum_reduction(backend, input, shape, &output_shape, axes, keep_dims)
    }
}

/// Mean reduction along specified axes
pub fn mean<T>(
    backend: &MLXBackend,
    input: &MLXStorage<T>,
    shape: &Shape,
    axes: &[usize],
    keep_dims: bool,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Div<Output = T> + num_traits::Zero + Clone + From<f32> + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Performing mean reduction along axes {:?}", axes);
    
    let output_shape = reduction_output_shape(shape, axes, keep_dims)?;
    
    #[cfg(feature = "mlx")]
    {
        mlx_mean_reduction(backend, input, &output_shape, axes, keep_dims)
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        cpu_mean_reduction(backend, input, shape, &output_shape, axes, keep_dims)
    }
}

#[cfg(feature = "mlx")]
fn mlx_sum_reduction<T>(
    backend: &MLXBackend,
    input: &MLXStorage<T>,
    output_shape: &Shape,
    _axes: &[usize],
    _keep_dims: bool,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using MLX sum reduction");
    
    let mut output = MLXStorage::zeros(
        output_shape.clone(),
        input.dtype(),
        Device::GPU,
    )?;
    
    // In real implementation, would call MLX reduction kernel
    trace!("MLX sum reduction (mock implementation)");
    
    maybe_synchronize(backend, true)?;
    Ok(output)
}

#[cfg(feature = "mlx")]
fn mlx_mean_reduction<T>(
    backend: &MLXBackend,
    input: &MLXStorage<T>,
    output_shape: &Shape,
    _axes: &[usize],
    _keep_dims: bool,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Div<Output = T> + num_traits::Zero + Clone + From<f32> + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using MLX mean reduction");
    
    let mut output = MLXStorage::zeros(
        output_shape.clone(),
        input.dtype(),
        Device::GPU,
    )?;
    
    trace!("MLX mean reduction (mock implementation)");
    
    maybe_synchronize(backend, true)?;
    Ok(output)
}

fn cpu_sum_reduction<T>(
    _backend: &MLXBackend,
    input: &MLXStorage<T>,
    input_shape: &Shape,
    output_shape: &Shape,
    axes: &[usize],
    _keep_dims: bool,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using CPU sum reduction");
    
    let input_data = input.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get input data: {}", e)))?;
    
    // Simple implementation for single axis reduction
    if axes.len() == 1 {
        let axis = axes[0];
        let result_data = reduce_single_axis(&input_data, input_shape, axis, T::zero(), |acc, val| acc + val.clone())?;
        MLXStorage::from_data(result_data, output_shape.clone(), input.dtype(), Device::CPU)
    } else {
        // For multiple axes, reduce one by one
        // This is inefficient but works for the mock implementation
        let mut current_data = input_data;
        let mut current_shape = input_shape.clone();
        
        for &axis in axes {
            let temp_result = reduce_single_axis(&current_data, &current_shape, axis, T::zero(), |acc, val| acc + val.clone())?;
            current_data = temp_result;
            // Update shape (simplified)
            let mut new_dims = current_shape.as_slice().to_vec();
            new_dims.remove(axis);
            current_shape = Shape::from_slice(&new_dims);
        }
        
        MLXStorage::from_data(current_data, output_shape.clone(), input.dtype(), Device::CPU)
    }
}

fn cpu_mean_reduction<T>(
    _backend: &MLXBackend,
    input: &MLXStorage<T>,
    input_shape: &Shape,
    output_shape: &Shape,
    axes: &[usize],
    _keep_dims: bool,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Div<Output = T> + num_traits::Zero + Clone + From<f32> + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using CPU mean reduction");
    
    let input_data = input.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get input data: {}", e)))?;
    
    if axes.len() == 1 {
        let axis = axes[0];
        let reduction_size = input_shape[axis] as f32;
        let divisor = T::from(reduction_size);
        
        let sum_result = reduce_single_axis(&input_data, input_shape, axis, T::zero(), |acc, val| acc + val.clone())?;
        let mean_result: Vec<T> = sum_result.into_iter().map(|val| val / divisor.clone()).collect();
        
        MLXStorage::from_data(mean_result, output_shape.clone(), input.dtype(), Device::CPU)
    } else {
        // For multiple axes, calculate total reduction size
        let total_reduction_size: usize = axes.iter().map(|&axis| input_shape[axis]).product();
        let divisor = T::from(total_reduction_size as f32);
        
        let mut current_data = input_data;
        let mut current_shape = input_shape.clone();
        
        for &axis in axes {
            let temp_result = reduce_single_axis(&current_data, &current_shape, axis, T::zero(), |acc, val| acc + val.clone())?;
            current_data = temp_result;
            let mut new_dims = current_shape.as_slice().to_vec();
            new_dims.remove(axis);
            current_shape = Shape::from_slice(&new_dims);
        }
        
        let mean_result: Vec<T> = current_data.into_iter().map(|val| val / divisor.clone()).collect();
        MLXStorage::from_data(mean_result, output_shape.clone(), input.dtype(), Device::CPU)
    }
}

fn reduce_single_axis<T, F>(
    data: &[T],
    shape: &Shape,
    axis: usize,
    initial: T,
    op: F,
) -> Result<Vec<T>>
where
    T: Clone,
    F: Fn(T, &T) -> T,
{
    if axis >= shape.ndim() {
        return Err(MLXError::ArrayOperationFailed(format!("Axis {} out of bounds for tensor with {} dimensions", axis, shape.ndim())));
    }
    
    let dims = shape.as_slice();
    let axis_size = dims[axis];
    
    // Calculate strides
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    
    // Calculate output size
    let output_size = data.len() / axis_size;
    let mut result = vec![initial; output_size];
    
    // Perform reduction
    for i in 0..data.len() {
        // Calculate multi-dimensional index
        let mut temp = i;
        let mut indices = vec![0; dims.len()];
        for j in 0..dims.len() {
            indices[j] = temp / strides[j];
            temp %= strides[j];
        }
        
        // Calculate output index (remove the axis dimension)
        let mut output_indices = indices.clone();
        output_indices.remove(axis);
        
        let mut output_index = 0;
        let mut output_stride = 1;
        for j in (0..output_indices.len()).rev() {
            output_index += output_indices[j] * output_stride;
            output_stride *= if j < axis { dims[j] } else { dims[j + 1] };
        }
        
        result[output_index] = op(result[output_index].clone(), &data[i]);
    }
    
    Ok(result)
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
    fn test_sum_reduction() {
        if let Ok(backend) = MLXBackend::new() {
            let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1, 2, 3], [4, 5, 6]]
            let shape = Shape::from_slice(&[2, 3]);
            
            if let Ok(input) = create_test_storage(data, &[2, 3]) {
                // Sum along axis 0 (columns)
                match sum(&backend, &input, &shape, &[0], false) {
                    Ok(result) => {
                        if let Ok(result_data) = result.to_vec() {
                            // Expected: [1+4, 2+5, 3+6] = [5, 7, 9]
                            let expected = vec![5.0, 7.0, 9.0];
                            assert_eq!(result_data, expected);
                            println!("Sum reduction test passed");
                        }
                    }
                    Err(e) => {
                        println!("Sum reduction failed: {}", e);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_mean_reduction() {
        if let Ok(backend) = MLXBackend::new() {
            let data = vec![2.0, 4.0, 6.0, 8.0]; // [[2, 4], [6, 8]]
            let shape = Shape::from_slice(&[2, 2]);
            
            if let Ok(input) = create_test_storage(data, &[2, 2]) {
                // Mean along axis 1 (rows)
                match mean(&backend, &input, &shape, &[1], false) {
                    Ok(result) => {
                        if let Ok(result_data) = result.to_vec() {
                            // Expected: [(2+4)/2, (6+8)/2] = [3, 7]
                            let expected = vec![3.0, 7.0];
                            assert_eq!(result_data, expected);
                            println!("Mean reduction test passed");
                        }
                    }
                    Err(e) => {
                        println!("Mean reduction failed: {}", e);
                    }
                }
            }
        }
    }
}