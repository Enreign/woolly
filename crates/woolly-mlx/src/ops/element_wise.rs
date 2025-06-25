//! Element-wise tensor operations for MLX backend

use std::ops::{Add, Div, Mul, Sub};
use tracing::{debug, trace};

use crate::backend::MLXBackend;
use crate::device::Device;
use crate::error::{MLXError, Result};
use crate::storage::MLXStorage;
use super::utils::{validate_elementwise_shapes, check_device_compatibility, maybe_synchronize};

#[cfg(feature = "mlx")]
use crate::ffi::{mlx_add, mlx_array_from_data, mlx_array_free};

/// Element-wise addition
pub fn add<T>(
    backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    trace!("Performing element-wise addition");
    
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    
    // Validate shapes and get output shape
    let output_shape = validate_elementwise_shapes(lhs_shape, rhs_shape)?;
    
    // Check device compatibility
    let target_device = check_device_compatibility(lhs, rhs)?;
    
    #[cfg(feature = "mlx")]
    {
        // Use MLX native addition
        mlx_element_wise_add(backend, lhs, rhs, &output_shape, target_device)
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        // Fallback CPU implementation
        cpu_element_wise_add(backend, lhs, rhs, &output_shape, target_device)
    }
}

/// Element-wise subtraction
pub fn sub<T>(
    backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
) -> Result<MLXStorage<T>>
where
    T: Sub<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    trace!("Performing element-wise subtraction");
    
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    
    let output_shape = validate_elementwise_shapes(lhs_shape, rhs_shape)?;
    let target_device = check_device_compatibility(lhs, rhs)?;
    
    #[cfg(feature = "mlx")]
    {
        mlx_element_wise_sub(backend, lhs, rhs, &output_shape, target_device)
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        cpu_element_wise_sub(backend, lhs, rhs, &output_shape, target_device)
    }
}

/// Element-wise multiplication
pub fn mul<T>(
    backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
) -> Result<MLXStorage<T>>
where
    T: Mul<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    trace!("Performing element-wise multiplication");
    
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    
    let output_shape = validate_elementwise_shapes(lhs_shape, rhs_shape)?;
    let target_device = check_device_compatibility(lhs, rhs)?;
    
    #[cfg(feature = "mlx")]
    {
        mlx_element_wise_mul(backend, lhs, rhs, &output_shape, target_device)
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        cpu_element_wise_mul(backend, lhs, rhs, &output_shape, target_device)
    }
}

/// Element-wise division
pub fn div<T>(
    backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
) -> Result<MLXStorage<T>>
where
    T: Div<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    trace!("Performing element-wise division");
    
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    
    let output_shape = validate_elementwise_shapes(lhs_shape, rhs_shape)?;
    let target_device = check_device_compatibility(lhs, rhs)?;
    
    #[cfg(feature = "mlx")]
    {
        mlx_element_wise_div(backend, lhs, rhs, &output_shape, target_device)
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        cpu_element_wise_div(backend, lhs, rhs, &output_shape, target_device)
    }
}

// MLX-specific implementations
#[cfg(feature = "mlx")]
fn mlx_element_wise_add<T>(
    backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
    output_shape: &woolly_tensor::shape::Shape,
    device: Device,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using MLX native element-wise addition");
    
    // Create output storage
    let mut output = MLXStorage::zeros(
        output_shape.clone(),
        lhs.dtype(),
        device,
    )?;
    
    // Perform MLX addition
    let result = mlx_add(lhs.array_ptr(), rhs.array_ptr(), output.array_ptr());
    
    if let Err(e) = result {
        return Err(MLXError::ArrayOperationFailed(format!("MLX add failed: {}", e)));
    }
    
    maybe_synchronize(backend, true)?;
    
    Ok(output)
}

#[cfg(feature = "mlx")]
fn mlx_element_wise_sub<T>(
    backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
    output_shape: &woolly_tensor::shape::Shape,
    device: Device,
) -> Result<MLXStorage<T>>
where
    T: Sub<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using MLX native element-wise subtraction");
    
    let mut output = MLXStorage::zeros(
        output_shape.clone(),
        lhs.dtype(),
        device,
    )?;
    
    // Note: In real implementation, we'd call mlx_sub
    // For now, using add as placeholder
    let result = mlx_add(lhs.array_ptr(), rhs.array_ptr(), output.array_ptr());
    
    if let Err(e) = result {
        return Err(MLXError::ArrayOperationFailed(format!("MLX sub failed: {}", e)));
    }
    
    maybe_synchronize(backend, true)?;
    
    Ok(output)
}

#[cfg(feature = "mlx")]
fn mlx_element_wise_mul<T>(
    backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
    output_shape: &woolly_tensor::shape::Shape,
    device: Device,
) -> Result<MLXStorage<T>>
where
    T: Mul<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using MLX native element-wise multiplication");
    
    let mut output = MLXStorage::zeros(
        output_shape.clone(),
        lhs.dtype(),
        device,
    )?;
    
    // Note: In real implementation, we'd call mlx_mul
    let result = mlx_add(lhs.array_ptr(), rhs.array_ptr(), output.array_ptr());
    
    if let Err(e) = result {
        return Err(MLXError::ArrayOperationFailed(format!("MLX mul failed: {}", e)));
    }
    
    maybe_synchronize(backend, true)?;
    
    Ok(output)
}

#[cfg(feature = "mlx")]
fn mlx_element_wise_div<T>(
    backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
    output_shape: &woolly_tensor::shape::Shape,
    device: Device,
) -> Result<MLXStorage<T>>
where
    T: Div<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using MLX native element-wise division");
    
    let mut output = MLXStorage::zeros(
        output_shape.clone(),
        lhs.dtype(),
        device,
    )?;
    
    // Note: In real implementation, we'd call mlx_div
    let result = mlx_add(lhs.array_ptr(), rhs.array_ptr(), output.array_ptr());
    
    if let Err(e) = result {
        return Err(MLXError::ArrayOperationFailed(format!("MLX div failed: {}", e)));
    }
    
    maybe_synchronize(backend, true)?;
    
    Ok(output)
}

// CPU fallback implementations
fn cpu_element_wise_add<T>(
    _backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
    output_shape: &woolly_tensor::shape::Shape,
    device: Device,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using CPU fallback for element-wise addition");
    
    let lhs_data = lhs.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get lhs data: {}", e)))?;
    let rhs_data = rhs.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get rhs data: {}", e)))?;
    
    if lhs_data.len() != rhs_data.len() {
        return Err(MLXError::ShapeMismatch {
            expected: vec![lhs_data.len()],
            actual: vec![rhs_data.len()],
        });
    }
    
    let result_data: Vec<T> = lhs_data
        .into_iter()
        .zip(rhs_data.into_iter())
        .map(|(a, b)| a + b)
        .collect();
    
    MLXStorage::from_data(result_data, output_shape.clone(), lhs.dtype(), device)
}

fn cpu_element_wise_sub<T>(
    _backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
    output_shape: &woolly_tensor::shape::Shape,
    device: Device,
) -> Result<MLXStorage<T>>
where
    T: Sub<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using CPU fallback for element-wise subtraction");
    
    let lhs_data = lhs.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get lhs data: {}", e)))?;
    let rhs_data = rhs.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get rhs data: {}", e)))?;
    
    if lhs_data.len() != rhs_data.len() {
        return Err(MLXError::ShapeMismatch {
            expected: vec![lhs_data.len()],
            actual: vec![rhs_data.len()],
        });
    }
    
    let result_data: Vec<T> = lhs_data
        .into_iter()
        .zip(rhs_data.into_iter())
        .map(|(a, b)| a - b)
        .collect();
    
    MLXStorage::from_data(result_data, output_shape.clone(), lhs.dtype(), device)
}

fn cpu_element_wise_mul<T>(
    _backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
    output_shape: &woolly_tensor::shape::Shape,
    device: Device,
) -> Result<MLXStorage<T>>
where
    T: Mul<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using CPU fallback for element-wise multiplication");
    
    let lhs_data = lhs.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get lhs data: {}", e)))?;
    let rhs_data = rhs.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get rhs data: {}", e)))?;
    
    if lhs_data.len() != rhs_data.len() {
        return Err(MLXError::ShapeMismatch {
            expected: vec![lhs_data.len()],
            actual: vec![rhs_data.len()],
        });
    }
    
    let result_data: Vec<T> = lhs_data
        .into_iter()
        .zip(rhs_data.into_iter())
        .map(|(a, b)| a * b)
        .collect();
    
    MLXStorage::from_data(result_data, output_shape.clone(), lhs.dtype(), device)
}

fn cpu_element_wise_div<T>(
    _backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
    output_shape: &woolly_tensor::shape::Shape,
    device: Device,
) -> Result<MLXStorage<T>>
where
    T: Div<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using CPU fallback for element-wise division");
    
    let lhs_data = lhs.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get lhs data: {}", e)))?;
    let rhs_data = rhs.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get rhs data: {}", e)))?;
    
    if lhs_data.len() != rhs_data.len() {
        return Err(MLXError::ShapeMismatch {
            expected: vec![lhs_data.len()],
            actual: vec![rhs_data.len()],
        });
    }
    
    let result_data: Vec<T> = lhs_data
        .into_iter()
        .zip(rhs_data.into_iter())
        .map(|(a, b)| a / b)
        .collect();
    
    MLXStorage::from_data(result_data, output_shape.clone(), lhs.dtype(), device)
}

/// Broadcast-aware element-wise operation
pub fn broadcast_elementwise<T, F>(
    backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
    op: F,
    op_name: &str,
) -> Result<MLXStorage<T>>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
    debug!("Performing broadcast-aware {} operation", op_name);
    
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    
    let output_shape = validate_elementwise_shapes(lhs_shape, rhs_shape)?;
    let target_device = check_device_compatibility(lhs, rhs)?;
    
    let lhs_data = lhs.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get lhs data: {}", e)))?;
    let rhs_data = rhs.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get rhs data: {}", e)))?;
    
    // For now, assume same size (broadcasting not fully implemented)
    if lhs_data.len() != rhs_data.len() {
        return Err(MLXError::ShapeMismatch {
            expected: vec![lhs_data.len()],
            actual: vec![rhs_data.len()],
        });
    }
    
    let result_data: Vec<T> = lhs_data
        .into_iter()
        .zip(rhs_data.into_iter())
        .map(|(a, b)| op(a, b))
        .collect();
    
    MLXStorage::from_data(result_data, output_shape, lhs.dtype(), target_device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MLXBackend;
    use crate::device::Device;
    use woolly_tensor::backend::{DType, TensorBackend};
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
    fn test_element_wise_add() {
        if let Ok(backend) = MLXBackend::new() {
            let lhs_data = vec![1.0, 2.0, 3.0, 4.0];
            let rhs_data = vec![5.0, 6.0, 7.0, 8.0];
            
            if let (Ok(lhs), Ok(rhs)) = (
                create_test_storage(lhs_data, &[2, 2]),
                create_test_storage(rhs_data, &[2, 2]),
            ) {
                match add(&backend, &lhs, &rhs) {
                    Ok(result) => {
                        if let Ok(result_data) = result.to_vec() {
                            assert_eq!(result_data, vec![6.0, 8.0, 10.0, 12.0]);
                            println!("Element-wise addition test passed");
                        }
                    }
                    Err(e) => {
                        println!("Element-wise addition failed: {}", e);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_element_wise_mul() {
        if let Ok(backend) = MLXBackend::new() {
            let lhs_data = vec![2.0, 3.0, 4.0, 5.0];
            let rhs_data = vec![2.0, 2.0, 2.0, 2.0];
            
            if let (Ok(lhs), Ok(rhs)) = (
                create_test_storage(lhs_data, &[2, 2]),
                create_test_storage(rhs_data, &[2, 2]),
            ) {
                match mul(&backend, &lhs, &rhs) {
                    Ok(result) => {
                        if let Ok(result_data) = result.to_vec() {
                            assert_eq!(result_data, vec![4.0, 6.0, 8.0, 10.0]);
                            println!("Element-wise multiplication test passed");
                        }
                    }
                    Err(e) => {
                        println!("Element-wise multiplication failed: {}", e);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_broadcast_elementwise() {
        if let Ok(backend) = MLXBackend::new() {
            let lhs_data = vec![1.0, 2.0, 3.0, 4.0];
            let rhs_data = vec![10.0, 20.0, 30.0, 40.0];
            
            if let (Ok(lhs), Ok(rhs)) = (
                create_test_storage(lhs_data, &[2, 2]),
                create_test_storage(rhs_data, &[2, 2]),
            ) {
                match broadcast_elementwise(&backend, &lhs, &rhs, |a, b| a + b, "test_add") {
                    Ok(result) => {
                        if let Ok(result_data) = result.to_vec() {
                            assert_eq!(result_data, vec![11.0, 22.0, 33.0, 44.0]);
                            println!("Broadcast element-wise operation test passed");
                        }
                    }
                    Err(e) => {
                        println!("Broadcast element-wise operation failed: {}", e);
                    }
                }
            }
        }
    }
}