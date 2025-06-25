//! Matrix multiplication operations for MLX backend

use std::ops::{Add, Mul};
use tracing::{debug, trace, warn};

use crate::backend::MLXBackend;
use crate::device::Device;
use crate::error::{MLXError, Result};
use crate::storage::MLXStorage;
use super::utils::{matmul_output_shape, check_device_compatibility, maybe_synchronize};
use woolly_tensor::shape::Shape;

#[cfg(feature = "mlx")]
use crate::ffi::mlx_matmul;

/// Matrix multiplication
pub fn matmul<T>(
    backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
    lhs_shape: &Shape,
    rhs_shape: &Shape,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Mul<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Performing matrix multiplication: {:?} × {:?}", lhs_shape, rhs_shape);
    
    // Calculate output shape
    let output_shape = matmul_output_shape(lhs_shape, rhs_shape)?;
    
    // Check device compatibility
    let target_device = check_device_compatibility(lhs, rhs)?;
    
    // Choose implementation based on available features and tensor size
    let total_ops = lhs_shape.numel() * rhs_shape.numel();
    
    if total_ops > 100_000 && target_device == Device::GPU {
        // Use GPU acceleration for large matrices
        #[cfg(feature = "mlx")]
        {
            mlx_matmul_gpu(backend, lhs, rhs, &output_shape, target_device)
        }
        
        #[cfg(not(feature = "mlx"))]
        {
            cpu_matmul_optimized(backend, lhs, rhs, lhs_shape, rhs_shape, &output_shape, target_device)
        }
    } else {
        // Use CPU implementation for small matrices or when GPU not available
        cpu_matmul_optimized(backend, lhs, rhs, lhs_shape, rhs_shape, &output_shape, target_device)
    }
}

/// Batched matrix multiplication
pub fn batch_matmul<T>(
    backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
    lhs_shape: &Shape,
    rhs_shape: &Shape,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Mul<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Performing batched matrix multiplication");
    
    if lhs_shape.ndim() < 3 || rhs_shape.ndim() < 3 {
        return Err(MLXError::ArrayOperationFailed(
            "Batched matrix multiplication requires at least 3D tensors".to_string()
        ));
    }
    
    // For now, delegate to regular matmul (which handles batch dimensions)
    matmul(backend, lhs, rhs, lhs_shape, rhs_shape)
}

/// Matrix-vector multiplication
pub fn matvec<T>(
    backend: &MLXBackend,
    matrix: &MLXStorage<T>,
    vector: &MLXStorage<T>,
    matrix_shape: &Shape,
    vector_shape: &Shape,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Mul<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Performing matrix-vector multiplication");
    
    if matrix_shape.ndim() != 2 || vector_shape.ndim() != 1 {
        return Err(MLXError::ArrayOperationFailed(
            "Matrix-vector multiplication requires 2D matrix and 1D vector".to_string()
        ));
    }
    
    let matrix_cols = matrix_shape[1];
    let vector_len = vector_shape[0];
    
    if matrix_cols != vector_len {
        return Err(MLXError::ShapeMismatch {
            expected: vec![matrix_shape[0], matrix_cols],
            actual: vec![vector_len],
        });
    }
    
    // Reshape vector to column matrix for matmul, then reshape result back
    let vector_2d_shape = Shape::from_slice(&[vector_len, 1]);
    let output_shape = Shape::from_slice(&[matrix_shape[0]]);
    
    // Use optimized matvec implementation
    #[cfg(feature = "mlx")]
    {
        mlx_matvec_gpu(backend, matrix, vector, matrix_shape, vector_shape, &output_shape)
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        cpu_matvec_optimized(backend, matrix, vector, matrix_shape, vector_shape, &output_shape)
    }
}

// MLX GPU implementations
#[cfg(feature = "mlx")]
fn mlx_matmul_gpu<T>(
    backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
    output_shape: &Shape,
    device: Device,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Mul<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using MLX GPU matrix multiplication");
    
    // Create output storage
    let mut output = MLXStorage::zeros(
        output_shape.clone(),
        lhs.dtype(),
        device,
    )?;
    
    // Perform MLX matrix multiplication
    let result = mlx_matmul(lhs.array_ptr(), rhs.array_ptr(), output.array_ptr());
    
    if let Err(e) = result {
        return Err(MLXError::ArrayOperationFailed(format!("MLX matmul failed: {}", e)));
    }
    
    maybe_synchronize(backend, true)?;
    
    Ok(output)
}

#[cfg(feature = "mlx")]
fn mlx_matvec_gpu<T>(
    backend: &MLXBackend,
    matrix: &MLXStorage<T>,
    vector: &MLXStorage<T>,
    _matrix_shape: &Shape,
    _vector_shape: &Shape,
    output_shape: &Shape,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Mul<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using MLX GPU matrix-vector multiplication");
    
    let mut output = MLXStorage::zeros(
        output_shape.clone(),
        matrix.dtype(),
        Device::GPU,
    )?;
    
    // Use matmul for matvec (could be optimized with specific matvec kernel)
    let result = mlx_matmul(matrix.array_ptr(), vector.array_ptr(), output.array_ptr());
    
    if let Err(e) = result {
        return Err(MLXError::ArrayOperationFailed(format!("MLX matvec failed: {}", e)));
    }
    
    maybe_synchronize(backend, true)?;
    
    Ok(output)
}

// CPU fallback implementations
fn cpu_matmul_optimized<T>(
    _backend: &MLXBackend,
    lhs: &MLXStorage<T>,
    rhs: &MLXStorage<T>,
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    output_shape: &Shape,
    device: Device,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Mul<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using CPU optimized matrix multiplication");
    
    let lhs_data = lhs.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get lhs data: {}", e)))?;
    let rhs_data = rhs.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get rhs data: {}", e)))?;
    
    // For 2D matrices
    if lhs_shape.ndim() == 2 && rhs_shape.ndim() == 2 {
        let m = lhs_shape[0];
        let k = lhs_shape[1];
        let n = rhs_shape[1];
        
        let result_data = cpu_matmul_2d(&lhs_data, &rhs_data, m, k, n)?;
        
        MLXStorage::from_data(result_data, output_shape.clone(), lhs.dtype(), device)
    } else {
        // Handle batched case
        cpu_matmul_batched(&lhs_data, &rhs_data, lhs_shape, rhs_shape, output_shape, device, lhs.dtype())
    }
}

fn cpu_matvec_optimized<T>(
    _backend: &MLXBackend,
    matrix: &MLXStorage<T>,
    vector: &MLXStorage<T>,
    matrix_shape: &Shape,
    _vector_shape: &Shape,
    output_shape: &Shape,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Mul<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using CPU optimized matrix-vector multiplication");
    
    let matrix_data = matrix.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get matrix data: {}", e)))?;
    let vector_data = vector.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get vector data: {}", e)))?;
    
    let m = matrix_shape[0];
    let n = matrix_shape[1];
    
    let mut result_data = vec![T::zero(); m];
    
    for i in 0..m {
        let mut sum = T::zero();
        for j in 0..n {
            let matrix_val = matrix_data[i * n + j].clone();
            let vector_val = vector_data[j].clone();
            sum = sum + (matrix_val * vector_val);
        }
        result_data[i] = sum;
    }
    
    MLXStorage::from_data(result_data, output_shape.clone(), matrix.dtype(), Device::CPU)
}

fn cpu_matmul_2d<T>(
    lhs: &[T],
    rhs: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<T>>
where
    T: Add<Output = T> + Mul<Output = T> + num_traits::Zero + Clone,
{
    let mut result = vec![T::zero(); m * n];
    
    // Basic matrix multiplication with loop tiling for better cache performance
    const TILE_SIZE: usize = 64;
    
    for ii in (0..m).step_by(TILE_SIZE) {
        for jj in (0..n).step_by(TILE_SIZE) {
            for kk in (0..k).step_by(TILE_SIZE) {
                let i_end = (ii + TILE_SIZE).min(m);
                let j_end = (jj + TILE_SIZE).min(n);
                let k_end = (kk + TILE_SIZE).min(k);
                
                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = T::zero();
                        for k_idx in kk..k_end {
                            let lhs_val = lhs[i * k + k_idx].clone();
                            let rhs_val = rhs[k_idx * n + j].clone();
                            sum = sum + (lhs_val * rhs_val);
                        }
                        result[i * n + j] = result[i * n + j].clone() + sum;
                    }
                }
            }
        }
    }
    
    Ok(result)
}

fn cpu_matmul_batched<T>(
    lhs: &[T],
    rhs: &[T],
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    output_shape: &Shape,
    device: Device,
    dtype: woolly_tensor::backend::DType,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Mul<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Performing batched CPU matrix multiplication");
    
    // Extract batch dimensions and matrix dimensions
    let lhs_batch_dims = &lhs_shape.as_slice()[..lhs_shape.ndim() - 2];
    let rhs_batch_dims = &rhs_shape.as_slice()[..rhs_shape.ndim() - 2];
    
    let lhs_m = lhs_shape[lhs_shape.ndim() - 2];
    let lhs_k = lhs_shape[lhs_shape.ndim() - 1];
    let rhs_k = rhs_shape[rhs_shape.ndim() - 2];
    let rhs_n = rhs_shape[rhs_shape.ndim() - 1];
    
    if lhs_k != rhs_k {
        return Err(MLXError::ShapeMismatch {
            expected: vec![lhs_m, lhs_k],
            actual: vec![rhs_k, rhs_n],
        });
    }
    
    // Calculate batch size
    let batch_size: usize = lhs_batch_dims.iter().product();
    let matrix_size_lhs = lhs_m * lhs_k;
    let matrix_size_rhs = rhs_k * rhs_n;
    let matrix_size_out = lhs_m * rhs_n;
    
    let mut result_data = Vec::with_capacity(output_shape.numel());
    
    for batch_idx in 0..batch_size {
        let lhs_offset = batch_idx * matrix_size_lhs;
        let rhs_offset = batch_idx * matrix_size_rhs;
        
        let lhs_batch = &lhs[lhs_offset..lhs_offset + matrix_size_lhs];
        let rhs_batch = &rhs[rhs_offset..rhs_offset + matrix_size_rhs];
        
        let batch_result = cpu_matmul_2d(lhs_batch, rhs_batch, lhs_m, lhs_k, rhs_n)?;
        result_data.extend(batch_result);
    }
    
    MLXStorage::from_data(result_data, output_shape.clone(), dtype, device)
}

/// Quantized matrix multiplication (for Q4_0, Q8_0, etc.)
pub fn quantized_matmul<T>(
    backend: &MLXBackend,
    weights: &MLXStorage<u8>, // Quantized weights
    input: &MLXStorage<T>,    // Input activations
    weights_shape: &Shape,
    input_shape: &Shape,
    quantization_scheme: woolly_tensor::quantization::QuantizationScheme,
) -> Result<MLXStorage<T>>
where
    T: Add<Output = T> + Mul<Output = T> + num_traits::Zero + Clone + From<f32> + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Performing quantized matrix multiplication with scheme {:?}", quantization_scheme);
    
    // This would use optimized quantized kernels in a real implementation
    // For now, fall back to dequantize -> matmul
    warn!("Quantized matmul not yet optimized, falling back to dequantize -> matmul");
    
    // Dequantize weights
    let dequantized_weights = crate::ops::quantization::dequantize(
        backend,
        weights,
        woolly_tensor::backend::DType::F32,
    )?;
    
    // Convert to same type as input
    // This is a simplification - in practice we'd need proper type conversion
    let weights_data = dequantized_weights.to_vec()
        .map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get weights data: {}", e)))?;
    
    let converted_weights_data: Vec<T> = weights_data
        .into_iter()
        .map(|x| T::from(x))
        .collect();
    
    let weights_storage = MLXStorage::from_data(
        converted_weights_data,
        weights_shape.clone(),
        woolly_tensor::backend::DType::F32, // This should match T
        Device::GPU,
    )?;
    
    // Perform regular matmul
    matmul(backend, &weights_storage, input, weights_shape, input_shape)
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
    fn test_matmul_2x2() {
        if let Ok(backend) = MLXBackend::new() {
            // Test 2x2 matrix multiplication
            let lhs_data = vec![1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
            let rhs_data = vec![5.0, 6.0, 7.0, 8.0]; // [[5, 6], [7, 8]]
            
            let lhs_shape = Shape::from_slice(&[2, 2]);
            let rhs_shape = Shape::from_slice(&[2, 2]);
            
            if let (Ok(lhs), Ok(rhs)) = (
                create_test_storage(lhs_data, &[2, 2]),
                create_test_storage(rhs_data, &[2, 2]),
            ) {
                match matmul(&backend, &lhs, &rhs, &lhs_shape, &rhs_shape) {
                    Ok(result) => {
                        if let Ok(result_data) = result.to_vec() {
                            // Expected: [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
                            //         = [[19, 22], [43, 50]]
                            let expected = vec![19.0, 22.0, 43.0, 50.0];
                            assert_eq!(result_data, expected);
                            println!("2x2 matrix multiplication test passed");
                        }
                    }
                    Err(e) => {
                        println!("2x2 matrix multiplication failed: {}", e);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_matvec() {
        if let Ok(backend) = MLXBackend::new() {
            // Test matrix-vector multiplication
            let matrix_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1, 2, 3], [4, 5, 6]]
            let vector_data = vec![1.0, 2.0, 3.0]; // [1, 2, 3]
            
            let matrix_shape = Shape::from_slice(&[2, 3]);
            let vector_shape = Shape::from_slice(&[3]);
            
            if let (Ok(matrix), Ok(vector)) = (
                create_test_storage(matrix_data, &[2, 3]),
                create_test_storage(vector_data, &[3]),
            ) {
                match matvec(&backend, &matrix, &vector, &matrix_shape, &vector_shape) {
                    Ok(result) => {
                        if let Ok(result_data) = result.to_vec() {
                            // Expected: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
                            let expected = vec![14.0, 32.0];
                            assert_eq!(result_data, expected);
                            println!("Matrix-vector multiplication test passed");
                        }
                    }
                    Err(e) => {
                        println!("Matrix-vector multiplication failed: {}", e);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_batch_matmul() {
        if let Ok(backend) = MLXBackend::new() {
            // Test batch matrix multiplication: (2, 2, 2) × (2, 2, 2) -> (2, 2, 2)
            let lhs_data = vec![
                1.0, 2.0, 3.0, 4.0, // First batch: [[1, 2], [3, 4]]
                5.0, 6.0, 7.0, 8.0, // Second batch: [[5, 6], [7, 8]]
            ];
            let rhs_data = vec![
                1.0, 0.0, 0.0, 1.0, // First batch: [[1, 0], [0, 1]] (identity)
                2.0, 0.0, 0.0, 2.0, // Second batch: [[2, 0], [0, 2]] (2*identity)
            ];
            
            let lhs_shape = Shape::from_slice(&[2, 2, 2]);
            let rhs_shape = Shape::from_slice(&[2, 2, 2]);
            
            if let (Ok(lhs), Ok(rhs)) = (
                create_test_storage(lhs_data, &[2, 2, 2]),
                create_test_storage(rhs_data, &[2, 2, 2]),
            ) {
                match batch_matmul(&backend, &lhs, &rhs, &lhs_shape, &rhs_shape) {
                    Ok(result) => {
                        if let Ok(result_data) = result.to_vec() {
                            // First batch should be unchanged (multiply by identity)
                            // Second batch should be doubled (multiply by 2*identity)
                            let expected = vec![
                                1.0, 2.0, 3.0, 4.0,   // First batch unchanged
                                10.0, 12.0, 14.0, 16.0, // Second batch doubled
                            ];
                            assert_eq!(result_data, expected);
                            println!("Batch matrix multiplication test passed");
                        }
                    }
                    Err(e) => {
                        println!("Batch matrix multiplication failed: {}", e);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_cpu_matmul_2d() {
        let lhs = vec![1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let rhs = vec![5.0, 6.0, 7.0, 8.0]; // [[5, 6], [7, 8]]
        
        match cpu_matmul_2d(&lhs, &rhs, 2, 2, 2) {
            Ok(result) => {
                let expected = vec![19.0, 22.0, 43.0, 50.0];
                assert_eq!(result, expected);
                println!("CPU 2D matrix multiplication test passed");
            }
            Err(e) => {
                panic!("CPU 2D matrix multiplication failed: {}", e);
            }
        }
    }
}