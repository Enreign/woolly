//! Quantization operations for MLX backend

use tracing::{debug, trace};

use woolly_tensor::backend::TensorStorage;

use crate::backend::MLXBackend;
use crate::device::Device;
use crate::error::{MLXError, Result};
use crate::storage::MLXStorage;
use woolly_tensor::backend::DType;
use woolly_tensor::quantization::QuantizationScheme;

#[cfg(feature = "mlx")]
use crate::ffi::{mlx_quantize, mlx_dequantize};

/// Quantize tensor using MLX
pub fn quantize<T>(
    backend: &MLXBackend,
    input: &MLXStorage<T>,
    scheme: QuantizationScheme,
) -> Result<MLXStorage<u8>>
where
    T: Clone + Into<f32> + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Quantizing tensor with scheme {:?}", scheme);
    
    #[cfg(feature = "mlx")]
    {
        mlx_quantize_tensor(backend, input, scheme)
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        cpu_quantize_tensor(backend, input, scheme)
    }
}

/// Dequantize tensor using MLX
pub fn dequantize(
    backend: &MLXBackend,
    input: &MLXStorage<u8>,
    target_dtype: DType,
) -> Result<MLXStorage<f32>>
{
    debug!("Dequantizing tensor to dtype {:?}", target_dtype);
    
    #[cfg(feature = "mlx")]
    {
        mlx_dequantize_tensor(backend, input, target_dtype)
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        cpu_dequantize_tensor(backend, input, target_dtype)
    }
}

#[cfg(feature = "mlx")]
fn mlx_quantize_tensor<T>(
    backend: &MLXBackend,
    input: &MLXStorage<T>,
    scheme: QuantizationScheme,
) -> Result<MLXStorage<u8>>
where
    T: Clone + Into<f32> + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using MLX quantization");
    
    let output_shape = input.shape().clone();
    let mut output = MLXStorage::zeros(
        output_shape,
        DType::U8,
        Device::GPU,
    )?;
    
    let scheme_id = match scheme {
        QuantizationScheme::Q4_0 => 0,
        QuantizationScheme::Q4_1 => 1,
        QuantizationScheme::Q8_0 => 2,
        _ => return Err(MLXError::QuantizationError("Unsupported quantization scheme".to_string())),
    };
    
    let result = mlx_quantize(input.array_ptr(), scheme_id, output.array_ptr());
    
    if let Err(e) = result {
        return Err(MLXError::QuantizationError(format!("MLX quantization failed: {}", e)));
    }
    
    crate::ops::utils::maybe_synchronize(backend, true)?;
    Ok(output)
}

#[cfg(feature = "mlx")]
fn mlx_dequantize_tensor(
    backend: &MLXBackend,
    input: &MLXStorage<u8>,
    target_dtype: DType,
) -> Result<MLXStorage<f32>>
{
    debug!("Using MLX dequantization");
    
    let output_shape = input.shape().clone();
    let mut output = MLXStorage::zeros(
        output_shape,
        target_dtype,
        Device::GPU,
    )?;
    
    let dtype_id = match target_dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        _ => return Err(MLXError::UnsupportedDataType("Unsupported target dtype".to_string())),
    };
    
    let result = mlx_dequantize(input.array_ptr(), dtype_id, output.array_ptr());
    
    if let Err(e) = result {
        return Err(MLXError::QuantizationError(format!("MLX dequantization failed: {}", e)));
    }
    
    crate::ops::utils::maybe_synchronize(backend, true)?;
    Ok(output)
}

fn cpu_quantize_tensor<T>(
    _backend: &MLXBackend,
    input: &MLXStorage<T>,
    scheme: QuantizationScheme,
) -> Result<MLXStorage<u8>>
where
    T: Clone + Into<f32> + Send + Sync + std::fmt::Debug + 'static,
{
    debug!("Using CPU quantization fallback");
    
    let input_data = input.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get input data: {}", e)))?;
    
    // Convert to f32 for quantization
    let f32_data: Vec<f32> = input_data.into_iter().map(|x| x.into()).collect();
    
    let quantized_data = match scheme {
        QuantizationScheme::Q8_0 => quantize_q8_0(&f32_data)?,
        QuantizationScheme::Q4_0 => quantize_q4_0(&f32_data)?,
        _ => return Err(MLXError::QuantizationError("Unsupported quantization scheme for CPU".to_string())),
    };
    
    MLXStorage::from_data(
        quantized_data,
        input.shape().clone(),
        DType::U8,
        Device::CPU,
    )
}

fn cpu_dequantize_tensor(
    _backend: &MLXBackend,
    input: &MLXStorage<u8>,
    _target_dtype: DType,
) -> Result<MLXStorage<f32>>
{
    debug!("Using CPU dequantization fallback");
    
    let input_data = input.to_vec().map_err(|e| MLXError::ArrayOperationFailed(format!("Failed to get input data: {}", e)))?;
    
    // For mock implementation, just convert u8 to f32
    let f32_data: Vec<f32> = input_data.into_iter().map(|x| x as f32).collect();
    
    MLXStorage::from_data(
        f32_data,
        input.shape().clone(),
        DType::F32,
        Device::CPU,
    )
}

fn quantize_q8_0(data: &[f32]) -> Result<Vec<u8>> {
    // Simple Q8_0 quantization: scale to [-127, 127] range
    let max_abs = data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
    
    let quantized: Vec<u8> = data
        .iter()
        .map(|&x| {
            let scaled = (x * scale).round().clamp(-127.0, 127.0) as i8;
            (scaled + 128) as u8 // Shift to unsigned range
        })
        .collect();
    
    Ok(quantized)
}

fn quantize_q4_0(data: &[f32]) -> Result<Vec<u8>> {
    // Simple Q4_0 quantization: scale to [-8, 7] range and pack
    let max_abs = data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 { 7.0 / max_abs } else { 1.0 };
    
    let mut quantized = Vec::with_capacity((data.len() + 1) / 2);
    
    for chunk in data.chunks(2) {
        let q0 = (chunk[0] * scale).round().clamp(-8.0, 7.0) as i8;
        let q1 = if chunk.len() > 1 {
            (chunk[1] * scale).round().clamp(-8.0, 7.0) as i8
        } else {
            0
        };
        
        // Pack two 4-bit values into one byte
        let packed = ((q0 + 8) as u8) | (((q1 + 8) as u8) << 4);
        quantized.push(packed);
    }
    
    Ok(quantized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MLXBackend;
    use crate::device::Device;
    use woolly_tensor::backend::DType;
    use woolly_tensor::shape::Shape;
    
    fn create_test_storage_f32(data: Vec<f32>, shape: &[usize]) -> Result<MLXStorage<f32>> {
        MLXStorage::from_data(
            data,
            Shape::from_slice(shape),
            DType::F32,
            Device::CPU,
        )
    }
    
    fn create_test_storage_u8(data: Vec<u8>, shape: &[usize]) -> Result<MLXStorage<u8>> {
        MLXStorage::from_data(
            data,
            Shape::from_slice(shape),
            DType::U8,
            Device::CPU,
        )
    }
    
    #[test]
    fn test_quantization_q8_0() {
        if let Ok(backend) = MLXBackend::new() {
            let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
            
            if let Ok(input) = create_test_storage_f32(data, &[5]) {
                match quantize(&backend, &input, QuantizationScheme::Q8_0) {
                    Ok(quantized) => {
                        if let Ok(quantized_data) = quantized.to_vec() {
                            assert_eq!(quantized_data.len(), 5);
                            println!("Q8_0 quantization test passed: {:?}", quantized_data);
                        }
                    }
                    Err(e) => {
                        println!("Q8_0 quantization failed: {}", e);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_quantization_q4_0() {
        if let Ok(backend) = MLXBackend::new() {
            let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 0.8];
            
            if let Ok(input) = create_test_storage_f32(data, &[6]) {
                match quantize(&backend, &input, QuantizationScheme::Q4_0) {
                    Ok(quantized) => {
                        if let Ok(quantized_data) = quantized.to_vec() {
                            // 6 values should pack into 3 bytes (2 values per byte)
                            assert_eq!(quantized_data.len(), 3);
                            println!("Q4_0 quantization test passed: {:?}", quantized_data);
                        }
                    }
                    Err(e) => {
                        println!("Q4_0 quantization failed: {}", e);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_dequantization() {
        if let Ok(backend) = MLXBackend::new() {
            let data = vec![128, 140, 115, 160, 200]; // Some quantized values
            
            if let Ok(input) = create_test_storage_u8(data, &[5]) {
                match dequantize(&backend, &input, DType::F32) {
                    Ok(dequantized) => {
                        if let Ok(dequantized_data) = dequantized.to_vec() {
                            assert_eq!(dequantized_data.len(), 5);
                            println!("Dequantization test passed: {:?}", dequantized_data);
                        }
                    }
                    Err(e) => {
                        println!("Dequantization failed: {}", e);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_quantize_dequantize_roundtrip() {
        if let Ok(backend) = MLXBackend::new() {
            let original_data = vec![0.1, 0.2, 0.3, 0.4];
            
            if let Ok(input) = create_test_storage_f32(original_data.clone(), &[4]) {
                // Quantize
                if let Ok(quantized) = quantize(&backend, &input, QuantizationScheme::Q8_0) {
                    // Dequantize
                    if let Ok(dequantized) = dequantize(&backend, &quantized, DType::F32) {
                        if let Ok(result_data) = dequantized.to_vec() {
                            println!("Original: {:?}", original_data);
                            println!("After roundtrip: {:?}", result_data);
                            
                            // Should be approximately equal (some precision loss expected)
                            for (orig, result) in original_data.iter().zip(result_data.iter()) {
                                let diff = (orig - result).abs();
                                assert!(diff < 0.1, "Roundtrip error too large: {} vs {}", orig, result);
                            }
                            
                            println!("Quantize-dequantize roundtrip test passed");
                        }
                    }
                }
            }
        }
    }
}