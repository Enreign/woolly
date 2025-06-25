//! MLX tensor backend implementation
//!
//! This module provides the main MLX backend that implements the TensorBackend trait,
//! enabling GPU-accelerated tensor operations on Apple Silicon.

use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use tracing::{debug, info, trace, warn};

use woolly_tensor::backend::{
    DType, Device as TensorDevice, QuantizationScheme, TensorBackend, TensorError, Result as TensorResult,
};
use woolly_tensor::shape::Shape;

use crate::device::{Device, MLXDevice};
use crate::error::{MLXError, Result};
use crate::ops::{self, MatMulOp, ElementWiseOp, ReductionOp, QuantizationOp};
use crate::storage::MLXStorage;

/// MLX backend for tensor operations
#[derive(Debug, Clone)]
pub struct MLXBackend {
    /// Primary device for operations
    device: MLXDevice,
    /// Whether to use automatic mixed precision
    use_amp: bool,
    /// Memory pool size limit (bytes)
    memory_limit: Option<u64>,
}

impl MLXBackend {
    /// Create a new MLX backend with the default device
    pub fn new() -> Result<Self> {
        crate::init()?;
        
        if !crate::is_available() {
            return Err(MLXError::NotAvailable(
                "MLX not available on this platform".to_string()
            ));
        }
        
        let device = MLXDevice::default()?;
        info!("Created MLX backend with device: {:?}", device.device_type());
        
        Ok(Self {
            device,
            use_amp: false,
            memory_limit: None,
        })
    }
    
    /// Create a new MLX backend with a specific device
    pub fn with_device(device_type: Device) -> Result<Self> {
        crate::init()?;
        
        if !crate::is_available() {
            return Err(MLXError::NotAvailable(
                "MLX not available on this platform".to_string()
            ));
        }
        
        let device = MLXDevice::new(device_type)?;
        info!("Created MLX backend with device: {:?}", device.device_type());
        
        Ok(Self {
            device,
            use_amp: false,
            memory_limit: None,
        })
    }
    
    /// Enable or disable automatic mixed precision
    pub fn with_amp(mut self, use_amp: bool) -> Self {
        self.use_amp = use_amp;
        self
    }
    
    /// Set memory limit
    pub fn with_memory_limit(mut self, limit: u64) -> Result<Self> {
        self.device.set_memory_limit(limit)?;
        self.memory_limit = Some(limit);
        Ok(self)
    }
    
    /// Get the current device
    pub fn device(&self) -> &MLXDevice {
        &self.device
    }
    
    /// Check if automatic mixed precision is enabled
    pub fn uses_amp(&self) -> bool {
        self.use_amp
    }
    
    /// Synchronize all operations on this backend
    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
    }
    
    /// Get memory usage statistics
    pub fn memory_usage(&self) -> Result<crate::device::MemoryUsage> {
        self.device.memory_usage()
    }
    
    /// Convert data type to MLX-compatible type
    fn dtype_to_mlx_compatible(&self, dtype: DType) -> DType {
        if self.use_amp {
            match dtype {
                DType::F32 => DType::F16, // Use F16 for AMP
                other => other,
            }
        } else {
            dtype
        }
    }
    
    /// Validate tensor shapes for operation
    fn validate_shapes(&self, lhs_shape: &Shape, rhs_shape: &Shape, op: &str) -> TensorResult<()> {
        if lhs_shape.numel() != rhs_shape.numel() {
            return Err(TensorError::incompatible_shapes(
                "MLX_BACKEND_SHAPE_MISMATCH",
                "Tensor shapes have incompatible element counts",
                "MLX backend operation",
                format!("{:?}", lhs_shape),
                format!("{:?}", rhs_shape),
                "Ensure tensors have the same total number of elements"
            ));
        }
        
        trace!("Shape validation passed for {} operation", op);
        Ok(())
    }
    
    /// Create storage with optimal device placement
    fn create_optimal_storage<T>(
        &self,
        shape: &Shape,
        dtype: DType,
    ) -> Result<MLXStorage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static + num_traits::Zero,
    {
        // Choose optimal device based on tensor size and operation type
        let device_type = if shape.numel() > 1024 {
            // Large tensors benefit from GPU
            Device::GPU
        } else {
            // Small tensors might be faster on CPU due to launch overhead
            Device::CPU
        };
        
        // Try to create on preferred device, fallback to available device
        match MLXStorage::zeros(shape.clone(), dtype, device_type) {
            Ok(storage) => Ok(storage),
            Err(_) => {
                // Fallback to backend's default device
                MLXStorage::zeros(shape.clone(), dtype, self.device.device_type())
            }
        }
    }
}

impl Default for MLXBackend {
    fn default() -> Self {
        Self::new().unwrap_or_else(|e| {
            warn!("Failed to create MLX backend: {}, using fallback", e);
            // Create a minimal fallback backend
            Self {
                device: MLXDevice::new(Device::CPU).expect("CPU device should always be available"),
                use_amp: false,
                memory_limit: None,
            }
        })
    }
}

impl TensorBackend for MLXBackend {
    type Storage<T> = MLXStorage<T>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static;
    
    fn name(&self) -> &'static str {
        "mlx"
    }
    
    fn device(&self) -> TensorDevice {
        self.device.device_type().into()
    }
    
    fn supports_dtype(&self, dtype: DType) -> bool {
        match dtype {
            DType::F32 => true,
            DType::F16 => self.device.supports_dtype("f16"),
            DType::BF16 => self.device.supports_dtype("bf16"),
            DType::I32 => true,
            DType::U8 => true,
            DType::I64 => true,
            DType::F64 => false, // MLX typically doesn't support F64
            DType::Bool => true,
            DType::Quantized(_) => true, // We support quantized types
        }
    }
    
    fn zeros<T>(&self, shape: &Shape, dtype: DType) -> TensorResult<Self::Storage<T>>
    where
        T: num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let mlx_dtype = self.dtype_to_mlx_compatible(dtype);
        trace!("Creating zeros tensor with shape {:?} and dtype {:?}", shape, mlx_dtype);
        
        self.create_optimal_storage(shape, mlx_dtype)
            .map_err(|e| e.into())
    }
    
    fn ones<T>(&self, shape: &Shape, dtype: DType) -> TensorResult<Self::Storage<T>>
    where
        T: num_traits::One + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let mlx_dtype = self.dtype_to_mlx_compatible(dtype);
        trace!("Creating ones tensor with shape {:?} and dtype {:?}", shape, mlx_dtype);
        
        MLXStorage::ones(shape.clone(), mlx_dtype, self.device.device_type())
            .map_err(|e| e.into())
    }
    
    fn full<T>(&self, shape: &Shape, value: T, dtype: DType) -> TensorResult<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let mlx_dtype = self.dtype_to_mlx_compatible(dtype);
        trace!("Creating full tensor with shape {:?}, value {:?}, and dtype {:?}", 
               shape, value, mlx_dtype);
        
        MLXStorage::full(shape.clone(), value, mlx_dtype, self.device.device_type())
            .map_err(|e| e.into())
    }
    
    fn from_slice<T>(&self, data: &[T], shape: &Shape, dtype: DType) -> TensorResult<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let mlx_dtype = self.dtype_to_mlx_compatible(dtype);
        trace!("Creating tensor from slice with shape {:?} and dtype {:?}", shape, mlx_dtype);
        
        if shape.numel() != data.len() {
            return Err(TensorError::incompatible_shapes(
                "MLX_FROM_SLICE_SIZE_MISMATCH",
                "Data length does not match shape elements",
                "MLX from_slice operation",
                format!("data length: {}", data.len()),
                format!("shape elements: {}", shape.numel()),
                "Ensure data length matches the total elements in shape"
            ));
        }
        
        MLXStorage::from_data(data.to_vec(), shape.clone(), mlx_dtype, self.device.device_type())
            .map_err(|e| e.into())
    }
    
    fn add<T>(&self, lhs: &Self::Storage<T>, rhs: &Self::Storage<T>) -> TensorResult<Self::Storage<T>>
    where
        T: Add<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        trace!("Performing element-wise addition");
        
        // Use MLX operations module for actual computation
        ops::element_wise::add(self, lhs, rhs).map_err(|e| e.into())
    }
    
    fn sub<T>(&self, lhs: &Self::Storage<T>, rhs: &Self::Storage<T>) -> TensorResult<Self::Storage<T>>
    where
        T: Sub<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        trace!("Performing element-wise subtraction");
        
        ops::element_wise::sub(self, lhs, rhs).map_err(|e| e.into())
    }
    
    fn mul<T>(&self, lhs: &Self::Storage<T>, rhs: &Self::Storage<T>) -> TensorResult<Self::Storage<T>>
    where
        T: Mul<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        trace!("Performing element-wise multiplication");
        
        ops::element_wise::mul(self, lhs, rhs).map_err(|e| e.into())
    }
    
    fn div<T>(&self, lhs: &Self::Storage<T>, rhs: &Self::Storage<T>) -> TensorResult<Self::Storage<T>>
    where
        T: Div<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        trace!("Performing element-wise division");
        
        ops::element_wise::div(self, lhs, rhs).map_err(|e| e.into())
    }
    
    fn matmul<T>(
        &self,
        lhs: &Self::Storage<T>,
        rhs: &Self::Storage<T>,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
    ) -> TensorResult<Self::Storage<T>>
    where
        T: Add<Output = T> + Mul<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        debug!("Performing matrix multiplication: {:?} × {:?}", lhs_shape, rhs_shape);
        
        // Validate matrix multiplication shapes
        if lhs_shape.ndim() < 2 || rhs_shape.ndim() < 2 {
            return Err(TensorError::invalid_shape(
                "MLX_MATMUL_NON_2D_TENSORS",
                "Matrix multiplication requires at least 2D tensors",
                format!("lhs: {:?}, rhs: {:?}", lhs_shape, rhs_shape),
                "MLX matrix multiplication",
                "Non-2D tensors provided",
                "Ensure both tensors have at least 2 dimensions"
            ));
        }
        
        let lhs_cols = lhs_shape[lhs_shape.ndim() - 1];
        let rhs_rows = rhs_shape[rhs_shape.ndim() - 2];
        
        if lhs_cols != rhs_rows {
            return Err(TensorError::incompatible_shapes(
                "MLX_MATMUL_INNER_DIM_MISMATCH",
                "Matrix multiplication inner dimensions do not match",
                "MLX matrix multiplication",
                format!("lhs cols: {}", lhs_cols),
                format!("rhs rows: {}", rhs_rows),
                "Ensure left matrix columns match right matrix rows"
            ));
        }
        
        ops::matmul::matmul(self, lhs, rhs, lhs_shape, rhs_shape).map_err(|e| e.into())
    }
    
    fn sum<T>(
        &self,
        input: &Self::Storage<T>,
        shape: &Shape,
        axes: &[usize],
        keep_dims: bool,
    ) -> TensorResult<Self::Storage<T>>
    where
        T: Add<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        trace!("Performing sum reduction along axes {:?}", axes);
        
        ops::reduction::sum(self, input, shape, axes, keep_dims).map_err(|e| e.into())
    }
    
    fn mean<T>(
        &self,
        input: &Self::Storage<T>,
        shape: &Shape,
        axes: &[usize],
        keep_dims: bool,
    ) -> TensorResult<Self::Storage<T>>
    where
        T: Add<Output = T> + Div<Output = T> + num_traits::Zero + Clone + From<f32> + Send + Sync + std::fmt::Debug + 'static,
    {
        trace!("Performing mean reduction along axes {:?}", axes);
        
        ops::reduction::mean(self, input, shape, axes, keep_dims).map_err(|e| e.into())
    }
    
    fn reshape<T>(
        &self,
        input: &Self::Storage<T>,
        old_shape: &Shape,
        new_shape: &Shape,
    ) -> TensorResult<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        trace!("Reshaping tensor from {:?} to {:?}", old_shape, new_shape);
        
        if old_shape.numel() != new_shape.numel() {
            return Err(TensorError::incompatible_shapes(
                "MLX_RESHAPE_SIZE_MISMATCH",
                "Reshape operation requires same total number of elements",
                "MLX reshape operation",
                format!("old: {} elements", old_shape.numel()),
                format!("new: {} elements", new_shape.numel()),
                "Ensure new shape has the same total number of elements"
            ));
        }
        
        ops::shape_ops::reshape(self, input, old_shape, new_shape).map_err(|e| e.into())
    }
    
    fn transpose<T>(
        &self,
        input: &Self::Storage<T>,
        shape: &Shape,
        axes: &[usize],
    ) -> TensorResult<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        trace!("Transposing tensor with axes {:?}", axes);
        
        if axes.len() != shape.ndim() {
            return Err(TensorError::invalid_shape(
                "MLX_TRANSPOSE_AXES_MISMATCH",
                format!("Transpose axes length {} doesn't match tensor dimensions {}", 
                    axes.len(), shape.ndim()),
                format!("{:?}", shape),
                "MLX transpose operation",
                "Axes length mismatch",
                "Provide exactly one axis index for each tensor dimension"
            ));
        }
        
        ops::shape_ops::transpose(self, input, shape, axes).map_err(|e| e.into())
    }
    
    fn to_device<T>(&self, input: &Self::Storage<T>, device: TensorDevice) -> TensorResult<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let target_device = match device {
            TensorDevice::Metal => Device::GPU,
            TensorDevice::Cpu => Device::CPU,
            _ => return Err(TensorError::UnsupportedOperation {
                code: "MLX_UNSUPPORTED_TARGET_DEVICE",
                message: "Unsupported target device for MLX".to_string(),
                operation: "device transfer".to_string(),
                backend: "MLX".to_string(),
                dtype: "unknown".to_string(),
                suggestion: "Use CPU or Metal device for MLX backend".to_string(),
            }),
        };
        
        trace!("Transferring tensor to device {:?}", target_device);
        
        input.to_device(target_device).map_err(|e| e.into())
    }
    
    fn quantize<T>(&self, input: &Self::Storage<T>, scheme: QuantizationScheme) -> TensorResult<Self::Storage<u8>>
    where
        T: Clone + Into<f32> + Send + Sync + std::fmt::Debug + 'static,
    {
        debug!("Quantizing tensor with scheme {:?}", scheme);
        
        ops::quantization::quantize(self, input, scheme).map_err(|e| e.into())
    }
    
    fn dequantize(&self, input: &Self::Storage<u8>, target_dtype: DType) -> TensorResult<Self::Storage<f32>> {
        debug!("Dequantizing tensor to dtype {:?}", target_dtype);
        
        ops::quantization::dequantize(self, input, target_dtype).map_err(|e| e.into())
    }
}

/// Builder for creating MLX backend with specific configuration
pub struct MLXBackendBuilder {
    device_type: Option<Device>,
    use_amp: bool,
    memory_limit: Option<u64>,
}

impl MLXBackendBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            device_type: None,
            use_amp: false,
            memory_limit: None,
        }
    }
    
    /// Set the device type
    pub fn device(mut self, device: Device) -> Self {
        self.device_type = Some(device);
        self
    }
    
    /// Enable automatic mixed precision
    pub fn with_amp(mut self) -> Self {
        self.use_amp = true;
        self
    }
    
    /// Set memory limit in bytes
    pub fn memory_limit(mut self, limit: u64) -> Self {
        self.memory_limit = Some(limit);
        self
    }
    
    /// Build the MLX backend
    pub fn build(self) -> Result<MLXBackend> {
        let mut backend = if let Some(device) = self.device_type {
            MLXBackend::with_device(device)?
        } else {
            MLXBackend::new()?
        };
        
        backend = backend.with_amp(self.use_amp);
        
        if let Some(limit) = self.memory_limit {
            backend = backend.with_memory_limit(limit)?;
        }
        
        Ok(backend)
    }
}

impl Default for MLXBackendBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use woolly_tensor::shape::Shape;
    
    #[test]
    fn test_backend_creation() {
        match MLXBackend::new() {
            Ok(backend) => {
                println!("Created MLX backend: {}", backend.name());
                assert_eq!(backend.name(), "mlx");
            }
            Err(e) => {
                println!("Failed to create MLX backend: {}", e);
                // This is expected on non-Apple Silicon platforms
            }
        }
    }
    
    #[test]
    fn test_backend_builder() {
        let result = MLXBackendBuilder::new()
            .device(Device::CPU)
            .with_amp()
            .memory_limit(1024 * 1024 * 1024) // 1GB
            .build();
        
        match result {
            Ok(backend) => {
                println!("Built MLX backend with custom config");
                assert!(backend.uses_amp());
            }
            Err(e) => {
                println!("Failed to build MLX backend: {}", e);
            }
        }
    }
    
    #[test]
    fn test_dtype_support() {
        if let Ok(backend) = MLXBackend::new() {
            assert!(backend.supports_dtype(DType::F32));
            assert!(backend.supports_dtype(DType::F16));
            assert!(backend.supports_dtype(DType::I32));
            assert!(backend.supports_dtype(DType::U8));
            assert!(!backend.supports_dtype(DType::F64)); // MLX doesn't support F64
            
            println!("Data type support verified");
        }
    }
    
    #[test]
    fn test_tensor_creation() {
        if let Ok(backend) = MLXBackend::new() {
            let shape = Shape::from_slice(&[2, 3]);
            
            // Test zeros
            match backend.zeros::<f32>(&shape, DType::F32) {
                Ok(tensor) => {
                    assert_eq!(tensor.len(), 6);
                    println!("Created zeros tensor successfully");
                }
                Err(e) => {
                    println!("Failed to create zeros tensor: {}", e);
                }
            }
            
            // Test ones
            match backend.ones::<f32>(&shape, DType::F32) {
                Ok(tensor) => {
                    assert_eq!(tensor.len(), 6);
                    println!("Created ones tensor successfully");
                }
                Err(e) => {
                    println!("Failed to create ones tensor: {}", e);
                }
            }
            
            // Test from slice
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            match backend.from_slice(&data, &shape, DType::F32) {
                Ok(tensor) => {
                    assert_eq!(tensor.len(), 6);
                    println!("Created tensor from slice successfully");
                }
                Err(e) => {
                    println!("Failed to create tensor from slice: {}", e);
                }
            }
        }
    }
    
    #[test]
    fn test_memory_stats() {
        if let Ok(backend) = MLXBackend::new() {
            match backend.memory_usage() {
                Ok(stats) => {
                    println!("Memory usage: {:?}", stats);
                    assert!(stats.total > 0);
                }
                Err(e) => {
                    println!("Failed to get memory stats: {}", e);
                }
            }
        }
    }
    
    #[test]
    fn test_device_info() {
        if let Ok(backend) = MLXBackend::new() {
            let device = backend.device();
            println!("Backend device: {:?}", device.device_type());
            println!("Device available: {}", device.is_available());
            
            let properties = device.properties();
            println!("Device properties: {:?}", properties);
        }
    }
}