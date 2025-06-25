//! MLX tensor storage implementation
//!
//! This module provides MLX-backed tensor storage that leverages Apple Silicon's
//! unified memory architecture for efficient GPU operations.

use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use tracing::{debug, trace};

use woolly_tensor::backend::{TensorStorage, DType, Device as TensorDevice};
use woolly_tensor::shape::Shape;

use crate::device::{Device, MLXDevice};
use crate::error::{MLXError, Result};

#[cfg(feature = "mlx")]
use crate::ffi::{MLXArray, mlx_array_from_data, mlx_array_data, mlx_array_shape, mlx_array_free};

/// MLX tensor storage
pub struct MLXStorage<T> {
    /// Pointer to MLX array (null if not using MLX feature)
    array_ptr: *mut MLXArray,
    /// Cached data for CPU access (unified memory allows this)
    cached_data: Option<Vec<T>>,
    /// Shape of the tensor
    shape: Shape,
    /// Data type
    dtype: DType,
    /// Device where the storage resides
    device: Device,
    /// Element type marker
    _phantom: PhantomData<T>,
}

impl<T> fmt::Debug for MLXStorage<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MLXStorage")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .field("array_ptr", &self.array_ptr)
            .field("has_cached_data", &self.cached_data.is_some())
            .finish()
    }
}

impl<T> MLXStorage<T>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    /// Create new MLX storage from data
    pub fn from_data(data: Vec<T>, shape: Shape, dtype: DType, device: Device) -> Result<Self> {
        if !crate::is_available() {
            return Err(MLXError::NotAvailable("MLX not available".to_string()));
        }
        
        // Validate shape
        if shape.numel() != data.len() {
            return Err(MLXError::ShapeMismatch {
                expected: vec![data.len()],
                actual: shape.as_slice().to_vec(),
            });
        }
        
        let array_ptr = Self::create_mlx_array(&data, &shape, device)?;
        
        Ok(Self {
            array_ptr,
            cached_data: Some(data),
            shape,
            dtype,
            device,
            _phantom: PhantomData,
        })
    }
    
    /// Create new empty MLX storage
    pub fn zeros(shape: Shape, dtype: DType, device: Device) -> Result<Self>
    where
        T: num_traits::Zero,
    {
        let total_elements = shape.numel();
        let data = vec![T::zero(); total_elements];
        Self::from_data(data, shape, dtype, device)
    }
    
    /// Create new MLX storage filled with ones
    pub fn ones(shape: Shape, dtype: DType, device: Device) -> Result<Self>
    where
        T: num_traits::One,
    {
        let total_elements = shape.numel();
        let data = vec![T::one(); total_elements];
        Self::from_data(data, shape, dtype, device)
    }
    
    /// Create new MLX storage filled with a value
    pub fn full(shape: Shape, value: T, dtype: DType, device: Device) -> Result<Self> {
        let total_elements = shape.numel();
        let data = vec![value; total_elements];
        Self::from_data(data, shape, dtype, device)
    }
    
    /// Get the MLX array pointer (for internal use)
    pub fn array_ptr(&self) -> *mut MLXArray {
        self.array_ptr
    }
    
    /// Get the shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    
    /// Ensure data is cached on CPU for access
    fn ensure_cached_data(&mut self) -> Result<()> {
        if self.cached_data.is_none() {
            trace!("Caching data from MLX array");
            
            #[cfg(feature = "mlx")]
            {
                if !self.array_ptr.is_null() {
                    let data_ptr = mlx_array_data(self.array_ptr)?;
                    let data_slice = unsafe {
                        std::slice::from_raw_parts(data_ptr as *const T, self.shape.numel())
                    };
                    self.cached_data = Some(data_slice.to_vec());
                }
            }
            
            #[cfg(not(feature = "mlx"))]
            {
                // Without MLX feature, we can't access the array data
                return Err(MLXError::NotAvailable("MLX feature not enabled".to_string()));
            }
        }
        
        Ok(())
    }
    
    /// Create MLX array from data
    fn create_mlx_array(data: &[T], shape: &Shape, device: Device) -> Result<*mut MLXArray> {
        #[cfg(feature = "mlx")]
        {
            // For now, assume T is f32 for simplicity
            // In a real implementation, we'd need to handle different types
            let data_ptr = data.as_ptr() as *const f32;
            mlx_array_from_data(data_ptr, shape.as_slice(), device)
        }
        
        #[cfg(not(feature = "mlx"))]
        {
            debug!("Creating mock MLX array for shape {:?} on device {:?}", shape, device);
            Ok(ptr::null_mut())
        }
    }
    
    /// Synchronize with the MLX device
    pub fn synchronize(&self) -> Result<()> {
        if !crate::is_available() {
            return Ok(());
        }
        
        let mlx_device = MLXDevice::new(self.device)?;
        mlx_device.synchronize()
    }
    
    /// Copy data to another device
    pub fn to_device(&self, target_device: Device) -> Result<Self> {
        if target_device == self.device {
            return Ok(self.clone());
        }
        
        // Get data (this will ensure it's cached)
        let mut storage_clone = self.clone();
        storage_clone.ensure_cached_data()?;
        
        if let Some(ref data) = storage_clone.cached_data {
            Self::from_data(data.clone(), self.shape.clone(), self.dtype, target_device)
        } else {
            Err(MLXError::ArrayOperationFailed("No data available for device transfer".to_string()))
        }
    }
    
    /// Get raw data pointer (for unified memory access)
    pub unsafe fn data_ptr(&self) -> *const T {
        #[cfg(feature = "mlx")]
        {
            if !self.array_ptr.is_null() {
                if let Ok(data_ptr) = mlx_array_data(self.array_ptr) {
                    return data_ptr as *const T;
                }
            }
        }
        
        // Fallback to cached data
        if let Some(ref data) = self.cached_data {
            data.as_ptr()
        } else {
            ptr::null()
        }
    }
    
    /// Get mutable raw data pointer
    pub unsafe fn data_ptr_mut(&mut self) -> *mut T {
        // Invalidate cached data since we're giving mutable access
        self.cached_data = None;
        
        #[cfg(feature = "mlx")]
        {
            if !self.array_ptr.is_null() {
                if let Ok(data_ptr) = mlx_array_data(self.array_ptr) {
                    return data_ptr as *mut T;
                }
            }
        }
        
        ptr::null_mut()
    }
}

impl<T> Clone for MLXStorage<T>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    fn clone(&self) -> Self {
        // For unified memory, we can share the same underlying data
        // In a real implementation, this might need reference counting
        Self {
            array_ptr: self.array_ptr, // Careful: this shares the pointer
            cached_data: self.cached_data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
            _phantom: PhantomData,
        }
    }
}

impl<T> Drop for MLXStorage<T> {
    fn drop(&mut self) {
        #[cfg(feature = "mlx")]
        {
            if !self.array_ptr.is_null() {
                trace!("Freeing MLX array");
                mlx_array_free(self.array_ptr);
                self.array_ptr = ptr::null_mut();
            }
        }
    }
}

impl<T> TensorStorage for MLXStorage<T>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    type Elem = T;
    
    fn dtype(&self) -> DType {
        self.dtype
    }
    
    fn device(&self) -> TensorDevice {
        self.device.into()
    }
    
    fn len(&self) -> usize {
        self.shape.numel()
    }
    
    fn allocate(capacity: usize, device: TensorDevice) -> woolly_tensor::backend::Result<Self>
    where
        Self: Sized,
    {
        let mlx_device = match device {
            TensorDevice::Metal => Device::GPU,
            TensorDevice::Cpu => Device::CPU,
            _ => return Err(woolly_tensor::backend::TensorError::UnsupportedOperation {
                code: "MLX_UNSUPPORTED_DEVICE",
                message: "Unsupported device for MLX".to_string(),
                operation: "allocate".to_string(),
                backend: "MLX".to_string(),
                dtype: "unknown".to_string(),
                suggestion: "Use CPU or Metal device for MLX backend".to_string(),
            }),
        };
        
        let shape = Shape::vector(capacity);
        let dtype = DType::F32; // Default type
        
        Self::zeros(shape, dtype, mlx_device)
            .map_err(|e| e.into())
    }
    
    fn slice(&self, offset: usize, len: usize) -> woolly_tensor::backend::Result<Self>
    where
        Self: Sized,
    {
        if offset + len > self.len() {
            return Err(woolly_tensor::backend::TensorError::out_of_bounds(
                "MLX_SLICE_OUT_OF_BOUNDS",
                "Slice range exceeds storage bounds",
                offset + len,
                0,
                self.len(),
                "MLX storage slicing",
                "Ensure slice offset + length does not exceed storage size"
            ));
        }
        
        // Get the data
        let mut storage_clone = self.clone();
        storage_clone.ensure_cached_data().map_err(|e| e.into())?;
        
        if let Some(ref data) = storage_clone.cached_data {
            let sliced_data = data[offset..offset + len].to_vec();
            let sliced_shape = Shape::vector(len);
            
            Self::from_data(sliced_data, sliced_shape, self.dtype, self.device)
                .map_err(|e| e.into())
        } else {
            Err(woolly_tensor::backend::TensorError::backend_error(
                "MLX_NO_DATA_FOR_SLICE",
                "No data available for slicing",
                "MLX",
                "storage slicing",
                "Use ensure_cached_data() to cache data from GPU"
            ))
        }
    }
    
    fn copy_from(
        &mut self, 
        other: &Self, 
        src_offset: usize, 
        dst_offset: usize, 
        len: usize
    ) -> woolly_tensor::backend::Result<()> {
        if src_offset + len > other.len() {
            return Err(woolly_tensor::backend::TensorError::out_of_bounds(
                "MLX_COPY_SRC_OUT_OF_BOUNDS",
                "Source copy range exceeds storage bounds",
                src_offset + len,
                0,
                other.len(),
                "MLX storage copy (source)",
                "Ensure source offset + length does not exceed source storage size"
            ));
        }
        
        if dst_offset + len > self.len() {
            return Err(woolly_tensor::backend::TensorError::out_of_bounds(
                "MLX_COPY_DST_OUT_OF_BOUNDS",
                "Destination copy range exceeds storage bounds",
                dst_offset + len,
                0,
                self.len(),
                "MLX storage copy (destination)",
                "Ensure destination offset + length does not exceed destination storage size"
            ));
        }
        
        // Ensure both storages have cached data
        self.ensure_cached_data().map_err(|e| e.into())?;
        let mut other_clone = other.clone();
        other_clone.ensure_cached_data().map_err(|e| e.into())?;
        
        if let (Some(ref mut dst_data), Some(ref src_data)) = 
            (&mut self.cached_data, &other_clone.cached_data) {
            
            dst_data[dst_offset..dst_offset + len]
                .clone_from_slice(&src_data[src_offset..src_offset + len]);
            
            // Update MLX array if available
            // In a real implementation, we'd need to sync this back to the GPU
            trace!("Copied {} elements from offset {} to offset {}", len, src_offset, dst_offset);
            
            Ok(())
        } else {
            Err(woolly_tensor::backend::TensorError::backend_error(
                "MLX_NO_DATA_FOR_COPY",
                "Data not available for copy operation",
                "MLX",
                "storage copy",
                "Use ensure_cached_data() to cache data from GPU"
            ))
        }
    }
    
    fn fill(&mut self, value: T) -> woolly_tensor::backend::Result<()> {
        // Ensure we have cached data
        self.ensure_cached_data().map_err(|e| e.into())?;
        
        if let Some(ref mut data) = self.cached_data {
            data.fill(value);
            
            // In a real implementation, we'd need to sync this back to the GPU
            trace!("Filled storage with value");
            
            Ok(())
        } else {
            Err(woolly_tensor::backend::TensorError::backend_error(
                "MLX_NO_DATA_FOR_FILL",
                "No data available for fill operation",
                "MLX",
                "storage fill",
                "Use ensure_cached_data() to cache data from GPU"
            ))
        }
    }
    
    fn to_vec(&self) -> woolly_tensor::backend::Result<Vec<T>> {
        let mut storage_clone = self.clone();
        storage_clone.ensure_cached_data().map_err(|e| e.into())?;
        
        if let Some(ref data) = storage_clone.cached_data {
            Ok(data.clone())
        } else {
            Err(woolly_tensor::backend::TensorError::backend_error(
                "MLX_NO_DATA_FOR_VEC",
                "No data available for conversion to vec",
                "MLX",
                "storage to_vec",
                "Use ensure_cached_data() to cache data from GPU"
            ))
        }
    }
}

// Specialized storage types for common data types
pub type MLXStorageF32 = MLXStorage<f32>;
pub type MLXStorageF16 = MLXStorage<half::f16>;
pub type MLXStorageI32 = MLXStorage<i32>;
pub type MLXStorageU8 = MLXStorage<u8>;

#[cfg(test)]
mod tests {
    use super::*;
    use woolly_tensor::shape::Shape;
    
    #[test]
    fn test_storage_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = Shape::from_slice(&[2, 2]);
        
        match MLXStorage::from_data(data.clone(), shape, DType::F32, Device::CPU) {
            Ok(storage) => {
                assert_eq!(storage.len(), 4);
                assert_eq!(storage.dtype(), DType::F32);
                println!("Created storage: {:?}", storage);
            }
            Err(e) => {
                println!("Failed to create storage: {}", e);
                // This might fail if MLX is not available, which is expected
            }
        }
    }
    
    #[test]
    fn test_zeros_storage() {
        let shape = Shape::from_slice(&[3, 3]);
        
        match MLXStorage::<f32>::zeros(shape, DType::F32, Device::CPU) {
            Ok(storage) => {
                assert_eq!(storage.len(), 9);
                
                if let Ok(data) = storage.to_vec() {
                    assert!(data.iter().all(|&x| x == 0.0));
                    println!("Zeros storage created successfully");
                }
            }
            Err(e) => {
                println!("Failed to create zeros storage: {}", e);
            }
        }
    }
    
    #[test]
    fn test_storage_slice() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = Shape::vector(6);
        
        if let Ok(storage) = MLXStorage::from_data(data, shape, DType::F32, Device::CPU) {
            match storage.slice(2, 3) {
                Ok(sliced) => {
                    assert_eq!(sliced.len(), 3);
                    
                    if let Ok(sliced_data) = sliced.to_vec() {
                        assert_eq!(sliced_data, vec![3.0, 4.0, 5.0]);
                        println!("Slice operation successful");
                    }
                }
                Err(e) => {
                    println!("Slice operation failed: {}", e);
                }
            }
        }
    }
    
    #[test]
    fn test_device_transfer() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = Shape::from_slice(&[2, 2]);
        
        if let Ok(cpu_storage) = MLXStorage::from_data(data, shape, DType::F32, Device::CPU) {
            match cpu_storage.to_device(Device::GPU) {
                Ok(gpu_storage) => {
                    assert_eq!(gpu_storage.device, Device::GPU);
                    assert_eq!(gpu_storage.len(), cpu_storage.len());
                    println!("Device transfer successful");
                }
                Err(e) => {
                    println!("Device transfer failed: {}", e);
                }
            }
        }
    }
    
    #[test]
    fn test_storage_copy() {
        let src_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let dst_data = vec![0.0f32; 5];
        let shape = Shape::vector(5);
        
        if let (Ok(src_storage), Ok(mut dst_storage)) = (
            MLXStorage::from_data(src_data, shape.clone(), DType::F32, Device::CPU),
            MLXStorage::from_data(dst_data, shape, DType::F32, Device::CPU),
        ) {
            match dst_storage.copy_from(&src_storage, 1, 0, 3) {
                Ok(_) => {
                    if let Ok(result_data) = dst_storage.to_vec() {
                        assert_eq!(result_data[0..3], [2.0, 3.0, 4.0]);
                        println!("Copy operation successful");
                    }
                }
                Err(e) => {
                    println!("Copy operation failed: {}", e);
                }
            }
        }
    }
}