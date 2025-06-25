//! Core tensor type and operations

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Sub, Mul};

use crate::backend::{TensorBackend, TensorStorage, TensorError, Result, DType, Device};
use crate::shape::{Shape, Strides};
use crate::quantization::{QuantizedStorage, QuantizationScheme, optimized};

/// Core tensor type
#[derive(Debug)]
pub struct Tensor<B: TensorBackend, T>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    /// The underlying storage
    storage: B::Storage<T>,
    /// The shape of the tensor
    shape: Shape,
    /// The strides of the tensor
    strides: Strides,
    /// Offset into the storage
    offset: usize,
    /// The backend used for operations
    backend: B,
    /// Phantom data for the element type
    _phantom: PhantomData<T>,
}

impl<B: TensorBackend, T> Tensor<B, T>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    /// Creates a new tensor from storage, shape, and strides
    pub fn new(
        storage: B::Storage<T>,
        shape: Shape,
        strides: Strides,
        offset: usize,
        backend: B,
    ) -> Result<Self> {
        // Validate that shape and strides are compatible
        if shape.ndim() != strides.ndim() {
            return Err(TensorError::invalid_shape(
                "TENSOR_SHAPE_STRIDE_MISMATCH",
                format!("Shape dimensions {} don't match stride dimensions {}", 
                    shape.ndim(), strides.ndim()),
                format!("{:?}", shape),
                "tensor creation",
                "Incompatible dimensions",
                "Ensure shape and strides have the same number of dimensions"
            ));
        }
        
        shape.validate()?;
        
        Ok(Self {
            storage,
            shape,
            strides,
            offset,
            backend,
            _phantom: PhantomData,
        })
    }
    
    /// Returns the shape of the tensor
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    
    /// Returns the strides of the tensor
    pub fn strides(&self) -> &Strides {
        &self.strides
    }
    
    /// Returns the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }
    
    /// Returns the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }
    
    /// Returns the data type of the tensor
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }
    
    /// Returns the device where the tensor is stored
    pub fn device(&self) -> Device {
        self.storage.device()
    }
    
    /// Returns a reference to the backend
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// Returns whether the tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }
    
    /// Creates a contiguous copy of the tensor if it's not already contiguous
    pub fn contiguous(&self) -> Result<Self>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        if self.is_contiguous() {
            // Already contiguous, return a clone
            Ok(Self {
                storage: self.storage.clone(),
                shape: self.shape.clone(),
                strides: self.strides.clone(),
                offset: self.offset,
                backend: self.backend.clone(),
                _phantom: PhantomData,
            })
        } else {
            // Need to create a contiguous copy
            todo!("Implement contiguous copy for non-contiguous tensors")
        }
    }
    
    /// Reshapes the tensor to a new shape
    pub fn reshape(&self, new_shape: &Shape) -> Result<Self>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        // Validate that the number of elements matches
        if self.numel() != new_shape.numel() {
            return Err(TensorError::invalid_shape(
                "TENSOR_RESHAPE_SIZE_MISMATCH",
                format!("Cannot reshape tensor with {} elements to shape with {} elements",
                    self.numel(), new_shape.numel()),
                format!("{:?}", new_shape),
                "tensor reshape",
                "Element count mismatch",
                "Ensure new shape has same total number of elements"
            ));
        }
        
        let new_storage = self.backend.reshape(&self.storage, &self.shape, new_shape)?;
        let new_strides = new_shape.default_strides();
        
        Self::new(new_storage, new_shape.clone(), new_strides, 0, self.backend.clone())
    }
    
    /// Transposes the tensor according to the given axes
    pub fn transpose(&self, axes: &[usize]) -> Result<Self>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        if axes.len() != self.ndim() {
            return Err(TensorError::invalid_shape(
                "TENSOR_TRANSPOSE_AXES_MISMATCH",
                format!("Transpose axes length {} doesn't match tensor dimensions {}",
                    axes.len(), self.ndim()),
                format!("{:?}", self.shape),
                "tensor transpose",
                "Axes length mismatch",
                "Provide exactly one axis index for each tensor dimension"
            ));
        }
        
        let new_storage = self.backend.transpose(&self.storage, &self.shape, axes)?;
        
        // Compute new shape and strides
        let mut new_shape_dims = vec![0; self.ndim()];
        for (i, &axis) in axes.iter().enumerate() {
            new_shape_dims[i] = self.shape[axis];
        }
        let new_shape = Shape::new(new_shape_dims);
        let new_strides = self.strides.transpose(axes)?;
        
        Self::new(new_storage, new_shape, new_strides, self.offset, self.backend.clone())
    }
    
    /// Creates a view of a slice of the tensor
    pub fn slice(&self, ranges: &[(usize, usize, isize)]) -> Result<Self> {
        if ranges.len() != self.ndim() {
            return Err(TensorError::invalid_shape(
                "TENSOR_SLICE_RANGES_MISMATCH",
                format!("Slice ranges length {} doesn't match tensor dimensions {}",
                    ranges.len(), self.ndim()),
                format!("{:?}", self.shape),
                "tensor slicing",
                "Ranges length mismatch",
                "Provide exactly one range for each tensor dimension"
            ));
        }
        
        // Compute new shape
        let mut new_shape_dims = Vec::with_capacity(self.ndim());
        for (i, &(start, end, step)) in ranges.iter().enumerate() {
            if start >= end || end > self.shape[i] {
                return Err(TensorError::out_of_bounds(
                    "TENSOR_SLICE_OUT_OF_BOUNDS",
                    format!("Slice end {} exceeds dimension {} size {}", end, i, self.shape[i]),
                    end,
                    i,
                    self.shape[i],
                    "tensor slicing",
                    "Ensure slice end does not exceed dimension size"
                ));
            }
            
            let size = if step > 0 {
                (end - start + step as usize - 1) / step as usize
            } else {
                return Err(TensorError::invalid_shape(
                    "TENSOR_SLICE_NEGATIVE_STEP",
                    "Negative slice step not supported",
                    format!("{:?}", self.shape),
                    "tensor slicing",
                    "Negative step values",
                    "Use positive step values for slicing"
                ));
            };
            
            new_shape_dims.push(size);
        }
        
        let new_shape = Shape::new(new_shape_dims);
        let (new_strides, stride_offset) = self.strides.slice(ranges)?;
        let new_offset = self.offset + stride_offset as usize;
        
        // Create a view with the same storage
        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
            backend: self.backend.clone(),
            _phantom: PhantomData,
        })
    }
    
    /// Broadcasts the tensor to a new shape
    pub fn broadcast_to(&self, target_shape: &Shape) -> Result<Self> {
        if !self.shape.is_broadcast_compatible(target_shape) {
            return Err(TensorError::incompatible_shapes(
                "TENSOR_BROADCAST_INCOMPATIBLE",
                "Cannot broadcast tensors with incompatible shapes",
                "tensor broadcasting",
                format!("{:?}", self.shape),
                format!("{:?}", target_shape),
                "Ensure shapes are compatible for broadcasting"
            ));
        }
        
        // Compute broadcast strides
        let mut new_strides = vec![0isize; target_shape.ndim()];
        let offset = target_shape.ndim() - self.ndim();
        
        for i in 0..self.ndim() {
            if self.shape[i] == target_shape[offset + i] {
                new_strides[offset + i] = self.strides[i];
            } else if self.shape[i] == 1 {
                new_strides[offset + i] = 0; // Broadcasting dimension
            }
        }
        
        Ok(Self {
            storage: self.storage.clone(),
            shape: target_shape.clone(),
            strides: Strides::new(new_strides),
            offset: self.offset,
            backend: self.backend.clone(),
            _phantom: PhantomData,
        })
    }
    
    /// Moves the tensor to a different device
    pub fn to_device(&self, device: Device) -> Result<Self>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        if self.device() == device {
            // Already on the target device
            return Ok(self.clone());
        }
        
        let new_storage = self.backend.to_device(&self.storage, device)?;
        
        Self::new(
            new_storage,
            self.shape.clone(),
            self.strides.clone(),
            self.offset,
            self.backend.clone(),
        )
    }
    
    /// Converts the tensor to a vector (must be contiguous)
    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        if !self.is_contiguous() {
            return Err(TensorError::invalid_shape(
                "TENSOR_NOT_CONTIGUOUS",
                "Tensor must be contiguous to convert to vector",
                format!("{:?}", self.shape),
                "tensor to_vec conversion",
                "Non-contiguous tensor",
                "Use contiguous() to create a contiguous copy first"
            ));
        }
        
        self.storage.to_vec()
    }
    
    /// Dequantizes a quantized tensor storage to f32
    /// This is a helper function for working with quantized tensors
    pub fn dequantize_storage(storage: &QuantizedStorage) -> Result<Vec<f32>> {
        optimized::dequantize_blocks(storage)
            .map_err(|e| TensorError::QuantizationError {
                code: "DEQUANTIZATION_ERROR",
                message: e.to_string(),
                scheme: "unknown".to_string(),
                dtype: "f32".to_string(),
                suggestion: "Check quantized storage format and values".to_string(),
            })
    }
    
    /// Creates a tensor from quantized storage by dequantizing it
    /// The resulting tensor will have f32 values
    pub fn from_quantized(
        storage: QuantizedStorage,
        shape: Shape,
        backend: B,
    ) -> Result<Tensor<B, f32>>
    where
        B: TensorBackend,
    {
        let f32_values = Self::dequantize_storage(&storage)?;
        
        // Create f32 storage from the dequantized values
        let f32_storage = backend.from_slice(&f32_values, &shape, DType::F32)?;
        let strides = shape.default_strides();
        
        Tensor::new(f32_storage, shape, strides, 0, backend)
    }
    
    /// Gets information about a quantized storage format
    pub fn quantization_info(storage: &QuantizedStorage) -> (QuantizationScheme, usize, f32) {
        match storage {
            QuantizedStorage::Q4_0(blocks) => (
                QuantizationScheme::Q4_0,
                blocks.len() * 32,
                8.0, // compression ratio
            ),
            QuantizedStorage::Q4_1(blocks) => (
                QuantizationScheme::Q4_1,
                blocks.len() * 32,
                8.0,
            ),
            QuantizedStorage::Q5_0(blocks) => (
                QuantizationScheme::Q5_0,
                blocks.len() * 32,
                6.4, // 32/5 compression ratio
            ),
            QuantizedStorage::Q5_1(blocks) => (
                QuantizationScheme::Q5_1,
                blocks.len() * 32,
                6.4,
            ),
            QuantizedStorage::Q8_0(blocks) => (
                QuantizationScheme::Q8_0,
                blocks.len() * 32,
                4.0, // 32/8 compression ratio
            ),
            QuantizedStorage::Int8(values) => (
                QuantizationScheme::Int8,
                values.len(),
                4.0,
            ),
            QuantizedStorage::Int4(values) => (
                QuantizationScheme::Int4,
                values.len() * 2, // 2 values per byte
                8.0,
            ),
        }
    }
}

// Factory methods
impl<B: TensorBackend, T> Tensor<B, T>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    /// Creates a tensor filled with zeros
    pub fn zeros(shape: Shape, dtype: DType, backend: B) -> Result<Self>
    where
        T: num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let storage = backend.zeros(&shape, dtype)?;
        let strides = shape.default_strides();
        Self::new(storage, shape, strides, 0, backend)
    }
    
    /// Creates a tensor filled with ones
    pub fn ones(shape: Shape, dtype: DType, backend: B) -> Result<Self>
    where
        T: num_traits::One + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let storage = backend.ones(&shape, dtype)?;
        let strides = shape.default_strides();
        Self::new(storage, shape, strides, 0, backend)
    }
    
    /// Creates a tensor filled with a scalar value
    pub fn full(shape: Shape, value: T, dtype: DType, backend: B) -> Result<Self>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let storage = backend.full(&shape, value, dtype)?;
        let strides = shape.default_strides();
        Self::new(storage, shape, strides, 0, backend)
    }
    
    /// Creates a tensor from a slice of data
    pub fn from_slice(data: &[T], shape: Shape, backend: B) -> Result<Self>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        if data.len() != shape.numel() {
            return Err(TensorError::invalid_shape(
                "TENSOR_DATA_SIZE_MISMATCH",
                format!("Data length {} doesn't match shape elements {}", 
                    data.len(), shape.numel()),
                format!("{:?}", shape),
                "tensor from_slice creation",
                "Data size mismatch",
                "Ensure data length matches shape total elements"
            ));
        }
        
        let dtype = DType::F32; // TODO: Infer from T
        let storage = backend.from_slice(data, &shape, dtype)?;
        let strides = shape.default_strides();
        Self::new(storage, shape, strides, 0, backend)
    }
}

// Arithmetic operations
impl<B: TensorBackend, T> Add for &Tensor<B, T>
where
    T: Add<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    type Output = Result<Tensor<B, T>>;
    
    fn add(self, rhs: Self) -> Self::Output {
        // Handle broadcasting
        let broadcast_shape = self.shape.broadcast_shape(&rhs.shape)?;
        
        let lhs = if &broadcast_shape != self.shape() {
            self.broadcast_to(&broadcast_shape)?
        } else {
            self.clone()
        };
        
        let rhs = if &broadcast_shape != rhs.shape() {
            rhs.broadcast_to(&broadcast_shape)?
        } else {
            rhs.clone()
        };
        
        let storage = self.backend.add(&lhs.storage, &rhs.storage)?;
        Tensor::new(storage, broadcast_shape, lhs.strides, 0, self.backend.clone())
    }
}

impl<B: TensorBackend, T> Sub for &Tensor<B, T>
where
    T: Sub<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    type Output = Result<Tensor<B, T>>;
    
    fn sub(self, rhs: Self) -> Self::Output {
        // Similar to add, with broadcasting
        let broadcast_shape = self.shape.broadcast_shape(&rhs.shape)?;
        
        let lhs = if &broadcast_shape != self.shape() {
            self.broadcast_to(&broadcast_shape)?
        } else {
            self.clone()
        };
        
        let rhs = if &broadcast_shape != rhs.shape() {
            rhs.broadcast_to(&broadcast_shape)?
        } else {
            rhs.clone()
        };
        
        let storage = self.backend.sub(&lhs.storage, &rhs.storage)?;
        Tensor::new(storage, broadcast_shape, lhs.strides, 0, self.backend.clone())
    }
}

impl<B: TensorBackend, T> Mul for &Tensor<B, T>
where
    T: Mul<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    type Output = Result<Tensor<B, T>>;
    
    fn mul(self, rhs: Self) -> Self::Output {
        let broadcast_shape = self.shape.broadcast_shape(&rhs.shape)?;
        
        let lhs = if &broadcast_shape != self.shape() {
            self.broadcast_to(&broadcast_shape)?
        } else {
            self.clone()
        };
        
        let rhs = if &broadcast_shape != rhs.shape() {
            rhs.broadcast_to(&broadcast_shape)?
        } else {
            rhs.clone()
        };
        
        let storage = self.backend.mul(&lhs.storage, &rhs.storage)?;
        Tensor::new(storage, broadcast_shape, lhs.strides, 0, self.backend.clone())
    }
}

impl<B: TensorBackend, T> Clone for Tensor<B, T>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            backend: self.backend.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<B: TensorBackend, T> fmt::Display for Tensor<B, T>
where
    T: fmt::Display + Clone + Send + Sync + std::fmt::Debug + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={}, dtype={:?}, device={:?})", 
            self.shape, self.dtype(), self.device())
    }
}

/// Builder for creating tensors with specific properties
pub struct TensorBuilder<B: TensorBackend> {
    backend: B,
    device: Option<Device>,
    dtype: Option<DType>,
}

impl<B: TensorBackend> TensorBuilder<B> {
    /// Creates a new tensor builder
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            device: None,
            dtype: None,
        }
    }
    
    /// Sets the device for tensor creation
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }
    
    /// Sets the data type for tensor creation
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    
    /// Creates a tensor filled with zeros
    pub fn zeros<T>(&self, shape: Shape) -> Result<Tensor<B, T>>
    where
        T: num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let dtype = self.dtype.unwrap_or(DType::F32);
        Tensor::zeros(shape, dtype, self.backend.clone())
    }
    
    /// Creates a tensor filled with ones
    pub fn ones<T>(&self, shape: Shape) -> Result<Tensor<B, T>>
    where
        T: num_traits::One + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let dtype = self.dtype.unwrap_or(DType::F32);
        Tensor::ones(shape, dtype, self.backend.clone())
    }
}