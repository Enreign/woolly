//! Tensor backend trait definitions and implementations

use std::fmt::Debug;
use std::ops::{Add, Mul, Sub, Div};
use thiserror::Error;

use crate::shape::Shape;
use crate::quantization::QuantizationScheme;

/// Errors that can occur during tensor operations
#[derive(Error, Debug)]
pub enum TensorError {
    /// Tensors have incompatible shapes for the operation
    #[error("Shape compatibility error [{code}]: {message}\nOperation: {operation}\nLeft shape: {left_shape}\nRight shape: {right_shape}\nSuggestion: {suggestion}")]
    IncompatibleShapes {
        /// Error code for programmatic handling
        code: &'static str,
        /// Human-readable error message
        message: String,
        /// The operation that failed
        operation: String,
        /// String representation of the left tensor shape
        left_shape: String,
        /// String representation of the right tensor shape
        right_shape: String,
        /// Suggested fix for the error
        suggestion: String,
    },
    
    /// The shape is invalid for the operation
    #[error("Invalid shape [{code}]: {message}\nShape: {shape}\nOperation: {operation}\nReason: {reason}\nSuggestion: {suggestion}")]
    InvalidShape {
        /// Error code for programmatic handling
        code: &'static str,
        /// Human-readable error message
        message: String,
        /// String representation of the invalid shape
        shape: String,
        /// The operation that failed
        operation: String,
        /// Reason why the shape is invalid
        reason: String,
        /// Suggested fix for the error
        suggestion: String,
    },
    
    /// Invalid tensor dimensions or size
    #[error("Dimension error [{code}]: {message}\nExpected: {expected}\nActual: {actual}\nOperation: {operation}\nSuggestion: {suggestion}")]
    InvalidDimensions {
        /// Error code for programmatic handling
        code: &'static str,
        /// Human-readable error message
        message: String,
        /// Expected dimensions description
        expected: String,
        /// Actual dimensions description
        actual: String,
        /// The operation that failed
        operation: String,
        /// Suggested fix for the error
        suggestion: String,
    },
    
    /// Generic backend error
    #[error("Backend error [{code}]: {message}\nBackend: {backend}\nOperation: {operation}\nSuggestion: {suggestion}")]
    BackendError {
        /// Error code for programmatic handling
        code: &'static str,
        /// Human-readable error message
        message: String,
        /// Name of the backend that failed
        backend: String,
        /// The operation that failed
        operation: String,
        /// Suggested fix for the error
        suggestion: String,
    },
    
    /// Index is out of bounds for the given dimension
    #[error("Index out of bounds [{code}]: {message}\nIndex: {index}, Dimension: {dim}, Size: {size}\nOperation: {operation}\nSuggestion: {suggestion}")]
    OutOfBounds { 
        /// Error code for programmatic handling
        code: &'static str,
        /// Human-readable error message
        message: String,
        /// The index that was out of bounds
        index: usize, 
        /// The dimension where the index was applied
        dim: usize, 
        /// The size of that dimension
        size: usize,
        /// The operation that failed
        operation: String,
        /// Suggested fix for the error
        suggestion: String,
    },
    
    /// Error during quantization operations
    #[error("Quantization error [{code}]: {message}\nScheme: {scheme}\nData type: {dtype}\nSuggestion: {suggestion}")]
    QuantizationError {
        /// Error code for programmatic handling
        code: &'static str,
        /// Human-readable error message
        message: String,
        /// The quantization scheme that failed
        scheme: String,
        /// The data type involved
        dtype: String,
        /// Suggested fix for the error
        suggestion: String,
    },
    
    /// Operation not supported by the backend
    #[error("Unsupported operation [{code}]: {message}\nOperation: {operation}\nBackend: {backend}\nData type: {dtype}\nSuggestion: {suggestion}")]
    UnsupportedOperation {
        /// Error code for programmatic handling
        code: &'static str,
        /// Human-readable error message
        message: String,
        /// The unsupported operation
        operation: String,
        /// The backend that doesn't support this operation
        backend: String,
        /// The data type involved
        dtype: String,
        /// Suggested fix for the error
        suggestion: String,
    },
    
    /// Memory allocation or management errors
    #[error("Memory error [{code}]: {message}\nRequested: {requested} bytes\nAvailable: {available}\nBackend: {backend}\nSuggestion: {suggestion}")]
    MemoryError {
        /// Error code for programmatic handling
        code: &'static str,
        /// Human-readable error message
        message: String,
        /// Number of bytes requested
        requested: u64,
        /// Available memory description
        available: String,
        /// The backend where memory allocation failed
        backend: String,
        /// Suggested fix for the error
        suggestion: String,
    },
    
    /// Device or compute errors
    #[error("Device error [{code}]: {message}\nDevice: {device}\nOperation: {operation}\nSuggestion: {suggestion}")]
    DeviceError {
        /// Error code for programmatic handling
        code: &'static str,
        /// Human-readable error message
        message: String,
        /// The device that encountered an error
        device: String,
        /// The operation that failed
        operation: String,
        /// Suggested fix for the error
        suggestion: String,
    },
    
    /// Data type conversion or compatibility errors
    #[error("Data type error [{code}]: {message}\nFrom: {from_dtype}\nTo: {to_dtype}\nOperation: {operation}\nSuggestion: {suggestion}")]
    DataTypeError {
        /// Error code for programmatic handling
        code: &'static str,
        /// Human-readable error message
        message: String,
        /// Source data type
        from_dtype: String,
        /// Target data type
        to_dtype: String,
        /// The operation that failed
        operation: String,
        /// Suggested fix for the error
        suggestion: String,
    },
}

/// Convenient result type for tensor operations
pub type Result<T> = std::result::Result<T, TensorError>;

impl TensorError {
    /// Create an incompatible shapes error
    pub fn incompatible_shapes<S1, S2, S3, S4, S5>(
        code: &'static str,
        message: S1,
        operation: S2,
        left_shape: S3,
        right_shape: S4,
        suggestion: S5,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
        S4: Into<String>,
        S5: Into<String>,
    {
        Self::IncompatibleShapes {
            code,
            message: message.into(),
            operation: operation.into(),
            left_shape: left_shape.into(),
            right_shape: right_shape.into(),
            suggestion: suggestion.into(),
        }
    }

    /// Create an invalid shape error
    pub fn invalid_shape<S1, S2, S3, S4, S5>(
        code: &'static str,
        message: S1,
        shape: S2,
        operation: S3,
        reason: S4,
        suggestion: S5,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
        S4: Into<String>,
        S5: Into<String>,
    {
        Self::InvalidShape {
            code,
            message: message.into(),
            shape: shape.into(),
            operation: operation.into(),
            reason: reason.into(),
            suggestion: suggestion.into(),
        }
    }

    /// Create a backend error
    pub fn backend_error<S1, S2, S3, S4>(
        code: &'static str,
        message: S1,
        backend: S2,
        operation: S3,
        suggestion: S4,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
        S4: Into<String>,
    {
        Self::BackendError {
            code,
            message: message.into(),
            backend: backend.into(),
            operation: operation.into(),
            suggestion: suggestion.into(),
        }
    }

    /// Create an out of bounds error
    pub fn out_of_bounds<S1, S2, S3>(
        code: &'static str,
        message: S1,
        index: usize,
        dim: usize,
        size: usize,
        operation: S2,
        suggestion: S3,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        Self::OutOfBounds {
            code,
            message: message.into(),
            index,
            dim,
            size,
            operation: operation.into(),
            suggestion: suggestion.into(),
        }
    }

    /// Create a memory error
    pub fn memory_error<S1, S2, S3>(
        code: &'static str,
        message: S1,
        requested: u64,
        available: String,
        backend: S2,
        suggestion: S3,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        Self::MemoryError {
            code,
            message: message.into(),
            requested,
            available,
            backend: backend.into(),
            suggestion: suggestion.into(),
        }
    }

    /// Create an unsupported operation error
    pub fn unsupported_operation<S1, S2, S3, S4, S5>(
        code: &'static str,
        message: S1,
        operation: S2,
        backend: S3,
        dtype: S4,
        suggestion: S5,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
        S4: Into<String>,
        S5: Into<String>,
    {
        Self::UnsupportedOperation {
            code,
            message: message.into(),
            operation: operation.into(),
            backend: backend.into(),
            dtype: dtype.into(),
            suggestion: suggestion.into(),
        }
    }

    /// Get the error code for programmatic handling
    pub fn code(&self) -> &'static str {
        match self {
            Self::IncompatibleShapes { code, .. } => code,
            Self::InvalidShape { code, .. } => code,
            Self::InvalidDimensions { code, .. } => code,
            Self::BackendError { code, .. } => code,
            Self::OutOfBounds { code, .. } => code,
            Self::QuantizationError { code, .. } => code,
            Self::UnsupportedOperation { code, .. } => code,
            Self::MemoryError { code, .. } => code,
            Self::DeviceError { code, .. } => code,
            Self::DataTypeError { code, .. } => code,
        }
    }

    /// Check if this is a shape-related error
    pub fn is_shape_error(&self) -> bool {
        matches!(self, Self::IncompatibleShapes { .. } | Self::InvalidShape { .. } | Self::InvalidDimensions { .. })
    }

    /// Check if this is a resource-related error
    pub fn is_resource_error(&self) -> bool {
        matches!(self, Self::MemoryError { .. } | Self::DeviceError { .. })
    }
}

/// Supported data types for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 64-bit floating point
    F64,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// Boolean
    Bool,
    /// Quantized type
    Quantized(QuantizationScheme),
}

impl DType {
    /// Returns the size of the data type in bytes
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F64 | DType::I64 => 8,
            DType::U8 | DType::Bool => 1,
            DType::Quantized(scheme) => scheme.bytes_per_element(),
        }
    }
    
    /// Returns whether this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16 | DType::F64)
    }
    
    /// Returns whether this is a quantized type
    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::Quantized(_))
    }
}

/// Device where tensor operations are performed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU device
    Cpu,
    /// CUDA GPU device with device index
    Cuda(usize),
    /// Metal GPU device
    Metal,
}

/// Trait for tensor storage implementations
pub trait TensorStorage: Debug + Send + Sync + Clone {
    /// The scalar type this storage holds
    type Elem;
    
    /// Returns the data type of the storage
    fn dtype(&self) -> DType;
    
    /// Returns the device where the storage resides
    fn device(&self) -> Device;
    
    /// Returns the total number of elements
    fn len(&self) -> usize;
    
    /// Returns whether the storage is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Allocates new storage with the given capacity
    fn allocate(capacity: usize, device: Device) -> Result<Self>
    where
        Self: Sized;
    
    /// Creates a view into a subset of the storage
    fn slice(&self, offset: usize, len: usize) -> Result<Self>
    where
        Self: Sized;
    
    /// Copies data from another storage
    fn copy_from(&mut self, other: &Self, src_offset: usize, dst_offset: usize, len: usize) -> Result<()>;
    
    /// Fills the storage with a scalar value
    fn fill(&mut self, value: Self::Elem) -> Result<()>
    where
        Self::Elem: Clone;
    
    /// Returns a CPU-accessible view of the data
    fn to_vec(&self) -> Result<Vec<Self::Elem>>
    where
        Self::Elem: Clone;
}

/// Trait for tensor backend implementations
pub trait TensorBackend: Debug + Send + Sync + Clone {
    /// The storage type used by this backend
    type Storage<T>: TensorStorage<Elem = T> 
    where 
        T: Clone + Send + Sync + std::fmt::Debug + 'static;
    
    /// Returns the name of the backend
    fn name(&self) -> &'static str;
    
    /// Returns the device associated with this backend
    fn device(&self) -> Device;
    
    /// Checks if the backend supports a specific data type
    fn supports_dtype(&self, dtype: DType) -> bool;
    
    /// Creates a tensor filled with zeros
    fn zeros<T>(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage<T>>
    where
        T: num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static;
    
    /// Creates a tensor filled with ones
    fn ones<T>(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage<T>>
    where
        T: num_traits::One + Clone + Send + Sync + std::fmt::Debug + 'static;
    
    /// Creates a tensor filled with a scalar value
    fn full<T>(&self, shape: &Shape, value: T, dtype: DType) -> Result<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static;
    
    /// Creates a tensor from a slice of data
    fn from_slice<T>(&self, data: &[T], shape: &Shape, dtype: DType) -> Result<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static;
    
    // Binary operations
    
    /// Element-wise addition
    fn add<T>(&self, lhs: &Self::Storage<T>, rhs: &Self::Storage<T>) -> Result<Self::Storage<T>>
    where
        T: Add<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static;
    
    /// Element-wise subtraction
    fn sub<T>(&self, lhs: &Self::Storage<T>, rhs: &Self::Storage<T>) -> Result<Self::Storage<T>>
    where
        T: Sub<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static;
    
    /// Element-wise multiplication
    fn mul<T>(&self, lhs: &Self::Storage<T>, rhs: &Self::Storage<T>) -> Result<Self::Storage<T>>
    where
        T: Mul<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static;
    
    /// Element-wise division
    fn div<T>(&self, lhs: &Self::Storage<T>, rhs: &Self::Storage<T>) -> Result<Self::Storage<T>>
    where
        T: Div<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static;
    
    // Matrix operations
    
    /// Matrix multiplication
    fn matmul<T>(&self, lhs: &Self::Storage<T>, rhs: &Self::Storage<T>, lhs_shape: &Shape, rhs_shape: &Shape) -> Result<Self::Storage<T>>
    where
        T: Add<Output = T> + Mul<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static;
    
    // Reduction operations
    
    /// Sum reduction along specified axes
    fn sum<T>(&self, input: &Self::Storage<T>, shape: &Shape, axes: &[usize], keep_dims: bool) -> Result<Self::Storage<T>>
    where
        T: Add<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static;
    
    /// Mean reduction along specified axes
    fn mean<T>(&self, input: &Self::Storage<T>, shape: &Shape, axes: &[usize], keep_dims: bool) -> Result<Self::Storage<T>>
    where
        T: Add<Output = T> + Div<Output = T> + num_traits::Zero + Clone + From<f32> + Send + Sync + std::fmt::Debug + 'static;
    
    // Shape operations
    
    /// Reshape a tensor
    fn reshape<T>(&self, input: &Self::Storage<T>, old_shape: &Shape, new_shape: &Shape) -> Result<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static;
    
    /// Transpose a tensor
    fn transpose<T>(&self, input: &Self::Storage<T>, shape: &Shape, axes: &[usize]) -> Result<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static;
    
    // Data movement
    
    /// Copy data to another device
    fn to_device<T>(&self, input: &Self::Storage<T>, device: Device) -> Result<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static;
    
    // Quantization operations
    
    /// Quantize a tensor
    fn quantize<T>(&self, input: &Self::Storage<T>, scheme: QuantizationScheme) -> Result<Self::Storage<u8>>
    where
        T: Clone + Into<f32> + Send + Sync + std::fmt::Debug + 'static;
    
    /// Dequantize a tensor  
    fn dequantize(&self, input: &Self::Storage<u8>, target_dtype: DType) -> Result<Self::Storage<f32>>;
}

/// CPU backend implementation
#[derive(Debug, Clone)]
pub struct CpuBackend {
    /// Whether to use SIMD operations
    pub use_simd: bool,
}

impl CpuBackend {
    /// Creates a new CPU backend
    pub fn new() -> Self {
        Self { use_simd: true }
    }
    
    /// Creates a CPU backend without SIMD
    pub fn new_no_simd() -> Self {
        Self { use_simd: false }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU storage implementation
#[derive(Debug, Clone)]
pub struct CpuStorage<T> {
    data: Vec<T>,
    dtype: DType,
}

impl<T> TensorStorage for CpuStorage<T>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    type Elem = T;

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> Device {
        Device::Cpu
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn allocate(capacity: usize, _device: Device) -> Result<Self> {
        Ok(Self {
            data: Vec::with_capacity(capacity),
            dtype: DType::F32, // Default, should be parameterized
        })
    }

    fn slice(&self, offset: usize, len: usize) -> Result<Self> {
        if offset + len > self.data.len() {
            return Err(TensorError::out_of_bounds(
                "TENSOR_SLICE_OUT_OF_BOUNDS",
                "Slice range exceeds tensor bounds",
                offset + len - 1,
                0,
                self.data.len(),
                "tensor slicing",
                "Ensure slice offset + length does not exceed tensor size"
            ));
        }
        Ok(Self {
            data: self.data[offset..offset + len].to_vec(),
            dtype: self.dtype,
        })
    }

    fn copy_from(&mut self, other: &Self, src_offset: usize, dst_offset: usize, len: usize) -> Result<()> {
        if src_offset + len > other.data.len() {
            return Err(TensorError::out_of_bounds(
                "TENSOR_COPY_SRC_OUT_OF_BOUNDS",
                "Source copy range exceeds tensor bounds",
                src_offset + len - 1,
                0,
                other.data.len(),
                "tensor copy (source)",
                "Ensure source offset + length does not exceed source tensor size"
            ));
        }
        if dst_offset + len > self.data.len() {
            return Err(TensorError::out_of_bounds(
                "TENSOR_COPY_DST_OUT_OF_BOUNDS",
                "Destination copy range exceeds tensor bounds",
                dst_offset + len - 1,
                0,
                self.data.len(),
                "tensor copy (destination)",
                "Ensure destination offset + length does not exceed destination tensor size"
            ));
        }
        self.data[dst_offset..dst_offset + len]
            .clone_from_slice(&other.data[src_offset..src_offset + len]);
        Ok(())
    }

    fn fill(&mut self, value: T) -> Result<()> {
        self.data.fill(value);
        Ok(())
    }

    fn to_vec(&self) -> Result<Vec<T>> {
        Ok(self.data.clone())
    }
}

// Stub implementation of TensorBackend for CpuBackend
// This is a minimal implementation to allow compilation
impl TensorBackend for CpuBackend {
    type Storage<T> = CpuStorage<T>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static;

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn device(&self) -> Device {
        Device::Cpu
    }

    fn supports_dtype(&self, _dtype: DType) -> bool {
        true // For now, support all types
    }

    fn zeros<T>(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage<T>>
    where
        T: num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let total_elements = shape.numel();
        Ok(CpuStorage {
            data: vec![T::zero(); total_elements],
            dtype,
        })
    }

    fn ones<T>(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage<T>>
    where
        T: num_traits::One + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let total_elements = shape.numel();
        Ok(CpuStorage {
            data: vec![T::one(); total_elements],
            dtype,
        })
    }

    fn full<T>(&self, shape: &Shape, value: T, dtype: DType) -> Result<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        let total_elements = shape.numel();
        Ok(CpuStorage {
            data: vec![value; total_elements],
            dtype,
        })
    }

    fn from_slice<T>(&self, data: &[T], _shape: &Shape, dtype: DType) -> Result<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        Ok(CpuStorage {
            data: data.to_vec(),
            dtype,
        })
    }

    fn add<T>(&self, lhs: &Self::Storage<T>, rhs: &Self::Storage<T>) -> Result<Self::Storage<T>>
    where
        T: Add<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        if lhs.len() != rhs.len() {
            return Err(TensorError::incompatible_shapes(
                "TENSOR_ADD_SHAPE_MISMATCH",
                "Cannot perform element-wise addition on tensors with different sizes",
                "element-wise addition",
                format!("[{}]", lhs.len()),
                format!("[{}]", rhs.len()),
                "Ensure both tensors have the same total number of elements, or use broadcasting"
            ));
        }
        let result: Vec<T> = lhs.data.iter().zip(rhs.data.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        Ok(CpuStorage {
            data: result,
            dtype: lhs.dtype,
        })
    }

    fn sub<T>(&self, lhs: &Self::Storage<T>, rhs: &Self::Storage<T>) -> Result<Self::Storage<T>>
    where
        T: Sub<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        if lhs.len() != rhs.len() {
            return Err(TensorError::incompatible_shapes(
                "TENSOR_SUB_SHAPE_MISMATCH",
                "Cannot perform element-wise subtraction on tensors with different sizes",
                "element-wise subtraction",
                format!("[{}]", lhs.len()),
                format!("[{}]", rhs.len()),
                "Ensure both tensors have the same total number of elements, or use broadcasting"
            ));
        }
        let result: Vec<T> = lhs.data.iter().zip(rhs.data.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();
        Ok(CpuStorage {
            data: result,
            dtype: lhs.dtype,
        })
    }

    fn mul<T>(&self, lhs: &Self::Storage<T>, rhs: &Self::Storage<T>) -> Result<Self::Storage<T>>
    where
        T: Mul<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        if lhs.len() != rhs.len() {
            return Err(TensorError::incompatible_shapes(
                "TENSOR_MUL_SHAPE_MISMATCH",
                "Cannot perform element-wise multiplication on tensors with different sizes",
                "element-wise multiplication",
                format!("[{}]", lhs.len()),
                format!("[{}]", rhs.len()),
                "Ensure both tensors have the same total number of elements, or use broadcasting"
            ));
        }
        let result: Vec<T> = lhs.data.iter().zip(rhs.data.iter())
            .map(|(a, b)| a.clone() * b.clone())
            .collect();
        Ok(CpuStorage {
            data: result,
            dtype: lhs.dtype,
        })
    }

    fn div<T>(&self, lhs: &Self::Storage<T>, rhs: &Self::Storage<T>) -> Result<Self::Storage<T>>
    where
        T: Div<Output = T> + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        if lhs.len() != rhs.len() {
            return Err(TensorError::incompatible_shapes(
                "TENSOR_DIV_SHAPE_MISMATCH",
                "Cannot perform element-wise division on tensors with different sizes",
                "element-wise division",
                format!("[{}]", lhs.len()),
                format!("[{}]", rhs.len()),
                "Ensure both tensors have the same total number of elements, or use broadcasting"
            ));
        }
        let result: Vec<T> = lhs.data.iter().zip(rhs.data.iter())
            .map(|(a, b)| a.clone() / b.clone())
            .collect();
        Ok(CpuStorage {
            data: result,
            dtype: lhs.dtype,
        })
    }

    fn matmul<T>(&self, _lhs: &Self::Storage<T>, _rhs: &Self::Storage<T>, _lhs_shape: &Shape, _rhs_shape: &Shape) -> Result<Self::Storage<T>>
    where
        T: Add<Output = T> + Mul<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        // TODO: Implement matrix multiplication
        Err(TensorError::unsupported_operation(
            "TENSOR_MATMUL_NOT_IMPLEMENTED",
            "Matrix multiplication is not yet implemented for CPU backend",
            "matrix multiplication",
            "CPU",
            "unknown",
            "Use a different backend or wait for CPU matmul implementation"
        ))
    }

    fn sum<T>(&self, _input: &Self::Storage<T>, _shape: &Shape, _axes: &[usize], _keep_dims: bool) -> Result<Self::Storage<T>>
    where
        T: Add<Output = T> + num_traits::Zero + Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        // TODO: Implement sum reduction
        Err(TensorError::unsupported_operation(
            "TENSOR_SUM_NOT_IMPLEMENTED",
            "Sum reduction is not yet implemented for CPU backend",
            "sum reduction",
            "CPU",
            "unknown",
            "Use a different backend or wait for CPU sum implementation"
        ))
    }

    fn mean<T>(&self, _input: &Self::Storage<T>, _shape: &Shape, _axes: &[usize], _keep_dims: bool) -> Result<Self::Storage<T>>
    where
        T: Add<Output = T> + Div<Output = T> + num_traits::Zero + Clone + From<f32> + Send + Sync + std::fmt::Debug + 'static,
    {
        // TODO: Implement mean reduction
        Err(TensorError::UnsupportedOperation {
            code: "UNSUPPORTED_MEAN",
            message: "Mean reduction not yet implemented".to_string(),
            operation: "mean".to_string(),
            backend: "cpu".to_string(),
            dtype: "generic".to_string(),
            suggestion: "Use alternative reduction operations or implement mean operation".to_string(),
        })
    }

    fn reshape<T>(&self, input: &Self::Storage<T>, _old_shape: &Shape, _new_shape: &Shape) -> Result<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        // For now, just clone the storage (reshape is mainly a metadata operation)
        Ok(input.clone())
    }

    fn transpose<T>(&self, _input: &Self::Storage<T>, _shape: &Shape, _axes: &[usize]) -> Result<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        // TODO: Implement transpose
        Err(TensorError::UnsupportedOperation {
            code: "UNSUPPORTED_TRANSPOSE",
            message: "Transpose not yet implemented".to_string(),
            operation: "transpose".to_string(),
            backend: "cpu".to_string(),
            dtype: "generic".to_string(),
            suggestion: "Use manual dimension reordering or implement transpose operation".to_string(),
        })
    }

    fn to_device<T>(&self, input: &Self::Storage<T>, device: Device) -> Result<Self::Storage<T>>
    where
        T: Clone + Send + Sync + std::fmt::Debug + 'static,
    {
        match device {
            Device::Cpu => Ok(input.clone()),
            _ => Err(TensorError::UnsupportedOperation {
                code: "UNSUPPORTED_DEVICE_TRANSFER",
                message: "Device transfer not yet implemented".to_string(),
                operation: "device_transfer".to_string(),
                backend: "generic".to_string(),
                dtype: "generic".to_string(),
                suggestion: "Use single device operations or implement device transfer".to_string(),
            }),
        }
    }

    fn quantize<T>(&self, _input: &Self::Storage<T>, _scheme: QuantizationScheme) -> Result<Self::Storage<u8>>
    where
        T: Clone + Into<f32> + Send + Sync + std::fmt::Debug + 'static,
    {
        // TODO: Implement quantization
        Err(TensorError::UnsupportedOperation {
            code: "UNSUPPORTED_QUANTIZATION",
            message: "Quantization not yet implemented".to_string(),
            operation: "quantize".to_string(),
            backend: "generic".to_string(),
            dtype: "generic".to_string(),
            suggestion: "Use non-quantized operations or implement quantization".to_string(),
        })
    }

    fn dequantize(&self, _input: &Self::Storage<u8>, _target_dtype: DType) -> Result<Self::Storage<f32>> {
        // TODO: Implement dequantization
        Err(TensorError::UnsupportedOperation {
            code: "UNSUPPORTED_DEQUANTIZATION",
            message: "Dequantization not yet implemented".to_string(),
            operation: "dequantize".to_string(),
            backend: "generic".to_string(),
            dtype: "generic".to_string(),
            suggestion: "Use non-quantized operations or implement dequantization".to_string(),
        })
    }
}

/// CUDA backend implementation placeholder
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaBackend {
    device_id: usize,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    pub fn new(device_id: usize) -> Result<Self> {
        // TODO: Initialize CUDA context
        Ok(Self { device_id })
    }
}

/// Metal backend implementation placeholder
#[cfg(feature = "metal")]
#[derive(Debug)]
pub struct MetalBackend;

#[cfg(feature = "metal")]
impl MetalBackend {
    pub fn new() -> Result<Self> {
        // TODO: Initialize Metal device
        Ok(Self)
    }
}