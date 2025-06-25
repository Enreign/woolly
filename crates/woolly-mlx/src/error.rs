//! Error types for MLX backend operations

use thiserror::Error;
use woolly_tensor::backend::TensorError;

/// MLX-specific error types
#[derive(Error, Debug)]
pub enum MLXError {
    /// MLX is not available on this platform
    #[error("MLX not available: {0}")]
    NotAvailable(String),
    
    /// MLX initialization failed
    #[error("MLX initialization failed: {0}")]
    InitializationFailed(String),
    
    /// MLX runtime error
    #[error("MLX runtime error: {0}")]
    RuntimeError(String),
    
    /// Memory allocation failed
    #[error("MLX memory allocation failed: {0}")]
    AllocationFailed(String),
    
    /// Invalid MLX device
    #[error("Invalid MLX device: {0}")]
    InvalidDevice(String),
    
    /// MLX array operation failed
    #[error("MLX array operation failed: {0}")]
    ArrayOperationFailed(String),
    
    /// Shape mismatch in MLX operation
    #[error("MLX shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    
    /// Data type not supported by MLX
    #[error("MLX does not support data type: {0}")]
    UnsupportedDataType(String),
    
    /// MLX operation not implemented
    #[error("MLX operation not implemented: {0}")]
    NotImplemented(String),
    
    /// MLX FFI error
    #[error("MLX FFI error: {0}")]
    FFIError(String),
    
    /// Quantization error specific to MLX
    #[error("MLX quantization error: {0}")]
    QuantizationError(String),
    
    /// GPU memory error
    #[error("GPU memory error: {0}")]
    GPUMemoryError(String),
    
    /// Device synchronization error
    #[error("Device synchronization error: {0}")]
    SynchronizationError(String),
}

impl From<MLXError> for TensorError {
    fn from(err: MLXError) -> Self {
        match err {
            MLXError::NotAvailable(msg) => TensorError::backend_error(
                "MLX_NOT_AVAILABLE",
                format!("MLX not available: {}", msg),
                "MLX",
                "backend initialization",
                "Ensure MLX is properly installed and available"
            ),
            MLXError::InitializationFailed(msg) => TensorError::backend_error(
                "MLX_INIT_FAILED",
                format!("MLX init failed: {}", msg),
                "MLX",
                "backend initialization",
                "Check MLX installation and system requirements"
            ),
            MLXError::RuntimeError(msg) => TensorError::backend_error(
                "MLX_RUNTIME_ERROR",
                format!("MLX runtime: {}", msg),
                "MLX",
                "runtime operation",
                "Check input parameters and operation validity"
            ),
            MLXError::AllocationFailed(msg) => TensorError::backend_error(
                "MLX_ALLOCATION_FAILED",
                format!("MLX allocation: {}", msg),
                "MLX",
                "memory allocation",
                "Reduce memory usage or increase available memory"
            ),
            MLXError::InvalidDevice(msg) => TensorError::backend_error(
                "MLX_INVALID_DEVICE",
                format!("MLX device: {}", msg),
                "MLX",
                "device selection",
                "Use a valid MLX device (CPU or GPU)"
            ),
            MLXError::ArrayOperationFailed(msg) => TensorError::backend_error(
                "MLX_ARRAY_OP_FAILED",
                format!("MLX array op: {}", msg),
                "MLX",
                "array operation",
                "Check array dimensions and operation parameters"
            ),
            MLXError::ShapeMismatch { expected, actual } => {
                TensorError::incompatible_shapes(
                    "MLX_SHAPE_MISMATCH",
                    "Shape mismatch in MLX operation",
                    "MLX operation",
                    format!("{:?}", expected),
                    format!("{:?}", actual),
                    "Ensure shapes are compatible for the operation"
                )
            }
            MLXError::UnsupportedDataType(msg) => TensorError::UnsupportedOperation {
                code: "MLX_UNSUPPORTED_DTYPE",
                message: format!("MLX dtype: {}", msg),
                operation: "data type operation".to_string(),
                backend: "MLX".to_string(),
                dtype: "unknown".to_string(),
                suggestion: "Use a supported data type for MLX operations".to_string(),
            },
            MLXError::NotImplemented(msg) => TensorError::UnsupportedOperation {
                code: "MLX_NOT_IMPLEMENTED",
                message: format!("MLX not impl: {}", msg),
                operation: "unimplemented operation".to_string(),
                backend: "MLX".to_string(),
                dtype: "unknown".to_string(),
                suggestion: "Use an alternative operation or backend".to_string(),
            },
            MLXError::FFIError(msg) => TensorError::backend_error(
                "MLX_FFI_ERROR",
                format!("MLX FFI: {}", msg),
                "MLX",
                "FFI operation",
                "Check MLX FFI bindings and system compatibility"
            ),
            MLXError::QuantizationError(msg) => TensorError::QuantizationError {
                code: "MLX_QUANTIZATION_ERROR",
                message: msg,
                scheme: "unknown".to_string(),
                dtype: "unknown".to_string(),
                suggestion: "Check quantization parameters and input data".to_string(),
            },
            MLXError::GPUMemoryError(msg) => TensorError::backend_error(
                "MLX_GPU_MEMORY_ERROR",
                format!("MLX GPU memory: {}", msg),
                "MLX",
                "GPU memory operation",
                "Reduce memory usage or use CPU backend"
            ),
            MLXError::SynchronizationError(msg) => TensorError::backend_error(
                "MLX_SYNC_ERROR",
                format!("MLX sync: {}", msg),
                "MLX",
                "device synchronization",
                "Retry operation or check device state"
            ),
        }
    }
}

/// Result type for MLX operations
pub type Result<T> = std::result::Result<T, MLXError>;

/// Helper function to create MLX errors from strings
pub fn mlx_error(msg: &str) -> MLXError {
    MLXError::RuntimeError(msg.to_string())
}

/// Helper function to create not available errors
pub fn not_available(msg: &str) -> MLXError {
    MLXError::NotAvailable(msg.to_string())
}

/// Helper function to create allocation errors
pub fn allocation_error(msg: &str) -> MLXError {
    MLXError::AllocationFailed(msg.to_string())
}

/// Helper function to create shape mismatch errors
pub fn shape_mismatch(expected: &[usize], actual: &[usize]) -> MLXError {
    MLXError::ShapeMismatch {
        expected: expected.to_vec(),
        actual: actual.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_conversion() {
        let mlx_err = MLXError::RuntimeError("test error".to_string());
        let tensor_err: TensorError = mlx_err.into();
        
        match tensor_err {
            TensorError::BackendError(msg) => {
                assert!(msg.contains("test error"));
            }
            _ => panic!("Expected BackendError"),
        }
    }
    
    #[test]
    fn test_shape_mismatch_error() {
        let err = shape_mismatch(&[2, 3], &[3, 2]);
        match err {
            MLXError::ShapeMismatch { expected, actual } => {
                assert_eq!(expected, vec![2, 3]);
                assert_eq!(actual, vec![3, 2]);
            }
            _ => panic!("Expected ShapeMismatch"),
        }
    }
    
    #[test]
    fn test_helper_functions() {
        let err1 = mlx_error("runtime error");
        assert!(matches!(err1, MLXError::RuntimeError(_)));
        
        let err2 = not_available("not available");
        assert!(matches!(err2, MLXError::NotAvailable(_)));
        
        let err3 = allocation_error("allocation failed");
        assert!(matches!(err3, MLXError::AllocationFailed(_)));
    }
}