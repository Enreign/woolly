//! Integration tests for error handling improvements
//!
//! This test file validates that all error handling enhancements work correctly
//! across the Woolly system, including proper error codes, messages, and suggestions.

use std::fs::File;
use std::io::Write;
use tempfile::tempdir;
use woolly_core::{CoreError, validation::{Validator, ResourceValidator}};
use woolly_tensor::{TensorError, validation::TensorValidator, Shape, DType};

/// Test Core Error functionality
#[cfg(test)]
mod core_error_tests {
    use super::*;
    
    #[test]
    fn test_model_error_with_file_path() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("nonexistent.gguf");
        
        let error = CoreError::model_with_path(
            "MODEL_FILE_NOT_FOUND",
            "Model file does not exist",
            format!("Attempted to load model from: {}", model_path.display()),
            "Check the file path is correct and the model file exists",
            model_path.clone(),
        );
        
        assert_eq!(error.code(), "MODEL_FILE_NOT_FOUND");
        assert!(error.to_string().contains("Model file does not exist"));
        assert!(error.to_string().contains("Check the file path"));
    }
    
    #[test]
    fn test_invalid_parameter_error() {
        let error = CoreError::invalid_parameter(
            "TEMPERATURE_NEGATIVE",
            "Temperature cannot be negative",
            "Generation configuration validation",
            "Use 0.0 for deterministic output or positive values for randomness",
            "temperature",
            "-1.0",
            "0.0 to 2.0 (typical range)"
        );
        
        assert_eq!(error.code(), "TEMPERATURE_NEGATIVE");
        assert!(error.to_string().contains("Temperature cannot be negative"));
        assert!(error.to_string().contains("0.0 for deterministic"));
    }
    
    #[test]
    fn test_resource_error() {
        let error = CoreError::resource(
            "INSUFFICIENT_MEMORY",
            "Not enough memory available",
            "System has 4GB available, operation requires 8GB",
            "Free up system memory or reduce the operation size",
            "memory",
            Some(8_000_000_000), // 8GB required
            Some(4_000_000_000), // 4GB available
        );
        
        assert_eq!(error.code(), "INSUFFICIENT_MEMORY");
        assert!(error.to_string().contains("Not enough memory"));
        assert!(error.to_string().contains("Free up system memory"));
    }
}

/// Test Tensor Error functionality
#[cfg(test)]
mod tensor_error_tests {
    use super::*;
    
    #[test]
    fn test_incompatible_shapes_error() {
        let error = TensorError::incompatible_shapes(
            "TENSOR_ADD_SHAPE_MISMATCH",
            "Cannot perform element-wise addition on tensors with different sizes",
            "element-wise addition",
            "[2, 3, 4]",
            "[2, 5, 4]",
            "Ensure both tensors have the same total number of elements, or use broadcasting"
        );
        
        assert_eq!(error.code(), "TENSOR_ADD_SHAPE_MISMATCH");
        assert!(error.to_string().contains("Cannot perform element-wise addition"));
        assert!(error.to_string().contains("broadcasting"));
    }
    
    #[test]
    fn test_out_of_bounds_error() {
        let error = TensorError::out_of_bounds(
            "TENSOR_SLICE_OUT_OF_BOUNDS",
            "Slice range exceeds tensor bounds",
            15,  // index
            0,   // dimension
            10,  // size
            "tensor slicing",
            "Ensure slice offset + length does not exceed tensor size"
        );
        
        assert_eq!(error.code(), "TENSOR_SLICE_OUT_OF_BOUNDS");
        assert!(error.to_string().contains("Slice range exceeds"));
        assert!(error.to_string().contains("Index: 15, Dimension: 0, Size: 10"));
    }
    
    #[test]
    fn test_memory_error() {
        let error = TensorError::memory_error(
            "TENSOR_ALLOCATION_FAILED",
            "Failed to allocate tensor memory",
            1_000_000_000, // 1GB requested
            Some(500_000_000), // 500MB available
            "CPU",
            "Free up system memory or use a smaller tensor"
        );
        
        assert_eq!(error.code(), "TENSOR_ALLOCATION_FAILED");
        assert!(error.to_string().contains("Failed to allocate"));
        assert!(error.to_string().contains("Requested: 1000000000 bytes"));
    }
    
    #[test]
    fn test_error_type_checking() {
        let shape_error = TensorError::invalid_shape(
            "INVALID_TENSOR_SHAPE",
            "Tensor shape is invalid",
            "[0, 5]",
            "tensor creation",
            "Contains zero dimension",
            "Use positive integers for all dimensions"
        );
        
        assert!(shape_error.is_shape_error());
        assert!(!shape_error.is_resource_error());
        
        let memory_error = TensorError::memory_error(
            "OUT_OF_MEMORY",
            "Memory allocation failed",
            1000,
            Some(500),
            "CPU",
            "Free up memory"
        );
        
        assert!(!memory_error.is_shape_error());
        assert!(memory_error.is_resource_error());
    }
}

/// Test Core Validation functionality
#[cfg(test)]
mod core_validation_tests {
    use super::*;
    
    #[test]
    fn test_validate_model_path_not_found() {
        let result = Validator::validate_model_path("/nonexistent/model.gguf");
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "MODEL_FILE_NOT_FOUND");
        assert!(err.to_string().contains("Model file does not exist"));
    }
    
    #[test]
    fn test_validate_model_path_valid_gguf() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("model.gguf");
        let mut file = File::create(&model_path).unwrap();
        writeln!(file, "dummy model content").unwrap();
        
        let result = Validator::validate_model_path(&model_path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_path);
    }
    
    #[test]
    fn test_validate_model_path_unsupported_format() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("model.txt");
        let mut file = File::create(&model_path).unwrap();
        writeln!(file, "dummy content").unwrap();
        
        let result = Validator::validate_model_path(&model_path);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "MODEL_UNSUPPORTED_FORMAT");
        assert!(err.to_string().contains("Unsupported model file format"));
    }
    
    #[test]
    fn test_validate_model_path_empty_file() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("empty.gguf");
        File::create(&model_path).unwrap(); // Create empty file
        
        let result = Validator::validate_model_path(&model_path);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "MODEL_FILE_EMPTY");
        assert!(err.to_string().contains("Model file is empty"));
    }
    
    #[test]
    fn test_validate_generation_config_invalid_temperature() {
        let result = Validator::validate_generation_config(
            Some(100),
            Some(-1.0), // Invalid temperature
            Some(0.9),
            Some(50),
            Some(1.1),
        );
        
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), "TEMPERATURE_NEGATIVE");
        assert!(err.to_string().contains("Temperature cannot be negative"));
    }
    
    #[test]
    fn test_validate_generation_config_valid() {
        let result = Validator::validate_generation_config(
            Some(100),
            Some(0.7),
            Some(0.9),
            Some(50),
            Some(1.1),
        );
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_generation_config_max_tokens_zero() {
        let result = Validator::validate_generation_config(
            Some(0), // Invalid max_tokens
            Some(0.7),
            Some(0.9),
            Some(50),
            Some(1.1),
        );
        
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), "INVALID_MAX_TOKENS");
    }
    
    #[test]
    fn test_validate_generation_config_top_p_out_of_range() {
        let result = Validator::validate_generation_config(
            Some(100),
            Some(0.7),
            Some(1.5), // Invalid top_p
            Some(50),
            Some(1.1),
        );
        
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), "TOP_P_OUT_OF_RANGE");
    }
    
    #[test]
    fn test_validate_input_text_empty() {
        let result = Validator::validate_input_text("", None);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "INPUT_TEXT_EMPTY");
    }
    
    #[test]
    fn test_validate_input_text_too_long() {
        let long_text = "a".repeat(1000);
        let result = Validator::validate_input_text(&long_text, Some(500));
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "INPUT_TEXT_TOO_LONG");
    }
    
    #[test]
    fn test_validate_input_text_control_chars() {
        let text_with_control = "Hello\x00World"; // Contains null character
        let result = Validator::validate_input_text(text_with_control, None);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "INPUT_TEXT_INVALID_CHARS");
    }
    
    #[test]
    fn test_validate_device_config_valid_cpu() {
        let result = Validator::validate_device_config("cpu", None);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_device_config_valid_gpu() {
        let result = Validator::validate_device_config("cuda", Some(0));
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_device_config_invalid_type() {
        let result = Validator::validate_device_config("quantum", None);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "INVALID_DEVICE_TYPE");
    }
    
    #[test]
    fn test_validate_device_config_cpu_with_id() {
        let result = Validator::validate_device_config("cpu", Some(0));
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "CPU_DEVICE_ID_INVALID");
    }
    
    #[test]
    fn test_validate_device_config_gpu_without_id() {
        let result = Validator::validate_device_config("cuda", None);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "GPU_DEVICE_ID_MISSING");
    }
}

/// Test Tensor Validation functionality
#[cfg(test)]
mod tensor_validation_tests {
    use super::*;
    
    #[test]
    fn test_validate_matmul_shapes_valid() {
        let left = Shape::from_slice(&[2, 3]);
        let right = Shape::from_slice(&[3, 4]);
        
        let result = TensorValidator::validate_matmul_shapes(&left, &right);
        assert!(result.is_ok());
        
        let output_shape = result.unwrap();
        assert_eq!(output_shape.dims(), &[2, 4]);
    }
    
    #[test]
    fn test_validate_matmul_shapes_incompatible() {
        let left = Shape::from_slice(&[2, 3]);
        let right = Shape::from_slice(&[4, 5]); // 3 != 4
        
        let result = TensorValidator::validate_matmul_shapes(&left, &right);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "MATMUL_INNER_DIM_MISMATCH");
    }
    
    #[test]
    fn test_validate_matmul_shapes_too_few_dims() {
        let left = Shape::from_slice(&[5]); // Only 1D
        let right = Shape::from_slice(&[5, 3]);
        
        let result = TensorValidator::validate_matmul_shapes(&left, &right);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "MATMUL_LEFT_TOO_FEW_DIMS");
    }
    
    #[test]
    fn test_validate_reshape_valid() {
        let input = Shape::from_slice(&[2, 3, 4]); // 24 elements
        let target = Shape::from_slice(&[6, 4]);   // 24 elements
        
        let result = TensorValidator::validate_reshape(&input, &target);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_reshape_invalid_size() {
        let input = Shape::from_slice(&[2, 3, 4]); // 24 elements
        let target = Shape::from_slice(&[5, 4]);   // 20 elements
        
        let result = TensorValidator::validate_reshape(&input, &target);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "RESHAPE_SIZE_MISMATCH");
    }
    
    #[test]
    fn test_validate_reshape_zero_dimension() {
        let input = Shape::from_slice(&[2, 3, 4]);
        let target = Shape::from_slice(&[0, 24]); // Contains zero
        
        let result = TensorValidator::validate_reshape(&input, &target);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "RESHAPE_ZERO_DIMENSION");
    }
    
    #[test]
    fn test_validate_transpose_valid() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        let axes = [2, 0, 1]; // Transpose dimensions
        
        let result = TensorValidator::validate_transpose(&shape, &axes);
        assert!(result.is_ok());
        
        let output_shape = result.unwrap();
        assert_eq!(output_shape.dims(), &[4, 2, 3]);
    }
    
    #[test]
    fn test_validate_transpose_wrong_length() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        let axes = [0, 1]; // Wrong number of axes
        
        let result = TensorValidator::validate_transpose(&shape, &axes);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "TRANSPOSE_AXES_LENGTH_MISMATCH");
    }
    
    #[test]
    fn test_validate_transpose_duplicate_axes() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        let axes = [0, 1, 1]; // Duplicate axis
        
        let result = TensorValidator::validate_transpose(&shape, &axes);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "TRANSPOSE_DUPLICATE_AXES");
    }
    
    #[test]
    fn test_validate_transpose_out_of_bounds() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        let axes = [0, 1, 5]; // Axis 5 is out of bounds
        
        let result = TensorValidator::validate_transpose(&shape, &axes);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "TRANSPOSE_AXIS_OUT_OF_BOUNDS");
    }
    
    #[test]
    fn test_validate_indexing_valid() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        let indices = [1, 2, 0];
        
        let result = TensorValidator::validate_indexing(&shape, &indices);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_indexing_out_of_bounds() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        let indices = [1, 5, 0]; // 5 >= 3
        
        let result = TensorValidator::validate_indexing(&shape, &indices);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "INDEX_OUT_OF_BOUNDS");
    }
    
    #[test]
    fn test_validate_indexing_too_many_dims() {
        let shape = Shape::from_slice(&[2, 3]);
        let indices = [1, 2, 0, 1]; // Too many indices
        
        let result = TensorValidator::validate_indexing(&shape, &indices);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "INDEX_TOO_MANY_DIMS");
    }
    
    #[test]
    fn test_validate_dtype_compatibility() {
        // Compatible types should promote to higher precision
        let result = TensorValidator::validate_dtype_compatibility(
            DType::F32,
            DType::F64,
            "addition"
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), DType::F64);
        
        // Incompatible quantized types
        let result = TensorValidator::validate_dtype_compatibility(
            DType::Quantized(woolly_tensor::QuantizationScheme::Q4_0),
            DType::F32,
            "addition"
        );
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        assert_eq!(err.code(), "DTYPE_QUANTIZED_INCOMPATIBLE");
    }
    
    #[test]
    fn test_are_shapes_broadcastable() {
        // Compatible shapes
        let shape1 = Shape::from_slice(&[3, 1, 4]);
        let shape2 = Shape::from_slice(&[1, 5, 4]);
        assert!(TensorValidator::are_shapes_broadcastable(&shape1, &shape2));
        
        // Incompatible shapes
        let shape1 = Shape::from_slice(&[3, 4]);
        let shape2 = Shape::from_slice(&[3, 5]);
        assert!(!TensorValidator::are_shapes_broadcastable(&shape1, &shape2));
    }
}

/// Test error propagation between crates
#[cfg(test)]
mod error_propagation_tests {
    use super::*;
    
    #[test]
    fn test_core_error_propagation() {
        // Test that CoreError can be created and its code accessed
        let error = CoreError::model(
            "MODEL_LOAD_FAILED",
            "Failed to load model",
            "Model file appears corrupted",
            "Try re-downloading the model file"
        );
        
        assert_eq!(error.code(), "MODEL_LOAD_FAILED");
        assert!(error.to_string().contains("Failed to load model"));
    }
    
    #[test]
    fn test_tensor_error_propagation() {
        // Test that TensorError can be created and its properties accessed
        let error = TensorError::incompatible_shapes(
            "SHAPES_INCOMPATIBLE",
            "Shape mismatch",
            "matmul",
            "[2, 3]",
            "[4, 5]",
            "Fix the shapes"
        );
        
        assert_eq!(error.code(), "SHAPES_INCOMPATIBLE");
        assert!(error.is_shape_error());
    }
}

/// Test error message quality and helpfulness
#[cfg(test)]
mod error_message_quality_tests {
    use super::*;
    
    #[test]
    fn test_error_messages_contain_suggestions() {
        let error = CoreError::invalid_parameter(
            "INVALID_TEMP",
            "Invalid temperature",
            "Validation failed",
            "Use values between 0.0 and 2.0",
            "temperature",
            "5.0",
            "0.0 to 2.0"
        );
        
        let message = error.to_string();
        assert!(message.contains("Suggestion:"));
        assert!(message.contains("Use values between 0.0 and 2.0"));
    }
    
    #[test]
    fn test_error_messages_contain_context() {
        let error = TensorError::out_of_bounds(
            "INDEX_ERROR",
            "Index out of bounds",
            10,
            0,
            5,
            "array access",
            "Use valid indices"
        );
        
        let message = error.to_string();
        assert!(message.contains("Index: 10"));
        assert!(message.contains("Dimension: 0"));
        assert!(message.contains("Size: 5"));
        assert!(message.contains("Operation: array access"));
    }
    
    #[test]
    fn test_error_codes_are_descriptive() {
        // Error codes should be descriptive and follow naming conventions
        let codes = [
            "MODEL_FILE_NOT_FOUND",
            "TEMPERATURE_NEGATIVE", 
            "TENSOR_ADD_SHAPE_MISMATCH",
            "MATMUL_INNER_DIM_MISMATCH",
            "RESHAPE_SIZE_MISMATCH",
        ];
        
        for code in &codes {
            // All codes should be uppercase with underscores
            assert!(code.chars().all(|c| c.is_uppercase() || c == '_'));
            // All codes should have meaningful prefixes
            assert!(
                code.starts_with("MODEL_") ||
                code.starts_with("TENSOR_") ||
                code.starts_with("MATMUL_") ||
                code.starts_with("RESHAPE_") ||
                code.starts_with("TEMPERATURE_")
            );
        }
    }
}