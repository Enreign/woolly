//! Input validation utilities for Woolly Core
//!
//! This module provides comprehensive validation for model paths, configurations,
//! parameters, and other inputs to ensure robust error handling and user feedback.

use std::path::{Path, PathBuf};
use std::fs;
use crate::{CoreError, Result};

/// Validation utilities for common operations
pub struct Validator;

impl Validator {
    /// Validate a model file path
    pub fn validate_model_path<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
        let path = path.as_ref();
        let path_buf = path.to_path_buf();
        
        // Check if path exists
        if !path.exists() {
            return Err(CoreError::model_with_path(
                "MODEL_FILE_NOT_FOUND",
                format!("Model file does not exist: {}", path.display()),
                format!("Attempted to load model from: {}", path.display()),
                "Check the file path is correct and the model file exists",
                path_buf.clone(),
            ));
        }
        
        // Check if it's a file (not a directory)
        if !path.is_file() {
            return Err(CoreError::model_with_path(
                "MODEL_PATH_NOT_FILE",
                format!("Model path is not a file: {}", path.display()),
                format!("Path points to: {}", if path.is_dir() { "directory" } else { "unknown" }),
                "Provide a path to a valid model file, not a directory",
                path_buf.clone(),
            ));
        }
        
        // Check file extension for common model formats
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            match ext_str.as_str() {
                "gguf" | "bin" | "safetensors" | "pt" | "pth" | "ckpt" => {
                    // Valid model format
                }
                _ => {
                    return Err(CoreError::model_with_path(
                        "MODEL_UNSUPPORTED_FORMAT",
                        format!("Unsupported model file format: .{}", ext_str),
                        format!("File: {}", path.display()),
                        "Supported formats: .gguf, .bin, .safetensors, .pt, .pth, .ckpt",
                        path_buf.clone(),
                    ));
                }
            }
        } else {
            return Err(CoreError::model_with_path(
                "MODEL_NO_EXTENSION",
                "Model file has no extension",
                format!("File: {}", path.display()),
                "Provide a model file with a valid extension (.gguf, .bin, etc.)",
                path_buf.clone(),
            ));
        }
        
        // Check file size (basic sanity check)
        match fs::metadata(path) {
            Ok(metadata) => {
                let size = metadata.len();
                if size == 0 {
                    return Err(CoreError::model_with_path(
                        "MODEL_FILE_EMPTY",
                        "Model file is empty",
                        format!("File: {} (0 bytes)", path.display()),
                        "Ensure the model file is properly downloaded and not corrupted",
                        path_buf.clone(),
                    ));
                }
                
                // Warn if file is suspiciously small for a model (less than 1MB)
                if size < 1_024_1024 {
                    // Note: We don't error here, just validate it's not empty
                    // Some models or test files might be small
                }
            }
            Err(e) => {
                return Err(CoreError::model_with_path(
                    "MODEL_FILE_ACCESS_ERROR",
                    format!("Cannot access model file: {}", e),
                    format!("File: {}", path.display()),
                    "Check file permissions and ensure the file is not locked",
                    path_buf.clone(),
                ));
            }
        }
        
        Ok(path_buf)
    }
    
    /// Validate generation parameters
    pub fn validate_generation_config(
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        repetition_penalty: Option<f32>,
    ) -> Result<()> {
        // Validate max_tokens
        if let Some(max_tokens) = max_tokens {
            if max_tokens == 0 {
                return Err(CoreError::invalid_parameter(
                    "INVALID_MAX_TOKENS",
                    "max_tokens cannot be zero",
                    "Generation configuration validation",
                    "Use a positive value to generate at least one token",
                    "max_tokens",
                    "0",
                    "1 to 4096 (typical range)"
                ));
            }
            if max_tokens > 100_000 {
                return Err(CoreError::invalid_parameter(
                    "MAX_TOKENS_TOO_LARGE",
                    "max_tokens is unreasonably large",
                    "Generation configuration validation",
                    "Use a smaller value to avoid memory issues and long generation times",
                    "max_tokens",
                    &max_tokens.to_string(),
                    "1 to 4096 (typical range)"
                ));
            }
        }
        
        // Validate temperature
        if let Some(temperature) = temperature {
            if temperature < 0.0 {
                return Err(CoreError::invalid_parameter(
                    "TEMPERATURE_NEGATIVE",
                    "Temperature cannot be negative",
                    "Generation configuration validation",
                    "Use 0.0 for deterministic output or positive values for randomness",
                    "temperature",
                    &temperature.to_string(),
                    "0.0 to 2.0 (typical range)"
                ));
            }
            if temperature > 10.0 {
                return Err(CoreError::invalid_parameter(
                    "TEMPERATURE_TOO_HIGH",
                    "Temperature is unreasonably high",
                    "Generation configuration validation",
                    "Use values between 0.0 and 2.0 for reasonable generation quality",
                    "temperature",
                    &temperature.to_string(),
                    "0.0 to 2.0 (typical range)"
                ));
            }
        }
        
        // Validate top_p
        if let Some(top_p) = top_p {
            if top_p <= 0.0 || top_p > 1.0 {
                return Err(CoreError::invalid_parameter(
                    "TOP_P_OUT_OF_RANGE",
                    "top_p must be between 0.0 and 1.0",
                    "Generation configuration validation",
                    "Use values between 0.0 and 1.0 for nucleus sampling",
                    "top_p",
                    &top_p.to_string(),
                    "0.0 to 1.0"
                ));
            }
        }
        
        // Validate top_k
        if let Some(top_k) = top_k {
            if top_k == 0 {
                return Err(CoreError::invalid_parameter(
                    "TOP_K_ZERO",
                    "top_k cannot be zero",
                    "Generation configuration validation",
                    "Use a positive value or disable top_k sampling",
                    "top_k",
                    "0",
                    "1 to 100 (typical range)"
                ));
            }
            if top_k > 10_000 {
                return Err(CoreError::invalid_parameter(
                    "TOP_K_TOO_LARGE",
                    "top_k is unreasonably large",
                    "Generation configuration validation",
                    "Use smaller values (1-100) for effective top_k sampling",
                    "top_k",
                    &top_k.to_string(),
                    "1 to 100 (typical range)"
                ));
            }
        }
        
        // Validate repetition_penalty
        if let Some(repetition_penalty) = repetition_penalty {
            if repetition_penalty <= 0.0 {
                return Err(CoreError::invalid_parameter(
                    "REPETITION_PENALTY_INVALID",
                    "Repetition penalty must be positive",
                    "Generation configuration validation",
                    "Use values > 1.0 to penalize repetition, < 1.0 to encourage it",
                    "repetition_penalty",
                    &repetition_penalty.to_string(),
                    "0.1 to 2.0 (typical range)"
                ));
            }
            if repetition_penalty > 10.0 {
                return Err(CoreError::invalid_parameter(
                    "REPETITION_PENALTY_TOO_HIGH",
                    "Repetition penalty is unreasonably high",
                    "Generation configuration validation",
                    "Use values between 1.0 and 2.0 for reasonable quality",
                    "repetition_penalty",
                    &repetition_penalty.to_string(),
                    "0.1 to 2.0 (typical range)"
                ));
            }
        }
        
        Ok(())
    }
    
    /// Validate context length parameters
    pub fn validate_context_config(
        context_length: usize,
        max_context_length: usize,
    ) -> Result<()> {
        if context_length == 0 {
            return Err(CoreError::invalid_parameter(
                "CONTEXT_LENGTH_ZERO",
                "Context length cannot be zero",
                "Context configuration validation",
                "Use a positive context length (typically 512, 1024, 2048, or 4096)",
                "context_length",
                "0",
                "512 to 8192 (typical range)"
            ));
        }
        
        if context_length > max_context_length {
            return Err(CoreError::invalid_parameter(
                "CONTEXT_LENGTH_EXCEEDS_MAX",
                format!("Context length {} exceeds maximum {}", context_length, max_context_length),
                "Context configuration validation",
                format!("Use a context length <= {}", max_context_length),
                "context_length",
                &context_length.to_string(),
                &format!("1 to {}", max_context_length)
            ));
        }
        
        // Check for reasonable context lengths (powers of 2 are common)
        let reasonable_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768];
        if !reasonable_lengths.contains(&context_length) {
            // Not an error, but could be a warning in logs
        }
        
        Ok(())
    }
    
    /// Validate tokenizer configuration
    pub fn validate_tokenizer_config(
        vocab_size: Option<usize>,
        tokenizer_type: &str,
    ) -> Result<()> {
        // Validate vocab size
        if let Some(vocab_size) = vocab_size {
            if vocab_size == 0 {
                return Err(CoreError::invalid_parameter(
                    "VOCAB_SIZE_ZERO",
                    "Vocabulary size cannot be zero",
                    "Tokenizer configuration validation",
                    "Ensure the tokenizer has a valid vocabulary",
                    "vocab_size",
                    "0",
                    "1000 to 100000 (typical range)"
                ));
            }
            
            if vocab_size > 1_000_000 {
                return Err(CoreError::invalid_parameter(
                    "VOCAB_SIZE_TOO_LARGE",
                    "Vocabulary size is unreasonably large",
                    "Tokenizer configuration validation",
                    "Check if the vocabulary size is correct",
                    "vocab_size",
                    &vocab_size.to_string(),
                    "1000 to 100000 (typical range)"
                ));
            }
        }
        
        // Validate tokenizer type
        let valid_types = ["bpe", "sentencepiece", "wordpiece", "character"];
        if !valid_types.contains(&tokenizer_type) {
            return Err(CoreError::invalid_parameter(
                "INVALID_TOKENIZER_TYPE",
                format!("Unknown tokenizer type: {}", tokenizer_type),
                "Tokenizer configuration validation",
                format!("Supported types: {}", valid_types.join(", ")),
                "tokenizer_type",
                tokenizer_type,
                "bpe, sentencepiece, wordpiece, character"
            ));
        }
        
        Ok(())
    }
    
    /// Validate device configuration
    pub fn validate_device_config(
        device_type: &str,
        device_id: Option<usize>,
    ) -> Result<()> {
        match device_type.to_lowercase().as_str() {
            "cpu" => {
                // CPU is always valid, device_id should be None
                if device_id.is_some() {
                    return Err(CoreError::invalid_parameter(
                        "CPU_DEVICE_ID_INVALID",
                        "CPU device should not have a device ID",
                        "Device configuration validation",
                        "Remove device_id when using CPU device",
                        "device_id",
                        &device_id.unwrap().to_string(),
                        "None (for CPU)"
                    ));
                }
            }
            "cuda" | "gpu" => {
                // GPU devices should have a valid device ID
                if let Some(id) = device_id {
                    if id > 7 {  // Most systems have at most 8 GPUs (0-7)
                        return Err(CoreError::invalid_parameter(
                            "GPU_DEVICE_ID_TOO_HIGH",
                            format!("GPU device ID {} is unusually high", id),
                            "Device configuration validation",
                            "Check available GPU devices and use a valid ID (typically 0-3)",
                            "device_id",
                            &id.to_string(),
                            "0 to 7 (typical range)"
                        ));
                    }
                } else {
                    return Err(CoreError::invalid_parameter(
                        "GPU_DEVICE_ID_MISSING",
                        "GPU device requires a device ID",
                        "Device configuration validation",
                        "Specify device_id (e.g., 0) when using GPU device",
                        "device_id",
                        "None",
                        "0 to 7 (typical range)"
                    ));
                }
            }
            "metal" => {
                // Metal is valid on macOS, device_id should be None
                if device_id.is_some() {
                    return Err(CoreError::invalid_parameter(
                        "METAL_DEVICE_ID_INVALID",
                        "Metal device should not have a device ID",
                        "Device configuration validation",
                        "Remove device_id when using Metal device",
                        "device_id",
                        &device_id.unwrap().to_string(),
                        "None (for Metal)"
                    ));
                }
            }
            _ => {
                return Err(CoreError::invalid_parameter(
                    "INVALID_DEVICE_TYPE",
                    format!("Unknown device type: {}", device_type),
                    "Device configuration validation",
                    "Supported devices: cpu, cuda, gpu, metal",
                    "device_type",
                    device_type,
                    "cpu, cuda, gpu, metal"
                ));
            }
        }
        
        Ok(())
    }
    
    /// Validate input text
    pub fn validate_input_text(text: &str, max_length: Option<usize>) -> Result<()> {
        if text.is_empty() {
            return Err(CoreError::invalid_input(
                "INPUT_TEXT_EMPTY",
                "Input text cannot be empty",
                "Text input validation",
                "Provide some text for the model to process"
            ));
        }
        
        if let Some(max_length) = max_length {
            if text.len() > max_length {
                return Err(CoreError::invalid_parameter(
                    "INPUT_TEXT_TOO_LONG",
                    format!("Input text is too long: {} characters", text.len()),
                    "Text input validation",
                    format!("Keep input text under {} characters", max_length),
                    "input_text_length",
                    &text.len().to_string(),
                    &format!("1 to {}", max_length)
                ));
            }
        }
        
        // Check for potentially problematic characters
        let control_chars: Vec<char> = text.chars()
            .filter(|c| c.is_control() && *c != '\n' && *c != '\t' && *c != '\r')
            .collect();
        
        if !control_chars.is_empty() {
            return Err(CoreError::invalid_input(
                "INPUT_TEXT_INVALID_CHARS",
                "Input text contains invalid control characters",
                format!("Found {} control characters", control_chars.len()),
                "Remove or replace control characters in the input text"
            ));
        }
        
        Ok(())
    }
}

/// Resource validation utilities
pub struct ResourceValidator;

impl ResourceValidator {
    /// Check available system memory
    pub fn check_memory_available(required_bytes: u64) -> Result<u64> {
        // This is a simplified check - in a real implementation, you'd use
        // system APIs to get actual memory information
        #[cfg(target_os = "linux")]
        {
            Self::check_memory_linux(required_bytes)
        }
        #[cfg(target_os = "macos")]
        {
            Self::check_memory_macos(required_bytes)
        }
        #[cfg(target_os = "windows")]
        {
            Self::check_memory_windows(required_bytes)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            // Default implementation - assume memory is available
            // This should be replaced with actual memory checking
            Ok(required_bytes)
        }
    }
    
    #[cfg(target_os = "linux")]
    fn check_memory_linux(required_bytes: u64) -> Result<u64> {
        // Read /proc/meminfo for available memory
        match std::fs::read_to_string("/proc/meminfo") {
            Ok(content) => {
                let available = content
                    .lines()
                    .find(|line| line.starts_with("MemAvailable:"))
                    .and_then(|line| {
                        line.split_whitespace()
                            .nth(1)
                            .and_then(|s| s.parse::<u64>().ok())
                            .map(|kb| kb * 1024) // Convert KB to bytes
                    });
                
                if let Some(available_bytes) = available {
                    if required_bytes > available_bytes {
                        return Err(CoreError::resource(
                            "INSUFFICIENT_MEMORY",
                            "Not enough memory available for operation",
                            format!("System has {} MB available, operation requires {} MB", 
                                    available_bytes / 1_024_1024, required_bytes / 1_024_1024),
                            "Free up system memory or reduce the operation size",
                            "memory",
                            Some(required_bytes),
                            Some(available_bytes)
                        ));
                    }
                    Ok(available_bytes)
                } else {
                    // Fallback if we can't parse memory info
                    Ok(required_bytes)
                }
            }
            Err(_) => {
                // Fallback if we can't read memory info
                Ok(required_bytes)
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    fn check_memory_macos(_required_bytes: u64) -> Result<u64> {
        // TODO: Implement macOS memory checking using sysctl
        // For now, just return success
        Ok(_required_bytes)
    }
    
    #[cfg(target_os = "windows")]
    fn check_memory_windows(_required_bytes: u64) -> Result<u64> {
        // TODO: Implement Windows memory checking using GlobalMemoryStatusEx
        // For now, just return success
        Ok(_required_bytes)
    }
    
    /// Check available disk space
    pub fn check_disk_space<P: AsRef<Path>>(path: P, required_bytes: u64) -> Result<()> {
        let path = path.as_ref();
        
        // Try to get the parent directory if the path doesn't exist
        let check_path = if path.exists() {
            path
        } else if let Some(parent) = path.parent() {
            parent
        } else {
            Path::new(".")
        };
        
        // For now, this is a placeholder. In a real implementation, you'd use
        // platform-specific APIs to check disk space
        match fs::metadata(check_path) {
            Ok(_) => {
                // TODO: Implement actual disk space checking
                // For now, just check if the directory is accessible
                Ok(())
            }
            Err(e) => {
                Err(CoreError::resource(
                    "DISK_ACCESS_ERROR",
                    format!("Cannot access disk path: {}", e),
                    format!("Path: {}", check_path.display()),
                    "Check disk permissions and availability",
                    "disk_space",
                    Some(required_bytes),
                    None
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;
    
    #[test]
    fn test_validate_model_path_not_found() {
        let result = Validator::validate_model_path("/nonexistent/model.gguf");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), "MODEL_FILE_NOT_FOUND");
    }
    
    #[test]
    fn test_validate_model_path_valid() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("model.gguf");
        let mut file = File::create(&model_path).unwrap();
        writeln!(file, "dummy model content").unwrap();
        
        let result = Validator::validate_model_path(&model_path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_path);
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
    fn test_validate_device_config_invalid_type() {
        let result = Validator::validate_device_config("quantum", None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code(), "INVALID_DEVICE_TYPE");
    }
    
    #[test]
    fn test_validate_device_config_cpu_valid() {
        let result = Validator::validate_device_config("cpu", None);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_validate_device_config_gpu_valid() {
        let result = Validator::validate_device_config("cuda", Some(0));
        assert!(result.is_ok());
    }
}