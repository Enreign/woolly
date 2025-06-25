//! Woolly Core - LLM Inference Engine
//!
//! This crate provides the core inference engine for the Woolly LLM system,
//! handling model execution, context management, and generation.

// Module declarations
pub mod config;
pub mod engine;
pub mod kv_cache;
pub mod model;
pub mod model_cache;
pub mod session;
pub mod tokenizer;
pub mod tensor_utils;
pub mod validation;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    /// Model loading or validation errors
    #[error("Model error [{code}]: {message}\nContext: {context}\nSuggestion: {suggestion}")]
    Model {
        code: &'static str,
        message: String,
        context: String,
        suggestion: String,
        file_path: Option<std::path::PathBuf>,
    },
    
    /// Tokenizer initialization or operation errors
    #[error("Tokenizer error [{code}]: {message}\nContext: {context}\nSuggestion: {suggestion}")]
    Tokenizer {
        code: &'static str,
        message: String,
        context: String,
        suggestion: String,
        tokenizer_type: Option<String>,
    },
    
    /// Text generation and inference errors
    #[error("Generation error [{code}]: {message}\nContext: {context}\nSuggestion: {suggestion}")]
    Generation {
        code: &'static str,
        message: String,
        context: String,
        suggestion: String,
        session_id: Option<String>,
    },
    
    /// Context window and sequence management errors
    #[error("Context error [{code}]: {message}\nContext: {context}\nSuggestion: {suggestion}")]
    Context {
        code: &'static str,
        message: String,
        context: String,
        suggestion: String,
        current_length: Option<usize>,
        max_length: Option<usize>,
    },
    
    /// KV cache and memory management errors
    #[error("Cache error [{code}]: {message}\nContext: {context}\nSuggestion: {suggestion}")]
    Cache {
        code: &'static str,
        message: String,
        context: String,
        suggestion: String,
        cache_size: Option<usize>,
        available_memory: Option<usize>,
    },
    
    /// Input validation and parameter errors
    #[error("Invalid input [{code}]: {message}\nContext: {context}\nSuggestion: {suggestion}")]
    InvalidInput {
        code: &'static str,
        message: String,
        context: String,
        suggestion: String,
        parameter_name: Option<String>,
        value: Option<String>,
        valid_range: Option<String>,
    },
    
    /// Tensor operation and backend errors
    #[error("Tensor error [{code}]: {message}\nContext: {context}\nSuggestion: {suggestion}")]
    Tensor {
        code: &'static str,
        message: String,
        context: String,
        suggestion: String,
        operation: Option<String>,
        shapes: Option<Vec<String>>,
    },
    
    /// Configuration parsing and validation errors
    #[error("Configuration error [{code}]: {message}\nContext: {context}\nSuggestion: {suggestion}")]
    Configuration {
        code: &'static str,
        message: String,
        context: String,
        suggestion: String,
        config_path: Option<std::path::PathBuf>,
        field_name: Option<String>,
    },
    
    /// Resource availability errors (memory, disk, etc.)
    #[error("Resource error [{code}]: {message}\nContext: {context}\nSuggestion: {suggestion}")]
    Resource {
        code: &'static str,
        message: String,
        context: String,
        suggestion: String,
        resource_type: String,
        required: Option<u64>,
        available: Option<u64>,
    },
    
    /// Device and backend compatibility errors
    #[error("Device error [{code}]: {message}\nContext: {context}\nSuggestion: {suggestion}")]
    Device {
        code: &'static str,
        message: String,
        context: String,
        suggestion: String,
        device_type: Option<String>,
        backend: Option<String>,
    },
    
    /// File system and I/O errors
    #[error("IO error [{code}]: {message}\nPath: {path:?}\nSuggestion: {suggestion}")]
    Io {
        code: &'static str,
        message: String,
        path: Option<std::path::PathBuf>,
        suggestion: String,
        #[source]
        source: std::io::Error,
    },
    
    /// Other unexpected errors
    #[error("Internal error [{code}]: {message}\nContext: {context}")]
    Internal {
        code: &'static str,
        message: String,
        context: String,
        #[source]
        source: Option<anyhow::Error>,
    },
}

pub type Result<T> = std::result::Result<T, CoreError>;

impl CoreError {
    /// Create a model error with context
    pub fn model<S1, S2, S3>(
        code: &'static str,
        message: S1,
        context: S2,
        suggestion: S3,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        Self::Model {
            code,
            message: message.into(),
            context: context.into(),
            suggestion: suggestion.into(),
            file_path: None,
        }
    }

    /// Create a model error with file path
    pub fn model_with_path<S1, S2, S3, P>(
        code: &'static str,
        message: S1,
        context: S2,
        suggestion: S3,
        path: P,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
        P: Into<std::path::PathBuf>,
    {
        Self::Model {
            code,
            message: message.into(),
            context: context.into(),
            suggestion: suggestion.into(),
            file_path: Some(path.into()),
        }
    }

    /// Create an invalid input error with parameter details
    pub fn invalid_input<S1, S2, S3>(
        code: &'static str,
        message: S1,
        context: S2,
        suggestion: S3,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        Self::InvalidInput {
            code,
            message: message.into(),
            context: context.into(),
            suggestion: suggestion.into(),
            parameter_name: None,
            value: None,
            valid_range: None,
        }
    }

    /// Create an invalid input error with parameter validation details
    pub fn invalid_parameter<S1, S2, S3, S4, S5, S6>(
        code: &'static str,
        message: S1,
        context: S2,
        suggestion: S3,
        param_name: S4,
        value: S5,
        valid_range: S6,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
        S4: Into<String>,
        S5: Into<String>,
        S6: Into<String>,
    {
        Self::InvalidInput {
            code,
            message: message.into(),
            context: context.into(),
            suggestion: suggestion.into(),
            parameter_name: Some(param_name.into()),
            value: Some(value.into()),
            valid_range: Some(valid_range.into()),
        }
    }

    /// Create a resource error with usage details
    pub fn resource<S1, S2, S3, S4>(
        code: &'static str,
        message: S1,
        context: S2,
        suggestion: S3,
        resource_type: S4,
        required: Option<u64>,
        available: Option<u64>,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
        S4: Into<String>,
    {
        Self::Resource {
            code,
            message: message.into(),
            context: context.into(),
            suggestion: suggestion.into(),
            resource_type: resource_type.into(),
            required,
            available,
        }
    }

    /// Create a configuration error
    pub fn configuration<S1, S2, S3>(
        code: &'static str,
        message: S1,
        context: S2,
        suggestion: S3,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        Self::Configuration {
            code,
            message: message.into(),
            context: context.into(),
            suggestion: suggestion.into(),
            config_path: None,
            field_name: None,
        }
    }

    /// Create a tensor error with operation details
    pub fn tensor<S1, S2, S3>(
        code: &'static str,
        message: S1,
        context: S2,
        suggestion: S3,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        Self::Tensor {
            code,
            message: message.into(),
            context: context.into(),
            suggestion: suggestion.into(),
            operation: None,
            shapes: None,
        }
    }

    /// Get the error code for programmatic handling
    pub fn code(&self) -> &'static str {
        match self {
            Self::Model { code, .. } => code,
            Self::Tokenizer { code, .. } => code,
            Self::Generation { code, .. } => code,
            Self::Context { code, .. } => code,
            Self::Cache { code, .. } => code,
            Self::InvalidInput { code, .. } => code,
            Self::Tensor { code, .. } => code,
            Self::Configuration { code, .. } => code,
            Self::Resource { code, .. } => code,
            Self::Device { code, .. } => code,
            Self::Io { code, .. } => code,
            Self::Internal { code, .. } => code,
        }
    }
}

impl From<std::io::Error> for CoreError {
    fn from(err: std::io::Error) -> Self {
        let (code, suggestion) = match err.kind() {
            std::io::ErrorKind::NotFound => (
                "IO_FILE_NOT_FOUND",
                "Check that the file path is correct and the file exists"
            ),
            std::io::ErrorKind::PermissionDenied => (
                "IO_PERMISSION_DENIED",
                "Check file permissions or run with appropriate privileges"
            ),
            std::io::ErrorKind::InvalidData => (
                "IO_INVALID_DATA",
                "The file may be corrupted or in an unexpected format"
            ),
            std::io::ErrorKind::OutOfMemory => (
                "IO_OUT_OF_MEMORY",
                "Free up system memory or reduce the operation size"
            ),
            _ => (
                "IO_UNKNOWN",
                "Check the file system and try the operation again"
            ),
        };

        Self::Io {
            code,
            message: err.to_string(),
            path: None,
            suggestion: suggestion.to_string(),
            source: err,
        }
    }
}

impl From<anyhow::Error> for CoreError {
    fn from(err: anyhow::Error) -> Self {
        Self::Internal {
            code: "INTERNAL_UNKNOWN",
            message: err.to_string(),
            context: "An unexpected error occurred".to_string(),
            source: Some(err),
        }
    }
}

/// Prelude module for common imports
pub mod prelude {
    pub use crate::{
        config::{EngineConfig, DeviceConfig, DeviceType},
        engine::{InferenceEngine, ModelInfo, SessionInfo},
        kv_cache::{OptimizedKVCache, KVCacheConfig, KVCacheStats, EvictionPolicy},
        model::{Model, ModelConfig, ModelOutput, KVCache, ModelFeature},
        model_cache::{GlobalModelCache, ModelCacheConfig, ModelCacheStats, global_cache},
        session::{InferenceSession, SessionConfig, SessionMemoryStats},
        generation::{GenerationConfig, GenerationResult, FinishReason},
        tokenizer::{Tokenizer, TokenizerConfig, TokenizerType, create_tokenizer},
        sampler::Sampler,
        validation::{Validator, ResourceValidator},
        Result, CoreError,
    };
}

// Re-export key types at the crate root
pub use config::EngineConfig;
pub use engine::InferenceEngine;
pub use model::Model;
pub use session::{InferenceSession, SessionConfig};

// Module placeholders for future implementation
pub mod context {
    pub struct Context {
        // Implementation details - to be implemented
    }
}

pub mod generation {
    #[derive(Debug, Clone)]
    pub struct GenerationConfig {
        pub max_tokens: usize,
        pub temperature: f32,
        pub top_p: f32,
        pub top_k: usize,
        pub repetition_penalty: f32,
    }
    
    #[derive(Debug, Clone)]
    pub struct GenerationResult {
        pub tokens: Vec<u32>,
        pub text: String,
        pub finish_reason: FinishReason,
    }
    
    #[derive(Debug, Clone, Copy)]
    pub enum FinishReason {
        MaxTokens,
        StopToken,
        EosToken,
    }
}

pub mod sampler {
    pub trait Sampler {
        fn sample(&self, logits: &[f32], config: &super::generation::GenerationConfig) -> u32;
    }
}


pub mod error {
    pub use super::{CoreError, Result};
}