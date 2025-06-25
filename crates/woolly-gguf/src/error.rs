//! Error types for GGUF loading

use std::io;
use thiserror::Error;

/// Result type alias for GGUF operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during GGUF loading and processing
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error occurred
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    
    /// Invalid GGUF magic number
    #[error("Invalid GGUF magic: expected 'GGUF', found {0:?}")]
    InvalidMagic([u8; 4]),
    
    /// Unsupported GGUF version
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    
    /// Invalid metadata format
    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),
    
    /// Invalid tensor information
    #[error("Invalid tensor info: {0}")]
    InvalidTensorInfo(String),
    
    /// Tensor not found
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
    
    /// Invalid tensor type
    #[error("Invalid tensor type: {0}")]
    InvalidTensorType(u32),
    
    /// Alignment error
    #[error("Alignment error: offset {offset} is not aligned to {alignment} bytes")]
    AlignmentError {
        offset: u64,
        alignment: u64,
    },
    
    /// Buffer too small
    #[error("Buffer too small: needed {needed} bytes, but only {available} available")]
    BufferTooSmall {
        needed: usize,
        available: usize,
    },
    
    /// Invalid UTF-8 string
    #[error("Invalid UTF-8 string: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),
    
    /// Invalid string encoding
    #[error("Invalid string encoding")]
    InvalidString,
    
    /// Memory map error
    #[error("Memory mapping failed: {0}")]
    MemoryMapError(String),
}