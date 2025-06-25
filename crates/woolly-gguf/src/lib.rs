//! Zero-copy, memory-mapped GGUF format loader for Rust
//!
//! This crate provides efficient loading of GGUF (GGML Universal File) format models
//! using memory-mapped I/O for minimal memory overhead and fast access.

pub mod error;
pub mod format;
pub mod loader;
pub mod metadata;
pub mod tensor_info;

pub use error::{Error, Result};
pub use format::{GGUFHeader, GGUFMagic, GGUFVersion};
pub use loader::GGUFLoader;
pub use metadata::{GGUFMetadata, MetadataValue, MetadataValueType};
pub use tensor_info::{GGMLType, TensorInfo};