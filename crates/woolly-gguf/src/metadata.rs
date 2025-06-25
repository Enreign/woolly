//! GGUF metadata structures
//!
//! This module contains structures for handling GGUF metadata,
//! including key-value pairs and various metadata value types.

use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::Read;

use crate::error::{Error, Result};

/// Metadata value types as defined in GGUF spec
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetadataValueType {
    /// 8-bit unsigned integer
    UInt8 = 0,
    /// 8-bit signed integer  
    Int8 = 1,
    /// 16-bit unsigned integer
    UInt16 = 2,
    /// 16-bit signed integer
    Int16 = 3,
    /// 32-bit unsigned integer
    UInt32 = 4,
    /// 32-bit signed integer
    Int32 = 5,
    /// 32-bit float
    Float32 = 6,
    /// Boolean
    Bool = 7,
    /// String
    String = 8,
    /// Array
    Array = 9,
    /// 64-bit unsigned integer
    UInt64 = 10,
    /// 64-bit signed integer
    Int64 = 11,
    /// 64-bit float
    Float64 = 12,
}

impl MetadataValueType {
    /// Try to create from a u32 value
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::UInt8),
            1 => Some(Self::Int8),
            2 => Some(Self::UInt16),
            3 => Some(Self::Int16),
            4 => Some(Self::UInt32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::UInt64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// A metadata value
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    /// Read a metadata value from a reader
    pub fn read_from<R: Read>(reader: &mut R, value_type: MetadataValueType) -> Result<Self> {
        match value_type {
            MetadataValueType::UInt8 => Ok(MetadataValue::UInt8(reader.read_u8()?)),
            MetadataValueType::Int8 => Ok(MetadataValue::Int8(reader.read_i8()?)),
            MetadataValueType::UInt16 => Ok(MetadataValue::UInt16(reader.read_u16::<LittleEndian>()?)),
            MetadataValueType::Int16 => Ok(MetadataValue::Int16(reader.read_i16::<LittleEndian>()?)),
            MetadataValueType::UInt32 => Ok(MetadataValue::UInt32(reader.read_u32::<LittleEndian>()?)),
            MetadataValueType::Int32 => Ok(MetadataValue::Int32(reader.read_i32::<LittleEndian>()?)),
            MetadataValueType::Float32 => Ok(MetadataValue::Float32(reader.read_f32::<LittleEndian>()?)),
            MetadataValueType::Bool => Ok(MetadataValue::Bool(reader.read_u8()? != 0)),
            MetadataValueType::String => {
                let len = reader.read_u64::<LittleEndian>()? as usize;
                let mut buf = vec![0u8; len];
                reader.read_exact(&mut buf)?;
                let string = String::from_utf8(buf)
                    .map_err(|_| Error::InvalidString)?;
                Ok(MetadataValue::String(string))
            }
            MetadataValueType::Array => {
                let array_type = reader.read_u32::<LittleEndian>()?;
                let array_type = MetadataValueType::from_u32(array_type)
                    .ok_or_else(|| Error::InvalidMetadata(format!("Invalid array type: {}", array_type)))?;
                let len = reader.read_u64::<LittleEndian>()? as usize;
                
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(Self::read_from(reader, array_type)?);
                }
                Ok(MetadataValue::Array(values))
            }
            MetadataValueType::UInt64 => Ok(MetadataValue::UInt64(reader.read_u64::<LittleEndian>()?)),
            MetadataValueType::Int64 => Ok(MetadataValue::Int64(reader.read_i64::<LittleEndian>()?)),
            MetadataValueType::Float64 => Ok(MetadataValue::Float64(reader.read_f64::<LittleEndian>()?)),
        }
    }
    
    /// Get value type
    pub fn value_type(&self) -> MetadataValueType {
        match self {
            MetadataValue::UInt8(_) => MetadataValueType::UInt8,
            MetadataValue::Int8(_) => MetadataValueType::Int8,
            MetadataValue::UInt16(_) => MetadataValueType::UInt16,
            MetadataValue::Int16(_) => MetadataValueType::Int16,
            MetadataValue::UInt32(_) => MetadataValueType::UInt32,
            MetadataValue::Int32(_) => MetadataValueType::Int32,
            MetadataValue::Float32(_) => MetadataValueType::Float32,
            MetadataValue::Bool(_) => MetadataValueType::Bool,
            MetadataValue::String(_) => MetadataValueType::String,
            MetadataValue::Array(_) => MetadataValueType::Array,
            MetadataValue::UInt64(_) => MetadataValueType::UInt64,
            MetadataValue::Int64(_) => MetadataValueType::Int64,
            MetadataValue::Float64(_) => MetadataValueType::Float64,
        }
    }
}

/// GGUF metadata container
#[derive(Debug, Clone, Default)]
pub struct GGUFMetadata {
    /// Key-value pairs
    pub kv_pairs: HashMap<String, MetadataValue>,
}

impl GGUFMetadata {
    /// Create a new empty metadata container
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Read metadata from a reader
    pub fn read_from<R: Read>(reader: &mut R, count: u64) -> Result<Self> {
        let mut metadata = Self::new();
        
        for _ in 0..count {
            // Read key
            let key_len = reader.read_u64::<LittleEndian>()? as usize;
            let mut key_buf = vec![0u8; key_len];
            reader.read_exact(&mut key_buf)?;
            let key = String::from_utf8(key_buf)
                .map_err(|_| Error::InvalidString)?;
            
            // Read value type
            let value_type = reader.read_u32::<LittleEndian>()?;
            let value_type = MetadataValueType::from_u32(value_type)
                .ok_or_else(|| Error::InvalidMetadata(format!("Invalid value type: {}", value_type)))?;
            
            // Read value
            let value = MetadataValue::read_from(reader, value_type)?;
            
            metadata.kv_pairs.insert(key, value);
        }
        
        Ok(metadata)
    }
    
    /// Get a value by key
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.kv_pairs.get(key)
    }
    
    /// Get a string value by key
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }
    
    /// Get a u32 value by key
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.get(key)? {
            MetadataValue::UInt32(v) => Some(*v),
            _ => None,
        }
    }
    
    /// Get a u64 value by key
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.get(key)? {
            MetadataValue::UInt64(v) => Some(*v),
            _ => None,
        }
    }
    
    /// Get an i32 value by key
    pub fn get_i32(&self, key: &str) -> Option<i32> {
        match self.get(key)? {
            MetadataValue::Int32(v) => Some(*v),
            _ => None,
        }
    }
    
    /// Get a float32 value by key
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.get(key)? {
            MetadataValue::Float32(v) => Some(*v),
            _ => None,
        }
    }
    
    /// Get a bool value by key
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key)? {
            MetadataValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// Common metadata keys
pub mod keys {
    /// General metadata keys
    pub const GENERAL_ARCHITECTURE: &str = "general.architecture";
    pub const GENERAL_QUANTIZATION_VERSION: &str = "general.quantization_version";
    pub const GENERAL_ALIGNMENT: &str = "general.alignment";
    pub const GENERAL_NAME: &str = "general.name";
    pub const GENERAL_AUTHOR: &str = "general.author";
    pub const GENERAL_URL: &str = "general.url";
    pub const GENERAL_DESCRIPTION: &str = "general.description";
    pub const GENERAL_LICENSE: &str = "general.license";
    pub const GENERAL_SOURCE_URL: &str = "general.source.url";
    pub const GENERAL_SOURCE_HF_REPO: &str = "general.source.huggingface.repository";
    
    /// Model-specific metadata keys
    pub const MODEL_TENSOR_DATA_LAYOUT: &str = "model.tensor_data_layout";
    
    /// LLaMA-specific metadata keys
    pub const LLAMA_CONTEXT_LENGTH: &str = "llama.context_length";
    pub const LLAMA_EMBEDDING_LENGTH: &str = "llama.embedding_length";
    pub const LLAMA_BLOCK_COUNT: &str = "llama.block_count";
    pub const LLAMA_FEED_FORWARD_LENGTH: &str = "llama.feed_forward_length";
    pub const LLAMA_ROPE_DIMENSION_COUNT: &str = "llama.rope.dimension_count";
    pub const LLAMA_ATTENTION_HEAD_COUNT: &str = "llama.attention.head_count";
    pub const LLAMA_ATTENTION_HEAD_COUNT_KV: &str = "llama.attention.head_count_kv";
    pub const LLAMA_ATTENTION_LAYER_NORM_RMS_EPSILON: &str = "llama.attention.layer_norm_rms_epsilon";
}