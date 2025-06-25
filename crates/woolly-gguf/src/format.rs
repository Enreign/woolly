//! GGUF format definitions
//!
//! This module contains the core format definitions for GGUF files,
//! including magic numbers, version information, and header structures.

use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Read;

use crate::error::{Error, Result};

/// GGUF magic number: "GGUF" in ASCII
pub const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];

/// Default alignment for tensor data
pub const GGUF_DEFAULT_ALIGNMENT: u32 = 32;

/// GGUF file magic identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GGUFMagic([u8; 4]);

impl GGUFMagic {
    /// Create from bytes
    pub fn from_bytes(bytes: [u8; 4]) -> Self {
        Self(bytes)
    }
    
    /// Check if this is a valid GGUF magic
    pub fn is_valid(&self) -> bool {
        self.0 == GGUF_MAGIC
    }
    
    /// Get the raw bytes
    pub fn as_bytes(&self) -> &[u8; 4] {
        &self.0
    }
}

/// GGUF version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GGUFVersion(pub u32);

impl GGUFVersion {
    /// Version 1
    pub const V1: Self = Self(1);
    
    /// Version 2
    pub const V2: Self = Self(2);
    
    /// Version 3 (current)
    pub const V3: Self = Self(3);
    
    /// Check if this version is supported
    pub fn is_supported(&self) -> bool {
        matches!(self.0, 1..=3)
    }
}

/// GGUF file header
#[derive(Debug, Clone)]
pub struct GGUFHeader {
    /// Magic number (should be "GGUF")
    pub magic: GGUFMagic,
    
    /// Format version
    pub version: GGUFVersion,
    
    /// Number of tensors
    pub tensor_count: u64,
    
    /// Number of metadata key-value pairs
    pub metadata_kv_count: u64,
}

impl GGUFHeader {
    /// Size of the fixed header in bytes
    pub const SIZE: usize = 4 + 4 + 8 + 8; // magic + version + tensor_count + metadata_kv_count
    
    /// Read header from a reader
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        // Read magic
        let mut magic_bytes = [0u8; 4];
        reader.read_exact(&mut magic_bytes)?;
        let magic = GGUFMagic::from_bytes(magic_bytes);
        
        if !magic.is_valid() {
            return Err(Error::InvalidMagic(magic_bytes));
        }
        
        // Read version
        let version = GGUFVersion(reader.read_u32::<LittleEndian>()?);
        if !version.is_supported() {
            return Err(Error::UnsupportedVersion(version.0));
        }
        
        // Read counts
        let tensor_count = reader.read_u64::<LittleEndian>()?;
        let metadata_kv_count = reader.read_u64::<LittleEndian>()?;
        
        Ok(Self {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }
    
    /// Parse header from a byte slice
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < Self::SIZE {
            return Err(Error::BufferTooSmall {
                needed: Self::SIZE,
                available: data.len(),
            });
        }
        
        let mut cursor = std::io::Cursor::new(data);
        Self::read_from(&mut cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gguf_magic() {
        let valid_magic = GGUFMagic::from_bytes(GGUF_MAGIC);
        assert!(valid_magic.is_valid());
        
        let invalid_magic = GGUFMagic::from_bytes([b'G', b'G', b'M', b'L']);
        assert!(!invalid_magic.is_valid());
    }
    
    #[test]
    fn test_gguf_version() {
        assert!(GGUFVersion::V1.is_supported());
        assert!(GGUFVersion::V2.is_supported());
        assert!(GGUFVersion::V3.is_supported());
        assert!(!GGUFVersion(0).is_supported());
        assert!(!GGUFVersion(4).is_supported());
    }
    
    #[test]
    fn test_header_size() {
        assert_eq!(GGUFHeader::SIZE, 24);
    }
}