//! Integration tests for woolly-gguf

use woolly_gguf::{Error, GGUFMagic, GGUFVersion, GGMLType};

#[test]
fn test_magic_validation() {
    let valid = GGUFMagic::from_bytes([b'G', b'G', b'U', b'F']);
    assert!(valid.is_valid());
    
    let invalid = GGUFMagic::from_bytes([b'G', b'G', b'M', b'L']);
    assert!(!invalid.is_valid());
}

#[test]
fn test_version_support() {
    assert!(GGUFVersion::V1.is_supported());
    assert!(GGUFVersion::V2.is_supported());
    assert!(GGUFVersion::V3.is_supported());
    assert!(!GGUFVersion(0).is_supported());
    assert!(!GGUFVersion(999).is_supported());
}

#[test]
fn test_ggml_type_conversion() {
    assert_eq!(GGMLType::from_u32(0), Some(GGMLType::F32));
    assert_eq!(GGMLType::from_u32(1), Some(GGMLType::F16));
    assert_eq!(GGMLType::from_u32(2), Some(GGMLType::Q4_0));
    assert_eq!(GGMLType::from_u32(999), None);
}

#[test]
fn test_error_types() {
    // Test that errors can be created
    let _io_err: Error = std::io::Error::new(std::io::ErrorKind::NotFound, "test").into();
    let _magic_err = Error::InvalidMagic([0, 1, 2, 3]);
    let _version_err = Error::UnsupportedVersion(999);
}

#[test]
fn test_tensor_size_calculations() {
    // Test non-quantized types
    assert_eq!(GGMLType::F32.element_size(), 4);
    assert_eq!(GGMLType::F16.element_size(), 2);
    assert_eq!(GGMLType::I32.element_size(), 4);
    
    // Test quantized types
    assert_eq!(GGMLType::Q4_0.block_size(), 32);
    assert_eq!(GGMLType::Q4_0.type_size(), 18);
    assert!(GGMLType::Q4_0.is_quantized());
    
    assert!(!GGMLType::F32.is_quantized());
}

// Note: More comprehensive tests would require creating actual GGUF test files
// or mocking the file format, which is beyond the scope of this basic structure.