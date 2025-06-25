//! Integration tests for loading GGUF model files
//!
//! These tests verify the complete functionality of the woolly-gguf crate,
//! including file loading, metadata parsing, tensor information extraction,
//! and memory-mapped data access.

use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{Cursor, Write};
use tempfile::NamedTempFile;
use woolly_gguf::{
    Error, GGMLType, GGUFLoader, GGUFVersion,
    MetadataValue,
};

/// Helper to create a minimal valid GGUF file for testing
struct MockGGUFBuilder {
    metadata: Vec<(String, MetadataValue)>,
    tensors: Vec<MockTensor>,
}

struct MockTensor {
    name: String,
    shape: Vec<u64>,
    ggml_type: GGMLType,
    data: Vec<u8>,
}

impl MockGGUFBuilder {
    fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    fn add_metadata(mut self, key: &str, value: MetadataValue) -> Self {
        self.metadata.push((key.to_string(), value));
        self
    }

    fn add_tensor(mut self, name: &str, shape: Vec<u64>, ggml_type: GGMLType, data: Vec<u8>) -> Self {
        self.tensors.push(MockTensor {
            name: name.to_string(),
            shape,
            ggml_type,
            data,
        });
        self
    }

    fn build(self) -> Vec<u8> {
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);

        // Write header
        cursor.write_all(&[b'G', b'G', b'U', b'F']).unwrap(); // Magic
        cursor.write_u32::<LittleEndian>(3).unwrap(); // Version 3
        cursor.write_u64::<LittleEndian>(self.tensors.len() as u64).unwrap(); // Tensor count
        cursor.write_u64::<LittleEndian>(self.metadata.len() as u64).unwrap(); // Metadata count

        // Write metadata
        for (key, value) in &self.metadata {
            // Write key
            cursor.write_u64::<LittleEndian>(key.len() as u64).unwrap();
            cursor.write_all(key.as_bytes()).unwrap();

            // Write value type
            cursor.write_u32::<LittleEndian>(value.value_type() as u32).unwrap();

            // Write value
            match value {
                MetadataValue::UInt8(v) => cursor.write_u8(*v).unwrap(),
                MetadataValue::Int8(v) => cursor.write_i8(*v).unwrap(),
                MetadataValue::UInt16(v) => cursor.write_u16::<LittleEndian>(*v).unwrap(),
                MetadataValue::Int16(v) => cursor.write_i16::<LittleEndian>(*v).unwrap(),
                MetadataValue::UInt32(v) => cursor.write_u32::<LittleEndian>(*v).unwrap(),
                MetadataValue::Int32(v) => cursor.write_i32::<LittleEndian>(*v).unwrap(),
                MetadataValue::Float32(v) => cursor.write_f32::<LittleEndian>(*v).unwrap(),
                MetadataValue::Bool(v) => cursor.write_u8(if *v { 1 } else { 0 }).unwrap(),
                MetadataValue::String(v) => {
                    cursor.write_u64::<LittleEndian>(v.len() as u64).unwrap();
                    cursor.write_all(v.as_bytes()).unwrap();
                }
                MetadataValue::UInt64(v) => cursor.write_u64::<LittleEndian>(*v).unwrap(),
                MetadataValue::Int64(v) => cursor.write_i64::<LittleEndian>(*v).unwrap(),
                MetadataValue::Float64(v) => cursor.write_f64::<LittleEndian>(*v).unwrap(),
                MetadataValue::Array(arr) => {
                    if !arr.is_empty() {
                        cursor.write_u32::<LittleEndian>(arr[0].value_type() as u32).unwrap();
                        cursor.write_u64::<LittleEndian>(arr.len() as u64).unwrap();
                        // For simplicity, we only support homogeneous arrays of u32
                        for item in arr {
                            if let MetadataValue::UInt32(v) = item {
                                cursor.write_u32::<LittleEndian>(*v).unwrap();
                            }
                        }
                    }
                }
            }
        }

        // Write tensor info
        let mut offset = 0u64;
        for tensor in &self.tensors {
            // Name
            cursor.write_u64::<LittleEndian>(tensor.name.len() as u64).unwrap();
            cursor.write_all(tensor.name.as_bytes()).unwrap();

            // Dimensions
            cursor.write_u32::<LittleEndian>(tensor.shape.len() as u32).unwrap();
            for dim in &tensor.shape {
                cursor.write_u64::<LittleEndian>(*dim).unwrap();
            }

            // Type
            cursor.write_u32::<LittleEndian>(tensor.ggml_type as u32).unwrap();

            // Offset
            cursor.write_u64::<LittleEndian>(offset).unwrap();

            // Update offset for next tensor
            let num_elements: u64 = tensor.shape.iter().product();
            if tensor.ggml_type.is_quantized() {
                let blocks = (num_elements as usize + tensor.ggml_type.block_size() - 1) 
                    / tensor.ggml_type.block_size();
                offset += (blocks * tensor.ggml_type.type_size()) as u64;
            } else {
                offset += num_elements * tensor.ggml_type.element_size() as u64;
            }
        }

        // Align to 32 bytes for data section
        let current_pos = cursor.position() as usize;
        let aligned_pos = (current_pos + 31) & !31;
        let padding = aligned_pos - current_pos;
        cursor.write_all(&vec![0u8; padding]).unwrap();

        // Write tensor data
        for tensor in &self.tensors {
            cursor.write_all(&tensor.data).unwrap();
        }

        buffer
    }
}

#[test]
fn test_load_empty_gguf() {
    // Create a minimal GGUF file with no tensors or metadata
    let mock_data = MockGGUFBuilder::new().build();

    // Write to temporary file
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&mock_data).unwrap();
    temp_file.flush().unwrap();

    // Load the file
    let loader = GGUFLoader::from_path(temp_file.path()).unwrap();

    // Verify header
    assert!(loader.header().magic.is_valid());
    assert_eq!(loader.header().version, GGUFVersion::V3);
    assert_eq!(loader.header().tensor_count, 0);
    assert_eq!(loader.header().metadata_kv_count, 0);

    // Verify empty collections
    assert_eq!(loader.tensors().len(), 0);
    assert_eq!(loader.metadata().kv_pairs.len(), 0);
}

#[test]
fn test_load_with_metadata() {
    // Create GGUF with various metadata types
    let mock_data = MockGGUFBuilder::new()
        .add_metadata("general.architecture", MetadataValue::String("test_arch".to_string()))
        .add_metadata("general.name", MetadataValue::String("Test Model".to_string()))
        .add_metadata("general.quantization_version", MetadataValue::UInt32(2))
        .add_metadata("general.alignment", MetadataValue::UInt32(32))
        .add_metadata("test.bool_value", MetadataValue::Bool(true))
        .add_metadata("test.float_value", MetadataValue::Float32(3.14))
        .add_metadata("test.array_value", MetadataValue::Array(vec![
            MetadataValue::UInt32(1),
            MetadataValue::UInt32(2),
            MetadataValue::UInt32(3),
        ]))
        .build();

    // Write to temporary file
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&mock_data).unwrap();
    temp_file.flush().unwrap();

    // Load the file
    let loader = GGUFLoader::from_path(temp_file.path()).unwrap();

    // Verify metadata
    assert_eq!(loader.header().metadata_kv_count, 7);
    assert_eq!(loader.architecture(), Some("test_arch"));
    assert_eq!(loader.model_name(), Some("Test Model"));
    assert_eq!(loader.quantization_version(), Some(2));
    assert_eq!(loader.alignment(), 32);

    // Check specific metadata values
    let metadata = loader.metadata();
    assert!(matches!(
        metadata.kv_pairs.get("test.bool_value"),
        Some(MetadataValue::Bool(true))
    ));
    assert!(matches!(
        metadata.kv_pairs.get("test.float_value"),
        Some(MetadataValue::Float32(v)) if (*v - 3.14).abs() < 0.001
    ));
}

#[test]
fn test_load_with_tensors() {
    // Create test tensor data
    let tensor1_data: Vec<u8> = (0..16).map(|i| i as u8).collect(); // 4x4 = 16 bytes for f32
    let tensor2_data: Vec<u8> = (0..32).collect(); // 4x4x2 = 32 bytes for f32

    let mock_data = MockGGUFBuilder::new()
        .add_metadata("general.architecture", MetadataValue::String("test".to_string()))
        .add_tensor("weight1", vec![4, 4], GGMLType::F32, tensor1_data.clone())
        .add_tensor("weight2", vec![4, 4, 2], GGMLType::F32, tensor2_data.clone())
        .build();

    // Write to temporary file
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&mock_data).unwrap();
    temp_file.flush().unwrap();

    // Load the file
    let loader = GGUFLoader::from_path(temp_file.path()).unwrap();

    // Verify tensor count
    assert_eq!(loader.header().tensor_count, 2);
    assert_eq!(loader.tensors().len(), 2);

    // Check tensor names
    let names = loader.tensor_names();
    assert!(names.contains(&"weight1"));
    assert!(names.contains(&"weight2"));

    // Check tensor info
    let tensor1_info = loader.tensor_info("weight1").unwrap();
    assert_eq!(tensor1_info.name, "weight1");
    assert_eq!(tensor1_info.shape(), &[4, 4]);
    assert_eq!(tensor1_info.ggml_type, GGMLType::F32);
    assert_eq!(tensor1_info.n_elements(), 16);
    assert_eq!(tensor1_info.data_size(), 64); // 16 * 4 bytes

    let tensor2_info = loader.tensor_info("weight2").unwrap();
    assert_eq!(tensor2_info.shape(), &[4, 4, 2]);
    assert_eq!(tensor2_info.n_elements(), 32);
}

#[test]
fn test_memory_mapped_tensor_access() {
    // Create a tensor with known pattern
    let mut tensor_data = vec![0u8; 64]; // 16 f32 values
    for i in 0..16 {
        let bytes = (i as f32).to_le_bytes();
        tensor_data[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
    }

    let mock_data = MockGGUFBuilder::new()
        .add_metadata("general.architecture", MetadataValue::String("test".to_string()))
        .add_tensor("test_tensor", vec![4, 4], GGMLType::F32, tensor_data)
        .build();

    // Write to temporary file
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&mock_data).unwrap();
    temp_file.flush().unwrap();

    // Load the file
    let loader = GGUFLoader::from_path(temp_file.path()).unwrap();

    // Access raw tensor data
    let raw_data = loader.tensor_data("test_tensor").unwrap();
    assert_eq!(raw_data.len(), 64); // 16 * 4 bytes

    // Access as typed array
    let float_data = loader.tensor_data_as::<f32>("test_tensor").unwrap();
    assert_eq!(float_data.len(), 16);
    
    // Verify values
    for (i, &value) in float_data.iter().enumerate() {
        assert_eq!(value, i as f32);
    }
}

#[test]
fn test_quantized_tensor_handling() {
    // Create a mock quantized tensor
    // For Q4_0: block_size=32, type_size=18
    // 64 elements = 2 blocks = 36 bytes
    let tensor_data = vec![0u8; 36];

    let mock_data = MockGGUFBuilder::new()
        .add_metadata("general.architecture", MetadataValue::String("test".to_string()))
        .add_tensor("quantized_weight", vec![64], GGMLType::Q4_0, tensor_data)
        .build();

    // Write to temporary file
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&mock_data).unwrap();
    temp_file.flush().unwrap();

    // Load the file
    let loader = GGUFLoader::from_path(temp_file.path()).unwrap();

    // Check tensor info
    let tensor_info = loader.tensor_info("quantized_weight").unwrap();
    assert_eq!(tensor_info.ggml_type, GGMLType::Q4_0);
    assert!(tensor_info.ggml_type.is_quantized());
    assert_eq!(tensor_info.data_size(), 36); // 2 blocks * 18 bytes

    // Verify we can access the raw data
    let raw_data = loader.tensor_data("quantized_weight").unwrap();
    assert_eq!(raw_data.len(), 36);

    // Verify typed access fails for quantized tensors
    let typed_result = loader.tensor_data_as::<f32>("quantized_weight");
    assert!(typed_result.is_err());
}

#[test]
fn test_error_handling() {
    // Test loading non-existent file
    let result = GGUFLoader::from_path("/non/existent/file.gguf");
    assert!(matches!(result, Err(Error::Io(_))));

    // Test invalid magic
    let mut bad_data = vec![b'B', b'A', b'D', b'!'];
    bad_data.extend_from_slice(&[0u8; 20]); // Add some padding

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&bad_data).unwrap();
    temp_file.flush().unwrap();

    let result = GGUFLoader::from_path(temp_file.path());
    assert!(matches!(result, Err(Error::InvalidMagic(_))));

    // Test unsupported version
    let mut bad_version = vec![b'G', b'G', b'U', b'F'];
    bad_version.write_u32::<LittleEndian>(999).unwrap(); // Unsupported version
    bad_version.extend_from_slice(&[0u8; 16]);

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&bad_version).unwrap();
    temp_file.flush().unwrap();

    let result = GGUFLoader::from_path(temp_file.path());
    assert!(matches!(result, Err(Error::UnsupportedVersion(999))));
}

#[test]
fn test_tensor_not_found() {
    let mock_data = MockGGUFBuilder::new()
        .add_metadata("general.architecture", MetadataValue::String("test".to_string()))
        .build();

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&mock_data).unwrap();
    temp_file.flush().unwrap();

    let loader = GGUFLoader::from_path(temp_file.path()).unwrap();

    // Try to access non-existent tensor
    assert!(loader.tensor_info("non_existent").is_none());

    let result = loader.tensor_data("non_existent");
    assert!(matches!(result, Err(Error::TensorNotFound(_))));
}

#[test]
fn test_file_size_and_stats() {
    let tensor_data = vec![0u8; 1024]; // 1KB of tensor data

    let mock_data = MockGGUFBuilder::new()
        .add_metadata("general.architecture", MetadataValue::String("test".to_string()))
        .add_tensor("large_tensor", vec![256], GGMLType::F32, tensor_data)
        .build();

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&mock_data).unwrap();
    temp_file.flush().unwrap();

    let loader = GGUFLoader::from_path(temp_file.path()).unwrap();

    // Check file size
    assert_eq!(loader.file_size(), mock_data.len());

    // Check total tensor size
    assert_eq!(loader.total_tensor_size(), 1024);
}

#[test]
fn test_all_metadata_types() {
    let mock_data = MockGGUFBuilder::new()
        .add_metadata("test.u8", MetadataValue::UInt8(255))
        .add_metadata("test.i8", MetadataValue::Int8(-128))
        .add_metadata("test.u16", MetadataValue::UInt16(65535))
        .add_metadata("test.i16", MetadataValue::Int16(-32768))
        .add_metadata("test.u32", MetadataValue::UInt32(4294967295))
        .add_metadata("test.i32", MetadataValue::Int32(-2147483648))
        .add_metadata("test.f32", MetadataValue::Float32(3.14159))
        .add_metadata("test.bool_true", MetadataValue::Bool(true))
        .add_metadata("test.bool_false", MetadataValue::Bool(false))
        .add_metadata("test.string", MetadataValue::String("Hello, GGUF!".to_string()))
        .add_metadata("test.u64", MetadataValue::UInt64(18446744073709551615))
        .add_metadata("test.i64", MetadataValue::Int64(-9223372036854775808))
        .add_metadata("test.f64", MetadataValue::Float64(2.718281828))
        .build();

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&mock_data).unwrap();
    temp_file.flush().unwrap();

    let loader = GGUFLoader::from_path(temp_file.path()).unwrap();
    let metadata = loader.metadata();

    // Verify all types loaded correctly
    assert!(matches!(metadata.get("test.u8"), Some(MetadataValue::UInt8(255))));
    assert!(matches!(metadata.get("test.i8"), Some(MetadataValue::Int8(-128))));
    assert!(matches!(metadata.get("test.u16"), Some(MetadataValue::UInt16(65535))));
    assert!(matches!(metadata.get("test.i16"), Some(MetadataValue::Int16(-32768))));
    assert_eq!(metadata.get_u32("test.u32"), Some(4294967295));
    assert_eq!(metadata.get_i32("test.i32"), Some(-2147483648));
    assert_eq!(metadata.get_bool("test.bool_true"), Some(true));
    assert_eq!(metadata.get_bool("test.bool_false"), Some(false));
    assert_eq!(metadata.get_string("test.string"), Some("Hello, GGUF!"));
    assert_eq!(metadata.get_u64("test.u64"), Some(18446744073709551615));
    assert!(matches!(metadata.get("test.i64"), Some(MetadataValue::Int64(-9223372036854775808))));
    
    // Check floats with epsilon
    assert!((metadata.get_f32("test.f32").unwrap() - 3.14159).abs() < 0.00001);
    if let Some(MetadataValue::Float64(v)) = metadata.get("test.f64") {
        assert!((v - 2.718281828).abs() < 0.000000001);
    } else {
        panic!("Expected Float64 metadata value");
    }
}

// Test documentation:
// 
// To run these tests:
// ```bash
// # Run all tests in the woolly-gguf crate
// cargo test -p woolly-gguf
// 
// # Run only the integration tests
// cargo test -p woolly-gguf --test load_model
// 
// # Run a specific test
// cargo test -p woolly-gguf test_load_with_metadata
// 
// # Run tests with output
// cargo test -p woolly-gguf -- --nocapture
// ```
// 
// These tests verify:
// - Loading empty GGUF files
// - Loading files with various metadata types
// - Loading files with tensor information
// - Memory-mapped tensor data access
// - Quantized tensor handling
// - Error handling for various failure cases
// - File statistics and size calculations