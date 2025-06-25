# woolly-gguf

Zero-copy, memory-mapped GGUF format loader for Rust.

## Features

- **Zero-copy loading**: Uses memory-mapped I/O for efficient access to large model files
- **Full GGUF support**: Handles all GGUF versions and tensor types
- **Type-safe API**: Strongly typed metadata and tensor information
- **Efficient memory usage**: Models are accessed directly from disk without loading into RAM

## Usage

```rust
use woolly_gguf::{GGUFLoader, Result};

fn main() -> Result<()> {
    // Load a GGUF file
    let loader = GGUFLoader::from_path("model.gguf")?;
    
    // Access metadata
    if let Some(arch) = loader.architecture() {
        println!("Architecture: {}", arch);
    }
    
    // List all tensors
    for name in loader.tensor_names() {
        let info = loader.tensor_info(name).unwrap();
        println!("{}: {:?} ({})", name, info.shape(), info.ggml_type);
    }
    
    // Access tensor data
    let tensor_data = loader.tensor_data("model.embed_tokens.weight")?;
    
    Ok(())
}
```

## Supported Tensor Types

- Floating point: F32, F16, BF16, F64
- Integer: I8, I16, I32, I64
- Quantized: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
- Advanced quantization: IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M

## Testing

The crate includes comprehensive integration tests that verify all functionality:

### Running Tests

```bash
# Run all tests for the woolly-gguf crate
cargo test -p woolly-gguf

# Run only the integration tests
cargo test -p woolly-gguf --test load_model

# Run only unit tests
cargo test -p woolly-gguf --lib

# Run a specific test
cargo test -p woolly-gguf test_load_with_metadata

# Run tests with output visible
cargo test -p woolly-gguf -- --nocapture

# Run tests in release mode (faster for large data)
cargo test -p woolly-gguf --release
```

### Integration Tests

The integration tests (`tests/load_model.rs`) cover:

- **Basic Loading**: Testing empty GGUF files and file format validation
- **Metadata Handling**: All metadata types (integers, floats, strings, arrays, booleans)
- **Tensor Information**: Loading tensor metadata, shapes, and types
- **Memory Mapping**: Accessing tensor data through memory-mapped I/O
- **Quantized Tensors**: Handling of quantized tensor formats
- **Error Handling**: Proper error reporting for invalid files and missing tensors
- **File Statistics**: File size and tensor size calculations

### Mock GGUF Files

Since the tests need valid GGUF files, the integration tests include a `MockGGUFBuilder` that creates minimal but valid GGUF files for testing. This allows testing without requiring actual model files.

### Test Coverage

The tests ensure:
- Header parsing with magic number and version validation
- All metadata value types are correctly parsed
- Tensor information is properly loaded
- Memory-mapped data access works correctly
- Quantized tensor handling respects block sizes
- Error conditions are properly handled

## License

MIT OR Apache-2.0