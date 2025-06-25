# 🦙 Woolly Tensor

[![Crates.io](https://img.shields.io/crates/v/woolly-tensor.svg)](https://crates.io/crates/woolly-tensor)
[![Documentation](https://docs.rs/woolly-tensor/badge.svg)](https://docs.rs/woolly-tensor)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../../LICENSE)

High-performance tensor operations library for Woolly, providing efficient mathematical operations with support for multiple backends including CPU (SIMD-accelerated), CUDA, and Metal.

## Features

- **🚀 Multiple Backends**: CPU, CUDA, and Metal support
- **⚡ SIMD Acceleration**: Optimized operations using AVX2/AVX-512 on x86_64 and NEON on ARM
- **📦 Quantization Support**: Various quantization schemes including llama.cpp compatible formats
- **🔍 Zero-copy Views**: Efficient tensor slicing and broadcasting without data copying
- **🛡️ Type Safety**: Strong typing with compile-time shape checking where possible
- **📐 Flexible Shapes**: Support for arbitrary tensor dimensions and strides
- **🔧 Custom Operations**: Extensible operation system for custom mathematical functions

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
woolly-tensor = "0.1"
```

### Basic Usage

```rust
use woolly_tensor::prelude::*;

fn main() -> Result<(), TensorError> {
    // Create a CPU backend
    let backend = CpuBackend::new();
    
    // Create tensors
    let shape = Shape::matrix(3, 4);
    let a = Tensor::zeros(&backend, shape.clone(), DType::F32)?;
    let b = Tensor::ones(&backend, shape, DType::F32)?;
    
    // Perform operations
    let c = a.add(&b)?;
    let d = c.mul_scalar(2.0)?;
    
    // Access data
    let data = d.to_vec()?;
    println!("Result: {:?}", data);
    
    Ok(())
}
```

### Matrix Operations

```rust
use woolly_tensor::prelude::*;
use woolly_tensor::ops::*;

fn main() -> Result<(), TensorError> {
    let backend = CpuBackend::new();
    
    // Create matrices
    let a = Tensor::from_data(&backend, vec![1.0, 2.0, 3.0, 4.0], Shape::matrix(2, 2), DType::F32)?;
    let b = Tensor::from_data(&backend, vec![5.0, 6.0, 7.0, 8.0], Shape::matrix(2, 2), DType::F32)?;
    
    // Matrix multiplication
    let c = MatMul::apply(&a, &b)?;
    
    // Element-wise operations
    let d = Add::apply(&a, &b)?;
    let e = ReLU::apply(&d)?;
    
    println!("Matrix multiplication result: {:?}", c.to_vec()?);
    println!("Element-wise addition + ReLU: {:?}", e.to_vec()?);
    
    Ok(())
}
```

### SIMD Operations

```rust
use woolly_tensor::prelude::*;
use woolly_tensor::ops::*;

fn main() -> Result<(), TensorError> {
    let backend = CpuBackend::new();
    
    // Large arrays benefit from SIMD acceleration
    let size = 1024 * 1024;
    let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
    
    let shape = Shape::vector(size);
    let a = Tensor::from_data(&backend, data_a, shape.clone(), DType::F32)?;
    let b = Tensor::from_data(&backend, data_b, shape, DType::F32)?;
    
    // SIMD-accelerated operations
    let start = std::time::Instant::now();
    let result = Add::apply(&a, &b)?;
    let duration = start.elapsed();
    
    println!("SIMD addition of {} elements took: {:?}", size, duration);
    
    Ok(())
}
```

## Tensor Creation

### From Data

```rust
use woolly_tensor::prelude::*;

let backend = CpuBackend::new();

// From Vec
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let tensor = Tensor::from_data(&backend, data, Shape::matrix(2, 3), DType::F32)?;

// From slice
let data = &[1.0f32, 2.0, 3.0, 4.0];
let tensor = Tensor::from_slice(&backend, data, Shape::vector(4), DType::F32)?;
```

### Initialization Patterns

```rust
use woolly_tensor::prelude::*;

let backend = CpuBackend::new();
let shape = Shape::from_slice(&[2, 3, 4]);

// Common initializations
let zeros = Tensor::zeros(&backend, shape.clone(), DType::F32)?;
let ones = Tensor::ones(&backend, shape.clone(), DType::F32)?;
let random = Tensor::randn(&backend, shape.clone(), DType::F32)?;

// Custom initialization
let tensor = Tensor::full(&backend, shape, 3.14, DType::F32)?;

// Range tensor
let range = Tensor::arange(&backend, 0.0, 10.0, 1.0, DType::F32)?;
```

### Shape Manipulation

```rust
use woolly_tensor::prelude::*;

let backend = CpuBackend::new();
let tensor = Tensor::arange(&backend, 0.0, 24.0, 1.0, DType::F32)?;

// Reshape
let reshaped = tensor.reshape(&Shape::from_slice(&[2, 3, 4]))?;

// Transpose
let transposed = reshaped.transpose(&[2, 0, 1])?;

// Squeeze and unsqueeze
let squeezed = transposed.squeeze()?;
let unsqueezed = squeezed.unsqueeze(1)?;

// Slice
let sliced = tensor.slice(&[0..12])?;
```

## Operations

### Unary Operations

```rust
use woolly_tensor::prelude::*;
use woolly_tensor::ops::*;

let backend = CpuBackend::new();
let tensor = Tensor::from_data(&backend, vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::vector(5), DType::F32)?;

// Activation functions
let relu = ReLU::apply(&tensor)?;
let sigmoid = Sigmoid::apply(&tensor)?;
let tanh = Tanh::apply(&tensor)?;
let gelu = GELU::apply(&tensor)?;

// Mathematical functions
let abs = Abs::apply(&tensor)?;
let exp = Exp::apply(&tensor)?;
let log = Log::apply(&tensor)?;
let sqrt = Sqrt::apply(&tensor)?;

// Trigonometric functions
let sin = Sin::apply(&tensor)?;
let cos = Cos::apply(&tensor)?;
```

### Binary Operations

```rust
use woolly_tensor::prelude::*;
use woolly_tensor::ops::*;

let backend = CpuBackend::new();
let a = Tensor::ones(&backend, Shape::matrix(3, 3), DType::F32)?;
let b = Tensor::full(&backend, Shape::matrix(3, 3), 2.0, DType::F32)?;

// Arithmetic operations
let add = Add::apply(&a, &b)?;
let sub = Sub::apply(&a, &b)?;
let mul = Mul::apply(&a, &b)?;
let div = Div::apply(&a, &b)?;

// Comparison operations
let eq = Equal::apply(&a, &b)?;
let gt = Greater::apply(&a, &b)?;
let lt = Less::apply(&a, &b)?;

// Logical operations
let and = And::apply(&a, &b)?;
let or = Or::apply(&a, &b)?;
```

### Reduction Operations

```rust
use woolly_tensor::prelude::*;
use woolly_tensor::ops::*;

let backend = CpuBackend::new();
let tensor = Tensor::from_data(&backend, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::matrix(2, 3), DType::F32)?;

// Sum operations
let sum_all = Sum::apply(&tensor, None)?;              // Sum all elements
let sum_dim0 = Sum::apply(&tensor, Some(0))?;          // Sum along dimension 0
let sum_dim1 = Sum::apply(&tensor, Some(1))?;          // Sum along dimension 1

// Other reductions
let mean = Mean::apply(&tensor, Some(1))?;
let max = Max::apply(&tensor, Some(0))?;
let min = Min::apply(&tensor, Some(0))?;
let argmax = ArgMax::apply(&tensor, 1)?;
```

### Linear Algebra

```rust
use woolly_tensor::prelude::*;
use woolly_tensor::ops::*;

let backend = CpuBackend::new();

// Matrix multiplication
let a = Tensor::randn(&backend, Shape::matrix(64, 128), DType::F32)?;
let b = Tensor::randn(&backend, Shape::matrix(128, 256), DType::F32)?;
let c = MatMul::apply(&a, &b)?;

// Batch matrix multiplication
let a_batch = Tensor::randn(&backend, Shape::from_slice(&[32, 64, 128]), DType::F32)?;
let b_batch = Tensor::randn(&backend, Shape::from_slice(&[32, 128, 256]), DType::F32)?;
let c_batch = BatchMatMul::apply(&a_batch, &b_batch)?;

// Vector operations
let v1 = Tensor::randn(&backend, Shape::vector(1024), DType::F32)?;
let v2 = Tensor::randn(&backend, Shape::vector(1024), DType::F32)?;
let dot_product = Dot::apply(&v1, &v2)?;
```

## Quantization

### Supported Quantization Schemes

```rust
use woolly_tensor::prelude::*;

let backend = CpuBackend::new();
let tensor = Tensor::randn(&backend, Shape::matrix(1024, 1024), DType::F32)?;

// Different quantization methods
let q4_0_quantizer = Q4_0Quantizer::new();
let q4_1_quantizer = Q4_1Quantizer::new();
let q8_0_quantizer = Q8_0Quantizer::new();
let int8_quantizer = Int8Quantizer::new();

// Quantize tensor
let quantized_q4_0 = q4_0_quantizer.quantize(&tensor)?;
let quantized_q8_0 = q8_0_quantizer.quantize(&tensor)?;

// Dequantize back to float
let dequantized = q4_0_quantizer.dequantize(&quantized_q4_0)?;

println!("Original size: {} bytes", tensor.size_in_bytes());
println!("Q4_0 size: {} bytes", quantized_q4_0.size_in_bytes());
println!("Compression ratio: {:.2}x", 
         tensor.size_in_bytes() as f32 / quantized_q4_0.size_in_bytes() as f32);
```

### Custom Quantization

```rust
use woolly_tensor::prelude::*;

struct CustomQuantizer {
    scale: f32,
    zero_point: i32,
}

impl Quantizer for CustomQuantizer {
    type QuantizedType = i8;
    
    fn quantize(&self, tensor: &Tensor) -> Result<Tensor, TensorError> {
        // Implement custom quantization logic
        todo!()
    }
    
    fn dequantize(&self, quantized: &Tensor) -> Result<Tensor, TensorError> {
        // Implement custom dequantization logic
        todo!()
    }
}
```

## Backend Support

### CPU Backend

```rust
use woolly_tensor::prelude::*;

// Create CPU backend with automatic SIMD detection
let backend = CpuBackend::new();

// Check available SIMD features
println!("AVX2 support: {}", backend.has_avx2());
println!("AVX-512 support: {}", backend.has_avx512());
println!("NEON support: {}", backend.has_neon());

// Configure thread pool
let backend = CpuBackend::with_threads(8);
```

### CUDA Backend

```rust
#[cfg(feature = "cuda")]
use woolly_tensor::prelude::*;

#[cfg(feature = "cuda")]
{
    // Create CUDA backend
    let backend = CudaBackend::new(0)?; // GPU device 0
    
    // Check device properties
    println!("Device name: {}", backend.device_name());
    println!("Memory: {} MB", backend.total_memory() / 1024 / 1024);
    println!("Compute capability: {:?}", backend.compute_capability());
    
    // Create tensors on GPU
    let tensor = Tensor::randn(&backend, Shape::matrix(1024, 1024), DType::F32)?;
    
    // Operations run on GPU
    let result = tensor.relu()?;
    
    // Copy to CPU if needed
    let cpu_backend = CpuBackend::new();
    let cpu_tensor = result.to_backend(&cpu_backend)?;
}
```

### Metal Backend (Apple Silicon)

```rust
#[cfg(feature = "metal")]
use woolly_tensor::prelude::*;

#[cfg(feature = "metal")]
{
    // Create Metal backend
    let backend = MetalBackend::new()?;
    
    // Check device properties
    println!("Device name: {}", backend.device_name());
    println!("Unified memory: {}", backend.has_unified_memory());
    
    // Operations use Metal Performance Shaders
    let tensor = Tensor::randn(&backend, Shape::matrix(2048, 2048), DType::F32)?;
    let result = tensor.matmul(&tensor.transpose(&[1, 0])?)?;
}
```

## Performance Optimization

### Memory Layout

```rust
use woolly_tensor::prelude::*;

let backend = CpuBackend::new();

// Row-major (C-style) layout - default
let row_major = Tensor::zeros(&backend, Shape::matrix(1000, 1000), DType::F32)?;

// Column-major (Fortran-style) layout
let col_major = row_major.transpose(&[1, 0])?;

// Contiguous memory access patterns are faster
let contiguous = col_major.contiguous()?;
```

### SIMD Optimization

```rust
use woolly_tensor::prelude::*;
use woolly_tensor::ops::simd::*;

// Direct SIMD operations for maximum performance
let a = vec![1.0f32; 1024];
let b = vec![2.0f32; 1024];
let mut result = vec![0.0f32; 1024];

// Use SIMD directly
#[cfg(target_arch = "x86_64")]
{
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            avx2_add_f32(&a, &b, &mut result);
        }
    }
}

#[cfg(target_arch = "aarch64")]
{
    unsafe {
        neon_add_f32(&a, &b, &mut result);
    }
}
```

### Batching and Threading

```rust
use woolly_tensor::prelude::*;

let backend = CpuBackend::with_threads(8);

// Process data in batches for better cache utilization
let batch_size = 64;
let total_samples = 10000;

for batch_start in (0..total_samples).step_by(batch_size) {
    let batch_end = (batch_start + batch_size).min(total_samples);
    let batch_data = &data[batch_start..batch_end];
    
    let batch_tensor = Tensor::from_slice(&backend, batch_data, 
                                         Shape::matrix(batch_end - batch_start, feature_size), 
                                         DType::F32)?;
    
    // Process batch
    let result = model.forward(&batch_tensor)?;
}
```

## Advanced Features

### Custom Operations

```rust
use woolly_tensor::prelude::*;
use woolly_tensor::ops::*;

struct CustomSwish;

impl UnaryOp for CustomSwish {
    fn apply_f32(input: &[f32], output: &mut [f32]) -> Result<(), TensorError> {
        for (i, &x) in input.iter().enumerate() {
            output[i] = x / (1.0 + (-x).exp());
        }
        Ok(())
    }
    
    fn apply_f16(input: &[f16], output: &mut [f16]) -> Result<(), TensorError> {
        // F16 implementation
        todo!()
    }
}

// Use custom operation
let tensor = Tensor::randn(&backend, Shape::vector(1000), DType::F32)?;
let result = CustomSwish::apply(&tensor)?;
```

### Memory Management

```rust
use woolly_tensor::prelude::*;

let backend = CpuBackend::new();

// Pre-allocate storage for better performance
let storage = TensorStorage::allocate(&backend, 1024 * 1024, DType::F32)?;
let tensor = Tensor::from_storage(storage, Shape::matrix(1024, 1024))?;

// Memory-mapped tensors for large datasets
let mmap_tensor = Tensor::from_file(&backend, "large_weights.bin", 
                                   Shape::from_slice(&[10000, 4096]), 
                                   DType::F32)?;

// Shared memory between tensors (zero-copy views)
let view = tensor.view(&[0..512, 0..512])?;
let slice = tensor.slice(&[100..200, 50..150])?;
```

## Error Handling

```rust
use woolly_tensor::prelude::*;

match tensor_operation() {
    Ok(result) => println!("Success: {:?}", result.shape()),
    Err(TensorError::ShapeMismatch { expected, actual }) => {
        eprintln!("Shape mismatch: expected {:?}, got {:?}", expected, actual);
    }
    Err(TensorError::InvalidDType { expected, actual }) => {
        eprintln!("Type mismatch: expected {:?}, got {:?}", expected, actual);
    }
    Err(TensorError::OutOfMemory { requested, available }) => {
        eprintln!("Out of memory: requested {} bytes, {} available", requested, available);
    }
    Err(TensorError::BackendError(msg)) => {
        eprintln!("Backend error: {}", msg);
    }
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

## Benchmarking

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark category
cargo bench tensor_ops
cargo bench matmul
cargo bench quantization

# Compare backends
cargo bench --features="cuda,metal"
```

Example benchmark results:
```
Matrix Multiplication (1024x1024):
CPU (AVX2):     45.2 ms
CUDA (RTX 4090): 2.3 ms
Metal (M2 Max):  3.1 ms

Element-wise Add (1M elements):
CPU (SIMD):      0.8 ms
CPU (scalar):    3.2 ms
CUDA:           0.1 ms
```

## Examples

- **[Basic Operations](examples/basic_ops.rs)**: Fundamental tensor operations
- **[Matrix Multiplication](examples/matmul.rs)**: Optimized matrix operations
- **[Quantization](examples/quantization.rs)**: Model quantization techniques
- **[Custom Backend](examples/custom_backend.rs)**: Implementing custom backends

## Features

- `cuda`: Enable NVIDIA CUDA support
- `metal`: Enable Apple Metal support
- `mkl`: Intel Math Kernel Library integration
- `blas`: Generic BLAS integration
- `benchmarks`: Include benchmarking utilities

## Contributing

We welcome contributions! Please see the [Contributing Guide](../../CONTRIBUTING.md) for details.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../../LICENSE-MIT))

at your option.