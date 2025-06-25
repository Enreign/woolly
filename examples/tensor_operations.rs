//! Tensor Operations Tutorial
//! 
//! This example demonstrates the fundamental tensor operations in woolly-tensor,
//! including creation, manipulation, SIMD operations, and memory management.

use woolly_tensor::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Woolly Tensor Operations Tutorial ===\n");

    // Example 1: Basic tensor creation
    basic_tensor_creation()?;
    
    // Example 2: Tensor operations
    tensor_operations()?;
    
    // Example 3: SIMD-accelerated operations
    simd_operations()?;
    
    // Example 4: Memory pool usage
    memory_pool_demo()?;
    
    // Example 5: Backend selection
    backend_selection()?;
    
    // Example 6: Zero-copy views and slicing
    tensor_views_and_slicing()?;
    
    // Example 7: Advanced operations
    advanced_operations()?;

    Ok(())
}

/// Example 1: Basic tensor creation and inspection
fn basic_tensor_creation() -> Result<(), Box<dyn std::error::Error>> {
    println!("### Example 1: Basic Tensor Creation ###");
    
    // Create backend (auto-selects best available)
    let backend = Backend::auto()?;
    println!("Using backend: {:?}", backend.name());
    
    // Create tensors with different methods
    let zeros = Tensor::zeros(&backend, Shape::vector(10), DType::F32)?;
    println!("Zeros tensor: shape={:?}, dtype={:?}", zeros.shape(), zeros.dtype());
    
    let ones = Tensor::ones(&backend, Shape::matrix(3, 4), DType::F32)?;
    println!("Ones tensor: shape={:?}", ones.shape());
    
    // Create from data
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let from_data = Tensor::from_slice(&backend, &data, Shape::matrix(2, 3))?;
    println!("From data tensor: shape={:?}", from_data.shape());
    print_tensor(&from_data, "From data")?;
    
    // Create with specific values
    let filled = Tensor::full(&backend, Shape::vector(5), 3.14, DType::F32)?;
    print_tensor(&filled, "Filled with 3.14")?;
    
    println!();
    Ok(())
}

/// Example 2: Basic tensor operations
fn tensor_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("### Example 2: Tensor Operations ###");
    
    let backend = Backend::auto()?;
    
    // Create test tensors
    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], Shape::vector(4))?;
    let b = Tensor::from_slice(&backend, &[5.0f32, 6.0, 7.0, 8.0], Shape::vector(4))?;
    
    // Element-wise operations
    let sum = (&a + &b)?;
    print_tensor(&sum, "a + b")?;
    
    let product = (&a * &b)?;
    print_tensor(&product, "a * b")?;
    
    let scaled = (&a * 2.0)?;
    print_tensor(&scaled, "a * 2.0")?;
    
    // Reduction operations
    let sum_all = a.sum()?;
    println!("Sum of all elements in a: {}", sum_all.to_scalar::<f32>()?);
    
    let mean = a.mean()?;
    println!("Mean of a: {}", mean.to_scalar::<f32>()?);
    
    // Matrix operations
    let mat_a = Tensor::from_slice(&backend, 
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 
        Shape::matrix(2, 3))?;
    let mat_b = Tensor::from_slice(&backend, 
        &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], 
        Shape::matrix(3, 2))?;
    
    let matmul = mat_a.matmul(&mat_b)?;
    print_tensor(&matmul, "Matrix multiplication (2x3) @ (3x2)")?;
    
    println!();
    Ok(())
}

/// Example 3: SIMD-accelerated operations
fn simd_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("### Example 3: SIMD-Accelerated Operations ###");
    
    let backend = Backend::cpu_with_features(CpuFeatures::all_available())?;
    println!("CPU features: {:?}", backend.cpu_features());
    
    // Large tensors to show SIMD performance
    let size = 1_000_000;
    let a = Tensor::rand(&backend, Shape::vector(size), DType::F32)?;
    let b = Tensor::rand(&backend, Shape::vector(size), DType::F32)?;
    
    // Benchmark SIMD operations
    let start = Instant::now();
    let result = (&a + &b)?;
    let simd_time = start.elapsed();
    println!("SIMD addition of {} elements: {:?}", size, simd_time);
    
    // Dot product (uses SIMD)
    let start = Instant::now();
    let dot = a.dot(&b)?;
    let dot_time = start.elapsed();
    println!("SIMD dot product: {:?} (result: {})", dot_time, dot.to_scalar::<f32>()?);
    
    // ReLU activation (SIMD optimized)
    let start = Instant::now();
    let relu_result = a.relu()?;
    let relu_time = start.elapsed();
    println!("SIMD ReLU activation: {:?}", relu_time);
    
    println!();
    Ok(())
}

/// Example 4: Memory pool usage for efficiency
fn memory_pool_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("### Example 4: Memory Pool Usage ###");
    
    // Get global memory pool stats before
    let pool = MemoryPool::global();
    let stats_before = pool.stats();
    println!("Pool stats before: {:?}", stats_before);
    
    let backend = Backend::auto()?;
    
    // Allocate and deallocate tensors using the pool
    {
        let _tensors: Vec<Tensor> = (0..10)
            .map(|_| Tensor::zeros(&backend, Shape::matrix(1000, 1000), DType::F32))
            .collect::<Result<Vec<_>, _>>()?;
        
        let stats_during = pool.stats();
        println!("Pool stats during allocation: {:?}", stats_during);
    } // Tensors dropped here, memory returned to pool
    
    let stats_after = pool.stats();
    println!("Pool stats after deallocation: {:?}", stats_after);
    
    // Clear pool to free memory
    pool.clear();
    let stats_cleared = pool.stats();
    println!("Pool stats after clear: {:?}", stats_cleared);
    
    println!();
    Ok(())
}

/// Example 5: Backend selection and capabilities
fn backend_selection() -> Result<(), Box<dyn std::error::Error>> {
    println!("### Example 5: Backend Selection ###");
    
    // Check available backends
    println!("Available backends:");
    
    // CPU backend (always available)
    let cpu_backend = Backend::cpu()?;
    println!("  - CPU: {:?}", cpu_backend.info());
    
    // Try CUDA backend
    match Backend::cuda(0) {
        Ok(cuda) => println!("  - CUDA: {:?}", cuda.info()),
        Err(_) => println!("  - CUDA: Not available"),
    }
    
    // Try Metal backend (macOS)
    #[cfg(target_os = "macos")]
    match Backend::metal() {
        Ok(metal) => println!("  - Metal: {:?}", metal.info()),
        Err(_) => println!("  - Metal: Not available"),
    }
    
    // Auto-select best backend
    let auto = Backend::auto()?;
    println!("\nAuto-selected backend: {:?}", auto.name());
    
    // Backend-specific operations
    let tensor = Tensor::rand(&auto, Shape::matrix(100, 100), DType::F32)?;
    println!("Created 100x100 tensor on {}", auto.name());
    
    println!();
    Ok(())
}

/// Example 6: Zero-copy views and slicing
fn tensor_views_and_slicing() -> Result<(), Box<dyn std::error::Error>> {
    println!("### Example 6: Zero-Copy Views and Slicing ###");
    
    let backend = Backend::auto()?;
    
    // Create a tensor
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::from_slice(&backend, &data, Shape::new(vec![2, 3, 4]))?;
    print_tensor(&tensor, "Original tensor (2x3x4)")?;
    
    // Create views (no data copying)
    let view1 = tensor.view(Shape::matrix(6, 4))?;
    print_tensor(&view1, "Reshaped view (6x4)")?;
    
    // Slicing
    let slice = tensor.slice(&[0..1, 1..3, ..])?;
    print_tensor(&slice, "Slice [0:1, 1:3, :]")?;
    
    // Transpose (creates a view)
    let matrix = Tensor::from_slice(&backend, 
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 
        Shape::matrix(2, 3))?;
    let transposed = matrix.transpose()?;
    print_tensor(&matrix, "Original matrix")?;
    print_tensor(&transposed, "Transposed view")?;
    
    // Permute dimensions
    let permuted = tensor.permute(&[2, 0, 1])?;
    println!("Permuted shape (was {:?}, now {:?})", tensor.shape(), permuted.shape());
    
    println!();
    Ok(())
}

/// Example 7: Advanced operations
fn advanced_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("### Example 7: Advanced Operations ###");
    
    let backend = Backend::auto()?;
    
    // Batch matrix multiplication
    let batch_a = Tensor::rand(&backend, Shape::new(vec![4, 3, 5]), DType::F32)?;
    let batch_b = Tensor::rand(&backend, Shape::new(vec![4, 5, 2]), DType::F32)?;
    let batch_result = batch_a.batch_matmul(&batch_b)?;
    println!("Batch matmul: {:?} @ {:?} = {:?}", 
             batch_a.shape(), batch_b.shape(), batch_result.shape());
    
    // Broadcasting
    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0], Shape::vector(3))?;
    let b = Tensor::from_slice(&backend, 
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], 
        Shape::matrix(2, 3))?;
    let broadcasted = (&a + &b)?;  // Broadcasting happens automatically
    print_tensor(&broadcasted, "Broadcasting: vector + matrix")?;
    
    // Activation functions
    let x = Tensor::from_slice(&backend, 
        &[-2.0f32, -1.0, 0.0, 1.0, 2.0], 
        Shape::vector(5))?;
    
    let relu = x.relu()?;
    print_tensor(&relu, "ReLU activation")?;
    
    let sigmoid = x.sigmoid()?;
    print_tensor(&sigmoid, "Sigmoid activation")?;
    
    let tanh = x.tanh()?;
    print_tensor(&tanh, "Tanh activation")?;
    
    // Softmax
    let logits = Tensor::from_slice(&backend, 
        &[1.0f32, 2.0, 3.0, 4.0], 
        Shape::vector(4))?;
    let softmax = logits.softmax(-1)?;
    print_tensor(&softmax, "Softmax")?;
    
    println!();
    Ok(())
}

/// Helper function to print tensor contents
fn print_tensor(tensor: &Tensor, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let data = tensor.to_vec::<f32>()?;
    let shape = tensor.shape();
    
    print!("{}: [", name);
    
    if data.len() <= 10 {
        // Print all elements for small tensors
        for (i, &val) in data.iter().enumerate() {
            if i > 0 { print!(", "); }
            print!("{:.2}", val);
        }
    } else {
        // Print first and last few elements for large tensors
        for i in 0..3 {
            if i > 0 { print!(", "); }
            print!("{:.2}", data[i]);
        }
        print!(", ...");
        for i in (data.len()-3)..data.len() {
            print!(", {:.2}", data[i]);
        }
    }
    
    println!("] shape={:?}", shape);
    Ok(())
}