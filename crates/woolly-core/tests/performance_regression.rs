//! Performance Regression Tests
//!
//! These tests ensure that performance optimizations are maintained
//! and alert developers if performance regresses below acceptable thresholds.

use woolly_core::{
    engine::InferenceEngine,
    generation::GenerationConfig,
    model::optimized_transformer::OptimizedTransformer,
    tensor_utils_simd::simd_matvec,
};
use woolly_gguf::loader::GgufLoader;
use woolly_tensor::Shape;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Performance thresholds (tokens/sec)
const MIN_SINGLE_TOKEN_THROUGHPUT: f64 = 0.5;  // Minimum acceptable
const TARGET_SINGLE_TOKEN_THROUGHPUT: f64 = 1.0;  // Target performance
const MIN_BATCH_THROUGHPUT: f64 = 2.0;  // For batch size 4
const MIN_SIMD_SPEEDUP: f64 = 3.0;  // Minimum SIMD vs scalar speedup

/// Helper to measure operation throughput
fn measure_throughput<F>(operation: F, iterations: usize) -> (Duration, f64)
where
    F: Fn(),
{
    // Warmup
    for _ in 0..3 {
        operation();
    }
    
    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        operation();
    }
    let duration = start.elapsed();
    let throughput = iterations as f64 / duration.as_secs_f64();
    
    (duration, throughput)
}

#[test]
#[ignore] // Run with --ignored flag for performance tests
fn test_single_token_generation_performance() {
    let model_path = "models/granite-3.3-8b-instruct-Q4_K_M.gguf";
    
    // Load model
    let loader = GgufLoader::new();
    let model_data = loader.load_model(model_path).expect("Failed to load model");
    let model = Arc::new(OptimizedTransformer::from_gguf(model_data).expect("Failed to create model"));
    let engine = InferenceEngine::new(model);
    
    // Generate tokens
    let config = GenerationConfig {
        max_tokens: 10,
        temperature: 1.0,
        top_p: 0.9,
        ..Default::default()
    };
    
    let start = Instant::now();
    let result = engine.generate("Test prompt", &config).expect("Generation failed");
    let duration = start.elapsed();
    
    let tokens_per_sec = result.tokens_generated as f64 / duration.as_secs_f64();
    
    println!("Single token generation: {:.3} tokens/sec", tokens_per_sec);
    
    assert!(
        tokens_per_sec >= MIN_SINGLE_TOKEN_THROUGHPUT,
        "Performance regression: {:.3} tokens/sec is below minimum threshold of {:.3}",
        tokens_per_sec,
        MIN_SINGLE_TOKEN_THROUGHPUT
    );
    
    if tokens_per_sec < TARGET_SINGLE_TOKEN_THROUGHPUT {
        println!(
            "Warning: Performance {:.3} tokens/sec is below target of {:.3}",
            tokens_per_sec,
            TARGET_SINGLE_TOKEN_THROUGHPUT
        );
    }
}

#[test]
fn test_simd_matvec_performance() {
    let sizes = vec![256, 512, 1024, 2048];
    
    for size in sizes {
        let matrix = vec![0.1f32; size * size];
        let vector = vec![0.2f32; size];
        let shape = Shape::matrix(size, size);
        
        // Create SimpleTensor for SIMD operations
        use woolly_core::tensor_utils::SimpleTensor;
        let matrix_tensor = SimpleTensor::new(matrix.clone(), shape.clone()).unwrap();
        let vector_tensor = SimpleTensor::new(vector.clone(), Shape::vector(size)).unwrap();
        
        // Measure SIMD performance
        let (simd_duration, _) = measure_throughput(
            || {
                let _ = simd_matvec(&matrix_tensor, &vector_tensor, false, 1.0, 0.0).unwrap();
            },
            100,
        );
        
        // Measure scalar performance
        std::env::set_var("WOOLLY_DISABLE_SIMD", "1");
        let (scalar_duration, _) = measure_throughput(
            || {
                let _ = simd_matvec(&matrix_tensor, &vector_tensor, false, 1.0, 0.0).unwrap();
            },
            100,
        );
        std::env::remove_var("WOOLLY_DISABLE_SIMD");
        
        let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();
        
        println!(
            "Matrix size {}x{}: SIMD speedup = {:.2}x",
            size, size, speedup
        );
        
        assert!(
            speedup >= MIN_SIMD_SPEEDUP,
            "SIMD performance regression for {}x{}: speedup {:.2}x is below minimum {:.2}x",
            size,
            size,
            speedup,
            MIN_SIMD_SPEEDUP
        );
    }
}

#[test]
fn test_memory_allocation_performance() {
    use woolly_core::model::memory_pool_enhanced::EnhancedTensorMemoryPool;
    
    let sizes = vec![1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024];
    let iterations = 1000;
    
    for size in sizes {
        // Measure raw allocation
        let (raw_duration, _) = measure_throughput(
            || {
                let _v: Vec<f32> = vec![0.0; size];
            },
            iterations,
        );
        
        // Measure pooled allocation
        let mut pool = EnhancedTensorMemoryPool::new();
        let (pool_duration, _) = measure_throughput(
            || {
                let buffer = pool.get_simd_buffer(size);
                pool.return_buffer(buffer);
            },
            iterations,
        );
        
        let speedup = raw_duration.as_secs_f64() / pool_duration.as_secs_f64();
        
        println!(
            "Allocation size {}: Pool speedup = {:.2}x",
            size, speedup
        );
        
        assert!(
            speedup >= 2.0,
            "Memory pool performance regression for size {}: speedup {:.2}x is too low",
            size,
            speedup
        );
    }
}

#[test]
fn test_cache_hit_performance() {
    use woolly_core::model::dequantization_cache::DequantizationCache;
    
    let mut cache = DequantizationCache::new(100 * 1024 * 1024); // 100MB
    let tensor_size = 4096 * 768;
    let tensor_data = vec![0.1f32; tensor_size];
    
    // Populate cache
    for i in 0..100 {
        let key = format!("tensor_{}", i);
        cache.insert(key, tensor_data.clone());
    }
    
    // Measure cache hits
    let mut hits = 0;
    let iterations = 10000;
    
    let (duration, _) = measure_throughput(
        || {
            let key = format!("tensor_{}", hits % 100);
            if cache.get(&key).is_some() {
                hits += 1;
            }
        },
        iterations,
    );
    
    let hit_rate = (hits as f64 / iterations as f64) * 100.0;
    let ops_per_sec = iterations as f64 / duration.as_secs_f64();
    
    println!("Cache hit rate: {:.1}%", hit_rate);
    println!("Cache operations/sec: {:.0}", ops_per_sec);
    
    assert!(
        hit_rate >= 95.0,
        "Cache hit rate {:.1}% is below expected 95%",
        hit_rate
    );
    
    assert!(
        ops_per_sec >= 1_000_000.0,
        "Cache performance {:.0} ops/sec is below expected 1M ops/sec",
        ops_per_sec
    );
}

#[test]
#[ignore] // Run with --ignored flag
fn test_batch_inference_performance() {
    let model_path = "models/granite-3.3-8b-instruct-Q4_K_M.gguf";
    
    // Load model
    let loader = GgufLoader::new();
    let model_data = loader.load_model(model_path).expect("Failed to load model");
    let model = Arc::new(OptimizedTransformer::from_gguf(model_data).expect("Failed to create model"));
    let engine = InferenceEngine::new(model);
    
    // Test batch sizes
    let batch_sizes = vec![1, 2, 4, 8];
    let prompts = vec![
        "What is machine learning?",
        "Explain quantum computing.",
        "How does the internet work?",
        "What is artificial intelligence?",
        "Describe cloud computing.",
        "What is blockchain?",
        "Explain neural networks.",
        "How do computers work?",
    ];
    
    for batch_size in batch_sizes {
        let batch_prompts = &prompts[..batch_size];
        let config = GenerationConfig {
            max_tokens: 20,
            temperature: 1.0,
            top_p: 0.9,
            ..Default::default()
        };
        
        let start = Instant::now();
        let mut total_tokens = 0;
        
        for prompt in batch_prompts {
            let result = engine.generate(prompt, &config).expect("Generation failed");
            total_tokens += result.tokens_generated;
        }
        
        let duration = start.elapsed();
        let tokens_per_sec = total_tokens as f64 / duration.as_secs_f64();
        
        println!(
            "Batch size {}: {:.3} tokens/sec",
            batch_size, tokens_per_sec
        );
        
        if batch_size == 4 {
            assert!(
                tokens_per_sec >= MIN_BATCH_THROUGHPUT,
                "Batch performance regression: {:.3} tokens/sec is below minimum {:.3}",
                tokens_per_sec,
                MIN_BATCH_THROUGHPUT
            );
        }
    }
}

#[test]
fn test_simd_operation_coverage() {
    // Test that all critical operations use SIMD
    let test_size = 1024;
    let a = vec![1.0f32; test_size];
    let b = vec![2.0f32; test_size];
    
    // These operations should all use SIMD internally
    let operations = vec![
        ("add", || { SimdOps::add(&a, &b).unwrap(); }),
        ("mul", || { SimdOps::mul(&a, &b).unwrap(); }),
        ("sum", || { SimdOps::sum(&a); }),
        ("softmax", || { SimdOps::softmax(&a).unwrap(); }),
    ];
    
    for (name, op) in operations {
        // Measure with SIMD
        let (simd_duration, _) = measure_throughput(op, 1000);
        
        // Measure without SIMD
        std::env::set_var("WOOLLY_DISABLE_SIMD", "1");
        let (scalar_duration, _) = measure_throughput(op, 1000);
        std::env::remove_var("WOOLLY_DISABLE_SIMD");
        
        let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();
        
        println!("Operation '{}': SIMD speedup = {:.2}x", name, speedup);
        
        assert!(
            speedup >= 2.0,
            "SIMD not effective for operation '{}': speedup only {:.2}x",
            name,
            speedup
        );
    }
}

/// Performance benchmark entry point for CI/CD
#[test]
#[ignore]
fn benchmark_performance_summary() {
    println!("\n=== Woolly Performance Regression Test Summary ===\n");
    
    // Run all performance tests and collect results
    let tests = vec![
        ("Single Token Generation", test_single_token_generation_performance as fn()),
        ("SIMD MatVec Operations", test_simd_matvec_performance as fn()),
        ("Memory Allocation", test_memory_allocation_performance as fn()),
        ("Cache Performance", test_cache_hit_performance as fn()),
        ("Batch Inference", test_batch_inference_performance as fn()),
        ("SIMD Coverage", test_simd_operation_coverage as fn()),
    ];
    
    let mut passed = 0;
    let mut failed = 0;
    
    for (name, test_fn) in tests {
        print!("Running {}... ", name);
        match std::panic::catch_unwind(test_fn) {
            Ok(_) => {
                println!("PASSED");
                passed += 1;
            }
            Err(_) => {
                println!("FAILED");
                failed += 1;
            }
        }
    }
    
    println!("\nResults: {} passed, {} failed", passed, failed);
    
    if failed > 0 {
        panic!("Performance regression detected!");
    }
}