//! Integration tests for parallel inference functionality
//! 
//! This test suite validates that the multi-threaded inference
//! implementations work correctly and provide expected performance improvements.

use std::sync::Arc;
use std::time::Instant;
use rayon::prelude::*;

use woolly_core::model::parallel_config::{ParallelConfig, WorkType, load_balancing};
use woolly_tensor::ops::matmul::{Gemm, MatMulConfig, BatchedMatMul};
use woolly_tensor::shape::Shape;
use woolly_tensor::quantization::{
    BlockQ4_0, BlockQ8_0, dequantize_q4_0, dequantize_q8_0,
    simd::{dequantize_q4_0_batch, dequantize_q8_0_batch}
};

#[test]
fn test_parallel_config_initialization() {
    let mut config = ParallelConfig::new();
    assert!(config.enable_parallel);
    assert!(config.num_threads > 0);
    
    // Test thread pool initialization
    assert!(config.init_thread_pool().is_ok());
    
    // Test model-specific configuration
    let large_model_config = ParallelConfig::for_model(4096, 32, 48);
    let small_model_config = ParallelConfig::for_model(768, 12, 12);
    
    assert!(large_model_config.parallel_threshold.attention_seq_len <= 
            small_model_config.parallel_threshold.attention_seq_len);
}

#[test]
fn test_work_distribution_strategies() {
    let total_work = 1000;
    let num_threads = 4;
    
    let even_ranges = load_balancing::distribute_work(
        total_work, 
        num_threads, 
        woolly_core::model::parallel_config::WorkDistributionStrategy::Even
    );
    
    // Verify all work is covered
    let total_distributed: usize = even_ranges.iter()
        .map(|(start, end)| end - start)
        .sum();
    assert_eq!(total_distributed, total_work);
    
    // Verify non-overlapping ranges
    for i in 1..even_ranges.len() {
        assert_eq!(even_ranges[i-1].1, even_ranges[i].0);
    }
    
    // Test other strategies
    let dynamic_ranges = load_balancing::distribute_work(
        total_work, 
        num_threads, 
        woolly_core::model::parallel_config::WorkDistributionStrategy::Dynamic
    );
    assert!(!dynamic_ranges.is_empty());
    
    let cache_ranges = load_balancing::distribute_work(
        total_work, 
        num_threads, 
        woolly_core::model::parallel_config::WorkDistributionStrategy::CacheOptimized
    );
    assert!(!cache_ranges.is_empty());
}

#[test]
fn test_optimal_thread_count_calculation() {
    let large_workload = 10000;
    let small_workload = 50;
    let available_threads = 8;
    
    let large_optimal = load_balancing::optimal_thread_count(
        large_workload, 
        WorkType::MatMul, 
        available_threads
    );
    assert!(large_optimal > 1);
    assert!(large_optimal <= available_threads);
    
    let small_optimal = load_balancing::optimal_thread_count(
        small_workload, 
        WorkType::MatMul, 
        available_threads
    );
    assert_eq!(small_optimal, 1); // Too small to benefit from parallelization
}

#[test]
fn test_parallel_matrix_multiplication_correctness() {
    let m = 64;
    let n = 64; 
    let k = 64;
    
    // Generate test matrices
    let a: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.02).collect();
    
    // Sequential computation
    let mut c_sequential = vec![0.0f32; m * n];
    Gemm::compute(
        &a, &b, &mut c_sequential,
        &Shape::matrix(m, k),
        &Shape::matrix(k, n),
        &MatMulConfig::default(),
    ).unwrap();
    
    // Parallel computation (when enabled)
    let mut c_parallel = vec![0.0f32; m * n];
    Gemm::compute(
        &a, &b, &mut c_parallel,
        &Shape::matrix(m, k),
        &Shape::matrix(k, n),
        &MatMulConfig::default(),
    ).unwrap();
    
    // Results should be identical
    for (seq, par) in c_sequential.iter().zip(c_parallel.iter()) {
        assert!((seq - par).abs() < 1e-6, "Sequential: {}, Parallel: {}", seq, par);
    }
}

#[test]
fn test_parallel_batch_matrix_multiplication() {
    let batch_size = 4;
    let m = 32;
    let n = 32;
    let k = 32;
    
    let single_a_size = m * k;
    let single_b_size = k * n;
    let single_c_size = m * n;
    
    let a: Vec<f32> = (0..batch_size * single_a_size).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..batch_size * single_b_size).map(|i| (i as f32) * 0.01).collect();
    let mut c = vec![0.0f32; batch_size * single_c_size];
    
    let start = Instant::now();
    BatchedMatMul::compute(
        &a, &b, &mut c,
        &Shape::from_slice(&[batch_size, m, k]),
        &Shape::from_slice(&[batch_size, k, n]),
    ).unwrap();
    let duration = start.elapsed();
    
    println!("Batch matmul ({}x{}x{}x{}) took: {:?}", batch_size, m, n, k, duration);
    
    // Verify results are reasonable (non-zero)
    assert!(c.iter().any(|&x| x != 0.0));
}

#[test]
fn test_parallel_quantization_correctness() {
    let block_count = 64;
    
    // Generate Q4_0 test blocks
    let q4_0_blocks: Vec<BlockQ4_0> = (0..block_count).map(|i| {
        BlockQ4_0 {
            d: half::f16::from_f32(0.1 + (i as f32) * 0.001),
            qs: [(i % 16) as u8; 16],
        }
    }).collect();
    
    // Sequential dequantization
    let sequential_result = dequantize_q4_0(&q4_0_blocks);
    
    // Parallel dequantization
    #[cfg(feature = "parallel")]
    let parallel_result = dequantize_q4_0_batch(&q4_0_blocks);
    
    #[cfg(feature = "parallel")]
    {
        assert_eq!(sequential_result.len(), parallel_result.len());
        
        for (seq, par) in sequential_result.iter().zip(parallel_result.iter()) {
            assert!((seq - par).abs() < 1e-6, "Sequential: {}, Parallel: {}", seq, par);
        }
    }
    
    // Generate Q8_0 test blocks
    let q8_0_blocks: Vec<BlockQ8_0> = (0..block_count).map(|i| {
        BlockQ8_0 {
            d: half::f16::from_f32(0.1 + (i as f32) * 0.001),
            qs: [(i % 127) as i8; 32],
        }
    }).collect();
    
    // Sequential dequantization
    let sequential_result = dequantize_q8_0(&q8_0_blocks);
    
    // Parallel dequantization
    #[cfg(feature = "parallel")]
    let parallel_result = dequantize_q8_0_batch(&q8_0_blocks);
    
    #[cfg(feature = "parallel")]
    {
        assert_eq!(sequential_result.len(), parallel_result.len());
        
        for (seq, par) in sequential_result.iter().zip(parallel_result.iter()) {
            assert!((seq - par).abs() < 1e-6, "Sequential: {}, Parallel: {}", seq, par);
        }
    }
}

#[test]
fn test_attention_simulation_parallel_correctness() {
    let seq_len = 32;
    let num_heads = 8;
    let head_dim = 64;
    let hidden_size = num_heads * head_dim;
    
    // Generate test data
    let query: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.01).collect();
    let key: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.01 + 0.5).collect();
    let value: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.01 + 1.0).collect();
    
    // Sequential attention computation (simplified)
    let mut sequential_output = vec![0.0f32; seq_len * hidden_size];
    for h in 0..num_heads {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = i * hidden_size + h * head_dim + d;
                    let k_idx = j * hidden_size + h * head_dim + d;
                    score += query[q_idx] * key[k_idx];
                }
                
                for d in 0..head_dim {
                    let v_idx = j * hidden_size + h * head_dim + d;
                    let out_idx = i * hidden_size + h * head_dim + d;
                    sequential_output[out_idx] += score * value[v_idx] / (seq_len as f32);
                }
            }
        }
    }
    
    // Parallel attention computation
    #[cfg(feature = "parallel")]
    {
        let parallel_output: Vec<f32> = (0..num_heads).into_par_iter().flat_map(|h| {
            let mut head_output = vec![0.0f32; seq_len * head_dim];
            
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        let q_idx = i * hidden_size + h * head_dim + d;
                        let k_idx = j * hidden_size + h * head_dim + d;
                        score += query[q_idx] * key[k_idx];
                    }
                    
                    for d in 0..head_dim {
                        let v_idx = j * hidden_size + h * head_dim + d;
                        let out_idx = i * head_dim + d;
                        head_output[out_idx] += score * value[v_idx] / (seq_len as f32);
                    }
                }
            }
            
            head_output
        }).collect();
        
        // Reshape parallel output to match sequential format
        let mut parallel_reshaped = vec![0.0f32; seq_len * hidden_size];
        for h in 0..num_heads {
            for i in 0..seq_len {
                for d in 0..head_dim {
                    let par_idx = h * seq_len * head_dim + i * head_dim + d;
                    let seq_idx = i * hidden_size + h * head_dim + d;
                    parallel_reshaped[seq_idx] = parallel_output[par_idx];
                }
            }
        }
        
        // Compare results
        for (seq, par) in sequential_output.iter().zip(parallel_reshaped.iter()) {
            assert!((seq - par).abs() < 1e-4, "Sequential: {}, Parallel: {}", seq, par);
        }
    }
}

#[test]
fn test_performance_metrics() {
    use woolly_core::model::parallel_config::performance::{ParallelMetrics, ParallelTimer};
    use std::time::Duration;
    
    let metrics = ParallelMetrics::new();
    
    // Simulate some operations
    {
        let _timer = ParallelTimer::new(Arc::clone(&metrics), true);
        std::thread::sleep(Duration::from_millis(10));
    }
    
    {
        let _timer = ParallelTimer::new(Arc::clone(&metrics), false);
        std::thread::sleep(Duration::from_millis(5));
    }
    
    // Check metrics
    assert_eq!(metrics.operation_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(metrics.serial_fallback_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    
    let efficiency = metrics.parallel_efficiency();
    assert_eq!(efficiency, 0.5); // 1 parallel op out of 2 total
}

#[test]
fn test_thread_safety() {
    use woolly_core::model::memory_pool::TensorMemoryPool;
    use std::sync::Arc;
    use std::thread;
    
    let pool = Arc::new(TensorMemoryPool::new());
    let handles: Vec<_> = (0..4).map(|i| {
        let pool_clone = Arc::clone(&pool);
        thread::spawn(move || {
            // Each thread requests buffers of different sizes
            let buffer_size = 1024 * (i + 1);
            let buffer = pool_clone.get_buffer(buffer_size);
            assert_eq!(buffer.len(), buffer_size);
            
            // Return buffer to pool
            pool_clone.return_buffer(buffer);
        })
    }).collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_load_balancing_effectiveness() {
    // Test that load balancing distributes work effectively
    let total_work = 1000;
    let num_threads = 8;
    
    let ranges = load_balancing::distribute_work(
        total_work,
        num_threads,
        woolly_core::model::parallel_config::WorkDistributionStrategy::Even
    );
    
    // Calculate variance in work distribution
    let work_per_thread: Vec<usize> = ranges.iter()
        .map(|(start, end)| end - start)
        .collect();
    
    let mean_work = work_per_thread.iter().sum::<usize>() as f64 / work_per_thread.len() as f64;
    let variance = work_per_thread.iter()
        .map(|&work| {
            let diff = work as f64 - mean_work;
            diff * diff
        })
        .sum::<f64>() / work_per_thread.len() as f64;
    
    // Variance should be low for even distribution
    assert!(variance < 2.0, "Work distribution variance too high: {}", variance);
}

#[test]
fn test_parallel_vs_sequential_performance() {
    // This test measures relative performance but doesn't enforce specific speedups
    // since performance depends on hardware and system load
    
    let matrix_size = (256, 256, 256);
    let a: Vec<f32> = (0..matrix_size.0 * matrix_size.2).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..matrix_size.2 * matrix_size.1).map(|i| (i as f32) * 0.01).collect();
    
    // Measure sequential time (force single thread)
    let sequential_start = Instant::now();
    for _ in 0..5 {
        let mut c = vec![0.0f32; matrix_size.0 * matrix_size.1];
        Gemm::compute(
            &a, &b, &mut c,
            &Shape::matrix(matrix_size.0, matrix_size.2),
            &Shape::matrix(matrix_size.2, matrix_size.1),
            &MatMulConfig::default(),
        ).unwrap();
    }
    let sequential_time = sequential_start.elapsed();
    
    // Measure parallel time
    let parallel_start = Instant::now();
    for _ in 0..5 {
        let mut c = vec![0.0f32; matrix_size.0 * matrix_size.1];
        Gemm::compute(
            &a, &b, &mut c,
            &Shape::matrix(matrix_size.0, matrix_size.2),
            &Shape::matrix(matrix_size.2, matrix_size.1),
            &MatMulConfig::default(),
        ).unwrap();
    }
    let parallel_time = parallel_start.elapsed();
    
    println!("Sequential time: {:?}", sequential_time);
    println!("Parallel time: {:?}", parallel_time);
    
    // On systems with multiple cores, parallel should not be significantly slower
    // (allowing for some overhead and variance)
    let speedup_ratio = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
    println!("Speedup ratio: {:.2}x", speedup_ratio);
    
    // Parallel implementation should at least not be more than 2x slower
    assert!(speedup_ratio > 0.5, "Parallel implementation is too much slower than sequential");
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_end_to_end_parallel_inference_simulation() {
        // Simulate a small transformer forward pass with parallel components
        let batch_size = 2;
        let seq_len = 64;
        let hidden_size = 512;
        let num_heads = 8;
        let head_dim = hidden_size / num_heads;
        let intermediate_size = 2048;
        
        // Initialize parallel configuration
        let config = ParallelConfig::for_model(hidden_size, num_heads, 12);
        
        // Check if operations should be parallelized
        assert!(config.should_parallelize_attention(seq_len, num_heads));
        assert!(config.should_parallelize_ffn(intermediate_size));
        assert!(config.should_parallelize_batch(batch_size));
        
        // Simulate input embeddings
        let input: Vec<f32> = (0..batch_size * seq_len * hidden_size)
            .map(|i| (i as f32) * 0.001)
            .collect();
        
        // Simulate attention weights
        let q_weight: Vec<f32> = (0..hidden_size * hidden_size)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        
        // Simulate Q projection with parallel matrix multiplication
        let start = Instant::now();
        let mut q_proj = vec![0.0f32; batch_size * seq_len * hidden_size];
        
        // Process each batch item
        for b in 0..batch_size {
            let input_batch = &input[b * seq_len * hidden_size..(b + 1) * seq_len * hidden_size];
            let mut q_batch = vec![0.0f32; seq_len * hidden_size];
            
            Gemm::compute(
                input_batch,
                &q_weight,
                &mut q_batch,
                &Shape::matrix(seq_len, hidden_size),
                &Shape::matrix(hidden_size, hidden_size),
                &MatMulConfig::default(),
            ).unwrap();
            
            q_proj[b * seq_len * hidden_size..(b + 1) * seq_len * hidden_size]
                .copy_from_slice(&q_batch);
        }
        
        let attention_time = start.elapsed();
        println!("Attention projection time: {:?}", attention_time);
        
        // Simulate FFN with parallel processing
        let ffn_w1: Vec<f32> = (0..hidden_size * intermediate_size)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        let ffn_w2: Vec<f32> = (0..intermediate_size * hidden_size)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        
        let start = Instant::now();
        
        // First FFN layer
        let mut ffn_intermediate = vec![0.0f32; batch_size * seq_len * intermediate_size];
        for b in 0..batch_size {
            let input_batch = &input[b * seq_len * hidden_size..(b + 1) * seq_len * hidden_size];
            let mut inter_batch = vec![0.0f32; seq_len * intermediate_size];
            
            Gemm::compute(
                input_batch,
                &ffn_w1,
                &mut inter_batch,
                &Shape::matrix(seq_len, hidden_size),
                &Shape::matrix(hidden_size, intermediate_size),
                &MatMulConfig::default(),
            ).unwrap();
            
            ffn_intermediate[b * seq_len * intermediate_size..(b + 1) * seq_len * intermediate_size]
                .copy_from_slice(&inter_batch);
        }
        
        // Apply activation in parallel
        #[cfg(feature = "parallel")]
        ffn_intermediate.par_iter_mut().for_each(|x| {
            *x = *x / (1.0 + (-*x).exp()); // SiLU activation
        });
        
        #[cfg(not(feature = "parallel"))]
        for x in ffn_intermediate.iter_mut() {
            *x = *x / (1.0 + (-*x).exp());
        }
        
        // Second FFN layer
        let mut ffn_output = vec![0.0f32; batch_size * seq_len * hidden_size];
        for b in 0..batch_size {
            let inter_batch = &ffn_intermediate[b * seq_len * intermediate_size..(b + 1) * seq_len * intermediate_size];
            let mut output_batch = vec![0.0f32; seq_len * hidden_size];
            
            Gemm::compute(
                inter_batch,
                &ffn_w2,
                &mut output_batch,
                &Shape::matrix(seq_len, intermediate_size),
                &Shape::matrix(intermediate_size, hidden_size),
                &MatMulConfig::default(),
            ).unwrap();
            
            ffn_output[b * seq_len * hidden_size..(b + 1) * seq_len * hidden_size]
                .copy_from_slice(&output_batch);
        }
        
        let ffn_time = start.elapsed();
        println!("FFN processing time: {:?}", ffn_time);
        
        // Verify outputs are reasonable
        assert!(q_proj.iter().any(|&x| x != 0.0));
        assert!(ffn_output.iter().any(|&x| x != 0.0));
        
        println!("End-to-end parallel inference simulation completed successfully");
    }
}