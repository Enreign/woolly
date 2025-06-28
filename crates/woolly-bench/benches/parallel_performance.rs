//! Benchmark for measuring multi-threaded inference performance improvements

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rayon::prelude::*;
use std::time::Duration;

// Import necessary types for benchmarking
use woolly_tensor::ops::matmul::{Gemm, MatMulConfig, BatchedMatMul, MatVec};
use woolly_tensor::shape::Shape;
use woolly_tensor::quantization::{
    BlockQ4_0, BlockQ8_0, dequantize_q4_0, dequantize_q8_0,
    simd::{dequantize_q4_0_batch, dequantize_q8_0_batch}
};

/// Benchmark parallel vs sequential matrix multiplication
fn bench_parallel_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    
    // Test different matrix sizes
    let sizes = vec![
        (128, 128, 128),
        (256, 256, 256), 
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ];
    
    for (m, n, k) in sizes {
        let size_label = format!("{}x{}x{}", m, n, k);
        group.throughput(Throughput::Elements((m * n * k) as u64));
        
        // Generate test data
        let a: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.01).collect();
        
        // Benchmark sequential implementation
        group.bench_with_input(
            BenchmarkId::new("sequential", &size_label),
            &(m, n, k),
            |bencher, &(_m, _n, _k)| {
                bencher.iter(|| {
                    let mut c = vec![0.0f32; m * n];
                    Gemm::compute(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c),
                        &Shape::matrix(m, k),
                        &Shape::matrix(k, n),
                        &MatMulConfig::default(),
                    ).unwrap();
                    black_box(c)
                });
            },
        );
        
        // Benchmark parallel implementation would be enabled with parallel feature
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("parallel", &size_label),
            &(m, n, k),
            |bencher, &(_m, _n, _k)| {
                bencher.iter(|| {
                    let mut c = vec![0.0f32; m * n];
                    // The parallel implementation is built into Gemm::compute when parallel feature is enabled
                    Gemm::compute(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c),
                        &Shape::matrix(m, k),
                        &Shape::matrix(k, n),
                        &MatMulConfig::default(),
                    ).unwrap();
                    black_box(c)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel vs sequential batch matrix multiplication
fn bench_parallel_batch_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_matrix_multiplication");
    
    let batch_sizes = vec![2, 4, 8, 16, 32];
    let matrix_size = (256, 256, 256); // Fixed matrix size
    
    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements((batch_size * matrix_size.0 * matrix_size.1 * matrix_size.2) as u64));
        
        // Generate test data for batch
        let single_a_size = matrix_size.0 * matrix_size.2;
        let single_b_size = matrix_size.2 * matrix_size.1;
        let single_c_size = matrix_size.0 * matrix_size.1;
        
        let a: Vec<f32> = (0..batch_size * single_a_size).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..batch_size * single_b_size).map(|i| (i as f32) * 0.01).collect();
        
        group.bench_with_input(
            BenchmarkId::new("batch_parallel", batch_size),
            &batch_size,
            |bencher, &_batch_size| {
                bencher.iter(|| {
                    let mut c = vec![0.0f32; batch_size * single_c_size];
                    BatchedMatMul::compute(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c),
                        &Shape::from_slice(&[batch_size, matrix_size.0, matrix_size.2]),
                        &Shape::from_slice(&[batch_size, matrix_size.2, matrix_size.1]),
                    ).unwrap();
                    black_box(c)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel vs sequential quantization operations
fn bench_parallel_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_dequantization");
    
    let block_counts = vec![16, 64, 256, 1024, 4096];
    
    for block_count in block_counts {
        group.throughput(Throughput::Elements((block_count * 32) as u64)); // 32 elements per block
        
        // Generate Q4_0 test data
        let q4_0_blocks: Vec<BlockQ4_0> = (0..block_count).map(|i| {
            BlockQ4_0 {
                d: half::f16::from_f32(0.1 + (i as f32) * 0.001),
                qs: [0x11; 16], // Some test pattern
            }
        }).collect();
        
        // Generate Q8_0 test data  
        let q8_0_blocks: Vec<BlockQ8_0> = (0..block_count).map(|i| {
            BlockQ8_0 {
                d: half::f16::from_f32(0.1 + (i as f32) * 0.001),
                qs: [10; 32], // Some test pattern
            }
        }).collect();
        
        // Benchmark Q4_0 sequential dequantization
        group.bench_with_input(
            BenchmarkId::new("q4_0_sequential", block_count),
            &block_count,
            |bencher, &_count| {
                bencher.iter(|| {
                    let result = dequantize_q4_0(black_box(&q4_0_blocks));
                    black_box(result)
                });
            },
        );
        
        // Benchmark Q4_0 parallel dequantization
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("q4_0_parallel", block_count),
            &block_count,
            |bencher, &_count| {
                bencher.iter(|| {
                    let result = dequantize_q4_0_batch(black_box(&q4_0_blocks));
                    black_box(result)
                });
            },
        );
        
        // Benchmark Q8_0 sequential dequantization
        group.bench_with_input(
            BenchmarkId::new("q8_0_sequential", block_count),
            &block_count,
            |bencher, &_count| {
                bencher.iter(|| {
                    let result = dequantize_q8_0(black_box(&q8_0_blocks));
                    black_box(result)
                });
            },
        );
        
        // Benchmark Q8_0 parallel dequantization
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("q8_0_parallel", block_count),
            &block_count,
            |bencher, &_count| {
                bencher.iter(|| {
                    let result = dequantize_q8_0_batch(black_box(&q8_0_blocks));
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel vs sequential attention head computation
fn bench_parallel_attention_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_head_simulation");
    
    let test_configs = vec![
        (32, 8, 64),   // seq_len, num_heads, head_dim
        (64, 16, 64),
        (128, 32, 64),
        (256, 32, 128),
        (512, 32, 128),
    ];
    
    for (seq_len, num_heads, head_dim) in test_configs {
        let config_label = format!("seq{}_heads{}_dim{}", seq_len, num_heads, head_dim);
        group.throughput(Throughput::Elements((seq_len * num_heads * head_dim) as u64));
        
        // Generate attention computation simulation data
        let hidden_size = num_heads * head_dim;
        let query: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let key: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.01 + 0.5).collect();
        let value: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.01 + 1.0).collect();
        
        // Sequential attention computation (simplified)
        group.bench_with_input(
            BenchmarkId::new("sequential", &config_label),
            &(seq_len, num_heads, head_dim),
            |bencher, &(seq_len, num_heads, head_dim)| {
                bencher.iter(|| {
                    let mut attention_output = vec![0.0f32; seq_len * num_heads * head_dim];
                    
                    // Simulate attention computation per head (sequential)
                    for h in 0..num_heads {
                        for i in 0..seq_len {
                            for j in 0..seq_len {
                                // Simplified attention score computation
                                let mut score = 0.0f32;
                                for d in 0..head_dim {
                                    let q_idx = i * num_heads * head_dim + h * head_dim + d;
                                    let k_idx = j * num_heads * head_dim + h * head_dim + d;
                                    score += query[q_idx] * key[k_idx];
                                }
                                
                                // Apply to output (simplified)
                                for d in 0..head_dim {
                                    let v_idx = j * num_heads * head_dim + h * head_dim + d;
                                    let out_idx = i * num_heads * head_dim + h * head_dim + d;
                                    attention_output[out_idx] += score * value[v_idx] / (seq_len as f32);
                                }
                            }
                        }
                    }
                    
                    black_box(attention_output)
                });
            },
        );
        
        // Parallel attention computation (simplified)
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("parallel", &config_label),
            &(seq_len, num_heads, head_dim),
            |bencher, &(seq_len, num_heads, head_dim)| {
                bencher.iter(|| {
                    let attention_output: Vec<f32> = (0..num_heads).into_par_iter().flat_map(|h| {
                        let mut head_output = vec![0.0f32; seq_len * head_dim];
                        
                        for i in 0..seq_len {
                            for j in 0..seq_len {
                                // Simplified attention score computation
                                let mut score = 0.0f32;
                                for d in 0..head_dim {
                                    let q_idx = i * num_heads * head_dim + h * head_dim + d;
                                    let k_idx = j * num_heads * head_dim + h * head_dim + d;
                                    score += query[q_idx] * key[k_idx];
                                }
                                
                                // Apply to output (simplified)
                                for d in 0..head_dim {
                                    let v_idx = j * num_heads * head_dim + h * head_dim + d;
                                    let out_idx = i * head_dim + d;
                                    head_output[out_idx] += score * value[v_idx] / (seq_len as f32);
                                }
                            }
                        }
                        
                        head_output
                    }).collect();
                    
                    black_box(attention_output)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark feed-forward network parallel vs sequential processing
fn bench_parallel_ffn_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffn_simulation");
    
    let test_configs = vec![
        (128, 512, 2048),   // seq_len, hidden_size, intermediate_size
        (256, 1024, 4096),
        (512, 2048, 8192),
        (1024, 4096, 16384),
    ];
    
    for (seq_len, hidden_size, intermediate_size) in test_configs {
        let config_label = format!("seq{}_hidden{}_inter{}", seq_len, hidden_size, intermediate_size);
        group.throughput(Throughput::Elements((seq_len * intermediate_size) as u64));
        
        // Generate FFN simulation data
        let input: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let w1: Vec<f32> = (0..hidden_size * intermediate_size).map(|i| (i as f32) * 0.001).collect();
        let w2: Vec<f32> = (0..intermediate_size * hidden_size).map(|i| (i as f32) * 0.001).collect();
        
        // Sequential FFN processing
        group.bench_with_input(
            BenchmarkId::new("sequential", &config_label),
            &(seq_len, hidden_size, intermediate_size),
            |bencher, &(seq_len, hidden_size, intermediate_size)| {
                bencher.iter(|| {
                    // First linear layer: input @ w1
                    let mut intermediate = vec![0.0f32; seq_len * intermediate_size];
                    for s in 0..seq_len {
                        for i in 0..intermediate_size {
                            let mut sum = 0.0f32;
                            for h in 0..hidden_size {
                                let input_idx = s * hidden_size + h;
                                let weight_idx = h * intermediate_size + i;
                                sum += input[input_idx] * w1[weight_idx];
                            }
                            intermediate[s * intermediate_size + i] = sum;
                        }
                    }
                    
                    // Apply activation (SiLU simulation)
                    for val in intermediate.iter_mut() {
                        *val = *val / (1.0 + (-*val).exp());
                    }
                    
                    // Second linear layer: intermediate @ w2
                    let mut output = vec![0.0f32; seq_len * hidden_size];
                    for s in 0..seq_len {
                        for h in 0..hidden_size {
                            let mut sum = 0.0f32;
                            for i in 0..intermediate_size {
                                let inter_idx = s * intermediate_size + i;
                                let weight_idx = i * hidden_size + h;
                                sum += intermediate[inter_idx] * w2[weight_idx];
                            }
                            output[s * hidden_size + h] = sum;
                        }
                    }
                    
                    black_box(output)
                });
            },
        );
        
        // Parallel FFN processing
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("parallel", &config_label),
            &(seq_len, hidden_size, intermediate_size),
            |bencher, &(seq_len, hidden_size, intermediate_size)| {
                bencher.iter(|| {
                    // First linear layer with parallel processing
                    let mut intermediate = vec![0.0f32; seq_len * intermediate_size];
                    intermediate.par_chunks_mut(intermediate_size).enumerate().for_each(|(s, seq_output)| {
                        for i in 0..intermediate_size {
                            let mut sum = 0.0f32;
                            for h in 0..hidden_size {
                                let input_idx = s * hidden_size + h;
                                let weight_idx = h * intermediate_size + i;
                                sum += input[input_idx] * w1[weight_idx];
                            }
                            seq_output[i] = sum;
                        }
                    });
                    
                    // Parallel activation
                    intermediate.par_iter_mut().for_each(|val| {
                        *val = *val / (1.0 + (-*val).exp());
                    });
                    
                    // Second linear layer with parallel processing
                    let mut output = vec![0.0f32; seq_len * hidden_size];
                    output.par_chunks_mut(hidden_size).enumerate().for_each(|(s, seq_output)| {
                        for h in 0..hidden_size {
                            let mut sum = 0.0f32;
                            for i in 0..intermediate_size {
                                let inter_idx = s * intermediate_size + i;
                                let weight_idx = i * hidden_size + h;
                                sum += intermediate[inter_idx] * w2[weight_idx];
                            }
                            seq_output[h] = sum;
                        }
                    });
                    
                    black_box(output)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark thread scaling efficiency
fn bench_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling");
    group.measurement_time(Duration::from_secs(10));
    
    let matrix_size = (1024, 1024, 1024);
    let a: Vec<f32> = (0..matrix_size.0 * matrix_size.2).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..matrix_size.2 * matrix_size.1).map(|i| (i as f32) * 0.01).collect();
    
    // Test different thread counts
    for thread_count in [1, 2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("matmul_threads", thread_count),
            &thread_count,
            |bencher, &thread_count| {
                bencher.iter(|| {
                    // Configure Rayon for this thread count
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(thread_count)
                        .build()
                        .unwrap();
                    
                    pool.install(|| {
                        let mut c = vec![0.0f32; matrix_size.0 * matrix_size.1];
                        Gemm::compute(
                            black_box(&a),
                            black_box(&b),
                            black_box(&mut c),
                            &Shape::matrix(matrix_size.0, matrix_size.2),
                            &Shape::matrix(matrix_size.2, matrix_size.1),
                            &MatMulConfig::default(),
                        ).unwrap();
                        black_box(c)
                    })
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    parallel_benches,
    bench_parallel_matmul,
    bench_parallel_batch_matmul,
    bench_parallel_quantization,
    bench_parallel_attention_simulation,
    bench_parallel_ffn_simulation,
    bench_thread_scaling
);

criterion_main!(parallel_benches);