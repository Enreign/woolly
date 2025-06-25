//! Comprehensive benchmarks for SIMD-optimized tensor operations
//!
//! These benchmarks measure the performance improvements of our SIMD optimizations
//! compared to scalar implementations for critical tensor operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use woolly_tensor::ops::simd::SimdF32;
use woolly_tensor::ops::neural::*;
use woolly_tensor::quantization::{BlockQ4_0, BlockQ8_0, simd};
use half::f16;

/// Generate test data for benchmarks
fn generate_test_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32 * 0.1) % 10.0 - 5.0).collect()
}

/// Generate Q4_0 test blocks
fn generate_q4_0_blocks(num_blocks: usize) -> Vec<BlockQ4_0> {
    (0..num_blocks).map(|i| {
        let scale = 0.5 + (i as f32 * 0.1) % 1.0;
        let mut qs = [0u8; 16];
        for j in 0..16 {
            qs[j] = ((i + j) % 16) as u8;
        }
        BlockQ4_0 {
            d: f16::from_f32(scale),
            qs,
        }
    }).collect()
}

/// Generate Q8_0 test blocks
fn generate_q8_0_blocks(num_blocks: usize) -> Vec<BlockQ8_0> {
    (0..num_blocks).map(|i| {
        let scale = 0.5 + (i as f32 * 0.1) % 1.0;
        let mut qs = [0i8; 32];
        for j in 0..32 {
            qs[j] = ((i + j) % 256 - 128) as i8;
        }
        BlockQ8_0 {
            d: f16::from_f32(scale),
            qs,
        }
    }).collect()
}

/// Benchmark SIMD vs scalar element-wise operations
fn bench_element_wise_ops(c: &mut Criterion) {
    let sizes = vec![1024, 4096, 16384, 65536];
    
    for size in sizes {
        let a = generate_test_data(size);
        let b = generate_test_data(size);
        let mut output = vec![0.0f32; size];
        
        let mut group = c.benchmark_group("element_wise_add");
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bencher, &_size| {
            bencher.iter(|| {
                SimdF32::add(black_box(&a), black_box(&b), black_box(&mut output));
            });
        });
        
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bencher, &_size| {
            bencher.iter(|| {
                for ((a_val, b_val), out) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
                    *out = black_box(*a_val + *b_val);
                }
            });
        });
        
        group.finish();
    }
}

/// Benchmark SIMD vs scalar dot product
fn bench_dot_product(c: &mut Criterion) {
    let sizes = vec![1024, 4096, 16384, 65536];
    
    for size in sizes {
        let a = generate_test_data(size);
        let b = generate_test_data(size);
        
        let mut group = c.benchmark_group("dot_product");
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, &_size| {
            bench.iter(|| {
                SimdF32::dot_product(black_box(&a), black_box(&b))
            });
        });
        
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, &_size| {
            bench.iter(|| {
                let mut sum = 0.0f32;
                for (a_val, b_val) in a.iter().zip(b.iter()) {
                    sum += black_box(*a_val * *b_val);
                }
                sum
            });
        });
        
        group.finish();
    }
}

/// Benchmark matrix multiplication implementations
fn bench_matrix_multiplication(c: &mut Criterion) {
    use woolly_tensor::ops::matmul::{Gemm, MatMulConfig};
    use woolly_tensor::shape::Shape;
    
    let sizes = vec![(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 128)];
    
    for (m, n, k) in sizes {
        let a = generate_test_data(m * k);
        let b = generate_test_data(k * n);
        let mut output_c = vec![0.0f32; m * n];
        
        let a_shape = Shape::matrix(m, k);
        let b_shape = Shape::matrix(k, n);
        let config = MatMulConfig::default();
        
        let mut group = c.benchmark_group("matrix_multiplication");
        group.throughput(Throughput::Elements((m * n * k) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("optimized", format!("{}x{}x{}", m, n, k)), 
            &(m, n, k), 
            |bench, &_size| {
                bench.iter(|| {
                    Gemm::compute(
                        black_box(&a), 
                        black_box(&b), 
                        black_box(&mut output_c), 
                        &a_shape, 
                        &b_shape, 
                        &config
                    ).unwrap()
                });
            }
        );
        
        group.finish();
    }
}

/// Benchmark quantized operations
fn bench_quantized_ops(c: &mut Criterion) {
    let num_blocks = vec![32, 128, 512, 2048];
    
    for blocks in num_blocks {
        let q4_0_blocks = generate_q4_0_blocks(blocks);
        let q8_0_blocks = generate_q8_0_blocks(blocks);
        
        let mut group = c.benchmark_group("quantized_dequantization");
        group.throughput(Throughput::Elements((blocks * 32) as u64));
        
        // Q4_0 dequantization
        group.bench_with_input(
            BenchmarkId::new("q4_0_simd", blocks), 
            &blocks, 
            |b, &_blocks| {
                b.iter(|| {
                    simd::dequantize_q4_0_simd(black_box(&q4_0_blocks))
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("q4_0_scalar", blocks), 
            &blocks, 
            |b, &_blocks| {
                b.iter(|| {
                    woolly_tensor::quantization::dequantize_q4_0(black_box(&q4_0_blocks))
                });
            }
        );
        
        // Q8_0 dequantization
        group.bench_with_input(
            BenchmarkId::new("q8_0_simd", blocks), 
            &blocks, 
            |b, &_blocks| {
                b.iter(|| {
                    simd::dequantize_q8_0_simd(black_box(&q8_0_blocks))
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("q8_0_scalar", blocks), 
            &blocks, 
            |b, &_blocks| {
                b.iter(|| {
                    woolly_tensor::quantization::dequantize_q8_0(black_box(&q8_0_blocks))
                });
            }
        );
        
        group.finish();
    }
}

/// Benchmark quantized matrix-vector multiplication
fn bench_quantized_matvec(c: &mut Criterion) {
    use woolly_tensor::quantization::{QuantizedStorage, simd};
    
    let sizes = vec![(1024, 4096), (2048, 4096), (4096, 4096), (8192, 4096)];
    
    for (m, k) in sizes {
        assert_eq!(k % 32, 0); // Must be divisible by block size
        let num_blocks = m * (k / 32);
        
        let q4_0_blocks = generate_q4_0_blocks(num_blocks);
        let q8_0_blocks = generate_q8_0_blocks(num_blocks);
        let input = generate_test_data(k);
        let mut output = vec![0.0f32; m];
        
        let q4_0_storage = QuantizedStorage::Q4_0(q4_0_blocks);
        let q8_0_storage = QuantizedStorage::Q8_0(q8_0_blocks);
        
        let mut group = c.benchmark_group("quantized_matvec");
        group.throughput(Throughput::Elements((m * k) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("q4_0", format!("{}x{}", m, k)), 
            &(m, k), 
            |b, &_size| {
                b.iter(|| {
                    simd::quantized_matvec(
                        black_box(&q4_0_storage), 
                        black_box(&input), 
                        black_box(&mut output), 
                        m, 
                        k
                    ).unwrap()
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("q8_0", format!("{}x{}", m, k)), 
            &(m, k), 
            |b, &_size| {
                b.iter(|| {
                    simd::quantized_matvec(
                        black_box(&q8_0_storage), 
                        black_box(&input), 
                        black_box(&mut output), 
                        m, 
                        k
                    ).unwrap()
                });
            }
        );
        
        group.finish();
    }
}

/// Benchmark neural network operations
fn bench_neural_ops(c: &mut Criterion) {
    let sizes = vec![1024, 4096, 8192, 16384];
    
    for size in sizes {
        let input = generate_test_data(size);
        let mut output = vec![0.0f32; size];
        
        // Softmax benchmark
        let mut group = c.benchmark_group("softmax");
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::new("optimized", size), &size, |b, &_size| {
            b.iter(|| {
                Softmax::apply_f32(
                    black_box(&input), 
                    black_box(&mut output), 
                    1, 
                    size
                ).unwrap()
            });
        });
        
        group.finish();
        
        // GELU benchmark
        let mut group = c.benchmark_group("gelu");
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::new("optimized", size), &size, |b, &_size| {
            b.iter(|| {
                GELU::apply_f32(black_box(&input), black_box(&mut output)).unwrap()
            });
        });
        
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, &_size| {
            b.iter(|| {
                for (inp, out) in input.iter().zip(output.iter_mut()) {
                    let x = *inp;
                    *out = 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh());
                }
            });
        });
        
        group.finish();
        
        // RMSNorm benchmark
        let weight = generate_test_data(size);
        let mut group = c.benchmark_group("rmsnorm");
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::new("optimized", size), &size, |b, &_size| {
            b.iter(|| {
                RMSNorm::apply_f32(
                    black_box(&input), 
                    black_box(&weight),
                    black_box(&mut output),
                    1,
                    size,
                    1e-6
                ).unwrap()
            });
        });
        
        group.finish();
    }
}

/// Benchmark RoPE (Rotary Position Embedding)
fn bench_rope(c: &mut Criterion) {
    let configs = vec![
        (32, 64, 8, 64),   // seq_len, batch, n_heads, head_dim
        (128, 32, 8, 64),
        (512, 8, 8, 64),
        (1024, 4, 8, 64),
    ];
    
    for (seq_len, batch_size, n_heads, head_dim) in configs {
        let total_size = batch_size * seq_len * n_heads * head_dim;
        let input = generate_test_data(total_size);
        let mut output = vec![0.0f32; total_size];
        let positions: Vec<usize> = (0..seq_len).collect();
        
        let mut group = c.benchmark_group("rope");
        group.throughput(Throughput::Elements(total_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("optimized", format!("{}x{}x{}x{}", seq_len, batch_size, n_heads, head_dim)), 
            &(seq_len, batch_size, n_heads, head_dim), 
            |b, &_config| {
                b.iter(|| {
                    RoPE::apply_f32(
                        black_box(&input),
                        black_box(&mut output),
                        black_box(&positions),
                        10000.0,
                        batch_size,
                        seq_len,
                        n_heads,
                        head_dim,
                    ).unwrap()
                });
            }
        );
        
        group.finish();
    }
}

/// Benchmark memory access patterns for cache efficiency
fn bench_memory_patterns(c: &mut Criterion) {
    let size = 65536;
    let data = generate_test_data(size);
    
    let mut group = c.benchmark_group("memory_access");
    group.throughput(Throughput::Elements(size as u64));
    
    // Sequential access
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..size {
                sum += black_box(data[i]);
            }
            sum
        });
    });
    
    // Strided access (cache-unfriendly)
    group.bench_function("strided_8", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in (0..size).step_by(8) {
                sum += black_box(data[i]);
            }
            sum
        });
    });
    
    // Random access (very cache-unfriendly)
    let mut indices: Vec<usize> = (0..size).collect();
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    indices.sort_by_key(|&i| {
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        hasher.finish()
    });
    
    group.bench_function("random", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &i in indices.iter().take(size / 8) {
                sum += black_box(data[i]);
            }
            sum
        });
    });
    
    group.finish();
}

/// Comparative benchmark against common BLAS operations
fn bench_blas_comparison(c: &mut Criterion) {
    // These benchmarks help compare our implementations against optimized BLAS libraries
    // In a real scenario, you'd compare against libraries like OpenBLAS or Intel MKL
    
    let sizes = vec![(256, 256, 256), (512, 512, 512), (1024, 1024, 256)];
    
    for (m, n, k) in sizes {
        let a = generate_test_data(m * k);
        let b = generate_test_data(k * n);
        let mut output_c = vec![0.0f32; m * n];
        
        let mut group = c.benchmark_group("blas_comparison");
        group.throughput(Throughput::Elements((m * n * k) as u64));
        
        // Our optimized implementation
        group.bench_with_input(
            BenchmarkId::new("woolly_tensor", format!("{}x{}x{}", m, n, k)), 
            &(m, n, k), 
            |bench, &_size| {
                use woolly_tensor::ops::matmul::{Gemm, MatMulConfig};
                use woolly_tensor::shape::Shape;
                
                let a_shape = Shape::matrix(m, k);
                let b_shape = Shape::matrix(k, n);
                let config = MatMulConfig::default();
                
                bench.iter(|| {
                    Gemm::compute(
                        black_box(&a), 
                        black_box(&b), 
                        black_box(&mut output_c), 
                        &a_shape, 
                        &b_shape, 
                        &config
                    ).unwrap()
                });
            }
        );
        
        // Naive implementation for comparison
        group.bench_with_input(
            BenchmarkId::new("naive", format!("{}x{}x{}", m, n, k)), 
            &(m, n, k), 
            |bench, &_size| {
                bench.iter(|| {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0f32;
                            for l in 0..k {
                                sum += black_box(a[i * k + l] * b[l * n + j]);
                            }
                            output_c[i * n + j] = sum;
                        }
                    }
                });
            }
        );
        
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_element_wise_ops,
    bench_dot_product,
    bench_matrix_multiplication,
    bench_quantized_ops,
    bench_quantized_matvec,
    bench_neural_ops,
    bench_rope,
    bench_memory_patterns,
    bench_blas_comparison
);

criterion_main!(benches);