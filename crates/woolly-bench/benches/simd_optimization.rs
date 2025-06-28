//! SIMD optimization benchmarks
//! 
//! Comprehensive benchmarks to validate the 4-8x speedup from SIMD optimizations
//! for transformer-specific operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use woolly_tensor::{
    Shape,
    ops::{
        matmul::{Gemm, MatMulConfig},
        simd_matmul::{SimdMatVec, CacheAwareMatVec, TransformerSIMD, MatVecConfig},
    },
};

/// Benchmark matrix-vector multiplication performance across different sizes
fn bench_matrix_vector_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_vector_multiplication");
    
    // Test sizes typical for transformer models
    let test_sizes = vec![
        (128, 128),    // Small attention head
        (512, 512),    // Medium hidden size
        (1024, 1024),  // Large hidden size
        (2048, 2048),  // Very large hidden size
        (4096, 4096),  // XL model size
        (768, 3072),   // GPT-2 style FFN
        (1024, 4096),  // Larger FFN
        (2048, 8192),  // GPT-3 style FFN
    ];
    
    for (rows, cols) in test_sizes {
        let matrix = (0..rows * cols).map(|i| (i as f32) * 0.001).collect::<Vec<f32>>();
        let vector = (0..cols).map(|i| (i as f32) * 0.01).collect::<Vec<f32>>();
        let mut output = vec![0.0f32; rows];
        let shape = Shape::matrix(rows, cols);
        let config = MatVecConfig::default();
        
        group.throughput(Throughput::Elements((rows * cols) as u64));
        
        // Benchmark naive implementation
        group.bench_with_input(
            BenchmarkId::new("naive", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    for i in 0..rows {
                        let mut sum = 0.0f32;
                        for j in 0..cols {
                            sum += matrix[i * cols + j] * vector[j];
                        }
                        output[i] = sum;
                    }
                    black_box(&output);
                });
            },
        );
        
        // Benchmark SIMD implementation
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, _| {
                b.iter(|| {
                    SimdMatVec::compute(
                        black_box(&matrix),
                        black_box(&vector),
                        black_box(&mut output),
                        &shape,
                        &config,
                    ).unwrap();
                    black_box(&output);
                });
            },
        );
        
        // Benchmark cache-aware implementation for larger matrices
        if rows >= 512 && cols >= 512 {
            group.bench_with_input(
                BenchmarkId::new("cache_aware", format!("{}x{}", rows, cols)),
                &(rows, cols),
                |b, _| {
                    b.iter(|| {
                        CacheAwareMatVec::compute_blocked(
                            black_box(&matrix),
                            black_box(&vector),
                            black_box(&mut output),
                            &shape,
                            &config,
                        ).unwrap();
                        black_box(&output);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark RMSNorm performance
fn bench_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm");
    
    let test_sizes = vec![128, 256, 512, 768, 1024, 2048, 4096, 8192];
    
    for size in test_sizes {
        let input = (0..size).map(|i| (i as f32) * 0.001 - 0.5).collect::<Vec<f32>>();
        let weight = vec![1.0f32; size];
        let mut output = vec![0.0f32; size];
        let epsilon = 1e-6;
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark naive RMSNorm
        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
                    let rms = (sum_sq / size as f32 + epsilon).sqrt();
                    let scale = 1.0 / rms;
                    
                    for i in 0..size {
                        output[i] = input[i] * scale * weight[i];
                    }
                    black_box(&output);
                });
            },
        );
        
        // Benchmark SIMD RMSNorm
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &size,
            |b, _| {
                b.iter(|| {
                    TransformerSIMD::rms_norm(
                        black_box(&input),
                        black_box(&weight),
                        epsilon,
                        black_box(&mut output),
                    ).unwrap();
                    black_box(&output);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark SwiGLU activation performance
fn bench_swiglu(c: &mut Criterion) {
    let mut group = c.benchmark_group("swiglu");
    
    let test_sizes = vec![128, 256, 512, 1024, 2048, 4096, 8192, 16384];
    
    for size in test_sizes {
        let gate = (0..size).map(|i| (i as f32) * 0.001 - 1.0).collect::<Vec<f32>>();
        let up = (0..size).map(|i| (i as f32) * 0.002 + 0.5).collect::<Vec<f32>>();
        let mut output = vec![0.0f32; size];
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark naive SwiGLU
        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &size,
            |b, _| {
                b.iter(|| {
                    for i in 0..size {
                        let g = gate[i];
                        let swish = g / (1.0 + (-g).exp());
                        output[i] = swish * up[i];
                    }
                    black_box(&output);
                });
            },
        );
        
        // Benchmark SIMD SwiGLU
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &size,
            |b, _| {
                b.iter(|| {
                    TransformerSIMD::swiglu_activation(
                        black_box(&gate),
                        black_box(&up),
                        black_box(&mut output),
                    ).unwrap();
                    black_box(&output);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark element-wise operations
fn bench_elementwise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_operations");
    
    let test_sizes = vec![1024, 4096, 16384, 65536, 262144];
    
    for size in test_sizes {
        let a = (0..size).map(|i| (i as f32) * 0.001).collect::<Vec<f32>>();
        let b = (0..size).map(|i| (i as f32) * 0.002).collect::<Vec<f32>>();
        let mut result = vec![0.0f32; size];
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark naive element-wise addition
        group.bench_with_input(
            BenchmarkId::new("naive_add", size),
            &size,
            |b_bench, _| {
                b_bench.iter(|| {
                    for i in 0..size {
                        result[i] = a[i] + b[i];
                    }
                    black_box(&result);
                });
            },
        );
        
        // Benchmark SIMD element-wise addition
        group.bench_with_input(
            BenchmarkId::new("simd_add", size),
            &size,
            |b_bench, _| {
                b_bench.iter(|| {
                    result.copy_from_slice(&a);
                    // Simulate SIMD addition by using the fact that we have the add implementation
                    #[cfg(target_arch = "aarch64")]
                    {
                        unsafe {
                            use std::arch::aarch64::*;
                            let mut i = 0;
                            while i + 4 <= size {
                                let va = vld1q_f32(result.as_ptr().add(i));
                                let vb = vld1q_f32(b.as_ptr().add(i));
                                let sum = vaddq_f32(va, vb);
                                vst1q_f32(result.as_mut_ptr().add(i), sum);
                                i += 4;
                            }
                            while i < size {
                                result[i] += b[i];
                                i += 1;
                            }
                        }
                    }
                    #[cfg(target_arch = "x86_64")]
                    {
                        if std::arch::is_x86_feature_detected!("avx2") {
                            unsafe {
                                use std::arch::x86_64::*;
                                let mut i = 0;
                                while i + 8 <= size {
                                    let va = _mm256_loadu_ps(result.as_ptr().add(i));
                                    let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                                    let sum = _mm256_add_ps(va, vb);
                                    _mm256_storeu_ps(result.as_mut_ptr().add(i), sum);
                                    i += 8;
                                }
                                while i < size {
                                    result[i] += b[i];
                                    i += 1;
                                }
                            }
                        } else {
                            for i in 0..size {
                                result[i] += b[i];
                            }
                        }
                    }
                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    {
                        for i in 0..size {
                            result[i] += b[i];
                        }
                    }
                    black_box(&result);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark full matrix multiplication performance
fn bench_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    
    // Test transformer-typical sizes
    let test_configs = vec![
        (64, 64, 64),      // Small attention
        (128, 128, 128),   // Medium attention  
        (256, 256, 256),   // Large attention
        (512, 512, 512),   // Very large attention
        (768, 768, 3072),  // GPT-2 FFN
        (1024, 1024, 4096), // GPT-3 FFN
        (2048, 2048, 8192), // Large model FFN
    ];
    
    for (m, k, n) in test_configs {
        let a = (0..m * k).map(|i| (i as f32) * 0.001).collect::<Vec<f32>>();
        let b = (0..k * n).map(|i| (i as f32) * 0.001).collect::<Vec<f32>>();
        let mut c = vec![0.0f32; m * n];
        
        let a_shape = Shape::matrix(m, k);
        let b_shape = Shape::matrix(k, n);
        let config = MatMulConfig::default();
        
        group.throughput(Throughput::Elements((m * k * n) as u64));
        
        // Benchmark optimized GEMM
        group.bench_with_input(
            BenchmarkId::new("gemm_optimized", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |b_bench, _| {
                b_bench.iter(|| {
                    Gemm::compute(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c),
                        &a_shape,
                        &b_shape,
                        &config,
                    ).unwrap();
                    black_box(&c);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory bandwidth for different access patterns
fn bench_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_bandwidth");
    
    let sizes = vec![1024, 4096, 16384, 65536, 262144, 1048576];
    
    for size in sizes {
        let data = vec![1.0f32; size];
        
        group.throughput(Throughput::Bytes((size * 4) as u64)); // f32 = 4 bytes
        
        // Sequential read
        group.bench_with_input(
            BenchmarkId::new("sequential_read", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for &val in &data {
                        sum += val;
                    }
                    black_box(sum);
                });
            },
        );
        
        // Strided read (simulate matrix column access)
        if size >= 1024 {
            let stride = (size as f32).sqrt() as usize;
            group.bench_with_input(
                BenchmarkId::new("strided_read", format!("{}_stride_{}", size, stride)),
                &size,
                |b, _| {
                    b.iter(|| {
                        let mut sum = 0.0f32;
                        let mut i = 0;
                        while i < size {
                            sum += data[i];
                            i += stride;
                        }
                        black_box(sum);
                    });
                },
            );
        }
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_matrix_vector_multiplication,
    bench_rmsnorm,
    bench_swiglu,
    bench_elementwise_ops,
    bench_matrix_multiplication,
    bench_memory_bandwidth
);
criterion_main!(benches);