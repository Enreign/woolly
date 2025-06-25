//! Performance validation tests to ensure our optimizations work correctly
//! 
//! This benchmark suite validates that our SIMD implementations produce
//! correct results while achieving performance improvements.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use woolly_tensor::ops::simd::SimdF32;
use woolly_tensor::ops::neural::*;
use woolly_tensor::quantization::{BlockQ4_0, BlockQ8_0, simd, dequantize_q4_0, dequantize_q8_0};
use half::f16;
use approx::assert_relative_eq;

/// Test data generation with known patterns for validation
fn generate_validation_data(size: usize, pattern: f32) -> Vec<f32> {
    (0..size).map(|i| (i as f32 * pattern).sin()).collect()
}

/// Generate deterministic Q4_0 blocks for validation
fn generate_validation_q4_0_blocks(num_blocks: usize) -> Vec<BlockQ4_0> {
    (0..num_blocks).map(|i| {
        let scale = 1.0 + (i as f32 * 0.1);
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

/// Generate deterministic Q8_0 blocks for validation
fn generate_validation_q8_0_blocks(num_blocks: usize) -> Vec<BlockQ8_0> {
    (0..num_blocks).map(|i| {
        let scale = 1.0 + (i as f32 * 0.1);
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

/// Validate SIMD element-wise operations produce correct results
fn validate_element_wise_ops(c: &mut Criterion) {
    let sizes = vec![1000, 4000, 16000];
    
    for size in sizes {
        let a = generate_validation_data(size, 0.1);
        let b = generate_validation_data(size, 0.2);
        let mut simd_output = vec![0.0f32; size];
        let mut scalar_output = vec![0.0f32; size];
        
        let mut group = c.benchmark_group("validate_element_wise");
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::new("simd_add_validated", size), &size, |bencher, &_size| {
            bencher.iter(|| {
                SimdF32::add(black_box(&a), black_box(&b), black_box(&mut simd_output));
                
                // Validate against scalar computation
                for ((a_val, b_val), out) in a.iter().zip(b.iter()).zip(scalar_output.iter_mut()) {
                    *out = *a_val + *b_val;
                }
                
                // Check that SIMD and scalar results match
                for (simd_val, scalar_val) in simd_output.iter().zip(scalar_output.iter()) {
                    assert_relative_eq!(simd_val, scalar_val, epsilon = 1e-6);
                }
            });
        });
        
        group.finish();
    }
}

/// Validate quantized operations produce correct results
fn validate_quantized_ops(c: &mut Criterion) {
    let num_blocks_list = vec![10, 50, 200];
    
    for num_blocks in num_blocks_list {
        let q4_0_blocks = generate_validation_q4_0_blocks(num_blocks);
        let q8_0_blocks = generate_validation_q8_0_blocks(num_blocks);
        
        let mut group = c.benchmark_group("validate_quantized");
        group.throughput(Throughput::Elements((num_blocks * 32) as u64));
        
        // Validate Q4_0 dequantization
        group.bench_with_input(
            BenchmarkId::new("q4_0_validated", num_blocks), 
            &num_blocks, 
            |b, &_blocks| {
                b.iter(|| {
                    let simd_result = simd::dequantize_q4_0_simd(black_box(&q4_0_blocks));
                    let scalar_result = dequantize_q4_0(black_box(&q4_0_blocks));
                    
                    // Validate results match
                    assert_eq!(simd_result.len(), scalar_result.len());
                    for (simd_val, scalar_val) in simd_result.iter().zip(scalar_result.iter()) {
                        assert_relative_eq!(simd_val, scalar_val, epsilon = 1e-5);
                    }
                });
            }
        );
        
        // Validate Q8_0 dequantization
        group.bench_with_input(
            BenchmarkId::new("q8_0_validated", num_blocks), 
            &num_blocks, 
            |b, &_blocks| {
                b.iter(|| {
                    let simd_result = simd::dequantize_q8_0_simd(black_box(&q8_0_blocks));
                    let scalar_result = dequantize_q8_0(black_box(&q8_0_blocks));
                    
                    // Validate results match
                    assert_eq!(simd_result.len(), scalar_result.len());
                    for (simd_val, scalar_val) in simd_result.iter().zip(scalar_result.iter()) {
                        assert_relative_eq!(simd_val, scalar_val, epsilon = 1e-6);
                    }
                });
            }
        );
        
        group.finish();
    }
}

/// Validate neural network operations
fn validate_neural_ops(c: &mut Criterion) {
    let sizes = vec![1000, 4000];
    
    for size in sizes {
        let input = generate_validation_data(size, 0.05);
        let mut optimized_output = vec![0.0f32; size];
        let mut reference_output = vec![0.0f32; size];
        
        let mut group = c.benchmark_group("validate_neural");
        group.throughput(Throughput::Elements(size as u64));
        
        // Validate Softmax
        group.bench_with_input(BenchmarkId::new("softmax_validated", size), &size, |b, &_size| {
            b.iter(|| {
                // Our optimized implementation
                Softmax::apply_f32(
                    black_box(&input), 
                    black_box(&mut optimized_output), 
                    1, 
                    size
                ).unwrap();
                
                // Reference implementation
                let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;
                for (inp, out) in input.iter().zip(reference_output.iter_mut()) {
                    *out = (inp - max_val).exp();
                    sum += *out;
                }
                let inv_sum = 1.0 / sum;
                for out in reference_output.iter_mut() {
                    *out *= inv_sum;
                }
                
                // Validate results
                for (opt_val, ref_val) in optimized_output.iter().zip(reference_output.iter()) {
                    assert_relative_eq!(opt_val, ref_val, epsilon = 1e-4);
                }
                
                // Validate sum is approximately 1.0
                let total_sum: f32 = optimized_output.iter().sum();
                assert_relative_eq!(total_sum, 1.0, epsilon = 1e-5);
            });
        });
        
        // Validate GELU
        group.bench_with_input(BenchmarkId::new("gelu_validated", size), &size, |b, &_size| {
            b.iter(|| {
                // Our optimized implementation
                GELU::apply_f32(black_box(&input), black_box(&mut optimized_output)).unwrap();
                
                // Reference implementation
                for (inp, out) in input.iter().zip(reference_output.iter_mut()) {
                    let x = *inp;
                    let tanh_arg = 0.7978845608 * (x + 0.044715 * x * x * x);
                    *out = 0.5 * x * (1.0 + tanh_arg.tanh());
                }
                
                // Validate results
                for (opt_val, ref_val) in optimized_output.iter().zip(reference_output.iter()) {
                    assert_relative_eq!(opt_val, ref_val, epsilon = 1e-3);
                }
            });
        });
        
        // Validate RMSNorm
        let weight = generate_validation_data(size, 0.02);
        group.bench_with_input(BenchmarkId::new("rmsnorm_validated", size), &size, |b, &_size| {
            b.iter(|| {
                // Our optimized implementation
                RMSNorm::apply_f32(
                    black_box(&input), 
                    black_box(&weight),
                    black_box(&mut optimized_output),
                    1,
                    size,
                    1e-6
                ).unwrap();
                
                // Reference implementation
                let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
                let rms = (sum_sq / size as f32 + 1e-6).sqrt();
                let inv_rms = 1.0 / rms;
                for ((inp, w), out) in input.iter().zip(weight.iter()).zip(reference_output.iter_mut()) {
                    *out = inp * inv_rms * w;
                }
                
                // Validate results
                for (opt_val, ref_val) in optimized_output.iter().zip(reference_output.iter()) {
                    assert_relative_eq!(opt_val, ref_val, epsilon = 1e-5);
                }
            });
        });
        
        group.finish();
    }
}

/// Validate matrix multiplication accuracy
fn validate_matrix_multiplication(c: &mut Criterion) {
    use woolly_tensor::ops::matmul::{Gemm, MatMulConfig};
    use woolly_tensor::shape::Shape;
    
    let sizes = vec![(32, 32, 32), (64, 64, 64), (128, 128, 64)];
    
    for (m, n, k) in sizes {
        let a = generate_validation_data(m * k, 0.1);
        let b = generate_validation_data(k * n, 0.2);
        let mut optimized_c = vec![0.0f32; m * n];
        let mut reference_c = vec![0.0f32; m * n];
        
        let a_shape = Shape::matrix(m, k);
        let b_shape = Shape::matrix(k, n);
        let config = MatMulConfig::default();
        
        let mut group = c.benchmark_group("validate_matmul");
        group.throughput(Throughput::Elements((m * n * k) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("matmul_validated", format!("{}x{}x{}", m, n, k)), 
            &(m, n, k), 
            |bench, &_size| {
                bench.iter(|| {
                    // Our optimized implementation
                    Gemm::compute(
                        black_box(&a), 
                        black_box(&b), 
                        black_box(&mut optimized_c), 
                        &a_shape, 
                        &b_shape, 
                        &config
                    ).unwrap();
                    
                    // Reference implementation (naive)
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0f32;
                            for l in 0..k {
                                sum += a[i * k + l] * b[l * n + j];
                            }
                            reference_c[i * n + j] = sum;
                        }
                    }
                    
                    // Validate results
                    for (opt_val, ref_val) in optimized_c.iter().zip(reference_c.iter()) {
                        assert_relative_eq!(opt_val, ref_val, epsilon = 1e-4);
                    }
                });
            }
        );
        
        group.finish();
    }
}

/// Performance regression test - ensure we're faster than naive implementations
fn performance_regression_test(c: &mut Criterion) {
    use std::time::Instant;
    
    let size = 16384;
    let a = generate_validation_data(size, 0.1);
    let b = generate_validation_data(size, 0.2);
    let mut simd_output = vec![0.0f32; size];
    let mut scalar_output = vec![0.0f32; size];
    
    // Measure SIMD performance
    let start = Instant::now();
    for _ in 0..100 {
        SimdF32::add(&a, &b, &mut simd_output);
    }
    let simd_duration = start.elapsed();
    
    // Measure scalar performance
    let start = Instant::now();
    for _ in 0..100 {
        for ((a_val, b_val), out) in a.iter().zip(b.iter()).zip(scalar_output.iter_mut()) {
            *out = *a_val + *b_val;
        }
    }
    let scalar_duration = start.elapsed();
    
    // We should be at least 2x faster with SIMD on most systems
    let speedup = scalar_duration.as_nanos() as f64 / simd_duration.as_nanos() as f64;
    
    let mut group = c.benchmark_group("performance_regression");
    
    group.bench_function("speedup_validation", |b| {
        b.iter(|| {
            // Log the achieved speedup for monitoring
            println!("SIMD speedup: {:.2}x", speedup);
            
            // On systems with proper SIMD support, we should see improvement
            // Note: This may not hold on all systems, especially in CI environments
            if cfg!(target_feature = "avx2") {
                // Only enforce speedup requirement if AVX2 is available
                assert!(speedup > 1.0, "SIMD implementation should be faster than scalar");
            }
            
            speedup
        });
    });
    
    group.finish();
}

/// Memory bandwidth utilization test
fn memory_bandwidth_test(c: &mut Criterion) {
    let sizes = vec![1024, 8192, 65536, 262144];
    
    for size in sizes {
        let data = generate_validation_data(size, 0.1);
        
        let mut group = c.benchmark_group("memory_bandwidth");
        group.throughput(Throughput::Bytes((size * std::mem::size_of::<f32>()) as u64));
        
        // Sequential memory access (should achieve near peak bandwidth)
        group.bench_with_input(BenchmarkId::new("sequential_read", size), &size, |b, &_size| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for &val in &data {
                    sum += black_box(val);
                }
                sum
            });
        });
        
        // SIMD memory access
        group.bench_with_input(BenchmarkId::new("simd_add", size), &size, |b, &_size| {
            let mut output = vec![0.0f32; size];
            b.iter(|| {
                SimdF32::add(black_box(&data), black_box(&data), black_box(&mut output));
            });
        });
        
        group.finish();
    }
}

criterion_group!(
    validation_benches,
    validate_element_wise_ops,
    validate_quantized_ops,
    validate_neural_ops,
    validate_matrix_multiplication,
    performance_regression_test,
    memory_bandwidth_test
);

criterion_main!(validation_benches);