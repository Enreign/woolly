//! Benchmark comparing original SIMD vs optimized SIMD implementations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use woolly_tensor::{
    Shape,
    ops::{
        simd_matmul::{SimdMatVec, MatVecConfig},
        simd_optimized::{SimdOpsOptimized, OptimizedSimdMatVec, OptimizedMatVecConfig},
    },
};
use woolly_core::model::memory_pool::TensorMemoryPool;
use woolly_core::model::memory_pool_enhanced::EnhancedTensorMemoryPool;

/// Test different matrix sizes
const TEST_SIZES: &[(usize, usize)] = &[
    (128, 128),      // Small - overhead dominates
    (256, 256),      // Medium - threshold size
    (512, 512),      // Large - SIMD benefits
    (1024, 1024),    // XLarge - cache effects
    (2048, 2048),    // XXLarge - memory bandwidth
];

fn bench_original_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("original_simd_matvec");
    
    for &(m, n) in TEST_SIZES {
        let matrix = vec![0.1f32; m * n];
        let vector = vec![0.2f32; n];
        let matrix_shape = Shape::matrix(m, n);
        
        group.bench_with_input(
            BenchmarkId::new("original", format!("{}x{}", m, n)),
            &(m, n),
            |b, _| {
                b.iter(|| {
                    let mut output = vec![0.0f32; m];
                    SimdMatVec::compute(
                        black_box(&matrix),
                        black_box(&vector),
                        black_box(&mut output),
                        &matrix_shape,
                        &MatVecConfig::default(),
                    ).unwrap();
                    output
                });
            },
        );
    }
    
    group.finish();
}

fn bench_optimized_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimized_simd_matvec");
    
    for &(m, n) in TEST_SIZES {
        let matrix = vec![0.1f32; m * n];
        let vector = vec![0.2f32; n];
        let matrix_shape = Shape::matrix(m, n);
        
        group.bench_with_input(
            BenchmarkId::new("optimized", format!("{}x{}", m, n)),
            &(m, n),
            |b, _| {
                b.iter(|| {
                    let output = SimdOpsOptimized::matvec(
                        black_box(&matrix),
                        black_box(&vector),
                        &matrix_shape,
                        false,
                    ).unwrap();
                    output
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    
    // Benchmark raw allocation
    group.bench_function("raw_allocation_1024", |b| {
        b.iter(|| {
            let v = vec![0.0f32; 1024];
            black_box(v)
        });
    });
    
    // Benchmark original memory pool
    group.bench_function("original_pool_1024", |b| {
        let mut pool = TensorMemoryPool::new();
        b.iter(|| {
            let v = pool.get_buffer(1024);
            pool.return_buffer(v);
        });
    });
    
    // Benchmark enhanced memory pool
    group.bench_function("enhanced_pool_1024", |b| {
        let mut pool = EnhancedTensorMemoryPool::new();
        b.iter(|| {
            let v = pool.get_simd_buffer(1024);
            pool.return_buffer(v);
        });
    });
    
    group.finish();
}

fn bench_cpu_feature_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_feature_detection");
    
    // Benchmark runtime detection
    group.bench_function("runtime_detection", |b| {
        b.iter(|| {
            #[cfg(target_arch = "x86_64")]
            {
                black_box(std::is_x86_feature_detected!("avx2"));
                black_box(std::is_x86_feature_detected!("fma"));
            }
            #[cfg(target_arch = "aarch64")]
            {
                black_box(true); // NEON is always available
            }
        });
    });
    
    // Benchmark cached detection
    group.bench_function("cached_detection", |b| {
        b.iter(|| {
            black_box(SimdOpsOptimized::cpu_features());
        });
    });
    
    group.finish();
}

fn bench_simd_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_threshold");
    
    // Test performance at different sizes to validate threshold
    for size in [64, 128, 256, 512, 1024] {
        let matrix = vec![0.1f32; size * size];
        let vector = vec![0.2f32; size];
        let matrix_shape = Shape::matrix(size, size);
        
        // Benchmark with SIMD forced on
        group.bench_with_input(
            BenchmarkId::new("simd_forced", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut output = vec![0.0f32; size];
                    OptimizedSimdMatVec::compute_pooled(
                        black_box(&matrix),
                        black_box(&vector),
                        black_box(&mut output),
                        &matrix_shape,
                        &OptimizedMatVecConfig {
                            transpose: false,
                            alpha: 1.0,
                            beta: 0.0,
                            simd_threshold: 0, // Force SIMD
                        },
                    ).unwrap();
                    output
                });
            },
        );
        
        // Benchmark with threshold check
        group.bench_with_input(
            BenchmarkId::new("simd_threshold", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let output = SimdOpsOptimized::matvec(
                        black_box(&matrix),
                        black_box(&vector),
                        &matrix_shape,
                        false,
                    ).unwrap();
                    output
                });
            },
        );
    }
    
    group.finish();
}

fn bench_transformer_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformer_ops");
    
    // Typical transformer dimensions
    let seq_len = 512;
    let hidden_size = 768;
    let intermediate_size = 3072;
    
    // Benchmark attention projection (Q, K, V)
    let hidden_states = vec![0.1f32; seq_len * hidden_size];
    let qkv_weight = vec![0.01f32; hidden_size * hidden_size * 3];
    
    group.bench_function("attention_qkv_projection", |b| {
        let mut pool = EnhancedTensorMemoryPool::new();
        b.iter(|| {
            let buffers = pool.get_buffers(&[
                seq_len * hidden_size,  // Q
                seq_len * hidden_size,  // K  
                seq_len * hidden_size,  // V
            ]);
            
            // Simulate QKV projection
            for (i, mut buffer) in buffers.into_iter().enumerate() {
                let weight_offset = i * hidden_size * hidden_size;
                let weight_slice = &qkv_weight[weight_offset..weight_offset + hidden_size * hidden_size];
                
                // Simple matmul simulation
                for s in 0..seq_len {
                    for h in 0..hidden_size {
                        let mut sum = 0.0f32;
                        for k in 0..hidden_size {
                            sum += hidden_states[s * hidden_size + k] * weight_slice[k * hidden_size + h];
                        }
                        buffer[s * hidden_size + h] = sum;
                    }
                }
                
                pool.return_buffer(buffer);
            }
        });
    });
    
    // Benchmark FFN operations
    let ffn_input = vec![0.1f32; seq_len * hidden_size];
    let gate_weight = vec![0.01f32; hidden_size * intermediate_size];
    let up_weight = vec![0.01f32; hidden_size * intermediate_size];
    
    group.bench_function("ffn_gated_projection", |b| {
        let mut pool = EnhancedTensorMemoryPool::new();
        b.iter(|| {
            let buffers = pool.get_buffers(&[
                seq_len * intermediate_size,  // gate
                seq_len * intermediate_size,  // up
                seq_len * intermediate_size,  // swiglu output
            ]);
            
            // Return buffers
            pool.return_buffers(buffers);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_original_simd,
    bench_optimized_simd,
    bench_memory_allocation,
    bench_cpu_feature_detection,
    bench_simd_threshold,
    bench_transformer_operations
);
criterion_main!(benches);