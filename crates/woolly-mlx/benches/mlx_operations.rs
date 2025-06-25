//! Benchmarks for MLX operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use woolly_mlx::{MLXBackend, MLXStorage};
use woolly_tensor::backend::{DType, TensorBackend};
use woolly_tensor::shape::Shape;

fn create_test_backend() -> Option<MLXBackend> {
    MLXBackend::new().ok()
}

fn create_test_storage_f32(data: Vec<f32>, shape: &[usize]) -> Option<MLXStorage<f32>> {
    MLXStorage::from_data(
        data,
        Shape::from_slice(shape),
        DType::F32,
        woolly_mlx::Device::CPU,
    ).ok()
}

fn bench_element_wise_operations(c: &mut Criterion) {
    if let Some(backend) = create_test_backend() {
        let sizes = vec![100, 1000, 10000];
        
        for size in sizes {
            let data1: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let data2: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
            
            if let (Some(tensor1), Some(tensor2)) = (
                create_test_storage_f32(data1, &[size]),
                create_test_storage_f32(data2, &[size]),
            ) {
                c.bench_with_input(
                    BenchmarkId::new("element_wise_add", size),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            let result = backend.add(
                                black_box(&tensor1),
                                black_box(&tensor2),
                            );
                            black_box(result)
                        })
                    },
                );
                
                c.bench_with_input(
                    BenchmarkId::new("element_wise_mul", size),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            let result = backend.mul(
                                black_box(&tensor1),
                                black_box(&tensor2),
                            );
                            black_box(result)
                        })
                    },
                );
            }
        }
    }
}

fn bench_matrix_multiplication(c: &mut Criterion) {
    if let Some(backend) = create_test_backend() {
        let sizes = vec![(32, 32), (64, 64), (128, 128), (256, 256)];
        
        for (m, n) in sizes {
            let k = m; // Square matrices
            let data1: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
            let data2: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.01).collect();
            
            if let (Some(lhs), Some(rhs)) = (
                create_test_storage_f32(data1, &[m, k]),
                create_test_storage_f32(data2, &[k, n]),
            ) {
                let lhs_shape = Shape::from_slice(&[m, k]);
                let rhs_shape = Shape::from_slice(&[k, n]);
                
                c.bench_with_input(
                    BenchmarkId::new("matmul", format!("{}x{}x{}", m, k, n)),
                    &(m, k, n),
                    |b, _| {
                        b.iter(|| {
                            let result = backend.matmul(
                                black_box(&lhs),
                                black_box(&rhs),
                                black_box(&lhs_shape),
                                black_box(&rhs_shape),
                            );
                            black_box(result)
                        })
                    },
                );
            }
        }
    }
}

fn bench_reductions(c: &mut Criterion) {
    if let Some(backend) = create_test_backend() {
        let sizes = vec![(100, 100), (500, 500), (1000, 1000)];
        
        for (rows, cols) in sizes {
            let data: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.001).collect();
            
            if let Some(tensor) = create_test_storage_f32(data, &[rows, cols]) {
                let shape = Shape::from_slice(&[rows, cols]);
                
                c.bench_with_input(
                    BenchmarkId::new("sum_reduction", format!("{}x{}", rows, cols)),
                    &(rows, cols),
                    |b, _| {
                        b.iter(|| {
                            let result = backend.sum(
                                black_box(&tensor),
                                black_box(&shape),
                                black_box(&[0]), // Sum along first axis
                                black_box(false),
                            );
                            black_box(result)
                        })
                    },
                );
                
                c.bench_with_input(
                    BenchmarkId::new("mean_reduction", format!("{}x{}", rows, cols)),
                    &(rows, cols),
                    |b, _| {
                        b.iter(|| {
                            let result = backend.mean(
                                black_box(&tensor),
                                black_box(&shape),
                                black_box(&[1]), // Mean along second axis
                                black_box(false),
                            );
                            black_box(result)
                        })
                    },
                );
            }
        }
    }
}

fn bench_tensor_creation(c: &mut Criterion) {
    if let Some(backend) = create_test_backend() {
        let sizes = vec![1000, 10000, 100000];
        
        for size in sizes {
            let shape = Shape::from_slice(&[size]);
            
            c.bench_with_input(
                BenchmarkId::new("zeros_creation", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = backend.zeros::<f32>(
                            black_box(&shape),
                            black_box(DType::F32),
                        );
                        black_box(result)
                    })
                },
            );
            
            c.bench_with_input(
                BenchmarkId::new("ones_creation", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = backend.ones::<f32>(
                            black_box(&shape),
                            black_box(DType::F32),
                        );
                        black_box(result)
                    })
                },
            );
        }
    }
}

fn bench_quantization(c: &mut Criterion) {
    if let Some(backend) = create_test_backend() {
        let sizes = vec![1000, 10000, 100000];
        
        for size in sizes {
            let data: Vec<f32> = (0..size).map(|i| (i as f32 - size as f32 / 2.0) * 0.01).collect();
            
            if let Some(tensor) = create_test_storage_f32(data, &[size]) {
                c.bench_with_input(
                    BenchmarkId::new("quantize_q8_0", size),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            let result = backend.quantize(
                                black_box(&tensor),
                                black_box(woolly_tensor::quantization::QuantizationScheme::Q8_0),
                            );
                            black_box(result)
                        })
                    },
                );
                
                c.bench_with_input(
                    BenchmarkId::new("quantize_q4_0", size),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            let result = backend.quantize(
                                black_box(&tensor),
                                black_box(woolly_tensor::quantization::QuantizationScheme::Q4_0),
                            );
                            black_box(result)
                        })
                    },
                );
            }
        }
    }
}

fn bench_memory_operations(c: &mut Criterion) {
    if let Some(backend) = create_test_backend() {
        let sizes = vec![1000, 10000, 100000];
        
        for size in sizes {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            
            if let Some(tensor) = create_test_storage_f32(data.clone(), &[size]) {
                c.bench_with_input(
                    BenchmarkId::new("to_vec", size),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            let result = black_box(&tensor).to_vec();
                            black_box(result)
                        })
                    },
                );
                
                let shape = Shape::from_slice(&[size]);
                c.bench_with_input(
                    BenchmarkId::new("from_slice", size),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            let result = backend.from_slice(
                                black_box(&data),
                                black_box(&shape),
                                black_box(DType::F32),
                            );
                            black_box(result)
                        })
                    },
                );
            }
        }
    }
}

criterion_group!(
    benches,
    bench_element_wise_operations,
    bench_matrix_multiplication,
    bench_reductions,
    bench_tensor_creation,
    bench_quantization,
    bench_memory_operations
);

criterion_main!(benches);