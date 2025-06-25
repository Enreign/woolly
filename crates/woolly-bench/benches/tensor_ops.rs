use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use woolly_bench::{Benchmark, BenchmarkResult, run_benchmark_iterations};
use woolly_tensor::{Tensor, TensorOps, Shape, DType};
use std::time::Duration;

/// Benchmark for tensor creation
struct TensorCreationBench {
    sizes: Vec<usize>,
}

impl TensorCreationBench {
    fn new() -> Self {
        Self {
            sizes: vec![100, 1000, 10000, 100000],
        }
    }
}

impl Benchmark for TensorCreationBench {
    fn name(&self) -> &str {
        "Tensor Creation"
    }
    
    fn run(&mut self) -> anyhow::Result<BenchmarkResult> {
        let mut total_time = Duration::ZERO;
        let iterations = 100;
        
        for &size in &self.sizes {
            let result = run_benchmark_iterations(
                &format!("create_tensor_{}", size),
                iterations,
                || {
                    let _tensor = Tensor::zeros(&[size], DType::F32);
                    Ok(())
                }
            )?;
            total_time += result.mean_time;
        }
        
        Ok(BenchmarkResult {
            name: self.name().to_string(),
            iterations: iterations * self.sizes.len() as u32,
            total_time,
            mean_time: total_time / self.sizes.len() as u32,
            min_time: total_time,
            max_time: total_time,
            stddev: None,
            throughput: None,
            metadata: serde_json::json!({
                "sizes": self.sizes,
            }),
        })
    }
}

/// Benchmark for matrix multiplication
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    
    for size in [64, 128, 256, 512].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // Create matrices
            let a = Tensor::randn(&[size, size], DType::F32, 0.0, 1.0);
            let b = Tensor::randn(&[size, size], DType::F32, 0.0, 1.0);
            
            b.iter(|| {
                // Perform matrix multiplication
                let _c = black_box(a.matmul(&b));
            });
        });
    }
    
    group.finish();
}

/// Benchmark for element-wise operations
fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise");
    
    let sizes = vec![1000, 10000, 100000, 1000000];
    
    for size in sizes {
        // Addition
        group.bench_with_input(
            BenchmarkId::new("add", size),
            &size,
            |b, &size| {
                let a = Tensor::randn(&[size], DType::F32, 0.0, 1.0);
                let b = Tensor::randn(&[size], DType::F32, 0.0, 1.0);
                
                b.iter(|| {
                    let _c = black_box(&a + &b);
                });
            }
        );
        
        // Multiplication
        group.bench_with_input(
            BenchmarkId::new("mul", size),
            &size,
            |b, &size| {
                let a = Tensor::randn(&[size], DType::F32, 0.0, 1.0);
                let b = Tensor::randn(&[size], DType::F32, 0.0, 1.0);
                
                b.iter(|| {
                    let _c = black_box(&a * &b);
                });
            }
        );
        
        // Activation functions
        group.bench_with_input(
            BenchmarkId::new("relu", size),
            &size,
            |b, &size| {
                let a = Tensor::randn(&[size], DType::F32, -1.0, 1.0);
                
                b.iter(|| {
                    let _c = black_box(a.relu());
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("softmax", size),
            &size,
            |b, &size| {
                let a = Tensor::randn(&[size], DType::F32, 0.0, 1.0);
                
                b.iter(|| {
                    let _c = black_box(a.softmax(-1));
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark for reduction operations
fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions");
    
    let sizes = vec![(100, 100), (1000, 100), (100, 1000), (1000, 1000)];
    
    for (rows, cols) in sizes {
        let size_str = format!("{}x{}", rows, cols);
        
        // Sum reduction
        group.bench_with_input(
            BenchmarkId::new("sum", &size_str),
            &(rows, cols),
            |b, &(rows, cols)| {
                let a = Tensor::randn(&[rows, cols], DType::F32, 0.0, 1.0);
                
                b.iter(|| {
                    let _sum = black_box(a.sum(None));
                });
            }
        );
        
        // Mean reduction
        group.bench_with_input(
            BenchmarkId::new("mean", &size_str),
            &(rows, cols),
            |b, &(rows, cols)| {
                let a = Tensor::randn(&[rows, cols], DType::F32, 0.0, 1.0);
                
                b.iter(|| {
                    let _mean = black_box(a.mean(None));
                });
            }
        );
        
        // Max reduction
        group.bench_with_input(
            BenchmarkId::new("max", &size_str),
            &(rows, cols),
            |b, &(rows, cols)| {
                let a = Tensor::randn(&[rows, cols], DType::F32, 0.0, 1.0);
                
                b.iter(|| {
                    let _max = black_box(a.max(None));
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark for tensor reshaping operations
fn bench_reshape(c: &mut Criterion) {
    let mut group = c.benchmark_group("reshape");
    
    // Different reshape scenarios
    let scenarios = vec![
        ("flatten_2d", vec![1000, 1000], vec![1000000]),
        ("unflatten_1d", vec![1000000], vec![1000, 1000]),
        ("transpose_2d", vec![512, 768], vec![768, 512]),
        ("reshape_3d", vec![100, 100, 100], vec![1000, 1000]),
    ];
    
    for (name, from_shape, to_shape) in scenarios {
        group.bench_function(name, |b| {
            let tensor = Tensor::randn(&from_shape, DType::F32, 0.0, 1.0);
            
            b.iter(|| {
                let _reshaped = black_box(tensor.reshape(&to_shape));
            });
        });
    }
    
    group.finish();
}

/// Benchmark for tensor slicing operations
fn bench_slicing(c: &mut Criterion) {
    let mut group = c.benchmark_group("slicing");
    
    let tensor_2d = Tensor::randn(&[1000, 1000], DType::F32, 0.0, 1.0);
    let tensor_3d = Tensor::randn(&[100, 100, 100], DType::F32, 0.0, 1.0);
    
    // 2D slicing
    group.bench_function("slice_2d_row", |b| {
        b.iter(|| {
            let _slice = black_box(tensor_2d.slice(vec![(0, 100), (0, 1000)]));
        });
    });
    
    group.bench_function("slice_2d_col", |b| {
        b.iter(|| {
            let _slice = black_box(tensor_2d.slice(vec![(0, 1000), (0, 100)]));
        });
    });
    
    group.bench_function("slice_2d_block", |b| {
        b.iter(|| {
            let _slice = black_box(tensor_2d.slice(vec![(100, 200), (100, 200)]));
        });
    });
    
    // 3D slicing
    group.bench_function("slice_3d", |b| {
        b.iter(|| {
            let _slice = black_box(tensor_3d.slice(vec![(10, 20), (10, 20), (10, 20)]));
        });
    });
    
    group.finish();
}

/// Benchmark for batch operations
fn bench_batch_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_ops");
    
    // Batch matrix multiplication
    group.bench_function("batch_matmul", |b| {
        let batch_size = 32;
        let size = 64;
        let a = Tensor::randn(&[batch_size, size, size], DType::F32, 0.0, 1.0);
        let b = Tensor::randn(&[batch_size, size, size], DType::F32, 0.0, 1.0);
        
        b.iter(|| {
            let _c = black_box(a.batch_matmul(&b));
        });
    });
    
    // Batch normalization
    group.bench_function("batch_norm", |b| {
        let batch_size = 32;
        let channels = 256;
        let height = 32;
        let width = 32;
        let input = Tensor::randn(&[batch_size, channels, height, width], DType::F32, 0.0, 1.0);
        
        b.iter(|| {
            let mean = input.mean(Some(vec![0, 2, 3]));
            let var = input.var(Some(vec![0, 2, 3]));
            let _normalized = black_box((&input - &mean) / (&var + 1e-5).sqrt());
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_elementwise,
    bench_reductions,
    bench_reshape,
    bench_slicing,
    bench_batch_ops
);
criterion_main!(benches);