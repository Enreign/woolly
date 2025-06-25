use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use woolly_bench::{Benchmark, BenchmarkResult, run_benchmark_iterations};
use woolly_gguf::{GgufFile, GgufTensor};
use woolly_core::ModelConfig;
use std::path::Path;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use std::fs;

/// Create a mock GGUF file for benchmarking
fn create_mock_gguf_file(path: &Path, tensor_count: usize, tensor_size: usize) -> anyhow::Result<()> {
    use std::io::Write;
    
    // Create a simple mock GGUF file structure
    // This is a simplified version for benchmarking purposes
    let mut file = fs::File::create(path)?;
    
    // Write mock header
    file.write_all(b"GGUF")?; // Magic
    file.write_all(&3u32.to_le_bytes())?; // Version
    file.write_all(&(tensor_count as u64).to_le_bytes())?; // Tensor count
    file.write_all(&0u64.to_le_bytes())?; // Metadata KV count
    
    // Write mock tensors
    for i in 0..tensor_count {
        // Tensor name
        let name = format!("tensor_{}", i);
        file.write_all(&(name.len() as u32).to_le_bytes())?;
        file.write_all(name.as_bytes())?;
        
        // Tensor dimensions (2D for simplicity)
        file.write_all(&2u32.to_le_bytes())?; // n_dims
        file.write_all(&(tensor_size as u64).to_le_bytes())?; // dim 0
        file.write_all(&(tensor_size as u64).to_le_bytes())?; // dim 1
        
        // Tensor type (F32)
        file.write_all(&0u32.to_le_bytes())?;
        
        // Tensor offset
        file.write_all(&0u64.to_le_bytes())?;
    }
    
    // Write mock tensor data
    let data_size = tensor_count * tensor_size * tensor_size * 4; // F32 = 4 bytes
    let data = vec![0u8; data_size];
    file.write_all(&data)?;
    
    Ok(())
}

/// Benchmark for loading GGUF files
struct GgufLoadingBench {
    temp_dir: TempDir,
    file_sizes: Vec<(usize, usize)>, // (tensor_count, tensor_size)
}

impl GgufLoadingBench {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            temp_dir: TempDir::new()?,
            file_sizes: vec![
                (10, 128),      // Small model
                (50, 256),      // Medium model
                (100, 512),     // Large model
                (200, 1024),    // XL model
            ],
        })
    }
}

impl Benchmark for GgufLoadingBench {
    fn name(&self) -> &str {
        "GGUF File Loading"
    }
    
    fn warmup(&mut self) -> anyhow::Result<()> {
        // Create mock files
        for (i, &(tensor_count, tensor_size)) in self.file_sizes.iter().enumerate() {
            let path = self.temp_dir.path().join(format!("model_{}.gguf", i));
            create_mock_gguf_file(&path, tensor_count, tensor_size)?;
        }
        Ok(())
    }
    
    fn run(&mut self) -> anyhow::Result<BenchmarkResult> {
        let mut results = Vec::new();
        
        for (i, &(tensor_count, tensor_size)) in self.file_sizes.iter().enumerate() {
            let path = self.temp_dir.path().join(format!("model_{}.gguf", i));
            let iterations = 10;
            
            let result = run_benchmark_iterations(
                &format!("load_gguf_{}tensors_{}size", tensor_count, tensor_size),
                iterations,
                || {
                    let _file = GgufFile::open(&path)?;
                    Ok(())
                }
            )?;
            
            results.push(result);
        }
        
        // Aggregate results
        let total_time: Duration = results.iter().map(|r| r.total_time).sum();
        let mean_time = total_time / results.len() as u32;
        
        Ok(BenchmarkResult {
            name: self.name().to_string(),
            iterations: results.iter().map(|r| r.iterations).sum(),
            total_time,
            mean_time,
            min_time: results.iter().map(|r| r.min_time).min().unwrap_or_default(),
            max_time: results.iter().map(|r| r.max_time).max().unwrap_or_default(),
            stddev: None,
            throughput: None,
            metadata: serde_json::json!({
                "file_sizes": self.file_sizes,
            }),
        })
    }
}

/// Benchmark for model configuration parsing
fn bench_config_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_parsing");
    
    // Different config sizes
    let configs = vec![
        ("small", serde_json::json!({
            "model_type": "llama",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "vocab_size": 32000,
        })),
        ("medium", serde_json::json!({
            "model_type": "llama",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "vocab_size": 32000,
            "intermediate_size": 5504,
            "max_position_embeddings": 2048,
        })),
        ("large", serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "norm_eps": 1e-5,
        })),
    ];
    
    for (name, config) in configs {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &config,
            |b, config| {
                let config_str = config.to_string();
                b.iter(|| {
                    let _parsed: ModelConfig = black_box(
                        serde_json::from_str(&config_str).unwrap()
                    );
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark for memory mapping operations
fn bench_memory_mapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_mapping");
    
    // Create temporary files of different sizes
    let temp_dir = TempDir::new().unwrap();
    let file_sizes = vec![
        ("1MB", 1024 * 1024),
        ("10MB", 10 * 1024 * 1024),
        ("100MB", 100 * 1024 * 1024),
        ("1GB", 1024 * 1024 * 1024),
    ];
    
    // Create files
    for (name, size) in &file_sizes {
        let path = temp_dir.path().join(format!("file_{}", name));
        let data = vec![0u8; *size];
        fs::write(&path, data).unwrap();
    }
    
    // Benchmark memory mapping
    for (name, _size) in &file_sizes {
        let path = temp_dir.path().join(format!("file_{}", name));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &path,
            |b, path| {
                b.iter(|| {
                    let file = fs::File::open(path).unwrap();
                    let _mmap = unsafe {
                        black_box(memmap2::Mmap::map(&file).unwrap())
                    };
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark for tensor metadata parsing
fn bench_tensor_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_metadata");
    
    // Different numbers of tensors
    let tensor_counts = vec![10, 100, 1000, 10000];
    
    for count in tensor_counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &count,
            |b, &count| {
                // Create mock tensor metadata
                let metadata: Vec<_> = (0..count)
                    .map(|i| serde_json::json!({
                        "name": format!("layer_{}.weight", i),
                        "shape": [768, 768],
                        "dtype": "f32",
                        "offset": i * 768 * 768 * 4,
                    }))
                    .collect();
                
                let metadata_str = serde_json::to_string(&metadata).unwrap();
                
                b.iter(|| {
                    let _parsed: Vec<serde_json::Value> = black_box(
                        serde_json::from_str(&metadata_str).unwrap()
                    );
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark for model initialization
fn bench_model_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_init");
    
    // Mock model sizes
    let model_sizes = vec![
        ("small", 125_000_000),   // 125M parameters
        ("medium", 355_000_000),  // 355M parameters
        ("large", 1_300_000_000), // 1.3B parameters
        ("xl", 7_000_000_000),    // 7B parameters
    ];
    
    for (name, param_count) in model_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &param_count,
            |b, &param_count| {
                b.iter(|| {
                    // Simulate model initialization
                    // In real scenario, this would allocate tensors
                    let tensor_size = 768 * 768; // Typical layer size
                    let num_tensors = param_count / tensor_size;
                    
                    let mut tensors = Vec::with_capacity(num_tensors);
                    for i in 0..num_tensors {
                        tensors.push(black_box(format!("tensor_{}", i)));
                    }
                    
                    black_box(tensors);
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark for checkpoint loading patterns
fn bench_checkpoint_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_patterns");
    
    // Simulate different loading patterns
    group.bench_function("sequential_load", |b| {
        b.iter(|| {
            // Simulate sequential tensor loading
            for i in 0..100 {
                let _tensor = black_box(vec![0.0f32; 1024 * 1024]);
                std::thread::sleep(Duration::from_micros(10)); // Simulate I/O
            }
        });
    });
    
    group.bench_function("parallel_load", |b| {
        use std::sync::Arc;
        use std::sync::Mutex;
        
        b.iter(|| {
            let tensors = Arc::new(Mutex::new(Vec::new()));
            let handles: Vec<_> = (0..4)
                .map(|thread_id| {
                    let tensors = Arc::clone(&tensors);
                    std::thread::spawn(move || {
                        for i in 0..25 {
                            let tensor = vec![0.0f32; 1024 * 1024];
                            tensors.lock().unwrap().push((thread_id * 25 + i, tensor));
                            std::thread::sleep(Duration::from_micros(10)); // Simulate I/O
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            black_box(tensors);
        });
    });
    
    group.bench_function("lazy_load", |b| {
        b.iter(|| {
            // Simulate lazy loading with metadata only
            let metadata: Vec<_> = (0..100)
                .map(|i| {
                    black_box((
                        format!("tensor_{}", i),
                        vec![1024, 1024], // shape
                        i * 1024 * 1024 * 4, // offset
                    ))
                })
                .collect();
            
            // Simulate loading only specific tensors
            for i in vec![0, 10, 20, 30, 40] {
                if i < metadata.len() {
                    let _tensor = black_box(vec![0.0f32; 1024 * 1024]);
                    std::thread::sleep(Duration::from_micros(10)); // Simulate I/O
                }
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_config_parsing,
    bench_memory_mapping,
    bench_tensor_metadata,
    bench_model_init,
    bench_checkpoint_patterns
);
criterion_main!(benches);