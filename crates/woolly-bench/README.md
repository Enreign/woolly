# Woolly Benchmarking Framework

A comprehensive benchmarking framework for Woolly that provides tensor operation benchmarks, model loading benchmarks, and comparison capabilities with external implementations like llama.cpp.

## Features

- **Tensor Operation Benchmarks**: Matrix multiplication, element-wise operations, reductions, reshaping, and more
- **Model Loading Benchmarks**: GGUF file loading, configuration parsing, memory mapping
- **Comparison Framework**: Compare Woolly performance with external implementations
- **Flexible Profiles**: Quick, standard, and comprehensive benchmark profiles
- **CLI Tool**: Easy-to-use command-line interface for running benchmarks
- **Multiple Output Formats**: JSON, Markdown, and table formats for results

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
woolly-bench = { path = "crates/woolly-bench" }
```

## Quick Start

### Using the CLI Tool

```bash
# List available benchmarks
cargo run --bin woolly-bench list

# Run tensor operation benchmarks
cargo run --bin woolly-bench tensor --profile standard

# Run model loading benchmarks
cargo run --bin woolly-bench model --model-path models/test.gguf

# Run comparison benchmarks
cargo run --bin woolly-bench compare \
  --woolly-model models/test.gguf \
  --llama-cpp /path/to/llama.cpp/main

# Run all benchmarks
cargo run --bin woolly-bench all --model-path models/test.gguf --profile comprehensive

# Show results
cargo run --bin woolly-bench show --results-file bench_results/results_20241224_120000.json --format table
```

### Using Criterion Benchmarks

```bash
# Run tensor operation benchmarks with Criterion
cargo bench --bench tensor_ops

# Run model loading benchmarks
cargo bench --bench model_loading

# Run comparison benchmarks
cargo bench --bench comparison
```

## Benchmark Profiles

### Quick Profile
- **Purpose**: Fast development and CI testing
- **Tensor sizes**: 100, 1000
- **Batch sizes**: 1, 8
- **Iterations**: 10
- **Warmup**: 2 iterations

### Standard Profile
- **Purpose**: Regular benchmarking and performance tracking
- **Tensor sizes**: 100, 1000, 10000
- **Batch sizes**: 1, 8, 32
- **Iterations**: 100
- **Warmup**: 10 iterations

### Comprehensive Profile
- **Purpose**: Thorough performance analysis
- **Tensor sizes**: 100, 1000, 10000, 100000
- **Batch sizes**: 1, 8, 32, 128
- **Iterations**: 1000
- **Warmup**: 100 iterations

## Available Benchmarks

### Tensor Operations
- Matrix multiplication (various sizes)
- Element-wise operations (add, multiply, relu, softmax)
- Reduction operations (sum, mean, max)
- Reshape and transpose operations
- Tensor slicing and indexing
- Batch operations

### Model Operations
- GGUF file loading and parsing
- Model configuration parsing
- Memory mapping operations
- Tensor metadata parsing
- Model initialization
- Checkpoint loading patterns

### Comparison Scenarios
- Text generation performance
- Model loading time comparison
- Inference speed comparison
- Memory usage comparison

## Using the Framework Programmatically

### Basic Usage

```rust
use woolly_bench::{Benchmark, BenchmarkResult, run_benchmark_iterations};

// Create a custom benchmark
struct MyBenchmark;

impl Benchmark for MyBenchmark {
    fn name(&self) -> &str {
        "my_benchmark"
    }
    
    fn run(&mut self) -> anyhow::Result<BenchmarkResult> {
        run_benchmark_iterations("my_operation", 100, || {
            // Your benchmark code here
            Ok(())
        })
    }
}

// Run the benchmark
let mut benchmark = MyBenchmark;
let result = benchmark.run()?;
println!("Mean time: {:?}", result.mean_time);
```

### Using the Comparison Framework

```rust
use woolly_bench::ComparisonFramework;

let mut framework = ComparisonFramework::new();

// Add benchmarks
framework.add_benchmark(Box::new(WoollyBenchmark::new()));
framework.add_benchmark(Box::new(ExternalBenchmark::llama_cpp("/path/to/llama.cpp", "model.gguf")));

// Run comparisons
framework.run_all()?;

// Generate report
let report = framework.generate_report();
println!("{}", report.to_markdown());

// Save results
framework.save_results("comparison_results.json")?;
```

### Using the Benchmark Runner

```rust
use woolly_bench::runner::{BenchmarkRunner, BenchmarkProfile};

let mut runner = BenchmarkRunner::new("output_dir");

// Add benchmarks
runner.add_benchmark(Box::new(MyBenchmark));

// Run all benchmarks
runner.run_all().await?;

// Results are automatically saved to the output directory
```

## External Implementation Comparison

The framework supports comparing Woolly with external implementations:

### llama.cpp Integration

```rust
use woolly_bench::ExternalBenchmark;

let llama_bench = ExternalBenchmark::llama_cpp(
    "/path/to/llama.cpp/main",
    "/path/to/model.gguf"
);
```

### Custom External Implementation

```rust
let custom_bench = ExternalBenchmark::custom(
    "my_implementation",
    "/path/to/executable",
    vec!["--model", "model.gguf", "--prompt", "Hello"]
);
```

## Output Formats

### JSON Results
```json
{
  "name": "matrix_multiplication",
  "iterations": 100,
  "total_time": {"secs": 1, "nanos": 500000000},
  "mean_time": {"secs": 0, "nanos": 15000000},
  "min_time": {"secs": 0, "nanos": 12000000},
  "max_time": {"secs": 0, "nanos": 18000000},
  "stddev": 2000000.0,
  "throughput": 66.67,
  "metadata": {"tensor_size": 1000}
}
```

### Markdown Report
```markdown
# Benchmark Comparison Report

**Fastest**: woolly
**Slowest**: llama.cpp

## Results

| Benchmark | Mean Time | Min Time | Max Time | Relative to Fastest |
|-----------|-----------|----------|----------|---------------------|
| woolly    | 15.00ms   | 12.00ms  | 18.00ms  | 1.00x               |
| llama.cpp | 22.50ms   | 20.00ms  | 25.00ms  | 1.50x               |
```

## Configuration

### Environment Variables

- `WOOLLY_BENCH_OUTPUT_DIR`: Default output directory for results
- `WOOLLY_BENCH_PROFILE`: Default benchmark profile
- `WOOLLY_BENCH_LLAMA_CPP_PATH`: Path to llama.cpp executable for comparisons

### Configuration File

Create a `woolly-bench.toml` file for persistent configuration:

```toml
[benchmark]
default_profile = "standard"
output_dir = "bench_results"

[comparison]
llama_cpp_path = "/usr/local/bin/llama.cpp"
enable_external_comparisons = true

[tensor]
default_sizes = [100, 1000, 10000]
default_batch_sizes = [1, 8, 32]

[model]
test_model_path = "models/test.gguf"
```

## Contributing

To add new benchmarks:

1. Implement the `Benchmark` trait for your benchmark
2. Add it to the appropriate benchmark file in `benches/`
3. Update the CLI tool to include your benchmark
4. Add tests for your benchmark

Example:

```rust
struct NewOperationBenchmark {
    // benchmark configuration
}

impl Benchmark for NewOperationBenchmark {
    fn name(&self) -> &str {
        "new_operation"
    }
    
    fn run(&mut self) -> anyhow::Result<BenchmarkResult> {
        // benchmark implementation
    }
}
```

## Performance Tips

1. **Use appropriate profiles**: Quick for development, comprehensive for releases
2. **Warm up properly**: Ensure consistent results by warming up operations
3. **Monitor system resources**: Check CPU and memory usage during benchmarks
4. **Run multiple times**: Average results across multiple runs for reliability
5. **Control environment**: Minimize system load during benchmarking

## License

This project is licensed under MIT OR Apache-2.0.