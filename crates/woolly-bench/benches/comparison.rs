use criterion::{criterion_group, criterion_main, Criterion};
use woolly_bench::{
    Benchmark, BenchmarkResult, ComparisonFramework, ComparisonReport,
    run_benchmark_iterations,
};
use std::process::Command;
use std::time::{Duration, Instant};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// External implementation benchmark wrapper
struct ExternalBenchmark {
    name: String,
    command: String,
    args: Vec<String>,
    parser: Box<dyn Fn(&str) -> anyhow::Result<ExternalBenchmarkResult>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExternalBenchmarkResult {
    pub execution_time: Duration,
    pub tokens_per_second: Option<f64>,
    pub memory_usage: Option<usize>,
    pub metadata: serde_json::Value,
}

impl ExternalBenchmark {
    /// Create a benchmark for llama.cpp
    pub fn llama_cpp(executable_path: &str, model_path: &str) -> Self {
        Self {
            name: "llama.cpp".to_string(),
            command: executable_path.to_string(),
            args: vec![
                "-m".to_string(),
                model_path.to_string(),
                "-p".to_string(),
                "Hello, world!".to_string(),
                "-n".to_string(),
                "32".to_string(),
                "--no-display-prompt".to_string(),
            ],
            parser: Box::new(parse_llama_cpp_output),
        }
    }
    
    /// Create a benchmark for another implementation
    pub fn custom(name: &str, command: &str, args: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            command: command.to_string(),
            args,
            parser: Box::new(parse_generic_output),
        }
    }
}

impl Benchmark for ExternalBenchmark {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn run(&mut self) -> anyhow::Result<BenchmarkResult> {
        let iterations = 5;
        let mut times = Vec::new();
        let mut tokens_per_second_samples = Vec::new();
        let mut memory_samples = Vec::new();
        
        for _ in 0..iterations {
            let start = Instant::now();
            
            let output = Command::new(&self.command)
                .args(&self.args)
                .output()?;
            
            let duration = start.elapsed();
            times.push(duration);
            
            let stdout = String::from_utf8_lossy(&output.stdout);
            let result = (self.parser)(&stdout)?;
            
            if let Some(tps) = result.tokens_per_second {
                tokens_per_second_samples.push(tps);
            }
            if let Some(mem) = result.memory_usage {
                memory_samples.push(mem);
            }
        }
        
        let total_time: Duration = times.iter().sum();
        let mean_time = total_time / iterations;
        let min_time = times.iter().min().copied().unwrap_or_default();
        let max_time = times.iter().max().copied().unwrap_or_default();
        
        let avg_tps = if !tokens_per_second_samples.is_empty() {
            Some(tokens_per_second_samples.iter().sum::<f64>() / tokens_per_second_samples.len() as f64)
        } else {
            None
        };
        
        let avg_memory = if !memory_samples.is_empty() {
            Some(memory_samples.iter().sum::<usize>() / memory_samples.len())
        } else {
            None
        };
        
        Ok(BenchmarkResult {
            name: self.name().to_string(),
            iterations,
            total_time,
            mean_time,
            min_time,
            max_time,
            stddev: None,
            throughput: avg_tps,
            metadata: serde_json::json!({
                "command": self.command,
                "args": self.args,
                "avg_memory_bytes": avg_memory,
            }),
        })
    }
}

/// Parse llama.cpp output format
fn parse_llama_cpp_output(output: &str) -> anyhow::Result<ExternalBenchmarkResult> {
    let mut tokens_per_second = None;
    let mut execution_time = Duration::from_secs(0);
    
    // Look for timing information in llama.cpp output
    for line in output.lines() {
        if line.contains("eval time") {
            // Extract timing from format: "llama_print_timings:     eval time = 1234.56 ms"
            if let Some(time_str) = line.split('=').nth(1) {
                if let Some(ms_str) = time_str.trim().split_whitespace().next() {
                    if let Ok(ms) = ms_str.parse::<f64>() {
                        execution_time = Duration::from_secs_f64(ms / 1000.0);
                    }
                }
            }
        }
        
        if line.contains("eval:") && line.contains("tokens/s") {
            // Extract tokens/s from format: "llama_print_timings:     eval: 123.45 tokens/s"
            if let Some(tps_part) = line.split("eval:").nth(1) {
                if let Some(tps_str) = tps_part.trim().split_whitespace().next() {
                    if let Ok(tps) = tps_str.parse::<f64>() {
                        tokens_per_second = Some(tps);
                    }
                }
            }
        }
    }
    
    Ok(ExternalBenchmarkResult {
        execution_time,
        tokens_per_second,
        memory_usage: None,
        metadata: serde_json::json!({
            "implementation": "llama.cpp",
        }),
    })
}

/// Parse generic benchmark output
fn parse_generic_output(output: &str) -> anyhow::Result<ExternalBenchmarkResult> {
    // Try to parse JSON output first
    if let Ok(json_result) = serde_json::from_str::<ExternalBenchmarkResult>(output) {
        return Ok(json_result);
    }
    
    // Fallback to basic parsing
    let mut execution_time = Duration::from_secs(0);
    
    // Look for common timing patterns
    for line in output.lines() {
        if line.contains("time:") || line.contains("duration:") {
            if let Some(time_str) = line.split(':').nth(1) {
                let time_str = time_str.trim();
                if time_str.ends_with("ms") {
                    if let Ok(ms) = time_str.trim_end_matches("ms").trim().parse::<f64>() {
                        execution_time = Duration::from_secs_f64(ms / 1000.0);
                    }
                } else if time_str.ends_with("s") {
                    if let Ok(s) = time_str.trim_end_matches("s").trim().parse::<f64>() {
                        execution_time = Duration::from_secs_f64(s);
                    }
                }
            }
        }
    }
    
    Ok(ExternalBenchmarkResult {
        execution_time,
        tokens_per_second: None,
        memory_usage: None,
        metadata: serde_json::json!({}),
    })
}

/// Woolly implementation benchmark
struct WoollyBenchmark {
    model_path: PathBuf,
    prompt: String,
    max_tokens: usize,
}

impl WoollyBenchmark {
    pub fn new(model_path: impl AsRef<Path>, prompt: &str, max_tokens: usize) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            prompt: prompt.to_string(),
            max_tokens,
        }
    }
}

impl Benchmark for WoollyBenchmark {
    fn name(&self) -> &str {
        "woolly"
    }
    
    fn run(&mut self) -> anyhow::Result<BenchmarkResult> {
        run_benchmark_iterations("woolly_inference", 5, || {
            // TODO: Implement actual Woolly inference when available
            // For now, simulate with mock timing
            std::thread::sleep(Duration::from_millis(100));
            Ok(())
        })
    }
}

/// Comparison scenarios
struct ComparisonScenario {
    pub name: String,
    pub description: String,
    pub benchmarks: Vec<Box<dyn Benchmark>>,
}

impl ComparisonScenario {
    /// Text generation comparison
    pub fn text_generation(woolly_model: &Path, llama_cpp_path: Option<&str>) -> Self {
        let mut benchmarks: Vec<Box<dyn Benchmark>> = vec![
            Box::new(WoollyBenchmark::new(woolly_model, "Hello, world!", 32)),
        ];
        
        if let Some(llama_path) = llama_cpp_path {
            benchmarks.push(Box::new(ExternalBenchmark::llama_cpp(
                llama_path,
                woolly_model.to_str().unwrap(),
            )));
        }
        
        Self {
            name: "text_generation".to_string(),
            description: "Compare text generation performance".to_string(),
            benchmarks,
        }
    }
    
    /// Model loading comparison
    pub fn model_loading(woolly_model: &Path, llama_cpp_path: Option<&str>) -> Self {
        let mut benchmarks: Vec<Box<dyn Benchmark>> = vec![
            Box::new(ModelLoadingBenchmark::woolly(woolly_model)),
        ];
        
        if let Some(llama_path) = llama_cpp_path {
            benchmarks.push(Box::new(ModelLoadingBenchmark::llama_cpp(
                llama_path,
                woolly_model,
            )));
        }
        
        Self {
            name: "model_loading".to_string(),
            description: "Compare model loading times".to_string(),
            benchmarks,
        }
    }
}

/// Model loading benchmark wrapper
struct ModelLoadingBenchmark {
    name: String,
    loader: Box<dyn Fn() -> anyhow::Result<Duration>>,
}

impl ModelLoadingBenchmark {
    pub fn woolly(model_path: &Path) -> Self {
        let path = model_path.to_path_buf();
        Self {
            name: "woolly_loading".to_string(),
            loader: Box::new(move || {
                let start = Instant::now();
                // TODO: Implement actual model loading
                std::thread::sleep(Duration::from_millis(50));
                Ok(start.elapsed())
            }),
        }
    }
    
    pub fn llama_cpp(executable: &str, model_path: &Path) -> Self {
        let exe = executable.to_string();
        let model = model_path.to_path_buf();
        Self {
            name: "llama_cpp_loading".to_string(),
            loader: Box::new(move || {
                let start = Instant::now();
                let output = Command::new(&exe)
                    .args(&[
                        "-m",
                        model.to_str().unwrap(),
                        "-p",
                        "test",
                        "-n",
                        "1",
                        "--no-display-prompt",
                    ])
                    .output()?;
                
                if !output.status.success() {
                    anyhow::bail!("llama.cpp execution failed");
                }
                
                Ok(start.elapsed())
            }),
        }
    }
}

impl Benchmark for ModelLoadingBenchmark {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn run(&mut self) -> anyhow::Result<BenchmarkResult> {
        let iterations = 3;
        let mut times = Vec::new();
        
        for _ in 0..iterations {
            let duration = (self.loader)()?;
            times.push(duration);
        }
        
        let total_time: Duration = times.iter().sum();
        let mean_time = total_time / iterations;
        let min_time = times.iter().min().copied().unwrap_or_default();
        let max_time = times.iter().max().copied().unwrap_or_default();
        
        Ok(BenchmarkResult {
            name: self.name().to_string(),
            iterations,
            total_time,
            mean_time,
            min_time,
            max_time,
            stddev: None,
            throughput: None,
            metadata: serde_json::json!({}),
        })
    }
}

/// Run comparison benchmarks
fn run_comparisons(c: &mut Criterion) {
    // This is a framework demonstration
    // In practice, you would configure this with actual model paths
    
    let mut framework = ComparisonFramework::new();
    
    // Add mock benchmarks for demonstration
    framework.add_benchmark(Box::new(WoollyBenchmark::new(
        "models/test.gguf",
        "Test prompt",
        32,
    )));
    
    // You can add external benchmarks when paths are configured
    // framework.add_benchmark(Box::new(ExternalBenchmark::llama_cpp(
    //     "/path/to/llama.cpp/main",
    //     "models/test.gguf",
    // )));
    
    // Run comparisons
    if let Err(e) = framework.run_all() {
        eprintln!("Error running comparisons: {}", e);
        return;
    }
    
    // Generate and print report
    let report = framework.generate_report();
    println!("{}", report.to_markdown());
    
    // Save results
    if let Err(e) = framework.save_results(Path::new("comparison_results.json")) {
        eprintln!("Error saving results: {}", e);
    }
}

/// Benchmark group for criterion integration
fn comparison_benchmarks(c: &mut Criterion) {
    // Run the comparison framework
    run_comparisons(c);
    
    // You can also add specific criterion benchmarks here
    let mut group = c.benchmark_group("implementation_comparison");
    
    group.bench_function("mock_woolly", |b| {
        b.iter(|| {
            // Mock Woolly execution
            std::thread::sleep(Duration::from_micros(100));
        });
    });
    
    group.bench_function("mock_external", |b| {
        b.iter(|| {
            // Mock external implementation
            std::thread::sleep(Duration::from_micros(150));
        });
    });
    
    group.finish();
}

criterion_group!(benches, comparison_benchmarks);
criterion_main!(benches);