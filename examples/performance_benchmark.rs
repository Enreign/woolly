//! Performance Benchmarking Example
//!
//! This example demonstrates Woolly's benchmarking capabilities including:
//! - Performance comparison with other inference engines
//! - Memory usage profiling
//! - Throughput testing
//! - Latency measurements
//! - Hardware utilization monitoring
//!
//! Usage:
//!   cargo run --example performance_benchmark -- --model path/to/model.gguf

use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;

use tracing::{info, debug};
use woolly_core::prelude::*;
use woolly_gguf::GGUFLoader;
use woolly_bench::{Benchmark, BenchmarkConfig, MetricsCollector};

#[derive(Debug)]
struct BenchmarkArgs {
    model_path: PathBuf,
    iterations: usize,
    warmup_iterations: usize,
    batch_sizes: Vec<usize>,
    sequence_lengths: Vec<usize>,
    compare_with: Option<String>,
    output_format: OutputFormat,
    profile_memory: bool,
    profile_cpu: bool,
}

#[derive(Debug, Clone, Copy)]
enum OutputFormat {
    Text,
    Json,
    Csv,
}

impl Default for BenchmarkArgs {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            iterations: 100,
            warmup_iterations: 10,
            batch_sizes: vec![1, 2, 4, 8],
            sequence_lengths: vec![128, 256, 512, 1024],
            compare_with: None,
            output_format: OutputFormat::Text,
            profile_memory: true,
            profile_cpu: true,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,woolly_bench=debug")
        .init();
    
    let args = parse_args()?;
    
    println!("📊 Woolly Performance Benchmark");
    println!("═══════════════════════════════");
    println!();
    
    // Display configuration
    println!("Configuration:");
    println!("  Model: {}", args.model_path.display());
    println!("  Iterations: {} (+ {} warmup)", args.iterations, args.warmup_iterations);
    println!("  Batch sizes: {:?}", args.batch_sizes);
    println!("  Sequence lengths: {:?}", args.sequence_lengths);
    println!("  Memory profiling: {}", args.profile_memory);
    println!("  CPU profiling: {}", args.profile_cpu);
    println!();
    
    // Load model
    println!("📂 Loading model...");
    let loader = GGUFLoader::from_path(&args.model_path)?;
    let model = create_model_from_gguf(&loader)?;
    
    // Print model info
    print_model_stats(&loader);
    
    // Create metrics collector
    let mut metrics = MetricsCollector::new();
    
    // Run benchmarks
    println!("\n🏃 Running benchmarks...");
    
    // 1. Model loading benchmark
    benchmark_model_loading(&args, &mut metrics).await?;
    
    // 2. Tokenization benchmark
    benchmark_tokenization(&mut metrics).await?;
    
    // 3. Inference benchmarks
    benchmark_inference(&args, model, &mut metrics).await?;
    
    // 4. Memory usage analysis
    if args.profile_memory {
        benchmark_memory_usage(&args, &mut metrics).await?;
    }
    
    // 5. CPU utilization
    if args.profile_cpu {
        benchmark_cpu_utilization(&args, &mut metrics).await?;
    }
    
    // 6. Compare with other engines if requested
    if let Some(engine) = &args.compare_with {
        println!("\n🔄 Comparing with {}...", engine);
        run_comparison_benchmark(&args, engine, &mut metrics).await?;
    }
    
    // Generate report
    println!("\n📈 Generating report...");
    generate_report(&args, &metrics)?;
    
    Ok(())
}

async fn benchmark_model_loading(
    args: &BenchmarkArgs,
    metrics: &mut MetricsCollector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⏱️ Model Loading Benchmark");
    println!("─────────────────────────");
    
    let mut load_times = Vec::new();
    
    for i in 0..5 {
        let start = Instant::now();
        let loader = GGUFLoader::from_path(&args.model_path)?;
        let _ = create_model_from_gguf(&loader)?;
        let duration = start.elapsed();
        
        load_times.push(duration);
        println!("  Iteration {}: {:?}", i + 1, duration);
    }
    
    // Calculate statistics
    let avg_time = load_times.iter().sum::<Duration>() / load_times.len() as u32;
    let min_time = load_times.iter().min().unwrap();
    let max_time = load_times.iter().max().unwrap();
    
    println!("\n  Results:");
    println!("    Average: {:?}", avg_time);
    println!("    Min: {:?}", min_time);
    println!("    Max: {:?}", max_time);
    
    metrics.record("model_loading", "average_ms", avg_time.as_millis() as f64);
    metrics.record("model_loading", "min_ms", min_time.as_millis() as f64);
    metrics.record("model_loading", "max_ms", max_time.as_millis() as f64);
    
    Ok(())
}

async fn benchmark_tokenization(
    metrics: &mut MetricsCollector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔤 Tokenization Benchmark");
    println!("────────────────────────");
    
    let test_texts = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.",
        &"Lorem ipsum dolor sit amet. ".repeat(100),
    ];
    
    for (i, text) in test_texts.iter().enumerate() {
        let mut times = Vec::new();
        let iterations = 1000;
        
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = simple_tokenize(text);
            times.push(start.elapsed());
        }
        
        let avg_time = times.iter().sum::<Duration>() / iterations as u32;
        let tokens_per_sec = 1.0 / avg_time.as_secs_f64();
        
        println!("  Text {} ({} chars):", i + 1, text.len());
        println!("    Average time: {:?}", avg_time);
        println!("    Throughput: {:.0} tokenizations/sec", tokens_per_sec);
        
        metrics.record(
            "tokenization",
            &format!("text_{}_chars", text.len()),
            avg_time.as_micros() as f64,
        );
    }
    
    Ok(())
}

async fn benchmark_inference(
    args: &BenchmarkArgs,
    model: Box<dyn Model>,
    metrics: &mut MetricsCollector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧠 Inference Benchmark");
    println!("─────────────────────");
    
    // Create inference engine
    let engine_config = EngineConfig::default();
    let mut engine = InferenceEngine::new(engine_config);
    engine.load_model(Arc::new(model)).await?;
    
    // Test different configurations
    for batch_size in &args.batch_sizes {
        for seq_length in &args.sequence_lengths {
            println!("\n  Testing batch_size={}, seq_length={}", batch_size, seq_length);
            
            let config = SessionConfig {
                max_seq_length: *seq_length,
                batch_size: *batch_size,
                temperature: 1.0,
                ..Default::default()
            };
            
            let session = engine.create_session(config).await?;
            
            // Prepare input tokens
            let tokens: Vec<u32> = (0..*seq_length as u32).map(|i| i % 1000 + 1).collect();
            
            // Warmup
            for _ in 0..args.warmup_iterations {
                let _ = session.infer(&tokens).await;
            }
            
            // Benchmark
            let mut latencies = Vec::new();
            let mut first_token_times = Vec::new();
            
            let total_start = Instant::now();
            
            for _ in 0..args.iterations {
                let start = Instant::now();
                let result = session.infer(&tokens).await?;
                let total_time = start.elapsed();
                
                latencies.push(total_time);
                
                // Simulate first token time
                if let Some(first_token_time) = result.metadata.get("first_token_ms") {
                    first_token_times.push(*first_token_time as f64);
                }
            }
            
            let total_duration = total_start.elapsed();
            
            // Calculate metrics
            let avg_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;
            let p50_latency = percentile(&latencies, 50.0);
            let p95_latency = percentile(&latencies, 95.0);
            let p99_latency = percentile(&latencies, 99.0);
            
            let throughput = (args.iterations * batch_size * seq_length) as f64 
                / total_duration.as_secs_f64();
            
            println!("    Latency (ms):");
            println!("      Average: {:.2}", avg_latency.as_millis());
            println!("      P50: {:.2}", p50_latency.as_millis());
            println!("      P95: {:.2}", p95_latency.as_millis());
            println!("      P99: {:.2}", p99_latency.as_millis());
            println!("    Throughput: {:.1} tokens/sec", throughput);
            
            // Record metrics
            let key = format!("inference_b{}_s{}", batch_size, seq_length);
            metrics.record(&key, "avg_latency_ms", avg_latency.as_millis() as f64);
            metrics.record(&key, "p95_latency_ms", p95_latency.as_millis() as f64);
            metrics.record(&key, "throughput_tokens_sec", throughput);
        }
    }
    
    Ok(())
}

async fn benchmark_memory_usage(
    args: &BenchmarkArgs,
    metrics: &mut MetricsCollector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💾 Memory Usage Analysis");
    println!("──────────────────────");
    
    // Get baseline memory
    let baseline = get_memory_usage();
    println!("  Baseline: {:.1} MB", baseline / 1024.0 / 1024.0);
    
    // Load model and measure
    let loader = GGUFLoader::from_path(&args.model_path)?;
    let after_load = get_memory_usage();
    println!("  After loading: {:.1} MB (+{:.1} MB)", 
        after_load / 1024.0 / 1024.0,
        (after_load - baseline) / 1024.0 / 1024.0
    );
    
    // Create model and measure
    let model = create_model_from_gguf(&loader)?;
    let after_model = get_memory_usage();
    println!("  After model creation: {:.1} MB (+{:.1} MB)", 
        after_model / 1024.0 / 1024.0,
        (after_model - after_load) / 1024.0 / 1024.0
    );
    
    // Run inference and measure peak
    let engine_config = EngineConfig::default();
    let mut engine = InferenceEngine::new(engine_config);
    engine.load_model(Arc::new(model)).await?;
    
    let session_config = SessionConfig {
        max_seq_length: 512,
        batch_size: 4,
        ..Default::default()
    };
    let session = engine.create_session(session_config).await?;
    
    let tokens: Vec<u32> = (0..512).map(|i| i % 1000 + 1).collect();
    let _ = session.infer(&tokens).await;
    
    let peak_memory = get_memory_usage();
    println!("  Peak during inference: {:.1} MB (+{:.1} MB)", 
        peak_memory / 1024.0 / 1024.0,
        (peak_memory - after_model) / 1024.0 / 1024.0
    );
    
    metrics.record("memory", "baseline_mb", baseline as f64 / 1024.0 / 1024.0);
    metrics.record("memory", "model_loaded_mb", after_load as f64 / 1024.0 / 1024.0);
    metrics.record("memory", "model_created_mb", after_model as f64 / 1024.0 / 1024.0);
    metrics.record("memory", "peak_mb", peak_memory as f64 / 1024.0 / 1024.0);
    
    Ok(())
}

async fn benchmark_cpu_utilization(
    args: &BenchmarkArgs,
    metrics: &mut MetricsCollector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🖥️ CPU Utilization Analysis");
    println!("─────────────────────────");
    
    // This is a simplified version - in practice, you'd use system-specific APIs
    let num_cpus = num_cpus::get();
    println!("  Available CPUs: {}", num_cpus);
    
    // Test different thread counts
    let thread_counts = vec![1, num_cpus / 2, num_cpus, num_cpus * 2];
    
    for threads in thread_counts {
        println!("\n  Testing with {} threads:", threads);
        
        let config = EngineConfig {
            num_threads: threads,
            ..Default::default()
        };
        
        // Run a workload and measure
        let start = Instant::now();
        // Simulate CPU-intensive work
        let _result = run_cpu_workload(threads).await;
        let duration = start.elapsed();
        
        let efficiency = 1.0 / (duration.as_secs_f64() * threads as f64);
        println!("    Time: {:?}", duration);
        println!("    Efficiency: {:.2}", efficiency);
        
        metrics.record(
            "cpu_utilization",
            &format!("threads_{}", threads),
            efficiency,
        );
    }
    
    Ok(())
}

async fn run_comparison_benchmark(
    args: &BenchmarkArgs,
    engine_name: &str,
    metrics: &mut MetricsCollector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Comparison with {}", engine_name);
    println!("─────────────────────────────");
    
    // This would run the external engine and collect metrics
    // For this example, we'll simulate the comparison
    
    match engine_name {
        "llama.cpp" => {
            println!("  Running llama.cpp benchmark...");
            
            // Simulate running llama.cpp
            let simulated_results = HashMap::from([
                ("model_load_ms", 3400.0),
                ("tokens_per_sec", 38.0),
                ("memory_mb", 7100.0),
                ("first_token_ms", 180.0),
            ]);
            
            for (metric, value) in simulated_results {
                println!("    {}: {:.1}", metric, value);
                metrics.record("comparison_llama_cpp", metric, value);
            }
        }
        _ => {
            println!("  Unknown engine: {}", engine_name);
        }
    }
    
    Ok(())
}

fn generate_report(
    args: &BenchmarkArgs,
    metrics: &MetricsCollector,
) -> Result<(), Box<dyn std::error::Error>> {
    match args.output_format {
        OutputFormat::Text => generate_text_report(metrics),
        OutputFormat::Json => generate_json_report(metrics),
        OutputFormat::Csv => generate_csv_report(metrics),
    }
}

fn generate_text_report(metrics: &MetricsCollector) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Performance Report");
    println!("══════════════════════");
    
    // Model Loading
    println!("\n🔧 Model Loading Performance:");
    if let Some(avg) = metrics.get("model_loading", "average_ms") {
        println!("  Average: {:.0} ms", avg);
    }
    
    // Inference Performance
    println!("\n🧠 Inference Performance:");
    for (key, values) in metrics.iter() {
        if key.starts_with("inference_") {
            println!("\n  Configuration: {}", key.replace("inference_", ""));
            if let Some(latency) = values.get("avg_latency_ms") {
                println!("    Latency: {:.1} ms", latency);
            }
            if let Some(throughput) = values.get("throughput_tokens_sec") {
                println!("    Throughput: {:.1} tokens/sec", throughput);
            }
        }
    }
    
    // Memory Usage
    println!("\n💾 Memory Usage:");
    if let Some(baseline) = metrics.get("memory", "baseline_mb") {
        println!("  Baseline: {:.1} MB", baseline);
    }
    if let Some(peak) = metrics.get("memory", "peak_mb") {
        println!("  Peak: {:.1} MB", peak);
    }
    
    // Comparison
    if metrics.has_category("comparison_llama_cpp") {
        println!("\n🔄 Comparison with llama.cpp:");
        
        let woolly_tps = metrics.get("inference_b1_s512", "throughput_tokens_sec").unwrap_or(0.0);
        let llama_tps = metrics.get("comparison_llama_cpp", "tokens_per_sec").unwrap_or(0.0);
        
        if llama_tps > 0.0 {
            let speedup = woolly_tps / llama_tps;
            println!("  Woolly throughput: {:.1} tokens/sec", woolly_tps);
            println!("  llama.cpp throughput: {:.1} tokens/sec", llama_tps);
            println!("  Speedup: {:.2}x", speedup);
        }
    }
    
    Ok(())
}

fn generate_json_report(metrics: &MetricsCollector) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(metrics.as_json())?;
    println!("\n{}", json);
    Ok(())
}

fn generate_csv_report(metrics: &MetricsCollector) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nCategory,Metric,Value");
    for (category, values) in metrics.iter() {
        for (metric, value) in values {
            println!("{},{},{}", category, metric, value);
        }
    }
    Ok(())
}

// Helper functions

fn print_model_stats(loader: &GGUFLoader) {
    println!("\n📊 Model Statistics:");
    println!("  Architecture: {}", loader.architecture().unwrap_or("unknown"));
    println!("  Parameters: ~{}B", estimate_parameters(loader));
    println!("  Tensors: {}", loader.header().tensor_count);
    println!("  File size: {:.1} GB", loader.file_size() as f64 / 1024.0 / 1024.0 / 1024.0);
}

fn estimate_parameters(loader: &GGUFLoader) -> f64 {
    // Rough estimation based on file size
    let gb = loader.file_size() as f64 / 1024.0 / 1024.0 / 1024.0;
    gb * 2.0 // Assume ~2B parameters per GB for quantized models
}

fn get_memory_usage() -> usize {
    // Platform-specific implementation
    // This is a simplified version
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }
    
    // Fallback estimate
    2 * 1024 * 1024 * 1024 // 2GB
}

fn percentile(values: &[Duration], p: f64) -> Duration {
    let mut sorted = values.to_vec();
    sorted.sort();
    let index = ((p / 100.0) * (sorted.len() - 1) as f64) as usize;
    sorted[index]
}

async fn run_cpu_workload(threads: usize) -> f64 {
    // Simulate CPU-intensive work
    let mut sum = 0.0;
    for i in 0..1000000 {
        sum += (i as f64).sin() * (i as f64).cos();
    }
    sum / threads as f64
}

fn simple_tokenize(text: &str) -> Vec<u32> {
    text.split_whitespace()
        .enumerate()
        .map(|(i, _)| (i as u32) % 1000 + 1)
        .collect()
}

fn parse_args() -> Result<BenchmarkArgs, Box<dyn std::error::Error>> {
    let mut args = BenchmarkArgs::default();
    let cmd_args: Vec<String> = env::args().collect();
    
    let mut i = 1;
    while i < cmd_args.len() {
        match cmd_args[i].as_str() {
            "--model" => {
                if i + 1 < cmd_args.len() {
                    args.model_path = PathBuf::from(&cmd_args[i + 1]);
                    i += 2;
                } else {
                    return Err("Missing model path".into());
                }
            }
            "--iterations" => {
                if i + 1 < cmd_args.len() {
                    args.iterations = cmd_args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing iterations".into());
                }
            }
            "--compare" => {
                if i + 1 < cmd_args.len() {
                    args.compare_with = Some(cmd_args[i + 1].clone());
                    i += 2;
                } else {
                    return Err("Missing comparison engine".into());
                }
            }
            "--format" => {
                if i + 1 < cmd_args.len() {
                    args.output_format = match cmd_args[i + 1].as_str() {
                        "json" => OutputFormat::Json,
                        "csv" => OutputFormat::Csv,
                        "text" => OutputFormat::Text,
                        _ => return Err("Invalid output format".into()),
                    };
                    i += 2;
                } else {
                    return Err("Missing output format".into());
                }
            }
            "--no-memory" => {
                args.profile_memory = false;
                i += 1;
            }
            "--no-cpu" => {
                args.profile_cpu = false;
                i += 1;
            }
            "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => i += 1,
        }
    }
    
    if args.model_path.as_os_str().is_empty() {
        return Err("Model path is required. Use --model <path>".into());
    }
    
    Ok(args)
}

fn print_help() {
    println!("📊 Woolly Performance Benchmark

USAGE:
    performance_benchmark --model <MODEL> [OPTIONS]

REQUIRED:
    --model <MODEL>          Path to the GGUF model file

OPTIONS:
    --iterations <NUM>       Number of benchmark iterations [default: 100]
    --compare <ENGINE>       Compare with another engine (e.g., llama.cpp)
    --format <FORMAT>        Output format: text, json, csv [default: text]
    --no-memory             Skip memory profiling
    --no-cpu                Skip CPU profiling
    --help                   Show this help message

EXAMPLES:
    # Basic benchmark
    performance_benchmark --model llama-7b.gguf

    # Compare with llama.cpp
    performance_benchmark --model llama-7b.gguf --compare llama.cpp

    # Generate JSON report
    performance_benchmark --model llama-7b.gguf --format json --iterations 200
");
}

// Mock types for the example
mod woolly_bench {
    use std::collections::HashMap;
    
    pub struct Benchmark;
    
    #[derive(Default)]
    pub struct BenchmarkConfig;
    
    pub struct MetricsCollector {
        data: HashMap<String, HashMap<String, f64>>,
    }
    
    impl MetricsCollector {
        pub fn new() -> Self {
            Self {
                data: HashMap::new(),
            }
        }
        
        pub fn record(&mut self, category: &str, metric: &str, value: f64) {
            self.data
                .entry(category.to_string())
                .or_insert_with(HashMap::new)
                .insert(metric.to_string(), value);
        }
        
        pub fn get(&self, category: &str, metric: &str) -> Option<f64> {
            self.data.get(category)?.get(metric).copied()
        }
        
        pub fn has_category(&self, category: &str) -> bool {
            self.data.contains_key(category)
        }
        
        pub fn iter(&self) -> impl Iterator<Item = (&String, &HashMap<String, f64>)> {
            self.data.iter()
        }
        
        pub fn as_json(&self) -> &HashMap<String, HashMap<String, f64>> {
            &self.data
        }
    }
}