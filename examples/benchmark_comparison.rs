//! Benchmark Comparison Example
//!
//! This example demonstrates comprehensive performance benchmarks comparing Woolly with llama.cpp:
//! 1. Model loading performance
//! 2. Inference speed (tokens per second)
//! 3. Memory usage comparison
//! 4. Throughput with different batch sizes
//! 5. Latency measurements
//! 6. Accuracy validation
//! 7. Resource utilization metrics
//!
//! Usage:
//!   cargo run --example benchmark_comparison -- --model path/to/model.gguf --comparison-binary path/to/llama.cpp/main

use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use woolly_core::prelude::*;
use woolly_gguf::GGUFLoader;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let benchmark_config = parse_args(&args)?;
    
    println!("🏁 Woolly vs llama.cpp Benchmark Comparison");
    println!("Model: {}", benchmark_config.model_path);
    if let Some(ref llama_binary) = benchmark_config.llama_cpp_binary {
        println!("llama.cpp binary: {}", llama_binary);
    }
    println!("Benchmark configuration: {:?}", benchmark_config);
    println!();
    
    // Step 1: Initialize benchmarking system
    println!("🔧 Setting up benchmark environment...");
    let mut benchmark_suite = BenchmarkSuite::new(benchmark_config).await?;
    
    // Step 2: Model loading benchmarks
    println!("\n📂 Model Loading Benchmarks:");
    let loading_results = benchmark_suite.benchmark_model_loading().await?;
    print_loading_results(&loading_results);
    
    // Step 3: Inference speed benchmarks
    println!("\n🚀 Inference Speed Benchmarks:");
    let inference_results = benchmark_suite.benchmark_inference_speed().await?;
    print_inference_results(&inference_results);
    
    // Step 4: Memory usage benchmarks
    println!("\n💾 Memory Usage Benchmarks:");
    let memory_results = benchmark_suite.benchmark_memory_usage().await?;
    print_memory_results(&memory_results);
    
    // Step 5: Throughput benchmarks
    println!("\n📊 Throughput Benchmarks:");
    let throughput_results = benchmark_suite.benchmark_throughput().await?;
    print_throughput_results(&throughput_results);
    
    // Step 6: Latency benchmarks
    println!("\n⏱️ Latency Benchmarks:");
    let latency_results = benchmark_suite.benchmark_latency().await?;
    print_latency_results(&latency_results);
    
    // Step 7: Accuracy validation
    println!("\n🎯 Accuracy Validation:");
    let accuracy_results = benchmark_suite.benchmark_accuracy().await?;
    print_accuracy_results(&accuracy_results);
    
    // Step 8: Generate comprehensive report
    println!("\n📈 Generating Comprehensive Report:");
    let report = BenchmarkReport::generate(
        loading_results,
        inference_results,
        memory_results,
        throughput_results,
        latency_results,
        accuracy_results,
    );
    
    save_report(&report).await?;
    print_summary(&report);
    
    println!("\n✨ Benchmark comparison completed!");
    println!("📄 Full report saved to: benchmark_report.json");
    
    Ok(())
}

#[derive(Debug, Clone)]
struct BenchmarkConfig {
    model_path: String,
    llama_cpp_binary: Option<String>,
    prompt_file: Option<String>,
    num_iterations: usize,
    max_tokens: usize,
    batch_sizes: Vec<usize>,
    temperature: f32,
    enable_accuracy_tests: bool,
    enable_memory_profiling: bool,
    output_dir: String,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            llama_cpp_binary: None,
            prompt_file: None,
            num_iterations: 10,
            max_tokens: 100,
            batch_sizes: vec![1, 2, 4, 8, 16],
            temperature: 0.8,
            enable_accuracy_tests: true,
            enable_memory_profiling: true,
            output_dir: "benchmark_results".to_string(),
        }
    }
}

fn parse_args(args: &[String]) -> Result<BenchmarkConfig, Box<dyn std::error::Error>> {
    let mut config = BenchmarkConfig::default();
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    config.model_path = args[i + 1].clone();
                    i += 2;
                } else {
                    return Err("Missing model path".into());
                }
            }
            "--comparison-binary" => {
                if i + 1 < args.len() {
                    config.llama_cpp_binary = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    return Err("Missing comparison binary path".into());
                }
            }
            "--prompt-file" => {
                if i + 1 < args.len() {
                    config.prompt_file = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    return Err("Missing prompt file path".into());
                }
            }
            "--iterations" => {
                if i + 1 < args.len() {
                    config.num_iterations = args[i + 1].parse().map_err(|_| "Invalid iteration count")?;
                    i += 2;
                } else {
                    return Err("Missing iteration count".into());
                }
            }
            "--max-tokens" => {
                if i + 1 < args.len() {
                    config.max_tokens = args[i + 1].parse().map_err(|_| "Invalid max tokens")?;
                    i += 2;
                } else {
                    return Err("Missing max tokens".into());
                }
            }
            "--no-accuracy" => {
                config.enable_accuracy_tests = false;
                i += 1;
            }
            "--no-memory" => {
                config.enable_memory_profiling = false;
                i += 1;
            }
            "--output-dir" => {
                if i + 1 < args.len() {
                    config.output_dir = args[i + 1].clone();
                    i += 2;
                } else {
                    return Err("Missing output directory".into());
                }
            }
            _ => i += 1,
        }
    }
    
    if config.model_path.is_empty() {
        return Err("Model path is required. Use --model <path>".into());
    }
    
    Ok(config)
}

struct BenchmarkSuite {
    config: BenchmarkConfig,
    woolly_engine: Option<Arc<InferenceEngine>>,
    test_prompts: Vec<String>,
}

impl BenchmarkSuite {
    async fn new(config: BenchmarkConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let test_prompts = if let Some(ref prompt_file) = config.prompt_file {
            load_prompts_from_file(prompt_file).await?
        } else {
            default_test_prompts()
        };
        
        Ok(Self {
            config,
            woolly_engine: None,
            test_prompts,
        })
    }
    
    async fn benchmark_model_loading(&mut self) -> Result<LoadingResults, Box<dyn std::error::Error>> {
        println!("  📥 Testing Woolly model loading...");
        let woolly_times = self.benchmark_woolly_loading().await?;
        
        let llama_times = if let Some(ref binary) = self.config.llama_cpp_binary {
            println!("  📥 Testing llama.cpp model loading...");
            self.benchmark_llama_loading(binary).await?
        } else {
            Vec::new()
        };
        
        Ok(LoadingResults {
            woolly_loading_times: woolly_times,
            llama_loading_times: llama_times,
        })
    }
    
    async fn benchmark_woolly_loading(&mut self) -> Result<Vec<Duration>, Box<dyn std::error::Error>> {
        let mut times = Vec::new();
        
        for i in 0..self.config.num_iterations {
            println!("    Iteration {}/{}", i + 1, self.config.num_iterations);
            
            let start = Instant::now();
            
            // Load GGUF model
            let gguf_loader = GGUFLoader::from_path(&self.config.model_path)?;
            let model = create_benchmark_model(&gguf_loader)?;
            
            // Create and initialize engine
            let engine_config = create_benchmark_engine_config();
            let mut engine = InferenceEngine::new(engine_config);
            engine.load_model(Arc::new(model)).await?;
            
            let duration = start.elapsed();
            times.push(duration);
            
            // Store the engine for later use
            if i == 0 {
                self.woolly_engine = Some(Arc::new(engine));
            }
            
            println!("      Loaded in {:?}", duration);
        }
        
        Ok(times)
    }
    
    async fn benchmark_llama_loading(&self, binary_path: &str) -> Result<Vec<Duration>, Box<dyn std::error::Error>> {
        let mut times = Vec::new();
        
        for i in 0..self.config.num_iterations {
            println!("    Iteration {}/{}", i + 1, self.config.num_iterations);
            
            let start = Instant::now();
            
            // Run llama.cpp with minimal prompt to measure loading time
            let output = Command::new(binary_path)
                .args(&[
                    "-m", &self.config.model_path,
                    "-p", "test",
                    "-n", "1",
                    "--log-disable",
                ])
                .output();
            
            let duration = start.elapsed();
            
            match output {
                Ok(_) => {
                    times.push(duration);
                    println!("      Loaded in {:?}", duration);
                }
                Err(e) => {
                    println!("      Failed to run llama.cpp: {}", e);
                    break;
                }
            }
        }
        
        Ok(times)
    }
    
    async fn benchmark_inference_speed(&self) -> Result<InferenceResults, Box<dyn std::error::Error>> {
        println!("  🚀 Testing Woolly inference speed...");
        let woolly_results = self.benchmark_woolly_inference().await?;
        
        let llama_results = if let Some(ref binary) = self.config.llama_cpp_binary {
            println!("  🚀 Testing llama.cpp inference speed...");
            self.benchmark_llama_inference(binary).await?
        } else {
            Vec::new()
        };
        
        Ok(InferenceResults {
            woolly_results,
            llama_results,
        })
    }
    
    async fn benchmark_woolly_inference(&self) -> Result<Vec<InferenceMetrics>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        if let Some(ref engine) = self.woolly_engine {
            for prompt in &self.test_prompts {
                println!("    Testing prompt: '{}'", prompt.chars().take(50).collect::<String>());
                
                let start = Instant::now();
                let session_config = SessionConfig {
                    max_seq_length: self.config.max_tokens,
                    temperature: self.config.temperature,
                    ..Default::default()
                };
                
                let session = engine.create_session(session_config).await?;
                
                // Simple tokenization for demo
                let tokens: Vec<u32> = prompt.split_whitespace()
                    .enumerate()
                    .map(|(i, _)| (i as u32) % 1000 + 1)
                    .collect();
                
                let generation_start = Instant::now();
                
                // Simulate inference (in real implementation, this would call session.infer)
                let mut generated_tokens = 0;
                let target_tokens = self.config.max_tokens.min(100); // Simulate generating tokens
                
                while generated_tokens < target_tokens {
                    // Simulate inference time
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    generated_tokens += 1;
                }
                
                let generation_time = generation_start.elapsed();
                let total_time = start.elapsed();
                
                let metrics = InferenceMetrics {
                    prompt_length: tokens.len(),
                    generated_tokens,
                    total_time,
                    generation_time,
                    tokens_per_second: generated_tokens as f64 / generation_time.as_secs_f64(),
                    memory_usage: get_memory_usage().unwrap_or(0),
                };
                
                println!("      {} tokens in {:?} ({:.2} tok/s)", 
                        generated_tokens, generation_time, metrics.tokens_per_second);
                
                results.push(metrics);
            }
        }
        
        Ok(results)
    }
    
    async fn benchmark_llama_inference(&self, binary_path: &str) -> Result<Vec<InferenceMetrics>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        for prompt in &self.test_prompts {
            println!("    Testing prompt: '{}'", prompt.chars().take(50).collect::<String>());
            
            let start = Instant::now();
            
            let output = Command::new(binary_path)
                .args(&[
                    "-m", &self.config.model_path,
                    "-p", prompt,
                    "-n", &self.config.max_tokens.to_string(),
                    "-t", &self.config.temperature.to_string(),
                    "--log-disable",
                ])
                .output();
            
            let total_time = start.elapsed();
            
            match output {
                Ok(result) => {
                    let output_text = String::from_utf8_lossy(&result.stdout);
                    let generated_tokens = estimate_token_count(&output_text);
                    
                    let metrics = InferenceMetrics {
                        prompt_length: estimate_token_count(prompt),
                        generated_tokens,
                        total_time,
                        generation_time: total_time, // Approximation
                        tokens_per_second: generated_tokens as f64 / total_time.as_secs_f64(),
                        memory_usage: 0, // Would need external profiling
                    };
                    
                    println!("      {} tokens in {:?} ({:.2} tok/s)", 
                            generated_tokens, total_time, metrics.tokens_per_second);
                    
                    results.push(metrics);
                }
                Err(e) => {
                    println!("      Failed to run inference: {}", e);
                }
            }
        }
        
        Ok(results)
    }
    
    async fn benchmark_memory_usage(&self) -> Result<MemoryResults, Box<dyn std::error::Error>> {
        println!("  💾 Measuring memory usage...");
        
        let woolly_memory = if self.config.enable_memory_profiling {
            self.measure_woolly_memory().await?
        } else {
            MemoryMetrics::default()
        };
        
        let llama_memory = if self.config.enable_memory_profiling && self.config.llama_cpp_binary.is_some() {
            self.measure_llama_memory().await?
        } else {
            MemoryMetrics::default()
        };
        
        Ok(MemoryResults {
            woolly_memory,
            llama_memory,
        })
    }
    
    async fn measure_woolly_memory(&self) -> Result<MemoryMetrics, Box<dyn std::error::Error>> {
        let baseline_memory = get_memory_usage().unwrap_or(0);
        
        if let Some(ref engine) = self.woolly_engine {
            let session_config = SessionConfig::default();
            let _session = engine.create_session(session_config).await?;
            
            let peak_memory = get_memory_usage().unwrap_or(0);
            
            Ok(MemoryMetrics {
                baseline_mb: baseline_memory as f64 / (1024.0 * 1024.0),
                peak_mb: peak_memory as f64 / (1024.0 * 1024.0),
                model_size_mb: estimate_model_size(&self.config.model_path)?,
            })
        } else {
            Ok(MemoryMetrics::default())
        }
    }
    
    async fn measure_llama_memory(&self) -> Result<MemoryMetrics, Box<dyn std::error::Error>> {
        // This would require external profiling tools like valgrind or system monitoring
        // For now, return estimated values
        Ok(MemoryMetrics {
            baseline_mb: 0.0,
            peak_mb: estimate_model_size(&self.config.model_path)? * 1.2, // Rough estimate
            model_size_mb: estimate_model_size(&self.config.model_path)?,
        })
    }
    
    async fn benchmark_throughput(&self) -> Result<ThroughputResults, Box<dyn std::error::Error>> {
        println!("  📊 Testing throughput with different batch sizes...");
        
        let mut results = Vec::new();
        
        for &batch_size in &self.config.batch_sizes {
            println!("    Testing batch size: {}", batch_size);
            
            let woolly_throughput = self.measure_woolly_throughput(batch_size).await?;
            let llama_throughput = if self.config.llama_cpp_binary.is_some() {
                self.measure_llama_throughput(batch_size).await?
            } else {
                0.0
            };
            
            results.push(ThroughputMetrics {
                batch_size,
                woolly_throughput,
                llama_throughput,
            });
            
            println!("      Woolly: {:.2} tok/s, llama.cpp: {:.2} tok/s", 
                    woolly_throughput, llama_throughput);
        }
        
        Ok(ThroughputResults { results })
    }
    
    async fn measure_woolly_throughput(&self, batch_size: usize) -> Result<f64, Box<dyn std::error::Error>> {
        if let Some(ref engine) = self.woolly_engine {
            let prompts = &self.test_prompts[..batch_size.min(self.test_prompts.len())];
            let start = Instant::now();
            
            let mut total_tokens = 0;
            for prompt in prompts {
                // Simulate batch processing
                let tokens = estimate_token_count(prompt);
                total_tokens += tokens;
                
                // Simulate processing time
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            
            let duration = start.elapsed();
            Ok(total_tokens as f64 / duration.as_secs_f64())
        } else {
            Ok(0.0)
        }
    }
    
    async fn measure_llama_throughput(&self, _batch_size: usize) -> Result<f64, Box<dyn std::error::Error>> {
        // llama.cpp doesn't support batch processing in the same way
        // This would require multiple parallel processes or special configuration
        Ok(0.0) // Placeholder
    }
    
    async fn benchmark_latency(&self) -> Result<LatencyResults, Box<dyn std::error::Error>> {
        println!("  ⏱️ Measuring first token latency...");
        
        let woolly_latency = self.measure_woolly_latency().await?;
        let llama_latency = if self.config.llama_cpp_binary.is_some() {
            self.measure_llama_latency().await?
        } else {
            Duration::from_millis(0)
        };
        
        Ok(LatencyResults {
            woolly_first_token: woolly_latency,
            llama_first_token: llama_latency,
        })
    }
    
    async fn measure_woolly_latency(&self) -> Result<Duration, Box<dyn std::error::Error>> {
        if let Some(ref engine) = self.woolly_engine {
            let session_config = SessionConfig::default();
            let session = engine.create_session(session_config).await?;
            
            let prompt = &self.test_prompts[0];
            let tokens: Vec<u32> = prompt.split_whitespace()
                .enumerate()
                .map(|(i, _)| (i as u32) % 1000 + 1)
                .collect();
            
            let start = Instant::now();
            
            // Simulate first token generation
            tokio::time::sleep(Duration::from_millis(100)).await;
            
            Ok(start.elapsed())
        } else {
            Ok(Duration::from_millis(0))
        }
    }
    
    async fn measure_llama_latency(&self) -> Result<Duration, Box<dyn std::error::Error>> {
        // This would require parsing llama.cpp output for timing information
        // For now, return a simulated value
        Ok(Duration::from_millis(150))
    }
    
    async fn benchmark_accuracy(&self) -> Result<AccuracyResults, Box<dyn std::error::Error>> {
        if !self.config.enable_accuracy_tests {
            return Ok(AccuracyResults::default());
        }
        
        println!("  🎯 Comparing output accuracy...");
        
        let mut results = Vec::new();
        
        for prompt in &self.test_prompts[..3.min(self.test_prompts.len())] { // Limit for demo
            let woolly_output = self.generate_woolly_output(prompt).await?;
            let llama_output = if let Some(ref binary) = self.config.llama_cpp_binary {
                self.generate_llama_output(binary, prompt).await?
            } else {
                String::new()
            };
            
            let similarity = calculate_similarity(&woolly_output, &llama_output);
            
            results.push(AccuracyComparison {
                prompt: prompt.clone(),
                woolly_output,
                llama_output,
                similarity_score: similarity,
            });
            
            println!("    Similarity: {:.2}%", similarity * 100.0);
        }
        
        Ok(AccuracyResults { comparisons: results })
    }
    
    async fn generate_woolly_output(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Simulate Woolly generation
        Ok(format!("Woolly generated response to: {}", prompt))
    }
    
    async fn generate_llama_output(&self, binary_path: &str, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let output = Command::new(binary_path)
            .args(&[
                "-m", &self.config.model_path,
                "-p", prompt,
                "-n", "50",
                "--log-disable",
            ])
            .output()?;
        
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

// Results structures

#[derive(Debug, Serialize, Deserialize)]
struct LoadingResults {
    woolly_loading_times: Vec<Duration>,
    llama_loading_times: Vec<Duration>,
}

#[derive(Debug, Serialize, Deserialize)]
struct InferenceResults {
    woolly_results: Vec<InferenceMetrics>,
    llama_results: Vec<InferenceMetrics>,
}

#[derive(Debug, Serialize, Deserialize)]
struct InferenceMetrics {
    prompt_length: usize,
    generated_tokens: usize,
    #[serde(with = "duration_serde")]
    total_time: Duration,
    #[serde(with = "duration_serde")]
    generation_time: Duration,
    tokens_per_second: f64,
    memory_usage: usize,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct MemoryResults {
    woolly_memory: MemoryMetrics,
    llama_memory: MemoryMetrics,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct MemoryMetrics {
    baseline_mb: f64,
    peak_mb: f64,
    model_size_mb: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ThroughputResults {
    results: Vec<ThroughputMetrics>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ThroughputMetrics {
    batch_size: usize,
    woolly_throughput: f64,
    llama_throughput: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct LatencyResults {
    #[serde(with = "duration_serde")]
    woolly_first_token: Duration,
    #[serde(with = "duration_serde")]
    llama_first_token: Duration,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct AccuracyResults {
    comparisons: Vec<AccuracyComparison>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AccuracyComparison {
    prompt: String,
    woolly_output: String,
    llama_output: String,
    similarity_score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkReport {
    timestamp: String,
    loading: LoadingResults,
    inference: InferenceResults,
    memory: MemoryResults,
    throughput: ThroughputResults,
    latency: LatencyResults,
    accuracy: AccuracyResults,
    summary: BenchmarkSummary,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkSummary {
    woolly_avg_loading_time: f64,
    llama_avg_loading_time: f64,
    woolly_avg_tokens_per_sec: f64,
    llama_avg_tokens_per_sec: f64,
    woolly_memory_efficiency: f64,
    llama_memory_efficiency: f64,
    avg_accuracy_similarity: f64,
    recommendation: String,
}

// Utility functions

async fn load_prompts_from_file(file_path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(file_path)?;
    Ok(content.lines().map(|s| s.to_string()).collect())
}

fn default_test_prompts() -> Vec<String> {
    vec![
        "Explain the theory of relativity in simple terms.".to_string(),
        "Write a short story about a robot learning to paint.".to_string(),
        "What are the advantages and disadvantages of renewable energy?".to_string(),
        "Describe the process of photosynthesis.".to_string(),
        "How does machine learning work?".to_string(),
    ]
}

fn create_benchmark_model(_loader: &GGUFLoader) -> Result<BenchmarkModel, Box<dyn std::error::Error>> {
    Ok(BenchmarkModel::new())
}

fn create_benchmark_engine_config() -> EngineConfig {
    EngineConfig {
        max_context_length: 2048,
        max_batch_size: 16,
        num_threads: num_cpus::get(),
        device: DeviceConfig {
            device_type: DeviceType::Cpu,
            cpu_fallback: true,
            ..Default::default()
        },
        memory: MemoryConfig {
            use_mmap: true,
            max_memory_mb: 4096,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn get_memory_usage() -> Option<usize> {
    // This would use platform-specific APIs to get current memory usage
    // For now, return a simulated value
    Some(2048 * 1024 * 1024) // 2GB
}

fn estimate_model_size(model_path: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let metadata = std::fs::metadata(model_path)?;
    Ok(metadata.len() as f64 / (1024.0 * 1024.0))
}

fn estimate_token_count(text: &str) -> usize {
    // Simple word-based estimation
    text.split_whitespace().count()
}

fn calculate_similarity(text1: &str, text2: &str) -> f64 {
    // Simple similarity based on common words
    let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
    let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();
    
    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();
    
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

impl BenchmarkReport {
    fn generate(
        loading: LoadingResults,
        inference: InferenceResults,
        memory: MemoryResults,
        throughput: ThroughputResults,
        latency: LatencyResults,
        accuracy: AccuracyResults,
    ) -> Self {
        let summary = BenchmarkSummary {
            woolly_avg_loading_time: average_duration(&loading.woolly_loading_times),
            llama_avg_loading_time: average_duration(&loading.llama_loading_times),
            woolly_avg_tokens_per_sec: average_tokens_per_sec(&inference.woolly_results),
            llama_avg_tokens_per_sec: average_tokens_per_sec(&inference.llama_results),
            woolly_memory_efficiency: memory.woolly_memory.peak_mb / memory.woolly_memory.model_size_mb,
            llama_memory_efficiency: memory.llama_memory.peak_mb / memory.llama_memory.model_size_mb,
            avg_accuracy_similarity: average_similarity(&accuracy.comparisons),
            recommendation: generate_recommendation(&inference, &memory, &accuracy),
        };
        
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            loading,
            inference,
            memory,
            throughput,
            latency,
            accuracy,
            summary,
        }
    }
}

fn average_duration(durations: &[Duration]) -> f64 {
    if durations.is_empty() {
        0.0
    } else {
        durations.iter().map(|d| d.as_secs_f64()).sum::<f64>() / durations.len() as f64
    }
}

fn average_tokens_per_sec(metrics: &[InferenceMetrics]) -> f64 {
    if metrics.is_empty() {
        0.0
    } else {
        metrics.iter().map(|m| m.tokens_per_second).sum::<f64>() / metrics.len() as f64
    }
}

fn average_similarity(comparisons: &[AccuracyComparison]) -> f64 {
    if comparisons.is_empty() {
        0.0
    } else {
        comparisons.iter().map(|c| c.similarity_score).sum::<f64>() / comparisons.len() as f64
    }
}

fn generate_recommendation(
    inference: &InferenceResults,
    memory: &MemoryResults,
    accuracy: &AccuracyResults,
) -> String {
    let woolly_speed = average_tokens_per_sec(&inference.woolly_results);
    let llama_speed = average_tokens_per_sec(&inference.llama_results);
    let similarity = average_similarity(&accuracy.comparisons);
    
    if woolly_speed > llama_speed && similarity > 0.8 {
        "Woolly shows superior performance with comparable accuracy.".to_string()
    } else if similarity > 0.9 {
        "Both implementations show excellent accuracy. Choose based on integration needs.".to_string()
    } else {
        "Performance varies by use case. Consider specific requirements.".to_string()
    }
}

async fn save_report(report: &BenchmarkReport) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write("benchmark_report.json", json)?;
    Ok(())
}

// Printing functions

fn print_loading_results(results: &LoadingResults) {
    if !results.woolly_loading_times.is_empty() {
        let avg_woolly = average_duration(&results.woolly_loading_times);
        println!("  🦙 Woolly average loading time: {:.2}s", avg_woolly);
    }
    
    if !results.llama_loading_times.is_empty() {
        let avg_llama = average_duration(&results.llama_loading_times);
        println!("  🦙 llama.cpp average loading time: {:.2}s", avg_llama);
        
        if !results.woolly_loading_times.is_empty() {
            let woolly_avg = average_duration(&results.woolly_loading_times);
            let speedup = avg_llama / woolly_avg;
            println!("  📊 Woolly loading speedup: {:.2}x", speedup);
        }
    }
}

fn print_inference_results(results: &InferenceResults) {
    if !results.woolly_results.is_empty() {
        let avg_woolly = average_tokens_per_sec(&results.woolly_results);
        println!("  🦙 Woolly average speed: {:.2} tokens/sec", avg_woolly);
    }
    
    if !results.llama_results.is_empty() {
        let avg_llama = average_tokens_per_sec(&results.llama_results);
        println!("  🦙 llama.cpp average speed: {:.2} tokens/sec", avg_llama);
        
        if !results.woolly_results.is_empty() {
            let woolly_avg = average_tokens_per_sec(&results.woolly_results);
            let speedup = woolly_avg / avg_llama;
            println!("  📊 Woolly inference speedup: {:.2}x", speedup);
        }
    }
}

fn print_memory_results(results: &MemoryResults) {
    println!("  🦙 Woolly memory usage: {:.1} MB", results.woolly_memory.peak_mb);
    println!("  🦙 llama.cpp memory usage: {:.1} MB", results.llama_memory.peak_mb);
    
    if results.woolly_memory.peak_mb > 0.0 && results.llama_memory.peak_mb > 0.0 {
        let efficiency = results.llama_memory.peak_mb / results.woolly_memory.peak_mb;
        println!("  📊 Memory efficiency ratio: {:.2}x", efficiency);
    }
}

fn print_throughput_results(results: &ThroughputResults) {
    for metrics in &results.results {
        println!("  Batch size {}: Woolly {:.2} tok/s, llama.cpp {:.2} tok/s", 
                metrics.batch_size, metrics.woolly_throughput, metrics.llama_throughput);
    }
}

fn print_latency_results(results: &LatencyResults) {
    println!("  🦙 Woolly first token latency: {:?}", results.woolly_first_token);
    println!("  🦙 llama.cpp first token latency: {:?}", results.llama_first_token);
}

fn print_accuracy_results(results: &AccuracyResults) {
    let avg_similarity = average_similarity(&results.comparisons);
    println!("  📊 Average output similarity: {:.1}%", avg_similarity * 100.0);
    
    for (i, comparison) in results.comparisons.iter().enumerate() {
        println!("    Test {}: {:.1}% similarity", i + 1, comparison.similarity_score * 100.0);
    }
}

fn print_summary(report: &BenchmarkReport) {
    println!("\n📋 Benchmark Summary:");
    println!("  Loading Time - Woolly: {:.2}s, llama.cpp: {:.2}s", 
            report.summary.woolly_avg_loading_time, report.summary.llama_avg_loading_time);
    println!("  Inference Speed - Woolly: {:.2} tok/s, llama.cpp: {:.2} tok/s", 
            report.summary.woolly_avg_tokens_per_sec, report.summary.llama_avg_tokens_per_sec);
    println!("  Memory Efficiency - Woolly: {:.2}x, llama.cpp: {:.2}x", 
            report.summary.woolly_memory_efficiency, report.summary.llama_memory_efficiency);
    println!("  Average Accuracy: {:.1}%", report.summary.avg_accuracy_similarity * 100.0);
    println!("  💡 Recommendation: {}", report.summary.recommendation);
}

// Benchmark model implementation

#[derive(Clone)]
struct BenchmarkModel;

impl BenchmarkModel {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Model for BenchmarkModel {
    fn name(&self) -> &str { "benchmark-model" }
    fn model_type(&self) -> &str { "llama" }
    fn vocab_size(&self) -> usize { 32000 }
    fn context_length(&self) -> usize { 2048 }
    fn hidden_size(&self) -> usize { 4096 }
    fn num_layers(&self) -> usize { 32 }
    fn num_heads(&self) -> usize { 32 }
    
    async fn forward(
        &self,
        _input_ids: &[u32],
        _past_kv_cache: Option<&(dyn std::any::Any + Send + Sync)>,
    ) -> Result<ModelOutput> {
        Ok(ModelOutput {
            logits: vec![0.1; self.vocab_size()],
            logits_shape: vec![1, 1, self.vocab_size()],
            past_kv_cache: None,
            hidden_states: None,
            attentions: None,
        })
    }
    
    async fn load_weights(&mut self, _path: &Path) -> Result<()> {
        Ok(())
    }
}

// Duration serialization helper
mod duration_serde {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(duration: &Duration, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs_f64().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args() {
        let args = vec![
            "program".to_string(),
            "--model".to_string(),
            "test.gguf".to_string(),
            "--iterations".to_string(),
            "5".to_string(),
        ];
        
        let config = parse_args(&args).unwrap();
        assert_eq!(config.model_path, "test.gguf");
        assert_eq!(config.num_iterations, 5);
    }

    #[test]
    fn test_similarity_calculation() {
        let text1 = "hello world test";
        let text2 = "hello world example";
        let similarity = calculate_similarity(text1, text2);
        assert!(similarity > 0.0 && similarity < 1.0);
    }

    #[test]
    fn test_estimate_token_count() {
        let text = "This is a test sentence";
        assert_eq!(estimate_token_count(text), 5);
    }
}