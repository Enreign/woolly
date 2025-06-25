//! Simplified benchmark command implementation
//!
//! This module implements basic benchmarking functionality for Woolly models.

use anyhow::{Context, Result};
use async_trait::async_trait;
use clap::Args;
use console::style;
use serde_json::json;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{debug, info};

use crate::commands::Command;
use crate::config::Config;
use crate::utils::{create_progress_bar, format_duration, print_output, print_success};

use woolly_gguf::GGUFLoader;

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum BenchmarkType {
    Loading,
    Basic,
    All,
}

#[derive(Args, Debug)]
pub struct BenchmarkCommand {
    /// Path to the model file (GGUF format)
    #[arg(short, long)]
    pub model: Option<PathBuf>,

    /// Type of benchmark to run
    #[arg(short, long, value_enum, default_value = "basic")]
    pub benchmark_type: BenchmarkType,

    /// Number of benchmark iterations
    #[arg(long, default_value = "5")]
    pub iterations: u32,

    /// Number of warmup iterations
    #[arg(long, default_value = "1")]
    pub warmup: u32,

    /// Output directory for detailed results
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,

    /// Export results to JSON
    #[arg(long)]
    pub json_export: bool,
}

#[derive(Debug)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: u32,
    pub times: Vec<Duration>,
    pub mean_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

impl BenchmarkResult {
    pub fn new(name: String, iterations: u32, times: Vec<Duration>) -> Self {
        let mean_time = Duration::from_nanos(
            (times.iter().map(|d| d.as_nanos()).sum::<u128>() / iterations as u128) as u64
        );
        let min_time = times.iter().min().copied().unwrap_or_default();
        let max_time = times.iter().max().copied().unwrap_or_default();

        Self {
            name,
            iterations,
            times,
            mean_time,
            min_time,
            max_time,
        }
    }
}

#[async_trait]
impl Command for BenchmarkCommand {
    async fn execute(&self, config: &Config, json_output: bool) -> Result<()> {
        debug!("Executing benchmark command: {:?}", self);

        // Determine model path
        let model_path = self.resolve_model_path(config)?;
        info!("Benchmarking model: {}", model_path.display());

        // Run benchmarks based on type
        let results = match self.benchmark_type {
            BenchmarkType::Loading => {
                vec![self.benchmark_model_loading(&model_path).await?]
            }
            BenchmarkType::Basic => {
                let mut results = Vec::new();
                results.push(self.benchmark_model_loading(&model_path).await?);
                results.push(self.benchmark_model_info(&model_path).await?);
                results
            }
            BenchmarkType::All => {
                let mut results = Vec::new();
                results.push(self.benchmark_model_loading(&model_path).await?);
                results.push(self.benchmark_model_info(&model_path).await?);
                results.push(self.benchmark_model_validation(&model_path).await?);
                results
            }
        };

        // Export results if requested
        if let Some(output_dir) = &self.output_dir {
            self.export_results(&results, output_dir).await?;
        }

        // Output results
        if json_output {
            let json_results = self.format_results_json(&results)?;
            print_output(&json_results, true)?;
        } else {
            self.print_results(&results)?;
        }

        Ok(())
    }
}

impl BenchmarkCommand {
    /// Resolve the model path from arguments or configuration
    fn resolve_model_path(&self, config: &Config) -> Result<PathBuf> {
        match &self.model {
            Some(path) => Ok(path.clone()),
            None => {
                if let Some(default_model) = &config.default_model {
                    config.find_model(&default_model.to_string_lossy())
                } else {
                    anyhow::bail!("No model specified. Use --model or set a default model in config.");
                }
            }
        }
    }

    /// Benchmark model loading performance
    async fn benchmark_model_loading(&self, model_path: &PathBuf) -> Result<BenchmarkResult> {
        println!("Running model loading benchmark...");
        let pb = create_progress_bar((self.warmup + self.iterations) as u64, "Model loading");
        
        let mut times = Vec::new();

        // Warmup
        for i in 0..self.warmup {
            let start = Instant::now();
            let _ = GGUFLoader::from_path(model_path)?;
            times.push(start.elapsed());
            pb.set_position(i as u64 + 1);
        }

        // Clear warmup times
        times.clear();

        // Actual benchmark
        for i in 0..self.iterations {
            let start = Instant::now();
            let _ = GGUFLoader::from_path(model_path)?;
            times.push(start.elapsed());
            pb.set_position((self.warmup + i + 1) as u64);
        }

        pb.finish_with_message("Model loading benchmark completed");
        Ok(BenchmarkResult::new("Model Loading".to_string(), self.iterations, times))
    }

    /// Benchmark model info extraction
    async fn benchmark_model_info(&self, model_path: &PathBuf) -> Result<BenchmarkResult> {
        println!("Running model info benchmark...");
        let pb = create_progress_bar((self.warmup + self.iterations) as u64, "Model info extraction");
        
        let mut times = Vec::new();

        // Load once for subsequent operations
        let gguf_file = GGUFLoader::from_path(model_path)?;

        // Warmup
        for i in 0..self.warmup {
            let start = Instant::now();
            let _ = gguf_file.metadata();
            let _ = gguf_file.tensors();
            times.push(start.elapsed());
            pb.set_position(i as u64 + 1);
        }

        // Clear warmup times
        times.clear();

        // Actual benchmark
        for i in 0..self.iterations {
            let start = Instant::now();
            let metadata = gguf_file.metadata();
            let tensors = gguf_file.tensors();
            
            // Do some processing to simulate real usage
            let _ = metadata.get("general.architecture");
            let _ = tensors.len();
            let _: u64 = tensors.values().map(|t| t.data_size()).sum();
            
            times.push(start.elapsed());
            pb.set_position((self.warmup + i + 1) as u64);
        }

        pb.finish_with_message("Model info benchmark completed");
        Ok(BenchmarkResult::new("Model Info Extraction".to_string(), self.iterations, times))
    }

    /// Benchmark model validation
    async fn benchmark_model_validation(&self, model_path: &PathBuf) -> Result<BenchmarkResult> {
        println!("Running model validation benchmark...");
        let pb = create_progress_bar((self.warmup + self.iterations) as u64, "Model validation");
        
        let mut times = Vec::new();

        // Warmup
        for i in 0..self.warmup {
            let start = Instant::now();
            let gguf_file = GGUFLoader::from_path(model_path)?;
            self.validate_model_simple(&gguf_file)?;
            times.push(start.elapsed());
            pb.set_position(i as u64 + 1);
        }

        // Clear warmup times
        times.clear();

        // Actual benchmark
        for i in 0..self.iterations {
            let start = Instant::now();
            let gguf_file = GGUFLoader::from_path(model_path)?;
            self.validate_model_simple(&gguf_file)?;
            times.push(start.elapsed());
            pb.set_position((self.warmup + i + 1) as u64);
        }

        pb.finish_with_message("Model validation benchmark completed");
        Ok(BenchmarkResult::new("Model Validation".to_string(), self.iterations, times))
    }

    /// Simple model validation
    fn validate_model_simple(&self, gguf_file: &woolly_gguf::GGUFLoader) -> Result<()> {
        let header = gguf_file.header();
        let metadata = gguf_file.metadata();
        let tensors = gguf_file.tensors();

        // Basic validation checks
        if header.version.0 < 2 {
            anyhow::bail!("GGUF version too old");
        }

        if !metadata.kv_pairs.contains_key("general.architecture") {
            anyhow::bail!("Missing architecture information");
        }

        if tensors.is_empty() {
            anyhow::bail!("No tensors found");
        }

        // Check tensor consistency
        for (name, tensor) in tensors {
            if name.is_empty() {
                anyhow::bail!("Empty tensor name");
            }
            if tensor.data_size() == 0 {
                anyhow::bail!("Zero-size tensor");
            }
        }

        Ok(())
    }

    /// Format results as JSON
    fn format_results_json(&self, results: &[BenchmarkResult]) -> Result<serde_json::Value> {
        let mut json_results = json!({});

        for result in results {
            json_results[&result.name] = json!({
                "iterations": result.iterations,
                "mean_time_ms": result.mean_time.as_secs_f64() * 1000.0,
                "min_time_ms": result.min_time.as_secs_f64() * 1000.0,
                "max_time_ms": result.max_time.as_secs_f64() * 1000.0,
            });
        }

        Ok(json_results)
    }

    /// Print benchmark results in human-readable format
    fn print_results(&self, results: &[BenchmarkResult]) -> Result<()> {
        println!();
        println!("{}", style("Benchmark Results").bold().cyan());
        println!("═══════════════════════════════════════════════════════════════════");
        println!();

        for result in results {
            println!("{}", style(&result.name).bold().green());
            println!("─────────────────────────────────────────────────");
            println!("Iterations: {}", result.iterations);
            println!("Mean Time: {}", format_duration(result.mean_time));
            println!("Min Time: {}", format_duration(result.min_time));
            println!("Max Time: {}", format_duration(result.max_time));
            
            // Calculate standard deviation
            let mean_nanos = result.mean_time.as_nanos() as f64;
            let variance: f64 = result.times.iter()
                .map(|d| {
                    let diff = d.as_nanos() as f64 - mean_nanos;
                    diff * diff
                })
                .sum::<f64>() / result.iterations as f64;
            let stddev = Duration::from_nanos(variance.sqrt() as u64);
            println!("Std Dev: {}", format_duration(stddev));
            
            println!();
        }

        // Print performance insights
        self.print_performance_insights(results)?;

        Ok(())
    }

    /// Print performance insights and recommendations
    fn print_performance_insights(&self, results: &[BenchmarkResult]) -> Result<()> {
        println!("{}", style("Performance Insights").bold().magenta());
        println!("─────────────────────────────────────────────────");

        for result in results {
            let mean_ms = result.mean_time.as_secs_f64() * 1000.0;
            
            match result.name.as_str() {
                "Model Loading" => {
                    if mean_ms > 5000.0 {
                        println!("⚠️  Slow model loading ({:.2}ms) - consider faster storage", mean_ms);
                    } else if mean_ms < 100.0 {
                        println!("✅ Very fast model loading ({:.2}ms)", mean_ms);
                    } else {
                        println!("👍 Good model loading performance ({:.2}ms)", mean_ms);
                    }
                }
                "Model Info Extraction" => {
                    if mean_ms > 10.0 {
                        println!("⚠️  Slow metadata extraction ({:.2}ms)", mean_ms);
                    } else {
                        println!("✅ Fast metadata extraction ({:.2}ms)", mean_ms);
                    }
                }
                "Model Validation" => {
                    if mean_ms > 100.0 {
                        println!("⚠️  Slow model validation ({:.2}ms)", mean_ms);
                    } else {
                        println!("✅ Fast model validation ({:.2}ms)", mean_ms);
                    }
                }
                _ => {}
            }
        }

        println!();
        println!("📊 Completed {} benchmark(s)", results.len());

        Ok(())
    }

    /// Export results to files
    async fn export_results(
        &self,
        results: &[BenchmarkResult],
        output_dir: &PathBuf,
    ) -> Result<()> {
        tokio::fs::create_dir_all(output_dir).await
            .context("Failed to create output directory")?;

        if self.json_export {
            let json_path = output_dir.join("benchmark_results.json");
            let json_data = self.format_results_json(results)?;
            tokio::fs::write(&json_path, serde_json::to_string_pretty(&json_data)?)
                .await
                .context("Failed to write JSON results")?;
            print_success(&format!("Results exported to: {}", json_path.display()));
        }

        Ok(())
    }
}

impl Default for BenchmarkCommand {
    fn default() -> Self {
        Self {
            model: None,
            benchmark_type: BenchmarkType::Basic,
            iterations: 5,
            warmup: 1,
            output_dir: None,
            json_export: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_benchmark_command() {
        let cmd = BenchmarkCommand::default();
        assert_eq!(cmd.iterations, 5);
        assert_eq!(cmd.warmup, 1);
        assert!(!cmd.json_export);
    }

    #[test]
    fn test_benchmark_result() {
        let times = vec![
            Duration::from_millis(100),
            Duration::from_millis(110),
            Duration::from_millis(90),
        ];
        let result = BenchmarkResult::new("Test".to_string(), 3, times.clone());
        
        assert_eq!(result.name, "Test");
        assert_eq!(result.iterations, 3);
        assert_eq!(result.min_time, Duration::from_millis(90));
        assert_eq!(result.max_time, Duration::from_millis(110));
    }
}