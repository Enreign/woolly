//! Woolly benchmarking framework
//!
//! This crate provides benchmarking utilities for Woolly, including:
//! - Tensor operation benchmarks
//! - Model loading benchmarks
//! - Comparison framework for external implementations (e.g., llama.cpp)

use std::time::{Duration, Instant};
use std::path::Path;
use serde::{Deserialize, Serialize};

pub mod runner;

/// Result of a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: u32,
    pub total_time: Duration,
    pub mean_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub stddev: Option<f64>,
    pub throughput: Option<f64>,
    pub metadata: serde_json::Value,
}

/// Trait for implementing comparable benchmarks
pub trait Benchmark {
    /// Name of the benchmark
    fn name(&self) -> &str;
    
    /// Run the benchmark and return results
    fn run(&mut self) -> anyhow::Result<BenchmarkResult>;
    
    /// Optional warmup before benchmarking
    fn warmup(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Framework for comparing different implementations
pub struct ComparisonFramework {
    benchmarks: Vec<Box<dyn Benchmark>>,
    results: Vec<BenchmarkResult>,
}

impl ComparisonFramework {
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
            results: Vec::new(),
        }
    }
    
    /// Add a benchmark to the comparison
    pub fn add_benchmark(&mut self, benchmark: Box<dyn Benchmark>) {
        self.benchmarks.push(benchmark);
    }
    
    /// Run all benchmarks
    pub fn run_all(&mut self) -> anyhow::Result<()> {
        self.results.clear();
        
        for benchmark in &mut self.benchmarks {
            tracing::info!("Running benchmark: {}", benchmark.name());
            
            // Warmup
            benchmark.warmup()?;
            
            // Run benchmark
            let result = benchmark.run()?;
            self.results.push(result);
        }
        
        Ok(())
    }
    
    /// Generate comparison report
    pub fn generate_report(&self) -> ComparisonReport {
        ComparisonReport::new(self.results.clone())
    }
    
    /// Save results to JSON file
    pub fn save_results(&self, path: &Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(&self.results)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Comparison report with analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub results: Vec<BenchmarkResult>,
    pub analysis: ComparisonAnalysis,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComparisonAnalysis {
    pub fastest: Option<String>,
    pub slowest: Option<String>,
    pub relative_performance: Vec<RelativePerformance>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RelativePerformance {
    pub name: String,
    pub relative_to_fastest: f64,
    pub relative_to_baseline: Option<f64>,
}

impl ComparisonReport {
    pub fn new(results: Vec<BenchmarkResult>) -> Self {
        let analysis = Self::analyze(&results);
        Self { results, analysis }
    }
    
    fn analyze(results: &[BenchmarkResult]) -> ComparisonAnalysis {
        if results.is_empty() {
            return ComparisonAnalysis {
                fastest: None,
                slowest: None,
                relative_performance: Vec::new(),
            };
        }
        
        let fastest = results
            .iter()
            .min_by_key(|r| r.mean_time)
            .map(|r| r.name.clone());
        
        let slowest = results
            .iter()
            .max_by_key(|r| r.mean_time)
            .map(|r| r.name.clone());
        
        let fastest_time = results
            .iter()
            .map(|r| r.mean_time.as_nanos() as f64)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);
        
        let baseline_time = results
            .first()
            .map(|r| r.mean_time.as_nanos() as f64);
        
        let relative_performance = results
            .iter()
            .map(|r| {
                let time = r.mean_time.as_nanos() as f64;
                RelativePerformance {
                    name: r.name.clone(),
                    relative_to_fastest: time / fastest_time,
                    relative_to_baseline: baseline_time.map(|b| time / b),
                }
            })
            .collect();
        
        ComparisonAnalysis {
            fastest,
            slowest,
            relative_performance,
        }
    }
    
    /// Generate markdown report
    pub fn to_markdown(&self) -> String {
        let mut report = String::new();
        
        report.push_str("# Benchmark Comparison Report\n\n");
        
        if let Some(fastest) = &self.analysis.fastest {
            report.push_str(&format!("**Fastest**: {}\n", fastest));
        }
        if let Some(slowest) = &self.analysis.slowest {
            report.push_str(&format!("**Slowest**: {}\n\n", slowest));
        }
        
        report.push_str("## Results\n\n");
        report.push_str("| Benchmark | Mean Time | Min Time | Max Time | Relative to Fastest |\n");
        report.push_str("|-----------|-----------|----------|----------|--------------------|\n");
        
        for (result, perf) in self.results.iter().zip(&self.analysis.relative_performance) {
            report.push_str(&format!(
                "| {} | {:.2}ms | {:.2}ms | {:.2}ms | {:.2}x |\n",
                result.name,
                result.mean_time.as_secs_f64() * 1000.0,
                result.min_time.as_secs_f64() * 1000.0,
                result.max_time.as_secs_f64() * 1000.0,
                perf.relative_to_fastest
            ));
        }
        
        report
    }
}

/// Helper function to measure execution time
pub fn measure_time<F, R>(f: F) -> (Duration, R)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (duration, result)
}

/// Helper to run a benchmark multiple times and collect statistics
pub fn run_benchmark_iterations<F>(
    name: &str,
    iterations: u32,
    mut f: F,
) -> anyhow::Result<BenchmarkResult>
where
    F: FnMut() -> anyhow::Result<()>,
{
    let mut times = Vec::new();
    let total_start = Instant::now();
    
    for _ in 0..iterations {
        let start = Instant::now();
        f()?;
        times.push(start.elapsed());
    }
    
    let total_time = total_start.elapsed();
    let mean_time = Duration::from_nanos(
        (times.iter().map(|d| d.as_nanos()).sum::<u128>() / iterations as u128) as u64
    );
    let min_time = times.iter().min().copied().unwrap_or_default();
    let max_time = times.iter().max().copied().unwrap_or_default();
    
    // Calculate standard deviation
    let mean_nanos = mean_time.as_nanos() as f64;
    let variance = times
        .iter()
        .map(|d| {
            let diff = d.as_nanos() as f64 - mean_nanos;
            diff * diff
        })
        .sum::<f64>() / iterations as f64;
    let stddev = variance.sqrt();
    
    Ok(BenchmarkResult {
        name: name.to_string(),
        iterations,
        total_time,
        mean_time,
        min_time,
        max_time,
        stddev: Some(stddev),
        throughput: None,
        metadata: serde_json::json!({}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_comparison_framework() {
        let mut framework = ComparisonFramework::new();
        assert_eq!(framework.results.len(), 0);
    }
    
    #[test]
    fn test_benchmark_result_serialization() {
        let result = BenchmarkResult {
            name: "test".to_string(),
            iterations: 100,
            total_time: Duration::from_millis(1000),
            mean_time: Duration::from_millis(10),
            min_time: Duration::from_millis(8),
            max_time: Duration::from_millis(12),
            stddev: Some(1.5),
            throughput: Some(100.0),
            metadata: serde_json::json!({"foo": "bar"}),
        };
        
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: BenchmarkResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.name, deserialized.name);
    }
}