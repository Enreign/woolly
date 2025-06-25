//! Benchmark runner utilities

use crate::{Benchmark, BenchmarkResult};
use std::path::Path;
use tracing::{info, warn};

/// Benchmark suite runner
pub struct BenchmarkRunner {
    benchmarks: Vec<Box<dyn Benchmark>>,
    results: Vec<BenchmarkResult>,
    output_dir: std::path::PathBuf,
}

impl BenchmarkRunner {
    pub fn new(output_dir: impl AsRef<Path>) -> Self {
        Self {
            benchmarks: Vec::new(),
            results: Vec::new(),
            output_dir: output_dir.as_ref().to_path_buf(),
        }
    }
    
    /// Add a benchmark to the suite
    pub fn add_benchmark(&mut self, benchmark: Box<dyn Benchmark>) {
        self.benchmarks.push(benchmark);
    }
    
    /// Run all benchmarks in the suite
    pub async fn run_all(&mut self) -> anyhow::Result<()> {
        info!("Starting benchmark suite with {} benchmarks", self.benchmarks.len());
        
        // Create output directory
        std::fs::create_dir_all(&self.output_dir)?;
        
        self.results.clear();
        
        let total_benchmarks = self.benchmarks.len();
        
        for (i, benchmark) in self.benchmarks.iter_mut().enumerate() {
            info!("Running benchmark {}/{}: {}", i + 1, total_benchmarks, benchmark.name());
            
            match benchmark.warmup() {
                Ok(_) => info!("Warmup completed for {}", benchmark.name()),
                Err(e) => warn!("Warmup failed for {}: {}", benchmark.name(), e),
            }
            
            match benchmark.run() {
                Ok(result) => {
                    info!(
                        "Benchmark {} completed: mean time = {:?}",
                        benchmark.name(),
                        result.mean_time
                    );
                    self.results.push(result);
                }
                Err(e) => {
                    warn!("Benchmark {} failed: {}", benchmark.name(), e);
                }
            }
        }
        
        // Save results
        self.save_results()?;
        
        Ok(())
    }
    
    /// Save benchmark results to disk
    fn save_results(&self) -> anyhow::Result<()> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        
        // Save JSON results
        let json_path = self.output_dir.join(format!("results_{}.json", timestamp));
        let json = serde_json::to_string_pretty(&self.results)?;
        std::fs::write(&json_path, json)?;
        info!("Results saved to {:?}", json_path);
        
        // Save markdown report
        let md_path = self.output_dir.join(format!("report_{}.md", timestamp));
        let report = self.generate_markdown_report();
        std::fs::write(&md_path, report)?;
        info!("Report saved to {:?}", md_path);
        
        Ok(())
    }
    
    /// Generate a markdown report of the results
    fn generate_markdown_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str(&format!(
            "# Woolly Benchmark Report\n\nGenerated: {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));
        
        report.push_str("## Summary\n\n");
        report.push_str(&format!("Total benchmarks run: {}\n\n", self.results.len()));
        
        report.push_str("## Results\n\n");
        report.push_str("| Benchmark | Iterations | Mean Time | Min Time | Max Time | Std Dev |\n");
        report.push_str("|-----------|------------|-----------|----------|----------|----------|\n");
        
        for result in &self.results {
            report.push_str(&format!(
                "| {} | {} | {:.3}ms | {:.3}ms | {:.3}ms | {:.3}ms |\n",
                result.name,
                result.iterations,
                result.mean_time.as_secs_f64() * 1000.0,
                result.min_time.as_secs_f64() * 1000.0,
                result.max_time.as_secs_f64() * 1000.0,
                result.stddev.unwrap_or(0.0) / 1_000_000.0, // Convert from nanos to ms
            ));
        }
        
        report.push_str("\n## Detailed Results\n\n");
        
        for result in &self.results {
            report.push_str(&format!("### {}\n\n", result.name));
            
            if let Some(throughput) = result.throughput {
                report.push_str(&format!("- **Throughput**: {:.2} ops/sec\n", throughput));
            }
            
            if !result.metadata.is_null() {
                report.push_str(&format!(
                    "- **Metadata**: {}\n",
                    serde_json::to_string_pretty(&result.metadata).unwrap_or_default()
                ));
            }
            
            report.push_str("\n");
        }
        
        report
    }
    
    /// Get the results
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }
}

/// Profile-based benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkProfile {
    pub name: String,
    pub tensor_sizes: Vec<usize>,
    pub batch_sizes: Vec<usize>,
    pub iterations: u32,
    pub warmup_iterations: u32,
}

impl BenchmarkProfile {
    /// Quick profile for fast benchmarking
    pub fn quick() -> Self {
        Self {
            name: "quick".to_string(),
            tensor_sizes: vec![100, 1000],
            batch_sizes: vec![1, 8],
            iterations: 10,
            warmup_iterations: 2,
        }
    }
    
    /// Standard profile for regular benchmarking
    pub fn standard() -> Self {
        Self {
            name: "standard".to_string(),
            tensor_sizes: vec![100, 1000, 10000],
            batch_sizes: vec![1, 8, 32],
            iterations: 100,
            warmup_iterations: 10,
        }
    }
    
    /// Comprehensive profile for thorough benchmarking
    pub fn comprehensive() -> Self {
        Self {
            name: "comprehensive".to_string(),
            tensor_sizes: vec![100, 1000, 10000, 100000],
            batch_sizes: vec![1, 8, 32, 128],
            iterations: 1000,
            warmup_iterations: 100,
        }
    }
}