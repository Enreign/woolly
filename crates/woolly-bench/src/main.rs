//! Woolly Benchmark CLI

use woolly_bench::{
    runner::{BenchmarkRunner, BenchmarkProfile},
    Benchmark, BenchmarkResult, ComparisonFramework,
};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber;

#[derive(Parser)]
#[command(name = "woolly-bench")]
#[command(about = "Woolly benchmarking framework CLI")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Output directory for benchmark results
    #[arg(short, long, default_value = "bench_results")]
    output: PathBuf,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run tensor operation benchmarks
    Tensor(TensorArgs),
    
    /// Run model loading benchmarks
    Model(ModelArgs),
    
    /// Run comparison benchmarks against external implementations
    Compare(CompareArgs),
    
    /// Run all benchmarks
    All(AllArgs),
    
    /// List available benchmarks
    List,
    
    /// Show benchmark results
    Show(ShowArgs),
}

#[derive(Args)]
struct TensorArgs {
    /// Benchmark profile to use (quick, standard, comprehensive)
    #[arg(short, long, default_value = "standard")]
    profile: String,
    
    /// Run only specific tensor operations
    #[arg(short, long)]
    operations: Option<Vec<String>>,
}

#[derive(Args)]
struct ModelArgs {
    /// Path to the model file
    #[arg(short, long)]
    model_path: Option<PathBuf>,
    
    /// Benchmark profile to use
    #[arg(short, long, default_value = "standard")]
    profile: String,
}

#[derive(Args)]
struct CompareArgs {
    /// Path to Woolly model
    #[arg(short, long)]
    woolly_model: PathBuf,
    
    /// Path to llama.cpp executable
    #[arg(long)]
    llama_cpp: Option<PathBuf>,
    
    /// Path to other implementation
    #[arg(long)]
    other_impl: Option<PathBuf>,
    
    /// Benchmark scenarios to run
    #[arg(short, long)]
    scenarios: Option<Vec<String>>,
}

#[derive(Args)]
struct AllArgs {
    /// Model path for comprehensive benchmarking
    #[arg(short, long)]
    model_path: Option<PathBuf>,
    
    /// Benchmark profile to use
    #[arg(short, long, default_value = "standard")]
    profile: String,
}

#[derive(Args)]
struct ShowArgs {
    /// Path to results file
    #[arg(short, long)]
    results_file: PathBuf,
    
    /// Output format (json, markdown, table)
    #[arg(short, long, default_value = "table")]
    format: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    let log_level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .init();
    
    info!("Starting Woolly benchmark CLI");
    
    // Create output directory
    std::fs::create_dir_all(&cli.output)?;
    
    match cli.command {
        Commands::Tensor(args) => run_tensor_benchmarks(args, &cli.output).await,
        Commands::Model(args) => run_model_benchmarks(args, &cli.output).await,
        Commands::Compare(args) => run_comparison_benchmarks(args, &cli.output).await,
        Commands::All(args) => run_all_benchmarks(args, &cli.output).await,
        Commands::List => list_benchmarks(),
        Commands::Show(args) => show_results(args),
    }
}

async fn run_tensor_benchmarks(args: TensorArgs, output_dir: &PathBuf) -> anyhow::Result<()> {
    info!("Running tensor operation benchmarks with profile: {}", args.profile);
    
    let profile = match args.profile.as_str() {
        "quick" => BenchmarkProfile::quick(),
        "standard" => BenchmarkProfile::standard(),
        "comprehensive" => BenchmarkProfile::comprehensive(),
        _ => {
            eprintln!("Unknown profile: {}. Using standard.", args.profile);
            BenchmarkProfile::standard()
        }
    };
    
    let mut runner = BenchmarkRunner::new(output_dir);
    
    // Add mock tensor benchmarks
    // In a real implementation, these would be actual benchmark implementations
    runner.add_benchmark(Box::new(MockTensorBenchmark::new("matrix_multiplication", profile.clone())));
    runner.add_benchmark(Box::new(MockTensorBenchmark::new("element_wise_addition", profile.clone())));
    runner.add_benchmark(Box::new(MockTensorBenchmark::new("softmax", profile.clone())));
    
    if let Some(operations) = args.operations {
        info!("Filtering operations: {:?}", operations);
        // TODO: Filter benchmarks based on operations
    }
    
    runner.run_all().await?;
    
    info!("Tensor benchmarks completed. Results saved to {:?}", output_dir);
    Ok(())
}

async fn run_model_benchmarks(args: ModelArgs, output_dir: &PathBuf) -> anyhow::Result<()> {
    info!("Running model loading benchmarks with profile: {}", args.profile);
    
    let profile = match args.profile.as_str() {
        "quick" => BenchmarkProfile::quick(),
        "standard" => BenchmarkProfile::standard(),
        "comprehensive" => BenchmarkProfile::comprehensive(),
        _ => BenchmarkProfile::standard(),
    };
    
    let mut runner = BenchmarkRunner::new(output_dir);
    
    // Add model loading benchmarks
    runner.add_benchmark(Box::new(MockModelBenchmark::new("gguf_loading", profile.clone())));
    runner.add_benchmark(Box::new(MockModelBenchmark::new("config_parsing", profile)));
    
    runner.run_all().await?;
    
    info!("Model benchmarks completed. Results saved to {:?}", output_dir);
    Ok(())
}

async fn run_comparison_benchmarks(args: CompareArgs, output_dir: &PathBuf) -> anyhow::Result<()> {
    info!("Running comparison benchmarks");
    
    let mut framework = ComparisonFramework::new();
    
    // Add Woolly benchmark
    framework.add_benchmark(Box::new(MockWoollyBenchmark::new(&args.woolly_model)));
    
    // Add external benchmarks if provided
    if let Some(llama_cpp_path) = args.llama_cpp {
        info!("Adding llama.cpp comparison");
        framework.add_benchmark(Box::new(MockExternalBenchmark::new(
            "llama.cpp",
            &llama_cpp_path,
        )));
    }
    
    if let Some(other_path) = args.other_impl {
        info!("Adding other implementation comparison");
        framework.add_benchmark(Box::new(MockExternalBenchmark::new(
            "other",
            &other_path,
        )));
    }
    
    framework.run_all()?;
    
    let report = framework.generate_report();
    let report_path = output_dir.join("comparison_report.md");
    std::fs::write(&report_path, report.to_markdown())?;
    
    framework.save_results(&output_dir.join("comparison_results.json"))?;
    
    info!("Comparison benchmarks completed. Report saved to {:?}", report_path);
    Ok(())
}

async fn run_all_benchmarks(args: AllArgs, output_dir: &PathBuf) -> anyhow::Result<()> {
    info!("Running all benchmarks with profile: {}", args.profile);
    
    // Run tensor benchmarks
    let tensor_args = TensorArgs {
        profile: args.profile.clone(),
        operations: None,
    };
    run_tensor_benchmarks(tensor_args, output_dir).await?;
    
    // Run model benchmarks
    let model_args = ModelArgs {
        model_path: args.model_path.clone(),
        profile: args.profile,
    };
    run_model_benchmarks(model_args, output_dir).await?;
    
    // Run comparison if model path is provided
    if let Some(model_path) = args.model_path {
        let compare_args = CompareArgs {
            woolly_model: model_path,
            llama_cpp: None,
            other_impl: None,
            scenarios: None,
        };
        run_comparison_benchmarks(compare_args, output_dir).await?;
    }
    
    info!("All benchmarks completed");
    Ok(())
}

fn list_benchmarks() -> anyhow::Result<()> {
    println!("Available Woolly Benchmarks:");
    println!();
    println!("Tensor Operations:");
    println!("  - matrix_multiplication");
    println!("  - element_wise_addition");
    println!("  - element_wise_multiplication");
    println!("  - softmax");
    println!("  - relu");
    println!("  - batch_operations");
    println!("  - reductions (sum, mean, max)");
    println!("  - reshape_operations");
    println!("  - slicing_operations");
    println!();
    println!("Model Operations:");
    println!("  - gguf_loading");
    println!("  - config_parsing");
    println!("  - memory_mapping");
    println!("  - tensor_metadata_parsing");
    println!("  - model_initialization");
    println!();
    println!("Comparison Scenarios:");
    println!("  - text_generation");
    println!("  - model_loading");
    println!("  - inference_speed");
    println!();
    println!("Profiles:");
    println!("  - quick: Fast benchmarking for development");
    println!("  - standard: Regular benchmarking with moderate precision");
    println!("  - comprehensive: Thorough benchmarking with high precision");
    
    Ok(())
}

fn show_results(args: ShowArgs) -> anyhow::Result<()> {
    info!("Loading results from {:?}", args.results_file);
    
    let content = std::fs::read_to_string(&args.results_file)?;
    let results: Vec<BenchmarkResult> = serde_json::from_str(&content)?;
    
    match args.format.as_str() {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        "table" => {
            println!("{:<30} {:<12} {:<12} {:<12} {:<12}", 
                     "Benchmark", "Iterations", "Mean (ms)", "Min (ms)", "Max (ms)");
            println!("{}", "-".repeat(80));
            
            for result in results {
                println!("{:<30} {:<12} {:<12.3} {:<12.3} {:<12.3}",
                         result.name,
                         result.iterations,
                         result.mean_time.as_secs_f64() * 1000.0,
                         result.min_time.as_secs_f64() * 1000.0,
                         result.max_time.as_secs_f64() * 1000.0);
            }
        }
        "markdown" => {
            println!("# Benchmark Results\n");
            println!("| Benchmark | Iterations | Mean Time | Min Time | Max Time |");
            println!("|-----------|------------|-----------|----------|----------|");
            
            for result in results {
                println!("| {} | {} | {:.3}ms | {:.3}ms | {:.3}ms |",
                         result.name,
                         result.iterations,
                         result.mean_time.as_secs_f64() * 1000.0,
                         result.min_time.as_secs_f64() * 1000.0,
                         result.max_time.as_secs_f64() * 1000.0);
            }
        }
        _ => {
            eprintln!("Unknown format: {}. Use json, table, or markdown.", args.format);
        }
    }
    
    Ok(())
}

// Mock benchmark implementations for demonstration
struct MockTensorBenchmark {
    name: String,
    profile: BenchmarkProfile,
}

impl MockTensorBenchmark {
    fn new(name: &str, profile: BenchmarkProfile) -> Self {
        Self {
            name: name.to_string(),
            profile,
        }
    }
}

impl Benchmark for MockTensorBenchmark {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn run(&mut self) -> anyhow::Result<BenchmarkResult> {
        use std::time::{Duration, Instant};
        
        let mut times = Vec::new();
        
        for _ in 0..self.profile.iterations {
            let start = Instant::now();
            // Simulate work
            std::thread::sleep(Duration::from_micros(100));
            times.push(start.elapsed());
        }
        
        let total_time: Duration = times.iter().sum();
        let mean_time = total_time / self.profile.iterations;
        let min_time = times.iter().min().copied().unwrap_or_default();
        let max_time = times.iter().max().copied().unwrap_or_default();
        
        Ok(BenchmarkResult {
            name: self.name().to_string(),
            iterations: self.profile.iterations,
            total_time,
            mean_time,
            min_time,
            max_time,
            stddev: Some(1000.0), // Mock stddev
            throughput: None,
            metadata: serde_json::json!({
                "profile": self.profile.name,
            }),
        })
    }
}

struct MockModelBenchmark {
    name: String,
    profile: BenchmarkProfile,
}

impl MockModelBenchmark {
    fn new(name: &str, profile: BenchmarkProfile) -> Self {
        Self {
            name: name.to_string(),
            profile,
        }
    }
}

impl Benchmark for MockModelBenchmark {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn run(&mut self) -> anyhow::Result<BenchmarkResult> {
        use std::time::{Duration, Instant};
        
        let mut times = Vec::new();
        
        for _ in 0..self.profile.iterations {
            let start = Instant::now();
            // Simulate model loading work
            std::thread::sleep(Duration::from_millis(50));
            times.push(start.elapsed());
        }
        
        let total_time: Duration = times.iter().sum();
        let mean_time = total_time / self.profile.iterations;
        let min_time = times.iter().min().copied().unwrap_or_default();
        let max_time = times.iter().max().copied().unwrap_or_default();
        
        Ok(BenchmarkResult {
            name: self.name().to_string(),
            iterations: self.profile.iterations,
            total_time,
            mean_time,
            min_time,
            max_time,
            stddev: Some(5000.0), // Mock stddev
            throughput: None,
            metadata: serde_json::json!({
                "profile": self.profile.name,
            }),
        })
    }
}

struct MockWoollyBenchmark {
    model_path: PathBuf,
}

impl MockWoollyBenchmark {
    fn new(model_path: &PathBuf) -> Self {
        Self {
            model_path: model_path.clone(),
        }
    }
}

impl Benchmark for MockWoollyBenchmark {
    fn name(&self) -> &str {
        "woolly"
    }
    
    fn run(&mut self) -> anyhow::Result<BenchmarkResult> {
        use std::time::{Duration, Instant};
        
        let start = Instant::now();
        // Mock Woolly inference
        std::thread::sleep(Duration::from_millis(100));
        let duration = start.elapsed();
        
        Ok(BenchmarkResult {
            name: self.name().to_string(),
            iterations: 1,
            total_time: duration,
            mean_time: duration,
            min_time: duration,
            max_time: duration,
            stddev: None,
            throughput: Some(10.0), // Mock tokens/sec
            metadata: serde_json::json!({
                "model_path": self.model_path,
            }),
        })
    }
}

struct MockExternalBenchmark {
    name: String,
    executable_path: PathBuf,
}

impl MockExternalBenchmark {
    fn new(name: &str, executable_path: &PathBuf) -> Self {
        Self {
            name: name.to_string(),
            executable_path: executable_path.clone(),
        }
    }
}

impl Benchmark for MockExternalBenchmark {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn run(&mut self) -> anyhow::Result<BenchmarkResult> {
        use std::time::{Duration, Instant};
        
        let start = Instant::now();
        // Mock external benchmark execution
        std::thread::sleep(Duration::from_millis(150));
        let duration = start.elapsed();
        
        Ok(BenchmarkResult {
            name: self.name().to_string(),
            iterations: 1,
            total_time: duration,
            mean_time: duration,
            min_time: duration,
            max_time: duration,
            stddev: None,
            throughput: Some(8.0), // Mock tokens/sec
            metadata: serde_json::json!({
                "executable_path": self.executable_path,
            }),
        })
    }
}