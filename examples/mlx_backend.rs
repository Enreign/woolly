//! Apple MLX Backend Example
//!
//! This example demonstrates how to use Woolly with Apple's MLX framework
//! for accelerated inference on Apple Silicon devices.
//!
//! Features demonstrated:
//! - MLX backend initialization
//! - GPU memory management
//! - Performance monitoring
//! - Mixed precision inference
//! - Batch processing with MLX
//!
//! Usage:
//!   cargo run --example mlx_backend --features mlx -- --model path/to/model.gguf

use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tracing::{info, error};
use woolly_core::prelude::*;
use woolly_gguf::GGUFLoader;
use woolly_mlx::{MLXBackend, MLXConfig, MLXDevice};

#[derive(Debug)]
struct Args {
    model_path: PathBuf,
    prompt: String,
    max_tokens: usize,
    batch_size: usize,
    use_fp16: bool,
    monitor_gpu: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            prompt: "The future of AI is".to_string(),
            max_tokens: 100,
            batch_size: 1,
            use_fp16: true,
            monitor_gpu: true,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,woolly_mlx=debug")
        .init();
    
    let args = parse_args()?;
    
    println!("🍎 Woolly MLX Backend Example");
    println!("═══════════════════════════════");
    
    // Check MLX availability
    if !MLXDevice::is_available() {
        error!("MLX is not available on this system");
        println!("❌ This example requires an Apple Silicon Mac with MLX support");
        return Err("MLX not available".into());
    }
    
    let device_info = MLXDevice::info()?;
    println!("📱 Device Information:");
    println!("  Name: {}", device_info.name);
    println!("  Memory: {:.1} GB", device_info.memory_gb);
    println!("  Compute Units: {}", device_info.compute_units);
    println!("  MLX Version: {}", device_info.mlx_version);
    println!();
    
    // Step 1: Load model with MLX optimizations
    println!("📂 Loading model with MLX optimizations...");
    let loader = GGUFLoader::from_path(&args.model_path)?;
    
    // Configure MLX backend
    let mlx_config = MLXConfig {
        device: MLXDevice::default(),
        use_fp16: args.use_fp16,
        use_unified_memory: true,
        metal_capture: false,
        profile_ops: args.monitor_gpu,
        max_memory_gb: None, // Use all available memory
        cache_size_mb: 512,
        compile_graphs: true, // Enable graph compilation
    };
    
    println!("⚙️ MLX Configuration:");
    println!("  Precision: {}", if args.use_fp16 { "FP16" } else { "FP32" });
    println!("  Unified Memory: {}", mlx_config.use_unified_memory);
    println!("  Graph Compilation: {}", mlx_config.compile_graphs);
    println!();
    
    // Create MLX-accelerated model
    let model = create_mlx_model(&loader, mlx_config).await?;
    
    // Step 2: Create inference engine with MLX backend
    println!("🚀 Initializing MLX inference engine...");
    let engine_config = EngineConfig {
        device: DeviceConfig {
            device_type: DeviceType::MLX,
            mlx_config: Some(mlx_config.clone()),
            ..Default::default()
        },
        optimizations: OptimizationConfig {
            use_flash_attention: true,
            operator_fusion: true,
            graph_optimization: true,
            mixed_precision: args.use_fp16,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut engine = InferenceEngine::new(engine_config);
    engine.load_model(Arc::new(model)).await?;
    
    // Step 3: Warm up GPU
    println!("🔥 Warming up GPU...");
    warmup_gpu(&engine).await?;
    
    // Step 4: Monitor GPU memory
    if args.monitor_gpu {
        let mem_before = MLXDevice::memory_usage()?;
        println!("💾 GPU Memory Usage:");
        println!("  Allocated: {:.1} MB", mem_before.allocated_mb);
        println!("  Cached: {:.1} MB", mem_before.cached_mb);
        println!("  Available: {:.1} MB", mem_before.available_mb);
        println!();
    }
    
    // Step 5: Run inference with batch processing
    if args.batch_size > 1 {
        println!("📦 Running batch inference (batch size: {})...", args.batch_size);
        run_batch_inference(&engine, &args).await?;
    } else {
        println!("🧠 Running single inference...");
        run_single_inference(&engine, &args).await?;
    }
    
    // Step 6: Performance analysis
    if args.monitor_gpu {
        println!("\n📊 Performance Analysis:");
        let mem_after = MLXDevice::memory_usage()?;
        println!("  Peak Memory: {:.1} MB", mem_after.peak_mb);
        println!("  Memory Bandwidth: {:.1} GB/s", mem_after.bandwidth_gbps);
        
        if let Ok(profile) = MLXDevice::get_profile() {
            println!("\n⏱️ Operation Timings:");
            for (op_name, timing) in profile.top_operations(5) {
                println!("  {}: {:.2} ms ({:.1}%)", 
                    op_name, timing.duration_ms, timing.percentage);
            }
        }
    }
    
    Ok(())
}

async fn create_mlx_model(
    loader: &GGUFLoader,
    config: MLXConfig,
) -> Result<Box<dyn Model>, Box<dyn std::error::Error>> {
    let start = Instant::now();
    
    // Create MLX backend
    let backend = MLXBackend::new(config)?;
    
    // Load model with MLX optimizations
    let model = backend.load_model(loader).await?;
    
    let load_time = start.elapsed();
    info!("Model loaded in {:?}", load_time);
    
    // Optimize model for MLX
    if let Some(mlx_model) = model.as_any().downcast_ref::<MLXModel>() {
        println!("🔧 Applying MLX optimizations:");
        
        // Quantize to FP16 if requested
        if config.use_fp16 {
            mlx_model.convert_to_fp16()?;
            println!("  ✓ Converted to FP16 precision");
        }
        
        // Compile computation graphs
        if config.compile_graphs {
            mlx_model.compile_graphs()?;
            println!("  ✓ Compiled computation graphs");
        }
        
        // Enable tensor cores
        mlx_model.enable_tensor_cores()?;
        println!("  ✓ Enabled tensor cores");
    }
    
    Ok(model)
}

async fn warmup_gpu(engine: &InferenceEngine) -> Result<(), Box<dyn std::error::Error>> {
    let warmup_tokens = vec![1, 2, 3, 4, 5]; // Simple warmup sequence
    
    let config = SessionConfig {
        max_seq_length: 10,
        temperature: 1.0,
        ..Default::default()
    };
    
    let session = engine.create_session(config).await?;
    
    // Run a few warmup iterations
    for i in 0..3 {
        let start = Instant::now();
        let _ = session.infer(&warmup_tokens).await;
        let duration = start.elapsed();
        info!("Warmup iteration {}: {:?}", i + 1, duration);
    }
    
    session.clear().await;
    Ok(())
}

async fn run_single_inference(
    engine: &InferenceEngine,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = SessionConfig {
        max_seq_length: args.max_tokens,
        temperature: 0.8,
        top_p: 0.9,
        ..Default::default()
    };
    
    let session = engine.create_session(config).await?;
    
    // Tokenize input
    let tokens = simple_tokenize(&args.prompt);
    println!("Input tokens: {} tokens", tokens.len());
    
    // Measure inference time
    let start = Instant::now();
    let result = session.infer(&tokens).await?;
    let inference_time = start.elapsed();
    
    println!("\n✅ Inference Results:");
    println!("  Time: {:?}", inference_time);
    println!("  Tokens/sec: {:.1}", tokens.len() as f64 / inference_time.as_secs_f64());
    println!("  Output shape: {:?}", result.logits_shape);
    
    // Get MLX-specific metrics
    if let Some(mlx_metrics) = result.backend_metrics.get("mlx") {
        println!("\n🍎 MLX Metrics:");
        if let Some(gpu_time) = mlx_metrics.get("gpu_time_ms") {
            println!("  GPU Time: {:.2} ms", gpu_time);
        }
        if let Some(memory_peak) = mlx_metrics.get("memory_peak_mb") {
            println!("  Peak Memory: {:.1} MB", memory_peak);
        }
        if let Some(ops_fused) = mlx_metrics.get("operations_fused") {
            println!("  Operations Fused: {}", ops_fused);
        }
    }
    
    Ok(())
}

async fn run_batch_inference(
    engine: &InferenceEngine,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create batch of prompts
    let prompts = vec![
        "The future of AI is",
        "Machine learning will",
        "Neural networks are",
        "Deep learning enables",
    ];
    
    let config = SessionConfig {
        max_seq_length: args.max_tokens,
        batch_size: args.batch_size,
        temperature: 0.8,
        ..Default::default()
    };
    
    let session = engine.create_session(config).await?;
    
    // Tokenize all prompts
    let token_batches: Vec<Vec<u32>> = prompts.iter()
        .take(args.batch_size)
        .map(|p| simple_tokenize(p))
        .collect();
    
    println!("Batch size: {}", token_batches.len());
    println!("Sequences: {:?}", prompts.iter().take(args.batch_size).collect::<Vec<_>>());
    
    // Run batch inference
    let start = Instant::now();
    let results = session.infer_batch(&token_batches).await?;
    let batch_time = start.elapsed();
    
    println!("\n✅ Batch Inference Results:");
    println!("  Total time: {:?}", batch_time);
    println!("  Time per sequence: {:?}", batch_time / args.batch_size as u32);
    println!("  Total tokens/sec: {:.1}", 
        token_batches.iter().map(|t| t.len()).sum::<usize>() as f64 / batch_time.as_secs_f64());
    
    // Show individual results
    for (i, result) in results.iter().enumerate() {
        println!("\n  Sequence {}: {} tokens generated", i + 1, result.generated_tokens.len());
    }
    
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args::default();
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
            "--prompt" => {
                if i + 1 < cmd_args.len() {
                    args.prompt = cmd_args[i + 1].clone();
                    i += 2;
                } else {
                    return Err("Missing prompt".into());
                }
            }
            "--max-tokens" => {
                if i + 1 < cmd_args.len() {
                    args.max_tokens = cmd_args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing max tokens".into());
                }
            }
            "--batch-size" => {
                if i + 1 < cmd_args.len() {
                    args.batch_size = cmd_args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing batch size".into());
                }
            }
            "--fp32" => {
                args.use_fp16 = false;
                i += 1;
            }
            "--no-monitor" => {
                args.monitor_gpu = false;
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
    println!("🍎 Woolly MLX Backend Example

USAGE:
    mlx_backend --model <MODEL> [OPTIONS]

REQUIRED:
    --model <MODEL>          Path to the GGUF model file

OPTIONS:
    --prompt <TEXT>          Input prompt [default: \"The future of AI is\"]
    --max-tokens <NUM>       Maximum tokens to generate [default: 100]
    --batch-size <NUM>       Batch size for inference [default: 1]
    --fp32                   Use FP32 precision instead of FP16
    --no-monitor            Disable GPU monitoring
    --help                   Show this help message

EXAMPLES:
    # Basic usage with FP16
    mlx_backend --model llama-7b.gguf

    # Batch inference
    mlx_backend --model llama-7b.gguf --batch-size 4

    # Custom prompt with FP32
    mlx_backend --model llama-7b.gguf --prompt \"Explain quantum computing\" --fp32
");
}

fn simple_tokenize(text: &str) -> Vec<u32> {
    text.split_whitespace()
        .enumerate()
        .map(|(i, _)| (i as u32) % 1000 + 1)
        .collect()
}

// Mock types for the example
// In a real implementation, these would come from woolly-mlx crate

mod woolly_mlx {
    use super::*;
    
    #[derive(Clone)]
    pub struct MLXConfig {
        pub device: MLXDevice,
        pub use_fp16: bool,
        pub use_unified_memory: bool,
        pub metal_capture: bool,
        pub profile_ops: bool,
        pub max_memory_gb: Option<f32>,
        pub cache_size_mb: usize,
        pub compile_graphs: bool,
    }
    
    #[derive(Clone, Default)]
    pub struct MLXDevice;
    
    impl MLXDevice {
        pub fn is_available() -> bool { true }
        
        pub fn info() -> Result<DeviceInfo, Box<dyn std::error::Error>> {
            Ok(DeviceInfo {
                name: "Apple M2 Max".to_string(),
                memory_gb: 32.0,
                compute_units: 38,
                mlx_version: "0.14.0".to_string(),
            })
        }
        
        pub fn memory_usage() -> Result<MemoryUsage, Box<dyn std::error::Error>> {
            Ok(MemoryUsage {
                allocated_mb: 2048.0,
                cached_mb: 512.0,
                available_mb: 30720.0,
                peak_mb: 2560.0,
                bandwidth_gbps: 400.0,
            })
        }
        
        pub fn get_profile() -> Result<Profile, Box<dyn std::error::Error>> {
            Ok(Profile::default())
        }
    }
    
    pub struct DeviceInfo {
        pub name: String,
        pub memory_gb: f32,
        pub compute_units: u32,
        pub mlx_version: String,
    }
    
    pub struct MemoryUsage {
        pub allocated_mb: f32,
        pub cached_mb: f32,
        pub available_mb: f32,
        pub peak_mb: f32,
        pub bandwidth_gbps: f32,
    }
    
    #[derive(Default)]
    pub struct Profile;
    
    impl Profile {
        pub fn top_operations(&self, _n: usize) -> Vec<(&str, OpTiming)> {
            vec![
                ("matmul", OpTiming { duration_ms: 12.5, percentage: 45.0 }),
                ("attention", OpTiming { duration_ms: 8.3, percentage: 30.0 }),
                ("layer_norm", OpTiming { duration_ms: 3.2, percentage: 11.5 }),
            ]
        }
    }
    
    pub struct OpTiming {
        pub duration_ms: f32,
        pub percentage: f32,
    }
    
    pub struct MLXBackend {
        config: MLXConfig,
    }
    
    impl MLXBackend {
        pub fn new(config: MLXConfig) -> Result<Self, Box<dyn std::error::Error>> {
            Ok(Self { config })
        }
        
        pub async fn load_model(&self, _loader: &GGUFLoader) -> Result<Box<dyn Model>, Box<dyn std::error::Error>> {
            Ok(Box::new(MLXModel))
        }
    }
    
    pub struct MLXModel;
    
    impl MLXModel {
        pub fn convert_to_fp16(&self) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
        pub fn compile_graphs(&self) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
        pub fn enable_tensor_cores(&self) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
    }
}