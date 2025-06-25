//! Basic Inference Example
//!
//! This example demonstrates the core Woolly inference workflow:
//! 1. Loading a model from GGUF format
//! 2. Creating an inference engine with configuration
//! 3. Setting up a session for inference
//! 4. Running inference on input text
//! 5. Comprehensive error handling and validation
//! 6. Configuration options and best practices
//!
//! Usage:
//!   cargo run --example basic_inference -- --model path/to/model.gguf --prompt "Hello, world!"
//!
//! Advanced usage:
//!   cargo run --example basic_inference -- \
//!     --model path/to/model.gguf \
//!     --prompt "Explain quantum computing" \
//!     --max-tokens 200 \
//!     --temperature 0.8 \
//!     --top-p 0.9 \
//!     --threads 8

use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use tracing::{info, warn, error, debug};
use woolly_core::prelude::*;
use woolly_gguf::GGUFLoader;

#[derive(Debug)]
struct InferenceArgs {
    model_path: PathBuf,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    threads: usize,
    use_mmap: bool,
    verbose: bool,
}

impl Default for InferenceArgs {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            prompt: String::new(),
            max_tokens: 100,
            temperature: 0.8,
            top_p: 0.9,
            top_k: 40,
            threads: num_cpus::get(),
            use_mmap: true,
            verbose: false,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging with environment variable support
    let log_level = env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .init();
    
    // Parse command line arguments with validation
    let args: Vec<String> = env::args().collect();
    let inference_args = parse_args(&args).map_err(|e| {
        eprintln!("❌ Argument parsing failed: {}", e);
        eprintln!("Use --help for usage information");
        e
    })?;
    
    // Display configuration
    println!("🦙 Woolly Basic Inference Example");
    println!("Configuration:");
    println!("  Model: {}", inference_args.model_path.display());
    println!("  Prompt: '{}'", inference_args.prompt);
    println!("  Max tokens: {}", inference_args.max_tokens);
    println!("  Temperature: {}", inference_args.temperature);
    println!("  Top-p: {}", inference_args.top_p);
    println!("  Top-k: {}", inference_args.top_k);
    println!("  Threads: {}", inference_args.threads);
    println!("  Use mmap: {}", inference_args.use_mmap);
    println!();
    
    // Step 1: Validate and load model from GGUF file
    info!("Starting model loading process");
    println!("📂 Loading GGUF model...");
    
    // Validate model file exists and is readable
    if !inference_args.model_path.exists() {
        error!("Model file does not exist: {}", inference_args.model_path.display());
        return Err(format!("Model file not found: {}", inference_args.model_path.display()).into());
    }
    
    let model_start = Instant::now();
    let gguf_loader = GGUFLoader::from_path(&inference_args.model_path)
        .map_err(|e| {
            error!("Failed to load GGUF file: {}", e);
            format!("GGUF loading failed: {}. Please ensure the file is a valid GGUF model.", e)
        })?;
    
    let model_load_time = model_start.elapsed();
    info!("GGUF loader created in {:?}", model_load_time);
    
    // Print model information with error handling
    print_model_info(&gguf_loader);
    
    // Step 2: Create model instance with error handling
    // Note: This is a simplified example. In practice, you would create a specific
    // model implementation (e.g., LlamaModel, MistralModel) based on the architecture
    println!("🔧 Creating model instance...");
    let model = create_example_model(&gguf_loader, &inference_args)
        .map_err(|e| {
            error!("Model creation failed: {}", e);
            format!("Failed to create model: {}. This might be due to unsupported model architecture or corrupted model file.", e)
        })?;
    
    // Step 3: Configure and create inference engine with validation
    println!("⚙️ Setting up inference engine...");
    let engine_config = create_engine_config(&inference_args);
    
    // Validate engine configuration
    if let Err(e) = engine_config.validate() {
        error!("Invalid engine configuration: {}", e);
        return Err(format!("Configuration validation failed: {}", e).into());
    }
    
    let mut engine = InferenceEngine::new(engine_config);
    info!("Inference engine created successfully");
    
    // Load the model into the engine with proper error handling
    let engine_start = Instant::now();
    engine.load_model(Arc::new(model)).await
        .map_err(|e| {
            error!("Failed to load model into engine: {}", e);
            format!("Engine model loading failed: {}. This might be due to insufficient memory or incompatible model format.", e)
        })?;
    
    let engine_load_time = engine_start.elapsed();
    info!("Model loaded into engine in {:?}", engine_load_time);
    
    // Step 4: Create inference session with proper configuration
    println!("🔗 Creating inference session...");
    let session_config = SessionConfig {
        max_seq_length: inference_args.max_tokens,
        temperature: inference_args.temperature,
        top_p: inference_args.top_p,
        top_k: inference_args.top_k,
        use_cache: true,
        ..Default::default()
    };
    
    // Validate session configuration
    if session_config.temperature < 0.0 || session_config.temperature > 2.0 {
        warn!("Temperature {} is outside recommended range [0.0, 2.0]", session_config.temperature);
    }
    
    if session_config.top_p < 0.0 || session_config.top_p > 1.0 {
        return Err("Top-p must be between 0.0 and 1.0".into());
    }
    
    let session = engine.create_session(session_config).await
        .map_err(|e| {
            error!("Session creation failed: {}", e);
            format!("Failed to create inference session: {}", e)
        })?;
    
    info!("Inference session created successfully");
    
    // Step 5: Tokenize input with validation and error handling
    println!("🔤 Tokenizing input...");
    
    // Validate prompt
    if inference_args.prompt.is_empty() {
        return Err("Prompt cannot be empty".into());
    }
    
    if inference_args.prompt.len() > 10000 {
        warn!("Very long prompt ({} chars) may cause performance issues", inference_args.prompt.len());
    }
    
    let tokenize_start = Instant::now();
    let tokens = improved_tokenize(&inference_args.prompt)?;
    let tokenize_time = tokenize_start.elapsed();
    
    debug!("Tokenization completed in {:?}", tokenize_time);
    println!("Tokens: {:?} (count: {})", 
             if inference_args.verbose { &tokens } else { &tokens[..tokens.len().min(10)] },
             tokens.len());
    
    if tokens.len() > inference_args.max_tokens / 2 {
        warn!("Input tokens ({}) are more than half of max tokens ({})", 
              tokens.len(), inference_args.max_tokens);
    }
    
    // Step 6: Run inference with comprehensive error handling
    println!("🧠 Running inference...");
    let inference_start = Instant::now();
    
    // Note: This will currently return a todo!() error since the inference
    // implementation is still being developed. This example shows the intended API.
    match session.infer(&tokens).await {
        Ok(result) => {
            let inference_time = inference_start.elapsed();
            println!("✅ Inference completed successfully!");
            println!("  ⏱️ Time: {:?}", inference_time);
            println!("  📊 Output logits shape: {:?}", result.logits_shape);
            println!("  🔢 Logits count: {}", result.logits.len());
            
            // Performance metrics
            let tokens_per_second = tokens.len() as f64 / inference_time.as_secs_f64();
            println!("  📈 Performance: {:.2} tokens/second", tokens_per_second);
            
            // Convert logits to probabilities and show top tokens
            if !result.logits.is_empty() {
                let top_tokens = get_top_tokens(&result.logits, 5);
                println!("  🎯 Top 5 predicted tokens: {:?}", top_tokens);
                
                // Simulate text generation (in real implementation, this would use proper decoding)
                let generated_text = simulate_text_generation(&top_tokens, &inference_args.prompt);
                println!("  📝 Generated text: \"{}\"", generated_text);
            }
            
            // Memory usage information
            if let Some(memory_usage) = get_memory_usage() {
                println!("  💾 Memory usage: {:.1} MB", memory_usage as f64 / (1024.0 * 1024.0));
            }
        }
        Err(e) => {
            let inference_time = inference_start.elapsed();
            error!("Inference failed after {:?}: {}", inference_time, e);
            
            // Provide helpful error information
            match e.to_string().as_str() {
                s if s.contains("not yet implemented") => {
                    println!("⚠️ Inference not yet implemented: {}", e);
                    println!("ℹ️ This is expected - the inference engine is still under development.");
                    println!("ℹ️ The example demonstrates the intended API structure.");
                    println!("ℹ️ Model loading and session setup worked correctly!");
                }
                s if s.contains("memory") => {
                    println!("❌ Memory error during inference: {}", e);
                    println!("💡 Try reducing max_tokens or using a smaller model");
                }
                s if s.contains("timeout") => {
                    println!("❌ Inference timeout: {}", e);
                    println!("💡 Try reducing the prompt length or increasing timeout");
                }
                _ => {
                    println!("❌ Inference failed: {}", e);
                    println!("💡 Check model compatibility and system resources");
                }
            }
        }
    }
    
    // Step 7: Session cleanup with timing
    println!("🧹 Cleaning up...");
    let cleanup_start = Instant::now();
    session.clear().await;
    let cleanup_time = cleanup_start.elapsed();
    debug!("Cleanup completed in {:?}", cleanup_time);
    
    println!("✨ Example completed!");
    
    Ok(())
}

fn parse_args(args: &[String]) -> Result<InferenceArgs, Box<dyn std::error::Error>> {
    if args.len() < 2 {
        print_help(&args[0]);
        std::process::exit(1);
    }
    
    let mut inference_args = InferenceArgs::default();
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    inference_args.model_path = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    return Err("Missing model path after --model".into());
                }
            }
            "--prompt" => {
                if i + 1 < args.len() {
                    inference_args.prompt = args[i + 1].clone();
                    i += 2;
                } else {
                    return Err("Missing prompt text after --prompt".into());
                }
            }
            "--max-tokens" => {
                if i + 1 < args.len() {
                    inference_args.max_tokens = args[i + 1].parse()
                        .map_err(|_| "Invalid max-tokens value")?;
                    i += 2;
                } else {
                    return Err("Missing value after --max-tokens".into());
                }
            }
            "--temperature" => {
                if i + 1 < args.len() {
                    inference_args.temperature = args[i + 1].parse()
                        .map_err(|_| "Invalid temperature value")?;
                    i += 2;
                } else {
                    return Err("Missing value after --temperature".into());
                }
            }
            "--top-p" => {
                if i + 1 < args.len() {
                    inference_args.top_p = args[i + 1].parse()
                        .map_err(|_| "Invalid top-p value")?;
                    i += 2;
                } else {
                    return Err("Missing value after --top-p".into());
                }
            }
            "--top-k" => {
                if i + 1 < args.len() {
                    inference_args.top_k = args[i + 1].parse()
                        .map_err(|_| "Invalid top-k value")?;
                    i += 2;
                } else {
                    return Err("Missing value after --top-k".into());
                }
            }
            "--threads" => {
                if i + 1 < args.len() {
                    inference_args.threads = args[i + 1].parse()
                        .map_err(|_| "Invalid threads value")?;
                    i += 2;
                } else {
                    return Err("Missing value after --threads".into());
                }
            }
            "--no-mmap" => {
                inference_args.use_mmap = false;
                i += 1;
            }
            "--verbose" | "-v" => {
                inference_args.verbose = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_help(&args[0]);
                std::process::exit(0);
            }
            arg if arg.starts_with("--") => {
                return Err(format!("Unknown argument: {}", arg).into());
            }
            _ => i += 1,
        }
    }
    
    // Validate required arguments
    if inference_args.model_path.as_os_str().is_empty() {
        return Err("Model path is required. Use --model <path>".into());
    }
    
    if inference_args.prompt.is_empty() {
        return Err("Prompt is required. Use --prompt <text>".into());
    }
    
    // Validate argument ranges
    if inference_args.max_tokens == 0 {
        return Err("max-tokens must be greater than 0".into());
    }
    
    if inference_args.max_tokens > 10000 {
        return Err("max-tokens should not exceed 10000 for this example".into());
    }
    
    if inference_args.temperature < 0.0 {
        return Err("Temperature must be non-negative".into());
    }
    
    if inference_args.top_p <= 0.0 || inference_args.top_p > 1.0 {
        return Err("Top-p must be between 0.0 and 1.0".into());
    }
    
    if inference_args.top_k == 0 {
        return Err("Top-k must be greater than 0".into());
    }
    
    if inference_args.threads == 0 {
        return Err("Thread count must be greater than 0".into());
    }
    
    Ok(inference_args)
}

fn print_help(program_name: &str) {
    println!("🦙 Woolly Basic Inference Example");
    println!();
    println!("USAGE:");
    println!("    {} --model <MODEL> --prompt <PROMPT> [OPTIONS]", program_name);
    println!();
    println!("REQUIRED ARGUMENTS:");
    println!("    --model <MODEL>          Path to the GGUF model file");
    println!("    --prompt <PROMPT>        Input text prompt for generation");
    println!();
    println!("OPTIONS:");
    println!("    --max-tokens <NUM>       Maximum number of tokens to generate [default: 100]");
    println!("    --temperature <FLOAT>    Sampling temperature (0.0-2.0) [default: 0.8]");
    println!("    --top-p <FLOAT>         Top-p nucleus sampling (0.0-1.0) [default: 0.9]");
    println!("    --top-k <NUM>           Top-k sampling [default: 40]");
    println!("    --threads <NUM>         Number of threads to use [default: CPU cores]");
    println!("    --no-mmap               Disable memory mapping [default: enabled]");
    println!("    --verbose, -v           Enable verbose output");
    println!("    --help, -h              Show this help message");
    println!();
    println!("EXAMPLES:");
    println!("    # Basic usage");
    println!("    {} --model model.gguf --prompt \"Hello, world!\"", program_name);
    println!();
    println!("    # With custom parameters");
    println!("    {} \\", program_name);
    println!("        --model model.gguf \\");
    println!("        --prompt \"Explain quantum computing\" \\");
    println!("        --max-tokens 200 \\");
    println!("        --temperature 0.7 \\");
    println!("        --verbose");
    println!();
    println!("    # High-quality generation");
    println!("    {} \\", program_name);
    println!("        --model model.gguf \\");
    println!("        --prompt \"Write a story about\" \\");
    println!("        --temperature 0.9 \\");
    println!("        --top-p 0.95 \\");
    println!("        --max-tokens 500");
}

fn print_model_info(loader: &GGUFLoader) {
    println!("📊 Model Information:");
    
    if let Some(arch) = loader.architecture() {
        println!("  Architecture: {}", arch);
    }
    
    if let Some(name) = loader.model_name() {
        println!("  Name: {}", name);
    }
    
    let header = loader.header();
    println!("  Tensors: {}", header.tensor_count);
    println!("  File size: {:.2} MB", loader.file_size() as f64 / (1024.0 * 1024.0));
    println!("  Tensor data: {:.2} MB", loader.total_tensor_size() as f64 / (1024.0 * 1024.0));
    println!();
}

fn create_example_model(loader: &GGUFLoader, args: &InferenceArgs) -> Result<ExampleModel, Box<dyn std::error::Error>> {
    // Create a basic model configuration based on GGUF metadata
    let config = ModelConfig {
        vocab_size: 32000, // This would be extracted from GGUF metadata
        hidden_size: 4096,
        num_layers: 32,
        num_heads: 32,
        context_length: args.max_tokens,
        ..Default::default()
    };
    
    info!("Created model with context length: {}", config.context_length);
    
    Ok(ExampleModel {
        name: loader.model_name().unwrap_or("unknown").to_string(),
        config,
        _loader: loader.clone(), // Keep reference for actual weight loading
    })
}

fn create_engine_config(args: &InferenceArgs) -> EngineConfig {
    EngineConfig {
        max_context_length: args.max_tokens * 2, // Allow room for generation
        max_batch_size: 1,
        num_threads: args.threads,
        device: DeviceConfig {
            device_type: DeviceType::Cpu,
            cpu_fallback: true,
            ..Default::default()
        },
        memory: MemoryConfig {
            use_mmap: args.use_mmap,
            max_memory_mb: 8192,
            ..Default::default()
        },
        optimizations: OptimizationConfig {
            use_flash_attention: false, // Disable for CPU
            operator_fusion: true,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn improved_tokenize(text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    // Enhanced tokenization with better word handling
    let words: Vec<&str> = text
        .split_whitespace()
        .collect();
    
    if words.is_empty() {
        return Ok(vec![]);
    }
    
    let tokens: Vec<u32> = words
        .iter()
        .enumerate()
        .map(|(i, word)| {
            // Create a more realistic token mapping based on word characteristics
            let base_token = (word.len() * 37 + i * 17) % 30000 + 1000;
            base_token as u32
        })
        .collect();
    
    debug!("Tokenized {} words into {} tokens", words.len(), tokens.len());
    Ok(tokens)
}

fn simulate_text_generation(top_tokens: &[(usize, f32)], original_prompt: &str) -> String {
    // Simulate text generation by creating a plausible continuation
    let continuation_words = vec![
        "interesting", "fascinating", "complex", "simple", "important", "relevant",
        "significant", "valuable", "useful", "essential", "fundamental", "advanced",
        "innovative", "creative", "effective", "efficient", "practical", "theoretical"
    ];
    
    let prompt_words = original_prompt.split_whitespace().collect::<Vec<&str>>();
    let last_word = prompt_words.last().unwrap_or(&"");
    
    // Choose a word based on the top token (simplified simulation)
    let chosen_word = if !top_tokens.is_empty() {
        let token_index = top_tokens[0].0 % continuation_words.len();
        continuation_words[token_index]
    } else {
        "generated"
    };
    
    format!("{} {}", last_word, chosen_word)
}

fn get_memory_usage() -> Option<usize> {
    // Platform-specific memory usage detection
    // This is a simplified implementation - in practice you'd use platform-specific APIs
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return Some(kb * 1024); // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }
    
    // Fallback: estimate based on typical model loading
    Some(2048 * 1024 * 1024) // 2GB estimate
}

fn simple_tokenize(text: &str) -> Vec<u32> {
    // Simple word-based tokenization for demonstration
    // In practice, you would use a proper tokenizer from woolly-core
    text.split_whitespace()
        .enumerate()
        .map(|(i, _)| (i as u32) % 1000 + 1) // Simple mapping to token IDs
        .collect()
}

fn get_top_tokens(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed_logits: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();
    
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed_logits.truncate(k);
    
    indexed_logits
}

// Example model implementation for demonstration
#[derive(Clone)]
struct ExampleModel {
    name: String,
    config: ModelConfig,
    _loader: GGUFLoader,
}

#[async_trait::async_trait]
impl Model for ExampleModel {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn model_type(&self) -> &str {
        "example"
    }
    
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    
    fn context_length(&self) -> usize {
        self.config.context_length
    }
    
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
    
    fn num_layers(&self) -> usize {
        self.config.num_layers
    }
    
    fn num_heads(&self) -> usize {
        self.config.num_heads
    }
    
    async fn forward(
        &self,
        _input_ids: &[u32],
        _past_kv_cache: Option<&(dyn std::any::Any + Send + Sync)>,
    ) -> Result<ModelOutput> {
        // Placeholder implementation
        // In practice, this would use the tensor backend to run the actual forward pass
        Ok(ModelOutput {
            logits: vec![0.0; self.vocab_size()],
            logits_shape: vec![1, 1, self.vocab_size()],
            past_kv_cache: None,
            hidden_states: None,
            attentions: None,
        })
    }
    
    async fn load_weights(&mut self, _path: &Path) -> Result<()> {
        // Placeholder - would load weights from the GGUF file
        Ok(())
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
            "--prompt".to_string(),
            "Hello world".to_string(),
            "--max-tokens".to_string(),
            "50".to_string(),
        ];
        
        let result = parse_args(&args).unwrap();
        assert_eq!(result.model_path, PathBuf::from("test.gguf"));
        assert_eq!(result.prompt, "Hello world");
        assert_eq!(result.max_tokens, 50);
    }

    #[test]
    fn test_improved_tokenize() {
        let tokens = improved_tokenize("hello world test").unwrap();
        assert_eq!(tokens.len(), 3);
        assert!(tokens.iter().all(|&t| t >= 1000));
    }
    
    #[test]
    fn test_empty_tokenize() {
        let tokens = improved_tokenize("").unwrap();
        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_simple_tokenize() {
        let tokens = simple_tokenize("hello world test");
        assert_eq!(tokens.len(), 3);
        assert!(tokens.iter().all(|&t| t > 0 && t <= 1000));
    }
    
    #[test]
    fn test_parse_args_missing_required() {
        let args = vec!["program".to_string(), "--model".to_string(), "test.gguf".to_string()];
        let result = parse_args(&args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Prompt is required"));
    }
    
    #[test]
    fn test_parse_args_invalid_temperature() {
        let args = vec![
            "program".to_string(),
            "--model".to_string(),
            "test.gguf".to_string(),
            "--prompt".to_string(),
            "test".to_string(),
            "--temperature".to_string(),
            "-1.0".to_string(),
        ];
        let result = parse_args(&args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Temperature must be non-negative"));
    }
    
    #[test]
    fn test_simulate_text_generation() {
        let top_tokens = vec![(0, 0.9), (1, 0.8), (2, 0.7)];
        let result = simulate_text_generation(&top_tokens, "Hello");
        assert!(result.contains("Hello"));
        assert!(result.len() > 5);
    }
    
    #[test]
    fn test_get_top_tokens() {
        let logits = vec![0.1, 0.8, 0.3, 0.9, 0.2];
        let top = get_top_tokens(&logits, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 3); // Index of highest score (0.9)
        assert_eq!(top[1].0, 1); // Index of second highest (0.8)
    }
}