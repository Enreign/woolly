# Getting Started with Woolly

Welcome to Woolly! This guide will help you get up and running with high-performance LLM inference in Rust.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Loading Models](#loading-models)
4. [Running Inference](#running-inference)
5. [Configuration](#configuration)
6. [Platform-Specific Features](#platform-specific-features)
7. [Performance Tips](#performance-tips)
8. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- **Rust 1.75 or later**: Install from [rustup.rs](https://rustup.rs/)
- **C++ compiler**: For building native dependencies
  - macOS: Xcode Command Line Tools
  - Linux: gcc or clang
  - Windows: Visual Studio 2019 or later
- **CMake** (optional): For building certain features

### Adding Woolly to Your Project

Add Woolly to your `Cargo.toml`:

```toml
[dependencies]
# Core functionality
woolly-core = "0.1"
woolly-gguf = "0.1"

# Optional features
woolly-mcp = "0.1"   # Model Context Protocol support
woolly-mlx = "0.1"   # Apple MLX support (macOS only)
woolly-bench = "0.1" # Benchmarking tools
```

### Feature Flags

Enable specific features based on your needs:

```toml
[dependencies]
woolly-core = { version = "0.1", features = ["mlx", "metrics"] }
```

Available features:
- `mlx`: Apple MLX support for Apple Silicon
- `cuda`: NVIDIA GPU support (coming soon)
- `metrics`: Performance monitoring
- `mcp`: Model Context Protocol server
- `distributed`: Multi-node inference

## Quick Start

Here's a minimal example to get you started:

```rust
use woolly_core::prelude::*;
use woolly_gguf::GGUFLoader;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a GGUF model
    let loader = GGUFLoader::from_path("model.gguf")?;
    let model = create_model_from_gguf(&loader)?;
    
    // Create inference engine
    let config = EngineConfig::default();
    let mut engine = InferenceEngine::new(config);
    engine.load_model(Arc::new(model)).await?;
    
    // Create a session
    let session = engine.create_session(SessionConfig::default()).await?;
    
    // Run inference
    let tokens = vec![1, 2, 3, 4, 5]; // Your tokenized input
    let result = session.infer(&tokens).await?;
    
    println!("Generated {} tokens", result.tokens.len());
    
    Ok(())
}
```

## Loading Models

### Supported Formats

Woolly currently supports GGUF format models, compatible with llama.cpp:

```rust
// Load from file path
let loader = GGUFLoader::from_path("models/llama-7b.gguf")?;

// Load with memory mapping (recommended for large models)
let loader = GGUFLoader::from_path_mmap("models/llama-70b.gguf")?;

// Load from bytes
let model_data: Vec<u8> = std::fs::read("model.gguf")?;
let loader = GGUFLoader::from_bytes(&model_data)?;
```

### Model Information

Get information about a loaded model:

```rust
// Model architecture
if let Some(arch) = loader.architecture() {
    println!("Architecture: {}", arch); // e.g., "llama", "mistral"
}

// Model metadata
println!("Tensors: {}", loader.header().tensor_count);
println!("File size: {:.1} GB", loader.file_size() as f64 / 1e9);

// Supported quantizations
let quant_info = loader.quantization_info();
println!("Quantization: {:?}", quant_info);
```

### Creating Model Instances

Woolly automatically detects the model architecture:

```rust
// Automatic model creation based on architecture
let model = create_model_from_gguf(&loader)?;

// Or manually specify model type
let model = match loader.architecture() {
    Some("llama") => LlamaModel::from_gguf(&loader)?,
    Some("mistral") => MistralModel::from_gguf(&loader)?,
    Some("phi") => PhiModel::from_gguf(&loader)?,
    _ => return Err("Unsupported model architecture".into()),
};
```

## Running Inference

### Basic Inference

```rust
// Configure the session
let config = SessionConfig {
    max_seq_length: 512,
    temperature: 0.8,
    top_p: 0.9,
    top_k: 40,
    use_cache: true,
    ..Default::default()
};

let session = engine.create_session(config).await?;

// Tokenize your input (using your preferred tokenizer)
let prompt = "The capital of France is";
let tokens = tokenizer.encode(prompt)?;

// Run inference
let result = session.infer(&tokens).await?;

// Process results
for token_id in result.tokens {
    let token_text = tokenizer.decode(&[token_id])?;
    print!("{}", token_text);
}
```

### Streaming Inference

For real-time text generation:

```rust
use futures::StreamExt;

let mut stream = session.infer_stream(&tokens).await?;

while let Some(token_result) = stream.next().await {
    match token_result {
        Ok(token) => {
            let text = tokenizer.decode(&[token.id])?;
            print!("{}", text);
            std::io::stdout().flush()?;
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### Batch Inference

Process multiple prompts efficiently:

```rust
let prompts = vec![
    "The weather today is",
    "AI will help us",
    "The future of technology",
];

// Tokenize all prompts
let token_batches: Vec<Vec<u32>> = prompts.iter()
    .map(|p| tokenizer.encode(p))
    .collect::<Result<Vec<_>, _>>()?;

// Configure for batch processing
let config = SessionConfig {
    batch_size: prompts.len(),
    ..Default::default()
};

let session = engine.create_session(config).await?;
let results = session.infer_batch(&token_batches).await?;

for (i, result) in results.iter().enumerate() {
    println!("Prompt {}: Generated {} tokens", i + 1, result.tokens.len());
}
```

## Configuration

### Engine Configuration

```rust
let config = EngineConfig {
    // Threading
    num_threads: num_cpus::get(), // Use all CPU cores
    
    // Memory settings
    memory: MemoryConfig {
        use_mmap: true,           // Memory-mapped file loading
        max_memory_mb: 8192,      // Maximum memory usage
        cache_size_mb: 1024,      // KV cache size
        use_memory_pool: true,    // Pre-allocated memory pools
    },
    
    // Device configuration
    device: DeviceConfig {
        device_type: DeviceType::Cpu, // or DeviceType::MLX for Apple Silicon
        cpu_features: CpuFeatures::auto_detect(),
    },
    
    // Optimizations
    optimizations: OptimizationConfig {
        use_flash_attention: false,  // CPU: false, GPU: true
        operator_fusion: true,
        graph_optimization: true,
        mixed_precision: false,
    },
    
    ..Default::default()
};
```

### Session Configuration

```rust
let config = SessionConfig {
    // Generation parameters
    max_seq_length: 2048,      // Maximum sequence length
    temperature: 0.8,          // Sampling temperature (0.0 - 2.0)
    top_p: 0.9,               // Nucleus sampling threshold
    top_k: 40,                // Top-k sampling
    repetition_penalty: 1.1,   // Penalize repetitions
    
    // Performance settings
    batch_size: 1,            // Number of sequences to process
    use_cache: true,          // Enable KV caching
    
    // Stopping criteria
    stop_tokens: vec![2],     // EOS token ID
    max_new_tokens: 100,      // Maximum tokens to generate
    
    ..Default::default()
};
```

## Platform-Specific Features

### Apple Silicon (MLX)

For optimal performance on Apple Silicon Macs:

```rust
#[cfg(feature = "mlx")]
use woolly_mlx::{MLXBackend, MLXConfig};

let mlx_config = MLXConfig {
    use_fp16: true,           // Use half-precision
    use_unified_memory: true, // Leverage unified memory
    compile_graphs: true,     // JIT compilation
    ..Default::default()
};

let config = EngineConfig {
    device: DeviceConfig {
        device_type: DeviceType::MLX,
        mlx_config: Some(mlx_config),
        ..Default::default()
    },
    ..Default::default()
};
```

### CPU Optimizations

Woolly automatically detects and uses CPU features:

```rust
// Check available features
let features = CpuFeatures::detect();
println!("AVX2: {}", features.has_avx2);
println!("AVX-512: {}", features.has_avx512);
println!("ARM NEON: {}", features.has_neon);

// Configure for specific CPU
let config = EngineConfig {
    device: DeviceConfig {
        device_type: DeviceType::Cpu,
        cpu_features: features,
        prefer_simd: true,
        ..Default::default()
    },
    ..Default::default()
};
```

## Performance Tips

### 1. Memory Management

```rust
// Use memory mapping for large models
let loader = GGUFLoader::from_path_mmap("large-model.gguf")?;

// Pre-allocate memory pools
let config = EngineConfig {
    memory: MemoryConfig {
        use_memory_pool: true,
        pool_size_mb: 4096,
        ..Default::default()
    },
    ..Default::default()
};
```

### 2. Batch Processing

```rust
// Process multiple requests together
let config = SessionConfig {
    batch_size: 8, // Adjust based on available memory
    ..Default::default()
};
```

### 3. Quantization

Use quantized models for better performance:

```bash
# Q4_K_M offers good balance of quality and speed
model-q4_k_m.gguf  # ~4.1 GB for 7B model

# Q8_0 for higher quality
model-q8_0.gguf    # ~7.2 GB for 7B model
```

### 4. Thread Tuning

```rust
// Leave some cores for system tasks
let config = EngineConfig {
    num_threads: num_cpus::get() - 2,
    thread_affinity: true, // Pin threads to cores
    ..Default::default()
};
```

### 5. Monitoring Performance

```rust
#[cfg(feature = "metrics")]
{
    let metrics = engine.metrics();
    println!("Tokens/sec: {:.1}", metrics.tokens_per_second);
    println!("First token latency: {:?}", metrics.first_token_latency);
    println!("Memory usage: {:.1} MB", metrics.memory_usage_mb);
}
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Fails

```rust
// Check file exists and is readable
if !std::path::Path::new("model.gguf").exists() {
    eprintln!("Model file not found!");
}

// Verify GGUF format
match GGUFLoader::from_path("model.gguf") {
    Ok(loader) => println!("Valid GGUF file"),
    Err(e) => eprintln!("Invalid GGUF: {}", e),
}
```

#### 2. Out of Memory

```rust
// Reduce memory usage
let config = EngineConfig {
    memory: MemoryConfig {
        max_memory_mb: 4096, // Limit to 4GB
        use_mmap: true,      // Don't load entire file
        ..Default::default()
    },
    ..Default::default()
};

// Use smaller batch size
let session_config = SessionConfig {
    batch_size: 1,
    max_seq_length: 512, // Reduce context length
    ..Default::default()
};
```

#### 3. Slow Performance

```rust
// Enable all optimizations
let config = EngineConfig {
    optimizations: OptimizationConfig {
        operator_fusion: true,
        graph_optimization: true,
        use_fast_math: true,
        parallel_attention: true,
        ..Default::default()
    },
    ..Default::default()
};

// Check CPU features are being used
let features = CpuFeatures::detect();
if !features.has_avx2 && cfg!(target_arch = "x86_64") {
    eprintln!("Warning: AVX2 not available, performance will be limited");
}
```

### Debug Logging

Enable detailed logging for troubleshooting:

```bash
RUST_LOG=debug cargo run
RUST_LOG=woolly_core=trace cargo run  # Very detailed
```

```rust
// In code
use tracing_subscriber;

tracing_subscriber::fmt()
    .with_env_filter("debug,woolly_core=trace")
    .init();
```

### Getting Help

- **Documentation**: [https://docs.rs/woolly](https://docs.rs/woolly)
- **Examples**: Check the `examples/` directory
- **Issues**: [GitHub Issues](https://github.com/woolly/woolly/issues)
- **Discord**: Join our community for real-time help

## Next Steps

Now that you're up and running:

1. Check out the [examples](../examples/) for more complex use cases
2. Read the [performance guide](performance.md) for optimization tips
3. Explore [MCP integration](mcp-integration.md) for building AI services
4. Learn about [advanced features](advanced-features.md)

Happy inferencing with Woolly! 🦙