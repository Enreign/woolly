# 🦙 Woolly Core

[![Crates.io](https://img.shields.io/crates/v/woolly-core.svg)](https://crates.io/crates/woolly-core)
[![Documentation](https://docs.rs/woolly-core/badge.svg)](https://docs.rs/woolly-core)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../../LICENSE)

The core inference engine for Woolly, providing high-performance LLM inference capabilities with comprehensive session management, configuration, and extensible model support.

## Features

- **🚀 High-Performance Inference**: Optimized for both CPU and GPU inference
- **🔧 Flexible Configuration**: Comprehensive configuration system for all aspects of inference
- **📝 Session Management**: Stateful inference sessions with caching and context management
- **🎯 Multiple Model Architectures**: Support for LLaMA, Mistral, Qwen, and more
- **🔤 Advanced Tokenization**: BPE, SentencePiece, and custom tokenizer support
- **⚡ Async/Await**: First-class async support throughout the API
- **🛡️ Type Safety**: Compile-time guarantees for model operations

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
woolly-core = "0.1"
```

### Basic Usage

```rust
use woolly_core::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create engine configuration
    let config = EngineConfig {
        max_context_length: 2048,
        num_threads: 4,
        device: DeviceConfig {
            device_type: DeviceType::Cpu,
            ..Default::default()
        },
        ..Default::default()
    };
    
    // Initialize inference engine
    let mut engine = InferenceEngine::new(config);
    
    // Load a model (you need to implement or provide a model)
    // let model = YourModel::new();
    // engine.load_model(Arc::new(model)).await?;
    
    // Create an inference session
    let session_config = SessionConfig {
        max_seq_length: 512,
        temperature: 0.8,
        top_p: 0.9,
        ..Default::default()
    };
    
    let session = engine.create_session(session_config).await?;
    
    // Run inference
    let input_tokens = vec![1, 2, 3, 4, 5]; // Your tokenized input
    let result = session.infer(&input_tokens).await?;
    
    println!("Generated {} logits", result.logits.len());
    
    Ok(())
}
```

### With Custom Tokenizer

```rust
use woolly_core::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tokenizer
    let tokenizer_config = TokenizerConfig {
        vocab_path: Some("vocab.json".to_string()),
        ..Default::default()
    };
    
    let tokenizer = create_tokenizer(TokenizerType::BPE, tokenizer_config).await?;
    
    // Encode text
    let text = "Hello, world!";
    let tokens = tokenizer.encode(text).await?;
    println!("Tokens: {:?}", tokens);
    
    // Decode back to text
    let decoded = tokenizer.decode(&tokens).await?;
    println!("Decoded: {}", decoded);
    
    Ok(())
}
```

## Core Components

### Inference Engine

The `InferenceEngine` is the central component that manages model loading, session creation, and resource allocation:

```rust
use woolly_core::prelude::*;

// Create engine with custom configuration
let config = EngineConfig {
    max_context_length: 4096,
    max_batch_size: 16,
    num_threads: 8,
    device: DeviceConfig {
        device_type: DeviceType::Cpu,
        cpu_fallback: true,
        ..Default::default()
    },
    memory: MemoryConfig {
        use_mmap: true,
        max_memory_mb: 8192,
        ..Default::default()
    },
    optimizations: OptimizationConfig {
        use_flash_attention: false, // CPU doesn't support flash attention
        operator_fusion: true,
        ..Default::default()
    },
    ..Default::default()
};

let mut engine = InferenceEngine::new(config);
```

### Session Management

Sessions provide stateful inference with context management and caching:

```rust
use woolly_core::prelude::*;

let session_config = SessionConfig {
    max_seq_length: 1024,
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    use_cache: true,
    repetition_penalty: 1.1,
    ..Default::default()
};

let session = engine.create_session(session_config).await?;

// Run multiple inferences with persistent context
let tokens1 = vec![1, 2, 3];
let result1 = session.infer(&tokens1).await?;

let tokens2 = vec![4, 5, 6];
let result2 = session.infer(&tokens2).await?; // Continues from previous context
```

### Model Implementation

Implement the `Model` trait for custom model architectures:

```rust
use woolly_core::prelude::*;
use async_trait::async_trait;

struct MyCustomModel {
    // Model parameters and configuration
}

#[async_trait]
impl Model for MyCustomModel {
    fn name(&self) -> &str {
        "my-custom-model"
    }
    
    fn model_type(&self) -> &str {
        "custom"
    }
    
    fn vocab_size(&self) -> usize {
        32000
    }
    
    fn context_length(&self) -> usize {
        2048
    }
    
    fn hidden_size(&self) -> usize {
        4096
    }
    
    fn num_layers(&self) -> usize {
        32
    }
    
    fn num_heads(&self) -> usize {
        32
    }
    
    async fn forward(
        &self,
        input_ids: &[u32],
        past_kv_cache: Option<&(dyn std::any::Any + Send + Sync)>,
    ) -> Result<ModelOutput> {
        // Implement your forward pass here
        Ok(ModelOutput {
            logits: vec![0.0; self.vocab_size()],
            logits_shape: vec![1, input_ids.len(), self.vocab_size()],
            past_kv_cache: None,
            hidden_states: None,
            attentions: None,
        })
    }
    
    async fn load_weights(&mut self, path: &std::path::Path) -> Result<()> {
        // Load model weights from file
        Ok(())
    }
}
```

### Custom Tokenizers

Implement custom tokenization strategies:

```rust
use woolly_core::prelude::*;
use async_trait::async_trait;
use std::collections::HashMap;

struct MyTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_tokens: HashMap<String, u32>,
}

#[async_trait]
impl Tokenizer for MyTokenizer {
    async fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Implement encoding logic
        Ok(vec![])
    }
    
    async fn decode(&self, tokens: &[u32]) -> Result<String> {
        // Implement decoding logic
        Ok(String::new())
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    fn bos_token_id(&self) -> Option<u32> {
        self.special_tokens.get("<s>").copied()
    }
    
    fn eos_token_id(&self) -> Option<u32> {
        self.special_tokens.get("</s>").copied()
    }
    
    // ... implement other required methods
}
```

## Configuration

### Engine Configuration

```rust
use woolly_core::prelude::*;

let config = EngineConfig {
    max_context_length: 4096,
    max_batch_size: 16,
    num_threads: 8,
    
    device: DeviceConfig {
        device_type: DeviceType::Cpu,
        device_id: 0,
        cpu_fallback: true,
        cuda: Some(CudaConfig {
            compute_capability: (8, 0),
            use_tensorrt: false,
            memory_pool_size: 2048,
        }),
        ..Default::default()
    },
    
    memory: MemoryConfig {
        max_memory_mb: 8192,
        use_mmap: true,
        pin_memory: false,
        allocator: AllocatorType::System,
    },
    
    cache: CacheConfig {
        max_cache_size_mb: 2048,
        eviction_policy: EvictionPolicy::Lru,
        persistent: false,
        ..Default::default()
    },
    
    optimizations: OptimizationConfig {
        use_flash_attention: true,
        operator_fusion: true,
        quantization: QuantizationConfig {
            enabled: true,
            method: QuantizationMethod::Int8,
            weight_bits: 8,
            ..Default::default()
        },
        ..Default::default()
    },
    
    logging: LoggingConfig {
        level: LogLevel::Info,
        log_performance: true,
        performance_interval_secs: 60,
        ..Default::default()
    },
};

// Validate configuration
config.validate().expect("Invalid configuration");

// Save configuration to file
config.to_file(&std::path::PathBuf::from("config.toml")).unwrap();

// Load configuration from file
let loaded_config = EngineConfig::from_file(&std::path::PathBuf::from("config.toml")).unwrap();
```

### Session Configuration

```rust
use woolly_core::prelude::*;

let session_config = SessionConfig {
    max_seq_length: 2048,
    temperature: 0.8,
    top_p: 0.9,
    top_k: 40,
    repetition_penalty: 1.1,
    use_cache: true,
    cache_type: CacheType::KV,
    batch_size: 1,
    stream: false,
    stop_tokens: vec![],
    pad_token_id: Some(0),
    eos_token_id: Some(2),
};
```

## Device Support

### CPU Configuration

```rust
use woolly_core::prelude::*;

let cpu_config = DeviceConfig {
    device_type: DeviceType::Cpu,
    cpu_fallback: true,
    ..Default::default()
};

let engine_config = EngineConfig {
    device: cpu_config,
    num_threads: num_cpus::get(),
    optimizations: OptimizationConfig {
        use_flash_attention: false, // Not supported on CPU
        operator_fusion: true,
        ..Default::default()
    },
    ..Default::default()
};
```

### CUDA Configuration

```rust
use woolly_core::prelude::*;

#[cfg(feature = "cuda")]
{
    let cuda_config = DeviceConfig {
        device_type: DeviceType::Cuda,
        device_id: 0,
        cpu_fallback: true,
        cuda: Some(CudaConfig {
            compute_capability: (8, 0),
            use_tensorrt: true,
            memory_pool_size: 4096,
        }),
        ..Default::default()
    };
    
    let engine_config = EngineConfig {
        device: cuda_config,
        optimizations: OptimizationConfig {
            use_flash_attention: true,
            use_amp: true,
            ..Default::default()
        },
        ..Default::default()
    };
}
```

### Metal Configuration (Apple Silicon)

```rust
use woolly_core::prelude::*;

#[cfg(feature = "metal")]
{
    let metal_config = DeviceConfig {
        device_type: DeviceType::Metal,
        metal: Some(MetalConfig {
            use_mps: true,
            max_buffer_size: 2048,
        }),
        ..Default::default()
    };
}
```

## Error Handling

Woolly Core provides comprehensive error handling:

```rust
use woolly_core::prelude::*;

match session.infer(&tokens).await {
    Ok(result) => {
        println!("Inference successful: {} logits", result.logits.len());
    }
    Err(CoreError::Model(msg)) => {
        eprintln!("Model error: {}", msg);
    }
    Err(CoreError::Tokenizer(msg)) => {
        eprintln!("Tokenizer error: {}", msg);
    }
    Err(CoreError::Generation(msg)) => {
        eprintln!("Generation error: {}", msg);
    }
    Err(CoreError::InvalidInput(msg)) => {
        eprintln!("Invalid input: {}", msg);
    }
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

## Performance Optimization

### Memory Management

```rust
use woolly_core::prelude::*;

let memory_config = MemoryConfig {
    max_memory_mb: 8192,
    use_mmap: true,        // Enable memory mapping for large models
    pin_memory: true,      // Pin memory for faster GPU transfers
    allocator: AllocatorType::Jemalloc, // Use optimized allocator
};
```

### Batch Processing

```rust
use woolly_core::prelude::*;

let engine_config = EngineConfig {
    max_batch_size: 16,    // Process multiple sequences simultaneously
    ..Default::default()
};

// Process multiple inputs in a batch
let batch_inputs = vec![
    vec![1, 2, 3],
    vec![4, 5, 6],
    vec![7, 8, 9],
];

for input in batch_inputs {
    let result = session.infer(&input).await?;
    // Process result
}
```

### Quantization

```rust
use woolly_core::prelude::*;

let quantization_config = QuantizationConfig {
    enabled: true,
    method: QuantizationMethod::Int8,
    weight_bits: 8,
    activation_bits: Some(8),
};

let engine_config = EngineConfig {
    optimizations: OptimizationConfig {
        quantization: quantization_config,
        ..Default::default()
    },
    ..Default::default()
};
```

## Testing

```bash
# Run unit tests
cargo test

# Run integration tests
cargo test --test integration_test

# Run tests with all features
cargo test --all-features

# Run benchmarks
cargo bench
```

## Examples

See the [`examples/`](../../examples/) directory for complete working examples:

- **[Basic Inference](../../examples/basic_inference.rs)**: Simple model loading and text generation
- **[Custom Tokenizer](../../examples/custom_tokenizer.rs)**: Custom tokenizer implementations
- **[MCP Integration](../../examples/mcp_integration.rs)**: Model Context Protocol server setup

## Features

- `cuda`: Enable NVIDIA GPU support
- `metal`: Enable Apple Metal GPU support
- `benchmarks`: Include benchmarking utilities
- `serde`: Enable serialization support

## Contributing

We welcome contributions! Please see the [Contributing Guide](../../CONTRIBUTING.md) for details.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../../LICENSE-MIT))

at your option.