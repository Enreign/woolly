# Woolly Examples

This directory contains comprehensive examples demonstrating Woolly's features, from basic inference to advanced optimizations.

## Quick Start

All examples can be run with:
```bash
cargo run --example <example_name> -- [arguments]
```

For help with any example:
```bash
cargo run --example <example_name> -- --help
```

## Featured Examples

### 🌟 New Examples

1. **streaming_generation.rs** - Real-time token streaming
2. **tensor_operations.rs** - Comprehensive tensor operations tutorial  
3. **error_handling.rs** - Production-ready error handling patterns
4. **websocket_server.rs** - Full WebSocket server implementation

## Available Examples

### 1. basic_inference.rs - Getting Started
**Purpose**: Learn the fundamentals of loading models and running inference.

**Features demonstrated**:
- GGUF model loading with error handling
- Inference engine configuration
- Session management
- Token generation
- Performance monitoring

```bash
# Basic usage
cargo run --example basic_inference -- --model llama-7b.gguf --prompt "Hello, world!"

# Advanced options
cargo run --example basic_inference -- \
    --model llama-7b.gguf \
    --prompt "Explain quantum computing" \
    --max-tokens 200 \
    --temperature 0.8 \
    --threads 8 \
    --verbose
```

### 2. mlx_backend.rs - Apple Silicon Acceleration
**Purpose**: Leverage Apple MLX for GPU-accelerated inference on Mac.

**Features demonstrated**:
- MLX backend initialization
- GPU memory management
- Mixed precision (FP16) inference
- Batch processing on GPU
- Performance profiling

```bash
# Requires macOS with Apple Silicon
cargo run --example mlx_backend --features mlx -- --model llama-7b.gguf

# Batch inference with monitoring
cargo run --example mlx_backend --features mlx -- \
    --model llama-7b.gguf \
    --batch-size 4 \
    --prompt "The future of AI"
```

### 3. performance_benchmark.rs - Comprehensive Benchmarking
**Purpose**: Measure and analyze inference performance in detail.

**Features demonstrated**:
- Model loading benchmarks
- Tokenization performance
- Inference latency and throughput
- Memory usage profiling
- CPU utilization analysis
- Comparison with other engines

```bash
# Basic benchmark
cargo run --example performance_benchmark -- --model llama-7b.gguf

# Compare with llama.cpp
cargo run --example performance_benchmark -- \
    --model llama-7b.gguf \
    --compare llama.cpp \
    --iterations 200

# Generate detailed JSON report
cargo run --example performance_benchmark -- \
    --model llama-7b.gguf \
    --format json > benchmark_results.json
```

### 4. custom_tokenizer.rs - Tokenization Strategies
**Purpose**: Implement and compare different tokenization approaches.

**Features demonstrated**:
- Custom tokenizer implementation
- BPE, WordPiece, and SentencePiece strategies
- Performance comparison
- Unicode handling

```bash
# Compare tokenization strategies
cargo run --example custom_tokenizer -- --text "Hello, 世界!" --type bpe
cargo run --example custom_tokenizer -- --text "Hello, 世界!" --type wordpiece
cargo run --example custom_tokenizer -- --text "Hello, 世界!" --type sentencepiece
```

### 5. mcp_integration.rs - Model Context Protocol Server
**Purpose**: Build AI services with MCP integration.

**Features demonstrated**:
- MCP server setup
- Tool registration and management
- WebSocket and HTTP transports
- Request handling
- Resource management

```bash
# Start MCP server
cargo run --example mcp_integration -- --model llama-7b.gguf --port 8080

# With custom configuration
cargo run --example mcp_integration -- \
    --model llama-7b.gguf \
    --port 8080 \
    --max-connections 100 \
    --enable-cors
```

### 6. benchmark_comparison.rs - Performance Comparison
**Purpose**: Compare Woolly with other inference engines.

**Features demonstrated**:
- Side-by-side performance testing
- Automated benchmark execution
- Statistical analysis
- Report generation

```bash
# Compare with llama.cpp
cargo run --example benchmark_comparison -- \
    --model llama-7b.gguf \
    --comparison-binary /path/to/llama-main \
    --prompts prompts.txt
```

### 7. streaming_generation.rs - Real-time Token Streaming
**Purpose**: Demonstrate real-time text generation with streaming capabilities.

**Features demonstrated**:
- Token-by-token streaming
- Word-level streaming (groups tokens into words)
- Cancellable generation with timeout
- Streaming with detailed metadata (tokens/sec, logprobs)
- Custom callback processing

```bash
# Basic streaming
cargo run --example streaming_generation -- models/llama-2-7b-q4_k_m.gguf

# Stream with custom model
cargo run --example streaming_generation -- your-model.gguf
```

### 8. tensor_operations.rs - Tensor Operations Tutorial
**Purpose**: Learn the fundamentals of tensor operations in Woolly.

**Features demonstrated**:
- Basic tensor creation and manipulation
- Element-wise and reduction operations
- SIMD-accelerated operations (AVX2/NEON)
- Memory pool management
- Backend selection (CPU/CUDA/Metal)
- Zero-copy views and slicing
- Advanced operations (broadcasting, activations)

```bash
# Run the tutorial
cargo run --example tensor_operations

# The example includes:
# - Vector/matrix creation
# - Mathematical operations
# - Performance benchmarks
# - Memory management patterns
```

### 9. error_handling.rs - Production Error Handling
**Purpose**: Implement robust error handling for production deployments.

**Features demonstrated**:
- Comprehensive error types and patterns
- Model loading error recovery
- Inference error handling
- Resource constraint management
- Custom error context
- Retry with exponential backoff
- Circuit breaker pattern
- Production logging and metrics

```bash
# Run error handling examples
cargo run --example error_handling

# Demonstrates handling of:
# - Missing models
# - Out of memory errors
# - Invalid inputs
# - Network failures
# - Resource exhaustion
```

### 10. websocket_server.rs - WebSocket Streaming Server
**Purpose**: Build a production-ready WebSocket server for real-time inference.

**Features demonstrated**:
- Real-time bidirectional streaming
- Connection management
- Authentication and sessions
- Graceful disconnection handling
- Broadcasting to multiple clients
- Structured message protocol
- Error recovery

```bash
# Start the WebSocket server
cargo run --example websocket_server -- models/llama-2-7b-q4_k_m.gguf

# Connect with a WebSocket client:
# wscat -c ws://localhost:8080/ws -H 'Authorization: Bearer test-token'

# Send generation request:
# {"type":"Generate","prompt":"Hello, world!","options":{"temperature":0.8,"max_tokens":100}}
```

## Example Categories

### Basic Usage
- `basic_inference.rs` - Start here for simple inference
- `custom_tokenizer.rs` - Tokenization basics
- `streaming_generation.rs` - Real-time text streaming
- `tensor_operations.rs` - Tensor operations fundamentals

### Performance & Optimization
- `performance_benchmark.rs` - Comprehensive benchmarking
- `benchmark_comparison.rs` - Compare with other engines
- `mlx_backend.rs` - Apple Silicon GPU acceleration

### Integration & Deployment
- `mcp_integration.rs` - Build MCP services
- `websocket_server.rs` - WebSocket server for real-time inference

### Production Patterns
- `error_handling.rs` - Robust error handling strategies

## Common Patterns

### Error Handling
All examples demonstrate proper error handling:
```rust
// Graceful error handling with helpful messages
match result {
    Ok(output) => process_output(output),
    Err(e) => {
        eprintln!("Error: {}", e);
        // Provide helpful suggestions
    }
}
```

### Performance Monitoring
Examples include performance tracking:
```rust
let start = Instant::now();
let result = session.infer(&tokens).await?;
let duration = start.elapsed();
println!("Inference time: {:?}", duration);
println!("Tokens/sec: {:.1}", tokens.len() as f64 / duration.as_secs_f64());
```

### Configuration
Flexible configuration patterns:
```rust
// Use builder pattern for complex configs
let config = EngineConfig::builder()
    .with_threads(8)
    .with_memory_limit_mb(4096)
    .with_mlx_backend()
    .build();
```

## Tips for Running Examples

1. **Model Files**: Download GGUF models from Hugging Face or convert using llama.cpp
2. **Performance**: For best performance, use quantized models (Q4_K_M recommended)
3. **Memory**: Monitor memory usage, especially with large models
4. **Platform**: Some examples (like MLX) require specific platforms

## Troubleshooting

### Model Loading Issues
```bash
# Verify model file
file model.gguf  # Should show "data" or "GGUF"

# Check permissions
ls -la model.gguf
```

### Performance Issues
```bash
# Enable debug logging
RUST_LOG=debug cargo run --example basic_inference -- --model model.gguf

# Check CPU features
cargo run --example performance_benchmark -- --model model.gguf --no-cpu
```

### Memory Issues
```bash
# Use memory mapping for large models
cargo run --example basic_inference -- --model large-model.gguf --use-mmap

# Limit thread count
cargo run --example basic_inference -- --model model.gguf --threads 4
```

## Contributing Examples

When adding new examples:

1. **Clear Purpose**: Each example should demonstrate specific features
2. **Documentation**: Include comprehensive comments explaining the code
3. **Error Handling**: Show proper error handling patterns
4. **Help Text**: Provide detailed --help output
5. **Real-World**: Make examples practical and useful

Example structure:
```rust
//! Example Name
//!
//! This example demonstrates...
//!
//! Usage:
//!   cargo run --example name -- [options]
```

## Learning Path

For newcomers to Woolly, we recommend this order:

1. Start with `basic_inference.rs` to understand the basics
2. Try `custom_tokenizer.rs` to learn about tokenization
3. Run `performance_benchmark.rs` to see performance capabilities
4. Explore `mlx_backend.rs` if on Apple Silicon
5. Build services with `mcp_integration.rs`

## Resources

- [Woolly Documentation](https://docs.rs/woolly)
- [Performance Guide](../docs/performance.md)
- [Getting Started](../docs/getting-started.md)
- [API Reference](https://docs.rs/woolly-core)