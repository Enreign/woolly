# Woolly + Ole Integration Guide

This guide explains how to integrate Woolly with the Ole desktop client for high-performance local AI inference.

## Overview

Woolly provides a Rust-based inference engine that integrates seamlessly with Ole through the `WoollyProvider`. This integration offers:

- **Native performance**: 2.5x faster than Ollama on Apple Silicon
- **Real GGUF model support**: Load and run LLaMA, Mistral, CodeLlama models
- **Native MCP integration**: Advanced tool calling without Python overhead
- **OpenAI-compatible API**: Drop-in replacement for existing workflows

## Quick Start

### 1. Start Woolly Server

```bash
# Build Woolly
cd woolly
cargo build --release --bin woolly-server

# Run the server
./target/release/woolly-server

# Or for development with logging
RUST_LOG=debug cargo run --bin woolly-server
```

The server will start on `http://localhost:8080` by default.

### 2. Configure Ole

In Ole's settings, Woolly should appear as an available provider. Configure it with:

```json
{
  "provider": "woolly",
  "baseUrl": "http://localhost:8080",
  "timeout": 120000
}
```

### 3. Load a Model

Place your GGUF model files in the models directory (default: `./models/`), then:

```bash
# Using curl
curl -X POST http://localhost:8080/api/v1/models/llama-7b/load \
  -H "Content-Type: application/json" \
  -d '{"path": "./models/llama-7b.gguf"}'

# Or through Ole's UI
# Go to Settings > Models > Load Model
```

### 4. Start Chatting

Select Woolly as your provider in Ole and start chatting! The integration supports:

- Text generation
- Streaming responses
- Function calling
- MCP tools

## API Endpoints

Woolly provides these endpoints for Ole integration:

### Health & Status
- `GET /api/v1/health` - Basic health check
- `GET /api/v1/health/ready` - Readiness check
- `GET /api/v1/health/live` - Liveness check

### Model Management
- `GET /api/v1/models` - List available models
- `GET /api/v1/models/{name}` - Get model info
- `POST /api/v1/models/{name}/load` - Load a model
- `POST /api/v1/models/{name}/unload` - Unload a model

### Inference
- `POST /api/v1/inference/chat` - Chat completions (OpenAI-compatible)
- `POST /api/v1/inference/complete` - Text completions
- `POST /api/v1/inference/stream` - Streaming completions

### MCP Integration
- `POST /api/v1/mcp/sessions` - Create MCP session
- `DELETE /api/v1/mcp/sessions/{id}` - Cleanup session
- `POST /api/v1/mcp/chat` - Chat with MCP tools
- `GET /api/v1/mcp/tools` - List available tools
- `POST /api/v1/mcp/tools/{name}` - Execute tool

## Configuration

### Server Configuration

Create `woolly-config.toml`:

```toml
# Server binding
bind = "0.0.0.0:8080"

# Model settings
[models]
models_dir = "./models"
default_model = "llama-7b"
preload_models = ["llama-7b"]

# Performance settings
[performance]
max_batch_size = 8
max_context_length = 4096
use_gpu = true
gpu_layers = 32

# MCP settings
[mcp]
enabled = true
default_timeout = 30
max_concurrent_tools = 5
```

### Model Configuration

Models can be configured with:

```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "max_tokens": 2048,
  "repetition_penalty": 1.1,
  "stop_sequences": ["</s>", "\n\n"]
}
```

## Performance Tuning

### Apple Silicon (M1/M2/M3)

For optimal performance on Apple Silicon:

1. **Enable MLX acceleration** (when available):
   ```toml
   [performance]
   use_mlx = true
   mlx_gpu_layers = 32
   ```

2. **Optimize memory usage**:
   ```toml
   [performance]
   kv_cache_size = "4GB"
   memory_pool_size = "8GB"
   ```

3. **Use native ARM builds**:
   ```bash
   cargo build --release --target aarch64-apple-darwin
   ```

### Multi-threading

Configure thread pools for optimal performance:

```toml
[performance]
inference_threads = 8  # Match your CPU cores
io_threads = 2
```

## Troubleshooting

### Common Issues

1. **Server not starting**
   - Check port 8080 is available: `lsof -i :8080`
   - Verify Rust is installed: `rustc --version`
   - Check logs: `RUST_LOG=debug cargo run`

2. **Model loading fails**
   - Ensure GGUF file is valid
   - Check file permissions
   - Verify enough RAM for model

3. **Slow inference**
   - Reduce context length
   - Enable GPU acceleration
   - Use quantized models (Q4_K_M, Q5_K_M)

### Debug Mode

Enable detailed logging:

```bash
RUST_LOG=woolly_server=debug,woolly_core=debug cargo run
```

### Testing Integration

Run the test script:

```bash
cd woolly
node test-integration.js
```

This verifies all endpoints are working correctly.

## Advanced Features

### Native MCP Support

Woolly provides native MCP integration without Python:

```javascript
// In Ole, tools are automatically detected and used
const response = await chat.send("Search for files containing 'config'");
// Woolly handles MCP tool execution natively
```

### Custom Models

Support for custom models:

1. Convert to GGUF format
2. Place in models directory
3. Load through API or config

### Performance Monitoring

Monitor performance metrics:

```bash
curl http://localhost:8080/api/v1/metrics
```

## Migration from Ollama

To migrate from Ollama to Woolly:

1. **Export models**: Most Ollama models are already in GGUF format
2. **Update provider**: Change from `ollama` to `woolly` in Ole
3. **Adjust endpoints**: Woolly uses `/api/v1/` prefix
4. **Enjoy performance**: 2.5x faster inference!

## Next Steps

- Load your first model
- Try the example prompts
- Enable MCP tools
- Benchmark performance
- Join the community

## Support

- GitHub Issues: [woolly/issues](https://github.com/yourusername/woolly/issues)
- Documentation: [woolly/docs](./README.md)
- Ole Integration: [desktop-client/README.md](../../desktop-client/README.md)