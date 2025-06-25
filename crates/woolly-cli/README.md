# 🦙 Woolly CLI

[![Crates.io](https://img.shields.io/crates/v/woolly-cli.svg)](https://crates.io/crates/woolly-cli)
[![Documentation](https://docs.rs/woolly-cli/badge.svg)](https://docs.rs/woolly-cli)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../../LICENSE)

Command-line interface for Woolly, providing easy access to LLM inference, model management, benchmarking, and MCP server functionality through a comprehensive CLI tool.

## Features

- **🚀 Text Generation**: Interactive and batch text generation with customizable parameters
- **📊 Model Information**: Detailed model inspection and metadata extraction
- **⚡ Benchmarking**: Performance testing and comparison with other inference engines
- **🌐 MCP Server**: Start and manage Model Context Protocol servers
- **🔧 Configuration**: Flexible configuration management with profiles
- **📝 Interactive Mode**: REPL-style interface for exploratory inference
- **📄 Batch Processing**: Process multiple inputs from files or stdin
- **🎯 Output Formats**: JSON, text, and custom formatting options

## Installation

### From Crates.io

```bash
cargo install woolly-cli
```

### From Source

```bash
git clone https://github.com/woolly/woolly.git
cd woolly
cargo install --path crates/woolly-cli
```

### Using Cargo

```bash
# Install latest version
cargo install --git https://github.com/woolly/woolly.git woolly-cli

# Install specific version
cargo install woolly-cli --version 0.1.0
```

## Quick Start

```bash
# Generate text with a model
woolly run --model model.gguf --prompt "Explain quantum computing"

# Get model information
woolly info --model model.gguf

# Start an MCP server
woolly mcp --model model.gguf --port 8080

# Run benchmarks
woolly benchmark --model model.gguf --compare-with llama.cpp
```

## Commands

### Text Generation (`run`)

Generate text using a loaded model:

```bash
# Basic text generation
woolly run --model model.gguf --prompt "Hello, world!"

# With custom parameters
woolly run \
  --model model.gguf \
  --prompt "Write a story about a robot" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-p 0.9

# Interactive mode
woolly run --model model.gguf --interactive

# Batch processing from file
woolly run --model model.gguf --input prompts.txt --output results.json

# Streaming output
woolly run --model model.gguf --prompt "Long story..." --stream
```

#### Options

- `--model, -m <PATH>`: Path to the GGUF model file
- `--prompt, -p <TEXT>`: Input prompt for generation
- `--max-tokens <NUM>`: Maximum tokens to generate (default: 100)
- `--temperature <FLOAT>`: Sampling temperature (default: 0.8)
- `--top-p <FLOAT>`: Top-p nucleus sampling (default: 0.9)
- `--top-k <NUM>`: Top-k sampling (default: 40)
- `--repetition-penalty <FLOAT>`: Repetition penalty (default: 1.1)
- `--stop <TOKENS>`: Stop tokens (can be used multiple times)
- `--seed <NUM>`: Random seed for reproducible generation
- `--threads <NUM>`: Number of threads to use
- `--batch-size <NUM>`: Batch size for processing
- `--context-length <NUM>`: Maximum context length
- `--interactive, -i`: Interactive mode
- `--input <FILE>`: Input file (prompts, one per line)
- `--output <FILE>`: Output file (JSON format)
- `--format <FORMAT>`: Output format (text, json, csv)
- `--stream`: Stream output as it's generated
- `--verbose, -v`: Verbose output

#### Examples

```bash
# Creative writing with high temperature
woolly run -m model.gguf -p "Once upon a time" --temperature 1.2 --max-tokens 500

# Factual Q&A with low temperature
woolly run -m model.gguf -p "What is the capital of France?" --temperature 0.1

# Batch processing
echo -e "Hello\nWorld\nTest" | woolly run -m model.gguf --input - --format json

# With custom stop tokens
woolly run -m model.gguf -p "List items:" --stop "." --stop "!" --max-tokens 50
```

### Model Information (`info`)

Get detailed information about GGUF models:

```bash
# Basic model information
woolly info --model model.gguf

# Detailed metadata
woolly info --model model.gguf --detailed

# Export model info to JSON
woolly info --model model.gguf --output model_info.json

# Compare multiple models
woolly info --model model1.gguf --model model2.gguf --compare
```

#### Options

- `--model, -m <PATH>`: Path to the GGUF model file (can be used multiple times)
- `--detailed, -d`: Show detailed model information
- `--output <FILE>`: Output file for model information
- `--format <FORMAT>`: Output format (text, json, yaml)
- `--compare`: Compare multiple models
- `--tensors`: List all tensors in the model
- `--verify`: Verify model integrity

#### Examples

```bash
# Quick model overview
woolly info -m model.gguf

# Detailed analysis with tensor information
woolly info -m model.gguf --detailed --tensors --format json

# Compare two models
woolly info -m model1.gguf -m model2.gguf --compare --format yaml
```

### MCP Server (`mcp`)

Start a Model Context Protocol server:

```bash
# Basic MCP server
woolly mcp --model model.gguf --port 8080

# With custom configuration
woolly mcp \
  --model model.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  --max-connections 100 \
  --enable-cors

# With authentication
woolly mcp --model model.gguf --port 8080 --api-key secret-key

# Load custom tools
woolly mcp --model model.gguf --port 8080 --tools-dir ./custom-tools
```

#### Options

- `--model, -m <PATH>`: Path to the GGUF model file
- `--port <NUM>`: Server port (default: 8080)
- `--host <HOST>`: Server host (default: localhost)
- `--max-connections <NUM>`: Maximum concurrent connections (default: 100)
- `--timeout <SECONDS>`: Request timeout in seconds (default: 30)
- `--enable-cors`: Enable CORS headers
- `--cors-origins <ORIGINS>`: Allowed CORS origins (comma-separated)
- `--api-key <KEY>`: Require API key authentication
- `--rate-limit <REQUESTS_PER_MINUTE>`: Rate limiting
- `--tools-dir <DIR>`: Directory containing custom tools
- `--config <FILE>`: Configuration file
- `--log-level <LEVEL>`: Logging level (error, warn, info, debug, trace)
- `--metrics`: Enable metrics endpoint
- `--websocket`: Enable WebSocket support
- `--stdio`: Use stdio transport instead of HTTP

#### Examples

```bash
# Production server with authentication
woolly mcp -m model.gguf --port 443 --host 0.0.0.0 --api-key $(cat api-key.txt) --rate-limit 60

# Development server with detailed logging
woolly mcp -m model.gguf --port 8080 --log-level debug --metrics --websocket

# Stdio mode for process integration
woolly mcp -m model.gguf --stdio
```

### Benchmarking (`benchmark`)

Performance testing and comparison:

```bash
# Basic benchmark
woolly benchmark --model model.gguf

# Compare with llama.cpp
woolly benchmark --model model.gguf --compare-with llama.cpp --binary ./llama-main

# Custom benchmark configuration
woolly benchmark \
  --model model.gguf \
  --iterations 10 \
  --prompt-file test-prompts.txt \
  --max-tokens 200 \
  --output benchmark-results.json

# Memory profiling
woolly benchmark --model model.gguf --profile-memory --profile-cpu
```

#### Options

- `--model, -m <PATH>`: Path to the GGUF model file
- `--compare-with <ENGINE>`: Compare with other engines (llama.cpp, candle)
- `--binary <PATH>`: Path to comparison binary
- `--iterations <NUM>`: Number of benchmark iterations (default: 10)
- `--prompt-file <FILE>`: File containing test prompts
- `--max-tokens <NUM>`: Maximum tokens for benchmark (default: 100)
- `--temperature <FLOAT>`: Temperature for benchmark (default: 0.8)
- `--batch-sizes <SIZES>`: Comma-separated batch sizes to test
- `--profile-memory`: Enable memory profiling
- `--profile-cpu`: Enable CPU profiling
- `--output <FILE>`: Output file for results
- `--format <FORMAT>`: Output format (json, csv, markdown)
- `--warmup <NUM>`: Number of warmup iterations
- `--threads <NUM>`: Number of threads for testing

#### Examples

```bash
# Comprehensive benchmark with profiling
woolly benchmark -m model.gguf --iterations 20 --profile-memory --profile-cpu --format json

# Compare performance across batch sizes
woolly benchmark -m model.gguf --batch-sizes 1,4,8,16 --iterations 5

# Generate markdown report
woolly benchmark -m model.gguf --compare-with llama.cpp --format markdown --output report.md
```

## Configuration

### Configuration Files

Woolly CLI supports configuration files for persistent settings:

```toml
# ~/.config/woolly/config.toml

[default]
model = "/path/to/default/model.gguf"
temperature = 0.8
max_tokens = 100
threads = 8

[profiles.creative]
temperature = 1.2
top_p = 0.95
repetition_penalty = 1.0

[profiles.factual]
temperature = 0.1
top_p = 0.9
repetition_penalty = 1.1

[mcp]
port = 8080
host = "localhost"
enable_cors = true
max_connections = 100

[benchmark]
iterations = 10
warmup = 2
output_format = "json"
```

### Using Profiles

```bash
# Use a specific profile
woolly run --profile creative --prompt "Write a creative story"

# Override profile settings
woolly run --profile factual --temperature 0.2 --prompt "What is quantum computing?"

# List available profiles
woolly config list-profiles

# Create a new profile
woolly config create-profile --name technical --temperature 0.3 --max-tokens 200
```

### Environment Variables

```bash
# Set default model
export WOOLLY_MODEL="/path/to/model.gguf"

# Set default configuration directory
export WOOLLY_CONFIG_DIR="/custom/config/path"

# Set logging level
export WOOLLY_LOG_LEVEL="debug"

# Use environment variables
woolly run --prompt "Hello, world!"  # Uses WOOLLY_MODEL
```

## Interactive Mode

Start an interactive REPL session:

```bash
# Start interactive mode
woolly run --model model.gguf --interactive

# With custom settings
woolly run -m model.gguf -i --temperature 0.9 --max-tokens 150
```

### Interactive Commands

```
woolly> Hello, how are you?
AI: I'm doing well, thank you! How can I help you today?

woolly> /help
Available commands:
  /help              Show this help message
  /exit, /quit       Exit interactive mode
  /clear             Clear conversation history
  /save <file>       Save conversation to file
  /load <file>       Load conversation from file
  /config            Show current configuration
  /set <key> <value> Set configuration parameter
  /reset             Reset to default configuration
  /stats             Show session statistics

woolly> /set temperature 1.2
Temperature set to 1.2

woolly> /config
Model: model.gguf
Temperature: 1.2
Max tokens: 100
Top-p: 0.9
Top-k: 40

woolly> Write a haiku about programming
Code flows like water,
Variables dance in the light,
Logic finds its way.

woolly> /save conversation.txt
Conversation saved to conversation.txt

woolly> /exit
Goodbye!
```

## Input/Output Formats

### Input Formats

#### Single Prompt
```bash
woolly run -m model.gguf -p "Your prompt here"
```

#### From File
```bash
# One prompt per line
echo -e "Prompt 1\nPrompt 2\nPrompt 3" > prompts.txt
woolly run -m model.gguf --input prompts.txt
```

#### From stdin
```bash
echo "Hello, world!" | woolly run -m model.gguf --input -
```

#### JSON Input
```json
{
  "prompts": [
    {
      "text": "Explain AI",
      "max_tokens": 150,
      "temperature": 0.7
    },
    {
      "text": "Write a poem",
      "max_tokens": 200,
      "temperature": 1.0
    }
  ]
}
```

```bash
woolly run -m model.gguf --input prompts.json --format json
```

### Output Formats

#### Text Format (Default)
```
Prompt: Hello, world!
Response: Hello! How can I help you today?

Prompt: Explain quantum computing
Response: Quantum computing is a type of computation that...
```

#### JSON Format
```json
{
  "results": [
    {
      "prompt": "Hello, world!",
      "response": "Hello! How can I help you today?",
      "tokens_generated": 8,
      "time_ms": 1250,
      "tokens_per_second": 6.4
    }
  ],
  "summary": {
    "total_prompts": 1,
    "total_tokens": 8,
    "total_time_ms": 1250,
    "average_tokens_per_second": 6.4
  }
}
```

#### CSV Format
```csv
prompt,response,tokens_generated,time_ms,tokens_per_second
"Hello, world!","Hello! How can I help you today?",8,1250,6.4
```

## Error Handling

Woolly CLI provides comprehensive error handling with helpful messages:

```bash
# Model not found
$ woolly run -m nonexistent.gguf -p "test"
Error: Model file not found: nonexistent.gguf
Suggestion: Check the file path and ensure the model exists

# Invalid GGUF format
$ woolly run -m invalid_file.txt -p "test"
Error: Invalid GGUF format: invalid_file.txt
Suggestion: Ensure the file is a valid GGUF model file

# Out of memory
$ woolly run -m large_model.gguf -p "test"
Error: Insufficient memory to load model
Suggestion: Try using --threads 1 or a smaller model

# Configuration error
$ woolly run -m model.gguf -p "test" --temperature -1
Error: Invalid temperature value: -1
Suggestion: Temperature must be non-negative (typically 0.0-2.0)
```

## Advanced Usage

### Custom Tokenizers

```bash
# Use specific tokenizer
woolly run -m model.gguf -p "test" --tokenizer sentencepiece --vocab vocab.model

# Show tokenization
woolly tokenize -m model.gguf -t "Hello, world!" --show-tokens
```

### Model Conversion

```bash
# Convert model formats (if supported)
woolly convert --input model.safetensors --output model.gguf --format gguf

# Quantize model
woolly quantize --input model.gguf --output model_q4_0.gguf --format q4_0
```

### Plugin System

```bash
# List available plugins
woolly plugins list

# Install plugin
woolly plugins install database-tools

# Load plugin for MCP server
woolly mcp -m model.gguf --plugins database-tools,web-scraper
```

## Performance Tips

### Memory Optimization

```bash
# Use memory mapping (default)
woolly run -m model.gguf --use-mmap

# Reduce context length for large models
woolly run -m large_model.gguf --context-length 1024

# Single thread for memory-constrained systems
woolly run -m model.gguf --threads 1
```

### Speed Optimization

```bash
# Use all CPU cores
woolly run -m model.gguf --threads $(nproc)

# Increase batch size for multiple prompts
woolly run -m model.gguf --input prompts.txt --batch-size 16

# Disable memory mapping for small models on fast storage
woolly run -m small_model.gguf --no-mmap
```

### GPU Acceleration

```bash
# Use CUDA (if available)
woolly run -m model.gguf --device cuda --gpu-id 0

# Use Metal (Apple Silicon)
woolly run -m model.gguf --device metal

# Offload specific layers to GPU
woolly run -m model.gguf --device cuda --gpu-layers 32
```

## Debugging

### Verbose Output

```bash
# Enable verbose logging
woolly run -m model.gguf -p "test" --verbose

# Enable debug logging
WOOLLY_LOG_LEVEL=debug woolly run -m model.gguf -p "test"

# Trace-level logging
RUST_LOG=trace woolly run -m model.gguf -p "test"
```

### Profiling

```bash
# Profile memory usage
woolly run -m model.gguf -p "test" --profile-memory

# Profile CPU usage
woolly run -m model.gguf -p "test" --profile-cpu

# Save profiling data
woolly run -m model.gguf -p "test" --profile-memory --profile-output profile.json
```

## Integration Examples

### Shell Scripts

```bash
#!/bin/bash
# Generate multiple variations
prompts=("Explain AI" "Describe quantum computing" "What is rust?")
for prompt in "${prompts[@]}"; do
    echo "Generating for: $prompt"
    woolly run -m model.gguf -p "$prompt" --max-tokens 100 --format json > "output_$(echo $prompt | tr ' ' '_').json"
done
```

### Python Integration

```python
import subprocess
import json

def generate_text(prompt, max_tokens=100):
    result = subprocess.run([
        'woolly', 'run',
        '--model', 'model.gguf',
        '--prompt', prompt,
        '--max-tokens', str(max_tokens),
        '--format', 'json'
    ], capture_output=True, text=True)
    
    return json.loads(result.stdout)

response = generate_text("Explain machine learning")
print(response['results'][0]['response'])
```

### Docker Integration

```dockerfile
FROM rust:latest
RUN cargo install woolly-cli
COPY model.gguf /app/model.gguf
WORKDIR /app
ENTRYPOINT ["woolly", "mcp", "--model", "model.gguf", "--host", "0.0.0.0", "--port", "8080"]
```

## Troubleshooting

### Common Issues

1. **Model Loading Failed**
   ```bash
   # Verify model file
   woolly info -m model.gguf --verify
   
   # Check file permissions
   ls -la model.gguf
   ```

2. **Out of Memory**
   ```bash
   # Reduce threads
   woolly run -m model.gguf --threads 1
   
   # Use smaller context
   woolly run -m model.gguf --context-length 512
   ```

3. **Slow Performance**
   ```bash
   # Check system resources
   woolly benchmark -m model.gguf --profile-cpu --profile-memory
   
   # Use appropriate thread count
   woolly run -m model.gguf --threads $(nproc)
   ```

4. **Connection Issues (MCP)**
   ```bash
   # Check port availability
   netstat -ln | grep 8080
   
   # Test with curl
   curl http://localhost:8080/health
   ```

## Examples

For more examples, see the [`examples/`](../../examples/) directory:

- **[CLI Automation](examples/cli_automation.sh)**: Shell scripting with Woolly CLI
- **[Batch Processing](examples/batch_processing.py)**: Python integration
- **[MCP Client](examples/mcp_client.js)**: JavaScript MCP client

## Contributing

We welcome contributions! Please see the [Contributing Guide](../../CONTRIBUTING.md) for details.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../../LICENSE-MIT))

at your option.