# Getting Started with Woolly

Welcome to Woolly! This guide will help you run your first LLM inference in under 5 minutes.

## What You'll Learn

By the end of this guide, you'll be able to:
- ✓ Install Woolly and verify your system
- ✓ Download and load a model
- ✓ Generate your first text
- ✓ Understand basic configuration options

**Time required**: 15-30 minutes  
**Difficulty**: Beginner-friendly

## Prerequisites

Before we begin, let's ensure your system is ready.

### Required Software
- [ ] Rust 1.75 or later
- [ ] Git
- [ ] 8GB+ RAM (16GB recommended for larger models)
- [ ] 10GB+ free disk space

### Quick System Check

Run this command to verify your setup:

```bash
# Check Rust version
rustc --version  # Should show 1.75 or higher

# Check available memory
# macOS
sysctl hw.memsize | awk '{print $2/1024/1024/1024 " GB"}'

# Linux
free -h | grep Mem | awk '{print $2}'

# Windows (PowerShell)
Get-CimInstance Win32_ComputerSystem | Select-Object @{Name="TotalMemoryGB";Expression={[Math]::Round($_.TotalPhysicalMemory/1GB,2)}}
```

### Platform-Specific Requirements

<details>
<summary><b>macOS</b></summary>

```bash
# Install Xcode Command Line Tools (if needed)
xcode-select --install
```
</summary>
</details>

<details>
<summary><b>Linux (Ubuntu/Debian)</b></summary>

```bash
# Install build essentials
sudo apt update
sudo apt install build-essential pkg-config
```
</summary>
</details>

<details>
<summary><b>Windows</b></summary>

1. Install [Visual Studio 2019+](https://visualstudio.microsoft.com/downloads/) with C++ workload
2. Or install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
</summary>
</details>

## Getting Your First Model

Woolly uses GGUF format models. Let's download a small, fast model perfect for testing.

### Option 1: Quick Download (Recommended)

```bash
# Download TinyLlama - a small, fast model (500MB)
curl -L -o tinyllama-1.1b-q4_k_m.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Verify download
ls -lh tinyllama-1.1b-q4_k_m.gguf
# Should show ~500MB file
```

### Option 2: Use Existing Models

If you have llama.cpp models, they work directly with Woolly!

```bash
# Copy your existing model
cp ~/llama.cpp/models/your-model.gguf ./
```

### Option 3: Browse More Models

Find more models at [Hugging Face](https://huggingface.co/models?search=gguf).  
Look for models with "GGUF" in the name and "Q4_K_M" for good balance of size/quality.

## Your First Woolly Program

Let's create the simplest possible Woolly program!

### Step 1: Create a New Project

```bash
cargo new hello-woolly
cd hello-woolly
```

### Step 2: Add Woolly Dependency

Edit `Cargo.toml`:

```toml
[dependencies]
woolly = "0.1"  # Single dependency for beginners
tokio = { version = "1", features = ["full"] }
```

> **Note**: Using just `woolly` gives you everything you need to start. Advanced users can use individual crates later.

### Step 3: Write Your First Program

Create `src/main.rs`:

```rust
use woolly::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading model...");
    
    // Load model (auto-detects format)
    let model = Model::load("tinyllama-1.1b-q4_k_m.gguf")?;
    println!("✓ Model loaded!");
    
    // Create engine with defaults
    let engine = Engine::new(model)?;
    println!("✓ Engine ready!");
    
    // Generate text
    println!("\nGenerating text...");
    let response = engine.complete("Hello, my name is").await?;
    println!("\nResponse: {}", response);
    
    Ok(())
}
```

### Step 4: Run It!

```bash
cargo run
```

Expected output:
```
Loading model...
✓ Model loaded!
✓ Engine ready!

Generating text...

Response: Hello, my name is Alice and I'm excited to help you today!
```

🎉 **Congratulations!** You've just run your first LLM inference with Woolly!

## Understanding What Just Happened

Let's break down what each part does:

```rust
// 1. Load the model file
let model = Model::load("tinyllama-1.1b-q4_k_m.gguf")?;
//   - Automatically detects model format
//   - Validates the file
//   - Prepares for inference

// 2. Create the inference engine
let engine = Engine::new(model)?;
//   - Sets up compute backend (CPU/GPU)
//   - Allocates memory
//   - Optimizes for your hardware

// 3. Generate text
let response = engine.complete("Hello, my name is").await?;
//   - Tokenizes your prompt
//   - Runs the model
//   - Decodes the output
```

## Common First-Time Issues

### "Model file not found"

**Solution**: Ensure the model file is in your project directory:
```bash
ls *.gguf  # Should show your model file
```

If not found, re-download:
```bash
curl -L -o tinyllama-1.1b-q4_k_m.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### "Out of memory"

**Solution**: Use a smaller model or quantization:
```rust
// Use memory-efficient loading
let model = Model::load_mmap("tinyllama-1.1b-q4_k_m.gguf")?;
```

Or try an even smaller model (Q4_0 instead of Q4_K_M).

### "Slow performance"

**Solution**: Enable optimizations in `Cargo.toml`:
```toml
[profile.release]
lto = true
codegen-units = 1
```

Then run with: `cargo run --release`

## Next Steps: Adding Features

Now that you have a working example, let's add some features!

### Adding Temperature Control

Make responses more creative or focused:

```rust
let response = engine.complete("Hello, my name is")
    .temperature(0.8)  // 0.0 = focused, 1.0 = creative
    .await?;
```

### Streaming Responses

See text as it's generated:

```rust
let mut stream = engine.stream("Tell me a story:").await?;
while let Some(token) = stream.next().await {
    print!("{}", token?);
    std::io::stdout().flush()?;
}
```

### Using Different Models

Woolly automatically detects model architecture:

```rust
// Works with any GGUF model!
let llama = Model::load("llama-2-7b.gguf")?;
let mistral = Model::load("mistral-7b.gguf")?;
let phi = Model::load("phi-2.gguf")?;
```

### Setting Token Limits

Control response length:

```rust
let response = engine.complete("Write a haiku:")
    .max_tokens(50)  // Limit response length
    .await?;
```

## Quick Reference Card

### Loading Models
```rust
Model::load("model.gguf")?              // Auto-detect format
Model::load_mmap("large-model.gguf")?   // Memory-efficient
```

### Basic Inference
```rust
engine.complete("prompt").await?         // Simple completion
engine.complete("prompt")                // With options
    .temperature(0.7)
    .max_tokens(100)
    .await?
```

### Streaming
```rust
let mut stream = engine.stream("prompt").await?;
while let Some(token) = stream.next().await {
    print!("{}", token?);
}
```

### Common Patterns
```rust
// Check model info
println!("Model: {}", model.name());
println!("Size: {:.1}GB", model.size_gb());
println!("Architecture: {}", model.architecture());

// Monitor performance  
let start = Instant::now();
let response = engine.complete("test").await?;
println!("Time: {:?}", start.elapsed());
println!("Tokens/sec: {:.1}", response.tokens_per_second());
```

## Where to Go From Here

Based on what you want to build:

### 🚀 "I want maximum performance"
→ Read the [Performance Optimization Guide](performance.md)  
→ Try the `performance_benchmark` example

### 🔧 "I want to build an API"
→ Check out the [MCP Integration Guide](mcp-integration.md)  
→ Run the `websocket_server` example

### 🍎 "I'm on Apple Silicon"
→ See [Apple MLX Acceleration](apple-silicon.md)  
→ Enable with `features = ["mlx"]`

### 📚 "I want to understand the internals"
→ Read the [Architecture Guide](architecture.md)  
→ Explore the [API Reference](api-reference.md)

### 🤝 "I want to contribute"
→ Check [CONTRIBUTING.md](../CONTRIBUTING.md)  
→ Join our [Discord](https://discord.gg/woolly)

## Getting Help

### Quick Resources
- **Examples**: Check `examples/` directory for working code
- **FAQ**: Common questions answered in [FAQ.md](faq.md)
- **API Docs**: Full reference at [docs.rs/woolly](https://docs.rs/woolly)

### Community Support
- **Discord**: [discord.gg/woolly](https://discord.gg/woolly) - Get help in #help channel
- **GitHub Discussions**: [Ask questions](https://github.com/woolly/woolly/discussions)
- **Issues**: [Report bugs](https://github.com/woolly/woolly/issues)

### Debugging Tips

Enable debug logging:
```bash
RUST_LOG=debug cargo run
```

Get system info for bug reports:
```bash
woolly --version --system-info
```

---

**Ready for more?** Check out our [examples](../examples/) to see Woolly in action!

Happy coding with Woolly! 🦙