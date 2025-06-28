# Woolly Inference Implementation Guide

## Quick Implementation for Real Inference

To replace the mock responses in Woolly with actual inference, here's what needs to be done:

### 1. Update inference.rs Handler

```rust
// In crates/woolly-server/src/handlers/inference.rs

async fn generate_completion(
    state: ServerState,
    request: CompletionRequest,
    _user_id: String,
) -> ServerResult<CompletionResponse> {
    let engine = state.inference_engine.read().await;
    
    // Create generation config
    let generation_config = GenerationConfig {
        max_tokens: request.max_tokens.unwrap_or(100),
        temperature: request.temperature.unwrap_or(0.7),
        top_p: request.top_p.unwrap_or(0.9),
        top_k: request.top_k.unwrap_or(50),
        repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
    };

    // Get tokenizer and tokenize input
    let tokenizer = engine.tokenizer();
    let input_tokens = tokenizer.encode(&request.prompt).await?;
    
    // Create or get session
    let session_config = SessionConfig {
        max_seq_length: 2048,
        generation_config,
    };
    let session = engine.create_session(session_config).await?;
    
    // Run inference
    let output = session.infer(&input_tokens).await?;
    
    // Decode response
    let response_text = tokenizer.decode(&output.tokens).await?;
    
    Ok(CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model: engine.model_name().unwrap_or("woolly-model"),
        choices: vec![Choice {
            index: 0,
            text: Some(response_text),
            message: None,
            finish_reason: "stop".to_string(),
            logprobs: None,
        }],
        usage: Usage {
            prompt_tokens: input_tokens.len(),
            completion_tokens: output.tokens.len(),
            total_tokens: input_tokens.len() + output.tokens.len(),
        },
    })
}
```

### 2. Update Models Handler

```rust
// In crates/woolly-server/src/handlers/models.rs

pub async fn list_models(
    State(state): State<ServerState>,
    request: Request,
) -> ServerResult<Json<Vec<ModelInfo>>> {
    let _auth_context = extract_auth(&request)?;

    // Return proper OpenAI-compatible format
    let models = vec![
        ModelInfo {
            id: "llama-3.1-8b".to_string(),
            name: "Meta Llama 3.1 8B".to_string(),
            description: Some("Fast, efficient model for general tasks".to_string()),
            capabilities: ModelCapabilities {
                chat: true,
                mcp: true,
                vision: false,
                function_calling: true,
            },
            context_length: 8192,
            provider: "woolly".to_string(),
        },
        // Add more models as supported
    ];

    // Also return in OpenAI format
    let openai_format = OpenAIModelsResponse {
        data: models,
        object: "list".to_string(),
    };

    Ok(Json(openai_format))
}
```

### 3. Fix Response Format for Ole Compatibility

```rust
// Ensure the response matches what WoollyProvider expects
#[derive(Serialize)]
pub struct OpenAIModelsResponse {
    pub data: Vec<ModelInfo>,
    pub object: String,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub capabilities: ModelCapabilities,
    pub context_length: usize,
    pub provider: String,
}

#[derive(Serialize)]
pub struct ModelCapabilities {
    pub chat: bool,
    pub mcp: bool,
    pub vision: bool,
    pub function_calling: bool,
}
```

### 4. Implement Model Loading on Startup

```rust
// In server initialization
impl WoollyServer {
    pub fn new(config: ServerConfig) -> ServerResult<Self> {
        // ... existing code ...
        
        // Load default model if specified
        if let Some(default_model) = &config.models.default_model {
            let model_path = config.models.models_dir.join(default_model);
            if model_path.exists() {
                // Load model using woolly-gguf
                let loader = GGUFLoader::from_path(&model_path)?;
                let model = create_model_from_gguf(&loader)?;
                
                // Load into engine
                let mut engine = inference_engine.write().await;
                engine.load_model(Arc::new(model)).await?;
                
                info!("Loaded default model: {}", default_model);
            }
        }
        
        // ... rest of initialization
    }
}
```

### 5. Quick Test Commands

```bash
# Build Woolly with all optimizations
cd woolly
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run with a model
./target/release/woolly-server --model path/to/model.gguf

# Test the endpoints
curl http://localhost:8080/api/v1/health
curl http://localhost:8080/api/v1/models
curl -X POST http://localhost:8080/api/v1/inference/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

### 6. Performance Optimizations to Enable

In `Cargo.toml`:
```toml
[profile.release]
lto = true
opt-level = 3
codegen-units = 1
target-cpu = "native"  # Enable CPU-specific optimizations

[features]
default = ["simd", "parallel"]
simd = ["woolly-tensor/simd"]
parallel = ["rayon"]
mlx = ["woolly-mlx"]  # For Apple Silicon
```

### 7. Environment Variables for Testing

```bash
# Enable debug logging
export RUST_LOG=woolly_server=debug,woolly_core=debug

# Set model directory
export WOOLLY_MODELS_DIR=/path/to/models

# Enable performance tracking
export WOOLLY_PERF_TRACKING=true

# For Apple Silicon with MLX
export WOOLLY_USE_MLX=true
```

This should get basic inference working. The key is connecting the HTTP handlers to the actual inference engine and ensuring the response format matches what Ole expects.