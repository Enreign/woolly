# API Reference

Complete API documentation for Woolly's Rust library, HTTP endpoints, and WebSocket protocol.

## Table of Contents

1. [Rust API](#rust-api)
2. [HTTP API](#http-api)
3. [WebSocket API](#websocket-api)
4. [Configuration](#configuration)
5. [Error Handling](#error-handling)
6. [Examples](#examples)

## Rust API

### Core Types

#### Engine

The main inference engine that manages models and sessions.

```rust
pub struct Engine {
    // Private fields
}

impl Engine {
    /// Create a new engine with the given configuration
    pub fn new(config: EngineConfig) -> Result<Self, Error>;
    
    /// Load a model from the specified path
    pub fn load_model(&mut self, path: &Path) -> Result<(), Error>;
    
    /// Unload the current model
    pub fn unload_model(&mut self) -> Result<(), Error>;
    
    /// Get the name of the currently loaded model
    pub fn current_model_name(&self) -> Option<&str>;
    
    /// Create a new inference session
    pub fn create_session(&self) -> Result<Session, Error>;
    
    /// Get engine statistics
    pub fn stats(&self) -> EngineStats;
}
```

#### EngineConfig

Configuration for the inference engine.

```rust
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Maximum batch size for parallel inference
    pub max_batch_size: usize,
    
    /// Maximum sequence length
    pub max_sequence_length: usize,
    
    /// Enable GPU acceleration if available
    pub enable_gpu: bool,
    
    /// Number of threads for CPU inference
    pub num_threads: Option<usize>,
    
    /// Memory mapping strategy
    pub mmap_enabled: bool,
    
    /// Cache configuration
    pub cache_config: CacheConfig,
}

impl EngineConfig {
    /// Create a new configuration with default values
    pub fn default() -> Self;
    
    /// Builder pattern for configuration
    pub fn builder() -> EngineConfigBuilder;
}

pub struct EngineConfigBuilder {
    // Private fields
}

impl EngineConfigBuilder {
    pub fn max_batch_size(mut self, size: usize) -> Self;
    pub fn max_sequence_length(mut self, length: usize) -> Self;
    pub fn enable_gpu(mut self, enable: bool) -> Self;
    pub fn num_threads(mut self, threads: usize) -> Self;
    pub fn enable_mmap(mut self, enable: bool) -> Self;
    pub fn cache_size(mut self, size: usize) -> Self;
    pub fn build(self) -> EngineConfig;
}
```

#### Session

An inference session for generating text.

```rust
pub struct Session {
    // Private fields
}

impl Session {
    /// Generate text with the given prompt
    pub async fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<String, Error>;
    
    /// Stream tokens as they are generated
    pub async fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        tx: mpsc::Sender<String>,
    ) -> Result<(), Error>;
    
    /// Cancel ongoing generation
    pub fn cancel_generation(&mut self);
    
    /// Get session statistics
    pub fn stats(&self) -> SessionStats;
    
    /// Clear session context
    pub fn clear_context(&mut self);
}
```

#### Model

Represents a loaded model.

```rust
pub struct Model {
    // Private fields
}

impl Model {
    /// Load a model from file
    pub fn from_file(path: &Path) -> Result<Self, Error>;
    
    /// Get model metadata
    pub fn metadata(&self) -> &ModelMetadata;
    
    /// Get model architecture
    pub fn architecture(&self) -> Architecture;
    
    /// Get vocabulary
    pub fn vocabulary(&self) -> &Vocabulary;
    
    /// Validate model integrity
    pub fn validate(&self) -> Result<(), Error>;
}

#[derive(Clone, Debug)]
pub struct ModelMetadata {
    pub name: String,
    pub author: Option<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub parameters: u64,
    pub quantization: QuantizationType,
    pub file_size: u64,
}
```

#### Tokenizer

Text tokenization interface.

```rust
pub trait Tokenizer: Send + Sync {
    /// Tokenize text into token IDs
    fn tokenize(&self, text: &str) -> Result<Vec<TokenId>, Error>;
    
    /// Decode token IDs back to text
    fn decode(&self, tokens: &[TokenId]) -> Result<String, Error>;
    
    /// Get vocabulary size
    fn vocab_size(&self) -> usize;
    
    /// Get special tokens
    fn special_tokens(&self) -> &SpecialTokens;
}

pub struct BPETokenizer {
    // Private fields
}

impl BPETokenizer {
    pub fn new(vocab: Vocabulary, merges: Vec<(String, String)>) -> Self;
}

pub struct SentencePieceTokenizer {
    // Private fields
}

impl SentencePieceTokenizer {
    pub fn from_file(path: &Path) -> Result<Self, Error>;
}
```

### Advanced Features

#### Quantization

```rust
pub enum QuantizationType {
    F32,
    F16,
    Q8_0,
    Q5_0,
    Q5_1,
    Q4_0,
    Q4_1,
    Q2_K,
}

pub fn quantize_model(
    input_path: &Path,
    output_path: &Path,
    quantization: QuantizationType,
    progress_callback: impl Fn(f32),
) -> Result<(), Error>;
```

#### Memory Mapping

```rust
pub struct MmapConfig {
    pub enabled: bool,
    pub prefetch: bool,
    pub huge_pages: bool,
}

pub trait MemoryBackend {
    fn allocate(&self, size: usize) -> Result<*mut u8, Error>;
    fn deallocate(&self, ptr: *mut u8, size: usize);
    fn read(&self, offset: usize, buf: &mut [u8]) -> Result<(), Error>;
}
```

#### Custom Samplers

```rust
pub trait Sampler: Send + Sync {
    fn sample(
        &mut self,
        logits: &[f32],
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> TokenId;
}

pub struct DefaultSampler {
    // Private fields
}

pub struct MirostatSampler {
    pub tau: f32,
    pub eta: f32,
    pub mu: f32,
}
```

## HTTP API

### Base URL

```
http://localhost:11434
```

### Authentication

```http
Authorization: Bearer <api_key>
```

### Endpoints

#### Health Check

Check if the server is running.

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime": 3600
}
```

#### List Models

Get available models.

```http
GET /api/models
```

**Response:**
```json
[
  {
    "name": "llama-7b-q4",
    "size": 3825819648,
    "quantization": "Q4_0",
    "loaded": true,
    "metadata": {
      "author": "Meta",
      "parameters": "7B",
      "license": "Llama 2"
    }
  }
]
```

#### Load Model

Load a model into memory.

```http
POST /api/models/load
Content-Type: application/json

{
  "name": "llama-7b-q4",
  "gpu_layers": 32
}
```

**Response:**
```json
{
  "success": true,
  "message": "Model loaded successfully",
  "load_time_ms": 2340
}
```

#### Unload Model

Unload the current model.

```http
POST /api/models/unload
```

**Response:**
```json
{
  "success": true,
  "message": "Model unloaded"
}
```

#### Generate Completion

Generate text completion.

```http
POST /api/generate
Content-Type: application/json

{
  "prompt": "Once upon a time",
  "max_tokens": 200,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "repeat_penalty": 1.1,
  "stream": false,
  "stop": ["\n\n", "###"],
  "seed": 42
}
```

**Response (non-streaming):**
```json
{
  "text": "Once upon a time, in a faraway kingdom...",
  "tokens_generated": 50,
  "generation_time_ms": 1234,
  "tokens_per_second": 40.5,
  "finish_reason": "stop"
}
```

**Response (streaming):**
```
data: {"token": "Once", "id": 0}
data: {"token": " upon", "id": 1}
data: {"token": " a", "id": 2}
data: {"token": " time", "id": 3}
data: [DONE]
```

#### Chat Completion

Generate chat response.

```http
POST /api/chat
Content-Type: application/json

{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 200,
  "temperature": 0.7,
  "stream": false
}
```

**Response:**
```json
{
  "content": "The capital of France is Paris.",
  "role": "assistant",
  "tokens_generated": 8,
  "generation_time_ms": 234,
  "finish_reason": "stop"
}
```

#### Tokenize

Tokenize text.

```http
POST /api/tokenize
Content-Type: application/json

{
  "text": "Hello, world!"
}
```

**Response:**
```json
{
  "tokens": [15043, 29892, 3186, 29991],
  "count": 4
}
```

#### Detokenize

Convert tokens back to text.

```http
POST /api/detokenize
Content-Type: application/json

{
  "tokens": [15043, 29892, 3186, 29991]
}
```

**Response:**
```json
{
  "text": "Hello, world!"
}
```

#### Embeddings

Generate text embeddings.

```http
POST /api/embeddings
Content-Type: application/json

{
  "input": "The quick brown fox",
  "model": "llama-7b"
}
```

**Response:**
```json
{
  "embeddings": [[0.123, -0.456, 0.789, ...]],
  "dimensions": 4096,
  "model": "llama-7b"
}
```

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:11434/ws');
```

### Protocol

#### Client Messages

**Generate Request:**
```json
{
  "type": "generate",
  "id": "req-123",
  "prompt": "Tell me a story",
  "config": {
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

**Chat Request:**
```json
{
  "type": "chat",
  "id": "req-456",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "config": {
    "max_tokens": 200,
    "temperature": 0.7
  }
}
```

**Cancel Request:**
```json
{
  "type": "cancel",
  "id": "req-123"
}
```

#### Server Messages

**Token Event:**
```json
{
  "type": "token",
  "request_id": "req-123",
  "token": "Once",
  "index": 0,
  "timestamp": 1234567890
}
```

**Complete Event:**
```json
{
  "type": "complete",
  "request_id": "req-123",
  "text": "Once upon a time...",
  "tokens_generated": 50,
  "generation_time_ms": 1234,
  "finish_reason": "stop"
}
```

**Error Event:**
```json
{
  "type": "error",
  "request_id": "req-123",
  "error": "Model not loaded",
  "code": "MODEL_NOT_LOADED"
}
```

**Status Event:**
```json
{
  "type": "status",
  "model": "llama-7b",
  "active_sessions": 3,
  "memory_used": 4294967296,
  "gpu_memory_used": 2147483648
}
```

### Example WebSocket Client

```javascript
class WoollyWebSocket {
  constructor(url = 'ws://localhost:11434/ws') {
    this.url = url;
    this.ws = null;
    this.handlers = new Map();
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);
      
      this.ws.onopen = () => {
        console.log('Connected to Woolly');
        resolve();
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };
      
      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        this.handleMessage(message);
      };
    });
  }

  handleMessage(message) {
    switch (message.type) {
      case 'token':
        this.handlers.get(message.request_id)?.onToken?.(message);
        break;
      case 'complete':
        this.handlers.get(message.request_id)?.onComplete?.(message);
        this.handlers.delete(message.request_id);
        break;
      case 'error':
        this.handlers.get(message.request_id)?.onError?.(message);
        this.handlers.delete(message.request_id);
        break;
      case 'status':
        this.onStatus?.(message);
        break;
    }
  }

  generate(prompt, config, handlers) {
    const id = `req-${Date.now()}`;
    this.handlers.set(id, handlers);
    
    this.ws.send(JSON.stringify({
      type: 'generate',
      id,
      prompt,
      config
    }));
    
    return id;
  }

  cancel(requestId) {
    this.ws.send(JSON.stringify({
      type: 'cancel',
      id: requestId
    }));
  }

  close() {
    this.ws?.close();
  }
}

// Usage
const client = new WoollyWebSocket();
await client.connect();

client.generate(
  'Tell me a story',
  { max_tokens: 200, temperature: 0.7 },
  {
    onToken: (msg) => console.log('Token:', msg.token),
    onComplete: (msg) => console.log('Complete:', msg.text),
    onError: (msg) => console.error('Error:', msg.error)
  }
);
```

## Configuration

### Server Configuration

```yaml
# woolly.config.yaml
server:
  host: 0.0.0.0
  port: 11434
  cors:
    enabled: true
    origins: ["*"]
  
models:
  directory: ./models
  preload: []
  
inference:
  max_batch_size: 4
  max_sequence_length: 4096
  gpu:
    enabled: true
    layers: 32
  cpu:
    threads: 8
    
cache:
  enabled: true
  size: 1073741824  # 1GB
  ttl: 3600         # 1 hour
  
logging:
  level: info
  format: json
  output: stdout
  
security:
  api_key_required: false
  rate_limiting:
    enabled: true
    requests_per_minute: 60
```

### Environment Variables

```bash
# Server configuration
WOOLLY_HOST=0.0.0.0
WOOLLY_PORT=11434

# Model directory
WOOLLY_MODELS_DIR=/path/to/models

# GPU configuration
WOOLLY_GPU_ENABLED=true
WOOLLY_GPU_LAYERS=32

# CPU configuration
WOOLLY_CPU_THREADS=8

# Logging
WOOLLY_LOG_LEVEL=info
WOOLLY_LOG_FORMAT=json

# Security
WOOLLY_API_KEY=your-secret-key
WOOLLY_RATE_LIMIT=60
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "message": "Model not found: llama-7b",
    "code": "MODEL_NOT_FOUND",
    "details": {
      "model": "llama-7b",
      "available_models": ["llama-13b", "mistral-7b"]
    }
  }
}
```

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `MODEL_NOT_FOUND` | Requested model doesn't exist | 404 |
| `MODEL_NOT_LOADED` | No model is currently loaded | 400 |
| `MODEL_LOAD_FAILED` | Failed to load the model | 500 |
| `INVALID_REQUEST` | Request validation failed | 400 |
| `GENERATION_FAILED` | Text generation failed | 500 |
| `OUT_OF_MEMORY` | Insufficient memory | 507 |
| `GPU_ERROR` | GPU operation failed | 500 |
| `TOKENIZER_ERROR` | Tokenization failed | 500 |
| `RATE_LIMITED` | Too many requests | 429 |
| `UNAUTHORIZED` | Invalid or missing API key | 401 |
| `INTERNAL_ERROR` | Unexpected server error | 500 |

### Error Handling Best Practices

```rust
// Rust error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WoollyError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Model load failed: {0}")]
    ModelLoadFailed(#[source] Box<dyn std::error::Error>),
    
    #[error("Out of memory: need {needed} bytes, available {available}")]
    OutOfMemory { needed: usize, available: usize },
    
    #[error("GPU error: {0}")]
    GpuError(String),
}

// Convert to HTTP response
impl From<WoollyError> for HttpResponse {
    fn from(error: WoollyError) -> Self {
        match error {
            WoollyError::ModelNotFound(_) => {
                HttpResponse::NotFound().json(ErrorResponse::from(error))
            }
            WoollyError::OutOfMemory { .. } => {
                HttpResponse::InsufficientStorage().json(ErrorResponse::from(error))
            }
            _ => HttpResponse::InternalServerError().json(ErrorResponse::from(error))
        }
    }
}
```

## Examples

### Basic Text Generation

```rust
use woolly_core::{Engine, EngineConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create engine
    let config = EngineConfig::builder()
        .enable_gpu(true)
        .max_sequence_length(2048)
        .build();
    
    let mut engine = Engine::new(config)?;
    
    // Load model
    engine.load_model(Path::new("models/llama-7b-q4.gguf"))?;
    
    // Create session
    let mut session = engine.create_session()?;
    
    // Generate text
    let response = session.generate(
        "Once upon a time",
        100,  // max_tokens
        0.7,  // temperature
        0.9,  // top_p
    ).await?;
    
    println!("Generated: {}", response);
    
    Ok(())
}
```

### Streaming Generation

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = Engine::new(EngineConfig::default())?;
    engine.load_model(Path::new("models/llama-7b-q4.gguf"))?;
    
    let mut session = engine.create_session()?;
    
    // Create channel for streaming
    let (tx, mut rx) = mpsc::channel(100);
    
    // Start generation in background
    let handle = tokio::spawn(async move {
        session.generate_stream(
            "Tell me a story about",
            200,
            0.7,
            0.9,
            tx,
        ).await
    });
    
    // Process tokens as they arrive
    while let Some(token) = rx.recv().await {
        print!("{}", token);
        std::io::stdout().flush()?;
    }
    
    // Wait for completion
    handle.await??;
    
    Ok(())
}
```

### Custom Tokenizer

```rust
use woolly_core::tokenizer::{Tokenizer, BPETokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load custom tokenizer
    let vocab = Vocabulary::from_file("tokenizer/vocab.json")?;
    let merges = load_merges("tokenizer/merges.txt")?;
    
    let tokenizer = BPETokenizer::new(vocab, merges);
    
    // Tokenize text
    let text = "Hello, world!";
    let tokens = tokenizer.tokenize(text)?;
    println!("Tokens: {:?}", tokens);
    
    // Decode back
    let decoded = tokenizer.decode(&tokens)?;
    println!("Decoded: {}", decoded);
    
    Ok(())
}
```

### HTTP Client Example

```python
import requests
import json

class WoollyClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        
    def generate(self, prompt, **kwargs):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 200),
                "temperature": kwargs.get("temperature", 0.7),
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["text"]
    
    def generate_stream(self, prompt, **kwargs):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 200),
                "temperature": kwargs.get("temperature", 0.7),
                "stream": True
            },
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    yield json.loads(data)["token"]

# Usage
client = WoollyClient()

# Non-streaming
text = client.generate("Once upon a time")
print(text)

# Streaming
for token in client.generate_stream("Tell me a story"):
    print(token, end='', flush=True)
```

### JavaScript/TypeScript Client

```typescript
interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repeatPenalty?: number;
  stream?: boolean;
}

class WoollyClient {
  constructor(private baseUrl = 'http://localhost:11434') {}

  async generate(prompt: string, options: GenerateOptions = {}): Promise<string> {
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_tokens: options.maxTokens ?? 200,
        temperature: options.temperature ?? 0.7,
        top_p: options.topP ?? 0.9,
        stream: false
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.text;
  }

  async *generateStream(
    prompt: string, 
    options: GenerateOptions = {}
  ): AsyncGenerator<string> {
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_tokens: options.maxTokens ?? 200,
        temperature: options.temperature ?? 0.7,
        stream: true
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          
          try {
            const json = JSON.parse(data);
            yield json.token;
          } catch (e) {
            console.error('Failed to parse:', data);
          }
        }
      }
    }
  }
}

// Usage
const client = new WoollyClient();

// Non-streaming
const text = await client.generate('Once upon a time');
console.log(text);

// Streaming
for await (const token of client.generateStream('Tell me a story')) {
  process.stdout.write(token);
}
```