//! MCP Integration Example
//!
//! This example demonstrates a complete MCP (Model Context Protocol) workflow with Woolly:
//! 1. Setting up an MCP server
//! 2. Registering tools and resources
//! 3. Handling MCP protocol messages
//! 4. Creating custom plugins
//! 5. Integrating with the Woolly inference engine
//!
//! The example shows how to build a complete MCP-enabled LLM server that can handle
//! tool calls, resource access, and complex multi-step workflows.
//!
//! Usage:
//!   cargo run --example mcp_integration -- --model path/to/model.gguf --port 8080

use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::time::sleep;
use serde_json::Value;

use woolly_core::prelude::*;
use woolly_gguf::GGUFLoader;
use woolly_mcp::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let (model_path, port) = parse_args(&args)?;
    
    println!("🦙 Woolly MCP Integration Example");
    println!("Model: {}", model_path);
    println!("MCP Server Port: {}", port);
    println!();
    
    // Step 1: Initialize Woolly inference engine
    println!("🔧 Setting up Woolly inference engine...");
    let inference_engine = setup_inference_engine(&model_path).await?;
    
    // Step 2: Create MCP server with custom plugins
    println!("🌐 Creating MCP server...");
    let mcp_server = create_mcp_server(inference_engine).await?;
    
    // Step 3: Register tools and resources
    println!("🛠️ Registering MCP tools and resources...");
    register_tools_and_resources(&mcp_server).await?;
    
    // Step 4: Start the MCP server
    println!("🚀 Starting MCP server on port {}...", port);
    let server_handle = start_server(mcp_server, port).await?;
    
    // Step 5: Demonstrate MCP protocol usage
    println!("📡 Demonstrating MCP protocol...");
    demonstrate_mcp_protocol(port).await?;
    
    // Step 6: Handle shutdown gracefully
    println!("⏳ Running server (press Ctrl+C to stop)...");
    tokio::signal::ctrl_c().await?;
    
    println!("🛑 Shutting down server...");
    server_handle.shutdown().await?;
    
    println!("✨ MCP integration example completed!");
    Ok(())
}

fn parse_args(args: &[String]) -> Result<(String, u16), Box<dyn std::error::Error>> {
    let mut model_path = None;
    let mut port = 8080u16;
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    model_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    return Err("Missing model path".into());
                }
            }
            "--port" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse().map_err(|_| "Invalid port number")?;
                    i += 2;
                } else {
                    return Err("Missing port number".into());
                }
            }
            _ => i += 1,
        }
    }
    
    let model_path = model_path.ok_or("Model path not provided. Use --model <path>")?;
    Ok((model_path, port))
}

async fn setup_inference_engine(model_path: &str) -> Result<Arc<InferenceEngine>, Box<dyn std::error::Error>> {
    // Load GGUF model
    let gguf_loader = GGUFLoader::from_path(model_path)?;
    
    // Create model instance
    let model = create_model_from_gguf(&gguf_loader)?;
    
    // Configure engine
    let config = EngineConfig {
        max_context_length: 4096,
        max_batch_size: 4, // Support multiple concurrent requests
        num_threads: num_cpus::get(),
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
        ..Default::default()
    };
    
    // Create and initialize engine
    let mut engine = InferenceEngine::new(config);
    engine.load_model(Arc::new(model)).await?;
    
    Ok(Arc::new(engine))
}

async fn create_mcp_server(
    inference_engine: Arc<InferenceEngine>,
) -> Result<McpServer, Box<dyn std::error::Error>> {
    // Create MCP server with Woolly integration
    let server = McpServer::builder()
        .with_name("woolly-mcp-server")
        .with_version("0.1.0")
        .with_inference_engine(inference_engine)
        .with_protocol_version("2024-11-05")
        .build()
        .await?;
    
    Ok(server)
}

async fn register_tools_and_resources(
    server: &McpServer,
) -> Result<(), Box<dyn std::error::Error>> {
    // Register built-in tools
    register_built_in_tools(server).await?;
    
    // Register custom tools
    register_custom_tools(server).await?;
    
    // Register resources
    register_resources(server).await?;
    
    // Register prompts
    register_prompts(server).await?;
    
    Ok(())
}

async fn register_built_in_tools(
    server: &McpServer,
) -> Result<(), Box<dyn std::error::Error>> {
    // Text generation tool
    server.register_tool(TextGenerationTool::new()).await?;
    
    // Token counting tool
    server.register_tool(TokenCountTool::new()).await?;
    
    // Model information tool
    server.register_tool(ModelInfoTool::new()).await?;
    
    // Tokenization tool
    server.register_tool(TokenizationTool::new()).await?;
    
    println!("✅ Registered built-in tools");
    Ok(())
}

async fn register_custom_tools(
    server: &McpServer,
) -> Result<(), Box<dyn std::error::Error>> {
    // Custom code analysis tool
    server.register_tool(CodeAnalysisTool::new()).await?;
    
    // Custom summarization tool
    server.register_tool(SummarizationTool::new()).await?;
    
    // Custom batch processing tool
    server.register_tool(BatchProcessingTool::new()).await?;
    
    println!("✅ Registered custom tools");
    Ok(())
}

async fn register_resources(
    server: &McpServer,
) -> Result<(), Box<dyn std::error::Error>> {
    // Model configuration resource
    server.register_resource(ModelConfigResource::new()).await?;
    
    // Performance metrics resource
    server.register_resource(MetricsResource::new()).await?;
    
    // System information resource
    server.register_resource(SystemInfoResource::new()).await?;
    
    println!("✅ Registered resources");
    Ok(())
}

async fn register_prompts(
    server: &McpServer,
) -> Result<(), Box<dyn std::error::Error>> {
    // Code review prompt
    server.register_prompt(CodeReviewPrompt::new()).await?;
    
    // Documentation prompt
    server.register_prompt(DocumentationPrompt::new()).await?;
    
    // Translation prompt
    server.register_prompt(TranslationPrompt::new()).await?;
    
    println!("✅ Registered prompts");
    Ok(())
}

async fn start_server(
    server: McpServer,
    port: u16,
) -> Result<ServerHandle, Box<dyn std::error::Error>> {
    // Configure server endpoints
    let config = ServerConfig {
        port,
        host: "localhost".to_string(),
        enable_cors: true,
        max_connections: 100,
        request_timeout: Duration::from_secs(30),
        enable_metrics: true,
        enable_logging: true,
    };
    
    // Start the server
    let handle = server.start(config).await?;
    
    println!("🟢 MCP server running at http://localhost:{}", port);
    println!("📊 Metrics available at http://localhost:{}/metrics", port);
    println!("📚 API documentation at http://localhost:{}/docs", port);
    
    Ok(handle)
}

async fn demonstrate_mcp_protocol(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    // Wait for server to be ready
    sleep(Duration::from_millis(1000)).await;
    
    println!("\n🔍 Demonstrating MCP protocol interactions...");
    
    // Create MCP client
    let client = McpClient::new(&format!("http://localhost:{}", port)).await?;
    
    // 1. Initialize connection
    println!("1️⃣ Initializing MCP connection...");
    let init_result = client.initialize("woolly-mcp-example", "0.1.0").await?;
    println!("   Server capabilities: {:?}", init_result.capabilities);
    
    // 2. List available tools
    println!("2️⃣ Listing available tools...");
    let tools = client.list_tools().await?;
    for tool in &tools {
        println!("   🛠️ {}: {}", tool.name, tool.description);
    }
    
    // 3. Call text generation tool
    println!("3️⃣ Calling text generation tool...");
    let generation_args = serde_json::json!({
        "prompt": "Explain the benefits of Rust for systems programming",
        "max_tokens": 150,
        "temperature": 0.7
    });
    let generation_result = client.call_tool("text_generation", generation_args).await?;
    println!("   Generated text: {}", generation_result["content"]);
    
    // 4. Get model information
    println!("4️⃣ Getting model information...");
    let model_info = client.call_tool("model_info", serde_json::json!({})).await?;
    println!("   Model info: {}", model_info);
    
    // 5. List resources
    println!("5️⃣ Listing available resources...");
    let resources = client.list_resources().await?;
    for resource in &resources {
        println!("   📄 {}: {}", resource.uri, resource.description);
    }
    
    // 6. Access performance metrics
    println!("6️⃣ Accessing performance metrics...");
    let metrics = client.read_resource("metrics://performance").await?;
    println!("   Metrics: {}", metrics["content"]);
    
    // 7. List prompts
    println!("7️⃣ Listing available prompts...");
    let prompts = client.list_prompts().await?;
    for prompt in &prompts {
        println!("   💬 {}: {}", prompt.name, prompt.description);
    }
    
    // 8. Get a prompt template
    println!("8️⃣ Getting code review prompt...");
    let prompt_args = serde_json::json!({
        "language": "rust",
        "complexity": "high"
    });
    let prompt_result = client.get_prompt("code_review", Some(prompt_args)).await?;
    println!("   Prompt messages: {}", prompt_result["messages"].as_array().unwrap().len());
    
    println!("✅ MCP protocol demonstration completed!");
    Ok(())
}

fn create_model_from_gguf(_loader: &GGUFLoader) -> Result<impl Model, Box<dyn std::error::Error>> {
    // Create a simplified model for demonstration
    // In practice, this would parse the GGUF metadata and create the appropriate model type
    Ok(ExampleMcpModel::new())
}

// Example implementations of MCP components

#[derive(Clone)]
struct ExampleMcpModel;

impl ExampleMcpModel {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Model for ExampleMcpModel {
    fn name(&self) -> &str { "example-mcp-model" }
    fn model_type(&self) -> &str { "llama" }
    fn vocab_size(&self) -> usize { 32000 }
    fn context_length(&self) -> usize { 4096 }
    fn hidden_size(&self) -> usize { 4096 }
    fn num_layers(&self) -> usize { 32 }
    fn num_heads(&self) -> usize { 32 }
    
    async fn forward(
        &self,
        _input_ids: &[u32],
        _past_kv_cache: Option<&(dyn std::any::Any + Send + Sync)>,
    ) -> Result<ModelOutput> {
        // Simplified forward pass for demonstration
        Ok(ModelOutput {
            logits: vec![0.1; self.vocab_size()],
            logits_shape: vec![1, 1, self.vocab_size()],
            past_kv_cache: None,
            hidden_states: None,
            attentions: None,
        })
    }
    
    async fn load_weights(&mut self, _path: &Path) -> Result<()> {
        Ok(())
    }
}

// Tool implementations

struct TextGenerationTool;

impl TextGenerationTool {
    fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl ToolHandler for TextGenerationTool {
    fn name(&self) -> &str { "text_generation" }
    
    fn description(&self) -> &str {
        "Generate text using the Woolly inference engine"
    }
    
    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The input prompt for text generation"
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum number of tokens to generate",
                    "default": 100
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature",
                    "default": 0.8
                }
            },
            "required": ["prompt"]
        })
    }
    
    async fn handle(&self, args: Value) -> Result<Value, String> {
        let prompt = args["prompt"].as_str().ok_or("Missing prompt")?;
        let max_tokens = args["max_tokens"].as_u64().unwrap_or(100);
        let temperature = args["temperature"].as_f64().unwrap_or(0.8);
        
        // Simulate text generation
        let generated_text = format!(
            "Generated response to '{}' (max_tokens: {}, temperature: {}): \
            This is a simulated response from the Woolly inference engine. \
            In a real implementation, this would use the actual model for generation.",
            prompt, max_tokens, temperature
        );
        
        Ok(serde_json::json!({
            "content": generated_text,
            "tokens_generated": max_tokens.min(50),
            "finish_reason": "max_tokens"
        }))
    }
}

struct TokenCountTool;

impl TokenCountTool {
    fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl ToolHandler for TokenCountTool {
    fn name(&self) -> &str { "token_count" }
    
    fn description(&self) -> &str {
        "Count the number of tokens in the given text"
    }
    
    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to tokenize and count"
                }
            },
            "required": ["text"]
        })
    }
    
    async fn handle(&self, args: Value) -> Result<Value, String> {
        let text = args["text"].as_str().ok_or("Missing text")?;
        
        // Simple token counting (word-based for demo)
        let token_count = text.split_whitespace().count();
        
        Ok(serde_json::json!({
            "token_count": token_count,
            "character_count": text.len(),
            "word_count": token_count
        }))
    }
}

struct ModelInfoTool;

impl ModelInfoTool {
    fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl ToolHandler for ModelInfoTool {
    fn name(&self) -> &str { "model_info" }
    
    fn description(&self) -> &str {
        "Get information about the loaded model"
    }
    
    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }
    
    async fn handle(&self, _args: Value) -> Result<Value, String> {
        Ok(serde_json::json!({
            "name": "example-mcp-model",
            "type": "llama",
            "vocab_size": 32000,
            "context_length": 4096,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "parameters": "7B"
        }))
    }
}

struct TokenizationTool;

impl TokenizationTool {
    fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl ToolHandler for TokenizationTool {
    fn name(&self) -> &str { "tokenization" }
    
    fn description(&self) -> &str {
        "Tokenize text and optionally decode tokens back to text"
    }
    
    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to tokenize"
                },
                "tokens": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Tokens to decode (optional)"
                },
                "operation": {
                    "type": "string",
                    "enum": ["encode", "decode", "both"],
                    "default": "encode"
                }
            }
        })
    }
    
    async fn handle(&self, args: Value) -> Result<Value, String> {
        let operation = args["operation"].as_str().unwrap_or("encode");
        
        match operation {
            "encode" => {
                let text = args["text"].as_str().ok_or("Missing text for encoding")?;
                let tokens: Vec<u32> = text.split_whitespace()
                    .enumerate()
                    .map(|(i, _)| (i as u32) % 1000 + 1)
                    .collect();
                
                Ok(serde_json::json!({
                    "tokens": tokens,
                    "token_count": tokens.len()
                }))
            }
            "decode" => {
                let tokens = args["tokens"].as_array().ok_or("Missing tokens for decoding")?;
                let token_ids: Vec<u32> = tokens.iter()
                    .map(|t| t.as_u64().unwrap_or(0) as u32)
                    .collect();
                
                let decoded_text = format!("decoded_token_{}", token_ids.len());
                
                Ok(serde_json::json!({
                    "text": decoded_text,
                    "token_count": token_ids.len()
                }))
            }
            "both" => {
                let text = args["text"].as_str().ok_or("Missing text")?;
                let tokens: Vec<u32> = text.split_whitespace()
                    .enumerate()
                    .map(|(i, _)| (i as u32) % 1000 + 1)
                    .collect();
                
                Ok(serde_json::json!({
                    "original_text": text,
                    "tokens": tokens,
                    "decoded_text": format!("decoded_{}", text),
                    "token_count": tokens.len()
                }))
            }
            _ => Err("Invalid operation".to_string())
        }
    }
}

// Custom tool implementations

struct CodeAnalysisTool;

impl CodeAnalysisTool {
    fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl ToolHandler for CodeAnalysisTool {
    fn name(&self) -> &str { "code_analysis" }
    
    fn description(&self) -> &str {
        "Analyze code for complexity, patterns, and potential issues"
    }
    
    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to analyze"
                },
                "language": {
                    "type": "string",
                    "description": "Programming language",
                    "default": "rust"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["complexity", "security", "performance", "all"],
                    "default": "all"
                }
            },
            "required": ["code"]
        })
    }
    
    async fn handle(&self, args: Value) -> Result<Value, String> {
        let code = args["code"].as_str().ok_or("Missing code")?;
        let language = args["language"].as_str().unwrap_or("rust");
        let analysis_type = args["analysis_type"].as_str().unwrap_or("all");
        
        // Simulate code analysis
        Ok(serde_json::json!({
            "language": language,
            "lines_of_code": code.lines().count(),
            "complexity_score": 3.5,
            "security_issues": [],
            "performance_suggestions": [
                "Consider using iterators instead of loops",
                "Use Vec::with_capacity for better memory allocation"
            ],
            "analysis_type": analysis_type,
            "summary": "Code analysis completed successfully"
        }))
    }
}

struct SummarizationTool;

impl SummarizationTool {
    fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl ToolHandler for SummarizationTool {
    fn name(&self) -> &str { "summarization" }
    
    fn description(&self) -> &str {
        "Summarize long text into key points"
    }
    
    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarize"
                },
                "max_summary_length": {
                    "type": "integer",
                    "description": "Maximum length of summary in words",
                    "default": 100
                },
                "summary_type": {
                    "type": "string",
                    "enum": ["extractive", "abstractive", "bullet_points"],
                    "default": "abstractive"
                }
            },
            "required": ["text"]
        })
    }
    
    async fn handle(&self, args: Value) -> Result<Value, String> {
        let text = args["text"].as_str().ok_or("Missing text")?;
        let max_length = args["max_summary_length"].as_u64().unwrap_or(100);
        let summary_type = args["summary_type"].as_str().unwrap_or("abstractive");
        
        // Simulate summarization
        Ok(serde_json::json!({
            "summary": format!("This is a {} summary of the provided text (max {} words)", summary_type, max_length),
            "key_points": [
                "Main topic discussed",
                "Key arguments presented",
                "Conclusions reached"
            ],
            "original_length": text.len(),
            "summary_ratio": 0.1
        }))
    }
}

struct BatchProcessingTool;

impl BatchProcessingTool {
    fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl ToolHandler for BatchProcessingTool {
    fn name(&self) -> &str { "batch_processing" }
    
    fn description(&self) -> &str {
        "Process multiple texts in a single batch operation"
    }
    
    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of texts to process"
                },
                "operation": {
                    "type": "string",
                    "enum": ["tokenize", "generate", "summarize", "analyze"],
                    "description": "Operation to perform on each text"
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Number of texts to process simultaneously",
                    "default": 4
                }
            },
            "required": ["texts", "operation"]
        })
    }
    
    async fn handle(&self, args: Value) -> Result<Value, String> {
        let texts = args["texts"].as_array().ok_or("Missing texts array")?;
        let operation = args["operation"].as_str().ok_or("Missing operation")?;
        let batch_size = args["batch_size"].as_u64().unwrap_or(4);
        
        let mut results = Vec::new();
        
        for (i, text_value) in texts.iter().enumerate() {
            let text = text_value.as_str().unwrap_or("");
            let result = match operation {
                "tokenize" => serde_json::json!({
                    "index": i,
                    "token_count": text.split_whitespace().count()
                }),
                "generate" => serde_json::json!({
                    "index": i,
                    "generated": format!("Generated response for: {}", text)
                }),
                "summarize" => serde_json::json!({
                    "index": i,
                    "summary": format!("Summary of: {}", text)
                }),
                "analyze" => serde_json::json!({
                    "index": i,
                    "analysis": format!("Analysis of: {}", text)
                }),
                _ => serde_json::json!({
                    "index": i,
                    "error": "Unknown operation"
                })
            };
            results.push(result);
        }
        
        Ok(serde_json::json!({
            "results": results,
            "processed_count": texts.len(),
            "batch_size": batch_size,
            "operation": operation
        }))
    }
}

// Resource implementations

struct ModelConfigResource;

impl ModelConfigResource {
    fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl ResourceHandler for ModelConfigResource {
    fn uri(&self) -> &str { "config://model" }
    
    fn name(&self) -> &str { "Model Configuration" }
    
    fn description(&self) -> &str {
        "Current model configuration and parameters"
    }
    
    fn mime_type(&self) -> &str { "application/json" }
    
    async fn read(&self, _args: Option<Value>) -> Result<Value, String> {
        Ok(serde_json::json!({
            "model_type": "llama",
            "vocab_size": 32000,
            "context_length": 4096,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "parameters": "7B",
            "quantization": "f16",
            "device": "cpu"
        }))
    }
}

struct MetricsResource;

impl MetricsResource {
    fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl ResourceHandler for MetricsResource {
    fn uri(&self) -> &str { "metrics://performance" }
    
    fn name(&self) -> &str { "Performance Metrics" }
    
    fn description(&self) -> &str {
        "Real-time performance metrics and statistics"
    }
    
    fn mime_type(&self) -> &str { "application/json" }
    
    async fn read(&self, _args: Option<Value>) -> Result<Value, String> {
        Ok(serde_json::json!({
            "requests_processed": 42,
            "avg_response_time_ms": 125.7,
            "tokens_generated_per_sec": 35.2,
            "memory_usage_mb": 2048,
            "cpu_usage_percent": 45.3,
            "uptime_seconds": 3600,
            "cache_hit_rate": 0.85
        }))
    }
}

struct SystemInfoResource;

impl SystemInfoResource {
    fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl ResourceHandler for SystemInfoResource {
    fn uri(&self) -> &str { "system://info" }
    
    fn name(&self) -> &str { "System Information" }
    
    fn description(&self) -> &str {
        "System hardware and software information"
    }
    
    fn mime_type(&self) -> &str { "application/json" }
    
    async fn read(&self, _args: Option<Value>) -> Result<Value, String> {
        Ok(serde_json::json!({
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "cpu_cores": num_cpus::get(),
            "woolly_version": "0.1.0",
            "rust_version": "1.75.0"
        }))
    }
}

// Prompt implementations

struct CodeReviewPrompt;

impl CodeReviewPrompt {
    fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl PromptHandler for CodeReviewPrompt {
    fn name(&self) -> &str { "code_review" }
    
    fn description(&self) -> &str {
        "Generate a comprehensive code review prompt"
    }
    
    fn input_schema(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "Programming language",
                    "default": "rust"
                },
                "complexity": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "default": "medium"
                }
            }
        }))
    }
    
    async fn generate(&self, args: Option<Value>) -> Result<Vec<Value>, String> {
        let language = args.as_ref()
            .and_then(|a| a["language"].as_str())
            .unwrap_or("rust");
        let complexity = args.as_ref()
            .and_then(|a| a["complexity"].as_str())
            .unwrap_or("medium");
        
        Ok(vec![
            serde_json::json!({
                "role": "system",
                "content": format!(
                    "You are an expert {} developer conducting a {} complexity code review. \
                    Provide detailed feedback on code quality, performance, security, and best practices.",
                    language, complexity
                )
            }),
            serde_json::json!({
                "role": "user",
                "content": "Please review the following code and provide constructive feedback:"
            })
        ])
    }
}

struct DocumentationPrompt;

impl DocumentationPrompt {
    fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl PromptHandler for DocumentationPrompt {
    fn name(&self) -> &str { "documentation" }
    
    fn description(&self) -> &str {
        "Generate documentation writing prompts"
    }
    
    fn input_schema(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "doc_type": {
                    "type": "string",
                    "enum": ["api", "tutorial", "readme", "changelog"],
                    "default": "api"
                }
            }
        }))
    }
    
    async fn generate(&self, args: Option<Value>) -> Result<Vec<Value>, String> {
        let doc_type = args.as_ref()
            .and_then(|a| a["doc_type"].as_str())
            .unwrap_or("api");
        
        Ok(vec![
            serde_json::json!({
                "role": "system",
                "content": format!(
                    "You are a technical writer creating {} documentation. \
                    Write clear, comprehensive, and user-friendly documentation.",
                    doc_type
                )
            }),
            serde_json::json!({
                "role": "user",
                "content": "Please create documentation for the following:"
            })
        ])
    }
}

struct TranslationPrompt;

impl TranslationPrompt {
    fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl PromptHandler for TranslationPrompt {
    fn name(&self) -> &str { "translation" }
    
    fn description(&self) -> &str {
        "Generate translation prompts for different languages"
    }
    
    fn input_schema(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "source_lang": {
                    "type": "string",
                    "description": "Source language",
                    "default": "english"
                },
                "target_lang": {
                    "type": "string",
                    "description": "Target language",
                    "default": "spanish"
                }
            }
        }))
    }
    
    async fn generate(&self, args: Option<Value>) -> Result<Vec<Value>, String> {
        let source_lang = args.as_ref()
            .and_then(|a| a["source_lang"].as_str())
            .unwrap_or("english");
        let target_lang = args.as_ref()
            .and_then(|a| a["target_lang"].as_str())
            .unwrap_or("spanish");
        
        Ok(vec![
            serde_json::json!({
                "role": "system",
                "content": format!(
                    "You are a professional translator specializing in {} to {} translation. \
                    Provide accurate, natural, and culturally appropriate translations.",
                    source_lang, target_lang
                )
            }),
            serde_json::json!({
                "role": "user",
                "content": format!("Please translate the following {} text to {}:", source_lang, target_lang)
            })
        ])
    }
}

// Placeholder types for compilation - these would be defined in woolly-mcp

struct McpServer;
struct ServerHandle;
struct ServerConfig {
    port: u16,
    host: String,
    enable_cors: bool,
    max_connections: u32,
    request_timeout: Duration,
    enable_metrics: bool,
    enable_logging: bool,
}

struct McpClient;

impl McpServer {
    fn builder() -> McpServerBuilder { McpServerBuilder }
    async fn start(self, _config: ServerConfig) -> Result<ServerHandle, Box<dyn std::error::Error>> {
        Ok(ServerHandle)
    }
    async fn register_tool(&self, _tool: impl ToolHandler + Send + Sync + 'static) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
    async fn register_resource(&self, _resource: impl ResourceHandler + Send + Sync + 'static) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
    async fn register_prompt(&self, _prompt: impl PromptHandler + Send + Sync + 'static) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
}

struct McpServerBuilder;

impl McpServerBuilder {
    fn with_name(self, _name: &str) -> Self { self }
    fn with_version(self, _version: &str) -> Self { self }
    fn with_inference_engine(self, _engine: Arc<InferenceEngine>) -> Self { self }
    fn with_protocol_version(self, _version: &str) -> Self { self }
    async fn build(self) -> Result<McpServer, Box<dyn std::error::Error>> { Ok(McpServer) }
}

impl ServerHandle {
    async fn shutdown(self) -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
}

impl McpClient {
    async fn new(_url: &str) -> Result<Self, Box<dyn std::error::Error>> { Ok(McpClient) }
    async fn initialize(&self, _name: &str, _version: &str) -> Result<InitResult, Box<dyn std::error::Error>> {
        Ok(InitResult { capabilities: HashMap::new() })
    }
    async fn list_tools(&self) -> Result<Vec<ToolInfo>, Box<dyn std::error::Error>> { Ok(vec![]) }
    async fn call_tool(&self, _name: &str, _args: Value) -> Result<Value, Box<dyn std::error::Error>> {
        Ok(serde_json::json!({"content": "example response"}))
    }
    async fn list_resources(&self) -> Result<Vec<ResourceInfo>, Box<dyn std::error::Error>> { Ok(vec![]) }
    async fn read_resource(&self, _uri: &str) -> Result<Value, Box<dyn std::error::Error>> {
        Ok(serde_json::json!({"content": "example resource"}))
    }
    async fn list_prompts(&self) -> Result<Vec<PromptInfo>, Box<dyn std::error::Error>> { Ok(vec![]) }
    async fn get_prompt(&self, _name: &str, _args: Option<Value>) -> Result<Value, Box<dyn std::error::Error>> {
        Ok(serde_json::json!({"messages": []}))
    }
}

struct InitResult {
    capabilities: HashMap<String, Value>,
}

struct ToolInfo {
    name: String,
    description: String,
}

struct ResourceInfo {
    uri: String,
    description: String,
}

struct PromptInfo {
    name: String,
    description: String,
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
            "--port".to_string(),
            "9090".to_string(),
        ];
        
        let (model_path, port) = parse_args(&args).unwrap();
        assert_eq!(model_path, "test.gguf");
        assert_eq!(port, 9090);
    }

    #[tokio::test]
    async fn test_tools() {
        let tool = TextGenerationTool::new();
        assert_eq!(tool.name(), "text_generation");
        
        let args = serde_json::json!({
            "prompt": "Hello, world!",
            "max_tokens": 50
        });
        
        let result = tool.handle(args).await.unwrap();
        assert!(result["content"].is_string());
    }
}