# 🦙 Woolly MCP

[![Crates.io](https://img.shields.io/crates/v/woolly-mcp.svg)](https://crates.io/crates/woolly-mcp)
[![Documentation](https://docs.rs/woolly-mcp/badge.svg)](https://docs.rs/woolly-mcp)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../../LICENSE)

Model Context Protocol (MCP) implementation for Woolly, providing a standardized interface for AI model serving, tool integration, and resource management. This crate enables seamless integration with MCP-compatible clients and applications.

## Features

- **🌐 Full MCP Compliance**: Complete implementation of Model Context Protocol specification
- **🛠️ Tool Registry**: Extensible tool system for function calling and external integrations
- **📄 Resource Management**: Efficient handling of external resources and data sources
- **💬 Prompt Templates**: Dynamic prompt generation and management
- **🔌 Multiple Transports**: WebSocket, HTTP, and stdio transport support
- **🔗 Plugin System**: Extensible architecture for custom MCP extensions
- **⚡ Async/Await**: First-class async support throughout the protocol stack
- **📊 Built-in Monitoring**: Performance metrics and request tracing

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
woolly-mcp = "0.1"
woolly-core = "0.1"  # For inference integration
```

### Basic MCP Server

```rust
use woolly_mcp::prelude::*;
use woolly_core::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create inference engine
    let engine = setup_inference_engine().await?;
    
    // Create MCP server
    let server = McpServer::builder()
        .with_name("woolly-server")
        .with_version("0.1.0")
        .with_inference_engine(engine)
        .build()
        .await?;
    
    // Register built-in tools
    server.register_tool(TextGenerationTool::new()).await?;
    server.register_tool(TokenizationTool::new()).await?;
    
    // Start server
    let config = ServerConfig {
        port: 8080,
        host: "localhost".to_string(),
        enable_cors: true,
        ..Default::default()
    };
    
    let handle = server.start(config).await?;
    println!("🌐 MCP server running on http://localhost:8080");
    
    // Keep server running
    tokio::signal::ctrl_c().await?;
    handle.shutdown().await?;
    
    Ok(())
}

async fn setup_inference_engine() -> Result<Arc<InferenceEngine>, Box<dyn std::error::Error>> {
    // Setup your inference engine here
    todo!()
}
```

### MCP Client Usage

```rust
use woolly_mcp::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to MCP server
    let client = McpClient::new("http://localhost:8080").await?;
    
    // Initialize connection
    let init_result = client.initialize("my-client", "1.0.0").await?;
    println!("Server capabilities: {:?}", init_result.capabilities);
    
    // List available tools
    let tools = client.list_tools().await?;
    for tool in &tools {
        println!("📦 {}: {}", tool.name, tool.description);
    }
    
    // Call a tool
    let result = client.call_tool("text_generation", json!({
        "prompt": "Explain quantum computing",
        "max_tokens": 150,
        "temperature": 0.7
    })).await?;
    
    println!("Generated text: {}", result["content"]);
    
    Ok(())
}
```

## Tool Development

### Creating Custom Tools

```rust
use woolly_mcp::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};

struct CodeAnalysisTool {
    // Tool state and configuration
}

impl CodeAnalysisTool {
    fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl ToolHandler for CodeAnalysisTool {
    fn name(&self) -> &str {
        "code_analysis"
    }
    
    fn description(&self) -> &str {
        "Analyze code for complexity, patterns, and potential issues"
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to analyze"
                },
                "language": {
                    "type": "string",
                    "enum": ["rust", "python", "javascript", "typescript"],
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
        
        // Perform code analysis
        let complexity_score = analyze_complexity(code, language)?;
        let security_issues = analyze_security(code, language)?;
        let performance_suggestions = analyze_performance(code, language)?;
        
        Ok(json!({
            "language": language,
            "analysis_type": analysis_type,
            "complexity_score": complexity_score,
            "security_issues": security_issues,
            "performance_suggestions": performance_suggestions,
            "lines_of_code": code.lines().count(),
            "summary": format!("Analysis complete for {} code", language)
        }))
    }
}

// Register the tool
server.register_tool(CodeAnalysisTool::new()).await?;

// Helper functions
fn analyze_complexity(code: &str, language: &str) -> Result<f64, String> {
    // Implement complexity analysis
    Ok(3.5)
}

fn analyze_security(code: &str, language: &str) -> Result<Vec<String>, String> {
    // Implement security analysis
    Ok(vec!["Consider input validation".to_string()])
}

fn analyze_performance(code: &str, language: &str) -> Result<Vec<String>, String> {
    // Implement performance analysis
    Ok(vec!["Use Vec::with_capacity for better allocation".to_string()])
}
```

### Built-in Tools

#### Text Generation Tool

```rust
use woolly_mcp::prelude::*;

// Text generation with the inference engine
let text_tool = TextGenerationTool::new();
server.register_tool(text_tool).await?;

// Usage from client
let response = client.call_tool("text_generation", json!({
    "prompt": "Write a short story about a robot",
    "max_tokens": 200,
    "temperature": 0.8,
    "top_p": 0.9
})).await?;
```

#### Tokenization Tool

```rust
use woolly_mcp::prelude::*;

// Tokenization utilities
let tokenizer_tool = TokenizationTool::new();
server.register_tool(tokenizer_tool).await?;

// Usage
let response = client.call_tool("tokenization", json!({
    "text": "Hello, world!",
    "operation": "encode"
})).await?;

println!("Tokens: {:?}", response["tokens"]);
```

#### Model Information Tool

```rust
use woolly_mcp::prelude::*;

// Model metadata and statistics
let model_info_tool = ModelInfoTool::new();
server.register_tool(model_info_tool).await?;

// Get model information
let info = client.call_tool("model_info", json!({})).await?;
println!("Model: {}", info["name"]);
println!("Parameters: {}", info["parameters"]);
```

## Resource Management

### Creating Custom Resources

```rust
use woolly_mcp::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};

struct MetricsResource {
    // Resource state
}

impl MetricsResource {
    fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl ResourceHandler for MetricsResource {
    fn uri(&self) -> &str {
        "metrics://performance"
    }
    
    fn name(&self) -> &str {
        "Performance Metrics"
    }
    
    fn description(&self) -> &str {
        "Real-time performance metrics and statistics"
    }
    
    fn mime_type(&self) -> &str {
        "application/json"
    }
    
    async fn read(&self, args: Option<Value>) -> Result<Value, String> {
        // Collect performance metrics
        let metrics = collect_performance_metrics().await?;
        
        Ok(json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "metrics": metrics,
            "uptime_seconds": get_uptime_seconds(),
            "memory_usage_mb": get_memory_usage_mb(),
            "requests_processed": get_request_count()
        }))
    }
    
    async fn write(&self, content: Value, args: Option<Value>) -> Result<(), String> {
        // Update configuration or reset metrics
        if let Some(action) = args.and_then(|a| a["action"].as_str()) {
            match action {
                "reset" => reset_metrics().await?,
                "configure" => configure_metrics(content).await?,
                _ => return Err("Unknown action".to_string()),
            }
        }
        Ok(())
    }
}

// Register the resource
server.register_resource(MetricsResource::new()).await?;

// Access from client
let metrics = client.read_resource("metrics://performance").await?;
println!("Current metrics: {}", metrics["content"]);
```

### Built-in Resources

#### Model Configuration Resource

```rust
// Access model configuration
let config = client.read_resource("config://model").await?;
println!("Model configuration: {}", config["content"]);
```

#### System Information Resource

```rust
// Get system information
let system_info = client.read_resource("system://info").await?;
println!("System: {}", system_info["content"]);
```

## Prompt Templates

### Creating Custom Prompts

```rust
use woolly_mcp::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};

struct CodeReviewPrompt;

impl CodeReviewPrompt {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl PromptHandler for CodeReviewPrompt {
    fn name(&self) -> &str {
        "code_review"
    }
    
    fn description(&self) -> &str {
        "Generate a comprehensive code review prompt"
    }
    
    fn input_schema(&self) -> Option<Value> {
        Some(json!({
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
                },
                "focus": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["performance", "security", "maintainability", "style"]
                    },
                    "default": ["performance", "security", "maintainability"]
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
        let focus_areas = args.as_ref()
            .and_then(|a| a["focus"].as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_else(|| vec!["performance", "security", "maintainability"]);
        
        let system_prompt = format!(
            "You are an expert {} developer conducting a {} complexity code review. \
            Focus on: {}. Provide detailed, constructive feedback.",
            language, complexity, focus_areas.join(", ")
        );
        
        Ok(vec![
            json!({
                "role": "system",
                "content": system_prompt
            }),
            json!({
                "role": "user",
                "content": "Please review the following code:"
            })
        ])
    }
}

// Register the prompt
server.register_prompt(CodeReviewPrompt::new()).await?;

// Use from client
let prompt = client.get_prompt("code_review", Some(json!({
    "language": "rust",
    "complexity": "high",
    "focus": ["performance", "security"]
}))).await?;
```

## Transport Protocols

### WebSocket Server

```rust
use woolly_mcp::prelude::*;

let server_config = ServerConfig {
    port: 8080,
    host: "localhost".to_string(),
    websocket_enabled: true,
    websocket_path: "/ws".to_string(),
    max_connections: 100,
    ..Default::default()
};

let handle = server.start(server_config).await?;
```

### HTTP REST API

```rust
use woolly_mcp::prelude::*;

let server_config = ServerConfig {
    port: 8080,
    host: "localhost".to_string(),
    http_enabled: true,
    enable_cors: true,
    cors_origins: vec!["*".to_string()],
    ..Default::default()
};

// API endpoints available at:
// GET  /mcp/tools          - List tools
// POST /mcp/tools/{name}   - Call tool
// GET  /mcp/resources      - List resources
// GET  /mcp/resources/{uri} - Read resource
// GET  /mcp/prompts        - List prompts
// POST /mcp/prompts/{name} - Get prompt
```

### Stdio Transport

```rust
use woolly_mcp::prelude::*;

// For command-line integration
let server = McpServer::builder()
    .with_transport(Transport::Stdio)
    .build()
    .await?;

server.run_stdio().await?;
```

## Plugin System

### Creating Plugins

```rust
use woolly_mcp::prelude::*;
use async_trait::async_trait;

struct DatabasePlugin {
    connection_pool: DatabasePool,
}

#[async_trait]
impl Plugin for DatabasePlugin {
    fn name(&self) -> &str {
        "database"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    async fn initialize(&mut self, registry: &mut PluginRegistry) -> Result<(), String> {
        // Register database-related tools
        registry.register_tool(DatabaseQueryTool::new(self.connection_pool.clone())).await?;
        registry.register_resource(DatabaseSchemaResource::new(self.connection_pool.clone())).await?;
        
        Ok(())
    }
    
    async fn shutdown(&mut self) -> Result<(), String> {
        self.connection_pool.close().await?;
        Ok(())
    }
}

// Load plugin
let plugin = DatabasePlugin::new(connection_pool);
server.load_plugin(plugin).await?;
```

### Plugin Registry

```rust
use woolly_mcp::prelude::*;

// Discover and load plugins
let registry = PluginRegistry::new();

// Load from directory
registry.load_from_directory("./plugins").await?;

// Load specific plugin
registry.load_plugin_by_name("database").await?;

// List loaded plugins
for plugin_info in registry.list_plugins() {
    println!("Plugin: {} v{}", plugin_info.name, plugin_info.version);
}
```

## Configuration

### Server Configuration

```rust
use woolly_mcp::prelude::*;
use std::time::Duration;

let config = ServerConfig {
    // Network settings
    port: 8080,
    host: "localhost".to_string(),
    
    // Protocol settings
    http_enabled: true,
    websocket_enabled: true,
    stdio_enabled: false,
    
    // CORS settings
    enable_cors: true,
    cors_origins: vec!["https://myapp.com".to_string()],
    cors_methods: vec!["GET".to_string(), "POST".to_string()],
    
    // Connection limits
    max_connections: 100,
    max_request_size: 1024 * 1024, // 1MB
    request_timeout: Duration::from_secs(30),
    
    // Monitoring
    enable_metrics: true,
    enable_logging: true,
    log_level: LogLevel::Info,
    
    // Security
    require_authentication: false,
    api_key: None,
    rate_limit: Some(RateLimit {
        requests_per_minute: 60,
        burst_size: 10,
    }),
    
    ..Default::default()
};
```

### Client Configuration

```rust
use woolly_mcp::prelude::*;
use std::time::Duration;

let client_config = ClientConfig {
    connect_timeout: Duration::from_secs(10),
    request_timeout: Duration::from_secs(30),
    max_retries: 3,
    retry_delay: Duration::from_secs(1),
    keep_alive: true,
    user_agent: "woolly-mcp-client/0.1.0".to_string(),
    headers: vec![
        ("Authorization".to_string(), "Bearer token".to_string()),
    ],
};

let client = McpClient::with_config("http://localhost:8080", client_config).await?;
```

## Authentication and Security

### API Key Authentication

```rust
use woolly_mcp::prelude::*;

let server_config = ServerConfig {
    require_authentication: true,
    api_key: Some("your-secret-api-key".to_string()),
    ..Default::default()
};

// Client with authentication
let client_config = ClientConfig {
    headers: vec![
        ("X-API-Key".to_string(), "your-secret-api-key".to_string()),
    ],
    ..Default::default()
};
```

### Custom Authentication

```rust
use woolly_mcp::prelude::*;
use async_trait::async_trait;

struct JwtAuthenticator {
    secret: String,
}

#[async_trait]
impl Authenticator for JwtAuthenticator {
    async fn authenticate(&self, request: &McpRequest) -> Result<bool, String> {
        if let Some(auth_header) = request.headers.get("Authorization") {
            if let Some(token) = auth_header.strip_prefix("Bearer ") {
                return self.validate_jwt_token(token).await;
            }
        }
        Ok(false)
    }
}

// Set custom authenticator
server.set_authenticator(JwtAuthenticator::new("secret")).await?;
```

## Monitoring and Metrics

### Built-in Metrics

```rust
use woolly_mcp::prelude::*;

// Enable metrics collection
let server_config = ServerConfig {
    enable_metrics: true,
    metrics_endpoint: "/metrics".to_string(),
    ..Default::default()
};

// Access metrics
let metrics = client.read_resource("metrics://server").await?;
println!("Requests processed: {}", metrics["requests_total"]);
println!("Average response time: {}ms", metrics["response_time_avg"]);
```

### Custom Metrics

```rust
use woolly_mcp::prelude::*;

// Custom metric collection
struct CustomMetrics {
    tool_usage: HashMap<String, u64>,
    error_counts: HashMap<String, u64>,
}

impl CustomMetrics {
    fn record_tool_usage(&mut self, tool_name: &str) {
        *self.tool_usage.entry(tool_name.to_string()).or_insert(0) += 1;
    }
    
    fn record_error(&mut self, error_type: &str) {
        *self.error_counts.entry(error_type.to_string()).or_insert(0) += 1;
    }
}

// Integrate with server
server.set_metrics_collector(CustomMetrics::new()).await?;
```

## Error Handling

```rust
use woolly_mcp::prelude::*;

match client.call_tool("text_generation", args).await {
    Ok(result) => {
        println!("Success: {}", result);
    }
    Err(McpError::ToolNotFound(name)) => {
        eprintln!("Tool '{}' not found", name);
    }
    Err(McpError::InvalidArguments { tool, message }) => {
        eprintln!("Invalid arguments for tool '{}': {}", tool, message);
    }
    Err(McpError::ToolExecutionFailed { tool, error }) => {
        eprintln!("Tool '{}' failed: {}", tool, error);
    }
    Err(McpError::ConnectionFailed(msg)) => {
        eprintln!("Connection failed: {}", msg);
    }
    Err(McpError::Timeout) => {
        eprintln!("Request timed out");
    }
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

## Testing

```bash
# Run unit tests
cargo test

# Run integration tests
cargo test --test mcp_integration

# Run with all features
cargo test --all-features

# Test with real MCP clients
cargo test --test compatibility
```

## Examples

See the [`examples/`](../../examples/) directory:

- **[MCP Integration](../../examples/mcp_integration.rs)**: Complete MCP server setup
- **[Custom Tools](examples/custom_tools.rs)**: Creating custom tool implementations
- **[Resource Management](examples/resources.rs)**: Working with MCP resources
- **[Plugin Development](examples/plugins.rs)**: Building MCP plugins

## Features

- `websocket`: WebSocket transport support
- `http`: HTTP REST API support
- `stdio`: Standard I/O transport support
- `auth`: Authentication and security features
- `metrics`: Built-in metrics and monitoring
- `plugins`: Plugin system support

## Contributing

We welcome contributions! Please see the [Contributing Guide](../../CONTRIBUTING.md) for details.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../../LICENSE-MIT))

at your option.