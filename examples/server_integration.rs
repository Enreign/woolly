//! HTTP Server Integration Example
//!
//! This example demonstrates how to integrate Woolly into a web service
//! for serving LLM inference over HTTP/WebSocket.
//!
//! Features:
//! - RESTful API endpoints
//! - WebSocket streaming
//! - Request queuing
//! - Health checks
//! - Metrics endpoint
//!
//! Usage:
//!   cargo run --example server_integration -- --model path/to/model.gguf --port 3000

use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use axum::{
    extract::{State, WebSocketUpgrade, Json, Query},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use tracing::{info, error};
use woolly_core::prelude::*;
use woolly_gguf::GGUFLoader;

#[derive(Clone)]
struct AppState {
    engine: Arc<RwLock<InferenceEngine>>,
    model_name: String,
    start_time: Instant,
    request_count: Arc<RwLock<u64>>,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct CompletionResponse {
    id: String,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
    created: u64,
}

#[derive(Debug, Serialize)]
struct Choice {
    text: String,
    index: usize,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    model: String,
    uptime_seconds: u64,
    requests_served: u64,
}

#[derive(Debug, Serialize)]
struct MetricsResponse {
    requests_per_second: f64,
    average_latency_ms: f64,
    memory_usage_mb: f64,
    active_sessions: usize,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    messages: Vec<Message>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,woolly=debug")
        .init();
    
    let args = parse_args()?;
    
    println!("🌐 Woolly Server Integration Example");
    println!("═══════════════════════════════════");
    
    // Load model
    info!("Loading model from {}", args.model_path.display());
    let loader = GGUFLoader::from_path(&args.model_path)?;
    let model = create_model_from_gguf(&loader)?;
    let model_name = loader.model_name().unwrap_or("woolly").to_string();
    
    // Create inference engine
    let config = EngineConfig {
        num_threads: args.threads,
        ..Default::default()
    };
    
    let mut engine = InferenceEngine::new(config);
    engine.load_model(Arc::new(model)).await?;
    
    // Create shared state
    let state = AppState {
        engine: Arc::new(RwLock::new(engine)),
        model_name: model_name.clone(),
        start_time: Instant::now(),
        request_count: Arc::new(RwLock::new(0)),
    };
    
    // Build router
    let app = Router::new()
        // Inference endpoints
        .route("/v1/completions", post(completions_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        
        // Streaming endpoint
        .route("/v1/stream", get(websocket_handler))
        
        // Utility endpoints
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/models", get(models_handler))
        
        // Root endpoint
        .route("/", get(root_handler))
        
        .with_state(state);
    
    // Start server
    let addr = format!("{}:{}", args.host, args.port);
    println!("\n✅ Server starting on http://{}", addr);
    println!("📊 Model: {}", model_name);
    println!("🧵 Threads: {}", args.threads);
    println!("\nEndpoints:");
    println!("  POST   /v1/completions      - Text completion");
    println!("  POST   /v1/chat/completions - Chat completion");
    println!("  GET    /v1/stream          - WebSocket streaming");
    println!("  GET    /health             - Health check");
    println!("  GET    /metrics            - Performance metrics");
    println!("  GET    /models             - List models");
    
    axum::Server::bind(&addr.parse()?)
        .serve(app.into_make_service())
        .await?;
    
    Ok(())
}

async fn completions_handler(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    // Increment request counter
    {
        let mut count = state.request_count.write().await;
        *count += 1;
    }
    
    let start = Instant::now();
    
    // Create session
    let engine = state.engine.read().await;
    let session_config = SessionConfig {
        max_seq_length: request.max_tokens.unwrap_or(100),
        temperature: request.temperature.unwrap_or(0.8),
        top_p: request.top_p.unwrap_or(0.9),
        ..Default::default()
    };
    
    let session = engine.create_session(session_config).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Tokenize prompt
    let tokens = simple_tokenize(&request.prompt);
    let prompt_tokens = tokens.len();
    
    // Run inference
    let result = session.infer(&tokens).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let completion_tokens = result.tokens.len();
    let generated_text = decode_tokens(&result.tokens);
    
    let response = CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        model: state.model_name.clone(),
        choices: vec![Choice {
            text: generated_text,
            index: 0,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    let duration = start.elapsed();
    info!("Completion request processed in {:?}", duration);
    
    Ok(Json(response))
}

async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    // Convert chat messages to prompt
    let prompt = convert_messages_to_prompt(&request.messages);
    
    // Reuse completions handler logic
    let completion_request = CompletionRequest {
        prompt,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: None,
        stream: None,
    };
    
    completions_handler(State(state), Json(completion_request)).await
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_websocket(socket, state))
}

async fn handle_websocket(
    socket: axum::extract::ws::WebSocket,
    state: AppState,
) {
    use axum::extract::ws::{Message, WebSocket};
    use futures::{sink::SinkExt, stream::StreamExt};
    
    let (mut sender, mut receiver) = socket.split();
    
    // Handle incoming messages
    while let Some(msg) = receiver.next().await {
        if let Ok(msg) = msg {
            match msg {
                Message::Text(text) => {
                    // Parse request
                    if let Ok(request) = serde_json::from_str::<CompletionRequest>(&text) {
                        // Stream tokens
                        let engine = state.engine.read().await;
                        let session_config = SessionConfig {
                            max_seq_length: request.max_tokens.unwrap_or(100),
                            temperature: request.temperature.unwrap_or(0.8),
                            ..Default::default()
                        };
                        
                        if let Ok(session) = engine.create_session(session_config).await {
                            let tokens = simple_tokenize(&request.prompt);
                            
                            // Simulate streaming (in real implementation, use actual streaming)
                            for i in 0..10 {
                                let chunk = format!("Token {}", i);
                                let _ = sender.send(Message::Text(chunk)).await;
                                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                            }
                        }
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    }
}

async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    let uptime = state.start_time.elapsed().as_secs();
    let requests = *state.request_count.read().await;
    
    Json(HealthResponse {
        status: "healthy".to_string(),
        model: state.model_name.clone(),
        uptime_seconds: uptime,
        requests_served: requests,
    })
}

async fn metrics_handler(State(state): State<AppState>) -> Json<MetricsResponse> {
    let uptime = state.start_time.elapsed().as_secs_f64();
    let requests = *state.request_count.read().await as f64;
    
    Json(MetricsResponse {
        requests_per_second: requests / uptime,
        average_latency_ms: 45.0, // Placeholder
        memory_usage_mb: get_memory_usage() as f64 / 1024.0 / 1024.0,
        active_sessions: 1, // Placeholder
    })
}

async fn models_handler(State(state): State<AppState>) -> Json<Vec<ModelInfo>> {
    Json(vec![ModelInfo {
        id: state.model_name.clone(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "woolly".to_string(),
    }])
}

async fn root_handler() -> &'static str {
    "Woolly Inference Server - Visit /health for status"
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

// Helper functions

fn convert_messages_to_prompt(messages: &[Message]) -> String {
    messages.iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn simple_tokenize(text: &str) -> Vec<u32> {
    text.split_whitespace()
        .enumerate()
        .map(|(i, _)| (i as u32) % 1000 + 1)
        .collect()
}

fn decode_tokens(tokens: &[u32]) -> String {
    // Simplified decoding
    tokens.iter()
        .map(|_| "word ")
        .collect::<Vec<_>>()
        .join("")
}

fn get_memory_usage() -> usize {
    // Simplified memory usage
    2 * 1024 * 1024 * 1024 // 2GB placeholder
}

// Argument parsing

#[derive(Debug)]
struct Args {
    model_path: PathBuf,
    port: u16,
    host: String,
    threads: usize,
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        model_path: PathBuf::new(),
        port: 3000,
        host: "127.0.0.1".to_string(),
        threads: num_cpus::get(),
    };
    
    let cmd_args: Vec<String> = env::args().collect();
    let mut i = 1;
    
    while i < cmd_args.len() {
        match cmd_args[i].as_str() {
            "--model" => {
                if i + 1 < cmd_args.len() {
                    args.model_path = PathBuf::from(&cmd_args[i + 1]);
                    i += 2;
                } else {
                    return Err("Missing model path".into());
                }
            }
            "--port" => {
                if i + 1 < cmd_args.len() {
                    args.port = cmd_args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing port".into());
                }
            }
            "--host" => {
                if i + 1 < cmd_args.len() {
                    args.host = cmd_args[i + 1].clone();
                    i += 2;
                } else {
                    return Err("Missing host".into());
                }
            }
            "--threads" => {
                if i + 1 < cmd_args.len() {
                    args.threads = cmd_args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing threads".into());
                }
            }
            "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => i += 1,
        }
    }
    
    if args.model_path.as_os_str().is_empty() {
        return Err("Model path is required. Use --model <path>".into());
    }
    
    Ok(args)
}

fn print_help() {
    println!("🌐 Woolly HTTP Server Integration

USAGE:
    server_integration --model <MODEL> [OPTIONS]

REQUIRED:
    --model <MODEL>     Path to the GGUF model file

OPTIONS:
    --port <PORT>       Server port [default: 3000]
    --host <HOST>       Server host [default: 127.0.0.1]
    --threads <NUM>     Number of threads [default: CPU cores]
    --help              Show this help message

EXAMPLES:
    # Start server on default port
    server_integration --model llama-7b.gguf

    # Custom configuration
    server_integration --model llama-7b.gguf --port 8080 --host 0.0.0.0

API ENDPOINTS:
    POST /v1/completions
        Body: {
            \"prompt\": \"Hello\",
            \"max_tokens\": 100,
            \"temperature\": 0.8
        }

    GET /v1/stream (WebSocket)
        Send: {\"prompt\": \"Hello\", \"stream\": true}
        Receive: Streaming tokens

    GET /health
        Returns server health status

    GET /metrics
        Returns performance metrics
");
}

// Mock UUID module for the example
mod uuid {
    pub struct Uuid;
    impl Uuid {
        pub fn new_v4() -> String {
            format!("{:x}", rand::random::<u32>())
        }
    }
}

mod rand {
    pub fn random<T>() -> T
    where
        T: Default,
    {
        T::default()
    }
}