//! WebSocket Streaming Server Example
//!
//! This example demonstrates a complete WebSocket server with:
//! - Real-time bidirectional streaming
//! - Connection management
//! - Authentication and session handling
//! - Graceful disconnection
//! - Broadcasting to multiple clients

use axum::{
    extract::{ws::{WebSocket, WebSocketUpgrade}, State, Query},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use axum_extra::headers::authorization::{Authorization, Bearer};
use axum_extra::TypedHeader;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;
use woolly_core::prelude::*;
use woolly_server::auth::{AuthToken, validate_token};

#[derive(Clone)]
struct AppState {
    /// Loaded model for inference
    engine: Arc<InferenceEngine>,
    /// Active WebSocket connections
    connections: Arc<RwLock<HashMap<Uuid, ConnectionInfo>>>,
    /// Broadcast channel for server events
    broadcast_tx: broadcast::Sender<ServerEvent>,
}

#[derive(Clone)]
struct ConnectionInfo {
    id: Uuid,
    user_id: String,
    session: Arc<InferenceSession>,
    created_at: std::time::Instant,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
enum ServerEvent {
    UserConnected { user_id: String, connection_id: Uuid },
    UserDisconnected { user_id: String, connection_id: Uuid },
    SystemMessage { message: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ClientMessage {
    /// Start text generation
    Generate {
        prompt: String,
        #[serde(default)]
        options: GenerationOptions,
    },
    /// Stop current generation
    StopGeneration,
    /// Ping for keepalive
    Ping,
    /// Request server stats
    GetStats,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ServerMessage {
    /// Authentication result
    AuthResult { success: bool, message: String },
    /// Generated token
    Token { text: String, finish_reason: Option<String> },
    /// Generation completed
    GenerationComplete { tokens_generated: usize, time_ms: u64 },
    /// Error occurred
    Error { code: String, message: String },
    /// Pong response
    Pong,
    /// Server statistics
    Stats { connections: usize, uptime_secs: u64 },
    /// Broadcast event
    Event(ServerEvent),
}

#[derive(Debug, Deserialize, Default)]
struct GenerationOptions {
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stop_sequences: Vec<String>,
}

fn default_temperature() -> f32 { 0.7 }
fn default_max_tokens() -> usize { 500 }

#[derive(Deserialize)]
struct ConnectQuery {
    /// Optional session ID to resume
    session_id: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load model
    println!("Loading model...");
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/llama-2-7b-q4_k_m.gguf".to_string());
    
    let model = Model::load(&model_path)?;
    println!("✓ Model loaded: {}", model.name());

    // Create inference engine
    let engine = InferenceEngine::new(InferenceConfig::default())?;
    engine.load_model(model).await?;

    // Create broadcast channel for events
    let (broadcast_tx, _) = broadcast::channel(100);

    // Create app state
    let state = AppState {
        engine: Arc::new(engine),
        connections: Arc::new(RwLock::new(HashMap::new())),
        broadcast_tx,
    };

    // Build router
    let app = Router::new()
        .route("/ws", get(websocket_handler))
        .route("/health", get(health_check))
        .with_state(state);

    // Start server
    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    println!("WebSocket server listening on ws://{}/ws", addr);
    println!("Example client: wscat -c ws://localhost:8080/ws -H 'Authorization: Bearer test-token'");
    
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

async fn health_check() -> impl IntoResponse {
    "OK"
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(query): Query<ConnectQuery>,
    auth_header: Option<TypedHeader<Authorization<Bearer>>>,
) -> Response {
    // Validate authentication
    let user_id = match auth_header {
        Some(TypedHeader(Authorization(bearer))) => {
            match validate_token(bearer.token()) {
                Ok(token) => token.user_id,
                Err(_) => {
                    return (axum::http::StatusCode::UNAUTHORIZED, "Invalid token").into_response();
                }
            }
        }
        None => {
            // For demo purposes, allow anonymous with generated ID
            format!("anonymous-{}", Uuid::new_v4())
        }
    };

    ws.on_upgrade(move |socket| handle_socket(socket, state, user_id, query.session_id))
}

async fn handle_socket(
    socket: WebSocket,
    state: AppState,
    user_id: String,
    session_id: Option<String>,
) {
    let connection_id = Uuid::new_v4();
    let (mut sender, mut receiver) = socket.split();
    
    // Create or resume session
    let session = match create_or_resume_session(&state, &user_id, session_id).await {
        Ok(session) => Arc::new(session),
        Err(e) => {
            let _ = sender.send(axum::extract::ws::Message::Text(
                serde_json::to_string(&ServerMessage::Error {
                    code: "SESSION_ERROR".to_string(),
                    message: e.to_string(),
                }).unwrap()
            )).await;
            return;
        }
    };

    // Register connection
    let conn_info = ConnectionInfo {
        id: connection_id,
        user_id: user_id.clone(),
        session: session.clone(),
        created_at: std::time::Instant::now(),
    };
    
    state.connections.write().await.insert(connection_id, conn_info);

    // Send authentication success
    let _ = sender.send(axum::extract::ws::Message::Text(
        serde_json::to_string(&ServerMessage::AuthResult {
            success: true,
            message: format!("Connected as {}", user_id),
        }).unwrap()
    )).await;

    // Broadcast user connected event
    let _ = state.broadcast_tx.send(ServerEvent::UserConnected {
        user_id: user_id.clone(),
        connection_id,
    });

    // Subscribe to broadcast events
    let mut broadcast_rx = state.broadcast_tx.subscribe();

    // Handle messages
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);
    
    // Task to handle incoming messages
    let user_id_clone = user_id.clone();
    let session_clone = session.clone();
    let tx_clone = tx.clone();
    let incoming_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let axum::extract::ws::Message::Text(text) = msg {
                if let Ok(client_msg) = serde_json::from_str::<ClientMessage>(&text) {
                    if let Err(e) = handle_client_message(
                        client_msg,
                        &session_clone,
                        &tx_clone,
                        &user_id_clone,
                    ).await {
                        let _ = tx_clone.send(ServerMessage::Error {
                            code: "PROCESSING_ERROR".to_string(),
                            message: e.to_string(),
                        }).await;
                    }
                }
            }
        }
    });

    // Task to send outgoing messages
    let outgoing_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                // Handle generated messages
                Some(msg) = rx.recv() => {
                    if let Ok(json) = serde_json::to_string(&msg) {
                        if sender.send(axum::extract::ws::Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                }
                // Handle broadcast events
                Ok(event) = broadcast_rx.recv() => {
                    if let Ok(json) = serde_json::to_string(&ServerMessage::Event(event)) {
                        if sender.send(axum::extract::ws::Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                }
            }
        }
    });

    // Wait for tasks to complete
    tokio::select! {
        _ = incoming_task => {},
        _ = outgoing_task => {},
    }

    // Clean up connection
    state.connections.write().await.remove(&connection_id);
    
    // Broadcast disconnection
    let _ = state.broadcast_tx.send(ServerEvent::UserDisconnected {
        user_id,
        connection_id,
    });
}

async fn handle_client_message(
    msg: ClientMessage,
    session: &InferenceSession,
    tx: &tokio::sync::mpsc::Sender<ServerMessage>,
    user_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match msg {
        ClientMessage::Generate { prompt, options } => {
            tracing::info!("User {} requested generation: {}", user_id, prompt);
            
            // Configure generation
            let config = session.config()
                .with_temperature(options.temperature)
                .with_max_tokens(options.max_tokens);
            
            if !options.stop_sequences.is_empty() {
                // Add stop sequences to config
            }
            
            // Start streaming generation
            let start_time = std::time::Instant::now();
            let mut tokens_generated = 0;
            
            let stream = session.stream_text(&prompt).await?;
            tokio::pin!(stream);

            while let Some(result) = stream.next().await {
                match result {
                    Ok(token) => {
                        tokens_generated += 1;
                        tx.send(ServerMessage::Token {
                            text: token.text,
                            finish_reason: token.finish_reason.map(|r| format!("{:?}", r)),
                        }).await?;
                    }
                    Err(e) => {
                        tx.send(ServerMessage::Error {
                            code: "GENERATION_ERROR".to_string(),
                            message: e.to_string(),
                        }).await?;
                        break;
                    }
                }
            }
            
            // Send completion message
            let elapsed = start_time.elapsed();
            tx.send(ServerMessage::GenerationComplete {
                tokens_generated,
                time_ms: elapsed.as_millis() as u64,
            }).await?;
        }
        
        ClientMessage::StopGeneration => {
            // In a real implementation, you would cancel the ongoing generation
            tracing::info!("User {} requested stop generation", user_id);
            session.cancel_generation().await?;
        }
        
        ClientMessage::Ping => {
            tx.send(ServerMessage::Pong).await?;
        }
        
        ClientMessage::GetStats => {
            let connections = {
                let conns = session.engine().stats().active_sessions;
                conns
            };
            
            tx.send(ServerMessage::Stats {
                connections,
                uptime_secs: 0, // Would track actual uptime
            }).await?;
        }
    }
    
    Ok(())
}

async fn create_or_resume_session(
    state: &AppState,
    user_id: &str,
    session_id: Option<String>,
) -> Result<InferenceSession, CoreError> {
    // In a real implementation, you would:
    // 1. Check if session_id exists in a persistent store
    // 2. Resume the session with its context
    // 3. Or create a new session
    
    let config = SessionConfig::default()
        .max_tokens(1000)
        .temperature(0.7);
    
    let session = state.engine.create_session(config).await?;
    
    tracing::info!("Created new session for user {}", user_id);
    
    Ok(session)
}

// Example WebSocket client code (for testing)
#[cfg(test)]
mod test_client {
    use super::*;
    use tokio_tungstenite::{connect_async, tungstenite::Message};
    use futures_util::{SinkExt, StreamExt};

    #[tokio::test]
    async fn test_websocket_connection() {
        let url = "ws://localhost:8080/ws";
        
        // Connect with auth header
        let request = http::Request::builder()
            .uri(url)
            .header("Authorization", "Bearer test-token")
            .body(())
            .unwrap();
            
        let (ws_stream, _) = connect_async(request).await.expect("Failed to connect");
        let (mut write, mut read) = ws_stream.split();
        
        // Send generation request
        let generate_msg = ClientMessage::Generate {
            prompt: "Hello, world!".to_string(),
            options: GenerationOptions::default(),
        };
        
        write.send(Message::Text(serde_json::to_string(&generate_msg).unwrap())).await.unwrap();
        
        // Read responses
        while let Some(msg) = read.next().await {
            if let Ok(Message::Text(text)) = msg {
                let server_msg: ServerMessage = serde_json::from_str(&text).unwrap();
                match server_msg {
                    ServerMessage::Token { text, .. } => print!("{}", text),
                    ServerMessage::GenerationComplete { tokens_generated, time_ms } => {
                        println!("\nGenerated {} tokens in {}ms", tokens_generated, time_ms);
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
}