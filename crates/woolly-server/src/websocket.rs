//! WebSocket handling for MCP communication

use crate::{
    auth::AuthContext,
    error::{ServerError, ServerResult},
    server::ServerState,
};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Request, State,
    },
    response::Response,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde_json::{from_str, to_string};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[cfg(feature = "mcp")]
use woolly_mcp::types::{McpMessage, McpError};

/// WebSocket connection state
#[derive(Debug, Clone)]
pub struct WebSocketConnection {
    pub id: String,
    pub user_id: String,
    pub is_authenticated: bool,
    pub connected_at: chrono::DateTime<chrono::Utc>,
}

/// WebSocket handler - upgrades HTTP to WebSocket
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<ServerState>,
    request: Request,
) -> Result<Response, ServerError> {
    // Extract authentication context
    let auth_context = request
        .extensions()
        .get::<AuthContext>()
        .ok_or_else(|| ServerError::Internal("Auth context not found".to_string()))?
        .clone();

    info!(
        user_id = %auth_context.user_id,
        is_authenticated = %auth_context.is_authenticated,
        "WebSocket connection request"
    );

    Ok(ws.on_upgrade(move |socket| handle_websocket(socket, state, auth_context)))
}

/// Handle WebSocket connection
async fn handle_websocket(socket: WebSocket, state: ServerState, auth_context: AuthContext) {
    let connection = WebSocketConnection {
        id: Uuid::new_v4().to_string(),
        user_id: auth_context.user_id.clone(),
        is_authenticated: auth_context.is_authenticated,
        connected_at: chrono::Utc::now(),
    };

    info!(
        connection_id = %connection.id,
        user_id = %connection.user_id,
        "WebSocket connection established"
    );

    let (mut sender, mut receiver) = socket.split();

    // Send welcome message
    let welcome_message = create_welcome_message(&connection, &state).await;
    if let Ok(welcome_json) = to_string(&welcome_message) {
        if let Err(e) = sender.send(Message::Text(welcome_json)).await {
            error!("Failed to send welcome message: {}", e);
            return;
        }
    }

    // Handle incoming messages
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                debug!(
                    connection_id = %connection.id,
                    message_length = text.len(),
                    "Received text message"
                );

                match handle_text_message(&text, &state, &connection).await {
                    Ok(Some(response)) => {
                        if let Ok(response_json) = to_string(&response) {
                            if let Err(e) = sender.send(Message::Text(response_json)).await {
                                error!("Failed to send response: {}", e);
                                break;
                            }
                        }
                    }
                    Ok(None) => {
                        // No response needed (e.g., notification)
                    }
                    Err(e) => {
                        error!("Error handling message: {}", e);
                        let error_response = create_error_response(e);
                        if let Ok(error_json) = to_string(&error_response) {
                            let _ = sender.send(Message::Text(error_json)).await;
                        }
                    }
                }
            }
            Ok(Message::Binary(data)) => {
                debug!(
                    connection_id = %connection.id,
                    data_length = data.len(),
                    "Received binary message (not supported)"
                );
                
                let error_response = McpError {
                    code: -32600,
                    message: "Binary messages not supported".to_string(),
                    data: None,
                };
                
                if let Ok(error_json) = to_string(&error_response) {
                    let _ = sender.send(Message::Text(error_json)).await;
                }
            }
            Ok(Message::Ping(data)) => {
                debug!(connection_id = %connection.id, "Received ping");
                if let Err(e) = sender.send(Message::Pong(data)).await {
                    error!("Failed to send pong: {}", e);
                    break;
                }
            }
            Ok(Message::Pong(_)) => {
                debug!(connection_id = %connection.id, "Received pong");
            }
            Ok(Message::Close(close_frame)) => {
                info!(
                    connection_id = %connection.id,
                    close_code = ?close_frame.as_ref().map(|f| f.code),
                    close_reason = ?close_frame.as_ref().map(|f| &f.reason),
                    "WebSocket connection closed by client"
                );
                break;
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
        }
    }

    info!(
        connection_id = %connection.id,
        user_id = %connection.user_id,
        duration = ?(chrono::Utc::now() - connection.connected_at),
        "WebSocket connection closed"
    );
}

/// Create welcome message for new connections
async fn create_welcome_message(
    connection: &WebSocketConnection,
    state: &ServerState,
) -> serde_json::Value {
    serde_json::json!({
        "type": "welcome",
        "connection_id": connection.id,
        "server_info": {
            "name": "woolly-server",
            "version": crate::VERSION,
            "protocol_version": "0.1.0"
        },
        "capabilities": get_server_capabilities(state).await,
        "timestamp": chrono::Utc::now().to_rfc3339()
    })
}

/// Get server capabilities
async fn get_server_capabilities(state: &ServerState) -> serde_json::Value {
    let mut capabilities = serde_json::Map::new();
    
    // Basic capabilities
    capabilities.insert("inference".to_string(), serde_json::json!({
        "completion": true,
        "chat": true,
        "streaming": true
    }));
    
    capabilities.insert("models".to_string(), serde_json::json!({
        "load": true,
        "unload": true,
        "list": true,
        "info": true
    }));
    
    capabilities.insert("sessions".to_string(), serde_json::json!({
        "create": true,
        "delete": true,
        "list": true,
        "persistent": false
    }));

    // MCP capabilities
    #[cfg(feature = "mcp")]
    {
        if let Ok(mcp_caps) = state.mcp_state.get_capabilities().await {
            capabilities.insert("mcp".to_string(), serde_json::to_value(mcp_caps).unwrap_or_default());
        }
    }

    serde_json::Value::Object(capabilities)
}

/// Handle text message (assumed to be JSON)
async fn handle_text_message(
    text: &str,
    state: &ServerState,
    connection: &WebSocketConnection,
) -> ServerResult<Option<serde_json::Value>> {
    // Try to parse as MCP message first
    #[cfg(feature = "mcp")]
    {
        if let Ok(mcp_message) = from_str::<McpMessage>(text) {
            return handle_mcp_message(mcp_message, state, connection).await;
        }
    }

    // Try to parse as generic JSON
    match from_str::<serde_json::Value>(text) {
        Ok(json_value) => {
            // Handle non-MCP JSON messages
            handle_json_message(json_value, state, connection).await
        }
        Err(e) => {
            warn!(
                connection_id = %connection.id,
                error = %e,
                "Failed to parse message as JSON"
            );
            Err(ServerError::InvalidRequest(format!("Invalid JSON: {}", e)))
        }
    }
}

/// Handle MCP message
#[cfg(feature = "mcp")]
async fn handle_mcp_message(
    message: McpMessage,
    state: &ServerState,
    connection: &WebSocketConnection,
) -> ServerResult<Option<serde_json::Value>> {
    debug!(
        connection_id = %connection.id,
        message_type = ?std::mem::discriminant(&message),
        "Handling MCP message"
    );

    match state.mcp_state.handle_message(message, connection).await {
        Ok(Some(response)) => Ok(Some(serde_json::to_value(response)?)),
        Ok(None) => Ok(None),
        Err(e) => Err(ServerError::from(e)),
    }
}

/// Handle generic JSON message
async fn handle_json_message(
    _json: serde_json::Value,
    _state: &ServerState,
    connection: &WebSocketConnection,
) -> ServerResult<Option<serde_json::Value>> {
    // For now, just echo back that we received it
    debug!(
        connection_id = %connection.id,
        "Handling generic JSON message"
    );

    Ok(Some(serde_json::json!({
        "type": "ack",
        "message": "Message received but not processed",
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}

/// Create error response
fn create_error_response(error: ServerError) -> serde_json::Value {
    serde_json::json!({
        "type": "error",
        "error": {
            "message": error.to_string(),
            "code": match error {
                ServerError::Auth(_) => 401,
                ServerError::Authorization(_) => 403,
                ServerError::InvalidRequest(_) => 400,
                ServerError::ModelNotFound(_) => 404,
                ServerError::SessionNotFound(_) => 404,
                _ => 500,
            }
        },
        "timestamp": chrono::Utc::now().to_rfc3339()
    })
}