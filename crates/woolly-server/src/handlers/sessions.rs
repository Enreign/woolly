//! Session management handlers

use crate::{
    auth::extract_auth,
    error::{ServerError, ServerResult},
    server::ServerState,
};
use axum::{
    extract::{Path, Request, State},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use woolly_core::session::SessionConfig;

/// Session information
#[derive(Debug, Serialize)]
pub struct SessionInfo {
    pub id: String,
    pub user_id: String,
    pub created_at: String,
    pub last_accessed: String,
    pub message_count: usize,
    pub context_length: usize,
    pub max_context_length: usize,
    pub model_name: Option<String>,
    pub active: bool,
    pub metadata: Value,
}

/// Create session request
#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    pub config: Option<SessionConfig>,
    pub metadata: Option<Value>,
}

/// Create session response
#[derive(Debug, Serialize)]
pub struct CreateSessionResponse {
    pub session_id: String,
    pub message: String,
    pub session_info: SessionInfo,
}

/// List all sessions for the authenticated user
pub async fn list_sessions(
    State(_state): State<ServerState>,
    request: Request,
) -> ServerResult<Json<Vec<SessionInfo>>> {
    let auth_context = extract_auth(&request)?;

    // TODO: Implement actual session retrieval from storage
    // For now, return mock sessions
    let sessions = vec![
        SessionInfo {
            id: "session-1".to_string(),
            user_id: auth_context.user_id.clone(),
            created_at: chrono::Utc::now().to_rfc3339(),
            last_accessed: chrono::Utc::now().to_rfc3339(),
            message_count: 5,
            context_length: 150,
            max_context_length: 4096,
            model_name: Some("loaded-model".to_string()),
            active: true,
            metadata: json!({}),
        },
        SessionInfo {
            id: "session-2".to_string(),
            user_id: auth_context.user_id.clone(),
            created_at: (chrono::Utc::now() - chrono::Duration::hours(1)).to_rfc3339(),
            last_accessed: (chrono::Utc::now() - chrono::Duration::minutes(30)).to_rfc3339(),
            message_count: 12,
            context_length: 500,
            max_context_length: 4096,
            model_name: Some("loaded-model".to_string()),
            active: false,
            metadata: json!({}),
        },
    ];

    Ok(Json(sessions))
}

/// Create a new session
pub async fn create_session(
    State(state): State<ServerState>,
    Json(create_request): Json<CreateSessionRequest>,
) -> ServerResult<Json<CreateSessionResponse>> {
    // TODO: Get auth context from middleware
    let user_id = "default-user".to_string(); // Placeholder

    // Create session configuration
    let session_config = create_request.config.unwrap_or_default();

    // Create session through inference engine
    let session_result = {
        let engine = state.inference_engine.read().await;
        engine.create_session(session_config).await
    };

    match session_result {
        Ok(_session) => {
            // TODO: Store session in persistent storage
            let session_id = uuid::Uuid::new_v4().to_string();
            
            let session_info = SessionInfo {
                id: session_id.clone(),
                user_id: user_id,
                created_at: chrono::Utc::now().to_rfc3339(),
                last_accessed: chrono::Utc::now().to_rfc3339(),
                message_count: 0,
                context_length: 0,
                max_context_length: 4096, // TODO: Get from session config
                model_name: Some("loaded-model".to_string()),
                active: true,
                metadata: create_request.metadata.unwrap_or_default(),
            };

            Ok(Json(CreateSessionResponse {
                session_id: session_id.clone(),
                message: format!("Session '{}' created successfully", session_id),
                session_info,
            }))
        }
        Err(e) => Err(ServerError::Core(e)),
    }
}

/// Get information about a specific session
pub async fn get_session(
    State(_state): State<ServerState>,
    Path(session_id): Path<String>,
    request: Request,
) -> ServerResult<Json<SessionInfo>> {
    let auth_context = extract_auth(&request)?;

    // TODO: Retrieve session from storage and verify ownership
    // For now, return mock session info
    if session_id == "session-1" || session_id == "session-2" {
        let session_info = SessionInfo {
            id: session_id.clone(),
            user_id: auth_context.user_id.clone(),
            created_at: chrono::Utc::now().to_rfc3339(),
            last_accessed: chrono::Utc::now().to_rfc3339(),
            message_count: 5,
            context_length: 150,
            max_context_length: 4096,
            model_name: Some("loaded-model".to_string()),
            active: true,
            metadata: json!({}),
        };

        Ok(Json(session_info))
    } else {
        Err(ServerError::SessionNotFound(format!("Session '{}' not found", session_id)))
    }
}

/// Delete a session
pub async fn delete_session(
    State(_state): State<ServerState>,
    Path(session_id): Path<String>,
    request: Request,
) -> ServerResult<Json<Value>> {
    let _auth_context = extract_auth(&request)?;

    // TODO: Delete session from storage and verify ownership
    // For now, return success
    
    Ok(Json(json!({
        "success": true,
        "message": format!("Session '{}' deleted successfully", session_id),
        "session_id": session_id
    })))
}

/// Update session metadata
pub async fn update_session(
    State(_state): State<ServerState>,
    Path(session_id): Path<String>,
    request: Request,
    Json(metadata): Json<Value>,
) -> ServerResult<Json<Value>> {
    let _auth_context = extract_auth(&request)?;

    // TODO: Update session metadata in storage and verify ownership
    // For now, return success
    
    Ok(Json(json!({
        "success": true,
        "message": format!("Session '{}' updated successfully", session_id),
        "session_id": session_id,
        "metadata": metadata
    })))
}

/// Get session messages/history
pub async fn get_session_history(
    State(_state): State<ServerState>,
    Path(session_id): Path<String>,
    request: Request,
) -> ServerResult<Json<Value>> {
    let _auth_context = extract_auth(&request)?;

    // TODO: Retrieve session history from storage
    // For now, return mock history
    
    Ok(Json(json!({
        "session_id": session_id,
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?",
                "timestamp": chrono::Utc::now().to_rfc3339()
            },
            {
                "role": "assistant", 
                "content": "I'm doing well, thank you for asking! How can I help you today?",
                "timestamp": chrono::Utc::now().to_rfc3339()
            }
        ],
        "total_messages": 2
    })))
}

/// Clear session context
pub async fn clear_session(
    State(_state): State<ServerState>,
    Path(session_id): Path<String>,
    request: Request,
) -> ServerResult<Json<Value>> {
    let _auth_context = extract_auth(&request)?;

    // TODO: Clear session context/history and verify ownership
    // For now, return success
    
    Ok(Json(json!({
        "success": true,
        "message": format!("Session '{}' context cleared successfully", session_id),
        "session_id": session_id
    })))
}