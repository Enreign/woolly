//! Error types for the Woolly server

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

/// Server error types
#[derive(Error, Debug)]
pub enum ServerError {
    #[error("Core engine error: {0}")]
    Core(woolly_core::CoreError),

    #[cfg(feature = "mcp")]
    #[error("MCP protocol error: {0}")]
    Mcp(woolly_mcp::types::McpError),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Authorization error: {0}")]
    Authorization(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("WebSocket error: {0}")]
    WebSocket(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] hyper::Error),

    #[error("Internal server error: {0}")]
    Internal(String),
}

/// Result type for server operations
pub type ServerResult<T> = Result<T, ServerError>;

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, error_type, suggestion, details) = match &self {
            ServerError::Auth(msg) => (
                StatusCode::UNAUTHORIZED, 
                "authentication_failed",
                "Check your API key or authentication credentials",
                Some(json!({
                    "auth_method": "api_key",
                    "message": msg
                }))
            ),
            ServerError::Authorization(msg) => (
                StatusCode::FORBIDDEN, 
                "access_denied",
                "Ensure you have the required permissions for this operation",
                Some(json!({
                    "required_permission": "unknown",
                    "message": msg
                }))
            ),
            ServerError::RateLimit(msg) => (
                StatusCode::TOO_MANY_REQUESTS, 
                "rate_limit_exceeded",
                "Wait before making more requests or upgrade your plan",
                Some(json!({
                    "retry_after": "60",
                    "limit_type": "requests_per_minute",
                    "message": msg
                }))
            ),
            ServerError::InvalidRequest(msg) => (
                StatusCode::BAD_REQUEST, 
                "invalid_request",
                "Check the request format and required parameters",
                Some(json!({
                    "message": msg
                }))
            ),
            ServerError::ModelNotFound(msg) => (
                StatusCode::NOT_FOUND, 
                "model_not_found",
                "Check the model name and ensure it's properly loaded",
                Some(json!({
                    "available_models": [],
                    "message": msg
                }))
            ),
            ServerError::SessionNotFound(msg) => (
                StatusCode::NOT_FOUND, 
                "session_not_found",
                "Create a new session or check the session ID",
                Some(json!({
                    "session_timeout": "30 minutes",
                    "message": msg
                }))
            ),
            ServerError::WebSocket(msg) => (
                StatusCode::BAD_REQUEST, 
                "websocket_error",
                "Check WebSocket connection and message format",
                Some(json!({
                    "connection_state": "unknown",
                    "message": msg
                }))
            ),
            ServerError::Config(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR, 
                "configuration_error",
                "Contact system administrator - server configuration issue",
                Some(json!({
                    "component": "server_config",
                    "message": msg
                }))
            ),
            ServerError::Core(core_err) => {
                Self::map_core_error(core_err)
            },
            #[cfg(feature = "mcp")]
            ServerError::Mcp(mcp_err) => (
                StatusCode::BAD_GATEWAY, 
                "mcp_protocol_error",
                "Check MCP client configuration and protocol compliance",
                Some(json!({
                    "protocol": "mcp",
                    "message": mcp_err.to_string()
                }))
            ),
            ServerError::Io(err) => (
                StatusCode::INTERNAL_SERVER_ERROR, 
                "io_error",
                "Check file system permissions and disk space",
                Some(json!({
                    "io_operation": "unknown",
                    "message": err.to_string()
                }))
            ),
            ServerError::Json(err) => (
                StatusCode::BAD_REQUEST, 
                "json_parse_error",
                "Check JSON syntax and format",
                Some(json!({
                    "parser": "serde_json",
                    "message": err.to_string()
                }))
            ),
            ServerError::Http(err) => (
                StatusCode::BAD_GATEWAY, 
                "http_error",
                "Check network connectivity and upstream services",
                Some(json!({
                    "protocol": "http",
                    "message": err.to_string()
                }))
            ),
            ServerError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR, 
                "internal_server_error",
                "Contact support if this error persists",
                Some(json!({
                    "category": "unknown",
                    "message": msg
                }))
            ),
        };

        let mut error_response = json!({
            "error": {
                "type": error_type,
                "message": self.to_string(),
                "code": status.as_u16(),
                "suggestion": suggestion,
                "timestamp": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                "request_id": generate_request_id()
            }
        });

        if let Some(details) = details {
            error_response["error"]["details"] = details;
        }

        let body = Json(error_response);
        (status, body).into_response()
    }
}

impl ServerError {
    /// Map core errors to appropriate HTTP responses with context
    fn map_core_error(core_err: &woolly_core::CoreError) -> (StatusCode, &'static str, &'static str, Option<serde_json::Value>) {
        let error_code = core_err.code();
        
        match error_code {
            // Model errors
            code if code.starts_with("MODEL_") => (
                StatusCode::BAD_REQUEST,
                "model_error", 
                "Check model file path and format",
                Some(json!({
                    "error_code": code,
                    "component": "model_loading"
                }))
            ),
            
            // Input validation errors
            code if code.starts_with("INVALID_") => (
                StatusCode::BAD_REQUEST,
                "validation_error",
                "Check input parameters and values",
                Some(json!({
                    "error_code": code,
                    "component": "input_validation"
                }))
            ),
            
            // Resource errors
            code if code.starts_with("INSUFFICIENT_") => (
                StatusCode::SERVICE_UNAVAILABLE,
                "resource_unavailable",
                "Free up system resources or try again later",
                Some(json!({
                    "error_code": code,
                    "component": "resource_management"
                }))
            ),
            
            // Configuration errors
            code if code.starts_with("CONFIG_") => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "configuration_error",
                "Contact system administrator - configuration issue",
                Some(json!({
                    "error_code": code,
                    "component": "configuration"
                }))
            ),
            
            // Device errors
            code if code.starts_with("DEVICE_") => (
                StatusCode::BAD_REQUEST,
                "device_error",
                "Check device availability and configuration",
                Some(json!({
                    "error_code": code,
                    "component": "device_management"
                }))
            ),
            
            // Generation errors
            code if code.starts_with("GENERATION_") => (
                StatusCode::BAD_REQUEST,
                "generation_error",
                "Check generation parameters and model state",
                Some(json!({
                    "error_code": code,
                    "component": "text_generation"
                }))
            ),
            
            // Context errors
            code if code.starts_with("CONTEXT_") => (
                StatusCode::BAD_REQUEST,
                "context_error",
                "Reduce input length or increase context window",
                Some(json!({
                    "error_code": code,
                    "component": "context_management"
                }))
            ),
            
            // Tokenizer errors
            code if code.starts_with("TOKENIZER_") => (
                StatusCode::BAD_REQUEST,
                "tokenizer_error",
                "Check tokenizer configuration and input text",
                Some(json!({
                    "error_code": code,
                    "component": "tokenization"
                }))
            ),
            
            // IO errors
            code if code.starts_with("IO_") => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "io_error",
                "Check file permissions and disk space",
                Some(json!({
                    "error_code": code,
                    "component": "file_system"
                }))
            ),
            
            // Cache errors
            code if code.starts_with("CACHE_") => (
                StatusCode::SERVICE_UNAVAILABLE,
                "cache_error",
                "Clear cache or restart the service",
                Some(json!({
                    "error_code": code,
                    "component": "cache_management"
                }))
            ),
            
            // Tensor errors
            code if code.starts_with("TENSOR_") => (
                StatusCode::BAD_REQUEST,
                "tensor_error",
                "Check tensor dimensions and operations",
                Some(json!({
                    "error_code": code,
                    "component": "tensor_operations"
                }))
            ),
            
            // Default case
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "core_error",
                "An error occurred in the core engine",
                Some(json!({
                    "error_code": code,
                    "component": "core_engine"
                }))
            ),
        }
    }
}

/// Generate a unique request ID for error tracking
fn generate_request_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    format!("req_{:x}", timestamp)
}

/// Convert core errors to appropriate HTTP status codes
impl From<woolly_core::CoreError> for ServerError {
    fn from(err: woolly_core::CoreError) -> Self {
        match err {
            woolly_core::CoreError::InvalidInput(_) => {
                ServerError::InvalidRequest(err.to_string())
            }
            woolly_core::CoreError::Model(msg) if msg.contains("not found") => {
                ServerError::ModelNotFound(msg)
            }
            _ => ServerError::Core(err),
        }
    }
}

#[cfg(feature = "mcp")]
impl From<woolly_mcp::types::McpError> for ServerError {
    fn from(err: woolly_mcp::types::McpError) -> Self {
        ServerError::Mcp(err)
    }
}