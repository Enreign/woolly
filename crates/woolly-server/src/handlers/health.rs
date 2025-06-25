//! Health check handlers

use crate::{error::ServerResult, server::ServerState};
use axum::{extract::State, Json};
use serde_json::{json, Value};

/// Basic health check
pub async fn health_check() -> ServerResult<Json<Value>> {
    Ok(Json(json!({
        "status": "ok",
        "service": "woolly-server",
        "version": crate::VERSION,
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}

/// Readiness check - checks if server is ready to serve requests
pub async fn readiness_check(State(state): State<ServerState>) -> ServerResult<Json<Value>> {
    let mut ready = true;
    let mut checks = serde_json::Map::new();

    // Check if inference engine is available
    match state.inference_engine.try_read() {
        Ok(_) => {
            checks.insert("inference_engine".to_string(), json!({
                "status": "ok",
                "message": "Inference engine accessible"
            }));
        }
        Err(_) => {
            ready = false;
            checks.insert("inference_engine".to_string(), json!({
                "status": "error", 
                "message": "Inference engine not accessible"
            }));
        }
    }

    // Check MCP state if enabled
    #[cfg(feature = "mcp")]
    {
        if state.mcp_state.is_initialized() {
            checks.insert("mcp".to_string(), json!({
                "status": "ok",
                "message": "MCP server initialized"  
            }));
        } else {
            ready = false;
            checks.insert("mcp".to_string(), json!({
                "status": "error",
                "message": "MCP server not initialized"
            }));
        }
    }

    let status = if ready { "ready" } else { "not_ready" };

    Ok(Json(json!({
        "status": status,
        "service": "woolly-server",
        "version": crate::VERSION,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "checks": checks
    })))
}

/// Liveness check - checks if server is alive
pub async fn liveness_check(State(state): State<ServerState>) -> ServerResult<Json<Value>> {
    // Simple liveness check - just verify we can access basic state
    let config_available = !state.config.bind.to_string().is_empty();
    
    let status = if config_available { "alive" } else { "dead" };

    Ok(Json(json!({
        "status": status,
        "service": "woolly-server", 
        "version": crate::VERSION,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "uptime_seconds": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::ServerConfig, server::ServerState};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use woolly_core::engine::InferenceEngine;

    fn create_test_state() -> ServerState {
        let config = Arc::new(ServerConfig::default());
        let engine_config = woolly_core::config::EngineConfig::default();
        let inference_engine = Arc::new(RwLock::new(InferenceEngine::new(engine_config)));
        let auth_config = Arc::new(crate::config::AuthConfig {
            jwt_secret: "test-secret".to_string(),
            jwt_expiration: 3600,
            api_keys: vec!["test-key".to_string()],
            allow_anonymous: false,
        });
        
        ServerState {
            config: Arc::clone(&config),
            inference_engine,
            token_manager: Arc::new(crate::auth::TokenManager::new(auth_config)),
            rate_limiter: Arc::new(crate::middleware::RateLimiterState::new(&config.rate_limit)),
            concurrency_limiter: Arc::new(crate::middleware::ConcurrencyLimiter::new(10)),
            #[cfg(feature = "mcp")]
            mcp_state: Arc::new(crate::mcp::McpServerState::new(&config.mcp).unwrap()),
        }
    }

    #[tokio::test]
    async fn test_health_check() {
        let result = health_check().await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        let json_value = response.0;
        
        assert_eq!(json_value["status"], "ok");
        assert_eq!(json_value["service"], "woolly-server");
        assert_eq!(json_value["version"], crate::VERSION);
        assert!(json_value["timestamp"].is_string());
    }

    #[tokio::test]
    async fn test_readiness_check() {
        let state = create_test_state();
        let result = readiness_check(State(state)).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        let json_value = response.0;
        
        assert_eq!(json_value["service"], "woolly-server");
        assert_eq!(json_value["version"], crate::VERSION);
        assert!(json_value["checks"].is_object());
        assert!(json_value["checks"]["inference_engine"].is_object());
    }

    #[tokio::test]
    async fn test_liveness_check() {
        let state = create_test_state();
        let result = liveness_check(State(state)).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        let json_value = response.0;
        
        assert_eq!(json_value["status"], "alive");
        assert_eq!(json_value["service"], "woolly-server");
        assert_eq!(json_value["version"], crate::VERSION);
        assert!(json_value["uptime_seconds"].is_number());
    }
}