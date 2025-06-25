//! Authentication handlers

use crate::{
    auth::{extract_auth, Claims},
    error::{ServerError, ServerResult},
    server::ServerState,
};
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Token creation request
#[derive(Debug, Deserialize)]
pub struct CreateTokenRequest {
    pub user_id: String,
    pub api_key: Option<String>,
}

/// Token creation response
#[derive(Debug, Serialize)]
pub struct CreateTokenResponse {
    pub token: String,
    pub expires_in: u64,
    pub token_type: String,
}

/// Token validation request
#[derive(Debug, Deserialize)]
pub struct ValidateTokenRequest {
    pub token: String,
}

/// Token validation response
#[derive(Debug, Serialize)]
pub struct ValidateTokenResponse {
    pub valid: bool,
    pub claims: Option<Claims>,
    pub expires_at: Option<u64>,
}

/// Create a new JWT token
pub async fn create_token(
    State(state): State<ServerState>,
    Json(request): Json<CreateTokenRequest>,
) -> ServerResult<Json<CreateTokenResponse>> {
    // Validate API key if provided
    if let Some(ref api_key) = request.api_key {
        if !state.token_manager.validate_api_key(api_key) {
            return Err(ServerError::Auth("Invalid API key".to_string()));
        }
    }

    // Generate token
    let token = state.token_manager.generate_token(
        &request.user_id,
        request.api_key.clone(),
    )?;

    Ok(Json(CreateTokenResponse {
        token,
        expires_in: state.config.auth.jwt_expiration,
        token_type: "Bearer".to_string(),
    }))
}

/// Validate a JWT token
pub async fn validate_token(
    State(state): State<ServerState>,
    Json(request): Json<ValidateTokenRequest>,
) -> ServerResult<Json<ValidateTokenResponse>> {
    match state.token_manager.validate_token(&request.token) {
        Ok(claims) => Ok(Json(ValidateTokenResponse {
            valid: true,
            expires_at: Some(claims.exp),
            claims: Some(claims),
        })),
        Err(_) => Ok(Json(ValidateTokenResponse {
            valid: false,
            claims: None,
            expires_at: None,
        })),
    }
}

/// Get current user information
pub async fn get_user_info(
    request: axum::extract::Request,
) -> ServerResult<Json<Value>> {
    let auth_context = extract_auth(&request)?;

    Ok(Json(json!({
        "user_id": auth_context.user_id,
        "is_authenticated": auth_context.is_authenticated,
        "scopes": auth_context.scopes,
        "api_key_id": auth_context.api_key_id
    })))
}

/// Refresh an existing token
pub async fn refresh_token(
    State(state): State<ServerState>,
    request: axum::extract::Request,
) -> ServerResult<Json<CreateTokenResponse>> {
    let auth_context = extract_auth(&request)?;

    if !auth_context.is_authenticated {
        return Err(ServerError::Auth("Authentication required".to_string()));
    }

    // Generate new token
    let token = state.token_manager.generate_token(
        &auth_context.user_id,
        auth_context.api_key_id.clone(),
    )?;

    Ok(Json(CreateTokenResponse {
        token,
        expires_in: state.config.auth.jwt_expiration,
        token_type: "Bearer".to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::{AuthConfig, ServerConfig}, server::ServerState};
    use axum::Json;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use woolly_core::engine::InferenceEngine;

    fn create_test_state() -> ServerState {
        let config = Arc::new(ServerConfig::default());
        let engine_config = woolly_core::config::EngineConfig::default();
        let inference_engine = Arc::new(RwLock::new(InferenceEngine::new(engine_config)));
        let auth_config = Arc::new(AuthConfig {
            jwt_secret: "test-secret-key-for-jwt-tokens".to_string(),
            jwt_expiration: 3600,
            api_keys: vec!["test-api-key".to_string(), "another-valid-key".to_string()],
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
    async fn test_create_token_success() {
        let state = create_test_state();
        let request = CreateTokenRequest {
            user_id: "test-user".to_string(),
            api_key: Some("test-api-key".to_string()),
        };

        let result = create_token(axum::extract::State(state), Json(request)).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        let token_response = response.0;
        
        assert!(!token_response.token.is_empty());
        assert_eq!(token_response.expires_in, 3600);
        assert_eq!(token_response.token_type, "Bearer");
    }

    #[tokio::test]
    async fn test_create_token_invalid_api_key() {
        let state = create_test_state();
        let request = CreateTokenRequest {
            user_id: "test-user".to_string(),
            api_key: Some("invalid-api-key".to_string()),
        };

        let result = create_token(axum::extract::State(state), Json(request)).await;
        assert!(result.is_err());
        
        if let Err(ServerError::Auth(msg)) = result {
            assert_eq!(msg, "Invalid API key");
        } else {
            panic!("Expected auth error");
        }
    }

    #[tokio::test]
    async fn test_create_token_no_api_key() {
        let state = create_test_state();
        let request = CreateTokenRequest {
            user_id: "test-user".to_string(),
            api_key: None,
        };

        let result = create_token(axum::extract::State(state), Json(request)).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        let token_response = response.0;
        
        assert!(!token_response.token.is_empty());
        assert_eq!(token_response.expires_in, 3600);
        assert_eq!(token_response.token_type, "Bearer");
    }

    #[tokio::test]
    async fn test_validate_token_success() {
        let state = create_test_state();
        
        // First create a token
        let token_result = state.token_manager.generate_token("test-user", Some("test-api-key".to_string()));
        assert!(token_result.is_ok());
        let token = token_result.unwrap();
        
        // Verify token is not empty
        assert!(!token.is_empty());
        
        let request = ValidateTokenRequest { token };
        let result = validate_token(axum::extract::State(state), Json(request)).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        let validation_response = response.0;
        
        // Debug print if validation fails
        if !validation_response.valid {
            println!("Token validation failed, but token generation succeeded");
            println!("Claims: {:?}", validation_response.claims);
        }
        
        assert!(validation_response.valid, "Token should be valid");
        assert!(validation_response.claims.is_some());
        assert!(validation_response.expires_at.is_some());
        
        let claims = validation_response.claims.unwrap();
        assert_eq!(claims.sub, "test-user");
        assert_eq!(claims.api_key_id, Some("test-api-key".to_string()));
    }

    #[tokio::test]
    async fn test_validate_token_invalid() {
        let state = create_test_state();
        let request = ValidateTokenRequest {
            token: "invalid.jwt.token".to_string(),
        };

        let result = validate_token(axum::extract::State(state), Json(request)).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        let validation_response = response.0;
        
        assert!(!validation_response.valid);
        assert!(validation_response.claims.is_none());
        assert!(validation_response.expires_at.is_none());
    }
}