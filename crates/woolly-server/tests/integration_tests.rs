//! Integration tests for woolly-server
//!
//! These tests verify end-to-end functionality of the server including
//! HTTP handlers, authentication, and request/response flow.

use axum::{
    body::Body,
    http::{Request, StatusCode, Method},
    response::Response,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;
use woolly_core::engine::InferenceEngine;
use woolly_server::{
    config::{AuthConfig, ServerConfig},
    handlers::{
        auth::{CreateTokenRequest, ValidateTokenRequest},
        inference::{CompletionRequest, ChatCompletionRequest, ChatMessage},
    },
    server::{ServerState, create_router},
};

fn create_test_server_state() -> ServerState {
    let config = ServerConfig::default();
    let inference_engine = Arc::new(RwLock::new(InferenceEngine::new()));
    let auth_config = Arc::new(AuthConfig {
        jwt_secret: "test-secret-key-for-integration-tests".to_string(),
        jwt_expiration: 3600,
        api_keys: vec!["integration-test-key".to_string()],
        allow_anonymous: true,
    });

    ServerState {
        config,
        inference_engine,
        sessions: Arc::new(dashmap::DashMap::new()),
        token_manager: Arc::new(woolly_server::auth::TokenManager::new(auth_config)),
        #[cfg(feature = "mcp")]
        mcp_state: Arc::new(woolly_server::mcp::McpState::new()),
    }
}

async fn send_request(
    router: axum::Router,
    method: Method,
    uri: &str,
    body: Option<Value>,
) -> Response {
    let mut request = Request::builder()
        .method(method)
        .uri(uri);

    if let Some(json_body) = body {
        request = request.header("content-type", "application/json");
        let body_string = serde_json::to_string(&json_body).unwrap();
        request.body(Body::from(body_string)).unwrap()
    } else {
        request = request.body(Body::empty()).unwrap();
    }

    router.oneshot(request.body(Body::empty()).unwrap()).await.unwrap()
}

#[tokio::test]
async fn test_health_endpoints() {
    let state = create_test_server_state();
    let router = create_router(state);

    // Test basic health check
    let response = send_request(router.clone(), Method::GET, "/health", None).await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    assert_eq!(json["status"], "ok");
    assert_eq!(json["service"], "woolly-server");

    // Test readiness check
    let response = send_request(router.clone(), Method::GET, "/health/ready", None).await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    assert!(json["checks"].is_object());
    assert!(json["checks"]["inference_engine"].is_object());

    // Test liveness check
    let response = send_request(router, Method::GET, "/health/live", None).await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    assert_eq!(json["status"], "alive");
}

#[tokio::test]
async fn test_auth_flow() {
    let state = create_test_server_state();
    let router = create_router(state);

    // Test token creation with valid API key
    let create_request = CreateTokenRequest {
        user_id: "test-user".to_string(),
        api_key: Some("integration-test-key".to_string()),
    };

    let response = send_request(
        router.clone(),
        Method::POST,
        "/auth/token",
        Some(serde_json::to_value(create_request).unwrap()),
    ).await;
    
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    assert!(json["token"].is_string());
    assert_eq!(json["token_type"], "Bearer");
    let token = json["token"].as_str().unwrap().to_string();

    // Test token validation
    let validate_request = ValidateTokenRequest { token };

    let response = send_request(
        router.clone(),
        Method::POST,
        "/auth/validate",
        Some(serde_json::to_value(validate_request).unwrap()),
    ).await;
    
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    assert_eq!(json["valid"], true);
    assert!(json["claims"].is_object());

    // Test token creation with invalid API key
    let invalid_request = CreateTokenRequest {
        user_id: "test-user".to_string(),
        api_key: Some("invalid-key".to_string()),
    };

    let response = send_request(
        router,
        Method::POST,
        "/auth/token",
        Some(serde_json::to_value(invalid_request).unwrap()),
    ).await;
    
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_completion_endpoint() {
    let state = create_test_server_state();
    let router = create_router(state);

    let completion_request = CompletionRequest {
        prompt: "Hello, world!".to_string(),
        max_tokens: Some(50),
        temperature: Some(0.7),
        top_p: Some(0.9),
        top_k: Some(40),
        repetition_penalty: Some(1.1),
        stop_sequences: None,
        stream: Some(false),
    };

    let response = send_request(
        router,
        Method::POST,
        "/v1/completions",
        Some(serde_json::to_value(completion_request).unwrap()),
    ).await;
    
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["model"], "woolly-model");
    assert!(json["choices"].is_array());
    assert!(json["usage"].is_object());
    
    let choices = json["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 1);
    
    let choice = &choices[0];
    assert!(choice["text"].is_string());
    assert_eq!(choice["finish_reason"], "stop");
}

#[tokio::test]
async fn test_chat_completion_endpoint() {
    let state = create_test_server_state();
    let router = create_router(state);

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Hello, how are you?".to_string(),
        },
    ];

    let chat_request = ChatCompletionRequest {
        messages,
        max_tokens: Some(100),
        temperature: Some(0.8),
        top_p: None,
        top_k: None,
        repetition_penalty: None,
        stop_sequences: None,
        stream: Some(false),
    };

    let response = send_request(
        router,
        Method::POST,
        "/v1/chat/completions",
        Some(serde_json::to_value(chat_request).unwrap()),
    ).await;
    
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "woolly-model");
    assert!(json["choices"].is_array());
    
    let choices = json["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 1);
    
    let choice = &choices[0];
    assert!(choice["message"].is_object());
    assert_eq!(choice["finish_reason"], "stop");
    
    let message = &choice["message"];
    assert_eq!(message["role"], "assistant");
    assert!(message["content"].is_string());
}

#[tokio::test]
async fn test_error_handling() {
    let state = create_test_server_state();
    let router = create_router(state);

    // Test invalid JSON
    let response = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from("invalid json"))
        .unwrap();

    let response = router.clone().oneshot(response).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Test missing required fields
    let incomplete_request = json!({
        "max_tokens": 50
        // Missing required "prompt" field
    });

    let response = send_request(
        router,
        Method::POST,
        "/v1/completions",
        Some(incomplete_request),
    ).await;
    
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_cors_headers() {
    let state = create_test_server_state();
    let router = create_router(state);

    // Test OPTIONS request
    let response = send_request(router, Method::OPTIONS, "/health", None).await;
    
    // Should handle CORS properly
    assert!(response.status().is_success() || response.status() == StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn test_request_size_limits() {
    let state = create_test_server_state();
    let router = create_router(state);

    // Test with very large prompt
    let large_prompt = "x".repeat(1_000_000); // 1MB prompt
    let completion_request = CompletionRequest {
        prompt: large_prompt,
        max_tokens: Some(10),
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
        stop_sequences: None,
        stream: Some(false),
    };

    let response = send_request(
        router,
        Method::POST,
        "/v1/completions",
        Some(serde_json::to_value(completion_request).unwrap()),
    ).await;
    
    // Should either accept the request or reject with proper status
    assert!(response.status().is_success() || 
            response.status() == StatusCode::PAYLOAD_TOO_LARGE ||
            response.status() == StatusCode::BAD_REQUEST);
}