//! Middleware components for the server

use crate::{
    auth::AuthContext,
    config::{RateLimitConfig, RequestLimits},
    error::ServerError,
};
use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use governor::{
    clock::DefaultClock,
    middleware::NoOpMiddleware,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use std::{
    num::NonZeroU32,
    sync::Arc,
    time::Duration,
};
use tracing::{info, warn};

/// Rate limiter state
#[derive(Clone)]
pub struct RateLimiterState {
    pub authenticated_limiter: Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock, NoOpMiddleware>>,
    pub anonymous_limiter: Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock, NoOpMiddleware>>,
}

impl RateLimiterState {
    pub fn new(config: &RateLimitConfig) -> Self {
        let authenticated_quota = Quota::per_minute(
            NonZeroU32::new(config.authenticated_rpm).unwrap()
        ).allow_burst(NonZeroU32::new(config.burst_capacity).unwrap());
        
        let anonymous_quota = Quota::per_minute(
            NonZeroU32::new(config.anonymous_rpm).unwrap()
        ).allow_burst(NonZeroU32::new(config.burst_capacity / 2).unwrap());

        Self {
            authenticated_limiter: Arc::new(RateLimiter::direct(authenticated_quota)),
            anonymous_limiter: Arc::new(RateLimiter::direct(anonymous_quota)),
        }
    }
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(rate_limiter): State<Arc<RateLimiterState>>,
    request: Request,
    next: Next,
) -> Result<Response, ServerError> {
    let auth_context = request
        .extensions()
        .get::<AuthContext>()
        .ok_or_else(|| ServerError::Internal("Auth context not found".to_string()))?;

    let limiter = if auth_context.is_authenticated {
        &rate_limiter.authenticated_limiter
    } else {
        &rate_limiter.anonymous_limiter
    };

    match limiter.check() {
        Ok(_) => Ok(next.run(request).await),
        Err(_) => {
            warn!(
                user_id = %auth_context.user_id,
                "Rate limit exceeded"
            );
            Err(ServerError::RateLimit(
                "Too many requests, please try again later".to_string(),
            ))
        }
    }
}

/// Request timeout middleware
pub async fn timeout_middleware(
    State(limits): State<Arc<RequestLimits>>,
    request: Request,
    next: Next,
) -> Result<Response, ServerError> {
    let timeout_duration = Duration::from_secs(limits.request_timeout);
    
    match tokio::time::timeout(timeout_duration, next.run(request)).await {
        Ok(response) => Ok(response),
        Err(_) => Err(ServerError::Internal("Request timeout".to_string())),
    }
}

/// Request size limiting middleware
pub async fn request_size_middleware(
    State(limits): State<Arc<RequestLimits>>,
    request: Request,
    next: Next,
) -> Result<Response, ServerError> {
    // Check content-length header if present
    if let Some(content_length) = request.headers().get("content-length") {
        if let Ok(length_str) = content_length.to_str() {
            if let Ok(length) = length_str.parse::<usize>() {
                if length > limits.max_body_size {
                    return Err(ServerError::InvalidRequest(
                        format!("Request body too large: {} bytes (max: {})", length, limits.max_body_size)
                    ));
                }
            }
        }
    }

    Ok(next.run(request).await)
}

/// Logging middleware
pub async fn logging_middleware(
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let start = std::time::Instant::now();

    let user_id = {
        let auth_context = request.extensions().get::<AuthContext>();
        auth_context.map(|ctx| ctx.user_id.clone()).unwrap_or("unknown".to_string())
    };

    info!(
        method = %method,
        uri = %uri,
        user_id = %user_id,
        "Request started"
    );

    let response = next.run(request).await;
    let elapsed = start.elapsed();

    info!(
        method = %method,
        uri = %uri,
        user_id = %user_id,
        status = %response.status(),
        elapsed_ms = %elapsed.as_millis(),
        "Request completed"
    );

    response
}

/// CORS middleware
pub async fn cors_middleware(
    request: Request,
    next: Next,
) -> Result<Response, ServerError> {
    let method = request.method().clone();
    let mut response = next.run(request).await;

    let headers = response.headers_mut();
    
    // Basic CORS headers - can be made configurable
    headers.insert("Access-Control-Allow-Origin", "*".parse().unwrap());
    headers.insert(
        "Access-Control-Allow-Methods", 
        "GET, POST, PUT, DELETE, OPTIONS".parse().unwrap()
    );
    headers.insert(
        "Access-Control-Allow-Headers",
        "Content-Type, Authorization, X-API-Key".parse().unwrap()
    );
    headers.insert("Access-Control-Max-Age", "3600".parse().unwrap());

    // Handle preflight requests
    if method == axum::http::Method::OPTIONS {
        *response.status_mut() = StatusCode::NO_CONTENT;
    }

    Ok(response)
}

/// Security headers middleware
pub async fn security_headers_middleware(
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    
    let headers = response.headers_mut();
    
    // Security headers
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert(
        "Strict-Transport-Security",
        "max-age=31536000; includeSubDomains".parse().unwrap()
    );
    headers.insert(
        "Content-Security-Policy",
        "default-src 'self'".parse().unwrap()
    );
    
    response
}

/// Health check bypass middleware - allows health checks without auth
pub async fn health_check_bypass(
    request: Request,
    next: Next,
) -> Response {
    // Check if this is a health check endpoint
    if request.uri().path() == "/health" || request.uri().path() == "/health/ready" {
        // Create a simple OK response for health checks
        return Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"status":"ok","service":"woolly-server"}"#))
            .unwrap();
    }
    
    next.run(request).await
}

/// Concurrency limiting middleware
pub struct ConcurrencyLimiter {
    semaphore: Arc<tokio::sync::Semaphore>,
}

impl ConcurrencyLimiter {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
        }
    }

    pub async fn middleware(
        &self,
        request: Request,
        next: Next,
    ) -> Result<Response, ServerError> {
        let _permit = self.semaphore
            .acquire()
            .await
            .map_err(|_| ServerError::Internal("Failed to acquire concurrency permit".to_string()))?;
        
        Ok(next.run(request).await)
    }
}

/// Content negotiation middleware
pub async fn content_negotiation_middleware(
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    
    // Set default content type if not already set
    if !response.headers().contains_key("content-type") {
        response.headers_mut().insert(
            "content-type",
            "application/json".parse().unwrap()
        );
    }
    
    response
}