//! Authentication and authorization utilities

use crate::{config::AuthConfig, error::ServerError, error::ServerResult};
use axum::{
    extract::{Request, State},
    http::HeaderValue,
    middleware::Next,
    response::Response,
};
use headers::{Authorization, Header, HeaderMapExt};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header as JwtHeader, Validation};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// JWT claims structure
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,
    /// Issued at
    pub iat: u64,
    /// Expiration time
    pub exp: u64,
    /// Issuer
    pub iss: String,
    /// Audience
    pub aud: String,
    /// API key ID (optional)
    pub api_key_id: Option<String>,
}

/// Authentication context
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub user_id: String,
    pub api_key_id: Option<String>,
    pub is_authenticated: bool,
    pub scopes: Vec<String>,
}

/// API Key header
#[derive(Debug)]
pub struct ApiKey(pub String);

impl Header for ApiKey {
    fn name() -> &'static headers::HeaderName {
        static API_KEY_HEADER: std::sync::OnceLock<headers::HeaderName> = std::sync::OnceLock::new();
        API_KEY_HEADER.get_or_init(|| headers::HeaderName::from_static("x-api-key"))
    }

    fn decode<'i, I>(values: &mut I) -> Result<Self, headers::Error>
    where
        I: Iterator<Item = &'i HeaderValue>,
    {
        values
            .next()
            .and_then(|v| v.to_str().ok())
            .map(|s| ApiKey(s.to_string()))
            .ok_or_else(headers::Error::invalid)
    }

    fn encode<E>(&self, values: &mut E)
    where
        E: Extend<HeaderValue>,
    {
        if let Ok(value) = HeaderValue::from_str(&self.0) {
            values.extend(std::iter::once(value));
        }
    }
}

/// JWT token manager
#[derive(Clone)]
pub struct TokenManager {
    config: Arc<AuthConfig>,
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
}

impl TokenManager {
    /// Create a new token manager
    pub fn new(config: Arc<AuthConfig>) -> Self {
        let encoding_key = EncodingKey::from_secret(config.jwt_secret.as_bytes());
        let decoding_key = DecodingKey::from_secret(config.jwt_secret.as_bytes());

        Self {
            config,
            encoding_key,
            decoding_key,
        }
    }

    /// Generate a JWT token
    pub fn generate_token(&self, user_id: &str, api_key_id: Option<String>) -> ServerResult<String> {
        let now = chrono::Utc::now().timestamp() as u64;
        let claims = Claims {
            sub: user_id.to_string(),
            iat: now,
            exp: now + self.config.jwt_expiration,
            iss: "woolly-server".to_string(),
            aud: "woolly-api".to_string(),
            api_key_id,
        };

        encode(&JwtHeader::default(), &claims, &self.encoding_key)
            .map_err(|e| ServerError::Auth(format!("Failed to generate token: {}", e)))
    }

    /// Validate a JWT token
    pub fn validate_token(&self, token: &str) -> ServerResult<Claims> {
        let mut validation = Validation::default();
        // Set expected audience and issuer to match what we generate
        validation.set_audience(&["woolly-api"]);
        validation.set_issuer(&["woolly-server"]);
        
        decode::<Claims>(token, &self.decoding_key, &validation)
            .map(|data| data.claims)
            .map_err(|e| ServerError::Auth(format!("Invalid token: {}", e)))
    }

    /// Validate an API key
    pub fn validate_api_key(&self, api_key: &str) -> bool {
        self.config.api_keys.contains(&api_key.to_string())
    }
}

/// Authentication middleware
pub async fn auth_middleware(
    State(token_manager): State<Arc<TokenManager>>,
    mut request: Request,
    next: Next,
) -> Result<Response, ServerError> {
    let auth_context = extract_auth_context(&request, &token_manager).await?;
    
    // Insert auth context into request extensions
    request.extensions_mut().insert(auth_context);
    
    Ok(next.run(request).await)
}

/// Extract authentication context from request
async fn extract_auth_context(
    request: &Request,
    token_manager: &TokenManager,
) -> ServerResult<AuthContext> {
    let headers = request.headers();

    // Try JWT token first
    if let Some(auth_header) = headers.typed_get::<Authorization<headers::authorization::Bearer>>() {
        let token = auth_header.token();
        match token_manager.validate_token(token) {
            Ok(claims) => {
                return Ok(AuthContext {
                    user_id: claims.sub,
                    api_key_id: claims.api_key_id,
                    is_authenticated: true,
                    scopes: vec!["api:read".to_string(), "api:write".to_string()],
                });
            }
            Err(_) => {
                // Token invalid, try other methods
            }
        }
    }

    // Try API key
    if let Some(api_key_header) = headers.typed_get::<ApiKey>() {
        let api_key = &api_key_header.0;
        if token_manager.validate_api_key(api_key) {
            return Ok(AuthContext {
                user_id: format!("api-key-{}", Uuid::new_v4()),
                api_key_id: Some(api_key.clone()),
                is_authenticated: true,
                scopes: vec!["api:read".to_string(), "api:write".to_string()],
            });
        }
    }

    // Check if anonymous access is allowed
    if token_manager.config.allow_anonymous {
        Ok(AuthContext {
            user_id: format!("anonymous-{}", Uuid::new_v4()),
            api_key_id: None,
            is_authenticated: false,
            scopes: vec!["api:read".to_string()],
        })
    } else {
        Err(ServerError::Auth("Authentication required".to_string()))
    }
}

/// Require authentication middleware
pub async fn require_auth_middleware(
    request: Request,
    next: Next,
) -> Result<Response, ServerError> {
    let auth_context = request
        .extensions()
        .get::<AuthContext>()
        .ok_or_else(|| ServerError::Internal("Auth context not found".to_string()))?;

    if !auth_context.is_authenticated {
        return Err(ServerError::Authorization(
            "Authentication required for this endpoint".to_string(),
        ));
    }

    Ok(next.run(request).await)
}

/// Require specific scope middleware
pub fn require_scope(required_scope: &'static str) -> impl Fn(Request, Next) -> futures::future::BoxFuture<'static, Result<Response, ServerError>> + Clone {
    move |request: Request, next: Next| {
        Box::pin(async move {
            let auth_context = request
                .extensions()
                .get::<AuthContext>()
                .ok_or_else(|| ServerError::Internal("Auth context not found".to_string()))?;

            if !auth_context.scopes.contains(&required_scope.to_string()) {
                return Err(ServerError::Authorization(
                    format!("Scope '{}' required for this endpoint", required_scope),
                ));
            }

            Ok(next.run(request).await)
        })
    }
}

/// Extract auth context from request extensions
pub fn extract_auth(request: &Request) -> ServerResult<&AuthContext> {
    request
        .extensions()
        .get::<AuthContext>()
        .ok_or_else(|| ServerError::Internal("Auth context not found".to_string()))
}