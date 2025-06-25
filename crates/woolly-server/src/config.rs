//! Server configuration

use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf};

/// Main server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server binding address
    pub bind: SocketAddr,
    
    /// Authentication configuration
    pub auth: AuthConfig,
    
    /// Rate limiting configuration  
    pub rate_limit: RateLimitConfig,
    
    /// MCP configuration
    #[cfg(feature = "mcp")]
    pub mcp: McpConfig,
    
    /// CORS configuration
    pub cors: CorsConfig,
    
    /// TLS configuration (optional)
    pub tls: Option<TlsConfig>,
    
    /// Model paths and configuration
    pub models: ModelConfig,
    
    /// Request limits
    pub limits: RequestLimits,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// JWT secret key
    pub jwt_secret: String,
    
    /// JWT token expiration in seconds
    pub jwt_expiration: u64,
    
    /// API keys for authentication
    pub api_keys: Vec<String>,
    
    /// Enable anonymous access for certain endpoints
    pub allow_anonymous: bool,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per minute for authenticated users
    pub authenticated_rpm: u32,
    
    /// Requests per minute for anonymous users
    pub anonymous_rpm: u32,
    
    /// Burst capacity
    pub burst_capacity: u32,
}

/// MCP-specific configuration
#[cfg(feature = "mcp")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// Enable MCP protocol support
    pub enabled: bool,
    
    /// MCP protocol version to advertise
    pub protocol_version: String,
    
    /// Server information
    pub server_info: McpServerInfo,
    
    /// Tools directory for dynamic tool loading
    pub tools_dir: Option<PathBuf>,
    
    /// Resources directory for serving static resources
    pub resources_dir: Option<PathBuf>,
}

#[cfg(feature = "mcp")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerInfo {
    pub name: String,
    pub version: String,
}

/// CORS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorsConfig {
    /// Enable CORS
    pub enabled: bool,
    
    /// Allowed origins (empty means all)
    pub allowed_origins: Vec<String>,
    
    /// Allowed methods
    pub allowed_methods: Vec<String>,
    
    /// Allowed headers
    pub allowed_headers: Vec<String>,
    
    /// Max age for preflight requests
    pub max_age: u64,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Certificate file path
    pub cert_path: PathBuf,
    
    /// Private key file path
    pub key_path: PathBuf,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Default model directory
    pub models_dir: PathBuf,
    
    /// Maximum number of concurrent sessions
    pub max_sessions: usize,
    
    /// Default model to load on startup
    pub default_model: Option<String>,
    
    /// Preload models on startup
    pub preload_models: Vec<String>,
}

/// Request size and time limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestLimits {
    /// Maximum request body size in bytes
    pub max_body_size: usize,
    
    /// Request timeout in seconds
    pub request_timeout: u64,
    
    /// Maximum tokens per request
    pub max_tokens: usize,
    
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8080".parse().unwrap(),
            auth: AuthConfig::default(),
            rate_limit: RateLimitConfig::default(),
            #[cfg(feature = "mcp")]
            mcp: McpConfig::default(),
            cors: CorsConfig::default(),
            tls: None,
            models: ModelConfig::default(),
            limits: RequestLimits::default(),
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            jwt_secret: "default-jwt-secret-change-in-production".to_string(),
            jwt_expiration: 3600, // 1 hour
            api_keys: vec!["demo-api-key".to_string()],
            allow_anonymous: true,
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            authenticated_rpm: 60,
            anonymous_rpm: 10,
            burst_capacity: 10,
        }
    }
}

#[cfg(feature = "mcp")]
impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            protocol_version: "0.1.0".to_string(),
            server_info: McpServerInfo {
                name: "Woolly MCP Server".to_string(),
                version: crate::VERSION.to_string(),
            },
            tools_dir: None,
            resources_dir: None,
        }
    }
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "PUT".to_string(),
                "DELETE".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Authorization".to_string(),
                "X-API-Key".to_string(),
            ],
            max_age: 3600,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            models_dir: PathBuf::from("./models"),
            max_sessions: 10,
            default_model: None,
            preload_models: Vec::new(),
        }
    }
}

impl Default for RequestLimits {
    fn default() -> Self {
        Self {
            max_body_size: 10 * 1024 * 1024, // 10MB
            request_timeout: 300, // 5 minutes
            max_tokens: 4096,
            max_concurrent_requests: 100,
        }
    }
}

impl ServerConfig {
    /// Load configuration from file
    pub fn from_file(path: &PathBuf) -> Result<Self, config::ConfigError> {
        let settings = config::Config::builder()
            .add_source(config::File::from(path.as_path()))
            .add_source(config::Environment::with_prefix("WOOLLY"))
            .build()?;
        
        settings.try_deserialize()
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let toml_string = toml::to_string_pretty(self)?;
        std::fs::write(path, toml_string)?;
        Ok(())
    }
}