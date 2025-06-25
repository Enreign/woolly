//! Core server implementation

use crate::{
    auth::TokenManager,
    config::ServerConfig,
    error::{ServerError, ServerResult},
    handlers,
    middleware::{
        ConcurrencyLimiter, RateLimiterState,
    },
    websocket::websocket_handler,
};
#[cfg(feature = "mcp")]
use crate::mcp::McpServerState;

use axum::{
    routing::{get, post, delete},
    Router,
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::{
    trace::TraceLayer,
};
use tracing::{info, warn};
use woolly_core::{engine::InferenceEngine, config::EngineConfig};

/// Main server state
#[derive(Clone)]
pub struct ServerState {
    pub config: Arc<ServerConfig>,
    pub inference_engine: Arc<tokio::sync::RwLock<InferenceEngine>>,
    pub token_manager: Arc<TokenManager>,
    pub rate_limiter: Arc<RateLimiterState>,
    pub concurrency_limiter: Arc<ConcurrencyLimiter>,
    #[cfg(feature = "mcp")]
    pub mcp_state: Arc<McpServerState>,
}

/// Woolly HTTP/WebSocket server
pub struct WoollyServer {
    config: Arc<ServerConfig>,
    state: ServerState,
}

impl WoollyServer {
    /// Create a new server instance
    pub fn new(config: ServerConfig) -> ServerResult<Self> {
        let config = Arc::new(config);
        
        // Initialize inference engine
        let engine_config = EngineConfig::default(); // TODO: Make configurable
        let inference_engine = Arc::new(tokio::sync::RwLock::new(InferenceEngine::new(engine_config)));
        
        // Initialize authentication
        let token_manager = Arc::new(TokenManager::new(Arc::new(config.auth.clone())));
        
        // Initialize rate limiting
        let rate_limiter = Arc::new(RateLimiterState::new(&config.rate_limit));
        
        // Initialize concurrency limiting
        let concurrency_limiter = Arc::new(ConcurrencyLimiter::new(
            config.limits.max_concurrent_requests
        ));
        
        // Initialize MCP state
        #[cfg(feature = "mcp")]
        let mcp_state = Arc::new(McpServerState::new(&config.mcp)?);
        
        let state = ServerState {
            config: Arc::clone(&config),
            inference_engine,
            token_manager,
            rate_limiter,
            concurrency_limiter,
            #[cfg(feature = "mcp")]
            mcp_state,
        };

        Ok(Self { config, state })
    }

    /// Build the router with all routes and middleware
    fn build_router(&self) -> Router {
        let api_routes = Router::new()
            // Health endpoints
            .route("/health", get(handlers::health::health_check))
            .route("/health/ready", get(handlers::health::readiness_check))
            .route("/health/live", get(handlers::health::liveness_check))
            
            // Authentication endpoints
            .route("/auth/token", post(handlers::auth::create_token))
            .route("/auth/validate", post(handlers::auth::validate_token))
            
            // Model management endpoints
            .route("/models", get(handlers::models::list_models))
            .route("/models/:model_name", get(handlers::models::get_model_info))
            .route("/models/:model_name/load", post(handlers::models::load_model))
            .route("/models/:model_name/unload", post(handlers::models::unload_model))
            
            // Inference endpoints
            .route("/inference/complete", post(handlers::inference::complete))
            .route("/inference/stream", post(handlers::inference::stream))
            .route("/inference/chat", post(handlers::inference::chat))
            
            // Session management
            .route("/sessions", get(handlers::sessions::list_sessions))
            .route("/sessions", post(handlers::sessions::create_session))
            .route("/sessions/:session_id", get(handlers::sessions::get_session))
            .route("/sessions/:session_id", delete(handlers::sessions::delete_session))
            
            // MCP endpoints
            .route("/mcp/tools", get(handlers::mcp::list_tools))
            .route("/mcp/tools/:tool_name", post(handlers::mcp::execute_tool))
            .route("/mcp/resources", get(handlers::mcp::list_resources))
            .route("/mcp/resources/*path", get(handlers::mcp::get_resource))
            .route("/mcp/prompts", get(handlers::mcp::list_prompts))
            .route("/mcp/prompts/:prompt_name", post(handlers::mcp::get_prompt))
            
            // WebSocket endpoint for MCP
            .route("/ws", get(websocket_handler))
            .route("/mcp/ws", get(websocket_handler));

        Router::new()
            .nest("/api/v1", api_routes)
            .layer(TraceLayer::new_for_http())
            .with_state(self.state.clone())
    }

    /// Start the server
    pub async fn start(self) -> ServerResult<()> {
        let app = self.build_router();
        let addr = self.config.bind;

        info!("Starting Woolly server on {}", addr);
        
        // Preload models if configured
        if !self.config.models.preload_models.is_empty() {
            info!("Preloading models: {:?}", self.config.models.preload_models);
            // TODO: Implement model preloading
        }

        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| ServerError::Io(e))?;

        info!("Server listening on {}", addr);

        axum::serve(listener, app)
            .await
            .map_err(|e| ServerError::Internal(format!("Server error: {}", e)))?;

        Ok(())
    }

    /// Gracefully shutdown the server
    pub async fn shutdown(&self) -> ServerResult<()> {
        info!("Shutting down Woolly server");
        
        // TODO: Implement graceful shutdown
        // - Stop accepting new requests
        // - Wait for existing requests to complete
        // - Clean up resources
        
        Ok(())
    }

    /// Get server configuration
    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    /// Get server state
    pub fn state(&self) -> &ServerState {
        &self.state
    }
}

/// Create default server with configuration file
pub async fn create_server_from_config(config_path: Option<std::path::PathBuf>) -> ServerResult<WoollyServer> {
    let config = if let Some(path) = config_path {
        ServerConfig::from_file(&path)
            .map_err(|e| ServerError::Config(format!("Failed to load config: {}", e)))?
    } else {
        ServerConfig::default()
    };

    WoollyServer::new(config)
}

/// Run server with graceful shutdown handling
pub async fn run_server(server: WoollyServer) -> ServerResult<()> {
    // Set up graceful shutdown
    let shutdown_signal = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for shutdown signal");
        info!("Shutdown signal received");
    };

    // Create a shared reference to handle shutdown
    let server_ref = std::sync::Arc::new(server);
    let server_clone = Arc::clone(&server_ref);

    // Run server with shutdown handling
    tokio::select! {
        result = async {
            // Clone the server for the start operation
            let config = server_ref.config().clone();
            let state = server_ref.state().clone();
            let new_server = WoollyServer { config: Arc::new(config), state };
            new_server.start().await
        } => {
            if let Err(e) = result {
                warn!("Server error: {}", e);
                return Err(e);
            }
        }
        _ = shutdown_signal => {
            info!("Graceful shutdown initiated");
            server_clone.shutdown().await?;
        }
    }

    Ok(())
}