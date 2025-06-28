//! Woolly Server - HTTP/WebSocket server with MCP integration
//!
//! This crate provides a web server that exposes Woolly's capabilities over HTTP and WebSocket
//! connections with full Model Context Protocol (MCP) support.

/// Version of the woolly-server crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod auth;
pub mod config;
pub mod error;
pub mod handlers;
pub mod middleware;
pub mod server;
pub mod websocket;

#[cfg(feature = "mcp")]
pub mod mcp;

// Re-export commonly used types
pub use config::ServerConfig;
pub use error::{ServerError, ServerResult};
pub use server::WoollyServer;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        auth::*,
        config::*,
        error::*,
        handlers::*,
        middleware::*,
        server::*,
        websocket::*,
    };
    
    #[cfg(feature = "mcp")]
    pub use crate::mcp::*;
}