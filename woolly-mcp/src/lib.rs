//! Woolly MCP (Model Context Protocol) Implementation
//! 
//! This crate provides a comprehensive MCP implementation for the Woolly inference engine,
//! supporting multiple transport layers (stdio, HTTP, WebSocket) and extensible plugin architecture.

pub mod hooks;
pub mod protocol;
pub mod registry;
pub mod transport;
pub mod types;

// Re-export commonly used types
pub use protocol::{McpHandler, McpProtocol};
pub use registry::{PluginRegistry, PluginInfo};
pub use transport::{Transport, TransportError};
pub use types::{McpMessage, McpRequest, McpResponse, McpError};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::protocol::*;
    pub use crate::registry::*;
    pub use crate::transport::*;
    pub use crate::types::*;
    pub use crate::hooks::*;
}