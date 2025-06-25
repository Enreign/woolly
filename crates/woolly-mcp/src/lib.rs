//! Woolly MCP - Model Context Protocol Handler
//!
//! This crate implements the Model Context Protocol (MCP) for Woolly LLM,
//! providing a standardized interface for model serving and communication.

// Core modules
pub mod types;
pub mod protocol;
pub mod hooks;
pub mod registry;

// Re-export commonly used types
pub use protocol::{McpHandler, McpProtocol, ToolHandler, ResourceHandler, PromptHandler};
pub use registry::{PluginRegistry, PluginInfo};
pub use types::{McpMessage, McpRequest, McpResponse, McpError};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::protocol::*;
    pub use crate::registry::*;
    pub use crate::types::*;
    pub use crate::hooks::*;
}

