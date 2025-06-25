//! MCP protocol trait definitions

use async_trait::async_trait;
use std::collections::HashMap;
use crate::types::*;

/// Result type for MCP operations
pub type McpResult<T> = Result<T, McpError>;

/// Core MCP protocol trait
#[async_trait]
pub trait McpProtocol: Send + Sync {
    /// Initialize the MCP connection
    async fn initialize(&self, request: InitializeRequest) -> McpResult<InitializeResponse>;
    
    /// Handle an incoming MCP message
    async fn handle_message(&self, message: McpMessage) -> McpResult<Option<McpMessage>>;
    
    /// Get server capabilities
    fn capabilities(&self) -> &McpCapabilities;
    
    /// Shutdown the protocol handler
    async fn shutdown(&self) -> McpResult<()>;
}

/// MCP handler for processing specific method calls
#[async_trait]
pub trait McpHandler: Send + Sync {
    /// Get the method pattern this handler supports (e.g., "tools/*", "resources/*")
    fn method_pattern(&self) -> &str;
    
    /// Handle a request for this method
    async fn handle_request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> McpResult<serde_json::Value>;
    
    /// Handle a notification for this method
    async fn handle_notification(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> McpResult<()> {
        // Default implementation does nothing
        let _ = (method, params);
        Ok(())
    }
}

/// Tool handler trait
#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Get tool information
    fn tool_info(&self) -> ToolInfo;
    
    /// Execute the tool
    async fn execute(&self, arguments: serde_json::Value) -> McpResult<ToolCallResponse>;
}

/// Resource handler trait
#[async_trait]
pub trait ResourceHandler: Send + Sync {
    /// Get resource information
    fn resource_info(&self) -> ResourceInfo;
    
    /// Read the resource
    async fn read(&self, uri: &str) -> McpResult<ResourceReadResponse>;
}

/// Prompt handler trait
#[async_trait]
pub trait PromptHandler: Send + Sync {
    /// Get prompt information
    fn prompt_info(&self) -> PromptInfo;
    
    /// Get prompt messages
    async fn get(
        &self,
        arguments: HashMap<String, serde_json::Value>,
    ) -> McpResult<PromptGetResponse>;
}