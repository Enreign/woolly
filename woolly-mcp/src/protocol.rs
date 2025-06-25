//! MCP protocol trait definitions

use async_trait::async_trait;
use std::sync::Arc;
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

/// Default MCP protocol implementation
pub struct DefaultMcpProtocol {
    capabilities: McpCapabilities,
    handlers: dashmap::DashMap<String, Arc<dyn McpHandler>>,
    tools: dashmap::DashMap<String, Arc<dyn ToolHandler>>,
    resources: dashmap::DashMap<String, Arc<dyn ResourceHandler>>,
    prompts: dashmap::DashMap<String, Arc<dyn PromptHandler>>,
}

impl DefaultMcpProtocol {
    /// Create a new default MCP protocol instance
    pub fn new() -> Self {
        Self {
            capabilities: McpCapabilities::default(),
            handlers: dashmap::DashMap::new(),
            tools: dashmap::DashMap::new(),
            resources: dashmap::DashMap::new(),
            prompts: dashmap::DashMap::new(),
        }
    }
    
    /// Register a method handler
    pub fn register_handler(&self, handler: Arc<dyn McpHandler>) {
        let pattern = handler.method_pattern().to_string();
        self.handlers.insert(pattern, handler);
    }
    
    /// Register a tool handler
    pub fn register_tool(&self, handler: Arc<dyn ToolHandler>) {
        let info = handler.tool_info();
        self.tools.insert(info.name.clone(), handler);
        
        // Update capabilities
        // Note: This is simplified, in production you'd want proper synchronization
    }
    
    /// Register a resource handler
    pub fn register_resource(&self, handler: Arc<dyn ResourceHandler>) {
        let info = handler.resource_info();
        self.resources.insert(info.uri.clone(), handler);
    }
    
    /// Register a prompt handler
    pub fn register_prompt(&self, handler: Arc<dyn PromptHandler>) {
        let info = handler.prompt_info();
        self.prompts.insert(info.name.clone(), handler);
    }
    
    /// Find a handler for a given method
    fn find_handler(&self, method: &str) -> Option<Arc<dyn McpHandler>> {
        // First try exact match
        if let Some(handler) = self.handlers.get(method) {
            return Some(handler.clone());
        }
        
        // Then try pattern matching
        for entry in self.handlers.iter() {
            let pattern = entry.key();
            if Self::matches_pattern(pattern, method) {
                return Some(entry.value().clone());
            }
        }
        
        None
    }
    
    /// Check if a method matches a pattern
    fn matches_pattern(pattern: &str, method: &str) -> bool {
        if pattern.ends_with("/*") {
            let prefix = &pattern[..pattern.len() - 2];
            method.starts_with(prefix) && method.contains('/')
        } else {
            pattern == method
        }
    }
}

#[async_trait]
impl McpProtocol for DefaultMcpProtocol {
    async fn initialize(&self, request: InitializeRequest) -> McpResult<InitializeResponse> {
        // Validate protocol version
        if request.protocol_version != MCP_VERSION {
            return Err(McpError {
                code: error_codes::INVALID_REQUEST,
                message: format!(
                    "Unsupported protocol version: {}. Expected: {}",
                    request.protocol_version, MCP_VERSION
                ),
                data: None,
            });
        }
        
        // Build response with current capabilities
        let mut capabilities = self.capabilities.clone();
        
        // Add registered tools
        if !self.tools.is_empty() {
            let tools: Vec<_> = self.tools
                .iter()
                .map(|entry| entry.value().tool_info())
                .collect();
            capabilities.tools = Some(tools);
        }
        
        // Add registered resources
        if !self.resources.is_empty() {
            let resources: Vec<_> = self.resources
                .iter()
                .map(|entry| entry.value().resource_info())
                .collect();
            capabilities.resources = Some(resources);
        }
        
        // Add registered prompts
        if !self.prompts.is_empty() {
            let prompts: Vec<_> = self.prompts
                .iter()
                .map(|entry| entry.value().prompt_info())
                .collect();
            capabilities.prompts = Some(prompts);
        }
        
        Ok(InitializeResponse {
            protocol_version: MCP_VERSION.to_string(),
            capabilities,
            server_info: Some(ServerInfo {
                name: "woolly-mcp".to_string(),
                version: crate::VERSION.to_string(),
            }),
        })
    }
    
    async fn handle_message(&self, message: McpMessage) -> McpResult<Option<McpMessage>> {
        match message {
            McpMessage::Request(request) => {
                let response = self.handle_request(request).await?;
                Ok(Some(McpMessage::Response(response)))
            }
            McpMessage::Notification(notification) => {
                self.handle_notification(notification).await?;
                Ok(None)
            }
            _ => Err(McpError {
                code: error_codes::INVALID_REQUEST,
                message: "Unexpected message type".to_string(),
                data: None,
            }),
        }
    }
    
    fn capabilities(&self) -> &McpCapabilities {
        &self.capabilities
    }
    
    async fn shutdown(&self) -> McpResult<()> {
        // Clean shutdown logic
        Ok(())
    }
}

impl DefaultMcpProtocol {
    async fn handle_request(&self, request: McpRequest) -> McpResult<McpResponse> {
        let result = match request.method.as_str() {
            "initialize" => {
                // Handle initialize specially
                if let Some(params) = request.params {
                    let init_req: InitializeRequest = serde_json::from_value(params)
                        .map_err(|e| McpError {
                            code: error_codes::INVALID_PARAMS,
                            message: format!("Invalid initialize params: {}", e),
                            data: None,
                        })?;
                    let init_resp = self.initialize(init_req).await?;
                    serde_json::to_value(init_resp).map_err(|e| McpError {
                        code: error_codes::INTERNAL_ERROR,
                        message: format!("Failed to serialize response: {}", e),
                        data: None,
                    })?
                } else {
                    return Err(McpError {
                        code: error_codes::INVALID_PARAMS,
                        message: "Initialize requires parameters".to_string(),
                        data: None,
                    });
                }
            }
            "tools/call" => {
                // Handle tool calls
                if let Some(params) = request.params {
                    let tool_req: ToolCallRequest = serde_json::from_value(params)
                        .map_err(|e| McpError {
                            code: error_codes::INVALID_PARAMS,
                            message: format!("Invalid tool call params: {}", e),
                            data: None,
                        })?;
                    
                    if let Some(handler) = self.tools.get(&tool_req.name) {
                        let response = handler.execute(tool_req.arguments).await?;
                        serde_json::to_value(response).map_err(|e| McpError {
                            code: error_codes::INTERNAL_ERROR,
                            message: format!("Failed to serialize response: {}", e),
                            data: None,
                        })?
                    } else {
                        return Err(McpError {
                            code: error_codes::METHOD_NOT_FOUND,
                            message: format!("Tool not found: {}", tool_req.name),
                            data: None,
                        });
                    }
                } else {
                    return Err(McpError {
                        code: error_codes::INVALID_PARAMS,
                        message: "Tool call requires parameters".to_string(),
                        data: None,
                    });
                }
            }
            method => {
                // Try to find a registered handler
                if let Some(handler) = self.find_handler(method) {
                    handler.handle_request(method, request.params).await?
                } else {
                    return Err(McpError {
                        code: error_codes::METHOD_NOT_FOUND,
                        message: format!("Method not found: {}", method),
                        data: None,
                    });
                }
            }
        };
        
        Ok(McpResponse {
            id: request.id,
            result: Some(result),
            error: None,
            metadata: None,
        })
    }
    
    async fn handle_notification(&self, notification: McpNotification) -> McpResult<()> {
        if let Some(handler) = self.find_handler(&notification.method) {
            handler.handle_notification(&notification.method, notification.params).await
        } else {
            // Notifications can be ignored if no handler is found
            Ok(())
        }
    }
}
