//! MCP (Model Context Protocol) integration

use crate::{
    config::McpConfig,
    error::{ServerError, ServerResult},
    websocket::WebSocketConnection,
};
use async_trait::async_trait;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;
use woolly_mcp::{
    prelude::*,
};

/// MCP server state
pub struct McpServerState {
    config: McpConfig,
    protocol_handler: Arc<dyn McpProtocol>,
    tool_handlers: RwLock<HashMap<String, Arc<dyn ToolHandler>>>,
    resource_handlers: RwLock<HashMap<String, Arc<dyn ResourceHandler>>>,
    prompt_handlers: RwLock<HashMap<String, Arc<dyn PromptHandler>>>,
    capabilities: McpCapabilities,
}

impl McpServerState {
    /// Create new MCP server state
    pub fn new(config: &McpConfig) -> ServerResult<Self> {
        let capabilities = Self::build_capabilities(config)?;
        let protocol_handler = Arc::new(WoollyMcpProtocol::new(config.clone(), capabilities.clone()));

        Ok(Self {
            config: config.clone(),
            protocol_handler,
            tool_handlers: RwLock::new(HashMap::new()),
            resource_handlers: RwLock::new(HashMap::new()),
            prompt_handlers: RwLock::new(HashMap::new()),
            capabilities,
        })
    }

    /// Check if MCP server is initialized
    pub fn is_initialized(&self) -> bool {
        self.config.enabled
    }

    /// Get server capabilities
    pub async fn get_capabilities(&self) -> ServerResult<&McpCapabilities> {
        Ok(&self.capabilities)
    }

    /// Handle MCP message
    pub async fn handle_message(
        &self,
        message: McpMessage,
        _connection: &WebSocketConnection,
    ) -> ServerResult<Option<McpMessage>> {
        if !self.config.enabled {
            return Err(ServerError::Internal("MCP not enabled".to_string()));
        }

        match self.protocol_handler.handle_message(message).await {
            Ok(response) => Ok(response),
            Err(e) => Err(ServerError::Mcp(e)),
        }
    }

    /// Register a tool handler
    pub async fn register_tool(&self, handler: Arc<dyn ToolHandler>) -> ServerResult<()> {
        let tool_info = handler.tool_info();
        self.tool_handlers.write().await.insert(tool_info.name.clone(), handler);
        Ok(())
    }

    /// Register a resource handler
    pub async fn register_resource(&self, handler: Arc<dyn ResourceHandler>) -> ServerResult<()> {
        let resource_info = handler.resource_info();
        self.resource_handlers.write().await.insert(resource_info.uri.clone(), handler);
        Ok(())
    }

    /// Register a prompt handler
    pub async fn register_prompt(&self, handler: Arc<dyn PromptHandler>) -> ServerResult<()> {
        let prompt_info = handler.prompt_info();
        self.prompt_handlers.write().await.insert(prompt_info.name.clone(), handler);
        Ok(())
    }

    /// List available tools
    pub async fn list_tools(&self) -> ServerResult<Vec<ToolInfo>> {
        let handlers = self.tool_handlers.read().await;
        Ok(handlers.values().map(|h| h.tool_info()).collect())
    }

    /// Execute a tool
    pub async fn execute_tool(&self, name: &str, arguments: serde_json::Value) -> ServerResult<ToolCallResponse> {
        let handlers = self.tool_handlers.read().await;
        match handlers.get(name) {
            Some(handler) => {
                handler.execute(arguments).await.map_err(ServerError::Mcp)
            }
            None => Err(ServerError::InvalidRequest(format!("Tool '{}' not found", name))),
        }
    }

    /// List available resources
    pub async fn list_resources(&self) -> ServerResult<Vec<ResourceInfo>> {
        let handlers = self.resource_handlers.read().await;
        Ok(handlers.values().map(|h| h.resource_info()).collect())
    }

    /// Get a resource
    pub async fn get_resource(&self, uri: &str) -> ServerResult<ResourceReadResponse> {
        let handlers = self.resource_handlers.read().await;
        match handlers.get(uri) {
            Some(handler) => {
                handler.read(uri).await.map_err(ServerError::Mcp)
            }
            None => Err(ServerError::InvalidRequest(format!("Resource '{}' not found", uri))),
        }
    }

    /// List available prompts
    pub async fn list_prompts(&self) -> ServerResult<Vec<PromptInfo>> {
        let handlers = self.prompt_handlers.read().await;
        Ok(handlers.values().map(|h| h.prompt_info()).collect())
    }

    /// Get a prompt
    pub async fn get_prompt(
        &self,
        name: &str,
        arguments: HashMap<String, serde_json::Value>,
    ) -> ServerResult<PromptGetResponse> {
        let handlers = self.prompt_handlers.read().await;
        match handlers.get(name) {
            Some(handler) => {
                handler.get(arguments).await.map_err(ServerError::Mcp)
            }
            None => Err(ServerError::InvalidRequest(format!("Prompt '{}' not found", name))),
        }
    }

    /// Build server capabilities
    fn build_capabilities(_config: &McpConfig) -> ServerResult<McpCapabilities> {
        let mut capabilities = McpCapabilities::default();

        // Initialize with basic capabilities
        capabilities.tools = Some(vec![
            // Built-in inference tool
            ToolInfo {
                name: "inference".to_string(),
                description: Some("Run inference on the loaded model".to_string()),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "max_tokens": {"type": "number"},
                        "temperature": {"type": "number"},
                        "stream": {"type": "boolean"}
                    },
                    "required": ["prompt"]
                }),
            },
        ]);

        capabilities.resources = Some(vec![
            ResourceInfo {
                uri: "model://info".to_string(),
                name: "Model Information".to_string(),
                description: Some("Information about the currently loaded model".to_string()),
                mime_type: Some("application/json".to_string()),
            },
        ]);

        capabilities.prompts = Some(vec![
            PromptInfo {
                name: "system".to_string(),
                description: Some("System prompt template".to_string()),
                arguments: vec![
                    PromptArgument {
                        name: "task".to_string(),
                        description: Some("The task description".to_string()),
                        required: true,
                    },
                ],
            },
        ]);

        Ok(capabilities)
    }
}

/// Woolly MCP protocol implementation
struct WoollyMcpProtocol {
    config: McpConfig,
    capabilities: McpCapabilities,
}

impl WoollyMcpProtocol {
    fn new(config: McpConfig, capabilities: McpCapabilities) -> Self {
        Self { config, capabilities }
    }
}

#[async_trait]
impl McpProtocol for WoollyMcpProtocol {
    async fn initialize(&self, _request: InitializeRequest) -> McpResult<InitializeResponse> {
        Ok(InitializeResponse {
            protocol_version: self.config.protocol_version.clone(),
            capabilities: self.capabilities.clone(),
            server_info: Some(ServerInfo {
                name: self.config.server_info.name.clone(),
                version: self.config.server_info.version.clone(),
            }),
        })
    }

    async fn handle_message(&self, message: McpMessage) -> McpResult<Option<McpMessage>> {
        match message {
            McpMessage::Request(request) => {
                match request.method.as_str() {
                    "initialize" => {
                        if let Some(params) = request.params {
                            let init_request: InitializeRequest = serde_json::from_value(params)
                                .map_err(|e| McpError {
                                    code: error_codes::INVALID_PARAMS,
                                    message: format!("Invalid initialize params: {}", e),
                                    data: None,
                                })?;
                            
                            let response = self.initialize(init_request).await?;
                            
                            Ok(Some(McpMessage::Response(McpResponse {
                                id: request.id,
                                result: Some(serde_json::to_value(response).unwrap()),
                                error: None,
                                metadata: None,
                            })))
                        } else {
                            Err(McpError {
                                code: error_codes::INVALID_PARAMS,
                                message: "Missing initialize params".to_string(),
                                data: None,
                            })
                        }
                    }
                    "tools/list" => {
                        let tools = self.capabilities.tools.clone().unwrap_or_default();
                        Ok(Some(McpMessage::Response(McpResponse {
                            id: request.id,
                            result: Some(serde_json::json!({ "tools": tools })),
                            error: None,
                            metadata: None,
                        })))
                    }
                    "resources/list" => {
                        let resources = self.capabilities.resources.clone().unwrap_or_default();
                        Ok(Some(McpMessage::Response(McpResponse {
                            id: request.id,
                            result: Some(serde_json::json!({ "resources": resources })),
                            error: None,
                            metadata: None,
                        })))
                    }
                    "prompts/list" => {
                        let prompts = self.capabilities.prompts.clone().unwrap_or_default();
                        Ok(Some(McpMessage::Response(McpResponse {
                            id: request.id,
                            result: Some(serde_json::json!({ "prompts": prompts })),
                            error: None,
                            metadata: None,
                        })))
                    }
                    _ => {
                        Err(McpError {
                            code: error_codes::METHOD_NOT_FOUND,
                            message: format!("Method '{}' not found", request.method),
                            data: None,
                        })
                    }
                }
            }
            McpMessage::Notification(_) => {
                // Handle notifications (no response expected)
                Ok(None)
            }
            _ => {
                // Echo back other message types for now
                Ok(Some(message))
            }
        }
    }

    fn capabilities(&self) -> &McpCapabilities {
        &self.capabilities
    }

    async fn shutdown(&self) -> McpResult<()> {
        Ok(())
    }
}