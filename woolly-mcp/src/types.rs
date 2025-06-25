//! Core MCP types and messages

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// MCP protocol version
pub const MCP_VERSION: &str = "1.0.0";

/// Unique request/response ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(pub Uuid);

impl Default for MessageId {
    fn default() -> Self {
        Self(Uuid::new_v4())
    }
}

/// MCP message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpMessage {
    Request(McpRequest),
    Response(McpResponse),
    Notification(McpNotification),
    Error(McpError),
}

/// MCP request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub id: MessageId,
    pub method: String,
    pub params: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// MCP response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub id: MessageId,
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// MCP notification (no response expected)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpNotification {
    pub method: String,
    pub params: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// MCP error
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[error("{message}")]
pub struct McpError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Standard MCP error codes
pub mod error_codes {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
    pub const SERVER_ERROR: i32 = -32000;
}

/// MCP capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpCapabilities {
    /// Supported tools/functions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolInfo>>,
    
    /// Supported resources
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<Vec<ResourceInfo>>,
    
    /// Supported prompts
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<Vec<PromptInfo>>,
    
    /// Additional capabilities
    #[serde(flatten)]
    pub extensions: HashMap<String, serde_json::Value>,
}

/// Tool/function information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<serde_json::Value>,
}

/// Resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    pub uri: String,
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Prompt template information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptInfo {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Vec<PromptArgument>>,
}

/// Prompt argument definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    pub name: String,
    pub description: String,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
}

/// Initialize request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeRequest {
    pub protocol_version: String,
    pub capabilities: McpCapabilities,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_info: Option<ClientInfo>,
}

/// Initialize response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResponse {
    pub protocol_version: String,
    pub capabilities: McpCapabilities,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_info: Option<ServerInfo>,
}

/// Client information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

/// Server information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

/// Tool call request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Tool call response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResponse {
    pub content: Vec<ToolResponseContent>,
}

/// Tool response content item
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResponseContent {
    Text { text: String },
    Image { data: String, mime_type: String },
    Error { error: String },
}

/// Resource read request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReadRequest {
    pub uri: String,
}

/// Resource read response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReadResponse {
    pub contents: Vec<ResourceContent>,
}

/// Resource content item
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResourceContent {
    Text { uri: String, text: String },
    Binary { uri: String, data: String, mime_type: String },
}

/// Prompt get request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptGetRequest {
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

/// Prompt get response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptGetResponse {
    pub messages: Vec<PromptMessage>,
}

/// Prompt message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMessage {
    pub role: String,
    pub content: PromptMessageContent,
}

/// Prompt message content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromptMessageContent {
    Text(String),
    Complex(Vec<PromptContentItem>),
}

/// Prompt content item
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PromptContentItem {
    Text { text: String },
    Image { data: String, mime_type: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let request = McpRequest {
            id: MessageId::default(),
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({
                "name": "test_tool",
                "arguments": {}
            })),
            metadata: None,
        };

        let message = McpMessage::Request(request);
        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: McpMessage = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            McpMessage::Request(req) => {
                assert_eq!(req.method, "tools/call");
            }
            _ => panic!("Expected request message"),
        }
    }
}
