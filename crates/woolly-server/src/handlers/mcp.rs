//! MCP (Model Context Protocol) handlers

use crate::{
    auth::extract_auth,
    error::ServerResult,
    server::ServerState,
};
use axum::{
    extract::{Path, Request, State},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use woolly_mcp::types::{ToolInfo, ResourceInfo, PromptInfo, ToolCallResponse, ResourceReadResponse, PromptGetResponse};

/// Tool execution request
#[derive(Debug, Deserialize)]
pub struct ExecuteToolRequest {
    pub arguments: Option<Value>,
}

/// Tool execution response  
#[derive(Debug, Serialize)]
pub struct ExecuteToolResponse {
    pub success: bool,
    pub result: Option<ToolCallResponse>,
    pub error: Option<String>,
}

/// Resource request with optional parameters
#[derive(Debug, Deserialize)]
pub struct ResourceRequest {
    pub parameters: Option<HashMap<String, Value>>,
}

/// Prompt request
#[derive(Debug, Deserialize)]
pub struct PromptRequest {
    pub arguments: HashMap<String, Value>,
}

/// List available tools
pub async fn list_tools(
    State(state): State<ServerState>,
    request: Request,
) -> ServerResult<Json<Vec<ToolInfo>>> {
    let _auth_context = extract_auth(&request)?;

    #[cfg(feature = "mcp")]
    {
        let tools = state.mcp_state.list_tools().await?;
        Ok(Json(tools))
    }

    #[cfg(not(feature = "mcp"))]
    {
        Ok(Json(vec![]))
    }
}

/// Execute a tool
pub async fn execute_tool(
    State(state): State<ServerState>,
    Path(tool_name): Path<String>,
    Json(execute_request): Json<ExecuteToolRequest>,
) -> ServerResult<Json<ExecuteToolResponse>> {

    #[cfg(feature = "mcp")]
    {
        let arguments = execute_request.arguments.unwrap_or(json!({}));
        
        match state.mcp_state.execute_tool(&tool_name, arguments).await {
            Ok(result) => Ok(Json(ExecuteToolResponse {
                success: true,
                result: Some(result),
                error: None,
            })),
            Err(e) => Ok(Json(ExecuteToolResponse {
                success: false,
                result: None,
                error: Some(e.to_string()),
            })),
        }
    }

    #[cfg(not(feature = "mcp"))]
    {
        Err(ServerError::Internal("MCP not enabled".to_string()))
    }
}

/// List available resources
pub async fn list_resources(
    State(state): State<ServerState>,
    request: Request,
) -> ServerResult<Json<Vec<ResourceInfo>>> {
    let _auth_context = extract_auth(&request)?;

    #[cfg(feature = "mcp")]
    {
        let resources = state.mcp_state.list_resources().await?;
        Ok(Json(resources))
    }

    #[cfg(not(feature = "mcp"))]
    {
        Ok(Json(vec![]))
    }
}

/// Get a resource
pub async fn get_resource(
    State(state): State<ServerState>,
    Path(resource_path): Path<String>,
    request: Request,
) -> ServerResult<Json<ResourceReadResponse>> {
    let _auth_context = extract_auth(&request)?;

    #[cfg(feature = "mcp")]
    {
        // Reconstruct full path from wildcard capture
        let full_path = format!("/{}", resource_path);
        let response = state.mcp_state.get_resource(&full_path).await?;
        Ok(Json(response))
    }

    #[cfg(not(feature = "mcp"))]
    {
        Err(ServerError::Internal("MCP not enabled".to_string()))
    }
}

/// List available prompts
pub async fn list_prompts(
    State(state): State<ServerState>,
    request: Request,
) -> ServerResult<Json<Vec<PromptInfo>>> {
    let _auth_context = extract_auth(&request)?;

    #[cfg(feature = "mcp")]
    {
        let prompts = state.mcp_state.list_prompts().await?;
        Ok(Json(prompts))
    }

    #[cfg(not(feature = "mcp"))]
    {
        Ok(Json(vec![]))
    }
}

/// Get a prompt
pub async fn get_prompt(
    State(state): State<ServerState>,
    Path(prompt_name): Path<String>,
    Json(prompt_request): Json<PromptRequest>,
) -> ServerResult<Json<PromptGetResponse>> {

    #[cfg(feature = "mcp")]
    {
        let response = state.mcp_state.get_prompt(&prompt_name, prompt_request.arguments).await?;
        Ok(Json(response))
    }

    #[cfg(not(feature = "mcp"))]
    {
        Err(ServerError::Internal("MCP not enabled".to_string()))
    }
}

/// Get MCP server capabilities
pub async fn get_capabilities(
    State(state): State<ServerState>,
    request: Request,
) -> ServerResult<Json<Value>> {
    let _auth_context = extract_auth(&request)?;

    #[cfg(feature = "mcp")]
    {
        let capabilities = state.mcp_state.get_capabilities().await?;
        Ok(Json(serde_json::to_value(capabilities)?))
    }

    #[cfg(not(feature = "mcp"))]
    {
        Ok(Json(json!({
            "mcp": false,
            "message": "MCP not enabled"
        })))
    }
}

/// Initialize MCP connection (for HTTP clients)
pub async fn initialize_mcp(
    State(state): State<ServerState>,
    request: Request,
    Json(_init_request): Json<Value>,
) -> ServerResult<Json<Value>> {
    let _auth_context = extract_auth(&request)?;

    #[cfg(feature = "mcp")]
    {
        // TODO: Handle MCP initialization for HTTP clients
        // This would be used for clients that don't use WebSocket
        Ok(Json(json!({
            "success": true,
            "protocol_version": "0.1.0",
            "server_info": {
                "name": "woolly-server",
                "version": crate::VERSION
            },
            "capabilities": state.mcp_state.get_capabilities().await?
        })))
    }

    #[cfg(not(feature = "mcp"))]
    {
        Err(ServerError::Internal("MCP not enabled".to_string()))
    }
}

/// Health check for MCP subsystem
pub async fn mcp_health(
    State(state): State<ServerState>,
    request: Request,
) -> ServerResult<Json<Value>> {
    let _auth_context = extract_auth(&request)?;

    #[cfg(feature = "mcp")]
    {
        let is_initialized = state.mcp_state.is_initialized();
        let status = if is_initialized { "healthy" } else { "not_initialized" };
        
        Ok(Json(json!({
            "status": status,
            "initialized": is_initialized,
            "protocol_version": "0.1.0",
            "timestamp": chrono::Utc::now().to_rfc3339()
        })))
    }

    #[cfg(not(feature = "mcp"))]
    {
        Ok(Json(json!({
            "status": "disabled",
            "message": "MCP feature not enabled",
            "timestamp": chrono::Utc::now().to_rfc3339()
        })))
    }
}