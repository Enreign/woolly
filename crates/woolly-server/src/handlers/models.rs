//! Model management handlers

use crate::{
    auth::extract_auth,
    error::{ServerError, ServerResult},
    server::ServerState,
};
use axum::{
    extract::{Path, Request, State},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;

/// Model information
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub description: Option<String>,
    pub path: Option<String>,
    pub loaded: bool,
    pub size_bytes: Option<u64>,
    pub parameters: Option<u64>,
    pub architecture: Option<String>,
    pub context_length: Option<usize>,
    pub capabilities: Vec<String>,
    pub metadata: Value,
}

/// Load model request
#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub path: Option<String>,
    pub config: Option<Value>,
}

/// Load model response
#[derive(Debug, Serialize)]
pub struct LoadModelResponse {
    pub success: bool,
    pub message: String,
    pub model_info: Option<ModelInfo>,
}

/// List available models
pub async fn list_models(
    State(state): State<ServerState>,
    request: Request,
) -> ServerResult<Json<Vec<ModelInfo>>> {
    let _auth_context = extract_auth(&request)?;

    // Get models from configured directory
    let models_dir = &state.config.models.models_dir;
    let mut models = Vec::new();

    // Scan for model files
    if models_dir.exists() {
        match std::fs::read_dir(models_dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    if let Some(model_info) = scan_model_file(entry.path()).await? {
                        models.push(model_info);
                    }
                }
            }
            Err(e) => {
                return Err(ServerError::Internal(format!(
                    "Failed to read models directory: {}",
                    e
                )));
            }
        }
    }

    // Add currently loaded model info if any
    if let Ok(_engine) = state.inference_engine.try_read() {
        // TODO: Get actual loaded model info from engine
        // For now, add a placeholder
        models.push(ModelInfo {
            name: "loaded-model".to_string(),
            description: Some("Currently loaded model".to_string()),
            path: None,
            loaded: true,
            size_bytes: None,
            parameters: None,
            architecture: Some("transformer".to_string()),
            context_length: Some(4096),
            capabilities: vec!["completion".to_string(), "chat".to_string()],
            metadata: json!({}),
        });
    }

    Ok(Json(models))
}

/// Get information about a specific model
pub async fn get_model_info(
    State(state): State<ServerState>,
    Path(model_name): Path<String>,
    request: Request,
) -> ServerResult<Json<ModelInfo>> {
    let _auth_context = extract_auth(&request)?;

    // Check if it's the currently loaded model
    if model_name == "loaded" || model_name == "current" {
        if let Ok(_engine) = state.inference_engine.try_read() {
            // TODO: Get actual model info from engine
            return Ok(Json(ModelInfo {
                name: "loaded-model".to_string(),
                description: Some("Currently loaded model".to_string()),
                path: None,
                loaded: true,
                size_bytes: None,
                parameters: None,
                architecture: Some("transformer".to_string()),
                context_length: Some(4096),
                capabilities: vec!["completion".to_string(), "chat".to_string()],
                metadata: json!({}),
            }));
        } else {
            return Err(ServerError::ModelNotFound("No model currently loaded".to_string()));
        }
    }

    // Look for model file
    let models_dir = &state.config.models.models_dir;
    let model_path = models_dir.join(&model_name);

    if let Some(model_info) = scan_model_file(model_path).await? {
        Ok(Json(model_info))
    } else {
        Err(ServerError::ModelNotFound(format!("Model '{}' not found", model_name)))
    }
}

/// Load a model
pub async fn load_model(
    State(state): State<ServerState>,
    Path(model_name): Path<String>,
    Json(load_request): Json<LoadModelRequest>,
) -> ServerResult<Json<LoadModelResponse>> {

    // Determine model path
    let model_path = if let Some(path) = load_request.path {
        PathBuf::from(path)
    } else {
        state.config.models.models_dir.join(&model_name)
    };

    if !model_path.exists() {
        return Ok(Json(LoadModelResponse {
            success: false,
            message: format!("Model file not found: {:?}", model_path),
            model_info: None,
        }));
    }

    // TODO: Implement actual model loading
    // For now, return success
    let model_info = ModelInfo {
        name: model_name.clone(),
        description: Some("Loaded model".to_string()),
        path: Some(model_path.to_string_lossy().to_string()),
        loaded: true,
        size_bytes: None,
        parameters: None,
        architecture: Some("transformer".to_string()),
        context_length: Some(4096),
        capabilities: vec!["completion".to_string(), "chat".to_string()],
        metadata: json!({}),
    };

    Ok(Json(LoadModelResponse {
        success: true,
        message: format!("Model '{}' loaded successfully", model_name),
        model_info: Some(model_info),
    }))
}

/// Unload a model
pub async fn unload_model(
    State(_state): State<ServerState>,
    Path(model_name): Path<String>,
    request: Request,
) -> ServerResult<Json<Value>> {
    let _auth_context = extract_auth(&request)?;

    // TODO: Implement actual model unloading
    // For now, return success
    
    Ok(Json(json!({
        "success": true,
        "message": format!("Model '{}' unloaded successfully", model_name)
    })))
}

/// Scan a model file and extract information
async fn scan_model_file(path: PathBuf) -> ServerResult<Option<ModelInfo>> {
    if !path.exists() {
        return Ok(None);
    }

    let metadata = std::fs::metadata(&path)
        .map_err(|e| ServerError::Internal(format!("Failed to read file metadata: {}", e)))?;

    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    // Check if it's a supported model format
    match extension.to_lowercase().as_str() {
        "gguf" | "bin" | "safetensors" => {
            Ok(Some(ModelInfo {
                name,
                description: None,
                path: Some(path.to_string_lossy().to_string()),
                loaded: false,
                size_bytes: Some(metadata.len()),
                parameters: None,
                architecture: None,
                context_length: None,
                capabilities: vec!["completion".to_string()],
                metadata: json!({
                    "format": extension,
                    "modified": metadata.modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                }),
            }))
        }
        _ => Ok(None),
    }
}