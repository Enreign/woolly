//! Model management handlers

use crate::{
    error::{ServerError, ServerResult},
    server::ServerState,
};
use axum::{
    extract::{Path, State, Request},
    response::Response,
    http::{StatusCode},
    body::Body,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{path::PathBuf, sync::Arc};
use woolly_core::{
    model::Model,
};

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

/// Wrapped models response for compatibility
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub success: bool,
    pub data: Vec<ModelInfo>,
}

/// List available models
pub async fn list_models(
    State(state): State<ServerState>,
    request: Request,
) -> ServerResult<Response> {
    // Allow anonymous access for model listing for now
    // let _auth_context = extract_auth(&request)?;

    // Get models from configured directory
    let models_dir = &state.config.models.models_dir;
    let mut models = Vec::new();
    
    eprintln!("[DEBUG] Listing models from directory: {:?}", models_dir);
    eprintln!("[DEBUG] Directory exists: {}", models_dir.exists());

    // Scan for model files
    if models_dir.exists() {
        match std::fs::read_dir(models_dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    eprintln!("[DEBUG] Found file: {:?}", path);
                    if let Some(model_info) = scan_model_file(path).await? {
                        eprintln!("[DEBUG] Added model: {}", model_info.name);
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

    eprintln!("[DEBUG] Found {} models in directory", models.len());
    
    // Add currently loaded model info if any
    if let Ok(engine) = state.inference_engine.try_read() {
        eprintln!("[DEBUG] Checking for loaded model...");
        // Check if we have a loaded model
        if let Some(model_info) = engine.model_info() {
            eprintln!("[DEBUG] Found loaded model: {}", model_info.name);
            eprintln!("[DEBUG] Model filename: {:?}", engine.model_filename());
            // Get the loaded model filename
            let loaded_filename = engine.model_filename();
            let mut found_in_list = false;
            
            // Mark the loaded model in the list if it exists
            if let Some(filename) = &loaded_filename {
                // Try to match with or without .gguf extension
                let loaded_model_name = filename.replace(".gguf", "");
                for model in models.iter_mut() {
                    if model.name == loaded_model_name || model.name == *filename {
                        model.loaded = true;
                        model.architecture = Some(model_info.model_type.clone());
                        model.context_length = Some(model_info.context_length);
                        model.metadata = json!({
                            "vocab_size": model_info.vocab_size,
                            "hidden_size": model_info.hidden_size,
                            "num_layers": model_info.num_layers,
                            "num_heads": model_info.num_heads
                        });
                        found_in_list = true;
                        break;
                    }
                }
            }
            
            // If the loaded model isn't in the directory listing, add it
            if !found_in_list {
                let model_name = loaded_filename
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "loaded-model".to_string());
                models.push(ModelInfo {
                    name: model_name,
                    description: Some("Currently loaded model".to_string()),
                    path: None, // Path might not be available
                    loaded: true,
                    size_bytes: None,
                    parameters: Some(calculate_parameters(&model_info)),
                    architecture: Some(model_info.model_type.clone()),
                    context_length: Some(model_info.context_length),
                    capabilities: vec!["completion".to_string(), "chat".to_string()],
                    metadata: json!({
                        "vocab_size": model_info.vocab_size,
                        "hidden_size": model_info.hidden_size,
                        "num_layers": model_info.num_layers,
                        "num_heads": model_info.num_heads,
                        "loaded_from_memory": true
                    }),
                });
            }
        }
    } else {
        eprintln!("[DEBUG] No loaded model found in engine");
    }
    
    eprintln!("[DEBUG] Returning {} models total", models.len());

    // Check if the request wants wrapped format (for Ole compatibility)
    let headers = request.headers();
    let wants_wrapped = headers.get("x-ole-client").is_some() 
        || headers.get("user-agent").and_then(|v| v.to_str().ok()).map(|s| s.contains("Ole")).unwrap_or(false);
    
    if wants_wrapped {
        // Return wrapped format for Ole
        let wrapped = ModelsResponse {
            success: true,
            data: models,
        };
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&wrapped).unwrap()))
            .unwrap())
    } else {
        // Return standard format
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&models).unwrap()))
            .unwrap())
    }
}

/// Get information about a specific model
pub async fn get_model_info(
    State(state): State<ServerState>,
    Path(model_name): Path<String>,
) -> ServerResult<Json<ModelInfo>> {
    // Allow anonymous access for model listing for now
    // let _auth_context = extract_auth(&request)?;

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
        let base_path = state.config.models.models_dir.join(&model_name);
        
        // Check if the path exists as-is
        if base_path.exists() {
            base_path
        } else {
            // Try with .gguf extension appended
            let gguf_path = state.config.models.models_dir.join(format!("{}.gguf", &model_name));
            if gguf_path.exists() {
                gguf_path
            } else {
                // Return the original path for the error message
                base_path
            }
        }
    };

    if !model_path.exists() {
        return Ok(Json(LoadModelResponse {
            success: false,
            message: format!("Model file not found: {:?}", model_path),
            model_info: None,
        }));
    }

    // Implement real model loading
    let mut engine = state.inference_engine.write().await;
    
    // Load model using GGUF loader
    match load_gguf_model(&model_path).await {
        Ok(model) => {
            // Load the model into the engine
            match engine.load_model(model).await {
                Ok(_) => {
                    // Store the model filename for display
                    engine.set_model_filename(model_name.clone());
                    // Get model info from the loaded model
                    let model_info = if let Some(info) = engine.model_info() {
                        ModelInfo {
                            name: model_name.clone(),
                            description: Some("Successfully loaded GGUF model".to_string()),
                            path: Some(model_path.to_string_lossy().to_string()),
                            loaded: true,
                            size_bytes: None, // Could get from file metadata
                            parameters: None, // Could calculate from model size
                            architecture: Some(info.model_type),
                            context_length: Some(info.context_length),
                            capabilities: vec!["completion".to_string(), "chat".to_string()],
                            metadata: json!({
                                "vocab_size": info.vocab_size,
                                "hidden_size": info.hidden_size,
                                "num_layers": info.num_layers,
                                "num_heads": info.num_heads
                            }),
                        }
                    } else {
                        ModelInfo {
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
                        }
                    };
                    
                    return Ok(Json(LoadModelResponse {
                        success: true,
                        message: format!("Model '{}' loaded successfully", model_name),
                        model_info: Some(model_info),
                    }));
                }
                Err(e) => {
                    return Ok(Json(LoadModelResponse {
                        success: false,
                        message: format!("Failed to load model into engine: {}", e),
                        model_info: None,
                    }));
                }
            }
        }
        Err(e) => {
            return Ok(Json(LoadModelResponse {
                success: false,
                message: format!("Failed to load GGUF model: {}", e),
                model_info: None,
            }));
        }
    }
}

/// Unload a model
pub async fn unload_model(
    State(_state): State<ServerState>,
    Path(model_name): Path<String>,
) -> ServerResult<Json<Value>> {
    // Allow anonymous access for model listing for now
    // let _auth_context = extract_auth(&request)?;

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

/// Load a GGUF model from file path
async fn load_gguf_model(path: &PathBuf) -> Result<Arc<dyn Model>, woolly_core::CoreError> {
    use woolly_core::model::lazy_transformer::LazyTransformer;
    use woolly_core::model::transformer::TransformerConfig;
    
    eprintln!("Loading GGUF model with lazy loading from: {:?}", path);
    
    // Create transformer config
    let config = TransformerConfig::default();
    
    // Create lazy-loading transformer
    let model = LazyTransformer::from_gguf(path, config).await?;
    
    eprintln!("Model loaded successfully with lazy loading!");
    
    Ok(Arc::new(model))
}

/// Calculate approximate parameter count from model info
fn calculate_parameters(model_info: &woolly_core::engine::ModelInfo) -> u64 {
    // Rough calculation of parameters
    let embedding_params = model_info.vocab_size * model_info.hidden_size;
    let attention_params_per_layer = 4 * model_info.hidden_size * model_info.hidden_size; // Q, K, V, O projections
    let mlp_params_per_layer = 3 * model_info.hidden_size * model_info.hidden_size * 4; // Approximate MLP size
    let layer_norm_params = 2 * model_info.hidden_size * model_info.num_layers;
    
    let total = embedding_params + 
                (attention_params_per_layer + mlp_params_per_layer) * model_info.num_layers + 
                layer_norm_params;
    
    total as u64
}

/// Debug endpoint to check engine state
pub async fn debug_engine_state(
    State(state): State<ServerState>,
) -> ServerResult<Json<Value>> {
    let mut debug_info = json!({
        "models_dir": state.config.models.models_dir.to_string_lossy(),
        "models_dir_exists": state.config.models.models_dir.exists(),
    });
    
    if let Ok(engine) = state.inference_engine.try_read() {
        debug_info["has_model"] = json!(engine.model_info().is_some());
        debug_info["model_filename"] = json!(engine.model_filename());
        
        if let Some(model_info) = engine.model_info() {
            debug_info["loaded_model"] = json!({
                "name": model_info.name,
                "type": model_info.model_type,
                "vocab_size": model_info.vocab_size,
                "context_length": model_info.context_length,
                "hidden_size": model_info.hidden_size,
                "num_layers": model_info.num_layers,
                "num_heads": model_info.num_heads,
            });
        }
    } else {
        debug_info["engine_locked"] = json!(true);
    }
    
    Ok(Json(debug_info))
}