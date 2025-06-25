# Tauri Native Integration with Woolly

This guide demonstrates how to build a native desktop application using Tauri and Woolly for high-performance, memory-efficient AI capabilities.

## Why Tauri + Woolly?

Tauri is an ideal framework for Woolly integration because:
- Both are written in Rust (seamless integration)
- Native performance with small binary size
- Direct memory sharing (no IPC overhead)
- Secure by default architecture
- Cross-platform with native look and feel

## Project Setup

### 1. Initialize Tauri Project

```bash
cargo create-tauri-app woolly-tauri-app
cd woolly-tauri-app
```

### 2. Add Woolly Dependencies

Update `src-tauri/Cargo.toml`:

```toml
[package]
name = "woolly-tauri-app"
version = "0.1.0"
description = "AI-powered desktop app with Woolly"
authors = ["Your Name"]
edition = "2021"

[build-dependencies]
tauri-build = { version = "1.5", features = [] }

[dependencies]
tauri = { version = "1.5", features = ["shell-open"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.35", features = ["full"] }

# Woolly dependencies
woolly-core = { path = "../../../crates/woolly-core" }
woolly-server = { path = "../../../crates/woolly-server" }

# Additional dependencies
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
once_cell = "1.19"
parking_lot = "0.12"
futures = "0.3"

[features]
default = ["custom-protocol"]
custom-protocol = ["tauri/custom-protocol"]
```

## Complete Implementation

### src-tauri/src/main.rs

```rust
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::sync::Arc;
use parking_lot::Mutex;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use tauri::{Manager, State};
use tokio::sync::mpsc;
use tracing::{info, error};

use woolly_core::{Engine, EngineConfig, Model, Session};
use woolly_server::InferenceRequest;

// Global engine instance
static ENGINE: Lazy<Arc<Mutex<Option<Engine>>>> = Lazy::new(|| {
    Arc::new(Mutex::new(None))
});

// App state
struct AppState {
    model_path: Arc<Mutex<Option<String>>>,
    session: Arc<Mutex<Option<Session>>>,
}

#[derive(Clone, Serialize)]
struct ModelInfo {
    name: String,
    size: u64,
    quantization: String,
    loaded: bool,
}

#[derive(Clone, Serialize)]
struct GenerationProgress {
    token: String,
    total_tokens: usize,
    tokens_per_second: f32,
}

#[derive(Clone, Serialize)]
struct SystemInfo {
    total_memory: u64,
    available_memory: u64,
    cpu_count: usize,
    gpu_available: bool,
    gpu_name: Option<String>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
}

// Initialize the engine
fn initialize_engine() -> Result<Engine, anyhow::Error> {
    let config = EngineConfig::builder()
        .max_batch_size(1)
        .max_sequence_length(2048)
        .enable_gpu(true)
        .build();
    
    Engine::new(config).map_err(|e| anyhow::anyhow!("Failed to create engine: {}", e))
}

// Tauri commands
#[tauri::command]
async fn get_system_info() -> Result<SystemInfo, String> {
    use sysinfo::{System, SystemExt, CpuExt};
    
    let mut sys = System::new_all();
    sys.refresh_memory();
    sys.refresh_cpu();
    
    // Check for GPU
    let (gpu_available, gpu_name) = detect_gpu();
    
    Ok(SystemInfo {
        total_memory: sys.total_memory(),
        available_memory: sys.available_memory(),
        cpu_count: sys.cpus().len(),
        gpu_available,
        gpu_name,
    })
}

#[tauri::command]
async fn list_models(app_handle: tauri::AppHandle) -> Result<Vec<ModelInfo>, String> {
    let models_dir = app_handle
        .path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app data dir")?
        .join("models");
    
    let mut models = Vec::new();
    
    if models_dir.exists() {
        let entries = std::fs::read_dir(&models_dir)
            .map_err(|e| format!("Failed to read models directory: {}", e))?;
        
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                let metadata = std::fs::metadata(&path)
                    .map_err(|e| format!("Failed to get file metadata: {}", e))?;
                
                let name = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                
                // Check if this model is currently loaded
                let loaded = ENGINE.lock()
                    .as_ref()
                    .map(|e| e.current_model_name() == Some(&name))
                    .unwrap_or(false);
                
                models.push(ModelInfo {
                    name,
                    size: metadata.len(),
                    quantization: detect_quantization(&path),
                    loaded,
                });
            }
        }
    }
    
    Ok(models)
}

#[tauri::command]
async fn load_model(
    state: State<'_, AppState>,
    model_path: String,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    info!("Loading model: {}", model_path);
    
    // Emit loading started event
    app_handle.emit_all("model-loading-started", &model_path)
        .map_err(|e| format!("Failed to emit event: {}", e))?;
    
    // Initialize engine if needed
    let mut engine_guard = ENGINE.lock();
    if engine_guard.is_none() {
        match initialize_engine() {
            Ok(engine) => {
                *engine_guard = Some(engine);
            }
            Err(e) => {
                error!("Failed to initialize engine: {}", e);
                return Err(format!("Failed to initialize engine: {}", e));
            }
        }
    }
    
    // Load the model
    if let Some(engine) = engine_guard.as_mut() {
        match engine.load_model(&model_path) {
            Ok(()) => {
                // Update app state
                *state.model_path.lock() = Some(model_path.clone());
                
                // Create a new session
                let session = engine.create_session()
                    .map_err(|e| format!("Failed to create session: {}", e))?;
                *state.session.lock() = Some(session);
                
                // Emit success event
                app_handle.emit_all("model-loaded", &model_path)
                    .map_err(|e| format!("Failed to emit event: {}", e))?;
                
                info!("Model loaded successfully");
                Ok("Model loaded successfully".to_string())
            }
            Err(e) => {
                error!("Failed to load model: {}", e);
                Err(format!("Failed to load model: {}", e))
            }
        }
    } else {
        Err("Engine not initialized".to_string())
    }
}

#[tauri::command]
async fn generate_completion(
    state: State<'_, AppState>,
    request: GenerateRequest,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    let session_guard = state.session.lock();
    let session = session_guard.as_ref()
        .ok_or("No session available. Please load a model first.")?;
    
    let max_tokens = request.max_tokens.unwrap_or(200);
    let temperature = request.temperature.unwrap_or(0.7);
    let top_p = request.top_p.unwrap_or(0.9);
    let stream = request.stream.unwrap_or(false);
    
    if stream {
        // Handle streaming generation
        let (tx, mut rx) = mpsc::channel(100);
        let session_clone = session.clone();
        let prompt_clone = request.prompt.clone();
        
        // Spawn generation task
        tokio::spawn(async move {
            let result = session_clone.generate_stream(
                &prompt_clone,
                max_tokens,
                temperature,
                top_p,
                tx,
            ).await;
            
            if let Err(e) = result {
                error!("Generation error: {}", e);
            }
        });
        
        // Forward tokens to frontend
        tokio::spawn(async move {
            let mut total_tokens = 0;
            let start_time = std::time::Instant::now();
            let mut full_response = String::new();
            
            while let Some(token) = rx.recv().await {
                total_tokens += 1;
                full_response.push_str(&token);
                
                let elapsed = start_time.elapsed().as_secs_f32();
                let tokens_per_second = total_tokens as f32 / elapsed.max(0.001);
                
                let progress = GenerationProgress {
                    token,
                    total_tokens,
                    tokens_per_second,
                };
                
                if let Err(e) = app_handle.emit_all("generation-progress", &progress) {
                    error!("Failed to emit progress: {}", e);
                }
            }
            
            // Emit completion event
            if let Err(e) = app_handle.emit_all("generation-complete", &full_response) {
                error!("Failed to emit completion: {}", e);
            }
        });
        
        Ok("Streaming started".to_string())
    } else {
        // Non-streaming generation
        let response = session.generate(
            &request.prompt,
            max_tokens,
            temperature,
            top_p,
        ).await
        .map_err(|e| format!("Generation failed: {}", e))?;
        
        Ok(response)
    }
}

#[tauri::command]
async fn chat_completion(
    state: State<'_, AppState>,
    messages: Vec<ChatMessage>,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    let session_guard = state.session.lock();
    let session = session_guard.as_ref()
        .ok_or("No session available. Please load a model first.")?;
    
    // Convert messages to prompt
    let prompt = format_chat_prompt(&messages);
    
    // Generate response
    let request = GenerateRequest {
        prompt,
        max_tokens: Some(500),
        temperature: Some(0.7),
        top_p: Some(0.9),
        stream: Some(true),
    };
    
    generate_completion(state, request, app_handle).await
}

#[tauri::command]
async fn cancel_generation(state: State<'_, AppState>) -> Result<(), String> {
    let session_guard = state.session.lock();
    if let Some(session) = session_guard.as_ref() {
        session.cancel_generation();
        Ok(())
    } else {
        Err("No active session".to_string())
    }
}

#[tauri::command]
async fn download_model(
    url: String,
    model_name: String,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    use futures::StreamExt;
    
    let models_dir = app_handle
        .path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app data dir")?
        .join("models");
    
    // Ensure models directory exists
    std::fs::create_dir_all(&models_dir)
        .map_err(|e| format!("Failed to create models directory: {}", e))?;
    
    let model_path = models_dir.join(&model_name);
    
    // Download with progress
    let client = reqwest::Client::new();
    let response = client.get(&url)
        .send()
        .await
        .map_err(|e| format!("Failed to start download: {}", e))?;
    
    let total_size = response.content_length().unwrap_or(0);
    
    let mut file = tokio::fs::File::create(&model_path)
        .await
        .map_err(|e| format!("Failed to create file: {}", e))?;
    
    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download error: {}", e))?;
        
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
            .await
            .map_err(|e| format!("Failed to write chunk: {}", e))?;
        
        downloaded += chunk.len() as u64;
        
        let progress = json!({
            "downloaded": downloaded,
            "total": total_size,
            "percent": (downloaded as f64 / total_size as f64) * 100.0
        });
        
        app_handle.emit_all("download-progress", &progress)
            .map_err(|e| format!("Failed to emit progress: {}", e))?;
    }
    
    Ok(model_path.to_string_lossy().to_string())
}

// Helper functions
fn detect_gpu() -> (bool, Option<String>) {
    #[cfg(target_os = "macos")]
    {
        // Check for Metal support
        (true, Some("Apple Metal".to_string()))
    }
    
    #[cfg(target_os = "windows")]
    {
        // Check for CUDA/DirectML
        use wmi::{COMLibrary, WMIConnection};
        
        let com_con = COMLibrary::new().ok()?;
        let wmi_con = WMIConnection::new(com_con.into()).ok()?;
        
        let results: Vec<std::collections::HashMap<String, wmi::Variant>> = 
            wmi_con.raw_query("SELECT Name FROM Win32_VideoController").ok()?;
        
        for gpu in results {
            if let Some(name) = gpu.get("Name").and_then(|v| v.as_str()) {
                if name.contains("NVIDIA") || name.contains("AMD") {
                    return (true, Some(name.to_string()));
                }
            }
        }
        
        (false, None)
    }
    
    #[cfg(target_os = "linux")]
    {
        // Check for CUDA/ROCm
        use std::process::Command;
        
        let nvidia_check = Command::new("nvidia-smi")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        
        if nvidia_check {
            (true, Some("NVIDIA CUDA".to_string()))
        } else {
            let amd_check = Command::new("rocm-smi")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            
            if amd_check {
                (true, Some("AMD ROCm".to_string()))
            } else {
                (false, None)
            }
        }
    }
}

fn detect_quantization(path: &std::path::Path) -> String {
    let filename = path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    
    if filename.contains("q4_0") { "Q4_0".to_string() }
    else if filename.contains("q4_1") { "Q4_1".to_string() }
    else if filename.contains("q5_0") { "Q5_0".to_string() }
    else if filename.contains("q5_1") { "Q5_1".to_string() }
    else if filename.contains("q8_0") { "Q8_0".to_string() }
    else if filename.contains("f16") { "F16".to_string() }
    else if filename.contains("f32") { "F32".to_string() }
    else { "Unknown".to_string() }
}

fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    
    for message in messages {
        match message.role.as_str() {
            "system" => {
                prompt.push_str(&format!("System: {}\n\n", message.content));
            }
            "user" => {
                prompt.push_str(&format!("User: {}\n\n", message.content));
            }
            "assistant" => {
                prompt.push_str(&format!("Assistant: {}\n\n", message.content));
            }
            _ => {}
        }
    }
    
    prompt.push_str("Assistant: ");
    prompt
}

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    
    // Create app state
    let app_state = AppState {
        model_path: Arc::new(Mutex::new(None)),
        session: Arc::new(Mutex::new(None)),
    };
    
    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            get_system_info,
            list_models,
            load_model,
            generate_completion,
            chat_completion,
            cancel_generation,
            download_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### src/App.tsx - React Frontend

```tsx
import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';
import { appDataDir } from '@tauri-apps/api/path';
import { open } from '@tauri-apps/api/dialog';
import './App.css';

interface ModelInfo {
  name: string;
  size: number;
  quantization: string;
  loaded: boolean;
}

interface SystemInfo {
  total_memory: number;
  available_memory: number;
  cpu_count: number;
  gpu_available: boolean;
  gpu_name?: string;
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface GenerationProgress {
  token: string;
  total_tokens: number;
  tokens_per_second: number;
}

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentResponse, setCurrentResponse] = useState('');
  const [stats, setStats] = useState({ tokens: 0, tokensPerSecond: 0 });
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const unlistenProgress = useRef<Function | null>(null);
  const unlistenComplete = useRef<Function | null>(null);

  useEffect(() => {
    // Load system info
    loadSystemInfo();
    
    // Load available models
    loadModels();
    
    // Set up event listeners
    setupEventListeners();
    
    return () => {
      // Cleanup listeners
      if (unlistenProgress.current) unlistenProgress.current();
      if (unlistenComplete.current) unlistenComplete.current();
    };
  }, []);

  useEffect(() => {
    // Scroll to bottom when messages update
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentResponse]);

  const setupEventListeners = async () => {
    // Listen for generation progress
    unlistenProgress.current = await listen<GenerationProgress>(
      'generation-progress',
      (event) => {
        setCurrentResponse(prev => prev + event.payload.token);
        setStats({
          tokens: event.payload.total_tokens,
          tokensPerSecond: event.payload.tokens_per_second
        });
      }
    );
    
    // Listen for generation complete
    unlistenComplete.current = await listen<string>(
      'generation-complete',
      (event) => {
        const finalResponse = event.payload;
        setMessages(prev => [
          ...prev,
          { role: 'assistant', content: finalResponse }
        ]);
        setCurrentResponse('');
        setIsGenerating(false);
        setStats({ tokens: 0, tokensPerSecond: 0 });
      }
    );
  };

  const loadSystemInfo = async () => {
    try {
      const info = await invoke<SystemInfo>('get_system_info');
      setSystemInfo(info);
    } catch (error) {
      console.error('Failed to load system info:', error);
    }
  };

  const loadModels = async () => {
    try {
      const modelList = await invoke<ModelInfo[]>('list_models');
      setModels(modelList);
      
      // Select loaded model if any
      const loadedModel = modelList.find(m => m.loaded);
      if (loadedModel) {
        setSelectedModel(loadedModel.name);
      }
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  };

  const handleLoadModel = async () => {
    if (!selectedModel) return;
    
    try {
      const dataDir = await appDataDir();
      const modelPath = `${dataDir}/models/${selectedModel}.gguf`;
      
      await invoke('load_model', { modelPath });
      await loadModels(); // Refresh model list
    } catch (error) {
      console.error('Failed to load model:', error);
      alert(`Failed to load model: ${error}`);
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isGenerating) return;
    
    const userMessage: ChatMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsGenerating(true);
    setCurrentResponse('');
    
    try {
      // Include recent context
      const contextMessages = [...messages.slice(-10), userMessage];
      
      await invoke('chat_completion', {
        messages: contextMessages
      });
    } catch (error) {
      console.error('Generation failed:', error);
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `Error: ${error}` }
      ]);
      setIsGenerating(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleFileUpload = async () => {
    const selected = await open({
      multiple: false,
      filters: [{
        name: 'GGUF Model',
        extensions: ['gguf']
      }]
    });
    
    if (selected && typeof selected === 'string') {
      try {
        await invoke('load_model', { modelPath: selected });
        await loadModels();
      } catch (error) {
        console.error('Failed to load model:', error);
        alert(`Failed to load model: ${error}`);
      }
    }
  };

  const formatBytes = (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  return (
    <div className="container">
      <header className="header">
        <h1>Woolly AI Assistant</h1>
        <div className="header-controls">
          <select 
            value={selectedModel} 
            onChange={(e) => setSelectedModel(e.target.value)}
            className="model-select"
          >
            <option value="">Select a model...</option>
            {models.map(model => (
              <option key={model.name} value={model.name}>
                {model.name} ({formatBytes(model.size)}) - {model.quantization}
                {model.loaded && ' ✓'}
              </option>
            ))}
          </select>
          <button onClick={handleLoadModel} disabled={!selectedModel}>
            Load Model
          </button>
          <button onClick={handleFileUpload}>
            Upload Model
          </button>
        </div>
      </header>

      <div className="system-info">
        {systemInfo && (
          <>
            <span>Memory: {formatBytes(systemInfo.available_memory)} / {formatBytes(systemInfo.total_memory)}</span>
            <span>CPU: {systemInfo.cpu_count} cores</span>
            {systemInfo.gpu_available && (
              <span>GPU: {systemInfo.gpu_name || 'Available'}</span>
            )}
            {isGenerating && (
              <span>Speed: {stats.tokensPerSecond.toFixed(1)} tok/s</span>
            )}
          </>
        )}
      </div>

      <div className="chat-container">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-role">
              {msg.role === 'user' ? '👤' : '🤖'}
            </div>
            <div className="message-content">
              {msg.content}
            </div>
          </div>
        ))}
        
        {isGenerating && currentResponse && (
          <div className="message assistant">
            <div className="message-role">🤖</div>
            <div className="message-content">
              {currentResponse}
              <span className="cursor">▊</span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          disabled={isGenerating || !models.some(m => m.loaded)}
          className="message-input"
          rows={3}
        />
        <button 
          onClick={handleSendMessage}
          disabled={isGenerating || !input.trim() || !models.some(m => m.loaded)}
          className="send-button"
        >
          {isGenerating ? 'Generating...' : 'Send'}
        </button>
      </div>
    </div>
  );
}

export default App;
```

### src/App.css - Styling

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

:root {
  --primary: #2563eb;
  --primary-dark: #1d4ed8;
  --secondary: #10b981;
  --background: #0f172a;
  --surface: #1e293b;
  --surface-light: #334155;
  --text: #e2e8f0;
  --text-secondary: #94a3b8;
  --border: #334155;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  background: var(--background);
  color: var(--text);
  height: 100vh;
  overflow: hidden;
}

.container {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.header {
  background: var(--surface);
  padding: 1rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

h1 {
  font-size: 1.5rem;
  font-weight: 600;
  background: linear-gradient(to right, var(--primary), var(--secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.header-controls {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.model-select {
  padding: 0.5rem;
  background: var(--surface-light);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  min-width: 200px;
}

button {
  padding: 0.5rem 1rem;
  background: var(--primary);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: background 0.2s;
}

button:hover:not(:disabled) {
  background: var(--primary-dark);
}

button:disabled {
  background: var(--surface-light);
  cursor: not-allowed;
  opacity: 0.5;
}

.system-info {
  background: var(--surface);
  padding: 0.5rem 1rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  gap: 2rem;
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.message {
  display: flex;
  gap: 0.75rem;
  animation: fadeIn 0.3s;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message.user {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.message-role {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  flex-shrink: 0;
}

.message.user .message-role {
  background: var(--primary);
}

.message.assistant .message-role {
  background: var(--secondary);
}

.message-content {
  max-width: 70%;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  background: var(--surface);
  line-height: 1.5;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.message.user .message-content {
  background: var(--primary);
}

.cursor {
  animation: blink 1s infinite;
  margin-left: 2px;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

.input-container {
  padding: 1rem;
  background: var(--surface);
  border-top: 1px solid var(--border);
  display: flex;
  gap: 0.75rem;
}

.message-input {
  flex: 1;
  padding: 0.75rem;
  background: var(--surface-light);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  resize: none;
  font-family: inherit;
  font-size: 1rem;
}

.message-input:focus {
  outline: none;
  border-color: var(--primary);
}

.send-button {
  align-self: flex-end;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: var(--surface);
}

::-webkit-scrollbar-thumb {
  background: var(--surface-light);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--border);
}
```

## Building and Distribution

### Development
```bash
# Install dependencies
cd woolly-tauri-app
npm install

# Run in development mode
npm run tauri dev
```

### Building for Production
```bash
# Build for current platform
npm run tauri build

# The built app will be in src-tauri/target/release/bundle/
```

### Platform-Specific Configuration

#### macOS
Add to `tauri.conf.json`:
```json
{
  "tauri": {
    "bundle": {
      "identifier": "com.yourcompany.woolly",
      "icon": ["icons/icon.icns"],
      "macOS": {
        "entitlements": "./entitlements.plist",
        "exceptionDomain": "",
        "frameworks": [],
        "minimumSystemVersion": "10.15"
      }
    }
  }
}
```

#### Windows
Add to `tauri.conf.json`:
```json
{
  "tauri": {
    "bundle": {
      "windows": {
        "certificateThumbprint": null,
        "digestAlgorithm": "sha256",
        "timestampUrl": "",
        "wix": {
          "language": "en-US"
        }
      }
    }
  }
}
```

#### Linux
Add to `tauri.conf.json`:
```json
{
  "tauri": {
    "bundle": {
      "deb": {
        "depends": ["libwebkit2gtk-4.0-37", "libgtk-3-0"]
      },
      "appimage": {
        "bundleMediaFramework": true
      }
    }
  }
}
```

## Advanced Features

### 1. Model Quantization UI
```rust
#[tauri::command]
async fn quantize_model(
    input_path: String,
    output_path: String,
    quantization: String,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    use woolly_core::quantization::{quantize_model, QuantizationType};
    
    let q_type = match quantization.as_str() {
        "Q4_0" => QuantizationType::Q4_0,
        "Q4_1" => QuantizationType::Q4_1,
        "Q5_0" => QuantizationType::Q5_0,
        "Q5_1" => QuantizationType::Q5_1,
        "Q8_0" => QuantizationType::Q8_0,
        _ => return Err("Invalid quantization type".to_string()),
    };
    
    tokio::task::spawn_blocking(move || {
        quantize_model(&input_path, &output_path, q_type, |progress| {
            let _ = app_handle.emit_all("quantization-progress", &progress);
        })
    })
    .await
    .map_err(|e| format!("Quantization failed: {}", e))?
    .map_err(|e| format!("Quantization error: {}", e))
}
```

### 2. Plugin System
```rust
use tauri::plugin::{Builder, TauriPlugin};

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("woolly-plugin")
        .invoke_handler(tauri::generate_handler![
            plugin_command
        ])
        .build()
}

#[tauri::command]
async fn plugin_command() -> Result<String, String> {
    Ok("Plugin response".to_string())
}
```

### 3. Custom Protocol Handler
```rust
// In main.rs
.register_uri_scheme_protocol("woolly", |app, request| {
    let response = match request.uri().path() {
        "/model" => {
            // Serve model files
            tauri::http::ResponseBuilder::new()
                .header("Content-Type", "application/octet-stream")
                .body(model_data)
        }
        _ => {
            tauri::http::ResponseBuilder::new()
                .status(404)
                .body(Vec::new())
        }
    };
    response
})
```

## Performance Optimization

### 1. Lazy Model Loading
```rust
use once_cell::sync::Lazy;
use lru::LruCache;

static MODEL_CACHE: Lazy<Mutex<LruCache<String, Arc<Model>>>> = 
    Lazy::new(|| Mutex::new(LruCache::new(3)));
```

### 2. Background Processing
```rust
use tauri::async_runtime::spawn;

#[tauri::command]
async fn background_task(app_handle: tauri::AppHandle) -> Result<(), String> {
    spawn(async move {
        // Long-running task
        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;
            
            // Emit status update
            let _ = app_handle.emit_all("background-update", &status);
        }
    });
    
    Ok(())
}
```

### 3. Window State Persistence
```rust
use tauri::{Manager, WindowEvent};

// Save window state on close
.on_window_event(|event| match event.event() {
    WindowEvent::CloseRequested { .. } => {
        let window = event.window();
        if let Ok(size) = window.outer_size() {
            // Save size to config
        }
        if let Ok(position) = window.outer_position() {
            // Save position to config
        }
    }
    _ => {}
})
```

## Security Best Practices

1. **CSP Configuration**
   ```json
   {
     "tauri": {
       "security": {
         "csp": "default-src 'self'; img-src 'self' data: https:; script-src 'self'"
       }
     }
   }
   ```

2. **Allowlist Configuration**
   ```json
   {
     "tauri": {
       "allowlist": {
         "fs": {
           "scope": ["$APPDATA/models/*"]
         },
         "http": {
           "scope": ["https://huggingface.co/*"]
         }
       }
     }
   }
   ```

3. **Input Validation**
   ```rust
   fn validate_model_path(path: &str) -> Result<PathBuf, String> {
       let path = PathBuf::from(path);
       
       // Ensure path is within allowed directory
       let app_data = app_handle.path_resolver().app_data_dir()
           .ok_or("Failed to get app data dir")?;
       
       if !path.starts_with(&app_data) {
           return Err("Invalid model path".to_string());
       }
       
       // Check file extension
       if path.extension() != Some("gguf".as_ref()) {
           return Err("Invalid model format".to_string());
       }
       
       Ok(path)
   }
   ```