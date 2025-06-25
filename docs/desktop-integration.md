# Desktop Integration Guide

This guide covers how to integrate Woolly as an LLM inference engine in desktop applications across different platforms and frameworks.

## Overview

Woolly can be integrated into desktop applications in two primary ways:

1. **Embedded Library**: Link Woolly directly into your application
2. **Local Server**: Run Woolly as a local HTTP/WebSocket server

## Integration Approaches

### 1. Embedded Library Approach

The embedded approach provides the best performance and control by directly linking Woolly into your application.

**Advantages:**
- Direct memory access (no serialization overhead)
- Full control over model lifecycle
- No network latency
- Better security (no exposed ports)
- Single binary distribution

**Disadvantages:**
- Larger application size
- Must handle model loading/unloading
- Platform-specific builds required
- More complex error handling

**Best for:**
- Native applications (Rust, C++, C#)
- Performance-critical applications
- Offline-first applications
- Applications requiring tight integration

### 2. Local Server Approach

Running Woolly as a local server provides flexibility and language independence.

**Advantages:**
- Language agnostic (any language with HTTP client)
- Process isolation
- Easy updates (update server independently)
- Shared between multiple applications
- Simpler error recovery

**Disadvantages:**
- Network overhead (even locally)
- Port management required
- Additional process to manage
- Potential security concerns

**Best for:**
- Web technologies (Electron, Tauri)
- Interpreted languages (Python, JavaScript)
- Multi-application scenarios
- Rapid prototyping

## Memory and Resource Management

### Model Memory Requirements

| Model Size | RAM Required | VRAM (GPU) | Disk Space |
|------------|--------------|------------|------------|
| 7B params  | 4-8 GB      | 6-8 GB     | 4-13 GB    |
| 13B params | 8-16 GB     | 10-13 GB   | 7-26 GB    |
| 30B params | 16-32 GB    | 20-30 GB   | 15-60 GB   |
| 70B params | 32-64 GB    | 40-70 GB   | 35-140 GB  |

*Note: Requirements vary based on quantization level*

### Memory Management Strategies

```rust
// Example: Dynamic memory management
use woolly_core::{Engine, EngineConfig};

impl DesktopApp {
    fn configure_for_system(&self) -> EngineConfig {
        let total_ram = sys_info::mem_info().unwrap().total;
        let available_ram = sys_info::mem_info().unwrap().avail;
        
        EngineConfig::builder()
            .max_memory(available_ram * 0.7) // Use 70% of available RAM
            .enable_mmap(total_ram > 16_000_000) // Enable mmap for >16GB systems
            .cache_size(1024 * 1024 * 512) // 512MB cache
            .build()
    }
}
```

## Platform-Specific Considerations

### Windows

**Key Considerations:**
- Use Windows Memory Manager for large allocations
- Handle antivirus false positives
- Code signing required for distribution
- Consider UWP sandboxing restrictions

**Build Configuration:**
```toml
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-feature=+crt-static"]

[dependencies.windows]
version = "0.48"
features = ["Win32_System_Memory", "Win32_System_Threading"]
```

### macOS

**Key Considerations:**
- App notarization required
- Hardened runtime restrictions
- Metal Performance Shaders for GPU acceleration
- Universal binary support (Intel + Apple Silicon)

**Entitlements (entitlements.plist):**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.device.camera</key>
    <false/>
    <key>com.apple.security.device.microphone</key>
    <false/>
</dict>
</plist>
```

### Linux

**Key Considerations:**
- Various display servers (X11, Wayland)
- Distribution-specific packaging
- AppImage/Flatpak for universal distribution
- GPU driver compatibility

**Desktop Entry (woolly-app.desktop):**
```ini
[Desktop Entry]
Name=Woolly AI Assistant
Comment=Local AI-powered desktop assistant
Exec=/usr/bin/woolly-desktop
Icon=woolly-desktop
Terminal=false
Type=Application
Categories=Utility;AI;
```

## Security and Sandboxing

### Security Best Practices

1. **Model Verification**
   ```rust
   use sha2::{Sha256, Digest};
   
   fn verify_model(path: &Path, expected_hash: &str) -> Result<()> {
       let mut file = File::open(path)?;
       let mut hasher = Sha256::new();
       io::copy(&mut file, &mut hasher)?;
       let result = format!("{:x}", hasher.finalize());
       
       if result != expected_hash {
           return Err(Error::InvalidModelHash);
       }
       Ok(())
   }
   ```

2. **Sandboxed Execution**
   - Use OS-level sandboxing (AppArmor, SELinux)
   - Restrict file system access
   - Disable network access for offline models
   - Use separate process for inference

3. **Secure Model Storage**
   ```rust
   #[cfg(target_os = "macos")]
   fn get_secure_model_dir() -> PathBuf {
       dirs::data_local_dir()
           .unwrap()
           .join("YourApp")
           .join("Models")
   }
   
   #[cfg(target_os = "windows")]
   fn get_secure_model_dir() -> PathBuf {
       dirs::data_local_dir()
           .unwrap()
           .join("YourApp")
           .join("Models")
   }
   
   #[cfg(target_os = "linux")]
   fn get_secure_model_dir() -> PathBuf {
       dirs::data_dir()
           .unwrap()
           .join("your-app")
           .join("models")
   }
   ```

### Application Sandboxing

**macOS Sandbox Configuration:**
```xml
(version 1)
(deny default)
(allow file-read* file-write*
    (home-subpath "/Library/Application Support/YourApp"))
(allow file-read*
    (home-subpath "/Library/Application Support/YourApp/Models"))
(allow mach-lookup
    (global-name "com.apple.coreservices.launchservicesd"))
```

**Windows UWP Capabilities:**
```xml
<Package>
  <Capabilities>
    <Capability Name="broadFileSystemAccess" />
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>
</Package>
```

## Integration Patterns

### 1. Singleton Pattern for Embedded Use
```rust
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;

static ENGINE: Lazy<Arc<Mutex<Engine>>> = Lazy::new(|| {
    let config = EngineConfig::default();
    let engine = Engine::new(config).expect("Failed to create engine");
    Arc::new(Mutex::new(engine))
});

pub fn get_engine() -> Arc<Mutex<Engine>> {
    ENGINE.clone()
}
```

### 2. Service Pattern for Server Mode
```rust
use tokio::task::spawn_blocking;

pub struct WoollyService {
    client: reqwest::Client,
    base_url: String,
}

impl WoollyService {
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let response = self.client
            .post(&format!("{}/v1/completions", self.base_url))
            .json(&json!({
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.7
            }))
            .send()
            .await?;
            
        let result: CompletionResponse = response.json().await?;
        Ok(result.text)
    }
}
```

### 3. Event-Driven Architecture
```rust
use tokio::sync::mpsc;

#[derive(Clone)]
pub enum WoollyEvent {
    ModelLoaded { model_id: String },
    TokenGenerated { token: String },
    GenerationComplete { text: String },
    Error { message: String },
}

pub struct EventDrivenEngine {
    engine: Engine,
    event_tx: mpsc::Sender<WoollyEvent>,
}

impl EventDrivenEngine {
    pub async fn generate_stream(&self, prompt: &str) {
        let tx = self.event_tx.clone();
        
        spawn_blocking(move || {
            // Stream tokens as they're generated
            for token in engine.generate_tokens(prompt) {
                let _ = tx.blocking_send(WoollyEvent::TokenGenerated { 
                    token: token.to_string() 
                });
            }
        });
    }
}
```

## Framework-Specific Integration

### Electron
See [Electron Quick Start Example](examples/electron-quick-start.md)

### Tauri
See [Tauri Native Integration](examples/tauri-integration.md)

### Flutter
See [Flutter Desktop Example](examples/flutter-desktop.md)

### .NET (WPF/MAUI)
See [.NET Integration Guide](examples/dotnet-integration.md)

### Python (Tkinter/PyQt)
See [Python Desktop Apps](examples/python-desktop.md)

## Best Practices

1. **Lazy Loading**: Load models only when needed
2. **Background Processing**: Keep UI responsive during inference
3. **Progress Indication**: Show loading/processing states
4. **Error Recovery**: Handle model loading failures gracefully
5. **Update Mechanism**: Allow model updates without app updates
6. **Telemetry**: Track performance and usage (with consent)
7. **Offline Mode**: Design for offline-first operation
8. **Model Caching**: Implement intelligent model caching
9. **Resource Monitoring**: Monitor and limit resource usage
10. **Graceful Degradation**: Fallback options for low-resource systems

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Use quantized models
   - Implement model streaming
   - Add swap space warning

2. **Slow First Inference**
   - Preload models on startup
   - Use model warmup
   - Show loading progress

3. **GPU Not Detected**
   - Check driver versions
   - Verify CUDA/Metal support
   - Fallback to CPU inference

4. **Model Corruption**
   - Implement checksums
   - Add model repair functionality
   - Provide re-download option

## Next Steps

- Review [API Reference](api-reference.md) for detailed API documentation
- Check [Model Management Guide](model-management.md) for model handling
- See [Performance Tuning](desktop-performance.md) for optimization tips
- Explore [Example Desktop Apps](../examples/desktop/) for complete implementations