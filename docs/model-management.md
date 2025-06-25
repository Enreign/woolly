# Model Management Guide

This guide covers how to download, store, manage, and optimize models for desktop applications using Woolly.

## Table of Contents

1. [Model Formats](#model-formats)
2. [Downloading Models](#downloading-models)
3. [Storage Strategies](#storage-strategies)
4. [Model Formats & Quantization](#model-formats--quantization)
5. [Model Switching](#model-switching)
6. [Memory Management](#memory-management)
7. [Model Validation](#model-validation)
8. [Best Practices](#best-practices)

## Model Formats

### GGUF Format

Woolly uses the GGUF (GPT-Generated Unified Format) for model storage, which provides:

- **Efficient Storage**: Optimized binary format
- **Metadata Support**: Model information embedded in file
- **Quantization**: Multiple precision levels
- **Memory Mapping**: Direct file access without full loading
- **Cross-Platform**: Works on all operating systems

### Model Structure

```
model.gguf
├── Header
│   ├── Magic Number (GGUF)
│   ├── Version
│   └── Tensor Count
├── Metadata
│   ├── Architecture
│   ├── Quantization
│   ├── Vocabulary
│   └── Parameters
└── Tensor Data
    ├── Embeddings
    ├── Attention Weights
    └── Feed-Forward Weights
```

## Downloading Models

### Model Sources

#### 1. Hugging Face Hub

```rust
use reqwest;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use futures_util::StreamExt;

pub struct ModelDownloader {
    client: reqwest::Client,
    cache_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            client: reqwest::Client::new(),
            cache_dir,
        }
    }
    
    pub async fn download_from_huggingface(
        &self,
        repo_id: &str,
        filename: &str,
        progress_callback: impl Fn(u64, u64),
    ) -> Result<PathBuf, Error> {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, filename
        );
        
        let response = self.client
            .get(&url)
            .send()
            .await?;
            
        let total_size = response
            .content_length()
            .ok_or_else(|| Error::UnknownSize)?;
            
        let dest_path = self.cache_dir.join(filename);
        let mut file = File::create(&dest_path).await?;
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            progress_callback(downloaded, total_size);
        }
        
        file.flush().await?;
        Ok(dest_path)
    }
}
```

#### 2. Direct URL Download

```rust
pub async fn download_model_direct(
    url: &str,
    dest_path: &Path,
    mut on_progress: impl FnMut(f64),
) -> Result<(), Error> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3600)) // 1 hour timeout
        .build()?;
        
    let mut response = client.get(url).send().await?;
    let total_size = response.content_length();
    
    let mut file = File::create(dest_path).await?;
    let mut downloaded = 0u64;
    
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        
        if let Some(total) = total_size {
            on_progress(downloaded as f64 / total as f64);
        }
    }
    
    Ok(())
}
```

### Download Manager

```rust
use std::collections::HashMap;
use tokio::sync::RwLock;

pub struct DownloadManager {
    downloads: Arc<RwLock<HashMap<String, DownloadStatus>>>,
}

#[derive(Clone)]
pub struct DownloadStatus {
    pub id: String,
    pub url: String,
    pub destination: PathBuf,
    pub progress: f64,
    pub total_bytes: Option<u64>,
    pub downloaded_bytes: u64,
    pub state: DownloadState,
    pub error: Option<String>,
}

#[derive(Clone, PartialEq)]
pub enum DownloadState {
    Pending,
    Downloading,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

impl DownloadManager {
    pub fn new() -> Self {
        Self {
            downloads: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn start_download(
        &self,
        id: String,
        url: String,
        destination: PathBuf,
    ) -> Result<(), Error> {
        let status = DownloadStatus {
            id: id.clone(),
            url: url.clone(),
            destination: destination.clone(),
            progress: 0.0,
            total_bytes: None,
            downloaded_bytes: 0,
            state: DownloadState::Pending,
            error: None,
        };
        
        self.downloads.write().await.insert(id.clone(), status);
        
        let downloads = self.downloads.clone();
        tokio::spawn(async move {
            if let Err(e) = download_with_resume(&url, &destination, move |progress, total| {
                let downloads = downloads.clone();
                let id = id.clone();
                tokio::spawn(async move {
                    if let Some(status) = downloads.write().await.get_mut(&id) {
                        status.downloaded_bytes = progress;
                        status.total_bytes = Some(total);
                        status.progress = progress as f64 / total as f64;
                        status.state = DownloadState::Downloading;
                    }
                });
            }).await {
                if let Some(status) = downloads.write().await.get_mut(&id) {
                    status.state = DownloadState::Failed;
                    status.error = Some(e.to_string());
                }
            } else {
                if let Some(status) = downloads.write().await.get_mut(&id) {
                    status.state = DownloadState::Completed;
                    status.progress = 1.0;
                }
            }
        });
        
        Ok(())
    }
    
    pub async fn pause_download(&self, id: &str) -> Result<(), Error> {
        if let Some(status) = self.downloads.write().await.get_mut(id) {
            status.state = DownloadState::Paused;
        }
        Ok(())
    }
    
    pub async fn resume_download(&self, id: &str) -> Result<(), Error> {
        // Implement resume logic
        Ok(())
    }
    
    pub async fn cancel_download(&self, id: &str) -> Result<(), Error> {
        if let Some(status) = self.downloads.write().await.get_mut(id) {
            status.state = DownloadState::Cancelled;
        }
        Ok(())
    }
    
    pub async fn get_status(&self, id: &str) -> Option<DownloadStatus> {
        self.downloads.read().await.get(id).cloned()
    }
    
    pub async fn list_downloads(&self) -> Vec<DownloadStatus> {
        self.downloads.read().await.values().cloned().collect()
    }
}

// Resume support
async fn download_with_resume(
    url: &str,
    path: &Path,
    mut progress_callback: impl FnMut(u64, u64),
) -> Result<(), Error> {
    let client = reqwest::Client::new();
    let mut file_size = 0;
    
    // Check if partial file exists
    if path.exists() {
        file_size = tokio::fs::metadata(path).await?.len();
    }
    
    // Request with range header for resume
    let mut request = client.get(url);
    if file_size > 0 {
        request = request.header("Range", format!("bytes={}-", file_size));
    }
    
    let response = request.send().await?;
    let total_size = response.content_length().unwrap_or(0) + file_size;
    
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
        
    let mut stream = response.bytes_stream();
    let mut downloaded = file_size;
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        progress_callback(downloaded, total_size);
    }
    
    Ok(())
}
```

## Storage Strategies

### Directory Structure

```
~/.woolly/
├── models/
│   ├── llama-7b-q4_0.gguf
│   ├── llama-13b-q5_1.gguf
│   ├── mistral-7b-q4_k_m.gguf
│   └── metadata/
│       ├── llama-7b-q4_0.json
│       └── mistral-7b-q4_k_m.json
├── cache/
│   └── kv_cache/
└── config/
    └── models.toml
```

### Model Registry

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    models: HashMap<String, ModelEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
    pub hash: String,
    pub quantization: String,
    pub architecture: String,
    pub parameters: String,
    pub license: Option<String>,
    pub source_url: Option<String>,
    pub downloaded_at: DateTime<Utc>,
    pub last_used: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }
    
    pub async fn load_from_file(path: &Path) -> Result<Self, Error> {
        let content = tokio::fs::read_to_string(path).await?;
        Ok(toml::from_str(&content)?)
    }
    
    pub async fn save_to_file(&self, path: &Path) -> Result<(), Error> {
        let content = toml::to_string_pretty(self)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }
    
    pub fn register_model(&mut self, entry: ModelEntry) {
        self.models.insert(entry.id.clone(), entry);
    }
    
    pub fn get_model(&self, id: &str) -> Option<&ModelEntry> {
        self.models.get(id)
    }
    
    pub fn list_models(&self) -> Vec<&ModelEntry> {
        self.models.values().collect()
    }
    
    pub fn remove_model(&mut self, id: &str) -> Option<ModelEntry> {
        self.models.remove(id)
    }
    
    pub fn find_models_by_architecture(&self, arch: &str) -> Vec<&ModelEntry> {
        self.models
            .values()
            .filter(|m| m.architecture == arch)
            .collect()
    }
    
    pub fn get_total_size(&self) -> u64 {
        self.models.values().map(|m| m.size).sum()
    }
}
```

### Storage Management

```rust
pub struct StorageManager {
    models_dir: PathBuf,
    max_storage: Option<u64>,
    registry: Arc<RwLock<ModelRegistry>>,
}

impl StorageManager {
    pub fn new(models_dir: PathBuf) -> Self {
        Self {
            models_dir,
            max_storage: None,
            registry: Arc::new(RwLock::new(ModelRegistry::new())),
        }
    }
    
    pub fn with_max_storage(mut self, bytes: u64) -> Self {
        self.max_storage = Some(bytes);
        self
    }
    
    pub async fn scan_models_directory(&self) -> Result<Vec<ModelEntry>, Error> {
        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(&self.models_dir).await?;
        
        while let Some(entry) = dir.next_entry().await? {
            let path = entry.path();
            if path.extension() == Some(OsStr::new("gguf")) {
                if let Ok(model_entry) = self.analyze_model_file(&path).await {
                    entries.push(model_entry);
                }
            }
        }
        
        Ok(entries)
    }
    
    async fn analyze_model_file(&self, path: &Path) -> Result<ModelEntry, Error> {
        let metadata = tokio::fs::metadata(path).await?;
        let hash = calculate_file_hash(path).await?;
        
        // Read GGUF header for model info
        let model_info = read_gguf_metadata(path).await?;
        
        Ok(ModelEntry {
            id: path.file_stem().unwrap().to_string_lossy().to_string(),
            name: model_info.name.clone(),
            path: path.to_path_buf(),
            size: metadata.len(),
            hash,
            quantization: model_info.quantization,
            architecture: model_info.architecture,
            parameters: model_info.parameters,
            license: model_info.license,
            source_url: None,
            downloaded_at: Utc::now(),
            last_used: None,
            metadata: HashMap::new(),
        })
    }
    
    pub async fn cleanup_old_models(&self, keep_recent: usize) -> Result<Vec<String>, Error> {
        let mut registry = self.registry.write().await;
        let mut models: Vec<_> = registry.list_models().cloned().collect();
        
        // Sort by last used date
        models.sort_by_key(|m| m.last_used);
        
        let mut removed = Vec::new();
        
        // Keep the most recently used models
        while models.len() > keep_recent {
            if let Some(model) = models.remove(0) {
                // Delete the file
                tokio::fs::remove_file(&model.path).await?;
                registry.remove_model(&model.id);
                removed.push(model.id);
            }
        }
        
        Ok(removed)
    }
    
    pub async fn verify_model_integrity(&self, id: &str) -> Result<bool, Error> {
        let registry = self.registry.read().await;
        
        if let Some(model) = registry.get_model(id) {
            let current_hash = calculate_file_hash(&model.path).await?;
            Ok(current_hash == model.hash)
        } else {
            Err(Error::ModelNotFound(id.to_string()))
        }
    }
}

async fn calculate_file_hash(path: &Path) -> Result<String, Error> {
    use sha2::{Sha256, Digest};
    
    let mut file = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 8192];
    
    loop {
        let n = file.read(&mut buffer).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    
    Ok(format!("{:x}", hasher.finalize()))
}
```

## Model Formats & Quantization

### Quantization Options

| Type | Bits | Size Reduction | Quality | Use Case |
|------|------|----------------|---------|----------|
| F32 | 32 | 1x (baseline) | Perfect | Development/Testing |
| F16 | 16 | 2x | Excellent | High-quality inference |
| Q8_0 | 8 | 4x | Very Good | Balanced quality/size |
| Q5_0 | 5 | 6.4x | Good | General use |
| Q5_1 | 5.5 | 5.8x | Good | General use (better) |
| Q4_0 | 4 | 8x | Moderate | Memory-constrained |
| Q4_1 | 4.5 | 7.1x | Moderate | Memory-constrained (better) |
| Q4_K_S | 4 | 8x | Good | Modern quantization |
| Q4_K_M | 4 | 8x | Better | Modern quantization |
| Q3_K_S | 3 | 10.7x | Fair | Extreme compression |
| Q2_K | 2 | 16x | Poor | Maximum compression |

### Quantization Tool

```rust
use woolly_core::quantization::{quantize_model, QuantizationType};

pub struct ModelQuantizer {
    source_path: PathBuf,
    output_dir: PathBuf,
}

impl ModelQuantizer {
    pub fn new(source_path: PathBuf, output_dir: PathBuf) -> Self {
        Self {
            source_path,
            output_dir,
        }
    }
    
    pub async fn quantize_to_multiple_formats(
        &self,
        formats: &[QuantizationType],
        progress_callback: impl Fn(String, f32),
    ) -> Result<Vec<PathBuf>, Error> {
        let mut results = Vec::new();
        
        for (i, format) in formats.iter().enumerate() {
            let output_name = format!(
                "{}-{}.gguf",
                self.source_path.file_stem().unwrap().to_string_lossy(),
                format_to_string(format)
            );
            
            let output_path = self.output_dir.join(output_name);
            
            progress_callback(
                format!("Quantizing to {}", format_to_string(format)),
                i as f32 / formats.len() as f32,
            );
            
            quantize_model(
                &self.source_path,
                &output_path,
                *format,
                |p| progress_callback(format!("Progress: {:.1}%", p * 100.0), p),
            )?;
            
            results.push(output_path);
        }
        
        Ok(results)
    }
    
    pub fn estimate_quantized_size(
        original_size: u64,
        quantization: QuantizationType,
    ) -> u64 {
        let compression_ratio = match quantization {
            QuantizationType::F32 => 1.0,
            QuantizationType::F16 => 2.0,
            QuantizationType::Q8_0 => 4.0,
            QuantizationType::Q5_0 => 6.4,
            QuantizationType::Q5_1 => 5.8,
            QuantizationType::Q4_0 => 8.0,
            QuantizationType::Q4_1 => 7.1,
            QuantizationType::Q4_K_S => 8.0,
            QuantizationType::Q4_K_M => 8.0,
            QuantizationType::Q3_K_S => 10.7,
            QuantizationType::Q2_K => 16.0,
        };
        
        (original_size as f64 / compression_ratio) as u64
    }
}

fn format_to_string(format: &QuantizationType) -> &'static str {
    match format {
        QuantizationType::F32 => "f32",
        QuantizationType::F16 => "f16",
        QuantizationType::Q8_0 => "q8_0",
        QuantizationType::Q5_0 => "q5_0",
        QuantizationType::Q5_1 => "q5_1",
        QuantizationType::Q4_0 => "q4_0",
        QuantizationType::Q4_1 => "q4_1",
        QuantizationType::Q4_K_S => "q4_k_s",
        QuantizationType::Q4_K_M => "q4_k_m",
        QuantizationType::Q3_K_S => "q3_k_s",
        QuantizationType::Q2_K => "q2_k",
    }
}
```

### Choosing Quantization

```rust
pub fn recommend_quantization(
    available_memory: u64,
    model_size_f32: u64,
    quality_preference: QualityPreference,
) -> QuantizationType {
    let memory_ratio = available_memory as f64 / model_size_f32 as f64;
    
    match quality_preference {
        QualityPreference::Highest => {
            if memory_ratio >= 1.0 { QuantizationType::F32 }
            else if memory_ratio >= 0.5 { QuantizationType::F16 }
            else if memory_ratio >= 0.25 { QuantizationType::Q8_0 }
            else { QuantizationType::Q5_1 }
        }
        QualityPreference::Balanced => {
            if memory_ratio >= 0.25 { QuantizationType::Q8_0 }
            else if memory_ratio >= 0.17 { QuantizationType::Q5_1 }
            else if memory_ratio >= 0.125 { QuantizationType::Q4_K_M }
            else { QuantizationType::Q4_0 }
        }
        QualityPreference::MemoryEfficient => {
            if memory_ratio >= 0.125 { QuantizationType::Q4_K_M }
            else if memory_ratio >= 0.1 { QuantizationType::Q4_0 }
            else if memory_ratio >= 0.093 { QuantizationType::Q3_K_S }
            else { QuantizationType::Q2_K }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum QualityPreference {
    Highest,
    Balanced,
    MemoryEfficient,
}
```

## Model Switching

### Hot-Swapping Models

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct ModelManager {
    current_model: Arc<RwLock<Option<Model>>>,
    model_cache: Arc<RwLock<LruCache<String, Arc<Model>>>>,
    loading: Arc<RwLock<HashSet<String>>>,
}

impl ModelManager {
    pub fn new(cache_size: usize) -> Self {
        Self {
            current_model: Arc::new(RwLock::new(None)),
            model_cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
            loading: Arc::new(RwLock::new(HashSet::new())),
        }
    }
    
    pub async fn switch_model(&self, model_id: &str) -> Result<(), Error> {
        // Check if model is already loaded
        if let Some(model) = self.model_cache.read().await.peek(model_id) {
            *self.current_model.write().await = Some(model.clone());
            return Ok(());
        }
        
        // Check if model is currently being loaded
        if self.loading.read().await.contains(model_id) {
            // Wait for loading to complete
            while self.loading.read().await.contains(model_id) {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            
            // Try again
            return self.switch_model(model_id).await;
        }
        
        // Mark as loading
        self.loading.write().await.insert(model_id.to_string());
        
        // Load model
        let result = self.load_model_internal(model_id).await;
        
        // Remove from loading set
        self.loading.write().await.remove(model_id);
        
        result
    }
    
    async fn load_model_internal(&self, model_id: &str) -> Result<(), Error> {
        // Load model from disk
        let model_path = self.get_model_path(model_id)?;
        let model = Arc::new(Model::from_file(&model_path)?);
        
        // Add to cache
        self.model_cache.write().await.put(model_id.to_string(), model.clone());
        
        // Set as current model
        *self.current_model.write().await = Some(model);
        
        Ok(())
    }
    
    pub async fn preload_models(&self, model_ids: &[String]) -> Result<(), Error> {
        for model_id in model_ids {
            // Load in background
            let model_id = model_id.clone();
            let manager = self.clone();
            
            tokio::spawn(async move {
                if let Err(e) = manager.load_model_internal(&model_id).await {
                    eprintln!("Failed to preload model {}: {}", model_id, e);
                }
            });
        }
        
        Ok(())
    }
    
    pub async fn unload_model(&self, model_id: &str) -> Result<(), Error> {
        self.model_cache.write().await.pop(model_id);
        
        // If it was the current model, clear it
        let mut current = self.current_model.write().await;
        if let Some(model) = &*current {
            if model.id() == model_id {
                *current = None;
            }
        }
        
        Ok(())
    }
    
    pub async fn clear_cache(&self) {
        self.model_cache.write().await.clear();
        *self.current_model.write().await = None;
    }
    
    pub async fn get_cached_models(&self) -> Vec<String> {
        self.model_cache.read().await
            .iter()
            .map(|(k, _)| k.clone())
            .collect()
    }
}
```

### Model Comparison

```rust
pub struct ModelComparator {
    models: Vec<Model>,
}

impl ModelComparator {
    pub fn new(models: Vec<Model>) -> Self {
        Self { models }
    }
    
    pub async fn compare_outputs(
        &self,
        prompt: &str,
        generation_params: GenerationParams,
    ) -> Vec<ComparisonResult> {
        let mut results = Vec::new();
        
        for model in &self.models {
            let start = Instant::now();
            
            let output = model.generate(prompt, &generation_params).await?;
            
            let duration = start.elapsed();
            let tokens_per_second = output.tokens_generated as f64 / duration.as_secs_f64();
            
            results.push(ComparisonResult {
                model_id: model.id().to_string(),
                output: output.text,
                tokens_generated: output.tokens_generated,
                generation_time: duration,
                tokens_per_second,
                memory_used: model.memory_usage(),
            });
        }
        
        results
    }
    
    pub fn compare_performance(&self) -> PerformanceComparison {
        let mut comparison = PerformanceComparison::default();
        
        for model in &self.models {
            let stats = model.get_stats();
            comparison.add_model_stats(model.id(), stats);
        }
        
        comparison
    }
}

#[derive(Debug)]
pub struct ComparisonResult {
    pub model_id: String,
    pub output: String,
    pub tokens_generated: usize,
    pub generation_time: Duration,
    pub tokens_per_second: f64,
    pub memory_used: usize,
}
```

## Memory Management

### Memory Requirements Calculator

```rust
pub struct MemoryCalculator;

impl MemoryCalculator {
    pub fn calculate_model_memory(
        parameters: u64,
        quantization: QuantizationType,
        context_length: usize,
        batch_size: usize,
    ) -> MemoryRequirements {
        // Base model memory
        let bytes_per_param = Self::bytes_per_parameter(quantization);
        let model_memory = parameters * bytes_per_param;
        
        // KV cache memory
        let kv_cache_memory = Self::calculate_kv_cache(
            parameters,
            context_length,
            batch_size,
            quantization,
        );
        
        // Activation memory
        let activation_memory = Self::calculate_activation_memory(
            parameters,
            batch_size,
        );
        
        // Buffer for temporary allocations (10% overhead)
        let buffer_memory = (model_memory + kv_cache_memory + activation_memory) / 10;
        
        MemoryRequirements {
            model_memory,
            kv_cache_memory,
            activation_memory,
            buffer_memory,
            total: model_memory + kv_cache_memory + activation_memory + buffer_memory,
            recommended: (model_memory + kv_cache_memory + activation_memory + buffer_memory) * 1.2,
        }
    }
    
    fn bytes_per_parameter(quantization: QuantizationType) -> u64 {
        match quantization {
            QuantizationType::F32 => 4,
            QuantizationType::F16 => 2,
            QuantizationType::Q8_0 => 1,
            QuantizationType::Q5_0 | QuantizationType::Q5_1 => 5 / 8,
            QuantizationType::Q4_0 | QuantizationType::Q4_1 => 1 / 2,
            QuantizationType::Q4_K_S | QuantizationType::Q4_K_M => 1 / 2,
            QuantizationType::Q3_K_S => 3 / 8,
            QuantizationType::Q2_K => 1 / 4,
        }
    }
    
    fn calculate_kv_cache(
        parameters: u64,
        context_length: usize,
        batch_size: usize,
        quantization: QuantizationType,
    ) -> u64 {
        // Rough estimation based on model architecture
        let hidden_dim = (parameters as f64 / 1e9 * 4096.0) as u64;
        let n_layers = (parameters as f64 / 1e9 * 32.0) as u64;
        
        let kv_heads = 32; // Typical for most models
        let head_dim = hidden_dim / kv_heads;
        
        // K and V cache per layer
        let kv_per_layer = 2 * batch_size as u64 * context_length as u64 * head_dim * 4; // FP32
        
        kv_per_layer * n_layers
    }
    
    fn calculate_activation_memory(parameters: u64, batch_size: usize) -> u64 {
        // Rough estimation
        let hidden_dim = (parameters as f64 / 1e9 * 4096.0) as u64;
        hidden_dim * batch_size as u64 * 4 * 10 // 10 intermediate tensors
    }
}

#[derive(Debug)]
pub struct MemoryRequirements {
    pub model_memory: u64,
    pub kv_cache_memory: u64,
    pub activation_memory: u64,
    pub buffer_memory: u64,
    pub total: u64,
    pub recommended: u64,
}
```

### Memory Pool Manager

```rust
use std::alloc::{alloc, dealloc, Layout};

pub struct MemoryPool {
    pools: Vec<Pool>,
    total_allocated: AtomicUsize,
    max_memory: usize,
}

struct Pool {
    size_class: usize,
    free_blocks: Vec<*mut u8>,
    allocated_blocks: HashSet<*mut u8>,
}

impl MemoryPool {
    pub fn new(max_memory: usize) -> Self {
        let size_classes = vec![
            1024,           // 1KB
            4096,           // 4KB
            16384,          // 16KB
            65536,          // 64KB
            262144,         // 256KB
            1048576,        // 1MB
            4194304,        // 4MB
            16777216,       // 16MB
            67108864,       // 64MB
            268435456,      // 256MB
        ];
        
        let pools = size_classes
            .into_iter()
            .map(|size| Pool {
                size_class: size,
                free_blocks: Vec::new(),
                allocated_blocks: HashSet::new(),
            })
            .collect();
            
        Self {
            pools,
            total_allocated: AtomicUsize::new(0),
            max_memory,
        }
    }
    
    pub fn allocate(&mut self, size: usize) -> Result<*mut u8, Error> {
        // Check memory limit
        if self.total_allocated.load(Ordering::Relaxed) + size > self.max_memory {
            return Err(Error::OutOfMemory);
        }
        
        // Find appropriate pool
        let pool_idx = self.pools
            .iter()
            .position(|p| p.size_class >= size)
            .ok_or(Error::AllocationTooLarge)?;
            
        let pool = &mut self.pools[pool_idx];
        
        // Try to reuse a free block
        if let Some(ptr) = pool.free_blocks.pop() {
            pool.allocated_blocks.insert(ptr);
            return Ok(ptr);
        }
        
        // Allocate new block
        unsafe {
            let layout = Layout::from_size_align(pool.size_class, 64)?;
            let ptr = alloc(layout);
            
            if ptr.is_null() {
                return Err(Error::AllocationFailed);
            }
            
            pool.allocated_blocks.insert(ptr);
            self.total_allocated.fetch_add(pool.size_class, Ordering::Relaxed);
            
            Ok(ptr)
        }
    }
    
    pub fn deallocate(&mut self, ptr: *mut u8, size: usize) {
        // Find the pool
        let pool_idx = self.pools
            .iter()
            .position(|p| p.size_class >= size);
            
        if let Some(idx) = pool_idx {
            let pool = &mut self.pools[idx];
            
            if pool.allocated_blocks.remove(&ptr) {
                pool.free_blocks.push(ptr);
            }
        }
    }
    
    pub fn clear_unused(&mut self) {
        for pool in &mut self.pools {
            for ptr in pool.free_blocks.drain(..) {
                unsafe {
                    let layout = Layout::from_size_align(pool.size_class, 64).unwrap();
                    dealloc(ptr, layout);
                }
                self.total_allocated.fetch_sub(pool.size_class, Ordering::Relaxed);
            }
        }
    }
    
    pub fn memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            allocated: self.total_allocated.load(Ordering::Relaxed),
            max_memory: self.max_memory,
            pools: self.pools
                .iter()
                .map(|p| PoolUsage {
                    size_class: p.size_class,
                    allocated_blocks: p.allocated_blocks.len(),
                    free_blocks: p.free_blocks.len(),
                })
                .collect(),
        }
    }
}
```

## Model Validation

### Integrity Checking

```rust
pub struct ModelValidator;

impl ModelValidator {
    pub async fn validate_model(path: &Path) -> Result<ValidationReport, Error> {
        let mut report = ValidationReport::default();
        
        // Check file exists and is readable
        if !path.exists() {
            report.errors.push("Model file does not exist".to_string());
            return Ok(report);
        }
        
        // Validate GGUF format
        match Self::validate_gguf_format(path).await {
            Ok(()) => report.format_valid = true,
            Err(e) => report.errors.push(format!("Invalid GGUF format: {}", e)),
        }
        
        // Validate model architecture
        match Self::validate_architecture(path).await {
            Ok(arch) => {
                report.architecture = Some(arch);
                report.architecture_valid = true;
            }
            Err(e) => report.errors.push(format!("Invalid architecture: {}", e)),
        }
        
        // Validate tensors
        match Self::validate_tensors(path).await {
            Ok(tensor_info) => {
                report.tensor_count = tensor_info.count;
                report.tensors_valid = true;
            }
            Err(e) => report.errors.push(format!("Invalid tensors: {}", e)),
        }
        
        // Check for required metadata
        match Self::validate_metadata(path).await {
            Ok(metadata) => {
                report.metadata = Some(metadata);
                report.metadata_valid = true;
            }
            Err(e) => report.warnings.push(format!("Missing metadata: {}", e)),
        }
        
        report.valid = report.errors.is_empty();
        Ok(report)
    }
    
    async fn validate_gguf_format(path: &Path) -> Result<(), Error> {
        let mut file = tokio::fs::File::open(path).await?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).await?;
        
        if &magic != b"GGUF" {
            return Err(Error::InvalidFormat("Not a GGUF file".to_string()));
        }
        
        // Read version
        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes).await?;
        let version = u32::from_le_bytes(version_bytes);
        
        if version > 3 {
            return Err(Error::InvalidFormat(format!("Unsupported GGUF version: {}", version)));
        }
        
        Ok(())
    }
    
    async fn validate_architecture(path: &Path) -> Result<String, Error> {
        let metadata = read_gguf_metadata(path).await?;
        
        let supported_architectures = vec![
            "llama",
            "falcon",
            "gpt2",
            "gptj",
            "gptneox",
            "mpt",
            "baichuan",
            "starcoder",
            "mistral",
            "mixtral",
            "phi",
            "qwen",
        ];
        
        if !supported_architectures.contains(&metadata.architecture.as_str()) {
            return Err(Error::UnsupportedArchitecture(metadata.architecture));
        }
        
        Ok(metadata.architecture)
    }
    
    async fn validate_tensors(path: &Path) -> Result<TensorInfo, Error> {
        // Implementation depends on GGUF reading logic
        Ok(TensorInfo {
            count: 0, // Placeholder
        })
    }
    
    async fn validate_metadata(path: &Path) -> Result<ModelMetadata, Error> {
        read_gguf_metadata(path).await
    }
}

#[derive(Debug, Default)]
pub struct ValidationReport {
    pub valid: bool,
    pub format_valid: bool,
    pub architecture_valid: bool,
    pub tensors_valid: bool,
    pub metadata_valid: bool,
    pub architecture: Option<String>,
    pub tensor_count: usize,
    pub metadata: Option<ModelMetadata>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}
```

### Performance Benchmarking

```rust
pub struct ModelBenchmark {
    model: Model,
    results: BenchmarkResults,
}

impl ModelBenchmark {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            results: BenchmarkResults::default(),
        }
    }
    
    pub async fn run_full_benchmark(&mut self) -> Result<&BenchmarkResults, Error> {
        // Prompt processing speed
        self.benchmark_prompt_processing().await?;
        
        // Token generation speed
        self.benchmark_token_generation().await?;
        
        // Memory usage
        self.benchmark_memory_usage().await?;
        
        // Context length handling
        self.benchmark_context_lengths().await?;
        
        // Batch processing
        self.benchmark_batch_processing().await?;
        
        Ok(&self.results)
    }
    
    async fn benchmark_prompt_processing(&mut self) -> Result<(), Error> {
        let prompts = vec![
            "Hello",                           // 1 token
            "The quick brown fox jumps over",  // ~8 tokens
            "Lorem ipsum dolor sit amet...",   // ~50 tokens
            // Long prompt with ~500 tokens
        ];
        
        let mut prompt_results = Vec::new();
        
        for prompt in prompts {
            let start = Instant::now();
            let tokens = self.model.tokenize(prompt)?;
            let _ = self.model.process_prompt(&tokens).await?;
            let duration = start.elapsed();
            
            prompt_results.push(PromptBenchmark {
                prompt_length: prompt.len(),
                token_count: tokens.len(),
                processing_time: duration,
                tokens_per_second: tokens.len() as f64 / duration.as_secs_f64(),
            });
        }
        
        self.results.prompt_processing = prompt_results;
        Ok(())
    }
    
    async fn benchmark_token_generation(&mut self) -> Result<(), Error> {
        let token_counts = vec![10, 50, 100, 200, 500];
        let mut generation_results = Vec::new();
        
        for count in token_counts {
            let start = Instant::now();
            let result = self.model.generate(
                "Once upon a time",
                GenerationParams {
                    max_tokens: count,
                    temperature: 0.7,
                    ..Default::default()
                },
            ).await?;
            let duration = start.elapsed();
            
            generation_results.push(GenerationBenchmark {
                tokens_requested: count,
                tokens_generated: result.tokens_generated,
                generation_time: duration,
                tokens_per_second: result.tokens_generated as f64 / duration.as_secs_f64(),
                time_to_first_token: result.time_to_first_token,
            });
        }
        
        self.results.token_generation = generation_results;
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct BenchmarkResults {
    pub prompt_processing: Vec<PromptBenchmark>,
    pub token_generation: Vec<GenerationBenchmark>,
    pub memory_usage: MemoryBenchmark,
    pub context_handling: Vec<ContextBenchmark>,
    pub batch_processing: Vec<BatchBenchmark>,
}
```

## Best Practices

### 1. Model Organization

```rust
// models.toml configuration
[models]
default = "llama-7b-q4_k_m"

[[models.available]]
id = "llama-7b-q4_k_m"
name = "LLaMA 7B Q4_K_M"
path = "models/llama-7b-q4_k_m.gguf"
size = 3825819904
parameters = "7B"
quantization = "Q4_K_M"
context_length = 4096
license = "LLaMA 2"
tags = ["general", "chat", "instruct"]

[[models.available]]
id = "codellama-13b-q5_1"
name = "CodeLlama 13B Q5_1"
path = "models/codellama-13b-q5_1.gguf"
size = 8536870912
parameters = "13B"
quantization = "Q5_1"
context_length = 16384
license = "LLaMA 2"
tags = ["code", "programming", "completion"]
```

### 2. Automatic Model Selection

```rust
pub fn select_best_model(
    available_models: &[ModelEntry],
    system_info: &SystemInfo,
    task_type: TaskType,
) -> Option<&ModelEntry> {
    let available_memory = system_info.available_memory * 0.8; // Leave 20% buffer
    
    // Filter models that fit in memory
    let mut suitable_models: Vec<_> = available_models
        .iter()
        .filter(|m| m.size <= available_memory)
        .filter(|m| task_matches_model(task_type, &m.tags))
        .collect();
        
    // Sort by quality (larger models with better quantization first)
    suitable_models.sort_by_key(|m| {
        let param_score = parse_parameters(&m.parameters);
        let quant_score = quantization_quality_score(&m.quantization);
        -(param_score * quant_score) // Negative for descending order
    });
    
    suitable_models.first().copied()
}

fn task_matches_model(task: TaskType, tags: &[String]) -> bool {
    match task {
        TaskType::Chat => tags.contains(&"chat".to_string()),
        TaskType::Code => tags.contains(&"code".to_string()),
        TaskType::Completion => tags.contains(&"completion".to_string()),
        TaskType::Analysis => tags.contains(&"analysis".to_string()),
        TaskType::General => true,
    }
}
```

### 3. Model Lifecycle Management

```rust
pub struct ModelLifecycleManager {
    storage_manager: StorageManager,
    download_manager: DownloadManager,
    usage_tracker: UsageTracker,
}

impl ModelLifecycleManager {
    pub async fn ensure_model_available(
        &self,
        model_id: &str,
        source_url: Option<&str>,
    ) -> Result<PathBuf, Error> {
        // Check if model exists locally
        if let Some(path) = self.storage_manager.find_model(model_id).await? {
            self.usage_tracker.record_usage(model_id).await;
            return Ok(path);
        }
        
        // Download if source URL provided
        if let Some(url) = source_url {
            let path = self.download_manager
                .download_model(url, model_id)
                .await?;
                
            self.storage_manager.register_model(model_id, &path).await?;
            self.usage_tracker.record_download(model_id).await;
            
            return Ok(path);
        }
        
        Err(Error::ModelNotFound(model_id.to_string()))
    }
    
    pub async fn cleanup_unused_models(&self, keep_days: u32) -> Result<Vec<String>, Error> {
        let cutoff = Utc::now() - Duration::days(keep_days as i64);
        let unused_models = self.usage_tracker
            .get_models_unused_since(cutoff)
            .await?;
            
        let mut removed = Vec::new();
        
        for model_id in unused_models {
            if self.storage_manager.remove_model(&model_id).await? {
                removed.push(model_id);
            }
        }
        
        Ok(removed)
    }
}
```

### 4. Error Recovery

```rust
pub struct ResilientModelLoader {
    primary_source: String,
    fallback_sources: Vec<String>,
    retry_policy: RetryPolicy,
}

impl ResilientModelLoader {
    pub async fn load_model_with_fallback(
        &self,
        model_id: &str,
    ) -> Result<Model, Error> {
        let mut last_error = None;
        
        // Try primary source
        match self.try_load_from_source(&self.primary_source, model_id).await {
            Ok(model) => return Ok(model),
            Err(e) => last_error = Some(e),
        }
        
        // Try fallback sources
        for source in &self.fallback_sources {
            match self.try_load_from_source(source, model_id).await {
                Ok(model) => return Ok(model),
                Err(e) => last_error = Some(e),
            }
        }
        
        Err(last_error.unwrap_or_else(|| Error::ModelNotFound(model_id.to_string())))
    }
    
    async fn try_load_from_source(
        &self,
        source: &str,
        model_id: &str,
    ) -> Result<Model, Error> {
        let mut attempt = 0;
        
        loop {
            match self.download_and_load(source, model_id).await {
                Ok(model) => return Ok(model),
                Err(e) if self.retry_policy.should_retry(attempt, &e) => {
                    let delay = self.retry_policy.delay_for_attempt(attempt);
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }
}
```