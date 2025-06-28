//! Configuration management for Woolly CLI

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use woolly_core::prelude::*;

/// CLI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Default model path
    pub default_model: Option<PathBuf>,
    
    /// Model search directories
    pub model_dirs: Vec<PathBuf>,
    
    /// Default inference configuration
    pub inference: InferenceConfig,
    
    /// MCP configuration
    pub mcp: McpConfig,
    
    /// Benchmark configuration
    pub benchmark: BenchmarkConfig,
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    
    /// Temperature for sampling
    pub temperature: f32,
    
    /// Top-p sampling threshold
    pub top_p: f32,
    
    /// Top-k sampling limit
    pub top_k: usize,
    
    /// Repetition penalty
    pub repetition_penalty: f32,
    
    /// Context window size
    pub context_size: usize,
    
    /// Number of threads to use
    pub num_threads: Option<usize>,
    
    /// Device configuration
    pub device: DeviceConfig,
}

/// MCP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// Enable MCP server mode
    pub enable_server: bool,
    
    /// Server host
    pub host: String,
    
    /// Server port
    pub port: u16,
    
    /// Enable WebSocket transport
    pub websocket: bool,
    
    /// Enable HTTP transport
    pub http: bool,
    
    /// Enable stdio transport
    pub stdio: bool,
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of benchmark iterations
    pub iterations: u32,
    
    /// Warmup iterations
    pub warmup_iterations: u32,
    
    /// Include comparison with llama.cpp
    pub compare_llamacpp: bool,
    
    /// Output directory for results
    pub output_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            default_model: None,
            model_dirs: vec![
                dirs::home_dir().unwrap_or_default().join(".woolly/models"),
                PathBuf::from("./models"),
            ],
            inference: InferenceConfig::default(),
            mcp: McpConfig::default(),
            benchmark: BenchmarkConfig::default(),
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            context_size: 131072, // Support Granite model's 131k context
            num_threads: None,
            device: DeviceConfig::default(),
        }
    }
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enable_server: false,
            host: "localhost".to_string(),
            port: 8080,
            websocket: true,
            http: true,
            stdio: false,
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            warmup_iterations: 2,
            compare_llamacpp: false,
            output_dir: PathBuf::from("./bench_results"),
        }
    }
}

impl Config {
    /// Load configuration from file or create default
    pub fn load(config_path: Option<&Path>) -> Result<Self> {
        let config_path = match config_path {
            Some(path) => path.to_path_buf(),
            None => Self::default_config_path(),
        };

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
            
            let config: Config = toml::from_str(&content)
                .with_context(|| format!("Failed to parse config file: {}", config_path.display()))?;
            
            Ok(config)
        } else {
            // Create default config
            let config = Config::default();
            config.save(&config_path)?;
            Ok(config)
        }
    }

    /// Save configuration to file
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {}", parent.display()))?;
        }

        let content = toml::to_string_pretty(self)
            .context("Failed to serialize configuration")?;
        
        std::fs::write(path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;
        
        Ok(())
    }

    /// Get default configuration file path
    pub fn default_config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| dirs::home_dir().unwrap_or_default().join(".config"))
            .join("woolly")
            .join("config.toml")
    }

    /// Find model file in configured directories
    pub fn find_model(&self, model_name: &str) -> Result<PathBuf> {
        // If it's already a full path and exists, use it
        let model_path = Path::new(model_name);
        if model_path.exists() {
            return Ok(model_path.to_path_buf());
        }

        // Expand shell variables like ~ 
        let expanded = shellexpand::full(model_name)
            .context("Failed to expand shell variables in model path")?;
        let expanded_path = Path::new(expanded.as_ref());
        if expanded_path.exists() {
            return Ok(expanded_path.to_path_buf());
        }

        // Search in configured model directories
        for dir in &self.model_dirs {
            let candidate = dir.join(model_name);
            if candidate.exists() {
                return Ok(candidate);
            }
            
            // Also try with .gguf extension
            let with_ext = dir.join(format!("{}.gguf", model_name));
            if with_ext.exists() {
                return Ok(with_ext);
            }
        }

        anyhow::bail!("Model '{}' not found in any configured directory", model_name);
    }
}