//! Configuration for the inference engine

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for the inference engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Maximum context length supported by the engine
    pub max_context_length: usize,
    
    /// Maximum batch size for parallel inference
    pub max_batch_size: usize,
    
    /// Number of threads to use for CPU computation
    pub num_threads: usize,
    
    /// Device configuration
    pub device: DeviceConfig,
    
    /// Memory configuration
    pub memory: MemoryConfig,
    
    /// Cache configuration
    pub cache: CacheConfig,
    
    /// Optimization settings
    pub optimizations: OptimizationConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_context_length: 4096,
            max_batch_size: 16,
            num_threads: num_cpus::get(),
            device: DeviceConfig::default(),
            memory: MemoryConfig::default(),
            cache: CacheConfig::default(),
            optimizations: OptimizationConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Primary device type
    pub device_type: DeviceType,
    
    /// Device ID (for multi-GPU systems)
    pub device_id: usize,
    
    /// Whether to use CPU fallback
    pub cpu_fallback: bool,
    
    /// CUDA-specific settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda: Option<CudaConfig>,
    
    /// Metal-specific settings (for Apple Silicon)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metal: Option<MetalConfig>,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            device_id: 0,
            cpu_fallback: true,
            cuda: None,
            metal: None,
        }
    }
}

/// Device types supported by the engine
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    Cpu,
    Cuda,
    Metal,
    Rocm,
}

/// CUDA-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaConfig {
    /// CUDA compute capability
    pub compute_capability: (u8, u8),
    
    /// Whether to use TensorRT optimization
    pub use_tensorrt: bool,
    
    /// Memory pool size in MB
    pub memory_pool_size: usize,
}

/// Metal-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalConfig {
    /// Whether to use Metal Performance Shaders
    pub use_mps: bool,
    
    /// Maximum buffer size in MB
    pub max_buffer_size: usize,
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    
    /// Whether to use memory mapping for model loading
    pub use_mmap: bool,
    
    /// Whether to pin memory for faster GPU transfers
    pub pin_memory: bool,
    
    /// Allocator type
    pub allocator: AllocatorType,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 8192, // 8GB default
            use_mmap: true,
            pin_memory: false,
            allocator: AllocatorType::System,
        }
    }
}

/// Memory allocator types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AllocatorType {
    System,
    Jemalloc,
    Mimalloc,
    Custom,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// KV cache directory
    pub cache_dir: Option<PathBuf>,
    
    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,
    
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
    
    /// Whether to persist cache to disk
    pub persistent: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: None,
            max_cache_size_mb: 2048, // 2GB default
            eviction_policy: EvictionPolicy::Lru,
            persistent: false,
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvictionPolicy {
    Lru,
    Lfu,
    Fifo,
    Random,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Whether to use Flash Attention
    pub use_flash_attention: bool,
    
    /// Whether to use torch.compile (if available)
    pub use_torch_compile: bool,
    
    /// Whether to fuse operations
    pub operator_fusion: bool,
    
    /// Quantization settings
    pub quantization: QuantizationConfig,
    
    /// Whether to use AMP (Automatic Mixed Precision)
    pub use_amp: bool,
    
    /// Graph optimization level (0-3)
    pub graph_optimization_level: u8,
    
    /// Whether to use SIMD operations (can be disabled via WOOLLY_DISABLE_SIMD env var)
    pub use_simd: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        // Check environment variable for SIMD override
        let use_simd = std::env::var("WOOLLY_DISABLE_SIMD")
            .map(|v| v != "1" && v.to_lowercase() != "true")
            .unwrap_or(true);
            
        Self {
            use_flash_attention: true,
            use_torch_compile: false,
            operator_fusion: true,
            quantization: QuantizationConfig::default(),
            use_amp: false,
            graph_optimization_level: 2,
            use_simd,
        }
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Whether quantization is enabled
    pub enabled: bool,
    
    /// Quantization method
    pub method: QuantizationMethod,
    
    /// Bits for weight quantization
    pub weight_bits: u8,
    
    /// Bits for activation quantization
    pub activation_bits: Option<u8>,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: QuantizationMethod::None,
            weight_bits: 8,
            activation_bits: None,
        }
    }
}

/// Quantization methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationMethod {
    None,
    Int8,
    Gptq,
    Awq,
    Gguf,
    Bnb,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    
    /// Whether to log to file
    pub log_to_file: bool,
    
    /// Log file path
    pub log_file: Option<PathBuf>,
    
    /// Whether to log performance metrics
    pub log_performance: bool,
    
    /// Performance logging interval in seconds
    pub performance_interval_secs: u64,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            log_to_file: false,
            log_file: None,
            log_performance: false,
            performance_interval_secs: 60,
        }
    }
}

/// Log levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl EngineConfig {
    /// Load configuration from a file
    pub fn from_file(path: &PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a file
    pub fn to_file(&self, path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate context length
        if self.max_context_length == 0 {
            return Err("max_context_length must be greater than 0".to_string());
        }

        // Validate batch size
        if self.max_batch_size == 0 {
            return Err("max_batch_size must be greater than 0".to_string());
        }

        // Validate thread count
        if self.num_threads == 0 {
            return Err("num_threads must be greater than 0".to_string());
        }

        // Validate memory settings
        if self.memory.max_memory_mb == 0 {
            return Err("max_memory_mb must be greater than 0".to_string());
        }

        // Validate quantization settings
        if self.optimizations.quantization.enabled {
            if self.optimizations.quantization.weight_bits == 0
                || self.optimizations.quantization.weight_bits > 32
            {
                return Err("weight_bits must be between 1 and 32".to_string());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EngineConfig::default();
        assert_eq!(config.max_context_length, 4096);
        assert_eq!(config.max_batch_size, 16);
        assert!(config.num_threads > 0);
    }

    #[test]
    fn test_config_validation() {
        let mut config = EngineConfig::default();
        assert!(config.validate().is_ok());

        config.max_context_length = 0;
        assert!(config.validate().is_err());
    }
}