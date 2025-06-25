//! MLX GPU backend for Woolly tensor operations on Apple Silicon
//!
//! This crate provides GPU acceleration for tensor operations using Apple's MLX framework,
//! specifically optimized for Apple Silicon chips (M1, M2, M3, etc.) with unified memory architecture.
//!
//! Key features:
//! - GPU-accelerated tensor operations using MLX
//! - Unified memory model optimized for Apple Silicon
//! - Quantized inference support (Q4_0, Q8_0)
//! - Automatic fallback to CPU on unsupported platforms
//! - Memory-efficient operations leveraging unified memory
//!
//! # Platform Support
//!
//! MLX backend is only available on macOS with Apple Silicon. On other platforms,
//! operations will automatically fall back to CPU backend.
//!
//! # Examples
//!
//! ```rust,ignore
//! use woolly_mlx::MLXBackend;
//! use woolly_tensor::backend::{TensorBackend, Device};
//!
//! // Create MLX backend (only works on Apple Silicon)
//! let backend = MLXBackend::new()?;
//! 
//! // Check if MLX is available
//! if backend.is_available() {
//!     println!("MLX GPU acceleration available");
//! }
//! ```

use std::sync::Once;
use tracing::{debug, info, warn};

pub mod backend;
pub mod device;
pub mod error;
pub mod ops;
pub mod platform;
pub mod storage;

#[cfg(feature = "mlx")]
mod ffi;

// Re-exports
pub use backend::MLXBackend;
pub use device::{MLXDevice, Device as MLXDeviceEnum};
pub use error::{MLXError, Result};
pub use storage::MLXStorage;

static INIT: Once = Once::new();

/// Initialize MLX runtime
/// 
/// This function should be called once before using any MLX operations.
/// It's safe to call multiple times - initialization will only happen once.
pub fn init() -> Result<()> {
    INIT.call_once(|| {
        if platform::is_apple_silicon() {
            debug!("Initializing MLX runtime on Apple Silicon");
            
            #[cfg(feature = "mlx")]
            {
                match ffi::mlx_init() {
                    Ok(_) => {
                        info!("MLX runtime initialized successfully");
                    }
                    Err(e) => {
                        warn!("Failed to initialize MLX runtime: {}", e);
                    }
                }
            }
            
            #[cfg(not(feature = "mlx"))]
            {
                warn!("MLX feature not enabled, using fallback implementation");
            }
        } else {
            debug!("Not on Apple Silicon, MLX not available");
        }
    });
    
    Ok(())
}

/// Check if MLX is available and functional
pub fn is_available() -> bool {
    if !platform::is_apple_silicon() {
        return false;
    }
    
    #[cfg(feature = "mlx")]
    {
        ffi::mlx_is_available()
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        false
    }
}

/// Get MLX version information
pub fn version() -> String {
    if !is_available() {
        return "MLX not available".to_string();
    }
    
    #[cfg(feature = "mlx")]
    {
        ffi::mlx_version().unwrap_or_else(|_| "Unknown".to_string())
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        "MLX not available".to_string()
    }
}

/// Get memory usage statistics
pub fn memory_stats() -> Result<MemoryStats> {
    if !is_available() {
        return Err(MLXError::NotAvailable("MLX not available on this platform".to_string()));
    }
    
    #[cfg(feature = "mlx")]
    {
        ffi::mlx_memory_stats()
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        Err(MLXError::NotAvailable("MLX feature not enabled".to_string()))
    }
}

/// MLX memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total memory available to MLX (bytes)
    pub total_memory: u64,
    /// Currently allocated memory (bytes)
    pub allocated_memory: u64,
    /// Memory reserved but not allocated (bytes)
    pub reserved_memory: u64,
    /// Peak memory usage (bytes)
    pub peak_memory: u64,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            total_memory: 0,
            allocated_memory: 0,
            reserved_memory: 0,
            peak_memory: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        // Should not panic
        let _ = init();
    }

    #[test]
    fn test_availability_check() {
        // Should not panic
        let available = is_available();
        println!("MLX available: {}", available);
    }

    #[test]
    fn test_version() {
        let version = version();
        println!("MLX version: {}", version);
        assert!(!version.is_empty());
    }

    #[test]
    fn test_memory_stats() {
        if is_available() {
            match memory_stats() {
                Ok(stats) => {
                    println!("Memory stats: {:?}", stats);
                }
                Err(e) => {
                    println!("Failed to get memory stats: {}", e);
                }
            }
        }
    }
}