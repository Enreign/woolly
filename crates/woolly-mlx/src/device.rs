//! MLX device management and selection
//!
//! This module handles MLX device detection, selection, and management
//! for Apple Silicon GPU operations.

use std::fmt;
use std::sync::{Arc, Mutex, Once};
use tracing::{debug, info, warn};

use crate::error::{MLXError, Result};
use crate::platform;

/// MLX device types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// Apple Silicon GPU
    GPU,
    /// Apple Silicon CPU (unified memory)
    CPU,
    /// Automatic device selection
    Auto,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::GPU => write!(f, "mlx:gpu"),
            Device::CPU => write!(f, "mlx:cpu"),
            Device::Auto => write!(f, "mlx:auto"),
        }
    }
}

impl From<Device> for woolly_tensor::backend::Device {
    fn from(device: Device) -> Self {
        match device {
            Device::GPU => woolly_tensor::backend::Device::Metal,
            Device::CPU => woolly_tensor::backend::Device::Cpu,
            Device::Auto => woolly_tensor::backend::Device::Metal,
        }
    }
}

/// MLX device manager
#[derive(Debug)]
pub struct MLXDevice {
    device_type: Device,
    properties: DeviceProperties,
    is_available: bool,
}

/// Device properties and capabilities
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: Option<u64>,
    /// Available memory in bytes
    pub available_memory: Option<u64>,
    /// GPU cores (if applicable)
    pub gpu_cores: Option<u32>,
    /// Maximum threads per threadgroup
    pub max_threads_per_threadgroup: Option<u32>,
    /// Maximum buffer size
    pub max_buffer_size: Option<u64>,
    /// Supports unified memory
    pub unified_memory: bool,
    /// Supports half precision (f16)
    pub supports_f16: bool,
    /// Supports bfloat16
    pub supports_bf16: bool,
    /// Supports int8 operations
    pub supports_int8: bool,
    /// Maximum texture size
    pub max_texture_size: Option<(u32, u32)>,
}

impl Default for DeviceProperties {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            total_memory: None,
            available_memory: None,
            gpu_cores: None,
            max_threads_per_threadgroup: None,
            max_buffer_size: None,
            unified_memory: false,
            supports_f16: false,
            supports_bf16: false,
            supports_int8: false,
            max_texture_size: None,
        }
    }
}

static DEVICE_MANAGER: Once = Once::new();
static mut DEVICE_MANAGER_INSTANCE: Option<Arc<Mutex<DeviceManager>>> = None;

/// Global device manager
struct DeviceManager {
    devices: Vec<MLXDevice>,
    default_device: Option<Device>,
}

impl DeviceManager {
    fn new() -> Self {
        let mut manager = Self {
            devices: Vec::new(),
            default_device: None,
        };
        
        manager.discover_devices();
        manager
    }
    
    fn discover_devices(&mut self) {
        debug!("Discovering MLX devices");
        
        if !platform::is_apple_silicon() {
            warn!("Not on Apple Silicon, no MLX devices available");
            return;
        }
        
        // Discover GPU device
        if let Ok(gpu_device) = self.create_gpu_device() {
            self.devices.push(gpu_device);
            if self.default_device.is_none() {
                self.default_device = Some(Device::GPU);
            }
        }
        
        // Discover CPU device (always available on Apple Silicon)
        if let Ok(cpu_device) = self.create_cpu_device() {
            self.devices.push(cpu_device);
            if self.default_device.is_none() {
                self.default_device = Some(Device::CPU);
            }
        }
        
        info!("Discovered {} MLX devices", self.devices.len());
    }
    
    fn create_gpu_device(&self) -> Result<MLXDevice> {
        let system_info = platform::system_info();
        
        let properties = DeviceProperties {
            name: format!("Apple Silicon GPU ({} cores)", 
                system_info.gpu_cores.unwrap_or(8)),
            total_memory: system_info.total_memory,
            available_memory: system_info.total_memory.map(|m| (m as f64 * 0.8) as u64),
            gpu_cores: system_info.gpu_cores,
            max_threads_per_threadgroup: Some(1024), // Typical value for Apple Silicon
            max_buffer_size: Some(1 << 30), // 1GB max buffer size
            unified_memory: true,
            supports_f16: true,
            supports_bf16: true,
            supports_int8: true,
            max_texture_size: Some((16384, 16384)),
        };
        
        Ok(MLXDevice {
            device_type: Device::GPU,
            properties,
            is_available: true,
        })
    }
    
    fn create_cpu_device(&self) -> Result<MLXDevice> {
        let system_info = platform::system_info();
        
        let properties = DeviceProperties {
            name: "Apple Silicon CPU".to_string(),
            total_memory: system_info.total_memory,
            available_memory: system_info.total_memory,
            gpu_cores: None,
            max_threads_per_threadgroup: None,
            max_buffer_size: system_info.total_memory,
            unified_memory: true,
            supports_f16: true,
            supports_bf16: true,
            supports_int8: true,
            max_texture_size: None,
        };
        
        Ok(MLXDevice {
            device_type: Device::CPU,
            properties,
            is_available: true,
        })
    }
}

impl MLXDevice {
    /// Create a new MLX device of the specified type
    pub fn new(device_type: Device) -> Result<Self> {
        get_device_manager()
            .lock()
            .unwrap()
            .devices
            .iter()
            .find(|d| d.device_type == device_type)
            .cloned()
            .ok_or_else(|| MLXError::InvalidDevice(format!("Device {:?} not available", device_type)))
    }
    
    /// Get the default MLX device
    pub fn default() -> Result<Self> {
        let manager = get_device_manager();
        let manager = manager.lock().unwrap();
        
        if let Some(default_type) = manager.default_device {
            manager.devices
                .iter()
                .find(|d| d.device_type == default_type)
                .cloned()
                .ok_or_else(|| MLXError::InvalidDevice("Default device not found".to_string()))
        } else {
            Err(MLXError::NotAvailable("No MLX devices available".to_string()))
        }
    }
    
    /// Get the device type
    pub fn device_type(&self) -> Device {
        self.device_type
    }
    
    /// Get device properties
    pub fn properties(&self) -> &DeviceProperties {
        &self.properties
    }
    
    /// Check if device is available
    pub fn is_available(&self) -> bool {
        self.is_available
    }
    
    /// Get current memory usage
    pub fn memory_usage(&self) -> Result<MemoryUsage> {
        if !self.is_available {
            return Err(MLXError::InvalidDevice("Device not available".to_string()));
        }
        
        #[cfg(feature = "mlx")]
        {
            crate::ffi::mlx_device_memory_usage(self.device_type)
        }
        
        #[cfg(not(feature = "mlx"))]
        {
            // Return mock data for testing
            Ok(MemoryUsage {
                total: self.properties.total_memory.unwrap_or(0),
                allocated: 0,
                available: self.properties.available_memory.unwrap_or(0),
                peak: 0,
            })
        }
    }
    
    /// Synchronize device operations
    pub fn synchronize(&self) -> Result<()> {
        if !self.is_available {
            return Err(MLXError::InvalidDevice("Device not available".to_string()));
        }
        
        #[cfg(feature = "mlx")]
        {
            crate::ffi::mlx_device_synchronize(self.device_type)
        }
        
        #[cfg(not(feature = "mlx"))]
        {
            Ok(())
        }
    }
    
    /// Set memory limit for this device
    pub fn set_memory_limit(&self, limit: u64) -> Result<()> {
        if !self.is_available {
            return Err(MLXError::InvalidDevice("Device not available".to_string()));
        }
        
        #[cfg(feature = "mlx")]
        {
            crate::ffi::mlx_set_memory_limit(self.device_type, limit)
        }
        
        #[cfg(not(feature = "mlx"))]
        {
            debug!("Setting memory limit to {} bytes (mock)", limit);
            Ok(())
        }
    }
    
    /// Check if a specific data type is supported
    pub fn supports_dtype(&self, dtype: &str) -> bool {
        match dtype {
            "f32" => true,
            "f16" => self.properties.supports_f16,
            "bf16" => self.properties.supports_bf16,
            "i8" => self.properties.supports_int8,
            "u8" => true,
            "i32" => true,
            "bool" => true,
            _ => false,
        }
    }
    
    /// Get optimal batch size for operations on this device
    pub fn optimal_batch_size(&self) -> u32 {
        match self.device_type {
            Device::GPU => {
                // Base on GPU cores
                let gpu_cores = self.properties.gpu_cores.unwrap_or(8);
                (gpu_cores * 4).max(8).min(64)
            }
            Device::CPU => {
                // Conservative batch size for CPU
                8
            }
            Device::Auto => {
                // Use GPU settings if available
                let gpu_cores = self.properties.gpu_cores.unwrap_or(8);
                (gpu_cores * 4).max(8).min(32)
            }
        }
    }
}

impl Clone for MLXDevice {
    fn clone(&self) -> Self {
        Self {
            device_type: self.device_type,
            properties: self.properties.clone(),
            is_available: self.is_available,
        }
    }
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Total memory in bytes
    pub total: u64,
    /// Currently allocated memory in bytes
    pub allocated: u64,
    /// Available memory in bytes
    pub available: u64,
    /// Peak memory usage in bytes
    pub peak: u64,
}

/// Get the global device manager
fn get_device_manager() -> &'static Arc<Mutex<DeviceManager>> {
    unsafe {
        DEVICE_MANAGER.call_once(|| {
            DEVICE_MANAGER_INSTANCE = Some(Arc::new(Mutex::new(DeviceManager::new())));
        });
        DEVICE_MANAGER_INSTANCE.as_ref().unwrap()
    }
}

/// List all available MLX devices
pub fn list_devices() -> Vec<MLXDevice> {
    get_device_manager()
        .lock()
        .unwrap()
        .devices
        .clone()
}

/// Get the default MLX device
pub fn default_device() -> Result<MLXDevice> {
    MLXDevice::default()
}

/// Create a device of the specified type
pub fn create_device(device_type: Device) -> Result<MLXDevice> {
    MLXDevice::new(device_type)
}

/// Check if GPU is available
pub fn gpu_available() -> bool {
    get_device_manager()
        .lock()
        .unwrap()
        .devices
        .iter()
        .any(|d| d.device_type == Device::GPU && d.is_available)
}

/// Get best available device for the given operation
pub fn best_device_for_operation(op_type: &str, data_size: Option<u64>) -> Result<MLXDevice> {
    let devices = list_devices();
    
    if devices.is_empty() {
        return Err(MLXError::NotAvailable("No MLX devices available".to_string()));
    }
    
    // For small operations, CPU might be faster due to launch overhead
    let prefer_cpu = data_size.map_or(false, |size| size < 1024 * 1024); // < 1MB
    
    match op_type {
        "matmul" | "conv" | "attention" => {
            // Prefer GPU for compute-heavy operations
            if !prefer_cpu && gpu_available() {
                create_device(Device::GPU)
            } else {
                create_device(Device::CPU)
            }
        }
        "copy" | "slice" | "reshape" => {
            // Memory operations are fine on either device due to unified memory
            default_device()
        }
        _ => {
            // Default to best available device
            default_device()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_discovery() {
        let devices = list_devices();
        println!("Found {} MLX devices", devices.len());
        
        for device in &devices {
            println!("Device: {:?}", device);
            println!("  Properties: {:?}", device.properties());
        }
    }
    
    #[test]
    fn test_default_device() {
        match default_device() {
            Ok(device) => {
                println!("Default device: {:?}", device.device_type());
                println!("  Available: {}", device.is_available());
            }
            Err(e) => {
                println!("No default device: {}", e);
            }
        }
    }
    
    #[test]
    fn test_gpu_availability() {
        let gpu_avail = gpu_available();
        println!("GPU available: {}", gpu_avail);
    }
    
    #[test]
    fn test_device_properties() {
        if let Ok(device) = default_device() {
            let props = device.properties();
            println!("Device properties:");
            println!("  Name: {}", props.name);
            println!("  Unified memory: {}", props.unified_memory);
            println!("  Supports f16: {}", props.supports_f16);
            println!("  GPU cores: {:?}", props.gpu_cores);
            
            if let Some(total_mem) = props.total_memory {
                println!("  Total memory: {} GB", total_mem / (1024 * 1024 * 1024));
            }
        }
    }
    
    #[test]
    fn test_memory_usage() {
        if let Ok(device) = default_device() {
            match device.memory_usage() {
                Ok(usage) => {
                    println!("Memory usage:");
                    println!("  Total: {} MB", usage.total / (1024 * 1024));
                    println!("  Allocated: {} MB", usage.allocated / (1024 * 1024));
                    println!("  Available: {} MB", usage.available / (1024 * 1024));
                }
                Err(e) => {
                    println!("Failed to get memory usage: {}", e);
                }
            }
        }
    }
    
    #[test]
    fn test_best_device_selection() {
        // Test for different operation types
        let ops = vec![
            ("matmul", Some(1024 * 1024 * 4)), // 4MB
            ("copy", Some(1024)),              // 1KB
            ("conv", Some(1024 * 1024 * 16)),  // 16MB
        ];
        
        for (op, size) in ops {
            match best_device_for_operation(op, size) {
                Ok(device) => {
                    println!("Best device for {}: {:?}", op, device.device_type());
                }
                Err(e) => {
                    println!("Failed to select device for {}: {}", op, e);
                }
            }
        }
    }
}