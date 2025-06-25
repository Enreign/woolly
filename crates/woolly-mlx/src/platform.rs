//! Platform detection for MLX availability
//!
//! This module provides functions to detect Apple Silicon hardware and MLX availability.

use std::sync::Once;
use tracing::{debug, info, warn};

static PLATFORM_INIT: Once = Once::new();
static mut IS_APPLE_SILICON: bool = false;
static mut SYSTEM_INFO: Option<SystemInfo> = None;

/// System information for Apple Silicon detection
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// CPU architecture (e.g., "arm64", "x86_64")
    pub arch: String,
    /// Operating system (e.g., "macos", "linux")
    pub os: String,
    /// CPU brand (e.g., "Apple M1", "Apple M2")
    pub cpu_brand: Option<String>,
    /// Number of performance cores
    pub performance_cores: Option<u32>,
    /// Number of efficiency cores
    pub efficiency_cores: Option<u32>,
    /// Total memory in bytes
    pub total_memory: Option<u64>,
    /// GPU cores count
    pub gpu_cores: Option<u32>,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            arch: "unknown".to_string(),
            os: "unknown".to_string(),
            cpu_brand: None,
            performance_cores: None,
            efficiency_cores: None,
            total_memory: None,
            gpu_cores: None,
        }
    }
}

/// Initialize platform detection
fn init_platform_detection() {
    PLATFORM_INIT.call_once(|| {
        unsafe {
            let info = detect_system_info();
            IS_APPLE_SILICON = is_apple_silicon_system(&info);
            SYSTEM_INFO = Some(info);
            
            if IS_APPLE_SILICON {
                info!("Detected Apple Silicon system");
                if let Some(ref info) = SYSTEM_INFO {
                    debug!("System info: {:?}", info);
                }
            } else {
                debug!("Not an Apple Silicon system");
            }
        }
    });
}

/// Check if the current system is Apple Silicon
pub fn is_apple_silicon() -> bool {
    init_platform_detection();
    unsafe { IS_APPLE_SILICON }
}

/// Get detailed system information
pub fn system_info() -> SystemInfo {
    init_platform_detection();
    unsafe { 
        SYSTEM_INFO.clone().unwrap_or_default()
    }
}

/// Detect system information
fn detect_system_info() -> SystemInfo {
    let mut info = SystemInfo::default();
    
    // Detect architecture
    info.arch = std::env::consts::ARCH.to_string();
    info.os = std::env::consts::OS.to_string();
    
    #[cfg(target_os = "macos")]
    {
        info = detect_macos_info(info);
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        debug!("Not running on macOS, MLX not available");
    }
    
    info
}

/// Detect if this is an Apple Silicon system
fn is_apple_silicon_system(info: &SystemInfo) -> bool {
    // Must be macOS with ARM64 architecture
    info.os == "macos" && info.arch == "aarch64"
}

/// Detect macOS-specific system information
#[cfg(target_os = "macos")]
fn detect_macos_info(mut info: SystemInfo) -> SystemInfo {
    use std::ffi::CStr;
    use std::ptr;
    
    // Get CPU brand using sysctl
    if let Ok(brand) = get_sysctl_string("machdep.cpu.brand_string") {
        info.cpu_brand = Some(brand);
    }
    
    // Get core counts
    if let Ok(perf_cores) = get_sysctl_u32("hw.perflevel0.logicalcpu") {
        info.performance_cores = Some(perf_cores);
    }
    
    if let Ok(eff_cores) = get_sysctl_u32("hw.perflevel1.logicalcpu") {
        info.efficiency_cores = Some(eff_cores);
    }
    
    // Get total memory
    if let Ok(memory) = get_sysctl_u64("hw.memsize") {
        info.total_memory = Some(memory);
    }
    
    // Try to detect GPU cores (this is approximate)
    info.gpu_cores = detect_gpu_cores(&info);
    
    info
}

/// Get sysctl string value
#[cfg(target_os = "macos")]
fn get_sysctl_string(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    use std::ffi::CString;
    
    let name_cstr = CString::new(name)?;
    let mut size = 0;
    
    // First call to get the size
    let result = unsafe {
        libc::sysctlbyname(
            name_cstr.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null(),
            0,
        )
    };
    
    if result != 0 {
        return Err(format!("sysctlbyname failed for {}", name).into());
    }
    
    // Second call to get the actual value
    let mut buffer = vec![0u8; size];
    let result = unsafe {
        libc::sysctlbyname(
            name_cstr.as_ptr(),
            buffer.as_mut_ptr() as *mut libc::c_void,
            &mut size,
            std::ptr::null(),
            0,
        )
    };
    
    if result != 0 {
        return Err(format!("sysctlbyname failed for {}", name).into());
    }
    
    // Convert to string, removing null terminator
    if let Some(null_pos) = buffer.iter().position(|&b| b == 0) {
        buffer.truncate(null_pos);
    }
    
    Ok(String::from_utf8(buffer)?)
}

/// Get sysctl u32 value
#[cfg(target_os = "macos")]
fn get_sysctl_u32(name: &str) -> Result<u32, Box<dyn std::error::Error>> {
    use std::ffi::CString;
    
    let name_cstr = CString::new(name)?;
    let mut value: u32 = 0;
    let mut size = std::mem::size_of::<u32>();
    
    let result = unsafe {
        libc::sysctlbyname(
            name_cstr.as_ptr(),
            &mut value as *mut u32 as *mut libc::c_void,
            &mut size,
            std::ptr::null(),
            0,
        )
    };
    
    if result != 0 {
        return Err(format!("sysctlbyname failed for {}", name).into());
    }
    
    Ok(value)
}

/// Get sysctl u64 value
#[cfg(target_os = "macos")]
fn get_sysctl_u64(name: &str) -> Result<u64, Box<dyn std::error::Error>> {
    use std::ffi::CString;
    
    let name_cstr = CString::new(name)?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    
    let result = unsafe {
        libc::sysctlbyname(
            name_cstr.as_ptr(),
            &mut value as *mut u64 as *mut libc::c_void,
            &mut size,
            std::ptr::null(),
            0,
        )
    };
    
    if result != 0 {
        return Err(format!("sysctlbyname failed for {}", name).into());
    }
    
    Ok(value)
}

/// Attempt to detect GPU cores (approximate)
#[cfg(target_os = "macos")]
fn detect_gpu_cores(info: &SystemInfo) -> Option<u32> {
    // This is a rough estimation based on known Apple Silicon specs
    if let Some(ref brand) = info.cpu_brand {
        if brand.contains("M1") {
            if brand.contains("M1 Pro") {
                Some(16) // M1 Pro has 16 GPU cores
            } else if brand.contains("M1 Max") {
                Some(32) // M1 Max has 32 GPU cores
            } else if brand.contains("M1 Ultra") {
                Some(64) // M1 Ultra has 64 GPU cores
            } else {
                Some(8) // Base M1 has 8 GPU cores
            }
        } else if brand.contains("M2") {
            if brand.contains("M2 Pro") {
                Some(19) // M2 Pro has up to 19 GPU cores
            } else if brand.contains("M2 Max") {
                Some(38) // M2 Max has up to 38 GPU cores
            } else if brand.contains("M2 Ultra") {
                Some(76) // M2 Ultra has up to 76 GPU cores
            } else {
                Some(10) // M2 has up to 10 GPU cores
            }
        } else if brand.contains("M3") {
            if brand.contains("M3 Pro") {
                Some(18) // M3 Pro has up to 18 GPU cores
            } else if brand.contains("M3 Max") {
                Some(40) // M3 Max has up to 40 GPU cores
            } else {
                Some(10) // M3 has up to 10 GPU cores
            }
        } else {
            None
        }
    } else {
        None
    }
}

#[cfg(not(target_os = "macos"))]
fn detect_macos_info(info: SystemInfo) -> SystemInfo {
    info
}

/// Check if MLX framework is available
pub fn is_mlx_available() -> bool {
    if !is_apple_silicon() {
        return false;
    }
    
    // On Apple Silicon, check if MLX framework is available
    #[cfg(feature = "mlx")]
    {
        // In a real implementation, this would check for MLX library availability
        // For now, assume it's available on Apple Silicon
        true
    }
    
    #[cfg(not(feature = "mlx"))]
    {
        false
    }
}

/// Get recommended memory allocation for MLX
pub fn recommended_mlx_memory() -> Option<u64> {
    let info = system_info();
    
    if let Some(total_memory) = info.total_memory {
        // Reserve 80% of total memory for MLX (unified memory architecture)
        Some((total_memory as f64 * 0.8) as u64)
    } else {
        None
    }
}

/// Get optimal thread count for CPU operations
pub fn optimal_cpu_threads() -> u32 {
    let info = system_info();
    
    // Use performance cores for CPU operations
    if let Some(perf_cores) = info.performance_cores {
        perf_cores
    } else {
        // Fallback to std::thread::available_parallelism
        std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1)
    }
}

/// Performance characteristics of the detected Apple Silicon chip
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// GPU cores available
    pub gpu_cores: u32,
    /// Performance CPU cores
    pub performance_cores: u32,
    /// Efficiency CPU cores
    pub efficiency_cores: u32,
    /// Memory bandwidth (GB/s) - approximate
    pub memory_bandwidth: f32,
    /// Recommended batch size for ML operations
    pub recommended_batch_size: u32,
}

/// Get performance characteristics of the current Apple Silicon chip
pub fn performance_profile() -> Option<PerformanceProfile> {
    let info = system_info();
    
    if !is_apple_silicon() {
        return None;
    }
    
    let gpu_cores = info.gpu_cores.unwrap_or(8);
    let perf_cores = info.performance_cores.unwrap_or(4);
    let eff_cores = info.efficiency_cores.unwrap_or(4);
    
    // Approximate memory bandwidth based on chip type
    let memory_bandwidth = if let Some(ref brand) = info.cpu_brand {
        if brand.contains("M1 Ultra") {
            800.0 // M1 Ultra has ~800 GB/s
        } else if brand.contains("M1 Max") || brand.contains("M2 Max") {
            400.0 // M1/M2 Max have ~400 GB/s
        } else if brand.contains("M1 Pro") || brand.contains("M2 Pro") || brand.contains("M3 Pro") {
            200.0 // Pro variants have ~200 GB/s
        } else {
            100.0 // Base models have ~100 GB/s
        }
    } else {
        100.0 // Conservative estimate
    };
    
    // Recommended batch size based on GPU cores and memory
    let recommended_batch_size = (gpu_cores * 4).min(64).max(8);
    
    Some(PerformanceProfile {
        gpu_cores,
        performance_cores: perf_cores,
        efficiency_cores: eff_cores,
        memory_bandwidth,
        recommended_batch_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_platform_detection() {
        let is_as = is_apple_silicon();
        println!("Is Apple Silicon: {}", is_as);
        
        let info = system_info();
        println!("System info: {:?}", info);
    }
    
    #[test]
    fn test_mlx_availability() {
        let available = is_mlx_available();
        println!("MLX available: {}", available);
    }
    
    #[test]
    fn test_performance_profile() {
        if let Some(profile) = performance_profile() {
            println!("Performance profile: {:?}", profile);
        } else {
            println!("No performance profile available");
        }
    }
    
    #[test]
    fn test_memory_recommendations() {
        if let Some(mem) = recommended_mlx_memory() {
            println!("Recommended MLX memory: {} GB", mem / (1024 * 1024 * 1024));
        }
        
        let threads = optimal_cpu_threads();
        println!("Optimal CPU threads: {}", threads);
    }
}