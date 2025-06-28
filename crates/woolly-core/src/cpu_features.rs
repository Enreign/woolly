//! CPU Feature Detection for SIMD Optimization
//! 
//! This module provides runtime detection of CPU features to enable
//! optimal SIMD instruction selection similar to llama.cpp's approach.

use std::sync::OnceLock;

/// CPU feature detection results
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    // x86_64 features
    pub has_sse2: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_fma: bool,
    
    // ARM64 features
    pub has_neon: bool,
    pub has_dotprod: bool,
    
    // Architecture info
    pub is_x86_64: bool,
    pub is_aarch64: bool,
    pub cache_line_size: usize,
}

impl Default for CpuFeatures {
    fn default() -> Self {
        Self {
            has_sse2: false,
            has_avx: false,
            has_avx2: false,
            has_fma: false,
            has_neon: false,
            has_dotprod: false,
            is_x86_64: cfg!(target_arch = "x86_64"),
            is_aarch64: cfg!(target_arch = "aarch64"),
            cache_line_size: 64, // Default cache line size
        }
    }
}

impl CpuFeatures {
    /// Detect CPU features at runtime
    pub fn detect() -> Self {
        let mut features = Self::default();
        
        #[cfg(target_arch = "x86_64")]
        {
            features.has_sse2 = is_x86_feature_detected!("sse2");
            features.has_avx = is_x86_feature_detected!("avx");
            features.has_avx2 = is_x86_feature_detected!("avx2");
            features.has_fma = is_x86_feature_detected!("fma");
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            features.has_neon = true; // NEON is mandatory on aarch64
            // Note: Rust doesn't have std detection for ARM features yet
            // In production, we'd use OS-specific detection
            features.has_dotprod = Self::detect_arm_dotprod();
        }
        
        features.cache_line_size = Self::detect_cache_line_size();
        
        features
    }
    
    /// Get cached CPU features (computed once)
    pub fn get() -> &'static CpuFeatures {
        static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();
        CPU_FEATURES.get_or_init(CpuFeatures::detect)
    }
    
    /// Get the best available SIMD instruction set
    pub fn best_simd_level(&self) -> SimdLevel {
        #[cfg(target_arch = "x86_64")]
        {
            if self.has_avx2 {
                SimdLevel::Avx2
            } else if self.has_avx {
                SimdLevel::Avx
            } else if self.has_sse2 {
                SimdLevel::Sse2
            } else {
                SimdLevel::Scalar
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if self.has_dotprod {
                SimdLevel::NeonDotprod
            } else if self.has_neon {
                SimdLevel::Neon
            } else {
                SimdLevel::Scalar
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdLevel::Scalar
        }
    }
    
    /// Get optimal memory alignment for SIMD operations
    pub fn simd_alignment(&self) -> usize {
        match self.best_simd_level() {
            SimdLevel::Avx2 => 32,  // 256-bit alignment
            SimdLevel::Avx => 32,   // 256-bit alignment
            SimdLevel::Sse2 => 16,  // 128-bit alignment
            SimdLevel::Neon => 16,  // 128-bit alignment
            SimdLevel::NeonDotprod => 16, // 128-bit alignment
            SimdLevel::Scalar => 8, // Basic alignment
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    fn detect_arm_dotprod() -> bool {
        // In production, we'd read /proc/cpuinfo or use OS-specific APIs
        // For now, conservatively assume false
        false
    }
    
    fn detect_cache_line_size() -> usize {
        // Try to detect actual cache line size
        // Default to 64 bytes which is common
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size") {
                if let Ok(size) = content.trim().parse::<usize>() {
                    return size;
                }
            }
        }
        
        64 // Default cache line size
    }
}

/// SIMD instruction set levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    Scalar = 0,
    Sse2 = 1,
    Avx = 2,
    Avx2 = 3,
    Neon = 4,
    NeonDotprod = 5,
}

impl SimdLevel {
    /// Get the optimal number of f32 elements to process per SIMD instruction
    pub fn f32_lane_count(&self) -> usize {
        match self {
            SimdLevel::Avx2 | SimdLevel::Avx => 8,  // 8 x f32 in 256-bit
            SimdLevel::Sse2 | SimdLevel::Neon | SimdLevel::NeonDotprod => 4, // 4 x f32 in 128-bit
            SimdLevel::Scalar => 1,
        }
    }
    
    /// Get the optimal number of i8 elements to process per SIMD instruction
    pub fn i8_lane_count(&self) -> usize {
        match self {
            SimdLevel::Avx2 | SimdLevel::Avx => 32, // 32 x i8 in 256-bit
            SimdLevel::Sse2 | SimdLevel::Neon | SimdLevel::NeonDotprod => 16, // 16 x i8 in 128-bit
            SimdLevel::Scalar => 1,
        }
    }
}

/// Kernel dispatcher for SIMD operations
pub struct SimdDispatcher {
    cpu_features: &'static CpuFeatures,
    simd_level: SimdLevel,
}

impl SimdDispatcher {
    /// Create a new SIMD dispatcher
    pub fn new() -> Self {
        let cpu_features = CpuFeatures::get();
        let simd_level = cpu_features.best_simd_level();
        
        Self {
            cpu_features,
            simd_level,
        }
    }
    
    /// Get CPU features
    pub fn cpu_features(&self) -> &CpuFeatures {
        self.cpu_features
    }
    
    /// Get SIMD level
    pub fn simd_level(&self) -> SimdLevel {
        self.simd_level
    }
    
    /// Check if specific SIMD level is available
    pub fn supports_simd_level(&self, level: SimdLevel) -> bool {
        self.simd_level >= level
    }
}

impl Default for SimdDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_features_detection() {
        let features = CpuFeatures::detect();
        
        // Basic sanity checks
        #[cfg(target_arch = "x86_64")]
        {
            assert!(features.is_x86_64);
            assert!(!features.is_aarch64);
            // SSE2 is guaranteed on x86_64
            assert!(features.has_sse2);
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            assert!(features.is_aarch64);
            assert!(!features.is_x86_64);
            // NEON is mandatory on aarch64
            assert!(features.has_neon);
        }
        
        assert!(features.cache_line_size >= 16);
        assert!(features.cache_line_size <= 256);
    }
    
    #[test]
    fn test_simd_dispatcher() {
        let dispatcher = SimdDispatcher::new();
        let level = dispatcher.simd_level();
        
        // Should always support at least scalar
        assert!(dispatcher.supports_simd_level(SimdLevel::Scalar));
        
        // Lane counts should be reasonable
        assert!(level.f32_lane_count() >= 1);
        assert!(level.i8_lane_count() >= 1);
        assert!(level.f32_lane_count() <= 16);
        assert!(level.i8_lane_count() <= 64);
    }
    
    #[test]
    fn test_cached_features() {
        let features1 = CpuFeatures::get();
        let features2 = CpuFeatures::get();
        
        // Should be the same instance (cached)
        assert!(std::ptr::eq(features1, features2));
    }
}