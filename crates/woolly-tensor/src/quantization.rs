//! Quantization schemes and traits for tensor compression

use std::fmt::Debug;
use thiserror::Error;
use half;

/// Errors specific to quantization operations
#[derive(Error, Debug)]
pub enum QuantizationError {
    /// Quantization scheme is not supported
    #[error("Unsupported quantization scheme: {0}")]
    UnsupportedScheme(String),
    
    /// Invalid parameters provided for quantization
    #[error("Invalid quantization parameters: {0}")]
    InvalidParameters(String),
    
    /// Value cannot be represented in the target quantization format
    #[error("Quantization overflow: value {value} cannot be represented in {bits} bits")]
    Overflow { 
        /// The value that caused the overflow
        value: f32, 
        /// The target bit width
        bits: u8 
    },
}

/// Supported quantization schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationScheme {
    /// 8-bit integer quantization
    Int8,
    /// 4-bit integer quantization  
    Int4,
    /// Q4_0 - 4-bit quantization (llama.cpp style)
    Q4_0,
    /// Q4_1 - 4-bit quantization with offset (llama.cpp style)
    Q4_1,
    /// Q5_0 - 5-bit quantization (llama.cpp style)
    Q5_0,
    /// Q5_1 - 5-bit quantization with offset (llama.cpp style)
    Q5_1,
    /// Q8_0 - 8-bit quantization (llama.cpp style)
    Q8_0,
    /// K-quants: 2-bit quantization
    Q2K,
    /// K-quants: 3-bit quantization
    Q3K,
    /// K-quants: 4-bit quantization
    Q4K,
    /// K-quants: 5-bit quantization
    Q5K,
    /// K-quants: 6-bit quantization
    Q6K,
}

impl QuantizationScheme {
    /// Returns the number of bits per element for this scheme
    pub fn bits_per_element(&self) -> f32 {
        match self {
            QuantizationScheme::Int8 | QuantizationScheme::Q8_0 => 8.0,
            QuantizationScheme::Int4 | QuantizationScheme::Q4_0 | QuantizationScheme::Q4_1 => 4.0,
            QuantizationScheme::Q5_0 | QuantizationScheme::Q5_1 => 5.0,
            QuantizationScheme::Q2K => 2.625,  // Approximate due to block structure
            QuantizationScheme::Q3K => 3.4375, // Approximate
            QuantizationScheme::Q4K => 4.5,    // Approximate
            QuantizationScheme::Q5K => 5.5,    // Approximate
            QuantizationScheme::Q6K => 6.5625, // Approximate
        }
    }
    
    /// Returns the block size for this quantization scheme
    pub fn block_size(&self) -> usize {
        match self {
            QuantizationScheme::Int8 => 1,
            QuantizationScheme::Int4 => 1,
            QuantizationScheme::Q4_0 | QuantizationScheme::Q4_1 |
            QuantizationScheme::Q5_0 | QuantizationScheme::Q5_1 |
            QuantizationScheme::Q8_0 => 32,
            QuantizationScheme::Q2K | QuantizationScheme::Q3K |
            QuantizationScheme::Q4K | QuantizationScheme::Q5K |
            QuantizationScheme::Q6K => 256,
        }
    }
    
    /// Returns the bytes per element for storage
    pub fn bytes_per_element(&self) -> usize {
        // This is a simplified calculation - actual implementation would be more complex
        ((self.bits_per_element() * self.block_size() as f32 / 8.0).ceil() as usize + 
         std::mem::size_of::<f32>() * 2) / self.block_size()
    }
}

// llama.cpp compatible quantization block structures
/// Q4_0 block (32 elements per block)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    /// Scale factor (half precision)
    pub d: half::f16,
    /// Quantized values (4 bits per value, packed)
    pub qs: [u8; 16], // 32 values / 2 values per byte = 16 bytes
}

/// Q4_1 block (32 elements per block)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_1 {
    /// Scale factor (half precision)
    pub d: half::f16,
    /// Minimum value (half precision)
    pub m: half::f16,
    /// Quantized values (4 bits per value, packed)
    pub qs: [u8; 16], // 32 values / 2 values per byte = 16 bytes
}

/// Q5_0 block (32 elements per block)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5_0 {
    /// Scale factor (half precision)
    pub d: half::f16,
    /// High bits for 5th bit of each quantized value
    pub qh: [u8; 4],
    /// Quantized values (4 bits per value, packed)
    pub qs: [u8; 16], // 32 values / 2 values per byte = 16 bytes
}

/// Q5_1 block (32 elements per block)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5_1 {
    /// Scale factor (half precision)
    pub d: half::f16,
    /// Minimum value (half precision)
    pub m: half::f16,
    /// High bits for 5th bit of each quantized value
    pub qh: [u8; 4],
    /// Quantized values (4 bits per value, packed)
    pub qs: [u8; 16], // 32 values / 2 values per byte = 16 bytes
}

/// Q8_0 block (32 elements per block)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    /// Scale factor (half precision)
    pub d: half::f16,
    /// Quantized values (8 bits per value)
    pub qs: [i8; 32],
}

/// Quantized storage type for different schemes
#[derive(Debug, Clone)]
pub enum QuantizedStorage {
    /// Q4_0 quantization blocks (4-bit values with scale)
    Q4_0(Vec<BlockQ4_0>),
    /// Q4_1 quantization blocks (4-bit values with scale and minimum)
    Q4_1(Vec<BlockQ4_1>),
    /// Q5_0 quantization blocks (5-bit values with scale)
    Q5_0(Vec<BlockQ5_0>),
    /// Q5_1 quantization blocks (5-bit values with scale and minimum)
    Q5_1(Vec<BlockQ5_1>),
    /// Q8_0 quantization blocks (8-bit values with scale)
    Q8_0(Vec<BlockQ8_0>),
    /// Simple 8-bit integer quantization
    Int8(Vec<i8>),
    /// Simple 4-bit integer quantization (packed in u8)
    Int4(Vec<u8>),
}

/// Trait for quantization operations
pub trait Quantizer: Debug + Send + Sync {
    /// The quantized storage type
    type QuantizedStorage: Debug + Clone;
    
    /// Returns the quantization scheme
    fn scheme(&self) -> QuantizationScheme;
    
    /// Quantizes a slice of floating point values
    fn quantize(&self, values: &[f32]) -> Result<Self::QuantizedStorage, QuantizationError>;
    
    /// Dequantizes values back to floating point
    fn dequantize(&self, quantized: &Self::QuantizedStorage) -> Result<Vec<f32>, QuantizationError>;
    
    /// Returns the compression ratio achieved
    fn compression_ratio(&self) -> f32 {
        32.0 / self.scheme().bits_per_element()
    }
}

/// Parameters for symmetric quantization
#[derive(Debug, Clone)]
pub struct SymmetricQuantizationParams {
    /// Scale factor for quantization
    pub scale: f32,
    /// Number of bits for quantization
    pub bits: u8,
}

/// Parameters for asymmetric quantization
#[derive(Debug, Clone)]
pub struct AsymmetricQuantizationParams {
    /// Scale factor for quantization
    pub scale: f32,
    /// Zero point offset
    pub zero_point: i32,
    /// Number of bits for quantization
    pub bits: u8,
}

/// Block quantization parameters (llama.cpp style)
#[derive(Debug, Clone)]
pub struct BlockQuantizationParams {
    /// Block size
    pub block_size: usize,
    /// Scale per block
    pub scales: Vec<f32>,
    /// Optional offset per block
    pub offsets: Option<Vec<f32>>,
}

/// 8-bit integer quantizer
#[derive(Debug, Clone)]
pub struct Int8Quantizer {
    params: SymmetricQuantizationParams,
}

impl Int8Quantizer {
    /// Creates a new Int8 quantizer
    pub fn new() -> Self {
        Self {
            params: SymmetricQuantizationParams {
                scale: 1.0,
                bits: 8,
            }
        }
    }
    
    /// Computes quantization parameters from data
    pub fn compute_params(values: &[f32]) -> SymmetricQuantizationParams {
        let max_abs = values.iter()
            .map(|&v| v.abs())
            .fold(0.0f32, f32::max);
        
        let scale = if max_abs > 0.0 {
            max_abs / 127.0
        } else {
            1.0
        };
        
        SymmetricQuantizationParams { scale, bits: 8 }
    }
}

impl Quantizer for Int8Quantizer {
    type QuantizedStorage = QuantizedStorage;
    
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Int8
    }
    
    fn quantize(&self, values: &[f32]) -> Result<Self::QuantizedStorage, QuantizationError> {
        let mut quantized = Vec::with_capacity(values.len());
        for &val in values {
            let q = (val / self.params.scale).round() as i32;
            if q < -128 || q > 127 {
                return Err(QuantizationError::Overflow { value: val, bits: 8 });
            }
            quantized.push(q as i8);
        }
        Ok(QuantizedStorage::Int8(quantized))
    }
    
    fn dequantize(&self, quantized: &Self::QuantizedStorage) -> Result<Vec<f32>, QuantizationError> {
        match quantized {
            QuantizedStorage::Int8(values) => {
                Ok(values.iter().map(|&q| q as f32 * self.params.scale).collect())
            }
            _ => Err(QuantizationError::UnsupportedScheme("Expected Int8 storage".to_string())),
        }
    }
}

/// Q4_0 quantizer (llama.cpp compatible)
#[derive(Debug, Clone)]
pub struct Q4_0Quantizer;

impl Q4_0Quantizer {
    /// Creates a new Q4_0 quantizer
    pub fn new() -> Self {
        Self
    }
}

impl Quantizer for Q4_0Quantizer {
    type QuantizedStorage = QuantizedStorage;
    
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q4_0
    }
    
    fn quantize(&self, _values: &[f32]) -> Result<Self::QuantizedStorage, QuantizationError> {
        // TODO: Implement actual quantization for Q4_0
        // For now, return an error as we focus on dequantization
        Err(QuantizationError::UnsupportedScheme("Q4_0 quantization not yet implemented".to_string()))
    }
    
    fn dequantize(&self, quantized: &Self::QuantizedStorage) -> Result<Vec<f32>, QuantizationError> {
        match quantized {
            QuantizedStorage::Q4_0(blocks) => Ok(dequantize_q4_0(blocks)),
            _ => Err(QuantizationError::UnsupportedScheme("Expected Q4_0 storage".to_string())),
        }
    }
}

/// Q4_1 quantizer (llama.cpp compatible)
#[derive(Debug, Clone)]
pub struct Q4_1Quantizer;

impl Q4_1Quantizer {
    /// Creates a new Q4_1 quantizer
    pub fn new() -> Self {
        Self
    }
}

impl Quantizer for Q4_1Quantizer {
    type QuantizedStorage = QuantizedStorage;
    
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q4_1
    }
    
    fn quantize(&self, _values: &[f32]) -> Result<Self::QuantizedStorage, QuantizationError> {
        // TODO: Implement actual quantization for Q4_1
        // For now, return an error as we focus on dequantization
        Err(QuantizationError::UnsupportedScheme("Q4_1 quantization not yet implemented".to_string()))
    }
    
    fn dequantize(&self, quantized: &Self::QuantizedStorage) -> Result<Vec<f32>, QuantizationError> {
        match quantized {
            QuantizedStorage::Q4_1(blocks) => Ok(dequantize_q4_1(blocks)),
            _ => Err(QuantizationError::UnsupportedScheme("Expected Q4_1 storage".to_string())),
        }
    }
}

/// Q5_0 quantizer (llama.cpp compatible)
#[derive(Debug, Clone)]
pub struct Q5_0Quantizer;

impl Q5_0Quantizer {
    /// Creates a new Q5_0 quantizer
    pub fn new() -> Self {
        Self
    }
}

impl Quantizer for Q5_0Quantizer {
    type QuantizedStorage = QuantizedStorage;
    
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q5_0
    }
    
    fn quantize(&self, _values: &[f32]) -> Result<Self::QuantizedStorage, QuantizationError> {
        Err(QuantizationError::UnsupportedScheme("Q5_0 quantization not yet implemented".to_string()))
    }
    
    fn dequantize(&self, quantized: &Self::QuantizedStorage) -> Result<Vec<f32>, QuantizationError> {
        match quantized {
            QuantizedStorage::Q5_0(blocks) => Ok(dequantize_q5_0(blocks)),
            _ => Err(QuantizationError::UnsupportedScheme("Expected Q5_0 storage".to_string())),
        }
    }
}

/// Q5_1 quantizer (llama.cpp compatible)
#[derive(Debug, Clone)]
pub struct Q5_1Quantizer;

impl Q5_1Quantizer {
    /// Creates a new Q5_1 quantizer
    pub fn new() -> Self {
        Self
    }
}

impl Quantizer for Q5_1Quantizer {
    type QuantizedStorage = QuantizedStorage;
    
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q5_1
    }
    
    fn quantize(&self, _values: &[f32]) -> Result<Self::QuantizedStorage, QuantizationError> {
        Err(QuantizationError::UnsupportedScheme("Q5_1 quantization not yet implemented".to_string()))
    }
    
    fn dequantize(&self, quantized: &Self::QuantizedStorage) -> Result<Vec<f32>, QuantizationError> {
        match quantized {
            QuantizedStorage::Q5_1(blocks) => Ok(dequantize_q5_1(blocks)),
            _ => Err(QuantizationError::UnsupportedScheme("Expected Q5_1 storage".to_string())),
        }
    }
}

/// Q8_0 quantizer (llama.cpp compatible)
#[derive(Debug, Clone)]
pub struct Q8_0Quantizer;

impl Q8_0Quantizer {
    /// Creates a new Q8_0 quantizer
    pub fn new() -> Self {
        Self
    }
}

impl Quantizer for Q8_0Quantizer {
    type QuantizedStorage = QuantizedStorage;
    
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q8_0
    }
    
    fn quantize(&self, _values: &[f32]) -> Result<Self::QuantizedStorage, QuantizationError> {
        // TODO: Implement actual quantization for Q8_0
        // For now, return an error as we focus on dequantization
        Err(QuantizationError::UnsupportedScheme("Q8_0 quantization not yet implemented".to_string()))
    }
    
    fn dequantize(&self, quantized: &Self::QuantizedStorage) -> Result<Vec<f32>, QuantizationError> {
        match quantized {
            QuantizedStorage::Q8_0(blocks) => Ok(dequantize_q8_0(blocks)),
            _ => Err(QuantizationError::UnsupportedScheme("Expected Q8_0 storage".to_string())),
        }
    }
}

/// Trait for dequantization operations
pub trait Dequantizer: Debug + Send + Sync {
    /// The input quantized type
    type QuantizedInput;
    
    /// Dequantizes to f32
    fn dequantize_f32(&self, input: &Self::QuantizedInput) -> Result<Vec<f32>, QuantizationError>;
    
    /// Dequantizes to f16 (if supported)
    fn dequantize_f16(&self, input: &Self::QuantizedInput) -> Result<Vec<half::f16>, QuantizationError> {
        // Default implementation: dequantize to f32 then convert
        let f32_values = self.dequantize_f32(input)?;
        Ok(f32_values.into_iter().map(half::f16::from_f32).collect())
    }
}

// Constants for block sizes
const QK4_0: usize = 32;
const QK4_1: usize = 32;
const QK5_0: usize = 32;
const QK5_1: usize = 32;
const QK8_0: usize = 32;

/// Dequantizes Q4_0 blocks to f32 values
/// Implementation matches llama.cpp's dequantize_row_q4_0
pub fn dequantize_q4_0(blocks: &[BlockQ4_0]) -> Vec<f32> {
    let mut result = Vec::with_capacity(blocks.len() * QK4_0);
    
    for (i, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        
        // Each block produces 32 values
        for j in 0..(QK4_0 / 2) {
            let x0 = ((block.qs[j] & 0x0F) as i8) - 8;
            let x1 = ((block.qs[j] >> 4) as i8) - 8;
            
            // Follow llama.cpp layout: y[i*qk + j + 0] and y[i*qk + j + qk/2]
            let base_idx = i * QK4_0;
            result.resize(std::cmp::max(result.len(), base_idx + QK4_0), 0.0);
            result[base_idx + j] = x0 as f32 * d;
            result[base_idx + j + QK4_0 / 2] = x1 as f32 * d;
        }
    }
    
    result
}

/// Dequantizes Q4_1 blocks to f32 values
/// Implementation matches llama.cpp's dequantize_row_q4_1
pub fn dequantize_q4_1(blocks: &[BlockQ4_1]) -> Vec<f32> {
    let mut result = Vec::with_capacity(blocks.len() * QK4_1);
    result.resize(blocks.len() * QK4_1, 0.0);
    
    for (i, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let m = block.m.to_f32();
        
        // Each block produces 32 values
        for j in 0..(QK4_1 / 2) {
            let x0 = (block.qs[j] & 0x0F) as f32;
            let x1 = (block.qs[j] >> 4) as f32;
            
            // Follow llama.cpp layout: y[i*qk + j + 0] and y[i*qk + j + qk/2]
            let base_idx = i * QK4_1;
            result[base_idx + j] = x0 * d + m;
            result[base_idx + j + QK4_1 / 2] = x1 * d + m;
        }
    }
    
    result
}

/// Dequantizes Q5_0 blocks to f32 values
/// Implementation matches llama.cpp's dequantize_row_q5_0
pub fn dequantize_q5_0(blocks: &[BlockQ5_0]) -> Vec<f32> {
    let mut result = Vec::with_capacity(blocks.len() * QK5_0);
    result.resize(blocks.len() * QK5_0, 0.0);
    
    for (i, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        
        // Get the high bits as u32
        let qh = u32::from_le_bytes(block.qh);
        
        // Each block produces 32 values
        for j in 0..(QK5_0 / 2) {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            
            let x0 = (((block.qs[j] & 0x0F) as u32) | xh_0) as i32 - 16;
            let x1 = (((block.qs[j] >> 4) as u32) | xh_1) as i32 - 16;
            
            // Follow llama.cpp layout: y[i*qk + j + 0] and y[i*qk + j + qk/2]
            let base_idx = i * QK5_0;
            result[base_idx + j] = x0 as f32 * d;
            result[base_idx + j + QK5_0 / 2] = x1 as f32 * d;
        }
    }
    
    result
}

/// Dequantizes Q5_1 blocks to f32 values
/// Implementation matches llama.cpp's dequantize_row_q5_1
pub fn dequantize_q5_1(blocks: &[BlockQ5_1]) -> Vec<f32> {
    let mut result = Vec::with_capacity(blocks.len() * QK5_1);
    result.resize(blocks.len() * QK5_1, 0.0);
    
    for (i, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        let m = block.m.to_f32();
        
        // Get the high bits as u32
        let qh = u32::from_le_bytes(block.qh);
        
        // Each block produces 32 values
        for j in 0..(QK5_1 / 2) {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            
            let x0 = ((block.qs[j] & 0x0F) as u32) | xh_0;
            let x1 = ((block.qs[j] >> 4) as u32) | xh_1;
            
            // Follow llama.cpp layout: y[i*qk + j + 0] and y[i*qk + j + qk/2]
            let base_idx = i * QK5_1;
            result[base_idx + j] = x0 as f32 * d + m;
            result[base_idx + j + QK5_1 / 2] = x1 as f32 * d + m;
        }
    }
    
    result
}

/// Dequantizes Q8_0 blocks to f32 values
/// Implementation matches llama.cpp's dequantize_row_q8_0
pub fn dequantize_q8_0(blocks: &[BlockQ8_0]) -> Vec<f32> {
    let mut result = Vec::with_capacity(blocks.len() * QK8_0);
    result.resize(blocks.len() * QK8_0, 0.0);
    
    for (i, block) in blocks.iter().enumerate() {
        let d = block.d.to_f32();
        
        // Each block produces 32 values
        for (j, &q) in block.qs.iter().enumerate() {
            result[i * QK8_0 + j] = q as f32 * d;
        }
    }
    
    result
}

/// SIMD-optimized dequantization functions
pub mod simd {
    use super::*;
    
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    use std::arch::x86_64::*;
    
    #[cfg(target_feature = "avx2")]
    mod avx2 {
        use super::*;
        
        /// Highly optimized Q8_0 dequantization using AVX2
        #[target_feature(enable = "avx2")]
        pub unsafe fn dequantize_q8_0_avx2(blocks: &[BlockQ8_0]) -> Vec<f32> {
            let mut result = Vec::with_capacity(blocks.len() * QK8_0);
            result.resize(blocks.len() * QK8_0, 0.0);
            
            for (i, block) in blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let d_vec = _mm256_set1_ps(d);
                let base_idx = i * QK8_0;
                
                // Process 8 values at a time using AVX2
                for chunk_start in (0..QK8_0).step_by(8) {
                    // Load 8 int8 values
                    let q_i8 = _mm_loadl_epi64(block.qs.as_ptr().add(chunk_start) as *const __m128i);
                    
                    // Convert to int32
                    let q_i32 = _mm256_cvtepi8_epi32(q_i8);
                    
                    // Convert to float
                    let q_f32 = _mm256_cvtepi32_ps(q_i32);
                    
                    // Multiply by scale
                    let result_vec = _mm256_mul_ps(q_f32, d_vec);
                    
                    // Store result
                    _mm256_storeu_ps(result.as_mut_ptr().add(base_idx + chunk_start), result_vec);
                }
                
                // Handle remaining elements (should be 0 for QK8_0=32)
                for j in (QK8_0& !7)..QK8_0 {
                    result[base_idx + j] = block.qs[j] as f32 * d;
                }
            }
            
            result
        }
        
        /// Highly optimized Q4_0 dequantization using AVX2
        #[target_feature(enable = "avx2")]
        pub unsafe fn dequantize_q4_0_avx2(blocks: &[BlockQ4_0]) -> Vec<f32> {
            let mut result = Vec::with_capacity(blocks.len() * QK4_0);
            result.resize(blocks.len() * QK4_0, 0.0);
            
            // Constants for unpacking
            let mask_low = _mm_set1_epi8(0x0F);
            let offset = _mm256_set1_ps(-8.0);
            
            for (i, block) in blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let d_vec = _mm256_set1_ps(d);
                let base_idx = i * QK4_0;
                
                // Process 16 packed values (32 nibbles) at a time  
                for chunk_start in (0..16).step_by(8) {
                    // Load 8 bytes containing 16 packed 4-bit values
                    let packed = _mm_loadl_epi64(block.qs.as_ptr().add(chunk_start) as *const __m128i);
                    
                    // Unpack low nibbles
                    let low_nibbles = _mm_and_si128(packed, mask_low);
                    let low_i32 = _mm256_cvtepi8_epi32(low_nibbles);
                    let low_f32 = _mm256_cvtepi32_ps(low_i32);
                    let low_result = _mm256_fmadd_ps(low_f32, d_vec, _mm256_mul_ps(offset, d_vec));
                    
                    // Unpack high nibbles  
                    let high_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_low);
                    let high_i32 = _mm256_cvtepi8_epi32(high_nibbles);
                    let high_f32 = _mm256_cvtepi32_ps(high_i32);
                    let high_result = _mm256_fmadd_ps(high_f32, d_vec, _mm256_mul_ps(offset, d_vec));
                    
                    // Store results in the correct layout (matches llama.cpp)
                    _mm256_storeu_ps(result.as_mut_ptr().add(base_idx + chunk_start), low_result);
                    _mm256_storeu_ps(result.as_mut_ptr().add(base_idx + chunk_start + 16), high_result);
                }
            }
            
            result
        }
        
        /// Optimized Q4_1 dequantization using AVX2
        #[target_feature(enable = "avx2")]
        pub unsafe fn dequantize_q4_1_avx2(blocks: &[BlockQ4_1]) -> Vec<f32> {
            let mut result = Vec::with_capacity(blocks.len() * QK4_1);
            result.resize(blocks.len() * QK4_1, 0.0);
            
            let mask_low = _mm_set1_epi8(0x0F);
            
            for (i, block) in blocks.iter().enumerate() {
                let d = block.d.to_f32();
                let m = block.m.to_f32();
                let d_vec = _mm256_set1_ps(d);
                let m_vec = _mm256_set1_ps(m);
                let base_idx = i * QK4_1;
                
                for chunk_start in (0..16).step_by(8) {
                    let packed = _mm_loadl_epi64(block.qs.as_ptr().add(chunk_start) as *const __m128i);
                    
                    // Low nibbles
                    let low_nibbles = _mm_and_si128(packed, mask_low);
                    let low_i32 = _mm256_cvtepi8_epi32(low_nibbles);
                    let low_f32 = _mm256_cvtepi32_ps(low_i32);
                    let low_result = _mm256_fmadd_ps(low_f32, d_vec, m_vec);
                    
                    // High nibbles
                    let high_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_low);
                    let high_i32 = _mm256_cvtepi8_epi32(high_nibbles);
                    let high_f32 = _mm256_cvtepi32_ps(high_i32);
                    let high_result = _mm256_fmadd_ps(high_f32, d_vec, m_vec);
                    
                    _mm256_storeu_ps(result.as_mut_ptr().add(base_idx + chunk_start), low_result);
                    _mm256_storeu_ps(result.as_mut_ptr().add(base_idx + chunk_start + 16), high_result);
                }
            }
            
            result
        }
        
        /// Optimized quantized matrix-vector multiplication for Q4_0 weights
        #[target_feature(enable = "avx2")]
        pub unsafe fn qmatvec_q4_0_avx2(
            weights: &[BlockQ4_0], // Quantized weight matrix (M x K/32 blocks)
            input: &[f32],         // Input vector (K elements)
            output: &mut [f32],    // Output vector (M elements)
            m: usize,              // Number of rows
            k: usize,              // Number of columns (must be multiple of 32)
        ) {
            assert_eq!(k % QK4_0, 0);
            let num_blocks_per_row = k / QK4_0;
            
            for row in 0..m {
                let mut sum = _mm256_setzero_ps();
                let row_offset = row * num_blocks_per_row;
                
                for block_idx in 0..num_blocks_per_row {
                    let block = &weights[row_offset + block_idx];
                    let d = block.d.to_f32();
                    let d_vec = _mm256_set1_ps(d);
                    let input_offset = block_idx * QK4_0;
                    
                    // Process block in chunks of 8
                    for chunk in 0..4 { // 32/8 = 4 chunks
                        let chunk_offset = chunk * 8;
                        
                        // Load input values
                        let input_vec = _mm256_loadu_ps(input.as_ptr().add(input_offset + chunk_offset));
                        
                        // Unpack and dequantize weights
                        let packed_idx = chunk * 4; // 8 values packed in 4 bytes
                        let packed = _mm_loadl_epi64(block.qs.as_ptr().add(packed_idx) as *const __m128i);
                        
                        // Process low nibbles
                        let mask_low = _mm_set1_epi8(0x0F);
                        let low_nibbles = _mm_and_si128(packed, mask_low);
                        let low_i32 = _mm256_cvtepi8_epi32(low_nibbles);
                        let low_f32 = _mm256_cvtepi32_ps(low_i32);
                        let offset_vec = _mm256_set1_ps(-8.0);
                        let weight_vec = _mm256_fmadd_ps(low_f32, d_vec, _mm256_mul_ps(offset_vec, d_vec));
                        
                        // Multiply and accumulate
                        sum = _mm256_fmadd_ps(weight_vec, input_vec, sum);
                    }
                }
                
                // Horizontal sum and store
                let hi = _mm256_extractf128_ps(sum, 1);
                let lo = _mm256_castps256_ps128(sum);
                let sum_128 = _mm_add_ps(hi, lo);
                
                let shuf = _mm_movehdup_ps(sum_128);
                let sums = _mm_add_ps(sum_128, shuf);
                let shuf = _mm_movehl_ps(sums, sums);
                let final_sum = _mm_add_ss(sums, shuf);
                
                output[row] = _mm_cvtss_f32(final_sum);
            }
        }
        
        /// Optimized quantized matrix-vector multiplication for Q8_0 weights
        #[target_feature(enable = "avx2")]
        pub unsafe fn qmatvec_q8_0_avx2(
            weights: &[BlockQ8_0],
            input: &[f32],
            output: &mut [f32],
            m: usize,
            k: usize,
        ) {
            assert_eq!(k % QK8_0, 0);
            let num_blocks_per_row = k / QK8_0;
            
            for row in 0..m {
                let mut sum = _mm256_setzero_ps();
                let row_offset = row * num_blocks_per_row;
                
                for block_idx in 0..num_blocks_per_row {
                    let block = &weights[row_offset + block_idx];
                    let d = block.d.to_f32();
                    let d_vec = _mm256_set1_ps(d);
                    let input_offset = block_idx * QK8_0;
                    
                    // Process in chunks of 8
                    for chunk in 0..4 { // 32/8 = 4 chunks
                        let chunk_offset = chunk * 8;
                        
                        // Load input values
                        let input_vec = _mm256_loadu_ps(input.as_ptr().add(input_offset + chunk_offset));
                        
                        // Load and convert quantized weights
                        let q_i8 = _mm_loadl_epi64(block.qs.as_ptr().add(chunk_offset) as *const __m128i);
                        let q_i32 = _mm256_cvtepi8_epi32(q_i8);
                        let q_f32 = _mm256_cvtepi32_ps(q_i32);
                        let weight_vec = _mm256_mul_ps(q_f32, d_vec);
                        
                        // FMA
                        sum = _mm256_fmadd_ps(weight_vec, input_vec, sum);
                    }
                }
                
                // Horizontal sum
                let hi = _mm256_extractf128_ps(sum, 1);
                let lo = _mm256_castps256_ps128(sum);
                let sum_128 = _mm_add_ps(hi, lo);
                
                let shuf = _mm_movehdup_ps(sum_128);
                let sums = _mm_add_ps(sum_128, shuf);
                let shuf = _mm_movehl_ps(sums, sums);
                let final_sum = _mm_add_ss(sums, shuf);
                
                output[row] = _mm_cvtss_f32(final_sum);
            }
        }
    }
    
    /// Automatically chooses the best SIMD implementation for Q8_0
    pub fn dequantize_q8_0_simd(blocks: &[BlockQ8_0]) -> Vec<f32> {
        #[cfg(target_feature = "avx2")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { avx2::dequantize_q8_0_avx2(blocks) };
            }
        }
        super::dequantize_q8_0(blocks)
    }
    
    /// Automatically chooses the best SIMD implementation for Q4_0
    pub fn dequantize_q4_0_simd(blocks: &[BlockQ4_0]) -> Vec<f32> {
        #[cfg(target_feature = "avx2")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { avx2::dequantize_q4_0_avx2(blocks) };
            }
        }
        super::dequantize_q4_0(blocks)
    }
    
    /// Automatically chooses the best SIMD implementation for Q4_1
    pub fn dequantize_q4_1_simd(blocks: &[BlockQ4_1]) -> Vec<f32> {
        #[cfg(target_feature = "avx2")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { avx2::dequantize_q4_1_avx2(blocks) };
            }
        }
        super::dequantize_q4_1(blocks)
    }
    
    /// High-performance quantized matrix-vector multiplication
    pub fn quantized_matvec(
        weights: &QuantizedStorage,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        k: usize,
    ) -> Result<(), QuantizationError> {
        match weights {
            QuantizedStorage::Q4_0(blocks) => {
                #[cfg(target_feature = "avx2")]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            avx2::qmatvec_q4_0_avx2(blocks, input, output, m, k);
                        }
                        return Ok(());
                    }
                }
                
                // Fallback to scalar implementation
                quantized_matvec_q4_0_scalar(blocks, input, output, m, k)
            }
            QuantizedStorage::Q8_0(blocks) => {
                #[cfg(target_feature = "avx2")]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            avx2::qmatvec_q8_0_avx2(blocks, input, output, m, k);
                        }
                        return Ok(());
                    }
                }
                
                // Fallback to scalar implementation
                quantized_matvec_q8_0_scalar(blocks, input, output, m, k)
            }
            _ => Err(QuantizationError::UnsupportedScheme(
                "Quantized matvec not implemented for this scheme".to_string()
            )),
        }
    }
    
    /// Parallel Q8_0 dequantization for batch processing
    pub fn dequantize_q8_0_batch(blocks: &[BlockQ8_0]) -> Vec<f32> {
        // Use SIMD implementation if available and beneficial
        if blocks.len() > 1 {
            return dequantize_q8_0_simd(blocks);
        }
        
        // Process multiple blocks in parallel when beneficial
        #[cfg(feature = "parallel")]
        {
            if blocks.len() > 4 {
                use rayon::prelude::*;
                
                let mut result = Vec::with_capacity(blocks.len() * QK8_0);
                result.resize(blocks.len() * QK8_0, 0.0);
                
                result.par_chunks_mut(QK8_0)
                    .zip(blocks.par_iter())
                    .for_each(|(chunk, block)| {
                        let d = block.d.to_f32();
                        
                        // Parallel processing of the block
                        chunk.par_iter_mut()
                            .zip(block.qs.par_iter())
                            .for_each(|(out, &q)| {
                                *out = q as f32 * d;
                            });
                    });
                return result;
            }
        }
        
        // Use sequential version for small batches
        super::dequantize_q8_0(blocks)
    }

    /// Parallel Q4_1 dequantization for batch processing
    pub fn dequantize_q4_1_batch(blocks: &[BlockQ4_1]) -> Vec<f32> {
        // Use SIMD implementation if available and beneficial
        if blocks.len() > 1 {
            return dequantize_q4_1_simd(blocks);
        }
        
        // Process multiple blocks in parallel when beneficial
        #[cfg(feature = "parallel")]
        {
            if blocks.len() > 4 {
                use rayon::prelude::*;
                
                let mut result = Vec::with_capacity(blocks.len() * QK4_1);
                result.resize(blocks.len() * QK4_1, 0.0);
                
                result.par_chunks_mut(QK4_1)
                    .zip(blocks.par_iter())
                    .for_each(|(chunk, block)| {
                        let d = block.d.to_f32();
                        let m = block.m.to_f32();
                        
                        // Vectorize the inner loop
                        for j in 0..(QK4_1 / 2) {
                            let x0 = (block.qs[j] & 0x0F) as f32;
                            let x1 = (block.qs[j] >> 4) as f32;
                            
                            chunk[j] = x0 * d + m;
                            chunk[j + QK4_1 / 2] = x1 * d + m;
                        }
                    });
                return result;
            }
        }
        
        // Use sequential version for small batches
        super::dequantize_q4_1(blocks)
    }

    /// SIMD-optimized Q4_0 dequantization for batch processing
    pub fn dequantize_q4_0_batch(blocks: &[BlockQ4_0]) -> Vec<f32> {
        // Use SIMD implementation if available and beneficial
        if blocks.len() > 1 {
            return dequantize_q4_0_simd(blocks);
        }
        
        // Process multiple blocks in parallel when beneficial
        #[cfg(feature = "parallel")]
        {
            if blocks.len() > 4 {
                use rayon::prelude::*;
                
                let mut result = Vec::with_capacity(blocks.len() * QK4_0);
                result.resize(blocks.len() * QK4_0, 0.0);
                
                result.par_chunks_mut(QK4_0)
                    .zip(blocks.par_iter())
                    .for_each(|(chunk, block)| {
                        let d = block.d.to_f32();
                        
                        // Vectorize the inner loop
                        for j in 0..(QK4_0 / 2) {
                            let x0 = ((block.qs[j] & 0x0F) as i8) - 8;
                            let x1 = ((block.qs[j] >> 4) as i8) - 8;
                            
                            chunk[j] = x0 as f32 * d;
                            chunk[j + QK4_0 / 2] = x1 as f32 * d;
                        }
                    });
                return result;
            }
        }
        
        // Use sequential version for small batches or when parallel is not available
        super::dequantize_q4_0(blocks)
    }

    /// Parallel Q5_0 dequantization for batch processing
    pub fn dequantize_q5_0_batch(blocks: &[BlockQ5_0]) -> Vec<f32> {
        // Process multiple blocks in parallel when beneficial
        #[cfg(feature = "parallel")]
        {
            if blocks.len() > 4 {
                use rayon::prelude::*;
                
                let mut result = Vec::with_capacity(blocks.len() * QK5_0);
                result.resize(blocks.len() * QK5_0, 0.0);
                
                result.par_chunks_mut(QK5_0)
                    .zip(blocks.par_iter())
                    .for_each(|(chunk, block)| {
                        let d = block.d.to_f32();
                        let qh = u32::from_le_bytes(block.qh);
                        
                        // Vectorize the inner loop
                        for j in 0..(QK5_0 / 2) {
                            let xh_0 = ((qh >> j) << 4) & 0x10;
                            let xh_1 = (qh >> (j + 12)) & 0x10;
                            
                            let x0 = (((block.qs[j] & 0x0F) as u32) | xh_0) as i32 - 16;
                            let x1 = (((block.qs[j] >> 4) as u32) | xh_1) as i32 - 16;
                            
                            chunk[j] = x0 as f32 * d;
                            chunk[j + QK5_0 / 2] = x1 as f32 * d;
                        }
                    });
                return result;
            }
        }
        
        // Use sequential version for small batches
        super::dequantize_q5_0(blocks)
    }

    /// Parallel Q5_1 dequantization for batch processing
    pub fn dequantize_q5_1_batch(blocks: &[BlockQ5_1]) -> Vec<f32> {
        // Process multiple blocks in parallel when beneficial
        #[cfg(feature = "parallel")]
        {
            if blocks.len() > 4 {
                use rayon::prelude::*;
                
                let mut result = Vec::with_capacity(blocks.len() * QK5_1);
                result.resize(blocks.len() * QK5_1, 0.0);
                
                result.par_chunks_mut(QK5_1)
                    .zip(blocks.par_iter())
                    .for_each(|(chunk, block)| {
                        let d = block.d.to_f32();
                        let m = block.m.to_f32();
                        let qh = u32::from_le_bytes(block.qh);
                        
                        // Vectorize the inner loop
                        for j in 0..(QK5_1 / 2) {
                            let xh_0 = ((qh >> j) << 4) & 0x10;
                            let xh_1 = (qh >> (j + 12)) & 0x10;
                            
                            let x0 = ((block.qs[j] & 0x0F) as u32) | xh_0;
                            let x1 = ((block.qs[j] >> 4) as u32) | xh_1;
                            
                            chunk[j] = x0 as f32 * d + m;
                            chunk[j + QK5_1 / 2] = x1 as f32 * d + m;
                        }
                    });
                return result;
            }
        }
        
        // Use sequential version for small batches
        super::dequantize_q5_1(blocks)
    }
    
    /// Scalar fallback for quantized matrix-vector multiplication with Q4_0
    fn quantized_matvec_q4_0_scalar(
        weights: &[BlockQ4_0],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        k: usize,
    ) -> Result<(), QuantizationError> {
        assert_eq!(k % QK4_0, 0);
        let num_blocks_per_row = k / QK4_0;
        
        for row in 0..m {
            let mut sum = 0.0f32;
            let row_offset = row * num_blocks_per_row;
            
            for block_idx in 0..num_blocks_per_row {
                let block = &weights[row_offset + block_idx];
                let d = block.d.to_f32();
                let input_offset = block_idx * QK4_0;
                
                for j in 0..(QK4_0 / 2) {
                    let x0 = ((block.qs[j] & 0x0F) as i8) - 8;
                    let x1 = ((block.qs[j] >> 4) as i8) - 8;
                    
                    sum += (x0 as f32 * d) * input[input_offset + j];
                    sum += (x1 as f32 * d) * input[input_offset + j + QK4_0 / 2];
                }
            }
            
            output[row] = sum;
        }
        
        Ok(())
    }
    
    /// Scalar fallback for quantized matrix-vector multiplication with Q8_0
    fn quantized_matvec_q8_0_scalar(
        weights: &[BlockQ8_0],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        k: usize,
    ) -> Result<(), QuantizationError> {
        assert_eq!(k % QK8_0, 0);
        let num_blocks_per_row = k / QK8_0;
        
        for row in 0..m {
            let mut sum = 0.0f32;
            let row_offset = row * num_blocks_per_row;
            
            for block_idx in 0..num_blocks_per_row {
                let block = &weights[row_offset + block_idx];
                let d = block.d.to_f32();
                let input_offset = block_idx * QK8_0;
                
                for j in 0..QK8_0 {
                    sum += (block.qs[j] as f32 * d) * input[input_offset + j];
                }
            }
            
            output[row] = sum;
        }
        
        Ok(())
    }
}

/// Optimized dequantization dispatcher
pub mod optimized {
    use super::*;
    
    /// Smart dispatcher that chooses the best dequantization method
    pub fn dequantize_blocks(storage: &QuantizedStorage) -> Result<Vec<f32>, QuantizationError> {
        match storage {
            QuantizedStorage::Q4_0(blocks) => {
                if blocks.len() > 4 {
                    Ok(super::simd::dequantize_q4_0_batch(blocks))
                } else {
                    Ok(dequantize_q4_0(blocks))
                }
            }
            QuantizedStorage::Q4_1(blocks) => {
                if blocks.len() > 4 {
                    Ok(super::simd::dequantize_q4_1_batch(blocks))
                } else {
                    Ok(dequantize_q4_1(blocks))
                }
            }
            QuantizedStorage::Q5_0(blocks) => {
                if blocks.len() > 4 {
                    Ok(super::simd::dequantize_q5_0_batch(blocks))
                } else {
                    Ok(dequantize_q5_0(blocks))
                }
            }
            QuantizedStorage::Q5_1(blocks) => {
                if blocks.len() > 4 {
                    Ok(super::simd::dequantize_q5_1_batch(blocks))
                } else {
                    Ok(dequantize_q5_1(blocks))
                }
            }
            QuantizedStorage::Q8_0(blocks) => {
                if blocks.len() > 4 {
                    Ok(super::simd::dequantize_q8_0_batch(blocks))
                } else {
                    Ok(super::simd::dequantize_q8_0_simd(blocks))
                }
            }
            _ => Err(QuantizationError::UnsupportedScheme("Unsupported storage type".to_string())),
        }
    }
}

/// Quantization utilities
pub mod utils {
    use super::*;
    
    /// Computes the optimal quantization scheme for given data characteristics
    pub fn suggest_quantization_scheme(
        _values: &[f32],
        target_bits: Option<f32>,
        preserve_accuracy: bool,
    ) -> QuantizationScheme {
        let target_bits = target_bits.unwrap_or(4.0);
        
        if preserve_accuracy {
            // For high accuracy requirements
            if target_bits >= 8.0 {
                QuantizationScheme::Q8_0
            } else if target_bits >= 6.0 {
                QuantizationScheme::Q6K
            } else if target_bits >= 5.0 {
                QuantizationScheme::Q5K
            } else {
                QuantizationScheme::Q4K
            }
        } else {
            // For maximum compression
            if target_bits >= 8.0 {
                QuantizationScheme::Int8
            } else if target_bits >= 5.0 {
                QuantizationScheme::Q5_0
            } else if target_bits >= 4.0 {
                QuantizationScheme::Q4_0
            } else if target_bits >= 3.0 {
                QuantizationScheme::Q3K
            } else {
                QuantizationScheme::Q2K
            }
        }
    }
    
    /// Estimates the memory usage after quantization
    pub fn estimate_quantized_size(
        num_elements: usize,
        scheme: QuantizationScheme,
    ) -> usize {
        let bytes_per_elem = scheme.bytes_per_element();
        num_elements * bytes_per_elem
    }
    
    /// Computes theoretical compression ratio
    pub fn compression_ratio(
        original_dtype_bytes: usize,
        scheme: QuantizationScheme,
    ) -> f32 {
        original_dtype_bytes as f32 / scheme.bytes_per_element() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantization_schemes() {
        assert_eq!(QuantizationScheme::Int8.bits_per_element(), 8.0);
        assert_eq!(QuantizationScheme::Q4_0.bits_per_element(), 4.0);
        assert_eq!(QuantizationScheme::Q4_0.block_size(), 32);
    }
    
    #[test]
    fn test_compression_estimation() {
        let num_elements = 1024;
        let f32_size = num_elements * 4; // 4 bytes per f32
        
        let q4_size = utils::estimate_quantized_size(num_elements, QuantizationScheme::Q4_0);
        assert!(q4_size < f32_size);
        
        let ratio = utils::compression_ratio(4, QuantizationScheme::Q4_0);
        assert!(ratio > 1.0);
    }
    
    #[test]
    fn test_q4_0_dequantization() {
        // Test with a simple block where all values are the same
        let block = BlockQ4_0 {
            d: half::f16::from_f32(2.0),
            qs: [0x00; 16], // All values are 0, which becomes -8 after offset, then * scale
        };
        
        let result = dequantize_q4_0(&[block]);
        assert_eq!(result.len(), 32);
        // Each value should be -8 * 2.0 = -16.0
        for &val in &result {
            assert!((val - (-16.0)).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_q4_1_dequantization() {
        // Test with a simple block
        let block = BlockQ4_1 {
            d: half::f16::from_f32(1.0),
            m: half::f16::from_f32(5.0),
            qs: [0x01; 16], // Values are 1 and 0
        };
        
        let result = dequantize_q4_1(&[block]);
        assert_eq!(result.len(), 32);
        // First half should be 1 * 1.0 + 5.0 = 6.0
        // Second half should be 0 * 1.0 + 5.0 = 5.0
        for i in 0..16 {
            assert!((result[i] - 6.0).abs() < 1e-6);
            assert!((result[i + 16] - 5.0).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_q8_0_dequantization() {
        // Test with a simple block
        let mut qs = [0i8; 32];
        qs[0] = 10;
        qs[1] = -5;
        
        let block = BlockQ8_0 {
            d: half::f16::from_f32(0.5),
            qs,
        };
        
        let result = dequantize_q8_0(&[block]);
        assert_eq!(result.len(), 32);
        assert!((result[0] - 5.0).abs() < 1e-6);  // 10 * 0.5
        assert!((result[1] - (-2.5)).abs() < 1e-6); // -5 * 0.5
    }
    
    #[test]
    fn test_block_sizes() {
        use std::mem;
        
        // Verify block sizes match llama.cpp expectations
        assert_eq!(mem::size_of::<BlockQ4_0>(), 18); // 2 bytes (f16) + 16 bytes (qs)
        assert_eq!(mem::size_of::<BlockQ4_1>(), 20); // 4 bytes (2 x f16) + 16 bytes (qs)
        assert_eq!(mem::size_of::<BlockQ5_0>(), 22); // 2 bytes (f16) + 4 bytes (qh) + 16 bytes (qs)
        assert_eq!(mem::size_of::<BlockQ5_1>(), 24); // 4 bytes (2 x f16) + 4 bytes (qh) + 16 bytes (qs)
        assert_eq!(mem::size_of::<BlockQ8_0>(), 34); // 2 bytes (f16) + 32 bytes (qs)
    }
    
    #[test]
    fn test_optimized_dispatcher() {
        // Test the optimized dispatcher with Q4_0
        let blocks = vec![
            BlockQ4_0 {
                d: half::f16::from_f32(1.0),
                qs: [0x11; 16], // Each nibble is 1, so values are 1 and 1, offset by -8 = -7 and -7
            },
            BlockQ4_0 {
                d: half::f16::from_f32(2.0),
                qs: [0x22; 16], // Each nibble is 2, so values are 2 and 2, offset by -8 = -6 and -6
            },
        ];
        
        let storage = QuantizedStorage::Q4_0(blocks);
        let result = optimized::dequantize_blocks(&storage).unwrap();
        
        assert_eq!(result.len(), 64); // 2 blocks * 32 values each
        
        // First block: all values should be -7 * 1.0 = -7.0
        for i in 0..32 {
            assert!((result[i] - (-7.0)).abs() < 1e-6);
        }
        
        // Second block: all values should be -6 * 2.0 = -12.0
        for i in 32..64 {
            assert!((result[i] - (-12.0)).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_simd_batch_processing() {
        // Test the SIMD batch processing with multiple Q4_0 blocks
        let blocks: Vec<BlockQ4_0> = (0..8).map(|i| {
            BlockQ4_0 {
                d: half::f16::from_f32(i as f32 + 1.0),
                qs: [0x00; 16], // All zeros, becomes -8 after offset
            }
        }).collect();
        
        let result_simd = simd::dequantize_q4_0_batch(&blocks);
        let result_regular = dequantize_q4_0(&blocks);
        
        assert_eq!(result_simd.len(), result_regular.len());
        
        // Results should be identical
        for (simd_val, regular_val) in result_simd.iter().zip(result_regular.iter()) {
            assert!((simd_val - regular_val).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_q5_0_dequantization() {
        // Test Q5_0 with high bits
        let block = BlockQ5_0 {
            d: half::f16::from_f32(0.5),
            qh: [0xFF, 0xFF, 0xFF, 0xFF], // All high bits set
            qs: [0xFF; 16], // All low bits set (0xF in each nibble)
        };
        
        let result = dequantize_q5_0(&[block]);
        assert_eq!(result.len(), 32);
        
        // With high bit set (0x10) and low bits 0xF, we get 0x1F = 31, then subtract 16 = 15
        // So each value should be 15 * 0.5 = 7.5
        for &val in &result {
            assert!((val - 7.5).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_q5_1_dequantization() {
        // Test Q5_1 with offset
        let block = BlockQ5_1 {
            d: half::f16::from_f32(1.0),
            m: half::f16::from_f32(10.0),
            qh: [0x00, 0x00, 0x00, 0x00], // No high bits
            qs: [0x11; 16], // Each nibble is 1
        };
        
        let result = dequantize_q5_1(&[block]);
        assert_eq!(result.len(), 32);
        
        // Value is 1 (no high bit) * 1.0 + 10.0 = 11.0
        for &val in &result {
            assert!((val - 11.0).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_quantizer_traits() {
        let q4_0 = Q4_0Quantizer::new();
        assert_eq!(q4_0.scheme(), QuantizationScheme::Q4_0);
        assert!((q4_0.compression_ratio() - 8.0).abs() < 1e-6); // 32 / 4 = 8
        
        let q8_0 = Q8_0Quantizer::new();
        assert_eq!(q8_0.scheme(), QuantizationScheme::Q8_0);
        assert!((q8_0.compression_ratio() - 4.0).abs() < 1e-6); // 32 / 8 = 4
        
        let int8 = Int8Quantizer::new();
        assert_eq!(int8.scheme(), QuantizationScheme::Int8);
    }
}