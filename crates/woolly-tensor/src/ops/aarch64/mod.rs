//! AArch64 NEON SIMD implementations
//! 
//! Optimized dequantization kernels for ARM M4 and other AArch64 processors.
//! These implementations target Q4_K_M quantization specifically for maximum performance.

use crate::ops::SimdOps;
use half::f16;

/// NEON implementation for f32
#[derive(Debug, Clone, Copy)]
pub struct NeonF32;

/// Optimized quantization block structures for NEON processing
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct AlignedBlockQ4_K {
    /// 6-bit packed scales (12 bytes)
    pub scales: [u8; 12],
    /// Half precision scale factor
    pub d: f16,
    /// Half precision min scale factor
    pub dmin: f16,
    /// Quantized values - 4 bits per value, packed (128 bytes)
    pub qs: [u8; 128],
}

#[cfg(target_arch = "aarch64")]
impl SimdOps for NeonF32 {
    type Scalar = f32;
    type Vector = std::arch::aarch64::float32x4_t;
    
    const LANES: usize = 4;
    const ALIGN: usize = 16;
    
    #[inline]
    unsafe fn load(ptr: *const Self::Scalar) -> Self::Vector {
        std::arch::aarch64::vld1q_f32(ptr)
    }
    
    #[inline]
    unsafe fn load_aligned(ptr: *const Self::Scalar) -> Self::Vector {
        // NEON doesn't distinguish between aligned and unaligned loads
        std::arch::aarch64::vld1q_f32(ptr)
    }
    
    #[inline]
    unsafe fn store(ptr: *mut Self::Scalar, vec: Self::Vector) {
        std::arch::aarch64::vst1q_f32(ptr, vec)
    }
    
    #[inline]
    unsafe fn store_aligned(ptr: *mut Self::Scalar, vec: Self::Vector) {
        std::arch::aarch64::vst1q_f32(ptr, vec)
    }
    
    #[inline]
    fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vaddq_f32(a, b) }
    }
    
    #[inline]
    fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vsubq_f32(a, b) }
    }
    
    #[inline]
    fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vmulq_f32(a, b) }
    }
    
    #[inline]
    fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vdivq_f32(a, b) }
    }
    
    #[inline]
    fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vfmaq_f32(c, a, b) }
    }
    
    #[inline]
    fn hadd(vec: Self::Vector) -> Self::Scalar {
        unsafe {
            let sum = std::arch::aarch64::vaddvq_f32(vec);
            sum
        }
    }
    
    #[inline]
    fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vmaxq_f32(a, b) }
    }
    
    #[inline]
    fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vminq_f32(a, b) }
    }
    
    #[inline]
    fn sqrt(vec: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vsqrtq_f32(vec) }
    }
    
    #[inline]
    fn reciprocal(vec: Self::Vector) -> Self::Vector {
        unsafe { std::arch::aarch64::vrecpeq_f32(vec) }
    }
    
    #[inline]
    fn splat(value: Self::Scalar) -> Self::Vector {
        unsafe { std::arch::aarch64::vdupq_n_f32(value) }
    }
    
    #[inline]
    fn zero() -> Self::Vector {
        Self::splat(0.0)
    }
}

/// NEON implementation for f16
#[derive(Debug, Clone, Copy)]
pub struct NeonF16;

// F16 implementation would go here when half crate support is added

/// NEON implementation for i8
#[derive(Debug, Clone, Copy)]
pub struct NeonI8;

// I8 implementation would go here

/// NEON-optimized Q4_K_M dequantization specifically for ARM M4
#[cfg(target_arch = "aarch64")]
pub mod quantization {
    use super::*;
    use std::arch::aarch64::*;
    
    /// High-performance Q4_K_M dequantization using NEON intrinsics
    /// 
    /// This function is specifically optimized for the 90s/token bottleneck
    /// by using vectorized operations for all unpacking and scaling operations.
    /// 
    /// Performance target: 10-20x speedup over scalar implementation
    #[target_feature(enable = "neon")]
    pub unsafe fn dequantize_q4_k_neon(
        blocks: &[u8],          // Raw Q4_K block data
        output: &mut [f32],     // Output buffer (pre-allocated)
        num_blocks: usize,      // Number of Q4_K blocks
        block_size: usize,      // Size of each block in bytes (144 for Q4_K)
    ) {
        const QK_K: usize = 256;  // Elements per Q4_K block
        const BLOCK_SIZE: usize = 144;  // Bytes per Q4_K block
        
        // Precompute constants for bit manipulation
        let mask_0f = vdupq_n_u8(0x0F);
        let offset_8 = vdupq_n_s8(-8);
        
        for block_idx in 0..num_blocks {
            let block_offset = block_idx * BLOCK_SIZE;
            let block_data = &blocks[block_offset..block_offset + BLOCK_SIZE];
            let output_offset = block_idx * QK_K;
            
            // Extract scales - stored as 6-bit values packed in 12 bytes
            let mut scales = [0u8; 16];
            unpack_6bit_scales_neon(block_data, &mut scales);
            
            // Load d and dmin (half precision scale factors)
            let d = f16::from_le_bytes([block_data[12], block_data[13]]).to_f32();
            let dmin = f16::from_le_bytes([block_data[14], block_data[15]]).to_f32();
            
            let d_vec = vdupq_n_f32(d);
            let dmin_vec = vdupq_n_f32(dmin);
            
            // Process quantized values in chunks of 16 bytes (32 4-bit values)
            let qs = &block_data[16..144]; // 128 bytes of quantized data
            
            for chunk_idx in 0..8 { // 128 bytes / 16 bytes per chunk = 8 chunks
                let chunk_offset = chunk_idx * 16;
                let output_chunk_offset = output_offset + chunk_idx * 32;
                
                // Load 16 bytes of packed 4-bit values
                let packed_low = vld1q_u8(qs.as_ptr().add(chunk_offset));
                
                // Unpack low nibbles (first 16 values)
                let low_nibbles = vandq_u8(packed_low, mask_0f);
                
                // Unpack high nibbles (next 16 values)  
                let high_nibbles = vandq_u8(vshrq_n_u8(packed_low, 4), mask_0f);
                
                // Process low nibbles in batches of 4
                for i in 0..4 {
                    let scale_idx = (chunk_idx * 4 + i) / 4;
                    let scale = scales[scale_idx] as f32;
                    let scale_vec = vdupq_n_f32(scale);
                    
                    // Extract 4 values from low nibbles
                    let vals_u8 = vget_low_u8(vextq_u8(low_nibbles, low_nibbles, i * 4));
                    let vals_u32 = vmovl_u16(vget_low_u16(vmovl_u8(vals_u8)));
                    let vals_f32 = vcvtq_f32_u32(vals_u32);
                    
                    // Apply quantization formula: d * scale * val + dmin
                    let scaled = vmulq_f32(vmulq_f32(d_vec, scale_vec), vals_f32);
                    let result = vaddq_f32(scaled, dmin_vec);
                    
                    // Store results
                    vst1q_f32(output.as_mut_ptr().add(output_chunk_offset + i * 4), result);
                }
                
                // Process high nibbles in batches of 4
                for i in 0..4 {
                    let scale_idx = (chunk_idx * 4 + i + 16) / 4;
                    let scale = scales[scale_idx] as f32;
                    let scale_vec = vdupq_n_f32(scale);
                    
                    // Extract 4 values from high nibbles
                    let vals_u8 = vget_low_u8(vextq_u8(high_nibbles, high_nibbles, i * 4));
                    let vals_u32 = vmovl_u16(vget_low_u16(vmovl_u8(vals_u8)));
                    let vals_f32 = vcvtq_f32_u32(vals_u32);
                    
                    // Apply quantization formula
                    let scaled = vmulq_f32(vmulq_f32(d_vec, scale_vec), vals_f32);
                    let result = vaddq_f32(scaled, dmin_vec);
                    
                    // Store results
                    vst1q_f32(output.as_mut_ptr().add(output_chunk_offset + 16 + i * 4), result);
                }
            }
        }
    }
    
    /// Highly optimized bulk dequantization for entire layers
    /// 
    /// This function processes multiple tensors in a single call to reduce
    /// function call overhead and improve cache locality.
    #[target_feature(enable = "neon")]
    pub unsafe fn bulk_dequantize_layer_neon(
        tensors: &[(&[u8], usize)],  // (tensor_data, num_blocks) pairs
        outputs: &mut [&mut [f32]],  // Output buffers for each tensor
        prefetch_next: Option<*const u8>, // Optional prefetch pointer
    ) {
        // Process all tensors in the layer
        for (i, ((tensor_data, num_blocks), output)) in tensors.iter().zip(outputs.iter_mut()).enumerate() {
            // Prefetch next tensor data for better cache performance
            if let Some(next_ptr) = prefetch_next {
                if i + 1 < tensors.len() {
                    let next_tensor_data = tensors[i + 1].0.as_ptr();
                    // ARM's PLD (preload data) instruction via inline assembly
                    std::arch::asm!("prfm pldl1keep, [{}]", in(reg) next_tensor_data);
                }
            }
            
            dequantize_q4_k_neon(tensor_data, output, *num_blocks, 144);
        }
    }
    
    /// Smart caching system for frequently accessed weights
    /// 
    /// This maintains a cache of dequantized weights with LRU eviction
    /// and predictive prefetching based on layer access patterns.
    pub struct NeonQuantizationCache {
        cache: std::collections::HashMap<u64, Vec<f32>>,
        access_order: std::collections::VecDeque<u64>,
        max_entries: usize,
        total_memory: usize,
        max_memory: usize,
        hit_count: u64,
        miss_count: u64,
    }
    
    impl NeonQuantizationCache {
        pub fn new(max_memory_mb: usize) -> Self {
            Self {
                cache: std::collections::HashMap::new(),
                access_order: std::collections::VecDeque::new(),
                max_entries: 1024,
                total_memory: 0,
                max_memory: max_memory_mb * 1024 * 1024,
                hit_count: 0,
                miss_count: 0,
            }
        }
        
        /// Get dequantized weights from cache or compute them
        #[target_feature(enable = "neon")]
        pub unsafe fn get_or_dequantize(
            &mut self,
            key: u64,
            tensor_data: &[u8],
            num_blocks: usize,
        ) -> &[f32] {
            if let Some(cached) = self.cache.get(&key) {
                self.hit_count += 1;
                
                // Move to end of access order (most recently used)
                if let Some(pos) = self.access_order.iter().position(|&x| x == key) {
                    self.access_order.remove(pos);
                }
                self.access_order.push_back(key);
                
                return cached;
            }
            
            self.miss_count += 1;
            
            // Dequantize the tensor
            let output_size = num_blocks * 256; // 256 elements per Q4_K block
            let mut output = vec![0.0f32; output_size];
            dequantize_q4_k_neon(tensor_data, &mut output, num_blocks, 144);
            
            // Add to cache if there's space
            let tensor_memory = output.len() * std::mem::size_of::<f32>();
            if self.total_memory + tensor_memory <= self.max_memory {
                self.evict_if_needed(tensor_memory);
                self.total_memory += tensor_memory;
                self.cache.insert(key, output);
                self.access_order.push_back(key);
                
                self.cache.get(&key).unwrap()
            } else {
                // Return without caching if tensor is too large
                Box::leak(output.into_boxed_slice())
            }
        }
        
        fn evict_if_needed(&mut self, needed_memory: usize) {
            while self.total_memory + needed_memory > self.max_memory && !self.access_order.is_empty() {
                if let Some(oldest_key) = self.access_order.pop_front() {
                    if let Some(evicted) = self.cache.remove(&oldest_key) {
                        self.total_memory -= evicted.len() * std::mem::size_of::<f32>();
                    }
                }
            }
        }
        
        pub fn hit_rate(&self) -> f64 {
            let total = self.hit_count + self.miss_count;
            if total == 0 { 0.0 } else { self.hit_count as f64 / total as f64 }
        }
        
        pub fn memory_usage_mb(&self) -> f64 {
            self.total_memory as f64 / (1024.0 * 1024.0)
        }
        
        /// Prefetch weights for upcoming layers based on access patterns
        pub fn prefetch_layer_weights(&mut self, layer_indices: &[usize], weight_keys: &[u64]) {
            // Simple prefetching strategy: mark frequently accessed weights as high priority
            // In a real implementation, this would use more sophisticated prediction
            for &key in weight_keys {
                if self.cache.contains_key(&key) {
                    // Move frequently accessed items to the back of eviction queue
                    if let Some(pos) = self.access_order.iter().position(|&x| x == key) {
                        self.access_order.remove(pos);
                        self.access_order.push_back(key);
                    }
                }
            }
        }
    }
    
    /// Unpack 6-bit scales from Q4_K block data using NEON
    #[target_feature(enable = "neon")]
    unsafe fn unpack_6bit_scales_neon(block_data: &[u8], scales: &mut [u8; 16]) {
        // Q4_K stores scales as 6-bit values packed in 12 bytes
        // This is a complex bit manipulation that benefits from SIMD
        
        let scales_data = &block_data[0..12];
        
        // Load 12 bytes of packed 6-bit values
        let packed_low = vld1_u8(scales_data.as_ptr());
        let packed_high = vld1_u8(scales_data.as_ptr().add(8));
        
        // Unpack 6-bit values using bit manipulation
        // This is a simplified version - real implementation would be more complex
        for i in 0..8 {
            let src_idx = (i * 3) / 2;
            if i % 2 == 0 {
                scales[i] = scales_data[src_idx] & 63;
            } else {
                scales[i] = ((scales_data[src_idx] >> 6) | ((scales_data[src_idx + 1] & 15) << 2)) & 63;
            }
        }
        
        // Fill remaining scales (Q4_K has 16 scale values)
        for i in 8..16 {
            scales[i] = scales[i % 8];
        }
    }
    
    /// Optimized Q4 bit unpacking with vectorized operations
    #[target_feature(enable = "neon")]
    pub unsafe fn unpack_q4_bits_neon(
        packed: &[u8],        // Packed 4-bit values
        unpacked: &mut [u8],  // Output buffer for unpacked values
        count: usize,         // Number of 4-bit values to unpack
    ) {
        let mask_0f = vdupq_n_u8(0x0F);
        let chunks = count / 32; // Process 32 values (16 bytes) at a time
        
        for chunk_idx in 0..chunks {
            let input_offset = chunk_idx * 16;
            let output_offset = chunk_idx * 32;
            
            // Load 16 bytes of packed data
            let packed_data = vld1q_u8(packed.as_ptr().add(input_offset));
            
            // Extract low nibbles
            let low_nibbles = vandq_u8(packed_data, mask_0f);
            
            // Extract high nibbles
            let high_nibbles = vandq_u8(vshrq_n_u8(packed_data, 4), mask_0f);
            
            // Interleave and store
            let result_low = vzip1q_u8(low_nibbles, high_nibbles);
            let result_high = vzip2q_u8(low_nibbles, high_nibbles);
            
            vst1q_u8(unpacked.as_mut_ptr().add(output_offset), result_low);
            vst1q_u8(unpacked.as_mut_ptr().add(output_offset + 16), result_high);
        }
        
        // Handle remaining values
        let remaining = count % 32;
        if remaining > 0 {
            let start_idx = chunks * 32;
            for i in 0..remaining {
                let packed_idx = (start_idx + i) / 2;
                if (start_idx + i) % 2 == 0 {
                    unpacked[start_idx + i] = packed[packed_idx] & 0x0F;
                } else {
                    unpacked[start_idx + i] = (packed[packed_idx] >> 4) & 0x0F;
                }
            }
        }
    }
}

/// Public interface for optimized quantization operations
pub use quantization::*;