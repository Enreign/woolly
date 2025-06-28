//! Optimized Dequantization Kernels
//! 
//! High-performance SIMD-optimized dequantization implementations
//! similar to llama.cpp's approach for maximum performance.

use crate::cpu_features::{CpuFeatures, SimdDispatcher, SimdLevel};
use crate::CoreError;
use std::sync::OnceLock;

/// Optimized dequantization dispatcher
pub struct OptimizedDequantizer {
    dispatcher: SimdDispatcher,
}

impl OptimizedDequantizer {
    /// Create a new optimized dequantizer
    pub fn new() -> Self {
        Self {
            dispatcher: SimdDispatcher::new(),
        }
    }
    
    /// Get the global cached dequantizer instance
    pub fn get() -> &'static OptimizedDequantizer {
        static DEQUANTIZER: OnceLock<OptimizedDequantizer> = OnceLock::new();
        DEQUANTIZER.get_or_init(OptimizedDequantizer::new)
    }
    
    /// Dequantize Q4_K format with SIMD optimization
    pub fn dequantize_q4_k(&self, input: &[u8], output: &mut [f32]) -> Result<(), CoreError> {
        if input.is_empty() || output.is_empty() {
            return Ok(());
        }
        
        // Q4_K format: 256 elements per block, 144 bytes per block
        // Block structure: 12 bytes (scales) + 2 bytes (d) + 2 bytes (dmin) + 128 bytes (quantized values)
        const BLOCK_SIZE: usize = 256; // QK_K elements per block
        const BYTES_PER_BLOCK: usize = 144; // Total bytes per Q4_K block
        
        let num_blocks = (output.len() + BLOCK_SIZE - 1) / BLOCK_SIZE; // Round up division
        if input.len() < num_blocks * BYTES_PER_BLOCK {
            return Err(CoreError::tensor(
                "DEQUANT_INPUT_TOO_SMALL",
                format!("Input size {} too small for {} blocks (need {} bytes)", 
                       input.len(), num_blocks, num_blocks * BYTES_PER_BLOCK),
                "Q4_K dequantization",
                "Ensure input buffer matches expected Q4_K format"
            ));
        }
        
        let simd_level = self.dispatcher.simd_level();
        eprintln!("🔍 Q4_K dequantization - SIMD level: {:?}", simd_level);
        
        match simd_level {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => unsafe {
                eprintln!("🚀 Using AVX2 optimized path for Q4_K");
                self.dequantize_q4_k_avx2(input, output, num_blocks)?;
            },
            
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon | SimdLevel::NeonDotprod => unsafe {
                eprintln!("🚀 Using NEON optimized path for Q4_K");
                self.dequantize_q4_k_neon(input, output, num_blocks)?;
            },
            
            _ => {
                eprintln!("⚠️ Falling back to scalar Q4_K dequantization");
                self.dequantize_q4_k_scalar(input, output, num_blocks)?;
            }
        }
        
        Ok(())
    }
    
    /// Scalar implementation for Q4_K dequantization
    fn dequantize_q4_k_scalar(&self, input: &[u8], output: &mut [f32], num_blocks: usize) -> Result<(), CoreError> {
        const BLOCK_SIZE: usize = 256; // QK_K elements per block
        const BYTES_PER_BLOCK: usize = 144; // Total bytes per Q4_K block
        
        for block_idx in 0..num_blocks {
            let block_offset = block_idx * BYTES_PER_BLOCK;
            
            // Q4_K block structure:
            // - 12 bytes: scales (6-bit packed scale values)
            // - 2 bytes: d (f16 scale factor) 
            // - 2 bytes: dmin (f16 min scale factor)
            // - 128 bytes: quantized values (256 values × 4 bits = 1024 bits = 128 bytes)
            
            if block_offset + BYTES_PER_BLOCK > input.len() {
                break; // Incomplete block
            }
            
            // Extract d (main scale factor) as f16 at offset 12-13
            let d_bytes = &input[block_offset + 12..block_offset + 14];
            let d_bits = u16::from_le_bytes([d_bytes[0], d_bytes[1]]);
            let d = half::f16::from_bits(d_bits).to_f32();
            
            // Extract dmin (min scale factor) as f16 at offset 14-15  
            let dmin_bytes = &input[block_offset + 14..block_offset + 16];
            let dmin_bits = u16::from_le_bytes([dmin_bytes[0], dmin_bytes[1]]);
            let dmin = half::f16::from_bits(dmin_bits).to_f32();
            
            // Extract 6-bit scales from first 12 bytes
            let mut scales = [0u8; 16]; // Q4_K uses 16 scales per block
            for i in 0..12 {
                let byte = input[block_offset + i];
                // Pack 6-bit values into scales array (simplified extraction)
                scales[i] = byte >> 2; // Use upper 6 bits as scale
            }
            
            // Dequantize 4-bit values from offset 16 onwards (128 bytes of data)
            let data_start = block_offset + 16;
            for i in 0..128 {
                if data_start + i >= input.len() {
                    break;
                }
                
                let byte = input[data_start + i];
                let low_nibble = (byte & 0x0F) as i8;
                let high_nibble = ((byte >> 4) & 0x0F) as i8;
                
                // Calculate output indices
                let output_idx_low = block_idx * BLOCK_SIZE + i * 2;
                let output_idx_high = output_idx_low + 1;
                
                // Apply scale (simplified scaling for now)
                let scale_idx = (i / 8) % 16; // Use one of 16 scales per subgroup
                let scale = scales[scale_idx] as f32 * d + dmin;
                
                if output_idx_low < output.len() {
                    output[output_idx_low] = low_nibble as f32 * scale;
                }
                if output_idx_high < output.len() {
                    output[output_idx_high] = high_nibble as f32 * scale;
                }
            }
        }
        
        Ok(())
    }
    
    /// AVX2 implementation for Q4_K dequantization
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dequantize_q4_k_avx2(&self, input: &[u8], output: &mut [f32], num_blocks: usize) -> Result<(), CoreError> {
        eprintln!("🏁 Starting AVX2 Q4_K dequantization for {} blocks", num_blocks);
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            
            const BLOCK_SIZE: usize = 256; // QK_K elements per block
            const BYTES_PER_BLOCK: usize = 144; // Total bytes per Q4_K block
            
            // SIMD constants are created inline where needed
            
            for block_idx in 0..num_blocks {
                let block_offset = block_idx * BYTES_PER_BLOCK;
                
                if block_offset + BYTES_PER_BLOCK > input.len() {
                    break; // Incomplete block
                }
                
                // Extract d and dmin as f16
                let d_bytes = &input[block_offset + 12..block_offset + 14];
                let d = half::f16::from_bits(u16::from_le_bytes([d_bytes[0], d_bytes[1]])).to_f32();
                
                let dmin_bytes = &input[block_offset + 14..block_offset + 16];
                let dmin = half::f16::from_bits(u16::from_le_bytes([dmin_bytes[0], dmin_bytes[1]])).to_f32();
                
                // Extract and prepare scales (simplified for SIMD)
                let mut scales = [0f32; 16];
                for i in 0..12 {
                    scales[i] = (input[block_offset + i] >> 2) as f32 * d + dmin;
                }
                
                // Process quantized data in 32-byte chunks (64 4-bit values -> 64 floats)
                let data_start = block_offset + 16;
                let output_base = block_idx * BLOCK_SIZE;
                
                // Process 128 bytes of quantized data in chunks
                for chunk in 0..4 { // 4 chunks of 32 bytes each
                    let chunk_offset = chunk * 32;
                    let output_offset = output_base + chunk * 64;
                    
                    if data_start + chunk_offset + 32 > input.len() || output_offset + 64 > output.len() {
                        // Fall back to scalar for partial blocks
                        for i in 0..32 {
                            if data_start + chunk_offset + i < input.len() {
                                let byte = input[data_start + chunk_offset + i];
                                let idx_low = output_offset + i * 2;
                                let idx_high = idx_low + 1;
                                
                                if idx_low < output.len() {
                                    let scale_idx = ((chunk_offset + i) / 8) % 16;
                                    output[idx_low] = (byte & 0x0F) as f32 * scales[scale_idx];
                                }
                                if idx_high < output.len() {
                                    let scale_idx = ((chunk_offset + i) / 8) % 16;
                                    output[idx_high] = ((byte >> 4) & 0x0F) as f32 * scales[scale_idx];
                                }
                            }
                        }
                        continue;
                    }
                    
                    // Process 32 bytes at once - each byte contains two 4-bit values
                    // We'll process in 8-byte chunks for better pipeline utilization
                    
                    for i in 0..4 {
                        let byte_offset = i * 8;
                        let input_offset = data_start + chunk_offset + byte_offset;
                        let out_idx = output_offset + byte_offset * 2;
                        
                        // Load 8 bytes of packed 4-bit values
                        let packed = _mm_loadl_epi64((input.as_ptr().add(input_offset)) as *const __m128i);
                        
                        // Extract low nibbles (bottom 4 bits of each byte)
                        let low_nibbles = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
                        
                        // Extract high nibbles (top 4 bits of each byte)
                        let high_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
                        
                        // Get appropriate scale for this chunk
                        let scale_idx = ((chunk_offset + byte_offset) / 8) % 16;
                        let scale_vec = _mm256_set1_ps(scales[scale_idx]);
                        
                        // Convert low nibbles to 32-bit integers then to floats
                        let low_32 = _mm256_cvtepu8_epi32(low_nibbles);
                        let low_float = _mm256_cvtepi32_ps(low_32);
                        let low_scaled = _mm256_mul_ps(low_float, scale_vec);
                        
                        // Convert high nibbles to 32-bit integers then to floats
                        let high_32 = _mm256_cvtepu8_epi32(high_nibbles);
                        let high_float = _mm256_cvtepi32_ps(high_32);
                        let high_scaled = _mm256_mul_ps(high_float, scale_vec);
                        
                        // Store results - low nibbles first, then high nibbles
                        _mm256_storeu_ps(output.as_mut_ptr().add(out_idx), low_scaled);
                        _mm256_storeu_ps(output.as_mut_ptr().add(out_idx + 8), high_scaled);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// NEON implementation for Q4_K dequantization
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn dequantize_q4_k_neon(&self, input: &[u8], output: &mut [f32], num_blocks: usize) -> Result<(), CoreError> {
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            
            eprintln!("🏁 Starting NEON Q4_K dequantization for {} blocks", num_blocks);
            
            const BLOCK_SIZE: usize = 256;
            const BYTES_PER_BLOCK: usize = 144;
            
            for block_idx in 0..num_blocks {
                let block_offset = block_idx * BYTES_PER_BLOCK;
                
                if block_offset + BYTES_PER_BLOCK > input.len() {
                    break;
                }
                
                // Extract main scale factor (d) and minimum (dmin) as f16
                let d_bytes = &input[block_offset + 12..block_offset + 14];
                let d_bits = u16::from_le_bytes([d_bytes[0], d_bytes[1]]);
                let d = half::f16::from_bits(d_bits).to_f32();
                
                let dmin_bytes = &input[block_offset + 14..block_offset + 16];
                let dmin_bits = u16::from_le_bytes([dmin_bytes[0], dmin_bytes[1]]);
                let dmin = half::f16::from_bits(dmin_bits).to_f32();
                
                // Extract 6-bit scales (0-63)
                let scale_bytes = &input[block_offset..block_offset + 12];
                
                // Extract quantized values (4-bit packed)
                let ql_start = block_offset + 16;
                
                // Process 32 bytes at a time using NEON (64 elements)
                for sub_block in 0..4 {
                    let scale_idx = sub_block * 3;
                    
                    // Extract 6-bit scales for this sub-block
                    let scale1 = (scale_bytes[scale_idx] & 0x3F) as f32;
                    let scale2 = ((scale_bytes[scale_idx] >> 6) | ((scale_bytes[scale_idx + 1] & 0x0F) << 2)) as f32;
                    let scale3 = ((scale_bytes[scale_idx + 1] >> 4) | ((scale_bytes[scale_idx + 2] & 0x03) << 4)) as f32;
                    let scale4 = (scale_bytes[scale_idx + 2] >> 2) as f32;
                    
                    // Convert scales to actual values
                    let s1 = d * scale1 + dmin;
                    let s2 = d * scale2 + dmin;
                    let s3 = d * scale3 + dmin;
                    let s4 = d * scale4 + dmin;
                    
                    // Load scale values into NEON registers
                    let scales = [s1, s2, s3, s4];
                    let scale_vec = vld1q_f32(scales.as_ptr());
                    
                    // Process 16 bytes (32 4-bit values) at a time
                    for i in 0..2 {
                        let byte_offset = ql_start + sub_block * 32 + i * 16;
                        let output_offset = block_idx * BLOCK_SIZE + sub_block * 64 + i * 32;
                        
                        if byte_offset + 16 > input.len() || output_offset + 32 > output.len() {
                            break;
                        }
                        
                        // Load 16 bytes of quantized data
                        let q_data = vld1q_u8(input[byte_offset..].as_ptr());
                        
                        // Extract low and high nibbles
                        let mask_low = vdupq_n_u8(0x0F);
                        let q_low = vandq_u8(q_data, mask_low);
                        let q_high = vshrq_n_u8(q_data, 4);
                        
                        // Process each set of 16 values
                        for j in 0..2 {
                            let scale_idx = (i * 2 + j) % 4;
                            let curr_scale = vdupq_n_f32(scales[scale_idx]);
                            
                            // Get 8 values from low or high nibbles
                            let q_vals = if j == 0 {
                                vget_low_u8(q_low)
                            } else {
                                vget_low_u8(q_high)
                            };
                            
                            // Convert to u16 then u32 then f32
                            let q_u16 = vmovl_u8(q_vals);
                            let q_u32_low = vmovl_u16(vget_low_u16(q_u16));
                            let q_u32_high = vmovl_u16(vget_high_u16(q_u16));
                            
                            // Convert to float and subtract 8
                            let q_f32_low = vcvtq_f32_u32(q_u32_low);
                            let q_f32_high = vcvtq_f32_u32(q_u32_high);
                            
                            let eight = vdupq_n_f32(8.0);
                            let centered_low = vsubq_f32(q_f32_low, eight);
                            let centered_high = vsubq_f32(q_f32_high, eight);
                            
                            // Multiply by scale
                            let result_low = vmulq_f32(centered_low, curr_scale);
                            let result_high = vmulq_f32(centered_high, curr_scale);
                            
                            // Store results
                            let out_offset = output_offset + j * 16;
                            if out_offset + 8 <= output.len() {
                                vst1q_f32(output[out_offset..].as_mut_ptr(), result_low);
                                vst1q_f32(output[out_offset + 4..].as_mut_ptr(), result_high);
                            }
                        }
                    }
                }
            }
            
            eprintln!("✅ NEON Q4_K dequantization completed");
            return Ok(());
        }
        
        Ok(())
    }
    
    /// Dequantize Q6_K format with SIMD optimization
    pub fn dequantize_q6_k(&self, input: &[u8], output: &mut [f32]) -> Result<(), CoreError> {
        if input.is_empty() || output.is_empty() {
            return Ok(());
        }
        
        // Q6_K format: 256 elements per block, 210 bytes per block
        // Block structure: 128B (ql) + 64B (qh) + 16B (scales) + 2B (d)
        const BLOCK_SIZE: usize = 256; // QK_K elements per block
        const BYTES_PER_BLOCK: usize = 210; // Total bytes per Q6_K block
        
        let num_blocks = (output.len() + BLOCK_SIZE - 1) / BLOCK_SIZE; // Round up division
        if input.len() < num_blocks * BYTES_PER_BLOCK {
            return Err(CoreError::tensor(
                "DEQUANT_INPUT_TOO_SMALL",
                format!("Input size {} too small for {} Q6_K blocks (need {} bytes)", 
                       input.len(), num_blocks, num_blocks * BYTES_PER_BLOCK),
                "Q6_K dequantization",
                "Ensure input buffer matches expected Q6_K format"
            ));
        }
        
        let simd_level = self.dispatcher.simd_level();
        eprintln!("🔍 Q6_K dequantization - SIMD level: {:?}", simd_level);
        
        match simd_level {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => unsafe {
                eprintln!("🚀 Using AVX2 optimized path for Q6_K");
                self.dequantize_q6_k_avx2(input, output, num_blocks)?;
            },
            
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon | SimdLevel::NeonDotprod => unsafe {
                eprintln!("🚀 Using NEON optimized path for Q6_K");
                self.dequantize_q6_k_neon(input, output, num_blocks)?;
            },
            
            _ => {
                eprintln!("⚠️ Falling back to scalar Q6_K dequantization");
                self.dequantize_q6_k_scalar(input, output, num_blocks)?;
            }
        }
        
        Ok(())
    }
    
    /// Scalar implementation for Q6_K dequantization
    fn dequantize_q6_k_scalar(&self, input: &[u8], output: &mut [f32], num_blocks: usize) -> Result<(), CoreError> {
        const BLOCK_SIZE: usize = 256; // QK_K elements per block
        const BYTES_PER_BLOCK: usize = 210; // Total bytes per Q6_K block
        
        for block_idx in 0..num_blocks {
            let block_offset = block_idx * BYTES_PER_BLOCK;
            
            if block_offset + BYTES_PER_BLOCK > input.len() {
                break; // Incomplete block
            }
            
            // Q6_K block structure:
            // Bytes 0-127:   ql (quantized low bits) - 4-bit base values
            // Bytes 128-191: qh (quantized high bits) - 2-bit extensions  
            // Bytes 192-207: scales - 16 bytes (8-bit scales)
            // Bytes 208-209: d (main scale factor) - f16
            
            // Extract main scale factor (d) as f16 at offset 208-209
            let d_bytes = &input[block_offset + 208..block_offset + 210];
            let d_bits = u16::from_le_bytes([d_bytes[0], d_bytes[1]]);
            let d = half::f16::from_bits(d_bits).to_f32();
            
            // Extract scales (16 bytes starting at offset 192)
            let scales_start = block_offset + 192;
            
            // Process each element in the block
            for i in 0..BLOCK_SIZE {
                let output_idx = block_idx * BLOCK_SIZE + i;
                if output_idx >= output.len() {
                    break;
                }
                
                let is = i / 16; // Scale index (0-15)
                
                // Calculate complex indices for Q6_K bit packing
                let ql_idx = block_offset + 64 * (i / 128) + 32 * ((i % 128) / 64) + 16 * ((i % 64) / 32) + (i % 16);
                let qh_idx = block_offset + 128 + 32 * (i / 128) + 16 * ((i % 128) / 64) + 8 * ((i % 64) / 32) + (i % 32) / 4;
                let qh_shift = 2 * ((i % 32) % 4); // 0, 2, 4, or 6
                
                if ql_idx >= input.len() || qh_idx >= input.len() {
                    continue;
                }
                
                // Extract 2-bit extensions from qh
                let qh_val = input[qh_idx];
                let q1_ext = (qh_val >> qh_shift) & 3;
                let q2_ext = if qh_shift < 6 {
                    (qh_val >> (qh_shift + 2)) & 3
                } else {
                    // Handle wraparound for last position
                    if qh_idx + 1 < input.len() {
                        input[qh_idx + 1] & 3
                    } else {
                        0
                    }
                };
                
                // Extract 4-bit base values from ql
                let ql_val = input[ql_idx];
                let q1 = (ql_val & 0x0F) | (q1_ext << 4);
                let q2 = (ql_val >> 4) | (q2_ext << 4);
                
                // Select q1 or q2 based on position
                let q = if i < 128 { q1 } else { q2 };
                
                // Get scale for this group
                let scale = input[scales_start + is] as f32;
                
                // Final dequantization: d * scale * (q - 32) / 64
                // Convert to signed and apply scaling
                let dequantized = d * scale * ((q as i8 - 32) as f32) / 64.0;
                output[output_idx] = dequantized;
            }
        }
        
        Ok(())
    }
    
    /// AVX2 implementation for Q6_K dequantization
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dequantize_q6_k_avx2(&self, input: &[u8], output: &mut [f32], num_blocks: usize) -> Result<(), CoreError> {
        eprintln!("🏁 Starting AVX2 Q6_K dequantization for {} blocks", num_blocks);
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            
            const BLOCK_SIZE: usize = 256; // QK_K elements per block
            const BYTES_PER_BLOCK: usize = 210; // Total bytes per Q6_K block
            
            for block_idx in 0..num_blocks {
                let block_offset = block_idx * BYTES_PER_BLOCK;
                
                if block_offset + BYTES_PER_BLOCK > input.len() {
                    break; // Incomplete block
                }
                
                // Q6_K block structure:
                // Bytes 0-127:   ql (quantized low bits) - 4-bit base values
                // Bytes 128-191: qh (quantized high bits) - 2-bit extensions  
                // Bytes 192-207: scales - 16 bytes (8-bit scales)
                // Bytes 208-209: d (main scale factor) - f16
                
                // Extract main scale factor (d) as f16 at offset 208-209
                let d_bytes = &input[block_offset + 208..block_offset + 210];
                let d_bits = u16::from_le_bytes([d_bytes[0], d_bytes[1]]);
                let d = half::f16::from_bits(d_bits).to_f32();
                
                // Extract scales (16 bytes starting at offset 192)
                let scales_start = block_offset + 192;
                let mut scales = [0f32; 16];
                for i in 0..16 {
                    scales[i] = input[scales_start + i] as f32;
                }
                
                // Process in chunks for better SIMD utilization
                let output_base = block_idx * BLOCK_SIZE;
                
                // We'll process 16 elements at a time (using 2 AVX2 registers)
                for chunk in 0..16 {
                    let chunk_offset = chunk * 16;
                    if output_base + chunk_offset + 16 > output.len() {
                        // Fall back to scalar for partial blocks
                        for i in 0..16 {
                            let idx = chunk_offset + i;
                            if output_base + idx >= output.len() {
                                break;
                            }
                            
                            let is = idx / 16; // Scale index
                            
                            // Calculate indices for Q6_K bit packing
                            let ql_idx = block_offset + 64 * (idx / 128) + 32 * ((idx % 128) / 64) + 16 * ((idx % 64) / 32) + (idx % 16);
                            let qh_idx = block_offset + 128 + 32 * (idx / 128) + 16 * ((idx % 128) / 64) + 8 * ((idx % 64) / 32) + (idx % 32) / 4;
                            let qh_shift = 2 * ((idx % 32) % 4);
                            
                            if ql_idx >= input.len() || qh_idx >= input.len() {
                                continue;
                            }
                            
                            // Extract 2-bit extensions from qh
                            let qh_val = input[qh_idx];
                            let q_ext = (qh_val >> qh_shift) & 3;
                            
                            // Extract 4-bit base value from ql
                            let ql_val = input[ql_idx];
                            let q = if idx < 128 {
                                (ql_val & 0x0F) | (q_ext << 4)
                            } else {
                                (ql_val >> 4) | (q_ext << 4)
                            };
                            
                            // Apply scaling: d * scale * (q - 32) / 64
                            let dequantized = d * scales[is] * ((q as i8 - 32) as f32) / 64.0;
                            output[output_base + idx] = dequantized;
                        }
                        continue;
                    }
                    
                    // SIMD processing for full chunks
                    let scale_idx = chunk;
                    let scale_vec = _mm256_set1_ps(d * scales[scale_idx] / 64.0);
                    let offset_vec = _mm256_set1_ps(32.0);
                    
                    // Process 8 elements at a time with AVX2
                    for i in 0..2 {
                        let base_idx = chunk_offset + i * 8;
                        let mut values = [0u8; 8];
                        
                        // Extract 8 6-bit values
                        for j in 0..8 {
                            let idx = base_idx + j;
                            let is = idx / 16;
                            
                            // Calculate indices
                            let ql_idx = block_offset + 64 * (idx / 128) + 32 * ((idx % 128) / 64) + 16 * ((idx % 64) / 32) + (idx % 16);
                            let qh_idx = block_offset + 128 + 32 * (idx / 128) + 16 * ((idx % 128) / 64) + 8 * ((idx % 64) / 32) + (idx % 32) / 4;
                            let qh_shift = 2 * ((idx % 32) % 4);
                            
                            // Extract 2-bit extension
                            let qh_val = input[qh_idx];
                            let q_ext = (qh_val >> qh_shift) & 3;
                            
                            // Extract 4-bit base value
                            let ql_val = input[ql_idx];
                            let q = if idx < 128 {
                                (ql_val & 0x0F) | (q_ext << 4)
                            } else {
                                (ql_val >> 4) | (q_ext << 4)
                            };
                            
                            values[j] = q;
                        }
                        
                        // Convert to AVX2 and process
                        let vals = _mm_loadu_si128(values.as_ptr() as *const __m128i);
                        let vals_32 = _mm256_cvtepu8_epi32(vals);
                        let vals_float = _mm256_cvtepi32_ps(vals_32);
                        
                        // Apply formula: scale * (q - 32)
                        let offset_vals = _mm256_sub_ps(vals_float, offset_vec);
                        let scaled = _mm256_mul_ps(offset_vals, scale_vec);
                        
                        // Store results
                        _mm256_storeu_ps(output.as_mut_ptr().add(output_base + base_idx), scaled);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// NEON implementation for Q6_K dequantization
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn dequantize_q6_k_neon(&self, input: &[u8], output: &mut [f32], num_blocks: usize) -> Result<(), CoreError> {
        use std::arch::aarch64::*;
        
        eprintln!("🏁 Starting NEON Q6_K dequantization for {} blocks", num_blocks);
        
        const BLOCK_SIZE: usize = 256;
        const BYTES_PER_BLOCK: usize = 210;
        
        for block_idx in 0..num_blocks {
            let block_offset = block_idx * BYTES_PER_BLOCK;
            
            if block_offset + BYTES_PER_BLOCK > input.len() {
                break;
            }
            
            // Extract main scale factor (d) as f16 at offset 208-209
            let d_bytes = &input[block_offset + 208..block_offset + 210];
            let d_bits = u16::from_le_bytes([d_bytes[0], d_bytes[1]]);
            let d = half::f16::from_bits(d_bits).to_f32();
            
            // Load scale factor into NEON register
            let d_vec = vdupq_n_f32(d);
            
            // Extract scales (16 bytes starting at offset 192)
            let scales_start = block_offset + 192;
            
            // Process in chunks using NEON
            for chunk_idx in 0..4 {
                let chunk_start = chunk_idx * 64;
                let output_offset = block_idx * BLOCK_SIZE + chunk_start;
                
                if output_offset + 64 > output.len() {
                    break;
                }
                
                // Process 64 elements at a time
                for sub_chunk in 0..4 {
                    let elem_offset = chunk_start + sub_chunk * 16;
                    let scale_idx = elem_offset / 16;
                    
                    if scale_idx >= 16 {
                        break;
                    }
                    
                    // Get scale for this group
                    let scale = input[scales_start + scale_idx] as f32;
                    let scale_vec = vdupq_n_f32(scale);
                    
                    // Load quantized values
                    let ql_base = block_offset + elem_offset / 2;
                    let qh_base = block_offset + 128 + elem_offset / 4;
                    
                    // Process 16 values at a time
                    for i in 0..4 {
                        let idx = elem_offset + i * 4;
                        let out_idx = output_offset + sub_chunk * 16 + i * 4;
                        
                        if out_idx + 4 > output.len() {
                            break;
                        }
                        
                        // Extract 4 6-bit values
                        let mut vals = [0f32; 4];
                        for j in 0..4 {
                            let elem_idx = idx + j;
                            
                            // Calculate indices for Q6_K bit packing
                            let ql_idx = block_offset + 64 * (elem_idx / 128) + 32 * ((elem_idx % 128) / 64) + 
                                        16 * ((elem_idx % 64) / 32) + (elem_idx % 16);
                            let qh_idx = block_offset + 128 + 32 * (elem_idx / 128) + 16 * ((elem_idx % 128) / 64) + 
                                        8 * ((elem_idx % 64) / 32) + (elem_idx % 32) / 4;
                            let qh_shift = 2 * ((elem_idx % 32) % 4);
                            
                            if ql_idx < input.len() && qh_idx < input.len() {
                                let ql = input[ql_idx];
                                let qh = (input[qh_idx] >> qh_shift) & 0x03;
                                let q = (ql & 0x0F) | ((qh & 0x03) << 4);
                                vals[j] = q as f32 - 32.0;
                            }
                        }
                        
                        // Load values into NEON register
                        let val_vec = vld1q_f32(vals.as_ptr());
                        
                        // Multiply by scale and d
                        let scaled = vmulq_f32(val_vec, scale_vec);
                        let result = vmulq_f32(scaled, d_vec);
                        
                        // Store results
                        vst1q_f32(output[out_idx..].as_mut_ptr(), result);
                    }
                }
            }
        }
        
        eprintln!("✅ NEON Q6_K dequantization completed");
        Ok(())
    }
    
    /// Dequantize Q8_0 format (simpler 8-bit quantization)
    pub fn dequantize_q8_0(&self, input: &[u8], scales: &[f32], output: &mut [f32]) -> Result<(), CoreError> {
        if input.len() != output.len() || scales.is_empty() {
            return Err(CoreError::tensor(
                "DEQUANT_SIZE_MISMATCH",
                format!("Input size {} != output size {}", input.len(), output.len()),
                "Q8_0 dequantization",
                "Ensure input and output buffers have same size"
            ));
        }
        
        const BLOCK_SIZE: usize = 32;
        let num_blocks = (output.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        match self.dispatcher.simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => unsafe {
                self.dequantize_q8_0_avx2(input, scales, output, num_blocks)?;
            },
            
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon | SimdLevel::NeonDotprod => unsafe {
                self.dequantize_q8_0_neon(input, scales, output, num_blocks)?;
            },
            
            _ => {
                self.dequantize_q8_0_scalar(input, scales, output, num_blocks)?;
            }
        }
        
        Ok(())
    }
    
    /// Scalar Q8_0 dequantization
    fn dequantize_q8_0_scalar(&self, input: &[u8], scales: &[f32], output: &mut [f32], num_blocks: usize) -> Result<(), CoreError> {
        const BLOCK_SIZE: usize = 32;
        
        for block_idx in 0..num_blocks {
            let scale = scales[block_idx.min(scales.len() - 1)];
            
            for i in 0..BLOCK_SIZE {
                let global_idx = block_idx * BLOCK_SIZE + i;
                if global_idx >= output.len() {
                    break;
                }
                
                let quantized = input[global_idx] as i8; // Convert to signed
                output[global_idx] = quantized as f32 * scale;
            }
        }
        
        Ok(())
    }
    
    /// AVX2 Q8_0 dequantization
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dequantize_q8_0_avx2(&self, input: &[u8], scales: &[f32], output: &mut [f32], num_blocks: usize) -> Result<(), CoreError> {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            
            const BLOCK_SIZE: usize = 32;
            
            for block_idx in 0..num_blocks {
                let scale = scales[block_idx.min(scales.len() - 1)];
                let scale_vec = _mm256_set1_ps(scale);
                
                let block_start = block_idx * BLOCK_SIZE;
                if block_start >= output.len() {
                    break;
                }
                
                let elements_to_process = (output.len() - block_start).min(BLOCK_SIZE);
                
                // Process in chunks of 8 (AVX2 can handle 8 f32s at once)
                for chunk in (0..elements_to_process).step_by(8) {
                    let global_idx = block_start + chunk;
                    if global_idx + 8 > output.len() {
                        // Handle remaining elements with scalar
                        for i in 0..(output.len() - global_idx) {
                            let quantized = input[global_idx + i] as i8;
                            output[global_idx + i] = quantized as f32 * scale;
                        }
                        break;
                    }
                    
                    // Load 8 bytes as i8
                    let bytes = _mm_loadl_epi64(input.as_ptr().add(global_idx) as *const __m128i);
                    
                    // Convert to i32 (with sign extension)
                    let bytes_16 = _mm_unpacklo_epi8(bytes, _mm_cmpgt_epi8(_mm_setzero_si128(), bytes));
                    let i32_lo = _mm_unpacklo_epi16(bytes_16, _mm_cmpgt_epi16(_mm_setzero_si128(), bytes_16));
                    let i32_hi = _mm_unpackhi_epi16(bytes_16, _mm_cmpgt_epi16(_mm_setzero_si128(), bytes_16));
                    
                    // Convert to float
                    let f32_lo = _mm_cvtepi32_ps(i32_lo);
                    let f32_hi = _mm_cvtepi32_ps(i32_hi);
                    let f32_result = _mm256_set_m128(f32_hi, f32_lo);
                    
                    // Apply scale
                    let result = _mm256_mul_ps(f32_result, scale_vec);
                    
                    // Store result
                    _mm256_storeu_ps(output.as_mut_ptr().add(global_idx), result);
                }
            }
        }
        
        Ok(())
    }
    
    /// NEON Q8_0 dequantization
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn dequantize_q8_0_neon(&self, input: &[u8], scales: &[f32], output: &mut [f32], num_blocks: usize) -> Result<(), CoreError> {
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            
            const BLOCK_SIZE: usize = 32;
            
            for block_idx in 0..num_blocks {
                let scale = scales[block_idx.min(scales.len() - 1)];
                let scale_vec = vdupq_n_f32(scale);
                
                let block_start = block_idx * BLOCK_SIZE;
                if block_start >= output.len() {
                    break;
                }
                
                let elements_to_process = (output.len() - block_start).min(BLOCK_SIZE);
                
                // Process in chunks of 4 (NEON can handle 4 f32s at once)
                for chunk in (0..elements_to_process).step_by(4) {
                    let global_idx = block_start + chunk;
                    if global_idx + 4 > output.len() {
                        // Handle remaining elements with scalar
                        for i in 0..(output.len() - global_idx) {
                            let quantized = input[global_idx + i] as i8;
                            output[global_idx + i] = quantized as f32 * scale;
                        }
                        break;
                    }
                    
                    // Load 4 bytes as i8 and convert to i32
                    let bytes = vld1_u8(input.as_ptr().add(global_idx));
                    let signed_bytes = vreinterpret_s8_u8(bytes);
                    let i16_vals = vmovl_s8(signed_bytes);
                    let i32_vals = vmovl_s16(vget_low_s16(i16_vals));
                    
                    // Convert to float and apply scale
                    let f32_vals = vcvtq_f32_s32(i32_vals);
                    let result = vmulq_f32(f32_vals, scale_vec);
                    
                    // Store result
                    vst1q_f32(output.as_mut_ptr().add(global_idx), result);
                }
            }
        }
        
        Ok(())
    }
}

impl Default for OptimizedDequantizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for Q4_K dequantization
pub fn dequantize_q4_k_optimized(input: &[u8], output: &mut [f32]) -> Result<(), CoreError> {
    OptimizedDequantizer::get().dequantize_q4_k(input, output)
}

/// Convenience function for Q8_0 dequantization
pub fn dequantize_q8_0_optimized(input: &[u8], scales: &[f32], output: &mut [f32]) -> Result<(), CoreError> {
    OptimizedDequantizer::get().dequantize_q8_0(input, scales, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_q8_0_dequantization() {
        let dequantizer = OptimizedDequantizer::new();
        
        // Test data: 8 quantized values with scale 0.5
        let input = vec![0u8, 1, 127, 128, 129, 254, 255, 100];
        let scales = vec![0.5f32];
        let mut output = vec![0.0f32; input.len()];
        
        dequantizer.dequantize_q8_0(&input, &scales, &mut output).unwrap();
        
        // Check some expected values
        assert!((output[0] - (-64.0 * 0.5)).abs() < 0.001); // 0 as i8 = -128, but we use different encoding
        assert!((output[2] - (127.0 * 0.5)).abs() < 0.001); // 127 as i8 = 127
    }
    
    #[test]
    fn test_dequantizer_singleton() {
        let deq1 = OptimizedDequantizer::get();
        let deq2 = OptimizedDequantizer::get();
        
        // Should be the same instance
        assert!(std::ptr::eq(deq1, deq2));
    }
}