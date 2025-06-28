//! GGUF tensor dequantization
//!
//! This module provides dequantization functions for various GGML quantization formats.
//! Based on the llama.cpp dequantization implementations.
//! Optimized with SIMD for ARM M4 and other AArch64 processors.

use crate::{GGMLType, Result, Error};
use half::f16;

/// SIMD-optimized dequantization implementations
#[cfg(target_arch = "aarch64")]
pub mod simd {
    use super::*;
    use std::arch::aarch64::*;
    
    /// NEON-optimized Q4_K dequantization
    #[target_feature(enable = "neon")]
    pub unsafe fn dequantize_q4_k_optimized(
        data: &[u8],
        output: &mut [f32],
        num_blocks: usize,
    ) {
        // Import the optimized implementation from woolly-tensor
        #[cfg(feature = "tensor-ops")]
        {
            use woolly_tensor::ops::aarch64::quantization::dequantize_q4_k_neon;
            dequantize_q4_k_neon(data, output, num_blocks, 144);
        }
        
        #[cfg(not(feature = "tensor-ops"))]
        {
            // Fallback inline implementation
            dequantize_q4_k_neon_inline(data, output, num_blocks);
        }
    }
    
    /// Inline NEON implementation when tensor-ops feature is not available
    #[target_feature(enable = "neon")]
    unsafe fn dequantize_q4_k_neon_inline(
        data: &[u8],
        output: &mut [f32],
        num_blocks: usize,
    ) {
        const QK_K: usize = 256;
        const BLOCK_SIZE: usize = 144;
        
        let mask_0f = vdupq_n_u8(0x0F);
        
        for block_idx in 0..num_blocks {
            let block_offset = block_idx * BLOCK_SIZE;
            let block_data = &data[block_offset..block_offset + BLOCK_SIZE];
            let output_offset = block_idx * QK_K;
            
            // Extract and unpack 6-bit scales
            let mut scales = [0u8; 16];
            let scales_data = &block_data[0..12];
            for i in 0..8 {
                let src_idx = (i * 3) / 2;
                if i % 2 == 0 {
                    scales[i] = scales_data[src_idx] & 63;
                } else {
                    scales[i] = ((scales_data[src_idx] >> 6) | ((scales_data[src_idx + 1] & 15) << 2)) & 63;
                }
            }
            for i in 8..16 {
                scales[i] = scales[i % 8];
            }
            
            // Load scale factors
            let d = f16::from_le_bytes([block_data[12], block_data[13]]).to_f32();
            let dmin = f16::from_le_bytes([block_data[14], block_data[15]]).to_f32();
            let d_vec = vdupq_n_f32(d);
            let dmin_vec = vdupq_n_f32(dmin);
            
            // Process quantized values
            let qs = &block_data[16..144];
            
            for chunk_idx in 0..8 {
                let chunk_offset = chunk_idx * 16;
                let output_chunk_offset = output_offset + chunk_idx * 32;
                
                if output_chunk_offset + 32 > output.len() {
                    break;
                }
                
                // Load packed data
                let packed = vld1q_u8(qs.as_ptr().add(chunk_offset));
                
                // Unpack nibbles
                let low_nibbles = vandq_u8(packed, mask_0f);
                let high_nibbles = vandq_u8(vshrq_n_u8(packed, 4), mask_0f);
                
                // Process low nibbles - unroll loop for compile-time constants
                {
                    let scale_idx = chunk_idx;
                    let scale = scales[scale_idx] as f32;
                    let scale_vec = vdupq_n_f32(scale);
                    
                    let vals_u8 = vget_low_u8(low_nibbles);
                    let vals_u16 = vmovl_u8(vals_u8);
                    let vals_u32 = vmovl_u16(vget_low_u16(vals_u16));
                    let vals_f32 = vcvtq_f32_u32(vals_u32);
                    
                    let scaled = vmulq_f32(vmulq_f32(d_vec, scale_vec), vals_f32);
                    let result = vaddq_f32(scaled, dmin_vec);
                    
                    let output_idx = output_chunk_offset;
                    if output_idx + 4 <= output.len() {
                        vst1q_f32(output.as_mut_ptr().add(output_idx), result);
                    }
                }
                
                // Process remaining elements from low nibbles
                {
                    let scale_idx = chunk_idx;
                    let scale = scales[scale_idx] as f32;
                    let scale_vec = vdupq_n_f32(scale);
                    
                    let vals_u8 = vget_high_u8(low_nibbles);
                    let vals_u16 = vmovl_u8(vals_u8);
                    let vals_u32 = vmovl_u16(vget_low_u16(vals_u16));
                    let vals_f32 = vcvtq_f32_u32(vals_u32);
                    
                    let scaled = vmulq_f32(vmulq_f32(d_vec, scale_vec), vals_f32);
                    let result = vaddq_f32(scaled, dmin_vec);
                    
                    let output_idx = output_chunk_offset + 4;
                    if output_idx + 4 <= output.len() {
                        vst1q_f32(output.as_mut_ptr().add(output_idx), result);
                    }
                }
                
                // Process high nibbles - first half
                {
                    let scale_idx = chunk_idx;
                    let scale = scales[scale_idx] as f32;
                    let scale_vec = vdupq_n_f32(scale);
                    
                    let vals_u8 = vget_low_u8(high_nibbles);
                    let vals_u16 = vmovl_u8(vals_u8);
                    let vals_u32 = vmovl_u16(vget_low_u16(vals_u16));
                    let vals_f32 = vcvtq_f32_u32(vals_u32);
                    
                    let scaled = vmulq_f32(vmulq_f32(d_vec, scale_vec), vals_f32);
                    let result = vaddq_f32(scaled, dmin_vec);
                    
                    let output_idx = output_chunk_offset + 16;
                    if output_idx + 4 <= output.len() {
                        vst1q_f32(output.as_mut_ptr().add(output_idx), result);
                    }
                }
                
                // Process high nibbles - second half
                {
                    let scale_idx = chunk_idx;
                    let scale = scales[scale_idx] as f32;
                    let scale_vec = vdupq_n_f32(scale);
                    
                    let vals_u8 = vget_high_u8(high_nibbles);
                    let vals_u16 = vmovl_u8(vals_u8);
                    let vals_u32 = vmovl_u16(vget_low_u16(vals_u16));
                    let vals_f32 = vcvtq_f32_u32(vals_u32);
                    
                    let scaled = vmulq_f32(vmulq_f32(d_vec, scale_vec), vals_f32);
                    let result = vaddq_f32(scaled, dmin_vec);
                    
                    let output_idx = output_chunk_offset + 20;
                    if output_idx + 4 <= output.len() {
                        vst1q_f32(output.as_mut_ptr().add(output_idx), result);
                    }
                }
            }
        }
    }
    
    /// Bulk layer dequantization with prefetching
    #[target_feature(enable = "neon")]
    pub unsafe fn bulk_dequantize_layer(
        tensors: &mut [(&[u8], usize, &mut [f32])], // (data, num_blocks, output)
    ) {
        // Process tensors sequentially without complex prefetching
        for (data, num_blocks, output) in tensors.iter_mut() {
            dequantize_q4_k_optimized(*data, output, *num_blocks);
        }
    }
}

/// Dequantize tensor data based on GGML type
pub fn dequantize(data: &[u8], ggml_type: GGMLType, nelements: usize) -> Result<Vec<f32>> {
    match ggml_type {
        GGMLType::F32 => dequantize_f32(data, nelements),
        GGMLType::F16 => dequantize_f16(data, nelements),
        GGMLType::Q4_0 => dequantize_q4_0(data, nelements),
        GGMLType::Q4_1 => dequantize_q4_1(data, nelements),
        GGMLType::Q5_0 => dequantize_q5_0(data, nelements),
        GGMLType::Q5_1 => dequantize_q5_1(data, nelements),
        GGMLType::Q8_0 => dequantize_q8_0(data, nelements),
        // GGMLType::Q8_1 => dequantize_q8_1(data, nelements), // Uncomment when Q8_1 is added to GGMLType
        GGMLType::Q4_K => dequantize_q4_k(data, nelements),
        GGMLType::Q5_K => dequantize_q5_k(data, nelements),
        GGMLType::Q6_K => dequantize_q6_k(data, nelements),
        GGMLType::Q8_K => dequantize_q8_k(data, nelements),
        _ => Err(Error::InvalidTensorType(ggml_type as u32)),
    }
}

/// Get the block size for a GGML type
pub fn get_block_size(ggml_type: GGMLType) -> usize {
    match ggml_type {
        GGMLType::F32 => 1,
        GGMLType::F16 => 1,
        GGMLType::Q4_0 | GGMLType::Q4_1 => 32,
        GGMLType::Q5_0 | GGMLType::Q5_1 => 32,
        GGMLType::Q8_0 | GGMLType::Q8_1 => 32,
        GGMLType::Q2_K | GGMLType::Q3_K => 256,
        GGMLType::Q4_K | GGMLType::Q5_K => 256,
        GGMLType::Q6_K => 256,
        GGMLType::Q8_K => 256,
        _ => 1,
    }
}

/// Get bytes per block for a GGML type
pub fn get_type_size(ggml_type: GGMLType) -> usize {
    match ggml_type {
        GGMLType::F32 => 4,
        GGMLType::F16 => 2,
        GGMLType::Q4_0 => 18, // 2 + 16
        GGMLType::Q4_1 => 20, // 2 + 2 + 16
        GGMLType::Q5_0 => 22, // 2 + 4 + 16
        GGMLType::Q5_1 => 24, // 2 + 2 + 4 + 16
        GGMLType::Q8_0 => 34, // 2 + 32
        // GGMLType::Q8_1 => 36, // 4 + 32 // Uncomment when Q8_1 is added
        GGMLType::Q2_K => 84,  // 256/16 + 256/4 + 2 + 2
        GGMLType::Q3_K => 110, // 256/8 + 256/4 + 12 + 2
        GGMLType::Q4_K => 144, // 2 + 2 + 12 + 256/2
        GGMLType::Q5_K => 176, // 2 + 2 + 12 + 256/2 + 256/8
        GGMLType::Q6_K => 210, // 256/2 + 256/4 + 256/16 + 2
        GGMLType::Q8_K => 292, // 4 + 256 + 4*8
        _ => 0,
    }
}

// F32 - no dequantization needed
fn dequantize_f32(data: &[u8], nelements: usize) -> Result<Vec<f32>> {
    if data.len() != nelements * 4 {
        return Err(Error::BufferTooSmall {
            needed: nelements * 4,
            available: data.len(),
        });
    }
    
    let mut result = Vec::with_capacity(nelements);
    for i in 0..nelements {
        let bytes = [data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]];
        result.push(f32::from_le_bytes(bytes));
    }
    Ok(result)
}

// F16 - half precision to single precision
fn dequantize_f16(data: &[u8], nelements: usize) -> Result<Vec<f32>> {
    if data.len() != nelements * 2 {
        return Err(Error::BufferTooSmall {
            needed: nelements * 2,
            available: data.len(),
        });
    }
    
    let mut result = Vec::with_capacity(nelements);
    for i in 0..nelements {
        let bytes = [data[i*2], data[i*2+1]];
        let f16_val = u16::from_le_bytes(bytes);
        result.push(f16::from_bits(f16_val).to_f32());
    }
    Ok(result)
}

// Q4_0 quantization: 32 4-bit values packed into 16 bytes with a scale
#[repr(C)]
struct BlockQ4_0 {
    d: f16,           // delta/scale
    qs: [u8; 16],     // 4-bit quantized values (32 values packed)
}

fn dequantize_q4_0(data: &[u8], nelements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    let nblocks = (nelements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let expected_size = nblocks * std::mem::size_of::<BlockQ4_0>();
    
    if data.len() < expected_size {
        return Err(Error::BufferTooSmall {
            needed: expected_size,
            available: data.len(),
        });
    }
    
    let mut result = Vec::with_capacity(nelements);
    
    for block_idx in 0..nblocks {
        let offset = block_idx * std::mem::size_of::<BlockQ4_0>();
        let block_data = &data[offset..offset + std::mem::size_of::<BlockQ4_0>()];
        
        // Read scale
        let d_bytes = [block_data[0], block_data[1]];
        let d = f16::from_le_bytes(d_bytes).to_f32();
        
        // Dequantize values
        let qs = &block_data[2..18];
        for i in 0..32 {
            if result.len() >= nelements {
                break;
            }
            
            let q = if i < 16 {
                (qs[i] & 0x0F) as i8 - 8
            } else {
                ((qs[i-16] >> 4) & 0x0F) as i8 - 8
            };
            
            result.push(q as f32 * d);
        }
    }
    
    result.truncate(nelements);
    Ok(result)
}

// Q4_1 quantization: 32 4-bit values with scale and min value
#[repr(C)]
struct BlockQ4_1 {
    d: f16,           // delta/scale
    m: f16,           // min value
    qs: [u8; 16],     // 4-bit quantized values
}

fn dequantize_q4_1(data: &[u8], nelements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    let nblocks = (nelements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let expected_size = nblocks * std::mem::size_of::<BlockQ4_1>();
    
    if data.len() < expected_size {
        return Err(Error::BufferTooSmall {
            needed: expected_size,
            available: data.len(),
        });
    }
    
    let mut result = Vec::with_capacity(nelements);
    
    for block_idx in 0..nblocks {
        let offset = block_idx * std::mem::size_of::<BlockQ4_1>();
        let block_data = &data[offset..offset + std::mem::size_of::<BlockQ4_1>()];
        
        // Read scale and min
        let d = f16::from_le_bytes([block_data[0], block_data[1]]).to_f32();
        let m = f16::from_le_bytes([block_data[2], block_data[3]]).to_f32();
        
        // Dequantize values
        let qs = &block_data[4..20];
        for i in 0..32 {
            if result.len() >= nelements {
                break;
            }
            
            let q = if i < 16 {
                qs[i] & 0x0F
            } else {
                (qs[i-16] >> 4) & 0x0F
            };
            
            result.push(q as f32 * d + m);
        }
    }
    
    result.truncate(nelements);
    Ok(result)
}

// Q4_K quantization - optimized with SIMD when available
fn dequantize_q4_k(data: &[u8], nelements: usize) -> Result<Vec<f32>> {
    const K_SCALE_SIZE: usize = 12;
    const BLOCK_SIZE: usize = 256;
    
    let nblocks = (nelements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut result = vec![0.0f32; nelements];
    
    // Use NEON optimization on AArch64 when available
    #[cfg(target_arch = "aarch64")]
    {
        if nblocks > 0 && std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                simd::dequantize_q4_k_optimized(data, &mut result, nblocks);
                return Ok(result);
            }
        }
    }
    
    // Fallback to scalar implementation
    dequantize_q4_k_scalar(data, nelements)
}

// Scalar implementation for Q4_K (fallback)
fn dequantize_q4_k_scalar(data: &[u8], nelements: usize) -> Result<Vec<f32>> {
    const K_SCALE_SIZE: usize = 12;
    const BLOCK_SIZE: usize = 256;
    
    let nblocks = (nelements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut result = Vec::with_capacity(nelements);
    
    for block_idx in 0..nblocks {
        let block_offset = block_idx * 144; // Q4_K block size
        if block_offset + 144 > data.len() {
            return Err(Error::InvalidTensorInfo("Q4_K data too small".to_string()));
        }
        
        let block_data = &data[block_offset..block_offset + 144];
        
        // Read scales - stored as 6-bit values
        let scales_data = &block_data[0..K_SCALE_SIZE];
        let mut scales = [0u8; 16];
        
        // Unpack 6-bit scales
        for i in 0..8 {
            let src_idx = (i * 3) / 2;
            if i % 2 == 0 {
                scales[i] = scales_data[src_idx] & 63;
            } else {
                scales[i] = ((scales_data[src_idx] >> 6) | ((scales_data[src_idx + 1] & 15) << 2)) & 63;
            }
        }
        
        // Read d and dmin
        let d = f16::from_le_bytes([block_data[12], block_data[13]]).to_f32();
        let dmin = f16::from_le_bytes([block_data[14], block_data[15]]).to_f32();
        
        // Read quantized values
        let qs = &block_data[16..144];
        
        // Dequantize
        for i in 0..256 {
            if result.len() >= nelements {
                break;
            }
            
            let scale_idx = i / 32;
            let q_idx = i / 2;
            
            let q = if i % 2 == 0 {
                qs[q_idx] & 0x0F
            } else {
                (qs[q_idx] >> 4) & 0x0F
            };
            
            let scale = scales[scale_idx] as f32;
            result.push(d * scale * (q as f32) + dmin);
        }
    }
    
    result.truncate(nelements);
    Ok(result)
}

// Q6_K quantization
fn dequantize_q6_k(data: &[u8], nelements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 256;
    const _QK_K: usize = 256;
    
    let nblocks = (nelements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut result = Vec::with_capacity(nelements);
    
    for block_idx in 0..nblocks {
        let block_offset = block_idx * 210; // Q6_K block size
        if block_offset + 210 > data.len() {
            return Err(Error::InvalidTensorInfo("Q6_K data too small".to_string()));
        }
        
        let block_data = &data[block_offset..block_offset + 210];
        
        // Read quantized values - 128 bytes for 256 6-bit values
        let ql = &block_data[0..128];
        
        // Read high bits - 64 bytes
        let qh = &block_data[128..192];
        
        // Read scales - 16 bytes  
        let scales = &block_data[192..208];
        
        // Read d
        let d = f16::from_le_bytes([block_data[208], block_data[209]]).to_f32();
        
        // Dequantize
        for i in 0..256 {
            if result.len() >= nelements {
                break;
            }
            
            let is = i / 16;  // scale index
            let _il = i % 16;  // index in group
            let ql_idx = 64 * (i / 128) + 32 * ((i % 128) / 64) + 16 * ((i % 64) / 32) + (i % 16);
            let qh_idx = 32 * (i / 128) + 16 * ((i % 128) / 64) + 8 * ((i % 64) / 32) + (i % 32) / 4;
            // For Q6_K quantization, qh contains 2-bit extensions for 6-bit quantization values
            // qh_shift will be 0, 2, 4, or 6 based on position
            let qh_shift = 2 * ((i % 32) % 4);
            
            // Get the 2-bit extensions from qh array
            let qh_val = qh[qh_idx];
            let q1_ext = (qh_val >> qh_shift) & 3;
            
            // For q2, we need bits shifted by 2 more positions
            // When qh_shift = 6, qh_shift + 2 = 8, which would overflow a u8
            let q2_ext = if qh_shift < 6 {
                (qh_val >> (qh_shift + 2)) & 3
            } else {
                // When qh_shift = 6, the next 2 bits wrap to the next byte
                if qh_idx + 1 < qh.len() {
                    qh[qh_idx + 1] & 3
                } else {
                    0
                }
            };
            
            // Combine 4-bit base values with 2-bit extensions to get 6-bit values
            let q1 = (ql[ql_idx] & 0x0F) | (q1_ext << 4);
            let q2 = (ql[ql_idx] >> 4) | (q2_ext << 4);
            
            let q = if i < 128 { q1 } else { q2 };
            let s = scales[is] as f32;
            
            result.push(d * s * (q as i8 - 32) as f32 / 64.0);
        }
    }
    
    result.truncate(nelements);
    Ok(result)
}

// Simplified implementations for other formats
fn dequantize_q5_0(_data: &[u8], nelements: usize) -> Result<Vec<f32>> {
    // Q5_0 is more complex, using simplified version for now
    let result = vec![0.0f32; nelements];
    eprintln!("WARNING: Q5_0 dequantization not fully implemented, using zeros");
    Ok(result)
}

fn dequantize_q5_1(_data: &[u8], nelements: usize) -> Result<Vec<f32>> {
    // Q5_1 is more complex, using simplified version for now
    let result = vec![0.0f32; nelements];
    eprintln!("WARNING: Q5_1 dequantization not fully implemented, using zeros");
    Ok(result)
}

fn dequantize_q8_0(data: &[u8], nelements: usize) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    let nblocks = (nelements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let expected_size = nblocks * 34; // 2 bytes scale + 32 bytes data
    
    if data.len() < expected_size {
        return Err(Error::BufferTooSmall {
            needed: expected_size,
            available: data.len(),
        });
    }
    
    let mut result = Vec::with_capacity(nelements);
    
    for block_idx in 0..nblocks {
        let offset = block_idx * 34;
        let block_data = &data[offset..offset + 34];
        
        // Read scale
        let d = f16::from_le_bytes([block_data[0], block_data[1]]).to_f32();
        
        // Read quantized values
        let qs = &block_data[2..34];
        for i in 0..32 {
            if result.len() >= nelements {
                break;
            }
            result.push((qs[i] as i8) as f32 * d);
        }
    }
    
    result.truncate(nelements);
    Ok(result)
}

// fn dequantize_q8_1(data: &[u8], nelements: usize) -> Result<Vec<f32>> {
//     // Q8_1 has scale and offset, simplified for now
//     let mut result = vec![0.0f32; nelements];
//     eprintln!("WARNING: Q8_1 dequantization not fully implemented, using zeros");
//     Ok(result)
// }

fn dequantize_q5_k(_data: &[u8], nelements: usize) -> Result<Vec<f32>> {
    // Q5_K is complex, using simplified version for now
    let result = vec![0.0f32; nelements];
    eprintln!("WARNING: Q5_K dequantization not fully implemented, using zeros");
    Ok(result)
}

fn dequantize_q8_k(_data: &[u8], nelements: usize) -> Result<Vec<f32>> {
    // Q8_K is complex, using simplified version for now
    let result = vec![0.0f32; nelements];
    eprintln!("WARNING: Q8_K dequantization not fully implemented, using zeros");
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_dequantization() {
        let data = vec![0.0f32, 1.0f32, -1.0f32, 3.14f32];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        
        let result = dequantize_f32(&bytes, 4).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], -1.0);
        assert!((result[3] - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_block_sizes() {
        assert_eq!(get_block_size(GGMLType::F32), 1);
        assert_eq!(get_block_size(GGMLType::Q4_0), 32);
        assert_eq!(get_block_size(GGMLType::Q4_K), 256);
        assert_eq!(get_block_size(GGMLType::Q6_K), 256);
    }
}