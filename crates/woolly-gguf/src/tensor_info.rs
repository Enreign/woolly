//! Tensor information and mapping structures
//!
//! This module contains structures for handling tensor metadata
//! and memory mapping information.

use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Read;

use crate::error::{Error, Result};

/// GGML tensor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    /// 32-bit float
    F32 = 0,
    /// 16-bit float
    F16 = 1,
    /// 4-bit quantization (type 0)
    Q4_0 = 2,
    /// 4-bit quantization (type 1) 
    Q4_1 = 3,
    /// 5-bit quantization (type 0)
    Q5_0 = 6,
    /// 5-bit quantization (type 1)
    Q5_1 = 7,
    /// 8-bit quantization (type 0)
    Q8_0 = 8,
    /// 8-bit quantization (type 1)
    Q8_1 = 9,
    /// 2-bit quantization
    Q2_K = 10,
    /// 3-bit quantization
    Q3_K = 11,
    /// 4-bit quantization (K-type)
    Q4_K = 12,
    /// 5-bit quantization (K-type)
    Q5_K = 13,
    /// 6-bit quantization (K-type)
    Q6_K = 14,
    /// 8-bit quantization (K-type)
    Q8_K = 15,
    /// IQ2 quantization (XXS variant)
    IQ2_XXS = 16,
    /// IQ2 quantization (XS variant)
    IQ2_XS = 17,
    /// IQ3 quantization (XXS variant)
    IQ3_XXS = 18,
    /// IQ1 quantization (S variant)
    IQ1_S = 19,
    /// IQ4 quantization (NL variant)
    IQ4_NL = 20,
    /// IQ3 quantization (S variant)
    IQ3_S = 21,
    /// IQ2 quantization (S variant)
    IQ2_S = 22,
    /// IQ4 quantization (XS variant)
    IQ4_XS = 23,
    /// 8-bit integer
    I8 = 24,
    /// 16-bit integer
    I16 = 25,
    /// 32-bit integer
    I32 = 26,
    /// 64-bit integer
    I64 = 27,
    /// 64-bit float
    F64 = 28,
    /// IQ1 quantization (M variant)
    IQ1_M = 29,
    /// BF16 format
    BF16 = 30,
}

impl GGMLType {
    /// Try to create from a u32 value
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            16 => Some(Self::IQ2_XXS),
            17 => Some(Self::IQ2_XS),
            18 => Some(Self::IQ3_XXS),
            19 => Some(Self::IQ1_S),
            20 => Some(Self::IQ4_NL),
            21 => Some(Self::IQ3_S),
            22 => Some(Self::IQ2_S),
            23 => Some(Self::IQ4_XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1_M),
            30 => Some(Self::BF16),
            _ => None,
        }
    }
    
    /// Get the size of a single element in bytes
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            _ => panic!("element_size called on quantized type"),
        }
    }
    
    /// Get the block size for quantized types
    pub fn block_size(&self) -> usize {
        match self {
            Self::Q4_0 => 32,
            Self::Q4_1 => 32,
            Self::Q5_0 => 32,
            Self::Q5_1 => 32,
            Self::Q8_0 => 32,
            Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            Self::IQ2_XXS | Self::IQ2_XS | Self::IQ2_S => 256,
            Self::IQ3_XXS | Self::IQ3_S => 256,
            Self::IQ1_S | Self::IQ1_M => 256,
            Self::IQ4_NL | Self::IQ4_XS => 256,
            _ => 1,
        }
    }
    
    /// Get the size of a quantized block in bytes
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2_K => 82,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ3_XXS => 98,
            Self::IQ1_S => 50,
            Self::IQ4_NL => 130,
            Self::IQ3_S => 110,
            Self::IQ2_S => 82,
            Self::IQ4_XS => 136,
            Self::IQ1_M => 56,
        }
    }
    
    /// Check if this is a quantized type
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::F64 | Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }
}

/// Information about a tensor in the GGUF file
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    
    /// Number of dimensions
    pub n_dims: u32,
    
    /// Dimensions (shape)
    pub dims: Vec<u64>,
    
    /// Data type
    pub ggml_type: GGMLType,
    
    /// Offset in the file where tensor data starts
    pub offset: u64,
}

impl TensorInfo {
    /// Read tensor info from a reader
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        // Read name
        let name_len = reader.read_u64::<LittleEndian>()? as usize;
        let mut name_buf = vec![0u8; name_len];
        reader.read_exact(&mut name_buf)?;
        let name = String::from_utf8(name_buf)
            .map_err(|_| Error::InvalidString)?;
        
        // Read dimensions
        let n_dims = reader.read_u32::<LittleEndian>()?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(reader.read_u64::<LittleEndian>()?);
        }
        
        // Read type
        let type_value = reader.read_u32::<LittleEndian>()?;
        let ggml_type = GGMLType::from_u32(type_value)
            .ok_or_else(|| Error::InvalidTensorType(type_value))?;
        
        // Read offset
        let offset = reader.read_u64::<LittleEndian>()?;
        
        Ok(Self {
            name,
            n_dims,
            dims,
            ggml_type,
            offset,
        })
    }
    
    /// Calculate the number of elements in the tensor
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
    
    /// Calculate the size of the tensor data in bytes
    pub fn data_size(&self) -> u64 {
        let n_elements = self.n_elements();
        
        if self.ggml_type.is_quantized() {
            let block_size = self.ggml_type.block_size() as u64;
            let type_size = self.ggml_type.type_size() as u64;
            let n_blocks = (n_elements + block_size - 1) / block_size;
            n_blocks * type_size
        } else {
            n_elements * self.ggml_type.element_size() as u64
        }
    }
    
    /// Get the shape as a slice
    pub fn shape(&self) -> &[u64] {
        &self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ggml_type_sizes() {
        assert_eq!(GGMLType::F32.element_size(), 4);
        assert_eq!(GGMLType::F16.element_size(), 2);
        assert_eq!(GGMLType::I8.element_size(), 1);
        
        assert_eq!(GGMLType::Q4_0.block_size(), 32);
        assert_eq!(GGMLType::Q4_0.type_size(), 18);
        
        assert_eq!(GGMLType::Q2_K.block_size(), 256);
        assert_eq!(GGMLType::Q2_K.type_size(), 82);
    }
    
    #[test]
    fn test_tensor_info_size_calculation() {
        let tensor = TensorInfo {
            name: "test".to_string(),
            n_dims: 2,
            dims: vec![4, 8],
            ggml_type: GGMLType::F32,
            offset: 0,
        };
        
        assert_eq!(tensor.n_elements(), 32);
        assert_eq!(tensor.data_size(), 128); // 32 * 4 bytes
        
        let quantized_tensor = TensorInfo {
            name: "test_q".to_string(),
            n_dims: 2,
            dims: vec![64, 64],
            ggml_type: GGMLType::Q4_0,
            offset: 0,
        };
        
        assert_eq!(quantized_tensor.n_elements(), 4096);
        // 4096 elements / 32 block_size = 128 blocks
        // 128 blocks * 18 bytes per block = 2304 bytes
        assert_eq!(quantized_tensor.data_size(), 2304);
    }
}