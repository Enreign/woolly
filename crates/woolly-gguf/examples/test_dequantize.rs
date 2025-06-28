//! Test dequantization with a GGUF file

use woolly_gguf::{GGUFLoader, dequantize};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new("models/granite-3.3-8b-instruct-Q4_K_M.gguf");
    
    println!("Loading GGUF file: {:?}", model_path);
    let loader = GGUFLoader::from_path(model_path)?;
    
    println!("Model loaded successfully!");
    println!("Architecture: {:?}", loader.architecture());
    
    // Get first few tensor names
    let tensor_names = loader.tensor_names();
    println!("\nFound {} tensors", tensor_names.len());
    
    // Test dequantization on first few quantized tensors
    let mut tested = 0;
    for (i, name) in tensor_names.iter().enumerate() {
        if tested >= 3 {
            break;
        }
        
        if let Some(info) = loader.tensor_info(name) {
            println!("\n{}. Tensor: {} (type: {:?}, shape: {:?})", 
                i + 1, name, info.ggml_type, info.shape());
            
            // Only test quantized tensors
            match info.ggml_type {
                woolly_gguf::GGMLType::Q4_0 | 
                woolly_gguf::GGMLType::Q4_1 | 
                woolly_gguf::GGMLType::Q4_K | 
                woolly_gguf::GGMLType::Q6_K | 
                woolly_gguf::GGMLType::Q8_0 => {
                    println!("   Testing dequantization...");
                    
                    match loader.tensor_data(name) {
                        Ok(data) => {
                            let num_elements: usize = info.shape().iter()
                                .map(|&x| x as usize)
                                .product();
                            
                            match dequantize(&data, info.ggml_type, num_elements) {
                                Ok(dequantized) => {
                                    println!("   ✓ Successfully dequantized {} elements", dequantized.len());
                                    println!("   First few values: {:?}", &dequantized[..5.min(dequantized.len())]);
                                    tested += 1;
                                }
                                Err(e) => {
                                    println!("   ✗ Dequantization failed: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            println!("   ✗ Failed to get tensor data: {}", e);
                        }
                    }
                }
                _ => {
                    println!("   Skipping non-quantized tensor");
                }
            }
        }
    }
    
    println!("\nDequantization test complete!");
    Ok(())
}