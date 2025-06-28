//! BLAS-optimized attention mechanism for transformer models
//! 
//! Replaces manual dot product loops with efficient matrix operations

use crate::{Result, CoreError};
use crate::tensor_utils::SimpleTensor;
use crate::blas_matmul::{matmul_blas, is_blas_available};
use woolly_tensor::Shape;

/// Compute scaled dot-product attention using BLAS
/// 
/// attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
pub fn scaled_dot_product_attention_blas(
    queries: &[f32],      // [seq_len, num_heads, head_dim]
    keys: &[f32],         // [total_seq_len, num_heads, head_dim] 
    values: &[f32],       // [total_seq_len, num_heads, head_dim]
    seq_len: usize,
    total_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    // Reshape for batched matrix multiplication
    // We'll process all heads at once for efficiency
    
    let mut output = vec![0.0f32; seq_len * num_heads * head_dim];
    
    // Process each head
    for head in 0..num_heads {
        // Extract Q for this head: [seq_len, head_dim]
        let q_start = head * head_dim;
        let q_head = queries.chunks(num_heads * head_dim)
            .flat_map(|chunk| &chunk[q_start..q_start + head_dim])
            .collect::<Vec<_>>();
        
        // Extract K for this head: [total_seq_len, head_dim]
        let k_head: Vec<f32> = keys.chunks(num_heads * head_dim)
            .flat_map(|chunk| &chunk[q_start..q_start + head_dim])
            .copied()
            .collect();
        
        // Compute QK^T using BLAS: [seq_len, total_seq_len]
        let q_tensor = SimpleTensor {
            data: q_head.into_iter().copied().collect(),
            shape: Shape::matrix(seq_len, head_dim),
        };
        
        // K needs to be transposed for QK^T
        let k_transposed = transpose_matrix(&k_head, total_seq_len, head_dim);
        let k_tensor = SimpleTensor {
            data: k_transposed,
            shape: Shape::matrix(head_dim, total_seq_len),
        };
        
        // QK^T with BLAS
        let scores = if is_blas_available() {
            matmul_blas(&q_tensor, &k_tensor)
                .unwrap_or_else(|| compute_scores_fallback(&q_tensor, &k_tensor))
        } else {
            compute_scores_fallback(&q_tensor, &k_tensor)
        };
        
        // Apply scale and causal mask
        let mut attention_weights = scores.data;
        for i in 0..seq_len {
            for j in 0..total_seq_len {
                let idx = i * total_seq_len + j;
                attention_weights[idx] *= scale;
                
                // Causal mask: can't attend to future positions
                if j > (seq_len - 1 - i) + total_seq_len - seq_len {
                    attention_weights[idx] = f32::NEG_INFINITY;
                }
            }
        }
        
        // Softmax over each row
        apply_softmax_rows(&mut attention_weights, seq_len, total_seq_len);
        
        // Extract V for this head: [total_seq_len, head_dim]
        let v_head: Vec<f32> = values.chunks(num_heads * head_dim)
            .flat_map(|chunk| &chunk[q_start..q_start + head_dim])
            .copied()
            .collect();
        
        let v_tensor = SimpleTensor {
            data: v_head,
            shape: Shape::matrix(total_seq_len, head_dim),
        };
        
        let attn_tensor = SimpleTensor {
            data: attention_weights,
            shape: Shape::matrix(seq_len, total_seq_len),
        };
        
        // Compute attention output with BLAS
        let head_output = if is_blas_available() {
            matmul_blas(&attn_tensor, &v_tensor)
                .unwrap_or_else(|| compute_attention_fallback(&attn_tensor, &v_tensor))
        } else {
            compute_attention_fallback(&attn_tensor, &v_tensor)
        };
        
        // Copy to output
        for pos in 0..seq_len {
            let out_start = pos * num_heads * head_dim + head * head_dim;
            let head_start = pos * head_dim;
            output[out_start..out_start + head_dim]
                .copy_from_slice(&head_output.data[head_start..head_start + head_dim]);
        }
    }
    
    Ok(output)
}

/// Grouped Query Attention (GQA) with BLAS optimization
pub fn grouped_query_attention_blas(
    queries: &[f32],      // [seq_len, num_heads * head_dim]
    keys: &[f32],         // [total_seq_len, num_kv_heads * head_dim]
    values: &[f32],       // [total_seq_len, num_kv_heads * head_dim]
    seq_len: usize,
    total_seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let head_groups = num_heads / num_kv_heads;
    let mut output = vec![0.0f32; seq_len * num_heads * head_dim];
    
    eprintln!("🚀 Using BLAS for GQA attention (seq_len={}, heads={}, kv_heads={})", 
              seq_len, num_heads, num_kv_heads);
    
    // Process each KV head group
    for kv_head in 0..num_kv_heads {
        // Extract K and V for this KV head
        let kv_start = kv_head * head_dim;
        
        // Process all Q heads in this group together
        for group_idx in 0..head_groups {
            let q_head = kv_head * head_groups + group_idx;
            let q_start = q_head * head_dim;
            
            // Extract queries for this head
            let mut q_data = Vec::with_capacity(seq_len * head_dim);
            for pos in 0..seq_len {
                let offset = pos * num_heads * head_dim + q_start;
                q_data.extend_from_slice(&queries[offset..offset + head_dim]);
            }
            
            // Extract keys for this KV head
            let mut k_data = Vec::with_capacity(total_seq_len * head_dim);
            for pos in 0..total_seq_len {
                let offset = pos * num_kv_heads * head_dim + kv_start;
                k_data.extend_from_slice(&keys[offset..offset + head_dim]);
            }
            
            // Compute attention scores with BLAS
            let q_tensor = SimpleTensor {
                data: q_data,
                shape: Shape::matrix(seq_len, head_dim),
            };
            
            // Transpose K for QK^T
            let k_transposed = transpose_matrix(&k_data, total_seq_len, head_dim);
            let k_tensor = SimpleTensor {
                data: k_transposed,
                shape: Shape::matrix(head_dim, total_seq_len),
            };
            
            let scores = if is_blas_available() {
                matmul_blas(&q_tensor, &k_tensor)
                    .unwrap_or_else(|| compute_scores_fallback(&q_tensor, &k_tensor))
            } else {
                compute_scores_fallback(&q_tensor, &k_tensor)
            };
            
            // Apply scale and mask
            let mut attention_weights = scores.data;
            for i in 0..seq_len {
                for j in 0..total_seq_len {
                    let idx = i * total_seq_len + j;
                    attention_weights[idx] *= scale;
                    if j > (seq_len - 1 - i) + total_seq_len - seq_len {
                        attention_weights[idx] = f32::NEG_INFINITY;
                    }
                }
            }
            
            // Softmax
            apply_softmax_rows(&mut attention_weights, seq_len, total_seq_len);
            
            // Extract values
            let mut v_data = Vec::with_capacity(total_seq_len * head_dim);
            for pos in 0..total_seq_len {
                let offset = pos * num_kv_heads * head_dim + kv_start;
                v_data.extend_from_slice(&values[offset..offset + head_dim]);
            }
            
            let v_tensor = SimpleTensor {
                data: v_data,
                shape: Shape::matrix(total_seq_len, head_dim),
            };
            
            let attn_tensor = SimpleTensor {
                data: attention_weights,
                shape: Shape::matrix(seq_len, total_seq_len),
            };
            
            // Compute output with BLAS
            let head_output = if is_blas_available() {
                matmul_blas(&attn_tensor, &v_tensor)
                    .unwrap_or_else(|| compute_attention_fallback(&attn_tensor, &v_tensor))
            } else {
                compute_attention_fallback(&attn_tensor, &v_tensor)
            };
            
            // Copy to output
            for pos in 0..seq_len {
                let out_offset = pos * num_heads * head_dim + q_head * head_dim;
                let src_offset = pos * head_dim;
                output[out_offset..out_offset + head_dim]
                    .copy_from_slice(&head_output.data[src_offset..src_offset + head_dim]);
            }
        }
    }
    
    Ok(output)
}

/// Transpose a matrix stored in row-major order
fn transpose_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed[j * rows + i] = data[i * cols + j];
        }
    }
    transposed
}

/// Apply softmax to each row of a matrix
fn apply_softmax_rows(data: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row_start = i * cols;
        let row_end = row_start + cols;
        let row = &mut data[row_start..row_end];
        
        // Find max for numerical stability
        let max_val = row.iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        
        if max_val == f32::NEG_INFINITY {
            continue; // All masked
        }
        
        // Exp and sum
        let mut sum = 0.0;
        for val in row.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }
        
        // Normalize
        if sum > 0.0 {
            for val in row.iter_mut() {
                *val /= sum;
            }
        }
    }
}

/// Fallback matrix multiplication for scores
fn compute_scores_fallback(q: &SimpleTensor, k: &SimpleTensor) -> SimpleTensor {
    let (m, n) = (q.shape.dims()[0], k.shape.dims()[1]);
    let k_dim = q.shape.dims()[1];
    
    let mut result = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for idx in 0..k_dim {
                sum += q.data[i * k_dim + idx] * k.data[idx * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    
    SimpleTensor {
        data: result,
        shape: Shape::matrix(m, n),
    }
}

/// Fallback matrix multiplication for attention
fn compute_attention_fallback(attn: &SimpleTensor, v: &SimpleTensor) -> SimpleTensor {
    let (m, n) = (attn.shape.dims()[0], v.shape.dims()[1]);
    let k = attn.shape.dims()[1];
    
    let mut result = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for idx in 0..k {
                sum += attn.data[i * k + idx] * v.data[idx * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    
    SimpleTensor {
        data: result,
        shape: Shape::matrix(m, n),
    }
}