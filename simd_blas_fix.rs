// Clean replacement for compute_simd_gqa_attention function
// This version uses BLAS instead of manual loops

fn compute_simd_gqa_attention(
    &self,
    hidden_states: &[f32],
    layer_idx: usize,
    seq_len: usize,
    hidden_size: usize,
    weights: &mut LazyModelWeights,
    layer_prefix: &str,
) -> Result<Vec<f32>> {
    let num_heads = self.config.model_config.num_heads;
    let head_dim = hidden_size / num_heads;
    
    // Get memory pool
    let mut pool = self.memory_pool.lock().map_err(|_| {
        CoreError::model(
            "MEMORY_POOL_MUTEX_POISONED",
            "Memory pool mutex was poisoned due to a previous panic",
            "lazy transformer memory pool access",
            "This is likely due to a previous error in tensor operations"
        )
    })?;
    
    // Create tensors for projections
    let hidden_tensor = tensor_from_slice(hidden_states, Shape::matrix(seq_len, hidden_size))?;
    
    let q_tensor = {
        let q_weight_ref = weights.get_tensor(&format!("{}.attn_q.weight", layer_prefix))?;
        tensor_from_slice(q_weight_ref, Shape::matrix(hidden_size, hidden_size))?
    };
    
    let k_tensor = {
        let k_shape = weights.get_tensor_shape(&format!("{}.attn_k.weight", layer_prefix))?;
        let k_weight_ref = weights.get_tensor(&format!("{}.attn_k.weight", layer_prefix))?;
        tensor_from_slice(k_weight_ref, Shape::from_slice(&k_shape))?
    };
    
    let v_tensor = {
        let v_shape = weights.get_tensor_shape(&format!("{}.attn_v.weight", layer_prefix))?;
        let v_weight_ref = weights.get_tensor(&format!("{}.attn_v.weight", layer_prefix))?;
        tensor_from_slice(v_weight_ref, Shape::from_slice(&v_shape))?
    };
    
    let o_tensor = {
        let o_weight_ref = weights.get_tensor(&format!("{}.attn_output.weight", layer_prefix))?;
        tensor_from_slice(o_weight_ref, Shape::matrix(hidden_size, hidden_size))?
    };
    
    // Compute Q, K, V projections using SIMD
    let num_kv_heads = k_tensor.shape.dims()[1] / head_dim;
    let kv_hidden_size = num_kv_heads * head_dim;
    
    let mut pool_ref = pool;
    let (queries, keys, values) = simd_attention_projections(
        &hidden_tensor,
        &q_tensor,
        &k_tensor,
        &v_tensor,
        &mut pool_ref,
    )?;
    drop(pool_ref);
    
    // Use BLAS for attention computation instead of manual loops
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    eprintln!("  🚀 SIMD path: Using BLAS-optimized GQA attention");
    let output = grouped_query_attention_blas(
        &queries.data,
        &keys.data,
        &values.data,
        seq_len,
        seq_len,  // total_seq_len = seq_len for now (no past KV)
        num_heads,
        num_kv_heads,
        head_dim,
        scale
    )?;
    
    // Apply output projection using SIMD
    let output_tensor = tensor_from_slice(&output, Shape::matrix(seq_len, hidden_size))?;
    let projected = simd_matvec(&output_tensor, &o_tensor, false, 1.0, 0.0)?;
    
    Ok(projected.to_vec())
}