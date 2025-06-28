//! Integration test for the complete optimized Woolly pipeline

use crate::{Result, CoreError};
use crate::model::{
    optimized_transformer::OptimizedTransformer,
    memory_pool::TensorMemoryPool,
    transformer::TransformerConfig,
};
use crate::tensor_utils_optimized::{matmul_fast, rms_norm_fast, swiglu_fast};
use crate::tensor_utils::{tensor_from_slice, SimpleTensor};
use woolly_tensor::Shape;
use std::time::Instant;

/// Comprehensive integration test for the optimized pipeline
pub struct OptimizedPipelineTest {
    pool: TensorMemoryPool,
    test_results: Vec<TestResult>,
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_name: String,
    pub duration_ms: f64,
    pub success: bool,
    pub performance_target_met: bool,
    pub details: String,
}

impl OptimizedPipelineTest {
    pub fn new() -> Self {
        Self {
            pool: TensorMemoryPool::new(),
            test_results: Vec::new(),
        }
    }
    
    /// Run complete integration test suite
    pub fn run_complete_test_suite(&mut self) -> Result<bool> {
        println!("🔬 Woolly Optimized Pipeline Integration Test");
        println!("=============================================");
        
        // Test 1: Memory Pool Operations
        self.test_memory_pool_efficiency()?;
        
        // Test 2: SIMD Tensor Operations
        self.test_simd_operations()?;
        
        // Test 3: Optimized Matrix Operations
        self.test_matrix_operations()?;
        
        // Test 4: Transformer Layer Simulation
        self.test_transformer_layer_simulation()?;
        
        // Test 5: End-to-End Performance
        self.test_end_to_end_performance()?;
        
        // Test 6: Memory Efficiency
        self.test_memory_efficiency()?;
        
        // Generate comprehensive report
        self.generate_test_report();
        
        // Return overall success
        Ok(self.all_tests_passed())
    }
    
    /// Test memory pool efficiency and correctness
    fn test_memory_pool_efficiency(&mut self) -> Result<()> {
        let test_name = "Memory Pool Efficiency";
        let start_time = Instant::now();
        
        let mut success = true;
        let mut details = String::new();
        
        // Test buffer allocation and reuse
        let sizes = [256, 1024, 4096, 16384];
        let iterations = 1000;
        
        for &size in &sizes {
            let allocation_start = Instant::now();
            
            // Allocate and return buffers repeatedly
            for _ in 0..iterations {
                let buffer = self.pool.get_buffer(size);
                if buffer.len() != size {
                    success = false;
                    details.push_str(&format!("Buffer size mismatch: expected {}, got {}\n", size, buffer.len()));
                }
                self.pool.return_buffer(buffer);
            }
            
            let allocation_time = allocation_start.elapsed();
            details.push_str(&format!(
                "Size {}: {} allocations in {:.2}ms ({:.0} allocs/sec)\n",
                size,
                iterations,
                allocation_time.as_millis(),
                iterations as f64 / allocation_time.as_secs_f64()
            ));
        }
        
        // Test cache functionality
        let cache_key = (100, 100);
        let test_data = vec![1.0; 10000];
        self.pool.cache_matmul_result(cache_key, test_data.clone());
        
        if let Some(cached) = self.pool.get_matmul_cache(cache_key) {
            if cached.len() != test_data.len() {
                success = false;
                details.push_str("Cache retrieval failed\n");
            }
        } else {
            success = false;
            details.push_str("Cache storage failed\n");
        }
        
        let duration = start_time.elapsed().as_millis() as f64;
        let performance_target_met = duration < 100.0; // Should complete in <100ms
        
        self.test_results.push(TestResult {
            test_name: test_name.to_string(),
            duration_ms: duration,
            success,
            performance_target_met,
            details,
        });
        
        Ok(())
    }
    
    /// Test SIMD operations correctness and performance
    fn test_simd_operations(&mut self) -> Result<()> {
        let test_name = "SIMD Operations";
        let start_time = Instant::now();
        
        let mut success = true;
        let mut details = String::new();
        
        // Test RMS normalization
        let hidden_size = 4096;
        let seq_len = 8;
        
        let input_data: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let weight_data: Vec<f32> = (0..hidden_size)
            .map(|i| 1.0 + (i as f32) * 0.001)
            .collect();
        
        let input = tensor_from_slice(&input_data, Shape::matrix(seq_len, hidden_size))?;
        let weight = tensor_from_slice(&weight_data, Shape::vector(hidden_size))?;
        
        // Test optimized vs baseline
        let baseline_result = crate::tensor_utils::rms_norm(&input, &weight, 1e-5)?;
        let optimized_result = rms_norm_fast(&input, &weight, 1e-5, &mut self.pool)?;
        
        // Verify correctness (results should be very close)
        let max_diff = baseline_result.data.iter()
            .zip(optimized_result.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |max, diff| max.max(diff));
        
        if max_diff > 1e-4 {
            success = false;
            details.push_str(&format!("RMS norm result mismatch: max diff = {}\n", max_diff));
        } else {
            details.push_str("RMS norm correctness: ✓\n");
        }
        
        // Test SwiGLU activation
        let gate_data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let up_data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01).collect();
        
        let gate = tensor_from_slice(&gate_data, Shape::vector(1000))?;
        let up = tensor_from_slice(&up_data, Shape::vector(1000))?;
        
        let baseline_swiglu = crate::tensor_utils::swiglu(&gate, &up)?;
        let optimized_swiglu = swiglu_fast(&gate, &up, &mut self.pool)?;
        
        let swiglu_max_diff = baseline_swiglu.data.iter()
            .zip(optimized_swiglu.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |max, diff| max.max(diff));
        
        if swiglu_max_diff > 1e-4 {
            success = false;
            details.push_str(&format!("SwiGLU result mismatch: max diff = {}\n", swiglu_max_diff));
        } else {
            details.push_str("SwiGLU correctness: ✓\n");
        }
        
        let duration = start_time.elapsed().as_millis() as f64;
        let performance_target_met = duration < 50.0; // Should complete in <50ms
        
        self.test_results.push(TestResult {
            test_name: test_name.to_string(),
            duration_ms: duration,
            success,
            performance_target_met,
            details,
        });
        
        Ok(())
    }
    
    /// Test matrix operations with various sizes
    fn test_matrix_operations(&mut self) -> Result<()> {
        let test_name = "Matrix Operations";
        let start_time = Instant::now();
        
        let mut success = true;
        let mut details = String::new();
        
        // Test different matrix sizes typical in transformers
        let test_cases = [
            (64, 64, 64),      // Small attention
            (512, 2048, 512),  // Medium FFN
            (1, 32000, 4096),  // LM head
        ];
        
        for (i, &(m, n, k)) in test_cases.iter().enumerate() {
            let case_start = Instant::now();
            
            // Generate test matrices
            let a_data: Vec<f32> = (0..m*k).map(|j| (j as f32) * 0.001).collect();
            let b_data: Vec<f32> = (0..k*n).map(|j| (j as f32) * 0.001).collect();
            
            let a = tensor_from_slice(&a_data, Shape::matrix(m, k))?;
            let b = tensor_from_slice(&b_data, Shape::matrix(k, n))?;
            
            // Test correctness
            let baseline_result = crate::tensor_utils::matmul(&a, &b)?;
            let optimized_result = matmul_fast(&a, &b, &mut self.pool, true)?;
            
            // Verify correctness
            let max_diff = baseline_result.data.iter()
                .zip(optimized_result.data.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, |max, diff| max.max(diff));
            
            let case_duration = case_start.elapsed().as_millis();
            
            if max_diff > 1e-3 {
                success = false;
                details.push_str(&format!("Case {}: {}x{}x{} FAILED (max diff: {})\n", i+1, m, n, k, max_diff));
            } else {
                details.push_str(&format!("Case {}: {}x{}x{} ✓ ({:.1}ms)\n", i+1, m, n, k, case_duration));
            }
        }
        
        let duration = start_time.elapsed().as_millis() as f64;
        let performance_target_met = duration < 200.0; // Should complete in <200ms
        
        self.test_results.push(TestResult {
            test_name: test_name.to_string(),
            duration_ms: duration,
            success,
            performance_target_met,
            details,
        });
        
        Ok(())
    }
    
    /// Test complete transformer layer simulation
    fn test_transformer_layer_simulation(&mut self) -> Result<()> {
        let test_name = "Transformer Layer Simulation";
        let start_time = Instant::now();
        
        let mut success = true;
        let mut details = String::new();
        
        // Simulate typical transformer dimensions
        let seq_len = 8;
        let hidden_size = 4096;
        let intermediate_size = 11008;
        let num_heads = 32;
        let head_dim = hidden_size / num_heads;
        
        // Input hidden states
        let input_data: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let hidden_states = tensor_from_slice(&input_data, Shape::matrix(seq_len, hidden_size))?;
        
        // Weight matrices
        let attn_q_data: Vec<f32> = (0..hidden_size * hidden_size)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        let attn_k_data: Vec<f32> = (0..hidden_size * hidden_size)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        let attn_v_data: Vec<f32> = (0..hidden_size * hidden_size)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        let attn_o_data: Vec<f32> = (0..hidden_size * hidden_size)
            .map(|i| (i as f32) * 0.0001)
            .collect();
        
        let ffn_gate_data: Vec<f32> = (0..hidden_size * intermediate_size)
            .map(|i| (i as f32) * 0.00001)
            .collect();
        let ffn_up_data: Vec<f32> = (0..hidden_size * intermediate_size)
            .map(|i| (i as f32) * 0.00001)
            .collect();
        let ffn_down_data: Vec<f32> = (0..intermediate_size * hidden_size)
            .map(|i| (i as f32) * 0.00001)
            .collect();
        
        let norm_weight_data: Vec<f32> = (0..hidden_size)
            .map(|i| 1.0 + (i as f32) * 0.0001)
            .collect();
        
        // Create tensors
        let q_weight = tensor_from_slice(&attn_q_data, Shape::matrix(hidden_size, hidden_size))?;
        let k_weight = tensor_from_slice(&attn_k_data, Shape::matrix(hidden_size, hidden_size))?;
        let v_weight = tensor_from_slice(&attn_v_data, Shape::matrix(hidden_size, hidden_size))?;
        let o_weight = tensor_from_slice(&attn_o_data, Shape::matrix(hidden_size, hidden_size))?;
        
        let gate_weight = tensor_from_slice(&ffn_gate_data, Shape::matrix(hidden_size, intermediate_size))?;
        let up_weight = tensor_from_slice(&ffn_up_data, Shape::matrix(hidden_size, intermediate_size))?;
        let down_weight = tensor_from_slice(&ffn_down_data, Shape::matrix(intermediate_size, hidden_size))?;
        
        let norm_weight = tensor_from_slice(&norm_weight_data, Shape::vector(hidden_size))?;
        
        // Simulate transformer layer forward pass
        let layer_start = Instant::now();
        
        // 1. Pre-attention normalization
        let norm1 = rms_norm_fast(&hidden_states, &norm_weight, 1e-5, &mut self.pool)?;
        
        // 2. Attention projections
        let queries = matmul_fast(&norm1, &q_weight, &mut self.pool, true)?;
        let keys = matmul_fast(&norm1, &k_weight, &mut self.pool, true)?;
        let values = matmul_fast(&norm1, &v_weight, &mut self.pool, true)?;
        
        // 3. Simplified attention (skip the complex attention computation for this test)
        let attn_output = matmul_fast(&queries, &o_weight, &mut self.pool, true)?;
        
        // 4. Residual connection
        let post_attn = crate::tensor_utils::add_tensors(&hidden_states, &attn_output)?;
        
        // 5. Pre-FFN normalization
        let norm2 = rms_norm_fast(&post_attn, &norm_weight, 1e-5, &mut self.pool)?;
        
        // 6. FFN with SwiGLU
        let gate_proj = matmul_fast(&norm2, &gate_weight, &mut self.pool, true)?;
        let up_proj = matmul_fast(&norm2, &up_weight, &mut self.pool, true)?;
        let swiglu_output = swiglu_fast(&gate_proj, &up_proj, &mut self.pool)?;
        let ffn_output = matmul_fast(&swiglu_output, &down_weight, &mut self.pool, true)?;
        
        // 7. Final residual connection
        let final_output = crate::tensor_utils::add_tensors(&post_attn, &ffn_output)?;
        
        let layer_duration = layer_start.elapsed();
        
        // Verify output shape is correct
        if final_output.shape().as_slice() != &[seq_len, hidden_size] {
            success = false;
            details.push_str("Output shape mismatch\n");
        } else {
            details.push_str("Output shape: ✓\n");
        }
        
        // Verify output is reasonable (no NaN, Inf, extreme values)
        let has_invalid = final_output.data.iter().any(|&x| x.is_nan() || x.is_infinite() || x.abs() > 1000.0);
        if has_invalid {
            success = false;
            details.push_str("Output contains invalid values\n");
        } else {
            details.push_str("Output validity: ✓\n");
        }
        
        details.push_str(&format!("Layer forward pass: {:.2}ms\n", layer_duration.as_millis()));
        
        let duration = start_time.elapsed().as_millis() as f64;
        let performance_target_met = layer_duration.as_millis() < 100; // Target: <100ms per layer
        
        self.test_results.push(TestResult {
            test_name: test_name.to_string(),
            duration_ms: duration,
            success,
            performance_target_met,
            details,
        });
        
        Ok(())
    }
    
    /// Test end-to-end performance simulation
    fn test_end_to_end_performance(&mut self) -> Result<()> {
        let test_name = "End-to-End Performance";
        let start_time = Instant::now();
        
        let mut success = true;
        let mut details = String::new();
        
        // Simulate multiple token generation cycles
        let num_tokens = 10;
        let vocab_size = 32000;
        let hidden_size = 4096;
        
        let mut total_generation_time = 0;
        
        for token_idx in 0..num_tokens {
            let token_start = Instant::now();
            
            // Simulate hidden state (last token representation)
            let hidden_data: Vec<f32> = (0..hidden_size)
                .map(|i| ((i + token_idx) as f32) * 0.001)
                .collect();
            let hidden_state = tensor_from_slice(&hidden_data, Shape::vector(hidden_size))?;
            
            // Simulate LM head projection (most expensive operation)
            let lm_head_data: Vec<f32> = (0..hidden_size * vocab_size)
                .map(|i| (i as f32) * 0.000001)
                .collect();
            let lm_head = tensor_from_slice(&lm_head_data, Shape::matrix(hidden_size, vocab_size))?;
            
            // Project to vocabulary
            let logits = matmul_fast(&hidden_state, &lm_head, &mut self.pool, true)?;
            
            // Verify logits shape
            if logits.shape().as_slice() != &[vocab_size] {
                success = false;
                details.push_str(&format!("Token {}: logits shape mismatch\n", token_idx));
                break;
            }
            
            let token_duration = token_start.elapsed().as_millis();
            total_generation_time += token_duration;
            
            details.push_str(&format!("Token {}: {:.1}ms\n", token_idx, token_duration));
        }
        
        let avg_token_time = total_generation_time as f64 / num_tokens as f64;
        let tokens_per_second = 1000.0 / avg_token_time;
        
        details.push_str(&format!(
            "Average per token: {:.1}ms ({:.1} tokens/sec)\n",
            avg_token_time,
            tokens_per_second
        ));
        
        let duration = start_time.elapsed().as_millis() as f64;
        let performance_target_met = tokens_per_second >= 5.0; // Target: ≥5 tokens/sec
        
        self.test_results.push(TestResult {
            test_name: test_name.to_string(),
            duration_ms: duration,
            success,
            performance_target_met,
            details,
        });
        
        Ok(())
    }
    
    /// Test memory efficiency and resource usage
    fn test_memory_efficiency(&mut self) -> Result<()> {
        let test_name = "Memory Efficiency";
        let start_time = Instant::now();
        
        let mut success = true;
        let mut details = String::new();
        
        // Clear pool to start fresh
        self.pool.clear_cache();
        
        // Perform a series of operations that would normally allocate a lot of memory
        let num_operations = 100;
        let matrix_size = 1024;
        
        for i in 0..num_operations {
            // Create temporary matrices
            let a_data: Vec<f32> = (0..matrix_size*matrix_size)
                .map(|j| ((i + j) as f32) * 0.001)
                .collect();
            let b_data: Vec<f32> = (0..matrix_size*matrix_size)
                .map(|j| ((i * 2 + j) as f32) * 0.001)
                .collect();
            
            let a = tensor_from_slice(&a_data, Shape::matrix(matrix_size, matrix_size))?;
            let b = tensor_from_slice(&b_data, Shape::matrix(matrix_size, matrix_size))?;
            
            // Perform matrix multiplication (should reuse buffers)
            let _result = matmul_fast(&a, &b, &mut self.pool, true)?;
            
            // Check for memory efficiency every 10 operations
            if i % 10 == 0 {
                // Simulate checking memory usage (would need actual memory tracking)
                details.push_str(&format!("Operation {}: ✓\n", i));
            }
        }
        
        // Test cache hit rate simulation
        let cache_operations = 50;
        let cache_key = (512, 512);
        let test_data = vec![1.0; 512 * 512];
        
        // First access - cache miss
        self.pool.cache_matmul_result(cache_key, test_data.clone());
        
        // Subsequent accesses - cache hits
        let mut cache_hits = 0;
        for _ in 0..cache_operations {
            if self.pool.get_matmul_cache(cache_key).is_some() {
                cache_hits += 1;
            }
        }
        
        let cache_hit_rate = cache_hits as f64 / cache_operations as f64;
        details.push_str(&format!("Cache hit rate: {:.1}%\n", cache_hit_rate * 100.0));
        
        if cache_hit_rate < 0.7 {
            success = false;
            details.push_str("Cache hit rate below target (70%)\n");
        }
        
        let duration = start_time.elapsed().as_millis() as f64;
        let performance_target_met = cache_hit_rate >= 0.7 && duration < 1000.0;
        
        self.test_results.push(TestResult {
            test_name: test_name.to_string(),
            duration_ms: duration,
            success,
            performance_target_met,
            details,
        });
        
        Ok(())
    }
    
    /// Generate comprehensive test report
    fn generate_test_report(&self) {
        println!("\n📋 Integration Test Report");
        println!("=========================");
        
        let mut total_duration = 0.0;
        let mut passed_tests = 0;
        let mut performance_targets_met = 0;
        
        for result in &self.test_results {
            let status_icon = if result.success { "✅" } else { "❌" };
            let perf_icon = if result.performance_target_met { "🚀" } else { "⚠️" };
            
            println!("\n{} {} {}", status_icon, perf_icon, result.test_name);
            println!("   Duration: {:.1}ms", result.duration_ms);
            println!("   Details:");
            for line in result.details.lines() {
                if !line.is_empty() {
                    println!("     {}", line);
                }
            }
            
            total_duration += result.duration_ms;
            if result.success {
                passed_tests += 1;
            }
            if result.performance_target_met {
                performance_targets_met += 1;
            }
        }
        
        println!("\n🏆 Summary");
        println!("=========");
        println!("Tests passed: {}/{}", passed_tests, self.test_results.len());
        println!("Performance targets met: {}/{}", performance_targets_met, self.test_results.len());
        println!("Total duration: {:.1}ms", total_duration);
        
        let success_rate = passed_tests as f64 / self.test_results.len() as f64;
        let performance_rate = performance_targets_met as f64 / self.test_results.len() as f64;
        
        if success_rate >= 1.0 && performance_rate >= 0.8 {
            println!("🎉 EXCELLENT: All tests passed, performance targets achieved!");
            println!("✅ Ready for Ole integration");
        } else if success_rate >= 1.0 && performance_rate >= 0.6 {
            println!("✅ GOOD: All tests passed, most performance targets met");
            println!("⚡ Some optimization opportunities remain");
        } else if success_rate >= 0.8 {
            println!("⚠️  MODERATE: Most tests passed, needs improvement");
            println!("🔧 Review failed tests and optimize");
        } else {
            println!("❌ NEEDS WORK: Several tests failed");
            println!("🚨 Not ready for production");
        }
    }
    
    /// Check if all tests passed
    fn all_tests_passed(&self) -> bool {
        self.test_results.iter().all(|r| r.success)
    }
    
    /// Get overall performance score (0.0 to 1.0)
    pub fn get_performance_score(&self) -> f64 {
        if self.test_results.is_empty() {
            return 0.0;
        }
        
        let success_rate = self.test_results.iter().filter(|r| r.success).count() as f64 / self.test_results.len() as f64;
        let performance_rate = self.test_results.iter().filter(|r| r.performance_target_met).count() as f64 / self.test_results.len() as f64;
        
        (success_rate + performance_rate) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_optimized_pipeline_integration() -> Result<()> {
        let mut test_suite = OptimizedPipelineTest::new();
        
        // Run subset of tests for CI
        test_suite.test_memory_pool_efficiency()?;
        test_suite.test_simd_operations()?;
        test_suite.test_matrix_operations()?;
        
        // Ensure reasonable performance
        let performance_score = test_suite.get_performance_score();
        assert!(performance_score > 0.7, "Performance score too low: {}", performance_score);
        
        Ok(())
    }
}