//! Inference session management with memory budgets and automatic cleanup

use crate::{
    kv_cache::{OptimizedKVCache, KVCacheConfig},
    model::Model,
    CoreError, Result,
};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};
use uuid::Uuid;

/// Configuration for an inference session
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Whether to use KV caching
    pub use_cache: bool,
    /// Maximum sequence length for this session
    pub max_seq_length: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p value for nucleus sampling
    pub top_p: f32,
    /// Top-k value for sampling
    pub top_k: usize,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Whether to return hidden states
    pub output_hidden_states: bool,
    /// Whether to return attention weights
    pub output_attentions: bool,
    /// Memory budget for this session (bytes)
    pub memory_budget: u64,
    /// Maximum token history to keep
    pub max_token_history: usize,
    /// Auto-cleanup interval
    pub cleanup_interval: Duration,
    /// Enable memory pressure monitoring
    pub monitor_memory: bool,
    /// KV cache configuration
    pub kv_cache_config: Option<KVCacheConfig>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1,
            use_cache: true,
            max_seq_length: 2048,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 50,
            repetition_penalty: 1.0,
            output_hidden_states: false,
            output_attentions: false,
            memory_budget: 256 * 1024 * 1024, // 256MB per session
            max_token_history: 4096,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            monitor_memory: true,
            kv_cache_config: None, // Use default
        }
    }
}

/// Memory usage statistics for a session
#[derive(Debug, Clone, Default)]
pub struct SessionMemoryStats {
    /// Current memory usage (bytes)
    pub current_usage: u64,
    /// Peak memory usage (bytes)
    pub peak_usage: u64,
    /// KV cache memory usage (bytes)
    pub kv_cache_usage: u64,
    /// Token history memory usage (bytes)
    pub token_history_usage: u64,
    /// Number of memory pressure events
    pub pressure_events: u64,
    /// Number of cleanup operations
    pub cleanup_operations: u64,
    /// Memory budget utilization (0.0 to 1.0)
    pub budget_utilization: f64,
}

/// An inference session that maintains state across multiple forward passes
pub struct InferenceSession {
    /// Unique session ID
    id: String,
    /// Reference to the model
    model: Arc<dyn Model>,
    /// Session configuration
    config: SessionConfig,
    /// Optimized KV cache for this session
    kv_cache: Option<Arc<OptimizedKVCache>>,
    /// Current sequence of token IDs with size limit
    token_history: RwLock<Vec<u32>>,
    /// Number of tokens processed
    tokens_processed: AtomicUsize,
    /// Whether the session is active
    active: AtomicBool,
    /// Current memory usage tracking
    current_memory: AtomicU64,
    /// Peak memory usage
    peak_memory: AtomicU64,
    /// Last cleanup time
    last_cleanup: RwLock<Instant>,
    /// Memory statistics
    memory_stats: RwLock<SessionMemoryStats>,
    /// Session creation time
    created_at: Instant,
    /// Last access time
    last_accessed: RwLock<Instant>,
    /// MCP hooks for session events
    #[cfg(feature = "mcp")]
    mcp_registry: Option<woolly_mcp::PluginRegistry>,
}

impl InferenceSession {
    /// Create a new inference session with memory management
    pub fn new(
        model: Arc<dyn Model>,
        config: SessionConfig,
        #[cfg(feature = "mcp")] mcp_registry: Option<woolly_mcp::PluginRegistry>,
        #[cfg(not(feature = "mcp"))] _mcp_registry: Option<()>,
    ) -> Result<Self> {
        let id = Uuid::new_v4().to_string();
        let now = Instant::now();
        
        // Initialize KV cache if enabled
        let kv_cache = if config.use_cache {
            let kv_config = config.kv_cache_config.clone().unwrap_or_else(|| {
                let mut default_config = KVCacheConfig::default();
                default_config.max_memory = config.memory_budget / 2; // Half budget for KV cache
                default_config.max_seq_length = config.max_seq_length;
                default_config
            });
            
            Some(Arc::new(OptimizedKVCache::new(kv_config)))
        } else {
            None
        };

        // Trigger session creation hook if MCP is available
        #[cfg(feature = "mcp")]
        if let Some(_registry) = &mcp_registry {
            // TODO: Implement trigger_hook method
            // registry.trigger_hook("session.created", session_data);
        }

        info!("Created session '{}' with memory budget: {} MB, max seq length: {}",
              id, config.memory_budget / (1024 * 1024), config.max_seq_length);

        Ok(Self {
            id,
            model,
            config,
            kv_cache,
            token_history: RwLock::new(Vec::with_capacity(1024)),
            tokens_processed: AtomicUsize::new(0),
            active: AtomicBool::new(true),
            current_memory: AtomicU64::new(0),
            peak_memory: AtomicU64::new(0),
            last_cleanup: RwLock::new(now),
            memory_stats: RwLock::new(SessionMemoryStats::default()),
            created_at: now,
            last_accessed: RwLock::new(now),
            #[cfg(feature = "mcp")]
            mcp_registry,
        })
    }

    /// Get session ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Check if session is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Get number of tokens processed
    pub fn tokens_processed(&self) -> usize {
        self.tokens_processed.load(Ordering::Relaxed)
    }

    /// Run inference on input tokens with memory management
    pub async fn infer(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        if !self.is_active() {
            return Err(CoreError::Generation {
                code: "SESSION_INACTIVE",
                message: "Session is not active".to_string(),
                context: "Inference request".to_string(),
                suggestion: "Create a new session or reactivate this one".to_string(),
                session_id: Some(self.id.clone()),
            });
        }

        // Update last accessed time
        *self.last_accessed.write().await = Instant::now();

        // Validate input
        if tokens.is_empty() {
            return Err(CoreError::Generation {
                code: "EMPTY_INPUT",
                message: "Empty input tokens".to_string(),
                context: "Token validation".to_string(),
                suggestion: "Provide non-empty token sequence".to_string(),
                session_id: Some(self.id.clone()),
            });
        }

        // Check memory pressure before processing
        self.check_memory_pressure().await?;

        // Check sequence length and apply management if needed
        self.manage_token_history(tokens.len()).await?;

        // Run forward pass through the model
        let past_kv_cache = self.get_kv_cache_for_inference().await;

        // Execute the forward pass
        let model_output = self.model.forward(tokens, past_kv_cache.as_ref()).await?;

        // Update KV cache if using caching
        if let Some(ref kv_cache) = self.kv_cache {
            if let Some(ref new_kv_data) = model_output.past_kv_cache {
                // Extract KV data and store in cache
                // This is a simplified interface - in reality we'd need proper type handling
                self.update_kv_cache(new_kv_data, tokens).await?;
            }
        }

        // Update token history with size limit
        self.append_to_token_history(tokens).await?;
        
        // Update tokens processed
        self.tokens_processed
            .fetch_add(tokens.len(), Ordering::Relaxed);

        // Update memory usage tracking
        self.update_memory_usage().await;

        // Trigger cleanup if needed
        if self.should_cleanup().await {
            self.cleanup().await?;
        }

        // Trigger inference hook if MCP is available
        #[cfg(feature = "mcp")]
        if let Some(_registry) = &self.mcp_registry {
            // TODO: Implement trigger_hook method
            // registry.trigger_hook("session.inference", inference_data);
        }

        trace!("Inference completed for session '{}': {} tokens processed", 
               self.id, tokens.len());

        // Return the actual logits from the model
        Ok(model_output.logits)
    }

    /// Process a batch of sequences
    pub async fn infer_batch(&self, batch: Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>> {
        if batch.len() > self.config.max_batch_size {
            return Err(CoreError::Generation(format!(
                "Batch size {} exceeds maximum {}",
                batch.len(),
                self.config.max_batch_size
            )));
        }

        // Process each sequence in the batch
        let mut results = Vec::with_capacity(batch.len());
        for tokens in batch {
            results.push(self.infer(&tokens).await?);
        }

        Ok(results)
    }

    /// Clear session state and reset memory usage
    pub async fn clear(&self) {
        info!("Clearing session '{}'", self.id);
        
        // Clear token history
        self.token_history.write().await.clear();
        
        // Clear KV cache
        if let Some(ref kv_cache) = self.kv_cache {
            kv_cache.clear();
        }
        
        // Reset counters
        self.tokens_processed.store(0, Ordering::Relaxed);
        self.current_memory.store(0, Ordering::Relaxed);
        
        // Reset memory stats
        {
            let mut stats = self.memory_stats.write().await;
            *stats = SessionMemoryStats::default();
        }
        
        // Update cleanup time
        *self.last_cleanup.write().await = Instant::now();
        
        trace!("Session '{}' cleared successfully", self.id);
    }

    /// Deactivate the session
    pub fn deactivate(&self) {
        self.active.store(false, Ordering::Relaxed);
        
        // Trigger session deactivation hook
        #[cfg(feature = "mcp")]
        if let Some(_registry) = &self.mcp_registry {
            // TODO: Implement trigger_hook method
            // registry.trigger_hook("session.deactivated", deactivation_data);
        }
    }

    /// Get the current token history
    pub async fn token_history(&self) -> Vec<u32> {
        self.token_history.read().await.clone()
    }

    /// Get session configuration
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }
    
    /// Check for memory pressure and take action if needed
    async fn check_memory_pressure(&self) -> Result<()> {
        let current = self.current_memory.load(Ordering::Relaxed);
        let budget = self.config.memory_budget;
        let pressure_threshold = (budget as f64 * 0.8) as u64; // 80% of budget
        
        if current > pressure_threshold {
            warn!("Memory pressure detected in session '{}': {} MB / {} MB ({}%)",
                  self.id, current / (1024 * 1024), budget / (1024 * 1024),
                  (current * 100) / budget);
            
            // Update stats
            {
                let mut stats = self.memory_stats.write().await;
                stats.pressure_events += 1;
            }
            
            // Trigger cleanup
            self.cleanup().await?;
            
            // Check if still over budget after cleanup
            let current_after = self.current_memory.load(Ordering::Relaxed);
            if current_after > budget {
                return Err(CoreError::Resource {
                    code: "MEMORY_BUDGET_EXCEEDED",
                    message: format!("Session memory budget exceeded: {} bytes", current_after),
                    context: format!("Session '{}' memory management", self.id),
                    suggestion: "Reduce session size or increase memory budget".to_string(),
                    resource_type: "memory".to_string(),
                    required: Some(current_after),
                    available: Some(budget),
                });
            }
        }
        
        Ok(())
    }
    
    /// Manage token history size to stay within limits
    async fn manage_token_history(&self, new_tokens: usize) -> Result<()> {
        let mut history = self.token_history.write().await;
        let current_len = history.len();
        let total_after = current_len + new_tokens;
        
        // Check if we need to truncate history
        if total_after > self.config.max_token_history {
            let target_size = self.config.max_token_history / 2; // Keep half
            let tokens_to_remove = current_len.saturating_sub(target_size);
            
            if tokens_to_remove > 0 {
                history.drain(0..tokens_to_remove);
                debug!("Truncated token history for session '{}': removed {} tokens",
                       self.id, tokens_to_remove);
            }
        }
        
        // Check sequence length against model limits
        if total_after > self.config.max_seq_length {
            let target_size = self.config.max_seq_length * 3 / 4; // Keep 75%
            let tokens_to_remove = current_len.saturating_sub(target_size);
            
            if tokens_to_remove > 0 {
                history.drain(0..tokens_to_remove);
                debug!("Truncated for sequence length limit in session '{}': removed {} tokens",
                       self.id, tokens_to_remove);
            }
        }
        
        Ok(())
    }
    
    /// Append tokens to history with size management
    async fn append_to_token_history(&self, tokens: &[u32]) -> Result<()> {
        let mut history = self.token_history.write().await;
        history.extend_from_slice(tokens);
        
        // Ensure we don't exceed maximum history size
        if history.len() > self.config.max_token_history {
            let excess = history.len() - self.config.max_token_history;
            history.drain(0..excess);
            trace!("Removed {} excess tokens from history", excess);
        }
        
        Ok(())
    }
    
    /// Get KV cache data for inference
    async fn get_kv_cache_for_inference(&self) -> Option<Box<dyn std::any::Any + Send + Sync>> {
        // This is a placeholder - in reality we'd extract the proper KV data
        // from our OptimizedKVCache and convert it to the format expected by the model
        None
    }
    
    /// Update KV cache with new data
    async fn update_kv_cache(&self, new_kv_data: &dyn std::any::Any, tokens: &[u32]) -> Result<()> {
        if let Some(ref kv_cache) = self.kv_cache {
            // This is a placeholder - in reality we'd need to:
            // 1. Extract the KV tensors from new_kv_data
            // 2. Convert them to the format expected by OptimizedKVCache
            // 3. Store them in the cache
            
            // For now, just log the operation
            trace!("Updated KV cache for session '{}' with {} tokens", self.id, tokens.len());
        }
        Ok(())
    }
    
    /// Update memory usage statistics
    async fn update_memory_usage(&self) {
        let history_usage = {
            let history = self.token_history.read().await;
            (history.len() * std::mem::size_of::<u32>()) as u64
        };
        
        let kv_cache_usage = self.kv_cache.as_ref()
            .map(|cache| cache.memory_usage())
            .unwrap_or(0);
        
        let total_usage = history_usage + kv_cache_usage;
        
        // Update atomic counters
        self.current_memory.store(total_usage, Ordering::Relaxed);
        
        let peak = self.peak_memory.load(Ordering::Relaxed);
        if total_usage > peak {
            self.peak_memory.store(total_usage, Ordering::Relaxed);
        }
        
        // Update detailed stats
        {
            let mut stats = self.memory_stats.write().await;
            stats.current_usage = total_usage;
            stats.peak_usage = self.peak_memory.load(Ordering::Relaxed);
            stats.kv_cache_usage = kv_cache_usage;
            stats.token_history_usage = history_usage;
            stats.budget_utilization = total_usage as f64 / self.config.memory_budget as f64;
        }
    }
    
    /// Check if cleanup should be performed
    async fn should_cleanup(&self) -> bool {
        let last_cleanup = *self.last_cleanup.read().await;
        let elapsed = last_cleanup.elapsed();
        elapsed > self.config.cleanup_interval
    }
    
    /// Perform session cleanup
    pub async fn cleanup(&self) -> Result<()> {
        debug!("Running cleanup for session '{}'", self.id);
        
        let cleanup_start = Instant::now();
        
        // Clean up KV cache
        if let Some(ref kv_cache) = self.kv_cache {
            kv_cache.cleanup_expired();
        }
        
        // Trim token history if it's grown too large
        {
            let mut history = self.token_history.write().await;
            let target_size = self.config.max_token_history * 3 / 4; // Trim to 75%
            
            if history.len() > target_size {
                let excess = history.len() - target_size;
                history.drain(0..excess);
                debug!("Trimmed {} tokens from history during cleanup", excess);
            }
        }
        
        // Update cleanup time
        *self.last_cleanup.write().await = Instant::now();
        
        // Update memory usage after cleanup
        self.update_memory_usage().await;
        
        // Update stats
        {
            let mut stats = self.memory_stats.write().await;
            stats.cleanup_operations += 1;
        }
        
        let cleanup_duration = cleanup_start.elapsed();
        trace!("Cleanup completed for session '{}' in {:.2}ms", 
               self.id, cleanup_duration.as_secs_f64() * 1000.0);
        
        Ok(())
    }
    
    /// Get memory usage statistics
    pub async fn memory_stats(&self) -> SessionMemoryStats {
        self.memory_stats.read().await.clone()
    }
    
    /// Get current memory usage in bytes
    pub fn current_memory_usage(&self) -> u64 {
        self.current_memory.load(Ordering::Relaxed)
    }
    
    /// Get peak memory usage in bytes
    pub fn peak_memory_usage(&self) -> u64 {
        self.peak_memory.load(Ordering::Relaxed)
    }
    
    /// Get memory budget utilization as a percentage (0.0 to 1.0)
    pub fn memory_utilization(&self) -> f64 {
        let current = self.current_memory.load(Ordering::Relaxed) as f64;
        let budget = self.config.memory_budget as f64;
        if budget > 0.0 {
            current / budget
        } else {
            0.0
        }
    }
    
    /// Force memory optimization
    pub async fn optimize_memory(&self) -> Result<()> {
        info!("Optimizing memory for session '{}'", self.id);
        
        // Aggressive token history trimming
        {
            let mut history = self.token_history.write().await;
            let target_size = self.config.max_token_history / 2; // Trim to 50%
            
            if history.len() > target_size {
                let excess = history.len() - target_size;
                history.drain(0..excess);
                debug!("Aggressively trimmed {} tokens from history", excess);
            }
        }
        
        // Optimize KV cache
        if let Some(ref kv_cache) = self.kv_cache {
            kv_cache.optimize_cache()?;
        }
        
        // Update memory usage
        self.update_memory_usage().await;
        
        let current = self.current_memory.load(Ordering::Relaxed);
        info!("Memory optimization complete for session '{}': {} MB usage", 
              self.id, current / (1024 * 1024));
        
        Ok(())
    }
    
    /// Get session age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
    
    /// Get time since last access
    pub async fn idle_time(&self) -> Duration {
        self.last_accessed.read().await.elapsed()
    }
    
    /// Check if session should be considered for eviction
    pub async fn is_eligible_for_eviction(&self, max_idle_time: Duration) -> bool {
        !self.is_active() || self.idle_time().await > max_idle_time
    }

}

/// Statistics for a session with enhanced memory tracking
#[derive(Debug, Clone)]
pub struct SessionStats {
    pub id: String,
    pub tokens_processed: usize,
    pub memory_usage_bytes: u64,
    pub peak_memory_bytes: u64,
    pub cache_usage_bytes: u64,
    pub memory_utilization: f64,
    pub active: bool,
    pub created_at: std::time::Instant,
    pub last_accessed: std::time::Instant,
    pub age: Duration,
    pub idle_time: Duration,
    pub cleanup_operations: u64,
    pub pressure_events: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert_eq!(config.max_batch_size, 1);
        assert!(config.use_cache);
        assert_eq!(config.max_seq_length, 2048);
        assert_eq!(config.temperature, 1.0);
    }
}