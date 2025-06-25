//! Standard input/output transport for MCP

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use std::sync::Arc;

use crate::transport::{Transport, TransportError, TransportConfig};
use crate::types::McpMessage;

/// Stdio transport for MCP communication
pub struct StdioTransport {
    stdin: Arc<Mutex<BufReader<tokio::io::Stdin>>>,
    stdout: Arc<Mutex<tokio::io::Stdout>>,
    config: TransportConfig,
    connected: Arc<Mutex<bool>>,
}

impl StdioTransport {
    /// Create a new stdio transport
    pub fn new(config: TransportConfig) -> Self {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        
        Self {
            stdin: Arc::new(Mutex::new(BufReader::new(stdin))),
            stdout: Arc::new(Mutex::new(stdout)),
            config,
            connected: Arc::new(Mutex::new(true)),
        }
    }
    
    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(TransportConfig::default())
    }
}

#[async_trait]
impl Transport for StdioTransport {
    async fn send(&mut self, message: McpMessage) -> Result<(), TransportError> {
        let json = serde_json::to_string(&message)?;
        
        // Check message size
        if json.len() > self.config.max_message_size {
            return Err(TransportError::Transport(format!(
                "Message size {} exceeds maximum {}",
                json.len(),
                self.config.max_message_size
            )));
        }
        
        let mut stdout = self.stdout.lock().await;
        
        // Apply write timeout if configured
        if let Some(timeout_ms) = self.config.write_timeout_ms {
            let timeout = std::time::Duration::from_millis(timeout_ms);
            tokio::time::timeout(timeout, async {
                stdout.write_all(json.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
                Ok::<(), std::io::Error>(())
            })
            .await
            .map_err(|_| TransportError::Transport("Write timeout".to_string()))??;
        } else {
            stdout.write_all(json.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
        
        Ok(())
    }
    
    async fn receive(&mut self) -> Result<McpMessage, TransportError> {
        let mut line = String::new();
        let mut stdin = self.stdin.lock().await;
        
        // Apply read timeout if configured
        let bytes_read = if let Some(timeout_ms) = self.config.read_timeout_ms {
            let timeout = std::time::Duration::from_millis(timeout_ms);
            tokio::time::timeout(timeout, stdin.read_line(&mut line))
                .await
                .map_err(|_| TransportError::Transport("Read timeout".to_string()))??
        } else {
            stdin.read_line(&mut line).await?
        };
        
        if bytes_read == 0 {
            *self.connected.lock().await = false;
            return Err(TransportError::ConnectionClosed);
        }
        
        // Remove trailing newline
        if line.ends_with('\n') {
            line.pop();
            if line.ends_with('\r') {
                line.pop();
            }
        }
        
        // Check message size
        if line.len() > self.config.max_message_size {
            return Err(TransportError::Transport(format!(
                "Message size {} exceeds maximum {}",
                line.len(),
                self.config.max_message_size
            )));
        }
        
        let message: McpMessage = serde_json::from_str(&line)?;
        Ok(message)
    }
    
    fn is_connected(&self) -> bool {
        // Block on the async lock - this is safe in our use case
        futures::executor::block_on(async {
            *self.connected.lock().await
        })
    }
    
    async fn close(&mut self) -> Result<(), TransportError> {
        *self.connected.lock().await = false;
        // For stdio, we don't actually close stdin/stdout
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_stdio_transport_creation() {
        let transport = StdioTransport::default();
        assert!(transport.is_connected());
    }
    
    #[tokio::test]
    async fn test_stdio_transport_with_config() {
        let mut config = TransportConfig::default();
        config.read_timeout_ms = Some(5000);
        config.write_timeout_ms = Some(5000);
        
        let transport = StdioTransport::new(config);
        assert!(transport.is_connected());
    }
}