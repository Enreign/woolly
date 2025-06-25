//! Transport abstraction layer for MCP

use async_trait::async_trait;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncWrite};

pub mod http;
pub mod stdio;
pub mod websocket;

use crate::types::{McpMessage, McpError};

/// Transport error type
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Connection closed")]
    ConnectionClosed,
    
    #[error("Transport error: {0}")]
    Transport(String),
    
    #[error("Protocol error: {0}")]
    Protocol(String),
    
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<TransportError> for McpError {
    fn from(err: TransportError) -> Self {
        McpError {
            code: crate::types::error_codes::INTERNAL_ERROR,
            message: err.to_string(),
            data: None,
        }
    }
}

/// Core transport trait for MCP communication
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send a message through the transport
    async fn send(&mut self, message: McpMessage) -> Result<(), TransportError>;
    
    /// Receive a message from the transport
    async fn receive(&mut self) -> Result<McpMessage, TransportError>;
    
    /// Check if the transport is still connected
    fn is_connected(&self) -> bool;
    
    /// Close the transport connection
    async fn close(&mut self) -> Result<(), TransportError>;
}

/// Bidirectional stream transport
pub trait StreamTransport: AsyncRead + AsyncWrite + Send + Sync + Unpin {}

/// Generic stream-based transport implementation
pub struct StreamTransportAdapter<S: StreamTransport> {
    stream: S,
    read_buffer: Vec<u8>,
    write_buffer: Vec<u8>,
}

impl<S: StreamTransport> StreamTransportAdapter<S> {
    /// Create a new stream transport adapter
    pub fn new(stream: S) -> Self {
        Self {
            stream,
            read_buffer: Vec::with_capacity(8192),
            write_buffer: Vec::with_capacity(8192),
        }
    }
    
    /// Read a line from the stream
    async fn read_line(&mut self) -> Result<String, TransportError> {
        use tokio::io::AsyncBufReadExt;
        use tokio::io::BufReader;
        
        let mut reader = BufReader::new(&mut self.stream);
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        
        if line.is_empty() {
            return Err(TransportError::ConnectionClosed);
        }
        
        // Remove trailing newline
        if line.ends_with('\n') {
            line.pop();
            if line.ends_with('\r') {
                line.pop();
            }
        }
        
        Ok(line)
    }
    
    /// Write a line to the stream
    async fn write_line(&mut self, line: &str) -> Result<(), TransportError> {
        use tokio::io::AsyncWriteExt;
        
        self.stream.write_all(line.as_bytes()).await?;
        self.stream.write_all(b"\n").await?;
        self.stream.flush().await?;
        
        Ok(())
    }
}

#[async_trait]
impl<S: StreamTransport + 'static> Transport for StreamTransportAdapter<S> {
    async fn send(&mut self, message: McpMessage) -> Result<(), TransportError> {
        let json = serde_json::to_string(&message)?;
        self.write_line(&json).await
    }
    
    async fn receive(&mut self) -> Result<McpMessage, TransportError> {
        let line = self.read_line().await?;
        let message: McpMessage = serde_json::from_str(&line)?;
        Ok(message)
    }
    
    fn is_connected(&self) -> bool {
        // For stream-based transports, we assume connected unless proven otherwise
        true
    }
    
    async fn close(&mut self) -> Result<(), TransportError> {
        use tokio::io::AsyncWriteExt;
        self.stream.shutdown().await?;
        Ok(())
    }
}

/// Transport configuration
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Read timeout in milliseconds (None for no timeout)
    pub read_timeout_ms: Option<u64>,
    
    /// Write timeout in milliseconds (None for no timeout)
    pub write_timeout_ms: Option<u64>,
    
    /// Maximum message size in bytes
    pub max_message_size: usize,
    
    /// Buffer size for reading/writing
    pub buffer_size: usize,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            read_timeout_ms: Some(30000), // 30 seconds
            write_timeout_ms: Some(30000), // 30 seconds
            max_message_size: 10 * 1024 * 1024, // 10MB
            buffer_size: 8192,
        }
    }
}

/// Helper struct for implementing StreamTransport on compound types
pub struct DuplexStream<R, W> 
where
    R: AsyncRead + Send + Sync + Unpin,
    W: AsyncWrite + Send + Sync + Unpin,
{
    reader: R,
    writer: W,
}

impl<R, W> DuplexStream<R, W>
where
    R: AsyncRead + Send + Sync + Unpin,
    W: AsyncWrite + Send + Sync + Unpin,
{
    pub fn new(reader: R, writer: W) -> Self {
        Self { reader, writer }
    }
}

impl<R, W> AsyncRead for DuplexStream<R, W>
where
    R: AsyncRead + Send + Sync + Unpin,
    W: AsyncWrite + Send + Sync + Unpin,
{
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.reader).poll_read(cx, buf)
    }
}

impl<R, W> AsyncWrite for DuplexStream<R, W>
where
    R: AsyncRead + Send + Sync + Unpin,
    W: AsyncWrite + Send + Sync + Unpin,
{
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.writer).poll_write(cx, buf)
    }
    
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.writer).poll_flush(cx)
    }
    
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.writer).poll_shutdown(cx)
    }
}

impl<R, W> StreamTransport for DuplexStream<R, W>
where
    R: AsyncRead + Send + Sync + Unpin,
    W: AsyncWrite + Send + Sync + Unpin,
{}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{McpRequest, MessageId};
    
    #[tokio::test]
    async fn test_transport_config_default() {
        let config = TransportConfig::default();
        assert_eq!(config.read_timeout_ms, Some(30000));
        assert_eq!(config.write_timeout_ms, Some(30000));
        assert_eq!(config.max_message_size, 10 * 1024 * 1024);
        assert_eq!(config.buffer_size, 8192);
    }
}
