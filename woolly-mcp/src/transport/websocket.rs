//! WebSocket transport for MCP

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use tokio::sync::Mutex;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{Error as WsError, Message as WsMessage},
    MaybeTlsStream, WebSocketStream,
};
use std::sync::Arc;
use url::Url;

use crate::transport::{Transport, TransportError, TransportConfig};
use crate::types::McpMessage;

/// WebSocket transport for MCP communication
pub struct WebSocketTransport {
    ws_stream: Arc<Mutex<WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>>>,
    config: TransportConfig,
    connected: Arc<Mutex<bool>>,
}

impl WebSocketTransport {
    /// Create a new WebSocket transport by connecting to the given URL
    pub async fn connect(url: Url, config: TransportConfig) -> Result<Self, TransportError> {
        let (ws_stream, _) = connect_async(url).await
            .map_err(|e| TransportError::Transport(format!("WebSocket connection failed: {}", e)))?;
        
        Ok(Self {
            ws_stream: Arc::new(Mutex::new(ws_stream)),
            config,
            connected: Arc::new(Mutex::new(true)),
        })
    }
    
    /// Create with default configuration
    pub async fn connect_with_default_config(url: Url) -> Result<Self, TransportError> {
        Self::connect(url, TransportConfig::default()).await
    }
}

#[async_trait]
impl Transport for WebSocketTransport {
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
        
        let ws_message = WsMessage::Text(json);
        let mut ws_stream = self.ws_stream.lock().await;
        
        // Apply write timeout if configured
        if let Some(timeout_ms) = self.config.write_timeout_ms {
            let timeout = std::time::Duration::from_millis(timeout_ms);
            tokio::time::timeout(timeout, ws_stream.send(ws_message))
                .await
                .map_err(|_| TransportError::Transport("Write timeout".to_string()))?
                .map_err(|e| TransportError::Transport(format!("WebSocket send error: {}", e)))?;
        } else {
            ws_stream.send(ws_message).await
                .map_err(|e| TransportError::Transport(format!("WebSocket send error: {}", e)))?;
        }
        
        Ok(())
    }
    
    async fn receive(&mut self) -> Result<McpMessage, TransportError> {
        let mut ws_stream = self.ws_stream.lock().await;
        
        // Apply read timeout if configured
        let ws_message = if let Some(timeout_ms) = self.config.read_timeout_ms {
            let timeout = std::time::Duration::from_millis(timeout_ms);
            tokio::time::timeout(timeout, ws_stream.next())
                .await
                .map_err(|_| TransportError::Transport("Read timeout".to_string()))?
        } else {
            ws_stream.next().await
        };
        
        match ws_message {
            Some(Ok(WsMessage::Text(text))) => {
                // Check message size
                if text.len() > self.config.max_message_size {
                    return Err(TransportError::Transport(format!(
                        "Message size {} exceeds maximum {}",
                        text.len(),
                        self.config.max_message_size
                    )));
                }
                
                let message: McpMessage = serde_json::from_str(&text)?;
                Ok(message)
            }
            Some(Ok(WsMessage::Binary(data))) => {
                // Check message size
                if data.len() > self.config.max_message_size {
                    return Err(TransportError::Transport(format!(
                        "Message size {} exceeds maximum {}",
                        data.len(),
                        self.config.max_message_size
                    )));
                }
                
                let message: McpMessage = serde_json::from_slice(&data)?;
                Ok(message)
            }
            Some(Ok(WsMessage::Close(_))) => {
                *self.connected.lock().await = false;
                Err(TransportError::ConnectionClosed)
            }
            Some(Ok(_)) => {
                // Ignore ping/pong messages
                self.receive().await
            }
            Some(Err(e)) => {
                *self.connected.lock().await = false;
                Err(TransportError::Transport(format!("WebSocket error: {}", e)))
            }
            None => {
                *self.connected.lock().await = false;
                Err(TransportError::ConnectionClosed)
            }
        }
    }
    
    fn is_connected(&self) -> bool {
        futures::executor::block_on(async {
            *self.connected.lock().await
        })
    }
    
    async fn close(&mut self) -> Result<(), TransportError> {
        *self.connected.lock().await = false;
        
        let mut ws_stream = self.ws_stream.lock().await;
        ws_stream.close(None).await
            .map_err(|e| TransportError::Transport(format!("WebSocket close error: {}", e)))?;
        
        Ok(())
    }
}

/// WebSocket server transport
pub struct WebSocketServerTransport {
    bind_addr: std::net::SocketAddr,
    config: TransportConfig,
    client_stream: Option<Arc<Mutex<WebSocketStream<tokio::net::TcpStream>>>>,
    server_handle: Option<tokio::task::JoinHandle<()>>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    accept_tx: Option<tokio::sync::mpsc::Sender<WebSocketStream<tokio::net::TcpStream>>>,
    accept_rx: Arc<Mutex<tokio::sync::mpsc::Receiver<WebSocketStream<tokio::net::TcpStream>>>>,
}

impl WebSocketServerTransport {
    /// Create a new WebSocket server transport
    pub fn new(bind_addr: std::net::SocketAddr, config: TransportConfig) -> Self {
        let (accept_tx, accept_rx) = tokio::sync::mpsc::channel(1);
        
        Self {
            bind_addr,
            config,
            client_stream: None,
            server_handle: None,
            shutdown_tx: None,
            accept_tx: Some(accept_tx),
            accept_rx: Arc::new(Mutex::new(accept_rx)),
        }
    }
    
    /// Start the WebSocket server
    pub async fn start(&mut self) -> Result<(), TransportError> {
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;
        
        let listener = TcpListener::bind(self.bind_addr).await
            .map_err(|e| TransportError::Transport(format!("Failed to bind: {}", e)))?;
        
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        self.shutdown_tx = Some(shutdown_tx);
        
        let accept_tx = self.accept_tx.as_ref().unwrap().clone();
        
        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    accept_result = listener.accept() => {
                        match accept_result {
                            Ok((stream, _)) => {
                                match accept_async(stream).await {
                                    Ok(ws_stream) => {
                                        if accept_tx.send(ws_stream).await.is_err() {
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("WebSocket accept error: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("TCP accept error: {}", e);
                            }
                        }
                    }
                    _ = &mut shutdown_rx => {
                        break;
                    }
                }
            }
        });
        
        self.server_handle = Some(handle);
        
        // Wait for first client connection
        self.accept_client().await?;
        
        Ok(())
    }
    
    /// Accept a client connection
    async fn accept_client(&mut self) -> Result<(), TransportError> {
        let mut accept_rx = self.accept_rx.lock().await;
        
        match accept_rx.recv().await {
            Some(ws_stream) => {
                self.client_stream = Some(Arc::new(Mutex::new(ws_stream)));
                Ok(())
            }
            None => Err(TransportError::Transport("No client connected".to_string())),
        }
    }
    
    /// Stop the WebSocket server
    pub async fn stop(&mut self) -> Result<(), TransportError> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        
        if let Some(handle) = self.server_handle.take() {
            handle.await.map_err(|e| TransportError::Transport(format!("Server shutdown error: {}", e)))?;
        }
        
        Ok(())
    }
}

#[async_trait]
impl Transport for WebSocketServerTransport {
    async fn send(&mut self, message: McpMessage) -> Result<(), TransportError> {
        if let Some(ref ws_stream) = self.client_stream {
            let json = serde_json::to_string(&message)?;
            
            // Check message size
            if json.len() > self.config.max_message_size {
                return Err(TransportError::Transport(format!(
                    "Message size {} exceeds maximum {}",
                    json.len(),
                    self.config.max_message_size
                )));
            }
            
            let ws_message = WsMessage::Text(json);
            let mut stream = ws_stream.lock().await;
            
            stream.send(ws_message).await
                .map_err(|e| TransportError::Transport(format!("WebSocket send error: {}", e)))?;
            
            Ok(())
        } else {
            Err(TransportError::Transport("No client connected".to_string()))
        }
    }
    
    async fn receive(&mut self) -> Result<McpMessage, TransportError> {
        if let Some(ref ws_stream) = self.client_stream {
            let mut stream = ws_stream.lock().await;
            
            match stream.next().await {
                Some(Ok(WsMessage::Text(text))) => {
                    // Check message size
                    if text.len() > self.config.max_message_size {
                        return Err(TransportError::Transport(format!(
                            "Message size {} exceeds maximum {}",
                            text.len(),
                            self.config.max_message_size
                        )));
                    }
                    
                    let message: McpMessage = serde_json::from_str(&text)?;
                    Ok(message)
                }
                Some(Ok(WsMessage::Close(_))) => {
                    self.client_stream = None;
                    // Try to accept a new client
                    drop(stream);
                    self.accept_client().await?;
                    self.receive().await
                }
                Some(Ok(_)) => {
                    // Ignore ping/pong messages
                    drop(stream);
                    self.receive().await
                }
                Some(Err(e)) => {
                    self.client_stream = None;
                    Err(TransportError::Transport(format!("WebSocket error: {}", e)))
                }
                None => {
                    self.client_stream = None;
                    Err(TransportError::ConnectionClosed)
                }
            }
        } else {
            Err(TransportError::Transport("No client connected".to_string()))
        }
    }
    
    fn is_connected(&self) -> bool {
        self.client_stream.is_some()
    }
    
    async fn close(&mut self) -> Result<(), TransportError> {
        if let Some(ref ws_stream) = self.client_stream {
            let mut stream = ws_stream.lock().await;
            let _ = stream.close(None).await;
        }
        
        self.client_stream = None;
        self.stop().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_websocket_server_creation() {
        let addr = "127.0.0.1:8081".parse().unwrap();
        let transport = WebSocketServerTransport::new(addr, TransportConfig::default());
        assert!(!transport.is_connected());
    }
}