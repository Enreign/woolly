//! HTTP transport for MCP

use async_trait::async_trait;
use hyper::{Body, Client, Method, Request, Response, StatusCode};
use hyper::client::HttpConnector;
use hyper::header::{CONTENT_TYPE, CONTENT_LENGTH};
use tokio::sync::Mutex;
use std::sync::Arc;
use url::Url;

use crate::transport::{Transport, TransportError, TransportConfig};
use crate::types::{McpMessage, MessageId};

/// HTTP transport for MCP communication
pub struct HttpTransport {
    client: Client<HttpConnector>,
    endpoint: Url,
    config: TransportConfig,
    /// Queue for storing responses (since HTTP is request-response based)
    response_queue: Arc<Mutex<Vec<McpMessage>>>,
    connected: Arc<Mutex<bool>>,
}

impl HttpTransport {
    /// Create a new HTTP transport
    pub fn new(endpoint: Url, config: TransportConfig) -> Self {
        let client = Client::new();
        
        Self {
            client,
            endpoint,
            config,
            response_queue: Arc::new(Mutex::new(Vec::new())),
            connected: Arc::new(Mutex::new(true)),
        }
    }
    
    /// Create with default configuration
    pub fn with_endpoint(endpoint: Url) -> Self {
        Self::new(endpoint, TransportConfig::default())
    }
    
    /// Send HTTP request and get response
    async fn send_request(&self, message: &McpMessage) -> Result<Response<Body>, TransportError> {
        let json = serde_json::to_vec(&message)?;
        
        // Check message size
        if json.len() > self.config.max_message_size {
            return Err(TransportError::Transport(format!(
                "Message size {} exceeds maximum {}",
                json.len(),
                self.config.max_message_size
            )));
        }
        
        let request = Request::builder()
            .method(Method::POST)
            .uri(self.endpoint.as_str())
            .header(CONTENT_TYPE, "application/json")
            .header(CONTENT_LENGTH, json.len())
            .body(Body::from(json))
            .map_err(|e| TransportError::Transport(format!("Failed to build request: {}", e)))?;
        
        // Apply timeout if configured
        let response = if let Some(timeout_ms) = self.config.write_timeout_ms {
            let timeout = std::time::Duration::from_millis(timeout_ms);
            tokio::time::timeout(timeout, self.client.request(request))
                .await
                .map_err(|_| TransportError::Transport("Request timeout".to_string()))??
        } else {
            self.client.request(request).await?
        };
        
        Ok(response)
    }
    
    /// Parse response body as MCP message
    async fn parse_response(&self, response: Response<Body>) -> Result<McpMessage, TransportError> {
        if response.status() != StatusCode::OK {
            return Err(TransportError::Transport(format!(
                "HTTP error: {}",
                response.status()
            )));
        }
        
        let body_bytes = hyper::body::to_bytes(response.into_body()).await
            .map_err(|e| TransportError::Transport(format!("Failed to read response body: {}", e)))?;
        
        // Check response size
        if body_bytes.len() > self.config.max_message_size {
            return Err(TransportError::Transport(format!(
                "Response size {} exceeds maximum {}",
                body_bytes.len(),
                self.config.max_message_size
            )));
        }
        
        let message: McpMessage = serde_json::from_slice(&body_bytes)?;
        Ok(message)
    }
}

#[async_trait]
impl Transport for HttpTransport {
    async fn send(&mut self, message: McpMessage) -> Result<(), TransportError> {
        // For HTTP transport, we send and immediately wait for response
        let response = self.send_request(&message).await?;
        let response_message = self.parse_response(response).await?;
        
        // Store response in queue for later retrieval
        self.response_queue.lock().await.push(response_message);
        
        Ok(())
    }
    
    async fn receive(&mut self) -> Result<McpMessage, TransportError> {
        // Check if we have queued responses
        let mut queue = self.response_queue.lock().await;
        if let Some(message) = queue.pop() {
            return Ok(message);
        }
        drop(queue);
        
        // For HTTP transport, we need to poll or wait for incoming requests
        // This is a simplified implementation - in practice, you might want to:
        // 1. Run an HTTP server to receive incoming requests
        // 2. Use long polling or WebSockets for bidirectional communication
        // 3. Implement a proper event-driven architecture
        
        Err(TransportError::Protocol(
            "HTTP transport requires explicit request-response pattern".to_string()
        ))
    }
    
    fn is_connected(&self) -> bool {
        futures::executor::block_on(async {
            *self.connected.lock().await
        })
    }
    
    async fn close(&mut self) -> Result<(), TransportError> {
        *self.connected.lock().await = false;
        Ok(())
    }
}

/// HTTP server transport for receiving MCP requests
pub struct HttpServerTransport {
    bind_addr: std::net::SocketAddr,
    config: TransportConfig,
    request_queue: Arc<Mutex<Vec<McpMessage>>>,
    response_map: Arc<dashmap::DashMap<MessageId, McpMessage>>,
    server_handle: Option<tokio::task::JoinHandle<()>>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl HttpServerTransport {
    /// Create a new HTTP server transport
    pub fn new(bind_addr: std::net::SocketAddr, config: TransportConfig) -> Self {
        Self {
            bind_addr,
            config,
            request_queue: Arc::new(Mutex::new(Vec::new())),
            response_map: Arc::new(dashmap::DashMap::new()),
            server_handle: None,
            shutdown_tx: None,
        }
    }
    
    /// Start the HTTP server
    pub async fn start(&mut self) -> Result<(), TransportError> {
        use axum::{Router, Json, extract::State};
        use axum::routing::post;
        use std::sync::Arc;
        
        let request_queue = self.request_queue.clone();
        let response_map = self.response_map.clone();
        
        let app = Router::new()
            .route("/mcp", post(handle_mcp_request))
            .with_state((request_queue, response_map));
        
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        self.shutdown_tx = Some(shutdown_tx);
        
        let server = axum::Server::bind(&self.bind_addr)
            .serve(app.into_make_service())
            .with_graceful_shutdown(async {
                shutdown_rx.await.ok();
            });
        
        let handle = tokio::spawn(async move {
            if let Err(e) = server.await {
                eprintln!("Server error: {}", e);
            }
        });
        
        self.server_handle = Some(handle);
        Ok(())
    }
    
    /// Stop the HTTP server
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

/// Handle incoming MCP request
async fn handle_mcp_request(
    State((request_queue, response_map)): State<(
        Arc<Mutex<Vec<McpMessage>>>,
        Arc<dashmap::DashMap<MessageId, McpMessage>>
    )>,
    Json(message): Json<McpMessage>,
) -> Result<Json<McpMessage>, StatusCode> {
    match &message {
        McpMessage::Request(req) => {
            // Store request in queue
            request_queue.lock().await.push(message);
            
            // Wait for response (with timeout)
            let request_id = req.id;
            let start = tokio::time::Instant::now();
            let timeout = std::time::Duration::from_secs(30);
            
            loop {
                if let Some((_, response)) = response_map.remove(&request_id) {
                    return Ok(Json(response));
                }
                
                if start.elapsed() > timeout {
                    return Err(StatusCode::REQUEST_TIMEOUT);
                }
                
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }
        _ => Err(StatusCode::BAD_REQUEST),
    }
}

#[async_trait]
impl Transport for HttpServerTransport {
    async fn send(&mut self, message: McpMessage) -> Result<(), TransportError> {
        // For server transport, sending means storing a response
        if let McpMessage::Response(ref resp) = message {
            self.response_map.insert(resp.id, message);
            Ok(())
        } else {
            Err(TransportError::Protocol(
                "Server can only send responses".to_string()
            ))
        }
    }
    
    async fn receive(&mut self) -> Result<McpMessage, TransportError> {
        // Receive from request queue
        loop {
            let mut queue = self.request_queue.lock().await;
            if let Some(message) = queue.pop() {
                return Ok(message);
            }
            drop(queue);
            
            // Wait a bit before checking again
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
    
    fn is_connected(&self) -> bool {
        self.server_handle.is_some()
    }
    
    async fn close(&mut self) -> Result<(), TransportError> {
        self.stop().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_http_transport_creation() {
        let endpoint = Url::parse("http://localhost:8080/mcp").unwrap();
        let transport = HttpTransport::with_endpoint(endpoint);
        assert!(transport.is_connected());
    }
    
    #[tokio::test]
    async fn test_http_server_transport_creation() {
        let addr = "127.0.0.1:8080".parse().unwrap();
        let transport = HttpServerTransport::new(addr, TransportConfig::default());
        assert!(!transport.is_connected());
    }
}