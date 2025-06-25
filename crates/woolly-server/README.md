# Woolly Server

A high-performance HTTP/WebSocket server that exposes Woolly's LLM capabilities over HTTP and WebSocket connections with full Model Context Protocol (MCP) support.

## Features

- **HTTP REST API** for model inference and management
- **WebSocket support** for real-time communication and streaming
- **Model Context Protocol (MCP)** integration for tool execution and resource serving
- **JWT and API key authentication** with configurable access control
- **Rate limiting and request throttling** with per-user limits
- **Session management** for persistent conversations
- **Health monitoring** with readiness and liveness checks
- **Comprehensive middleware stack** with logging, CORS, and security headers

## Architecture

The server is built using a modular architecture with the following components:

### Core Components

- **Server State** (`server.rs`): Central state management for the inference engine, authentication, and MCP integration
- **Configuration** (`config.rs`): Comprehensive configuration system with TOML support
- **Error Handling** (`error.rs`): Structured error types with HTTP status code mapping
- **Authentication** (`auth.rs`): JWT and API key based authentication with scope management

### Middleware Stack

- **Authentication middleware**: JWT token and API key validation
- **Rate limiting**: Token bucket based rate limiting with user-specific limits
- **Request size limiting**: Configurable maximum request body sizes
- **Timeout middleware**: Request timeout handling
- **CORS**: Cross-origin request support
- **Security headers**: Standard security headers (HSTS, CSP, etc.)
- **Logging**: Structured request/response logging

### HTTP Endpoints

#### Health & Status
- `GET /api/v1/health` - Basic health check
- `GET /api/v1/health/ready` - Readiness check with dependency validation
- `GET /api/v1/health/live` - Liveness check

#### Authentication
- `POST /api/v1/auth/token` - Create JWT token
- `POST /api/v1/auth/validate` - Validate token

#### Model Management
- `GET /api/v1/models` - List available models
- `GET /api/v1/models/:name` - Get model information
- `POST /api/v1/models/:name/load` - Load a model
- `POST /api/v1/models/:name/unload` - Unload a model

#### Inference
- `POST /api/v1/inference/complete` - Text completion
- `POST /api/v1/inference/stream` - Streaming completion
- `POST /api/v1/inference/chat` - Chat completion

#### Session Management
- `GET /api/v1/sessions` - List user sessions
- `POST /api/v1/sessions` - Create new session
- `GET /api/v1/sessions/:id` - Get session info
- `DELETE /api/v1/sessions/:id` - Delete session

#### MCP Integration
- `GET /api/v1/mcp/tools` - List available tools
- `POST /api/v1/mcp/tools/:name` - Execute tool
- `GET /api/v1/mcp/resources` - List resources
- `GET /api/v1/mcp/resources/*path` - Get resource
- `GET /api/v1/mcp/prompts` - List prompts
- `POST /api/v1/mcp/prompts/:name` - Get prompt

### WebSocket Support

- **Endpoint**: `/api/v1/ws` and `/api/v1/mcp/ws`
- **Protocol**: Full MCP support over WebSocket
- **Features**: Real-time bidirectional communication, connection management, error handling
- **Authentication**: Same authentication system as HTTP endpoints

### MCP Integration

The server includes comprehensive Model Context Protocol support:

- **Protocol Implementation**: Full MCP message handling (requests, responses, notifications)
- **Tool Execution**: Dynamic tool registration and execution
- **Resource Serving**: File and metadata resource serving
- **Prompt Management**: Template-based prompt handling
- **Capabilities Advertisement**: Server capability discovery

## Configuration

The server uses a TOML configuration file with the following structure:

```toml
# Server binding
bind = "127.0.0.1:8080"

# Authentication
[auth]
jwt_secret = "your-jwt-secret"
jwt_expiration = 3600
api_keys = ["api-key-1", "api-key-2"]
allow_anonymous = false

# Rate limiting
[rate_limit]
authenticated_rpm = 60
anonymous_rpm = 10
burst_capacity = 10

# MCP configuration
[mcp]
enabled = true
protocol_version = "0.1.0"

[mcp.server_info]
name = "Woolly MCP Server"
version = "0.1.0"

# Model configuration
[models]
models_dir = "./models"
max_sessions = 10
default_model = "llama-7b"

# Request limits
[limits]
max_body_size = 10485760  # 10MB
request_timeout = 300     # 5 minutes
max_tokens = 4096
max_concurrent_requests = 100
```

## Usage

### Starting the Server

```bash
# Start with default configuration
woolly-server

# Start with custom configuration
woolly-server --config config.toml

# Start with custom bind address
woolly-server --bind 0.0.0.0:8080

# Generate default configuration
woolly-server config --output server.toml

# Validate configuration
woolly-server validate config.toml
```

### Example API Usage

#### Authentication
```bash
# Create token with API key
curl -X POST http://localhost:8080/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "api_key": "your-api-key"}'
```

#### Text Completion
```bash
# Non-streaming completion
curl -X POST http://localhost:8080/api/v1/inference/complete \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 100}'

# Streaming completion
curl -X POST http://localhost:8080/api/v1/inference/stream \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a story", "stream": true}'
```

#### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8080/api/v1/ws');

ws.onopen = function() {
    // Send MCP initialize message
    ws.send(JSON.stringify({
        type: "request",
        id: "1",
        method: "initialize",
        params: {
            protocol_version: "0.1.0",
            capabilities: {},
            client_info: {
                name: "WebClient",
                version: "1.0.0"
            }
        }
    }));
};
```

## Implementation Status

### Completed Components ✅

- **Core server architecture** with axum and tokio
- **HTTP endpoint handlers** for all major functionality
- **WebSocket implementation** with MCP protocol support
- **Authentication system** with JWT and API key support
- **Rate limiting** with token bucket algorithm
- **Middleware stack** with security and monitoring
- **Configuration system** with TOML support
- **Error handling** with proper HTTP status codes
- **MCP integration** with tool, resource, and prompt support

### Known Limitations

1. **Tensor Backend**: The underlying tensor computation backend requires additional implementation work to compile successfully. The server architecture is complete, but the tensor operations currently return stub responses.

2. **Model Loading**: Actual model loading and inference require completion of the tensor backend implementation.

3. **Persistence**: Session and model state is currently in-memory only. Production deployment would benefit from persistent storage.

### Next Steps for Production

1. **Complete Tensor Implementation**: Implement the missing tensor operations in the `woolly-tensor` crate
2. **Add Persistence**: Implement database storage for sessions and model state
3. **GPU Support**: Add CUDA/Metal backend implementations
4. **Performance Optimization**: Add connection pooling, caching, and other optimizations
5. **Monitoring**: Add metrics collection and observability
6. **Security Hardening**: Add additional security measures for production deployment

## Dependencies

The server builds on the following major dependencies:

- **axum**: HTTP server framework
- **tokio**: Async runtime
- **tower**: Middleware framework
- **serde**: Serialization
- **jsonwebtoken**: JWT handling
- **governor**: Rate limiting
- **tracing**: Logging and observability

## Contributing

The server architecture is designed to be modular and extensible. Key extension points include:

- **Custom authentication providers** via the auth module
- **Additional middleware** via the tower middleware stack
- **Custom MCP tools** via the tool registration system
- **Additional endpoints** via the route configuration

## License

This project is licensed under MIT OR Apache-2.0.