# Woolly + Ole Integration Testing Checklist

This document provides a step-by-step guide to test the Woolly integration with Ole.

## Pre-requisites

- [ ] Rust installed (`rustc --version`)
- [ ] Node.js installed (`node --version`)
- [ ] Ole desktop client built and ready
- [ ] GGUF model file available (optional for full testing)

## Phase 1: Build and Start Woolly

### 1.1 Build Woolly Server
```bash
cd woolly
cargo build --release --bin woolly-server
```
✅ Expected: Build completes without errors

### 1.2 Start Woolly Server
```bash
./start-woolly.sh
# Or manually:
RUST_LOG=info ./target/release/woolly-server
```
✅ Expected: Server starts on http://localhost:8080

### 1.3 Verify Health
```bash
curl http://localhost:8080/api/v1/health
```
✅ Expected: `{"status":"ok","service":"woolly-server",...}`

## Phase 2: Test API Endpoints

### 2.1 Run Integration Test
```bash
node test-integration.js
```
✅ Expected: All tests pass

### 2.2 Manual API Tests

#### List Models
```bash
curl http://localhost:8080/api/v1/models
```
✅ Expected: Empty array or list of models

#### Test Chat (without model)
```bash
curl -X POST http://localhost:8080/api/v1/inference/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```
✅ Expected: Response or model not loaded error

## Phase 3: Ole Integration

### 3.1 Start Ole
```bash
cd ../desktop-client
npm run dev
```
✅ Expected: Ole starts without errors

### 3.2 Configure Woolly Provider

1. Open Ole
2. Go to Settings → Providers
3. Find Woolly in the list
4. Verify settings:
   - Base URL: `http://localhost:8080`
   - API Key: (leave empty for local testing)
5. Click "Test Connection"

✅ Expected: "Connection successful" message

### 3.3 Test Basic Chat

1. Select Woolly as the active provider
2. Start a new chat
3. Send a simple message: "Hello"

✅ Expected: Response from Woolly (may be mock response without model)

## Phase 4: Model Loading (Optional)

### 4.1 Download a GGUF Model
```bash
# Example: Download a small model for testing
cd woolly/models
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf
```

### 4.2 Load Model via API
```bash
curl -X POST http://localhost:8080/api/v1/models/llama-2-7b/load \
  -H "Content-Type: application/json" \
  -d '{"path": "./models/llama-2-7b.Q4_K_M.gguf"}'
```
✅ Expected: Model loads successfully

### 4.3 Test with Real Model
1. In Ole, refresh models list
2. Select the loaded model
3. Send a test prompt: "What is the capital of France?"

✅ Expected: Coherent response about Paris

## Phase 5: Advanced Features

### 5.1 Test Streaming
1. In Ole, send a longer prompt
2. Watch for streaming tokens

✅ Expected: Response streams in real-time

### 5.2 Test MCP Integration
1. In Ole, enable MCP tools
2. Send: "What files are in the current directory?"

✅ Expected: Tool execution and file listing

### 5.3 Test Performance
1. Send multiple messages rapidly
2. Monitor response times

✅ Expected: Consistent, fast responses

## Troubleshooting

### Issue: Woolly won't start
- Check port 8080 is free: `lsof -i :8080`
- Check Rust version: `rustc --version` (needs 1.70+)
- Check logs: `RUST_LOG=debug ./start-woolly.sh debug`

### Issue: Ole can't connect
- Verify Woolly is running: `curl http://localhost:8080/api/v1/health`
- Check Ole console for errors: Open Developer Tools
- Verify firewall isn't blocking localhost connections

### Issue: Model won't load
- Check file path is correct
- Verify file is valid GGUF format
- Ensure enough RAM (7B model needs ~6GB)
- Check Woolly logs for detailed errors

### Issue: Slow inference
- Use quantized models (Q4_K_M, Q5_K_M)
- Reduce max_tokens in requests
- Enable GPU acceleration if available
- Check system resources (CPU, RAM usage)

## Success Criteria

- [x] Woolly server builds and runs
- [x] All API endpoints respond correctly
- [x] Ole connects to Woolly successfully
- [x] Basic chat functionality works
- [ ] Model loading works (optional)
- [ ] Streaming responses work
- [ ] MCP tools integrate properly
- [ ] Performance meets expectations

## Next Steps

Once all tests pass:

1. Load your preferred models
2. Configure performance settings
3. Set up authentication if needed
4. Integrate MCP servers
5. Benchmark against Ollama
6. Deploy for production use

## Reporting Issues

If you encounter any issues:

1. Note the exact error message
2. Save relevant logs from both Woolly and Ole
3. Check if the issue is listed in troubleshooting
4. Report with reproduction steps

Happy testing! 🚀