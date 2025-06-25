# Electron Quick Start with Woolly

This guide shows how to integrate Woolly into an Electron application for desktop AI capabilities.

## Project Setup

### 1. Initialize Electron Project

```bash
mkdir woolly-electron-app
cd woolly-electron-app
npm init -y
npm install electron --save-dev
npm install axios electron-store
```

### 2. Project Structure

```
woolly-electron-app/
├── package.json
├── main.js
├── preload.js
├── renderer.js
├── index.html
├── woolly-service.js
└── styles.css
```

## Complete Implementation

### package.json

```json
{
  "name": "woolly-electron-app",
  "version": "1.0.0",
  "description": "AI-powered desktop app with Woolly",
  "main": "main.js",
  "scripts": {
    "start": "electron .",
    "build": "electron-builder",
    "woolly-server": "woolly-server --port 11434"
  },
  "devDependencies": {
    "electron": "^27.0.0",
    "electron-builder": "^24.0.0"
  },
  "dependencies": {
    "axios": "^1.6.0",
    "electron-store": "^8.1.0"
  },
  "build": {
    "appId": "com.yourcompany.woolly-app",
    "productName": "Woolly AI Assistant",
    "directories": {
      "output": "dist"
    },
    "files": [
      "**/*",
      "!**/node_modules/*/{CHANGELOG.md,README.md,README,readme.md,readme}"
    ],
    "mac": {
      "category": "public.app-category.productivity"
    },
    "win": {
      "target": "nsis"
    },
    "linux": {
      "target": "AppImage"
    }
  }
}
```

### main.js - Main Process

```javascript
const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const Store = require('electron-store');

const store = new Store();
let mainWindow;
let woollyProcess;

// Start Woolly server
function startWoollyServer() {
  const woollyPath = store.get('woollyPath', 'woolly-server');
  const port = store.get('woollyPort', 11434);
  
  woollyProcess = spawn(woollyPath, [
    '--port', port.toString(),
    '--host', '127.0.0.1',
    '--models-dir', path.join(app.getPath('userData'), 'models')
  ]);
  
  woollyProcess.stdout.on('data', (data) => {
    console.log(`Woolly: ${data}`);
    if (mainWindow) {
      mainWindow.webContents.send('woolly-status', { 
        status: 'running', 
        message: data.toString() 
      });
    }
  });
  
  woollyProcess.stderr.on('data', (data) => {
    console.error(`Woolly Error: ${data}`);
    if (mainWindow) {
      mainWindow.webContents.send('woolly-status', { 
        status: 'error', 
        message: data.toString() 
      });
    }
  });
  
  woollyProcess.on('close', (code) => {
    console.log(`Woolly process exited with code ${code}`);
    if (mainWindow) {
      mainWindow.webContents.send('woolly-status', { 
        status: 'stopped', 
        code 
      });
    }
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    },
    icon: path.join(__dirname, 'assets/icon.png')
  });

  mainWindow.loadFile('index.html');
  
  // Start Woolly when window is ready
  mainWindow.webContents.on('did-finish-load', () => {
    startWoollyServer();
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  // Kill Woolly process
  if (woollyProcess) {
    woollyProcess.kill();
  }
  
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// IPC handlers
ipcMain.handle('get-models-path', () => {
  return path.join(app.getPath('userData'), 'models');
});

ipcMain.handle('get-app-version', () => {
  return app.getVersion();
});

ipcMain.handle('restart-woolly', async () => {
  if (woollyProcess) {
    woollyProcess.kill();
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  startWoollyServer();
});
```

### preload.js - Context Bridge

```javascript
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getModelsPath: () => ipcRenderer.invoke('get-models-path'),
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
  restartWoolly: () => ipcRenderer.invoke('restart-woolly'),
  onWoollyStatus: (callback) => {
    ipcRenderer.on('woolly-status', (event, status) => callback(status));
  }
});
```

### woolly-service.js - Woolly API Client

```javascript
class WoollyService {
  constructor(baseUrl = 'http://localhost:11434') {
    this.baseUrl = baseUrl;
    this.timeout = 60000; // 60 second timeout
  }
  
  async checkHealth() {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        timeout: 5000
      });
      return response.ok;
    } catch (error) {
      return false;
    }
  }
  
  async listModels() {
    const response = await fetch(`${this.baseUrl}/api/models`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }
  
  async loadModel(modelName) {
    const response = await fetch(`${this.baseUrl}/api/models/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: modelName })
    });
    if (!response.ok) {
      throw new Error(`Failed to load model: ${response.statusText}`);
    }
    return response.json();
  }
  
  async generateCompletion(prompt, options = {}) {
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_tokens: options.maxTokens || 200,
        temperature: options.temperature || 0.7,
        top_p: options.topP || 0.9,
        stream: options.stream || false
      })
    });
    
    if (!response.ok) {
      throw new Error(`Generation failed: ${response.statusText}`);
    }
    
    if (options.stream) {
      return this.handleStream(response);
    }
    
    return response.json();
  }
  
  async *handleStream(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            return;
          }
          try {
            yield JSON.parse(data);
          } catch (e) {
            console.error('Failed to parse streaming data:', e);
          }
        }
      }
    }
  }
  
  async chat(messages, options = {}) {
    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages,
        max_tokens: options.maxTokens || 200,
        temperature: options.temperature || 0.7,
        stream: options.stream || false
      })
    });
    
    if (!response.ok) {
      throw new Error(`Chat failed: ${response.statusText}`);
    }
    
    if (options.stream) {
      return this.handleStream(response);
    }
    
    return response.json();
  }
}
```

### renderer.js - Frontend Logic

```javascript
const woollyService = new WoollyService();

// State management
let currentModel = null;
let chatHistory = [];
let isGenerating = false;

// DOM elements
const modelSelect = document.getElementById('model-select');
const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const statusIndicator = document.getElementById('status-indicator');

// Initialize
async function initialize() {
  // Listen for Woolly status updates
  window.electronAPI.onWoollyStatus((status) => {
    updateStatus(status);
  });
  
  // Wait for Woolly to be ready
  await waitForWoolly();
  
  // Load available models
  await loadModels();
}

async function waitForWoolly(maxRetries = 30) {
  for (let i = 0; i < maxRetries; i++) {
    const isHealthy = await woollyService.checkHealth();
    if (isHealthy) {
      updateStatus({ status: 'running', message: 'Woolly is ready' });
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  throw new Error('Woolly failed to start');
}

async function loadModels() {
  try {
    const models = await woollyService.listModels();
    modelSelect.innerHTML = '';
    
    models.forEach(model => {
      const option = document.createElement('option');
      option.value = model.name;
      option.textContent = `${model.name} (${formatBytes(model.size)})`;
      modelSelect.appendChild(option);
    });
    
    if (models.length > 0) {
      await selectModel(models[0].name);
    }
  } catch (error) {
    console.error('Failed to load models:', error);
    showError('Failed to load models. Please check Woolly server.');
  }
}

async function selectModel(modelName) {
  try {
    showLoading('Loading model...');
    await woollyService.loadModel(modelName);
    currentModel = modelName;
    hideLoading();
    showSuccess(`Model ${modelName} loaded successfully`);
  } catch (error) {
    hideLoading();
    showError(`Failed to load model: ${error.message}`);
  }
}

async function sendMessage() {
  if (!currentModel || isGenerating) return;
  
  const message = messageInput.value.trim();
  if (!message) return;
  
  isGenerating = true;
  sendButton.disabled = true;
  messageInput.disabled = true;
  
  // Add user message to chat
  addMessage('user', message);
  messageInput.value = '';
  
  // Add placeholder for AI response
  const aiMessageId = addMessage('assistant', '', true);
  
  try {
    // Create chat messages array
    const messages = [
      ...chatHistory,
      { role: 'user', content: message }
    ];
    
    // Stream the response
    const stream = await woollyService.chat(messages, { stream: true });
    
    let fullResponse = '';
    for await (const chunk of stream) {
      if (chunk.content) {
        fullResponse += chunk.content;
        updateMessage(aiMessageId, fullResponse);
      }
    }
    
    // Update chat history
    chatHistory.push(
      { role: 'user', content: message },
      { role: 'assistant', content: fullResponse }
    );
    
  } catch (error) {
    updateMessage(aiMessageId, `Error: ${error.message}`);
    showError(`Failed to generate response: ${error.message}`);
  } finally {
    isGenerating = false;
    sendButton.disabled = false;
    messageInput.disabled = false;
    messageInput.focus();
  }
}

function addMessage(role, content, isStreaming = false) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${role}-message`;
  messageDiv.id = `message-${Date.now()}`;
  
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '👤' : '🤖';
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.textContent = content;
  
  if (isStreaming) {
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    cursor.textContent = '▊';
    contentDiv.appendChild(cursor);
  }
  
  messageDiv.appendChild(avatar);
  messageDiv.appendChild(contentDiv);
  chatContainer.appendChild(messageDiv);
  
  chatContainer.scrollTop = chatContainer.scrollHeight;
  
  return messageDiv.id;
}

function updateMessage(messageId, content) {
  const messageDiv = document.getElementById(messageId);
  if (!messageDiv) return;
  
  const contentDiv = messageDiv.querySelector('.message-content');
  contentDiv.textContent = content;
  
  // Add cursor if still streaming
  if (isGenerating) {
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    cursor.textContent = '▊';
    contentDiv.appendChild(cursor);
  }
  
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// UI helpers
function updateStatus(status) {
  statusIndicator.className = `status ${status.status}`;
  statusIndicator.title = status.message || status.status;
}

function showLoading(message) {
  const loader = document.getElementById('loader');
  loader.textContent = message;
  loader.style.display = 'block';
}

function hideLoading() {
  const loader = document.getElementById('loader');
  loader.style.display = 'none';
}

function showError(message) {
  showNotification(message, 'error');
}

function showSuccess(message) {
  showNotification(message, 'success');
}

function showNotification(message, type) {
  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.textContent = message;
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.remove();
  }, 5000);
}

function formatBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

// Event listeners
modelSelect.addEventListener('change', (e) => {
  selectModel(e.target.value);
});

messageInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendButton.addEventListener('click', sendMessage);

// Initialize on load
window.addEventListener('DOMContentLoaded', initialize);
```

### index.html - UI

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Woolly AI Assistant</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div id="app">
    <header>
      <h1>Woolly AI Assistant</h1>
      <div class="controls">
        <select id="model-select" class="model-select">
          <option>Loading models...</option>
        </select>
        <div id="status-indicator" class="status loading"></div>
      </div>
    </header>
    
    <main>
      <div id="chat-container" class="chat-container"></div>
      
      <div class="input-container">
        <textarea 
          id="message-input" 
          class="message-input"
          placeholder="Type your message..."
          rows="3"
        ></textarea>
        <button id="send-button" class="send-button">Send</button>
      </div>
    </main>
    
    <div id="loader" class="loader" style="display: none;"></div>
  </div>
  
  <script src="woolly-service.js"></script>
  <script src="renderer.js"></script>
</body>
</html>
```

### styles.css - Styling

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f5f5f5;
  color: #333;
  height: 100vh;
  overflow: hidden;
}

#app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

header {
  background: #fff;
  border-bottom: 1px solid #e0e0e0;
  padding: 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

h1 {
  font-size: 1.5rem;
  color: #2c3e50;
}

.controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.model-select {
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #fff;
  cursor: pointer;
}

.status {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  transition: background-color 0.3s;
}

.status.loading {
  background: #f39c12;
  animation: pulse 1s infinite;
}

.status.running {
  background: #27ae60;
}

.status.error {
  background: #e74c3c;
}

.status.stopped {
  background: #95a5a6;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
}

main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  background: #fff;
  margin: 1rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.message {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
  animation: fadeIn 0.3s;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.user-message {
  flex-direction: row-reverse;
}

.avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  background: #e0e0e0;
}

.user-message .avatar {
  background: #3498db;
}

.assistant-message .avatar {
  background: #2ecc71;
}

.message-content {
  max-width: 70%;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  background: #f8f9fa;
  line-height: 1.5;
  white-space: pre-wrap;
}

.user-message .message-content {
  background: #3498db;
  color: white;
}

.cursor {
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

.input-container {
  display: flex;
  gap: 0.5rem;
  padding: 1rem;
  background: #fff;
  border-top: 1px solid #e0e0e0;
}

.message-input {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  resize: none;
  font-family: inherit;
  font-size: 1rem;
}

.message-input:focus {
  outline: none;
  border-color: #3498db;
}

.send-button {
  padding: 0.75rem 1.5rem;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
  transition: background 0.3s;
}

.send-button:hover:not(:disabled) {
  background: #2980b9;
}

.send-button:disabled {
  background: #95a5a6;
  cursor: not-allowed;
}

.loader {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: rgba(0,0,0,0.8);
  color: white;
  padding: 1rem 2rem;
  border-radius: 4px;
  z-index: 1000;
}

.notification {
  position: fixed;
  top: 20px;
  right: 20px;
  padding: 1rem 1.5rem;
  border-radius: 4px;
  color: white;
  animation: slideIn 0.3s;
  z-index: 1001;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
  }
  to {
    transform: translateX(0);
  }
}

.notification.success {
  background: #27ae60;
}

.notification.error {
  background: #e74c3c;
}
```

## Running the Application

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Ensure Woolly Server is Available**
   ```bash
   # Install woolly-server globally or locally
   cargo install woolly-server
   ```

3. **Start the Application**
   ```bash
   npm start
   ```

## Building for Distribution

```bash
# Build for current platform
npm run build

# Build for all platforms
npm run build -- --mac --win --linux
```

## Advanced Features

### 1. Model Download Manager
```javascript
class ModelDownloadManager {
  constructor(modelsPath) {
    this.modelsPath = modelsPath;
    this.downloads = new Map();
  }
  
  async downloadModel(modelUrl, modelName, onProgress) {
    const response = await fetch(modelUrl);
    const reader = response.body.getReader();
    const contentLength = +response.headers.get('Content-Length');
    
    let receivedLength = 0;
    const chunks = [];
    
    while(true) {
      const {done, value} = await reader.read();
      if (done) break;
      
      chunks.push(value);
      receivedLength += value.length;
      
      if (onProgress) {
        onProgress({
          loaded: receivedLength,
          total: contentLength,
          percent: (receivedLength / contentLength) * 100
        });
      }
    }
    
    const blob = new Blob(chunks);
    const buffer = await blob.arrayBuffer();
    
    // Save to file system
    const fs = require('fs').promises;
    const modelPath = path.join(this.modelsPath, modelName);
    await fs.writeFile(modelPath, Buffer.from(buffer));
    
    return modelPath;
  }
}
```

### 2. GPU Detection and Configuration
```javascript
async function detectGPU() {
  try {
    const si = require('systeminformation');
    const graphics = await si.graphics();
    
    return {
      hasNvidia: graphics.controllers.some(c => 
        c.vendor.toLowerCase().includes('nvidia')
      ),
      hasAMD: graphics.controllers.some(c => 
        c.vendor.toLowerCase().includes('amd')
      ),
      hasMetal: process.platform === 'darwin',
      gpuInfo: graphics.controllers
    };
  } catch (error) {
    console.error('GPU detection failed:', error);
    return { hasNvidia: false, hasAMD: false, hasMetal: false };
  }
}
```

### 3. Auto-Update Integration
```javascript
const { autoUpdater } = require('electron-updater');

// Configure auto-updater
autoUpdater.checkForUpdatesAndNotify();

autoUpdater.on('update-available', () => {
  mainWindow.webContents.send('update-available');
});

autoUpdater.on('update-downloaded', () => {
  mainWindow.webContents.send('update-ready');
});

// In renderer
window.electronAPI.onUpdateReady(() => {
  if (confirm('Update is ready. Restart now?')) {
    window.electronAPI.restartApp();
  }
});
```

## Security Considerations

1. **Content Security Policy**
   ```html
   <meta http-equiv="Content-Security-Policy" 
         content="default-src 'self'; 
                  script-src 'self'; 
                  style-src 'self' 'unsafe-inline';">
   ```

2. **Input Sanitization**
   ```javascript
   function sanitizeInput(input) {
     return input
       .replace(/</g, '&lt;')
       .replace(/>/g, '&gt;')
       .replace(/"/g, '&quot;')
       .replace(/'/g, '&#x27;')
       .replace(/\//g, '&#x2F;');
   }
   ```

3. **Secure Storage**
   ```javascript
   const Store = require('electron-store');
   const store = new Store({
     encryptionKey: 'your-encryption-key',
     schema: {
       apiKeys: { type: 'object' },
       modelPaths: { type: 'array' }
     }
   });
   ```

## Performance Optimization

1. **Debounced Input**
   ```javascript
   function debounce(func, wait) {
     let timeout;
     return function executedFunction(...args) {
       const later = () => {
         clearTimeout(timeout);
         func(...args);
       };
       clearTimeout(timeout);
       timeout = setTimeout(later, wait);
     };
   }
   
   const debouncedSend = debounce(sendMessage, 300);
   ```

2. **Virtual Scrolling for Chat**
   ```javascript
   // Use virtual scrolling library for large chat histories
   const VirtualList = require('@tanstack/virtual-list');
   ```

3. **Worker Thread for Heavy Operations**
   ```javascript
   // In main process
   const { Worker } = require('worker_threads');
   
   const worker = new Worker('./woolly-worker.js');
   worker.on('message', (result) => {
     mainWindow.webContents.send('inference-result', result);
   });
   ```

## Troubleshooting

### Common Issues

1. **Woolly Server Won't Start**
   - Check if port is already in use
   - Verify woolly-server is in PATH
   - Check firewall settings

2. **Model Loading Fails**
   - Verify sufficient disk space
   - Check model file integrity
   - Ensure proper permissions

3. **Slow Performance**
   - Enable GPU acceleration
   - Use quantized models
   - Reduce context size

### Debug Mode
```javascript
// Enable debug logging
if (process.env.NODE_ENV === 'development') {
  require('electron-debug')();
  mainWindow.webContents.openDevTools();
}
```