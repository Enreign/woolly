# Python Desktop Apps with Woolly

This guide demonstrates how to build desktop applications in Python using Woolly as a local AI engine, with examples for both Tkinter and PyQt6.

## Overview

Python desktop applications can integrate Woolly through:
1. HTTP API calls to a local Woolly server
2. Process management for starting/stopping Woolly
3. Async operations for non-blocking UI

## Common Setup

### Requirements

```bash
pip install requests asyncio aiohttp PyQt6 customtkinter
```

### Woolly Client Class

```python
# woolly_client.py
import asyncio
import aiohttp
import json
import subprocess
import platform
import os
from typing import Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass

@dataclass
class ModelInfo:
    name: str
    size: int
    quantization: str
    loaded: bool

@dataclass
class GenerationConfig:
    max_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1

class WoollyClient:
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
        self.process: Optional[subprocess.Popen] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    def start_server(self, models_dir: str = None) -> bool:
        """Start the Woolly server process"""
        if self.process and self.process.poll() is None:
            return True  # Already running
            
        cmd = ["woolly-server", "--host", "127.0.0.1", "--port", str(self.port)]
        
        if models_dir:
            cmd.extend(["--models-dir", models_dir])
            
        try:
            # Platform-specific process creation
            if platform.system() == "Windows":
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
            
            # Wait for server to be ready
            import time
            for _ in range(30):  # 30 second timeout
                if self.check_health_sync():
                    return True
                time.sleep(1)
                
            return False
            
        except Exception as e:
            print(f"Failed to start Woolly server: {e}")
            return False
            
    def stop_server(self):
        """Stop the Woolly server process"""
        if self.process:
            if platform.system() == "Windows":
                self.process.terminate()
            else:
                import signal
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait()
            self.process = None
            
    def check_health_sync(self) -> bool:
        """Synchronous health check"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/health", timeout=1)
            return response.status_code == 200
        except:
            return False
            
    async def check_health(self) -> bool:
        """Check if Woolly server is running"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except:
            return False
            
    async def list_models(self) -> list[ModelInfo]:
        """List available models"""
        async with self.session.get(f"{self.base_url}/api/models") as response:
            if response.status != 200:
                raise Exception(f"Failed to list models: {response.status}")
            
            data = await response.json()
            return [
                ModelInfo(
                    name=m["name"],
                    size=m["size"],
                    quantization=m.get("quantization", "unknown"),
                    loaded=m.get("loaded", False)
                )
                for m in data
            ]
            
    async def load_model(self, model_name: str) -> bool:
        """Load a model"""
        async with self.session.post(
            f"{self.base_url}/api/models/load",
            json={"name": model_name}
        ) as response:
            return response.status == 200
            
    async def generate(
        self, 
        prompt: str, 
        config: GenerationConfig = GenerationConfig(),
        stream: bool = False
    ) -> str | AsyncIterator[str]:
        """Generate text completion"""
        
        payload = {
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repeat_penalty": config.repeat_penalty,
            "stream": stream
        }
        
        if not stream:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Generation failed: {response.status}")
                data = await response.json()
                return data["text"]
        else:
            return self._stream_generate(payload)
            
    async def _stream_generate(self, payload: dict) -> AsyncIterator[str]:
        """Stream generation tokens"""
        async with self.session.post(
            f"{self.base_url}/api/generate",
            json=payload
        ) as response:
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]
                        if data != '[DONE]':
                            try:
                                chunk = json.loads(data)
                                if 'token' in chunk:
                                    yield chunk['token']
                            except json.JSONDecodeError:
                                pass
                                
    async def chat(
        self,
        messages: list[Dict[str, str]],
        config: GenerationConfig = GenerationConfig(),
        stream: bool = False
    ) -> str | AsyncIterator[str]:
        """Chat completion"""
        
        payload = {
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": stream
        }
        
        if not stream:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Chat failed: {response.status}")
                data = await response.json()
                return data["content"]
        else:
            return self._stream_chat(payload)
            
    async def _stream_chat(self, payload: dict) -> AsyncIterator[str]:
        """Stream chat tokens"""
        async with self.session.post(
            f"{self.base_url}/api/chat",
            json=payload
        ) as response:
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]
                        if data != '[DONE]':
                            try:
                                chunk = json.loads(data)
                                if 'content' in chunk:
                                    yield chunk['content']
                            except json.JSONDecodeError:
                                pass
```

## Tkinter Implementation

### Modern Tkinter App with CustomTkinter

```python
# woolly_tkinter_app.py
import tkinter as tk
import customtkinter as ctk
from tkinter import scrolledtext, filedialog, messagebox
import asyncio
import threading
from datetime import datetime
import os
from pathlib import Path

from woolly_client import WoollyClient, GenerationConfig, ModelInfo

# Configure appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class WoollyTkinterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Woolly AI Assistant")
        self.geometry("1200x800")
        
        # Initialize Woolly client
        self.woolly = WoollyClient()
        self.current_model = None
        self.chat_history = []
        self.is_generating = False
        
        # Setup UI
        self.setup_ui()
        
        # Start async event loop in thread
        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.async_thread.start()
        
        # Start Woolly server
        self.after(100, self.start_woolly_server)
        
    def setup_ui(self):
        # Main container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        self.setup_header()
        
        # Chat area
        self.setup_chat_area()
        
        # Input area
        self.setup_input_area()
        
        # Status bar
        self.setup_status_bar()
        
    def setup_header(self):
        header_frame = ctk.CTkFrame(self.main_container)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame, 
            text="Woolly AI Assistant",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(side="left", padx=10)
        
        # Model controls
        controls_frame = ctk.CTkFrame(header_frame)
        controls_frame.pack(side="right", padx=10)
        
        self.model_var = tk.StringVar()
        self.model_dropdown = ctk.CTkComboBox(
            controls_frame,
            variable=self.model_var,
            values=["Loading..."],
            state="readonly",
            width=200
        )
        self.model_dropdown.pack(side="left", padx=5)
        
        self.load_model_btn = ctk.CTkButton(
            controls_frame,
            text="Load Model",
            command=self.load_model,
            width=100
        )
        self.load_model_btn.pack(side="left", padx=5)
        
        self.upload_model_btn = ctk.CTkButton(
            controls_frame,
            text="Upload Model",
            command=self.upload_model,
            width=100
        )
        self.upload_model_btn.pack(side="left", padx=5)
        
    def setup_chat_area(self):
        # Chat frame with canvas for custom styling
        chat_frame = ctk.CTkFrame(self.main_container)
        chat_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Chat display with custom text widget
        self.chat_display = ctk.CTkTextbox(
            chat_frame,
            wrap="word",
            font=ctk.CTkFont(size=14)
        )
        self.chat_display.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configure tags for message styling
        self.chat_display.tag_config("user", justify="right")
        self.chat_display.tag_config("assistant", justify="left")
        self.chat_display.tag_config("system", justify="center", foreground="gray")
        
    def setup_input_area(self):
        input_frame = ctk.CTkFrame(self.main_container)
        input_frame.pack(fill="x", padx=5, pady=5)
        
        # Message input
        self.message_input = ctk.CTkTextbox(
            input_frame,
            height=80,
            wrap="word",
            font=ctk.CTkFont(size=14)
        )
        self.message_input.pack(side="left", fill="both", expand=True, padx=5)
        
        # Bind Enter key
        self.message_input.bind("<Return>", self.send_message_event)
        self.message_input.bind("<Shift-Return>", lambda e: None)
        
        # Send button
        self.send_btn = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.send_message,
            width=80,
            height=80
        )
        self.send_btn.pack(side="right", padx=5)
        
    def setup_status_bar(self):
        status_frame = ctk.CTkFrame(self.main_container)
        status_frame.pack(fill="x", padx=5, pady=5)
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Starting Woolly server...",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=10)
        
        self.tokens_label = ctk.CTkLabel(
            status_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.tokens_label.pack(side="right", padx=10)
        
    def _run_async_loop(self):
        """Run async event loop in separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        
    def run_async(self, coro):
        """Run async coroutine from main thread"""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future
        
    def start_woolly_server(self):
        """Start Woolly server and load models"""
        models_dir = Path.home() / ".woolly" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        if self.woolly.start_server(str(models_dir)):
            self.status_label.configure(text="Woolly server running")
            self.after(1000, self.refresh_models)
        else:
            self.status_label.configure(text="Failed to start Woolly server")
            messagebox.showerror("Error", "Failed to start Woolly server")
            
    def refresh_models(self):
        """Refresh available models"""
        async def _refresh():
            async with WoollyClient() as client:
                try:
                    models = await client.list_models()
                    return models
                except Exception as e:
                    print(f"Failed to list models: {e}")
                    return []
                    
        future = self.run_async(_refresh())
        self.after(100, lambda: self._update_models(future))
        
    def _update_models(self, future):
        """Update models dropdown"""
        if future.done():
            models = future.result()
            if models:
                model_names = [m.name for m in models]
                self.model_dropdown.configure(values=model_names)
                
                # Select loaded model if any
                for model in models:
                    if model.loaded:
                        self.model_var.set(model.name)
                        self.current_model = model.name
                        break
            else:
                self.model_dropdown.configure(values=["No models found"])
        else:
            self.after(100, lambda: self._update_models(future))
            
    def load_model(self):
        """Load selected model"""
        model_name = self.model_var.get()
        if not model_name or model_name == "Loading..." or model_name == "No models found":
            return
            
        self.status_label.configure(text=f"Loading model: {model_name}")
        self.load_model_btn.configure(state="disabled")
        
        async def _load():
            async with WoollyClient() as client:
                try:
                    success = await client.load_model(model_name)
                    return success, None
                except Exception as e:
                    return False, str(e)
                    
        future = self.run_async(_load())
        self.after(100, lambda: self._handle_model_load(future, model_name))
        
    def _handle_model_load(self, future, model_name):
        """Handle model load result"""
        if future.done():
            success, error = future.result()
            self.load_model_btn.configure(state="normal")
            
            if success:
                self.current_model = model_name
                self.status_label.configure(text=f"Model loaded: {model_name}")
                self.add_system_message(f"Model '{model_name}' loaded successfully")
            else:
                self.status_label.configure(text="Failed to load model")
                messagebox.showerror("Error", f"Failed to load model: {error}")
        else:
            self.after(100, lambda: self._handle_model_load(future, model_name))
            
    def upload_model(self):
        """Upload a model file"""
        file_path = filedialog.askopenfilename(
            title="Select GGUF Model",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        
        if file_path:
            # Copy to models directory
            models_dir = Path.home() / ".woolly" / "models"
            dest_path = models_dir / Path(file_path).name
            
            try:
                import shutil
                shutil.copy2(file_path, dest_path)
                self.add_system_message(f"Model uploaded: {Path(file_path).name}")
                self.refresh_models()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to upload model: {e}")
                
    def send_message_event(self, event):
        """Handle Enter key press"""
        if not event.state & 0x1:  # Not Shift key
            self.send_message()
            return "break"
            
    def send_message(self):
        """Send user message"""
        if self.is_generating or not self.current_model:
            return
            
        message = self.message_input.get("1.0", "end-1c").strip()
        if not message:
            return
            
        # Clear input
        self.message_input.delete("1.0", "end")
        
        # Add user message
        self.add_message("user", message)
        self.chat_history.append({"role": "user", "content": message})
        
        # Generate response
        self.is_generating = True
        self.send_btn.configure(state="disabled", text="Generating...")
        
        async def _generate():
            async with WoollyClient() as client:
                try:
                    # Stream the response
                    full_response = ""
                    token_count = 0
                    start_time = asyncio.get_event_loop().time()
                    
                    async for token in await client.chat(
                        self.chat_history[-10:],  # Last 10 messages for context
                        GenerationConfig(),
                        stream=True
                    ):
                        full_response += token
                        token_count += 1
                        
                        # Update UI periodically
                        if token_count % 5 == 0:
                            elapsed = asyncio.get_event_loop().time() - start_time
                            tokens_per_sec = token_count / max(elapsed, 0.001)
                            
                            self.after(0, lambda r=full_response, t=tokens_per_sec: 
                                      self._update_response(r, t))
                                      
                    return full_response, None
                    
                except Exception as e:
                    return None, str(e)
                    
        future = self.run_async(_generate())
        self.after(100, lambda: self._handle_generation(future))
        
    def _update_response(self, response, tokens_per_sec):
        """Update streaming response"""
        # Update the last assistant message or create new one
        self.chat_display.delete("end-2c", "end-1c")
        self.chat_display.insert("end", response + "▊", "assistant")
        self.chat_display.see("end")
        
        # Update stats
        self.tokens_label.configure(text=f"{tokens_per_sec:.1f} tokens/s")
        
    def _handle_generation(self, future):
        """Handle generation completion"""
        if future.done():
            response, error = future.result()
            
            self.is_generating = False
            self.send_btn.configure(state="normal", text="Send")
            self.tokens_label.configure(text="")
            
            if response:
                # Remove cursor and finalize message
                self.chat_display.delete("end-2c", "end-1c")
                self.chat_display.insert("end", response, "assistant")
                self.chat_display.insert("end", "\n\n")
                
                self.chat_history.append({"role": "assistant", "content": response})
            else:
                self.add_system_message(f"Error: {error}")
        else:
            self.after(100, lambda: self._handle_generation(future))
            
    def add_message(self, role, content):
        """Add message to chat display"""
        timestamp = datetime.now().strftime("%H:%M")
        
        if role == "user":
            self.chat_display.insert("end", f"You ({timestamp}):\n", "user")
            self.chat_display.insert("end", f"{content}\n\n", "user")
        elif role == "assistant":
            self.chat_display.insert("end", f"Assistant ({timestamp}):\n", "assistant")
            self.chat_display.insert("end", f"{content}\n\n", "assistant")
            
        self.chat_display.see("end")
        
    def add_system_message(self, message):
        """Add system message to chat"""
        self.chat_display.insert("end", f"[System] {message}\n\n", "system")
        self.chat_display.see("end")
        
    def on_closing(self):
        """Clean up on close"""
        self.woolly.stop_server()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.destroy()

if __name__ == "__main__":
    app = WoollyTkinterApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
```

## PyQt6 Implementation

### Professional PyQt6 Application

```python
# woolly_pyqt_app.py
import sys
import asyncio
import qasync
from pathlib import Path
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from woolly_client import WoollyClient, GenerationConfig, ModelInfo

class WorkerSignals(QObject):
    """Signals for async operations"""
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str, float)
    finished = pyqtSignal()

class AsyncWorker(QRunnable):
    """Worker for running async operations"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()
        
    @pyqtSlot()
    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.fn())
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

class ChatWidget(QWidget):
    """Custom chat display widget"""
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Chat display
        self.chat_display = QTextBrowser()
        self.chat_display.setOpenExternalLinks(False)
        self.chat_display.setStyleSheet("""
            QTextBrowser {
                background-color: #1e1e1e;
                color: #ffffff;
                border: none;
                padding: 10px;
                font-size: 14px;
            }
        """)
        
        layout.addWidget(self.chat_display)
        
    def add_message(self, role: str, content: str, timestamp: Optional[str] = None):
        """Add a message to the chat"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M")
            
        if role == "user":
            html = f"""
            <div style="text-align: right; margin: 10px 0;">
                <span style="color: #888; font-size: 12px;">You ({timestamp})</span><br>
                <span style="background-color: #0084ff; color: white; padding: 8px 12px; 
                      border-radius: 18px; display: inline-block; max-width: 70%;">
                    {content}
                </span>
            </div>
            """
        elif role == "assistant":
            html = f"""
            <div style="text-align: left; margin: 10px 0;">
                <span style="color: #888; font-size: 12px;">Assistant ({timestamp})</span><br>
                <span style="background-color: #333; color: white; padding: 8px 12px; 
                      border-radius: 18px; display: inline-block; max-width: 70%;">
                    {content}
                </span>
            </div>
            """
        else:  # system
            html = f"""
            <div style="text-align: center; margin: 10px 0;">
                <span style="color: #888; font-size: 12px; font-style: italic;">
                    {content}
                </span>
            </div>
            """
            
        self.chat_display.append(html)
        
        # Scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

class WoollyPyQtApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.woolly = WoollyClient()
        self.current_model = None
        self.chat_history = []
        self.is_generating = False
        
        # Thread pool for async operations
        self.thread_pool = QThreadPool()
        
        self.init_ui()
        self.setStyleSheet(self.load_stylesheet())
        
        # Start Woolly server
        QTimer.singleShot(100, self.start_woolly_server)
        
    def init_ui(self):
        self.setWindowTitle("Woolly AI Assistant")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        self.create_header(main_layout)
        
        # Chat area
        self.chat_widget = ChatWidget()
        main_layout.addWidget(self.chat_widget)
        
        # Input area
        self.create_input_area(main_layout)
        
        # Status bar
        self.create_status_bar()
        
    def create_header(self, parent_layout):
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        
        # Title
        title_label = QLabel("Woolly AI Assistant")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Model controls
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.addItem("Loading models...")
        header_layout.addWidget(self.model_combo)
        
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        header_layout.addWidget(self.load_model_btn)
        
        self.upload_model_btn = QPushButton("Upload Model")
        self.upload_model_btn.clicked.connect(self.upload_model)
        header_layout.addWidget(self.upload_model_btn)
        
        parent_layout.addWidget(header_widget)
        
    def create_input_area(self, parent_layout):
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        
        # Message input
        self.message_input = QTextEdit()
        self.message_input.setMaximumHeight(80)
        self.message_input.setPlaceholderText("Type your message...")
        self.message_input.installEventFilter(self)
        input_layout.addWidget(self.message_input)
        
        # Send button
        self.send_btn = QPushButton("Send")
        self.send_btn.setMinimumHeight(80)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)
        
        parent_layout.addWidget(input_widget)
        
    def create_status_bar(self):
        self.status_bar = self.statusBar()
        
        # Status label
        self.status_label = QLabel("Starting Woolly server...")
        self.status_bar.addWidget(self.status_label)
        
        # Tokens/sec label
        self.tokens_label = QLabel("")
        self.status_bar.addPermanentWidget(self.tokens_label)
        
    def eventFilter(self, obj, event):
        """Handle Enter key in message input"""
        if obj == self.message_input and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.send_message()
                return True
        return super().eventFilter(obj, event)
        
    def start_woolly_server(self):
        """Start Woolly server"""
        models_dir = Path.home() / ".woolly" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        if self.woolly.start_server(str(models_dir)):
            self.status_label.setText("Woolly server running")
            self.refresh_models()
        else:
            self.status_label.setText("Failed to start Woolly server")
            QMessageBox.critical(self, "Error", "Failed to start Woolly server")
            
    def refresh_models(self):
        """Refresh available models"""
        async def _refresh():
            async with WoollyClient() as client:
                return await client.list_models()
                
        worker = AsyncWorker(_refresh)
        worker.signals.result.connect(self.on_models_loaded)
        worker.signals.error.connect(lambda e: print(f"Failed to load models: {e}"))
        self.thread_pool.start(worker)
        
    def on_models_loaded(self, models):
        """Update models dropdown"""
        self.model_combo.clear()
        
        if models:
            for model in models:
                self.model_combo.addItem(f"{model.name} ({self.format_bytes(model.size)})")
                if model.loaded:
                    self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
                    self.current_model = model.name
        else:
            self.model_combo.addItem("No models found")
            
    def load_model(self):
        """Load selected model"""
        if self.model_combo.currentText() in ["Loading models...", "No models found"]:
            return
            
        model_name = self.model_combo.currentText().split(" (")[0]
        self.status_label.setText(f"Loading model: {model_name}")
        self.load_model_btn.setEnabled(False)
        
        async def _load():
            async with WoollyClient() as client:
                await client.load_model(model_name)
                return model_name
                
        worker = AsyncWorker(_load)
        worker.signals.result.connect(self.on_model_loaded)
        worker.signals.error.connect(self.on_model_load_error)
        worker.signals.finished.connect(lambda: self.load_model_btn.setEnabled(True))
        self.thread_pool.start(worker)
        
    def on_model_loaded(self, model_name):
        """Handle successful model load"""
        self.current_model = model_name
        self.status_label.setText(f"Model loaded: {model_name}")
        self.chat_widget.add_message("system", f"Model '{model_name}' loaded successfully")
        
    def on_model_load_error(self, error):
        """Handle model load error"""
        self.status_label.setText("Failed to load model")
        QMessageBox.critical(self, "Error", f"Failed to load model: {error}")
        
    def upload_model(self):
        """Upload a model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GGUF Model",
            "",
            "GGUF Files (*.gguf);;All Files (*.*)"
        )
        
        if file_path:
            models_dir = Path.home() / ".woolly" / "models"
            dest_path = models_dir / Path(file_path).name
            
            try:
                import shutil
                shutil.copy2(file_path, dest_path)
                self.chat_widget.add_message("system", f"Model uploaded: {Path(file_path).name}")
                self.refresh_models()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to upload model: {e}")
                
    def send_message(self):
        """Send user message"""
        if self.is_generating or not self.current_model:
            return
            
        message = self.message_input.toPlainText().strip()
        if not message:
            return
            
        # Clear input
        self.message_input.clear()
        
        # Add user message
        self.chat_widget.add_message("user", message)
        self.chat_history.append({"role": "user", "content": message})
        
        # Generate response
        self.is_generating = True
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Generating...")
        
        # Create temporary message for streaming
        timestamp = datetime.now().strftime("%H:%M")
        self.streaming_message_start = len(self.chat_widget.chat_display.toPlainText())
        self.chat_widget.add_message("assistant", "▊", timestamp)
        
        async def _generate():
            async with WoollyClient() as client:
                full_response = ""
                token_count = 0
                start_time = asyncio.get_event_loop().time()
                
                async for token in await client.chat(
                    self.chat_history[-10:],
                    GenerationConfig(),
                    stream=True
                ):
                    full_response += token
                    token_count += 1
                    
                    # Emit progress
                    if token_count % 5 == 0:
                        elapsed = asyncio.get_event_loop().time() - start_time
                        tokens_per_sec = token_count / max(elapsed, 0.001)
                        yield (full_response, tokens_per_sec)
                        
                yield (full_response, 0)  # Final update
                
        worker = StreamingWorker(_generate)
        worker.signals.progress.connect(self.update_streaming_response)
        worker.signals.finished.connect(self.on_generation_complete)
        worker.signals.error.connect(self.on_generation_error)
        self.thread_pool.start(worker)
        
    def update_streaming_response(self, response, tokens_per_sec):
        """Update streaming response in chat"""
        # Update tokens/sec
        if tokens_per_sec > 0:
            self.tokens_label.setText(f"{tokens_per_sec:.1f} tokens/s")
            
        # Update message content
        cursor = self.chat_widget.chat_display.textCursor()
        cursor.setPosition(self.streaming_message_start)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        
        # Reinsert with updated content
        timestamp = datetime.now().strftime("%H:%M")
        html = f"""
        <div style="text-align: left; margin: 10px 0;">
            <span style="color: #888; font-size: 12px;">Assistant ({timestamp})</span><br>
            <span style="background-color: #333; color: white; padding: 8px 12px; 
                  border-radius: 18px; display: inline-block; max-width: 70%;">
                {response}▊
            </span>
        </div>
        """
        cursor.insertHtml(html)
        
        # Scroll to bottom
        scrollbar = self.chat_widget.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def on_generation_complete(self, final_response):
        """Handle generation completion"""
        self.is_generating = False
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send")
        self.tokens_label.setText("")
        
        # Remove cursor from final message
        cursor = self.chat_widget.chat_display.textCursor()
        cursor.setPosition(self.streaming_message_start)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        
        timestamp = datetime.now().strftime("%H:%M")
        html = f"""
        <div style="text-align: left; margin: 10px 0;">
            <span style="color: #888; font-size: 12px;">Assistant ({timestamp})</span><br>
            <span style="background-color: #333; color: white; padding: 8px 12px; 
                  border-radius: 18px; display: inline-block; max-width: 70%;">
                {final_response}
            </span>
        </div>
        """
        cursor.insertHtml(html)
        
        # Add to history
        self.chat_history.append({"role": "assistant", "content": final_response})
        
    def on_generation_error(self, error):
        """Handle generation error"""
        self.is_generating = False
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send")
        self.tokens_label.setText("")
        
        self.chat_widget.add_message("system", f"Error: {error}")
        
    def format_bytes(self, bytes):
        """Format bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024:
                return f"{bytes:.1f} {unit}"
            bytes /= 1024
        return f"{bytes:.1f} TB"
        
    def load_stylesheet(self):
        """Load dark theme stylesheet"""
        return """
        QMainWindow {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QWidget {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QPushButton {
            background-color: #0084ff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #0073e6;
        }
        
        QPushButton:pressed {
            background-color: #005bb5;
        }
        
        QPushButton:disabled {
            background-color: #555;
            color: #888;
        }
        
        QComboBox {
            background-color: #2d2d2d;
            border: 1px solid #555;
            padding: 5px;
            border-radius: 4px;
        }
        
        QComboBox::drop-down {
            border: none;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #888;
            margin-right: 5px;
        }
        
        QTextEdit {
            background-color: #2d2d2d;
            border: 1px solid #555;
            padding: 8px;
            border-radius: 4px;
        }
        
        QStatusBar {
            background-color: #2d2d2d;
            color: #888;
        }
        """
        
    def closeEvent(self, event):
        """Clean up on close"""
        self.woolly.stop_server()
        event.accept()

class StreamingWorker(AsyncWorker):
    """Special worker for streaming responses"""
    def __init__(self, async_gen_fn):
        self.async_gen_fn = async_gen_fn
        super().__init__(None)
        
    @pyqtSlot()
    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def _run():
                final_response = ""
                async for response, tokens_per_sec in self.async_gen_fn():
                    self.signals.progress.emit(response, tokens_per_sec)
                    final_response = response
                return final_response
                
            result = loop.run_until_complete(_run())
            self.signals.finished.emit(result)
            
        except Exception as e:
            self.signals.error.emit(str(e))

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Woolly AI Assistant")
    
    # Enable high DPI scaling
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    
    window = WoollyPyQtApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

## Running the Applications

### Tkinter App
```bash
pip install customtkinter aiohttp
python woolly_tkinter_app.py
```

### PyQt6 App
```bash
pip install PyQt6 qasync aiohttp
python woolly_pyqt_app.py
```

## Advanced Features

### 1. Model Download Manager
```python
class ModelDownloader(QThread):
    progress = pyqtSignal(int, int)  # downloaded, total
    finished = pyqtSignal(str)  # file path
    error = pyqtSignal(str)
    
    def __init__(self, url, dest_path):
        super().__init__()
        self.url = url
        self.dest_path = dest_path
        
    def run(self):
        try:
            response = requests.get(self.url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.dest_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    self.progress.emit(downloaded, total_size)
                    
            self.finished.emit(self.dest_path)
            
        except Exception as e:
            self.error.emit(str(e))
```

### 2. Settings Dialog
```python
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        
        layout = QFormLayout(self)
        
        # Temperature
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)
        self.temp_label = QLabel("0.7")
        
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_label)
        
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_label.setText(f"{v/100:.2f}")
        )
        
        layout.addRow("Temperature:", temp_layout)
        
        # Max tokens
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(50, 4096)
        self.max_tokens.setValue(200)
        layout.addRow("Max Tokens:", self.max_tokens)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
```

### 3. Export Chat History
```python
def export_chat(self):
    """Export chat history to file"""
    file_path, _ = QFileDialog.getSaveFileName(
        self,
        "Export Chat",
        f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        "Markdown Files (*.md);;Text Files (*.txt);;All Files (*.*)"
    )
    
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("# Woolly Chat Export\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.current_model}\n\n")
            
            for msg in self.chat_history:
                role = msg['role'].capitalize()
                content = msg['content']
                f.write(f"## {role}\n\n{content}\n\n")
```

## Packaging for Distribution

### PyInstaller Configuration
```python
# woolly_app.spec
a = Analysis(
    ['woolly_pyqt_app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['aiohttp', 'asyncio'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='WoollyAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='woolly.ico'
)

# Build with: pyinstaller woolly_app.spec
```

## Best Practices

1. **Async Operations**: Always use async for API calls to keep UI responsive
2. **Error Handling**: Gracefully handle connection errors and model failures
3. **Resource Management**: Monitor memory usage and provide feedback
4. **User Feedback**: Show progress for long operations
5. **Settings Persistence**: Save user preferences and window state
6. **Logging**: Implement proper logging for debugging
7. **Testing**: Use pytest-qt for PyQt testing
8. **Accessibility**: Ensure keyboard navigation and screen reader support