# Flutter Desktop Example with Woolly

This guide demonstrates how to build a cross-platform desktop application using Flutter with Woolly as the AI engine.

## Overview

Flutter desktop apps can integrate Woolly through:
1. **FFI (Foreign Function Interface)**: Direct Rust binding for maximum performance
2. **HTTP Client**: REST API communication for simplicity
3. **Platform Channels**: Native platform integration

## Project Setup

### 1. Create Flutter Desktop Project

```bash
flutter create woolly_flutter --platforms=windows,macos,linux
cd woolly_flutter
flutter config --enable-windows-desktop
flutter config --enable-macos-desktop  
flutter config --enable-linux-desktop
```

### 2. Add Dependencies

```yaml
# pubspec.yaml
name: woolly_flutter
description: AI-powered desktop app with Woolly

environment:
  sdk: '>=3.0.0 <4.0.0'

dependencies:
  flutter:
    sdk: flutter
  
  # HTTP client
  dio: ^5.4.0
  
  # State management
  provider: ^6.1.0
  flutter_riverpod: ^2.4.0
  
  # UI enhancements
  fluent_ui: ^4.8.0  # Windows 11 style
  macos_ui: ^2.0.0   # macOS style
  yaru: ^1.2.0       # Ubuntu style
  
  # Storage
  shared_preferences: ^2.2.0
  path_provider: ^2.1.0
  
  # FFI for native integration
  ffi: ^2.1.0
  
  # File picker
  file_picker: ^6.1.0
  
  # Markdown rendering
  flutter_markdown: ^0.6.18
  
  # Syntax highlighting
  flutter_highlight: ^0.7.0
  
  # Window management
  window_manager: ^0.3.7
  
  # System tray
  system_tray: ^2.0.3

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0
  build_runner: ^2.4.0
  freezed: ^2.4.0
  json_serializable: ^6.7.0
```

## Complete Implementation

### Models

```dart
// lib/models/chat_models.dart
import 'package:freezed_annotation/freezed_annotation.dart';

part 'chat_models.freezed.dart';
part 'chat_models.g.dart';

@freezed
class ModelInfo with _$ModelInfo {
  const factory ModelInfo({
    required String name,
    required int size,
    required String quantization,
    @Default(false) bool loaded,
  }) = _ModelInfo;

  factory ModelInfo.fromJson(Map<String, dynamic> json) =>
      _$ModelInfoFromJson(json);
}

@freezed
class ChatMessage with _$ChatMessage {
  const factory ChatMessage({
    required String role,
    required String content,
    DateTime? timestamp,
  }) = _ChatMessage;

  factory ChatMessage.fromJson(Map<String, dynamic> json) =>
      _$ChatMessageFromJson(json);
}

@freezed
class GenerationConfig with _$GenerationConfig {
  const factory GenerationConfig({
    @Default(200) int maxTokens,
    @Default(0.7) double temperature,
    @Default(0.9) double topP,
    @Default(40) int topK,
    @Default(1.1) double repeatPenalty,
  }) = _GenerationConfig;

  factory GenerationConfig.fromJson(Map<String, dynamic> json) =>
      _$GenerationConfigFromJson(json);
}

enum MessageType { user, assistant, system }

class ChatMessageDisplay {
  final String id;
  final MessageType type;
  final String content;
  final DateTime timestamp;
  final bool isStreaming;

  ChatMessageDisplay({
    required this.id,
    required this.type,
    required this.content,
    required this.timestamp,
    this.isStreaming = false,
  });

  ChatMessageDisplay copyWith({
    String? content,
    bool? isStreaming,
  }) {
    return ChatMessageDisplay(
      id: id,
      type: type,
      content: content ?? this.content,
      timestamp: timestamp,
      isStreaming: isStreaming ?? this.isStreaming,
    );
  }
}
```

### Woolly Service

```dart
// lib/services/woolly_service.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import '../models/chat_models.dart';

class WoollyService {
  final String host;
  final int port;
  final Dio _dio;
  Process? _serverProcess;
  
  WoollyService({
    this.host = 'localhost',
    this.port = 11434,
  }) : _dio = Dio(BaseOptions(
          baseUrl: 'http://$host:$port',
          connectTimeout: const Duration(seconds: 30),
          receiveTimeout: const Duration(minutes: 5),
        ));

  Future<bool> startServer() async {
    if (_serverProcess != null) {
      return await checkHealth();
    }

    try {
      final appDir = await getApplicationSupportDirectory();
      final modelsDir = Directory('${appDir.path}/models');
      if (!await modelsDir.exists()) {
        await modelsDir.create(recursive: true);
      }

      _serverProcess = await Process.start(
        'woolly-server',
        [
          '--host', '127.0.0.1',
          '--port', port.toString(),
          '--models-dir', modelsDir.path,
        ],
      );

      // Wait for server to be ready
      for (int i = 0; i < 30; i++) {
        await Future.delayed(const Duration(seconds: 1));
        if (await checkHealth()) {
          debugPrint('Woolly server started successfully');
          return true;
        }
      }

      return false;
    } catch (e) {
      debugPrint('Failed to start Woolly server: $e');
      return false;
    }
  }

  Future<void> stopServer() async {
    if (_serverProcess != null) {
      _serverProcess!.kill();
      await _serverProcess!.exitCode;
      _serverProcess = null;
    }
  }

  Future<bool> checkHealth() async {
    try {
      final response = await _dio.get('/health');
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  Future<List<ModelInfo>> listModels() async {
    try {
      final response = await _dio.get('/api/models');
      final List<dynamic> data = response.data;
      return data.map((json) => ModelInfo.fromJson(json)).toList();
    } catch (e) {
      throw Exception('Failed to list models: $e');
    }
  }

  Future<bool> loadModel(String modelName) async {
    try {
      final response = await _dio.post('/api/models/load', data: {
        'name': modelName,
      });
      return response.statusCode == 200;
    } catch (e) {
      throw Exception('Failed to load model: $e');
    }
  }

  Future<String> generate(
    String prompt, {
    GenerationConfig? config,
  }) async {
    config ??= const GenerationConfig();
    
    try {
      final response = await _dio.post('/api/generate', data: {
        'prompt': prompt,
        'max_tokens': config.maxTokens,
        'temperature': config.temperature,
        'top_p': config.topP,
        'top_k': config.topK,
        'repeat_penalty': config.repeatPenalty,
        'stream': false,
      });
      
      return response.data['text'];
    } catch (e) {
      throw Exception('Failed to generate: $e');
    }
  }

  Stream<String> generateStream(
    String prompt, {
    GenerationConfig? config,
  }) async* {
    config ??= const GenerationConfig();
    
    try {
      final response = await _dio.post<ResponseBody>(
        '/api/generate',
        data: {
          'prompt': prompt,
          'max_tokens': config.maxTokens,
          'temperature': config.temperature,
          'top_p': config.topP,
          'top_k': config.topK,
          'repeat_penalty': config.repeatPenalty,
          'stream': true,
        },
        options: Options(responseType: ResponseType.stream),
      );
      
      final stream = response.data!.stream;
      await for (final chunk in stream) {
        final lines = utf8.decode(chunk).split('\n');
        for (final line in lines) {
          if (line.startsWith('data: ')) {
            final data = line.substring(6);
            if (data == '[DONE]') return;
            
            try {
              final json = jsonDecode(data);
              if (json['token'] != null) {
                yield json['token'];
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } catch (e) {
      throw Exception('Failed to generate stream: $e');
    }
  }

  Future<String> chat(
    List<ChatMessage> messages, {
    GenerationConfig? config,
  }) async {
    config ??= const GenerationConfig();
    
    try {
      final response = await _dio.post('/api/chat', data: {
        'messages': messages.map((m) => m.toJson()).toList(),
        'max_tokens': config.maxTokens,
        'temperature': config.temperature,
        'top_p': config.topP,
        'stream': false,
      });
      
      return response.data['content'];
    } catch (e) {
      throw Exception('Failed to chat: $e');
    }
  }

  Stream<String> chatStream(
    List<ChatMessage> messages, {
    GenerationConfig? config,
  }) async* {
    config ??= const GenerationConfig();
    
    try {
      final response = await _dio.post<ResponseBody>(
        '/api/chat',
        data: {
          'messages': messages.map((m) => m.toJson()).toList(),
          'max_tokens': config.maxTokens,
          'temperature': config.temperature,
          'top_p': config.topP,
          'stream': true,
        },
        options: Options(responseType: ResponseType.stream),
      );
      
      final stream = response.data!.stream;
      await for (final chunk in stream) {
        final lines = utf8.decode(chunk).split('\n');
        for (final line in lines) {
          if (line.startsWith('data: ')) {
            final data = line.substring(6);
            if (data == '[DONE]') return;
            
            try {
              final json = jsonDecode(data);
              if (json['content'] != null) {
                yield json['content'];
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } catch (e) {
      throw Exception('Failed to chat stream: $e');
    }
  }

  Future<String> downloadModel({
    required String url,
    required String modelName,
    required Function(double progress) onProgress,
  }) async {
    final appDir = await getApplicationSupportDirectory();
    final modelsDir = Directory('${appDir.path}/models');
    final modelPath = '${modelsDir.path}/$modelName';

    await _dio.download(
      url,
      modelPath,
      onReceiveProgress: (received, total) {
        if (total > 0) {
          onProgress(received / total);
        }
      },
    );

    return modelPath;
  }
}
```

### State Management with Riverpod

```dart
// lib/providers/chat_provider.dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:uuid/uuid.dart';
import '../models/chat_models.dart';
import '../services/woolly_service.dart';

final woollyServiceProvider = Provider<WoollyService>((ref) {
  return WoollyService();
});

final serverStatusProvider = FutureProvider<bool>((ref) async {
  final service = ref.watch(woollyServiceProvider);
  return await service.checkHealth();
});

final modelsProvider = FutureProvider<List<ModelInfo>>((ref) async {
  final service = ref.watch(woollyServiceProvider);
  return await service.listModels();
});

final currentModelProvider = StateProvider<String?>((ref) => null);

final chatMessagesProvider = StateNotifierProvider<ChatMessagesNotifier, List<ChatMessageDisplay>>((ref) {
  return ChatMessagesNotifier();
});

class ChatMessagesNotifier extends StateNotifier<List<ChatMessageDisplay>> {
  ChatMessagesNotifier() : super([]);
  
  final _uuid = const Uuid();

  void addUserMessage(String content) {
    state = [
      ...state,
      ChatMessageDisplay(
        id: _uuid.v4(),
        type: MessageType.user,
        content: content,
        timestamp: DateTime.now(),
      ),
    ];
  }

  String addAssistantMessage({String content = '', bool isStreaming = true}) {
    final id = _uuid.v4();
    state = [
      ...state,
      ChatMessageDisplay(
        id: id,
        type: MessageType.assistant,
        content: content,
        timestamp: DateTime.now(),
        isStreaming: isStreaming,
      ),
    ];
    return id;
  }

  void updateMessage(String id, String content, {bool? isStreaming}) {
    state = state.map((message) {
      if (message.id == id) {
        return message.copyWith(
          content: content,
          isStreaming: isStreaming,
        );
      }
      return message;
    }).toList();
  }

  void addSystemMessage(String content) {
    state = [
      ...state,
      ChatMessageDisplay(
        id: _uuid.v4(),
        type: MessageType.system,
        content: content,
        timestamp: DateTime.now(),
      ),
    ];
  }

  void clear() {
    state = [];
  }
}

final isGeneratingProvider = StateProvider<bool>((ref) => false);

final generationStatsProvider = StateProvider<GenerationStats>((ref) {
  return GenerationStats(tokensPerSecond: 0, totalTokens: 0);
});

class GenerationStats {
  final double tokensPerSecond;
  final int totalTokens;

  GenerationStats({
    required this.tokensPerSecond,
    required this.totalTokens,
  });
}
```

### Main Application

```dart
// lib/main.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:window_manager/window_manager.dart';
import 'screens/chat_screen.dart';
import 'services/woolly_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize window manager
  await windowManager.ensureInitialized();
  
  WindowOptions windowOptions = const WindowOptions(
    size: Size(1200, 800),
    center: true,
    backgroundColor: Colors.transparent,
    skipTaskbar: false,
    titleBarStyle: TitleBarStyle.normal,
    title: 'Woolly AI Assistant',
  );
  
  windowManager.waitUntilReadyToShow(windowOptions, () async {
    await windowManager.show();
    await windowManager.focus();
  });

  runApp(const ProviderScope(child: WoollyApp()));
}

class WoollyApp extends ConsumerWidget {
  const WoollyApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return MaterialApp(
      title: 'Woolly AI Assistant',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
        fontFamily: getPlatformFont(),
      ),
      home: const AppShell(),
    );
  }
  
  String getPlatformFont() {
    switch (Theme.of(context).platform) {
      case TargetPlatform.macOS:
        return 'SF Pro Display';
      case TargetPlatform.windows:
        return 'Segoe UI';
      case TargetPlatform.linux:
        return 'Ubuntu';
      default:
        return 'Roboto';
    }
  }
}

class AppShell extends ConsumerStatefulWidget {
  const AppShell({super.key});

  @override
  ConsumerState<AppShell> createState() => _AppShellState();
}

class _AppShellState extends ConsumerState<AppShell> with WindowListener {
  @override
  void initState() {
    super.initState();
    windowManager.addListener(this);
    _initializeApp();
  }

  @override
  void dispose() {
    windowManager.removeListener(this);
    super.dispose();
  }

  Future<void> _initializeApp() async {
    final service = ref.read(woollyServiceProvider);
    await service.startServer();
  }

  @override
  void onWindowClose() async {
    final service = ref.read(woollyServiceProvider);
    await service.stopServer();
  }

  @override
  Widget build(BuildContext context) {
    return const ChatScreen();
  }
}
```

### Chat Screen UI

```dart
// lib/screens/chat_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:flutter_highlight/flutter_highlight.dart';
import 'package:flutter_highlight/themes/github-dark.dart';
import 'package:file_picker/file_picker.dart';
import '../models/chat_models.dart';
import '../providers/chat_provider.dart';
import '../widgets/message_bubble.dart';

class ChatScreen extends ConsumerStatefulWidget {
  const ChatScreen({super.key});

  @override
  ConsumerState<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends ConsumerState<ChatScreen> {
  final _messageController = TextEditingController();
  final _scrollController = ScrollController();
  final _focusNode = FocusNode();
  
  @override
  void dispose() {
    _messageController.dispose();
    _scrollController.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final serverStatus = ref.watch(serverStatusProvider);
    final models = ref.watch(modelsProvider);
    final currentModel = ref.watch(currentModelProvider);
    final messages = ref.watch(chatMessagesProvider);
    final isGenerating = ref.watch(isGeneratingProvider);
    final stats = ref.watch(generationStatsProvider);

    return Scaffold(
      backgroundColor: const Color(0xFF1E1E1E),
      body: Column(
        children: [
          // Header
          Container(
            height: 60,
            decoration: BoxDecoration(
              color: const Color(0xFF2D2D2D),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.2),
                  blurRadius: 4,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  // App title
                  const Text(
                    'Woolly AI Assistant',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const Spacer(),
                  
                  // Model selector
                  models.when(
                    data: (modelList) => SizedBox(
                      width: 250,
                      child: DropdownButtonFormField<String>(
                        value: currentModel,
                        decoration: const InputDecoration(
                          isDense: true,
                          contentPadding: EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 8,
                          ),
                          border: OutlineInputBorder(),
                          fillColor: Color(0xFF333333),
                          filled: true,
                        ),
                        dropdownColor: const Color(0xFF333333),
                        items: modelList.map((model) {
                          return DropdownMenuItem(
                            value: model.name,
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  model.name,
                                  style: const TextStyle(color: Colors.white),
                                ),
                                if (model.loaded)
                                  const Icon(
                                    Icons.check_circle,
                                    color: Colors.green,
                                    size: 16,
                                  ),
                              ],
                            ),
                          );
                        }).toList(),
                        onChanged: (value) {
                          if (value != null) {
                            ref.read(currentModelProvider.notifier).state = value;
                          }
                        },
                      ),
                    ),
                    loading: () => const CircularProgressIndicator(),
                    error: (e, s) => const Text('Error loading models'),
                  ),
                  
                  const SizedBox(width: 8),
                  
                  // Load model button
                  FilledButton.icon(
                    onPressed: currentModel == null ? null : _loadModel,
                    icon: const Icon(Icons.download),
                    label: const Text('Load Model'),
                  ),
                  
                  const SizedBox(width: 8),
                  
                  // Upload model button
                  IconButton(
                    onPressed: _uploadModel,
                    icon: const Icon(Icons.upload_file),
                    tooltip: 'Upload Model',
                  ),
                  
                  const SizedBox(width: 8),
                  
                  // Server status indicator
                  serverStatus.when(
                    data: (isRunning) => Container(
                      width: 12,
                      height: 12,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: isRunning ? Colors.green : Colors.red,
                      ),
                    ),
                    loading: () => Container(
                      width: 12,
                      height: 12,
                      decoration: const BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.orange,
                      ),
                    ),
                    error: (e, s) => Container(
                      width: 12,
                      height: 12,
                      decoration: const BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.red,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          
          // Chat messages
          Expanded(
            child: Container(
              color: const Color(0xFF1E1E1E),
              child: messages.isEmpty
                  ? Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.chat_bubble_outline,
                            size: 64,
                            color: Colors.grey[600],
                          ),
                          const SizedBox(height: 16),
                          Text(
                            'Start a conversation',
                            style: TextStyle(
                              fontSize: 18,
                              color: Colors.grey[600],
                            ),
                          ),
                        ],
                      ),
                    )
                  : ListView.builder(
                      controller: _scrollController,
                      padding: const EdgeInsets.all(16),
                      itemCount: messages.length,
                      itemBuilder: (context, index) {
                        final message = messages[index];
                        return MessageBubble(
                          message: message,
                          onCopy: () => _copyToClipboard(message.content),
                        );
                      },
                    ),
            ),
          ),
          
          // Input area
          Container(
            decoration: BoxDecoration(
              color: const Color(0xFF2D2D2D),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.2),
                  blurRadius: 4,
                  offset: const Offset(0, -2),
                ),
              ],
            ),
            padding: const EdgeInsets.all(16),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                // Message input
                Expanded(
                  child: Container(
                    constraints: const BoxConstraints(maxHeight: 120),
                    child: TextField(
                      controller: _messageController,
                      focusNode: _focusNode,
                      maxLines: null,
                      keyboardType: TextInputType.multiline,
                      textInputAction: TextInputAction.newline,
                      enabled: !isGenerating && currentModel != null,
                      decoration: InputDecoration(
                        hintText: 'Type your message...',
                        fillColor: const Color(0xFF333333),
                        filled: true,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8),
                          borderSide: BorderSide.none,
                        ),
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 12,
                        ),
                      ),
                      onSubmitted: (_) {
                        if (!HardwareKeyboard.instance.isShiftPressed) {
                          _sendMessage();
                        }
                      },
                    ),
                  ),
                ),
                
                const SizedBox(width: 8),
                
                // Send button
                FilledButton.icon(
                  onPressed: (isGenerating || currentModel == null) ? null : _sendMessage,
                  icon: Icon(isGenerating ? Icons.stop : Icons.send),
                  label: Text(isGenerating ? 'Stop' : 'Send'),
                  style: FilledButton.styleFrom(
                    minimumSize: const Size(100, 48),
                  ),
                ),
              ],
            ),
          ),
          
          // Status bar
          if (isGenerating || stats.tokensPerSecond > 0)
            Container(
              height: 30,
              color: const Color(0xFF252525),
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  if (isGenerating)
                    const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  const SizedBox(width: 8),
                  Text(
                    'Generating...',
                    style: TextStyle(
                      color: Colors.grey[400],
                      fontSize: 12,
                    ),
                  ),
                  const Spacer(),
                  if (stats.tokensPerSecond > 0)
                    Text(
                      '${stats.tokensPerSecond.toStringAsFixed(1)} tokens/s',
                      style: TextStyle(
                        color: Colors.grey[400],
                        fontSize: 12,
                      ),
                    ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Future<void> _loadModel() async {
    final model = ref.read(currentModelProvider);
    if (model == null) return;

    try {
      final service = ref.read(woollyServiceProvider);
      final success = await service.loadModel(model);
      
      if (success) {
        ref.read(chatMessagesProvider.notifier).addSystemMessage(
          'Model "$model" loaded successfully',
        );
        
        // Refresh models list
        ref.invalidate(modelsProvider);
      } else {
        _showError('Failed to load model');
      }
    } catch (e) {
      _showError('Error loading model: $e');
    }
  }

  Future<void> _uploadModel() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['gguf'],
    );

    if (result != null && result.files.single.path != null) {
      final file = result.files.single;
      final path = file.path!;
      
      // TODO: Copy file to models directory
      ref.read(chatMessagesProvider.notifier).addSystemMessage(
        'Model uploaded: ${file.name}',
      );
      
      // Refresh models list
      ref.invalidate(modelsProvider);
    }
  }

  Future<void> _sendMessage() async {
    final message = _messageController.text.trim();
    if (message.isEmpty) return;

    _messageController.clear();
    _focusNode.requestFocus();

    // Add user message
    ref.read(chatMessagesProvider.notifier).addUserMessage(message);
    
    // Scroll to bottom
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    });

    // Start generation
    ref.read(isGeneratingProvider.notifier).state = true;
    
    // Add assistant message placeholder
    final assistantId = ref.read(chatMessagesProvider.notifier)
        .addAssistantMessage(isStreaming: true);

    try {
      final service = ref.read(woollyServiceProvider);
      final messages = ref.read(chatMessagesProvider)
          .where((m) => m.type != MessageType.system)
          .take(10)
          .map((m) => ChatMessage(
                role: m.type == MessageType.user ? 'user' : 'assistant',
                content: m.content,
              ))
          .toList();
      
      messages.add(ChatMessage(role: 'user', content: message));

      String fullResponse = '';
      int tokenCount = 0;
      final startTime = DateTime.now();

      await for (final token in service.chatStream(messages)) {
        fullResponse += token;
        tokenCount++;

        // Update message
        ref.read(chatMessagesProvider.notifier).updateMessage(
          assistantId,
          fullResponse,
          isStreaming: true,
        );

        // Update stats periodically
        if (tokenCount % 5 == 0) {
          final elapsed = DateTime.now().difference(startTime).inMilliseconds / 1000.0;
          final tokensPerSecond = tokenCount / elapsed;
          
          ref.read(generationStatsProvider.notifier).state = GenerationStats(
            tokensPerSecond: tokensPerSecond,
            totalTokens: tokenCount,
          );
        }

        // Auto scroll
        if (_scrollController.hasClients) {
          _scrollController.animateTo(
            _scrollController.position.maxScrollExtent,
            duration: const Duration(milliseconds: 100),
            curve: Curves.easeOut,
          );
        }
      }

      // Finalize message
      ref.read(chatMessagesProvider.notifier).updateMessage(
        assistantId,
        fullResponse,
        isStreaming: false,
      );
    } catch (e) {
      ref.read(chatMessagesProvider.notifier).updateMessage(
        assistantId,
        'Error: $e',
        isStreaming: false,
      );
    } finally {
      ref.read(isGeneratingProvider.notifier).state = false;
      ref.read(generationStatsProvider.notifier).state = GenerationStats(
        tokensPerSecond: 0,
        totalTokens: 0,
      );
    }
  }

  void _copyToClipboard(String text) {
    Clipboard.setData(ClipboardData(text: text));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Copied to clipboard'),
        duration: Duration(seconds: 2),
      ),
    );
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
      ),
    );
  }
}
```

### Message Bubble Widget

```dart
// lib/widgets/message_bubble.dart
import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:flutter_highlight/flutter_highlight.dart';
import 'package:flutter_highlight/themes/monokai-sublime.dart';
import '../models/chat_models.dart';

class MessageBubble extends StatelessWidget {
  final ChatMessageDisplay message;
  final VoidCallback? onCopy;

  const MessageBubble({
    super.key,
    required this.message,
    this.onCopy,
  });

  @override
  Widget build(BuildContext context) {
    final isUser = message.type == MessageType.user;
    final isSystem = message.type == MessageType.system;

    if (isSystem) {
      return Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: const Color(0xFF555555),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(
          message.content,
          style: TextStyle(
            color: Colors.grey[300],
            fontSize: 12,
          ),
          textAlign: TextAlign.center,
        ),
      );
    }

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!isUser) ...[
            CircleAvatar(
              radius: 20,
              backgroundColor: const Color(0xFF10B981),
              child: const Icon(
                Icons.smart_toy,
                color: Colors.white,
                size: 20,
              ),
            ),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Container(
              constraints: BoxConstraints(
                maxWidth: MediaQuery.of(context).size.width * 0.7,
              ),
              decoration: BoxDecoration(
                color: isUser ? const Color(0xFF0084FF) : const Color(0xFF333333),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Stack(
                children: [
                  Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        if (!isUser && message.isStreaming)
                          Row(
                            children: [
                              const SizedBox(
                                width: 12,
                                height: 12,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white70,
                                ),
                              ),
                              const SizedBox(width: 8),
                              Text(
                                'Generating...',
                                style: TextStyle(
                                  color: Colors.grey[400],
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        if (isUser)
                          Text(
                            message.content,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                            ),
                          )
                        else
                          MarkdownBody(
                            data: message.content,
                            selectable: true,
                            styleSheet: MarkdownStyleSheet(
                              p: const TextStyle(
                                color: Colors.white,
                                fontSize: 14,
                              ),
                              code: TextStyle(
                                backgroundColor: Colors.black.withOpacity(0.3),
                                fontFamily: 'monospace',
                              ),
                              codeblockDecoration: BoxDecoration(
                                color: Colors.black.withOpacity(0.3),
                                borderRadius: BorderRadius.circular(8),
                              ),
                            ),
                            builders: {
                              'code': CodeBlockBuilder(),
                            },
                          ),
                        if (message.isStreaming)
                          const Text(
                            '▊',
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 14,
                            ),
                          ),
                      ],
                    ),
                  ),
                  if (!isUser && !message.isStreaming)
                    Positioned(
                      top: 4,
                      right: 4,
                      child: IconButton(
                        icon: const Icon(
                          Icons.copy,
                          size: 16,
                          color: Colors.white54,
                        ),
                        onPressed: onCopy,
                        padding: EdgeInsets.zero,
                        constraints: const BoxConstraints(),
                        splashRadius: 16,
                      ),
                    ),
                ],
              ),
            ),
          ),
          if (isUser) ...[
            const SizedBox(width: 8),
            CircleAvatar(
              radius: 20,
              backgroundColor: const Color(0xFF6366F1),
              child: const Icon(
                Icons.person,
                color: Colors.white,
                size: 20,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class CodeBlockBuilder extends MarkdownElementBuilder {
  @override
  Widget? visitElementAfter(element, TextStyle? preferredStyle) {
    if (element.tag == 'pre') {
      final code = element.textContent;
      final language = element.children?.first.attributes['class']
          ?.replaceAll('language-', '') ?? 'plaintext';

      return Container(
        margin: const EdgeInsets.symmetric(vertical: 8),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.8),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.3),
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(8),
                  topRight: Radius.circular(8),
                ),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    language,
                    style: const TextStyle(
                      color: Colors.white60,
                      fontSize: 12,
                    ),
                  ),
                  IconButton(
                    icon: const Icon(
                      Icons.copy,
                      size: 16,
                      color: Colors.white60,
                    ),
                    onPressed: () {
                      // Copy code to clipboard
                    },
                    padding: EdgeInsets.zero,
                    constraints: const BoxConstraints(),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(12),
              child: HighlightView(
                code,
                language: language,
                theme: monokaiSublimeTheme,
                textStyle: const TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 13,
                ),
              ),
            ),
          ],
        ),
      );
    }
    return null;
  }
}
```

## Platform-Specific Configuration

### Windows

```yaml
# windows/runner/Runner.rc
// Add app icon
IDI_APP_ICON ICON "resources\\app_icon.ico"

// Version info
VS_VERSION_INFO VERSIONINFO
 FILEVERSION 1,0,0,0
 PRODUCTVERSION 1,0,0,0
 FILEFLAGSMASK 0x3fL
 FILEFLAGS 0x0L
 FILEOS 0x40004L
 FILETYPE 0x1L
 FILESUBTYPE 0x0L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904b0"
        BEGIN
            VALUE "CompanyName", "Your Company"
            VALUE "FileDescription", "Woolly AI Assistant"
            VALUE "FileVersion", "1.0.0.0"
            VALUE "InternalName", "woolly_flutter"
            VALUE "LegalCopyright", "Copyright (C) 2024"
            VALUE "OriginalFilename", "woolly_flutter.exe"
            VALUE "ProductName", "Woolly AI Assistant"
            VALUE "ProductVersion", "1.0.0.0"
        END
    END
END
```

### macOS

```swift
// macos/Runner/AppDelegate.swift
import Cocoa
import FlutterMacOS

@NSApplicationMain
class AppDelegate: FlutterAppDelegate {
  override func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
    return true
  }
  
  override func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool {
    return true
  }
}
```

```xml
<!-- macos/Runner/Info.plist -->
<key>LSMinimumSystemVersion</key>
<string>10.14</string>
<key>NSHumanReadableCopyright</key>
<string>Copyright © 2024 Your Company. All rights reserved.</string>
```

### Linux

```cmake
# linux/CMakeLists.txt
# Add after project()
set(APPLICATION_ID "com.yourcompany.woolly_flutter")

# Add icon
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/assets/icon.png" 
        DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pixmaps"
        RENAME "${APPLICATION_ID}.png")

# Add desktop entry
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/woolly.desktop.in"
               "${CMAKE_CURRENT_BINARY_DIR}/woolly.desktop" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/woolly.desktop"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/share/applications")
```

## Advanced Features

### 1. FFI Integration

```dart
// lib/ffi/woolly_ffi.dart
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

typedef WoollyEngineNewNative = Pointer<Void> Function(Pointer<WoollyConfig>);
typedef WoollyEngineNew = Pointer<Void> Function(Pointer<WoollyConfig>);

typedef WoollyGenerateNative = Pointer<Utf8> Function(
  Pointer<Void> engine,
  Pointer<Utf8> prompt,
  Int32 maxTokens,
);
typedef WoollyGenerate = Pointer<Utf8> Function(
  Pointer<Void> engine,
  Pointer<Utf8> prompt,
  int maxTokens,
);

class WoollyConfig extends Struct {
  @Int32()
  external int maxBatchSize;
  
  @Int32()
  external int maxSequenceLength;
  
  @Bool()
  external bool enableGpu;
}

class WoollyFFI {
  late final DynamicLibrary _lib;
  late final WoollyEngineNew _engineNew;
  late final WoollyGenerate _generate;
  
  WoollyFFI() {
    _lib = _loadLibrary();
    _engineNew = _lib.lookupFunction<WoollyEngineNewNative, WoollyEngineNew>(
      'woolly_engine_new',
    );
    _generate = _lib.lookupFunction<WoollyGenerateNative, WoollyGenerate>(
      'woolly_generate',
    );
  }
  
  DynamicLibrary _loadLibrary() {
    if (Platform.isWindows) {
      return DynamicLibrary.open('woolly_core.dll');
    } else if (Platform.isMacOS) {
      return DynamicLibrary.open('libwoolly_core.dylib');
    } else if (Platform.isLinux) {
      return DynamicLibrary.open('libwoolly_core.so');
    }
    throw UnsupportedError('Platform not supported');
  }
  
  Pointer<Void> createEngine({
    int maxBatchSize = 1,
    int maxSequenceLength = 2048,
    bool enableGpu = true,
  }) {
    final config = calloc<WoollyConfig>();
    config.ref.maxBatchSize = maxBatchSize;
    config.ref.maxSequenceLength = maxSequenceLength;
    config.ref.enableGpu = enableGpu;
    
    final engine = _engineNew(config);
    calloc.free(config);
    
    return engine;
  }
  
  String generate(Pointer<Void> engine, String prompt, int maxTokens) {
    final promptPtr = prompt.toNativeUtf8();
    final resultPtr = _generate(engine, promptPtr, maxTokens);
    
    final result = resultPtr.toDartString();
    calloc.free(promptPtr);
    calloc.free(resultPtr);
    
    return result;
  }
}
```

### 2. System Tray Support

```dart
// lib/system_tray.dart
import 'package:system_tray/system_tray.dart';

class WoollySystemTray {
  final SystemTray _systemTray = SystemTray();
  final Menu _menu = Menu();
  
  Future<void> initialize() async {
    // Initialize system tray
    await _systemTray.initSystemTray(
      title: 'Woolly AI',
      iconPath: _getIconPath(),
      toolTip: 'Woolly AI Assistant',
    );
    
    // Create menu
    await _menu.buildFrom([
      MenuItemLabel(
        label: 'Show',
        onClicked: (menuItem) => _showWindow(),
      ),
      MenuItemLabel(
        label: 'Settings',
        onClicked: (menuItem) => _showSettings(),
      ),
      MenuSeparator(),
      MenuItemLabel(
        label: 'Quit',
        onClicked: (menuItem) => _quit(),
      ),
    ]);
    
    await _systemTray.setContextMenu(_menu);
    
    // Handle clicks
    _systemTray.registerSystemTrayEventHandler((eventName) {
      if (eventName == kSystemTrayEventClick) {
        _showWindow();
      }
    });
  }
  
  String _getIconPath() {
    if (Platform.isWindows) {
      return 'assets/icon.ico';
    } else if (Platform.isMacOS) {
      return 'assets/icon.png';
    } else {
      return 'assets/icon.png';
    }
  }
  
  void _showWindow() {
    // Show main window
    windowManager.show();
    windowManager.focus();
  }
  
  void _showSettings() {
    // Show settings dialog
  }
  
  void _quit() {
    // Quit application
    exit(0);
  }
}
```

### 3. Settings Management

```dart
// lib/services/settings_service.dart
import 'package:shared_preferences/shared_preferences.dart';

class SettingsService {
  static const String _keyTheme = 'theme';
  static const String _keyModelPath = 'model_path';
  static const String _keyTemperature = 'temperature';
  static const String _keyMaxTokens = 'max_tokens';
  
  final SharedPreferences _prefs;
  
  SettingsService(this._prefs);
  
  static Future<SettingsService> create() async {
    final prefs = await SharedPreferences.getInstance();
    return SettingsService(prefs);
  }
  
  // Theme
  String get theme => _prefs.getString(_keyTheme) ?? 'dark';
  set theme(String value) => _prefs.setString(_keyTheme, value);
  
  // Model path
  String? get modelPath => _prefs.getString(_keyModelPath);
  set modelPath(String? value) {
    if (value != null) {
      _prefs.setString(_keyModelPath, value);
    } else {
      _prefs.remove(_keyModelPath);
    }
  }
  
  // Generation settings
  double get temperature => _prefs.getDouble(_keyTemperature) ?? 0.7;
  set temperature(double value) => _prefs.setDouble(_keyTemperature, value);
  
  int get maxTokens => _prefs.getInt(_keyMaxTokens) ?? 200;
  set maxTokens(int value) => _prefs.setInt(_keyMaxTokens, value);
}
```

## Building and Distribution

### Windows
```bash
flutter build windows --release
# Output: build/windows/runner/Release/
```

### macOS
```bash
flutter build macos --release
# Create DMG for distribution
create-dmg build/macos/Build/Products/Release/woolly_flutter.app
```

### Linux
```bash
flutter build linux --release
# Create AppImage
./create-appimage.sh
```

## Best Practices

1. **State Management**: Use Riverpod for reactive state management
2. **Platform Adaptation**: Adapt UI to platform conventions
3. **Performance**: Use isolates for heavy computations
4. **Error Handling**: Implement comprehensive error handling
5. **Testing**: Write widget and integration tests
6. **Accessibility**: Support screen readers and keyboard navigation
7. **Theming**: Support light/dark themes
8. **Localization**: Prepare for internationalization
9. **Updates**: Implement auto-update mechanism
10. **Analytics**: Add privacy-respecting analytics