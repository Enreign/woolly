# .NET Integration Guide (WPF/MAUI)

This guide demonstrates how to integrate Woolly into .NET desktop applications using WPF and .NET MAUI.

## Overview

.NET applications can integrate Woolly through:
1. **P/Invoke**: Direct Rust library binding (fastest)
2. **HTTP Client**: REST API communication (simplest)
3. **gRPC**: High-performance RPC (recommended for production)

## Prerequisites

```xml
<!-- Add to your .csproj file -->
<ItemGroup>
  <PackageReference Include="System.Net.Http.Json" Version="8.0.0" />
  <PackageReference Include="Microsoft.Extensions.Hosting" Version="8.0.0" />
  <PackageReference Include="CommunityToolkit.Mvvm" Version="8.2.2" />
  <PackageReference Include="Grpc.Net.Client" Version="2.59.0" />
</ItemGroup>
```

## Woolly Client Implementation

### Base Client Class

```csharp
// WoollyClient.cs
using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace WoollyDesktop.Services;

public record ModelInfo(
    string Name,
    long Size,
    string Quantization,
    bool Loaded
);

public record GenerationConfig(
    int MaxTokens = 200,
    float Temperature = 0.7f,
    float TopP = 0.9f,
    int TopK = 40,
    float RepeatPenalty = 1.1f
);

public record ChatMessage(
    string Role,
    string Content
);

public interface IWoollyClient : IDisposable
{
    Task<bool> CheckHealthAsync(CancellationToken cancellationToken = default);
    Task<IEnumerable<ModelInfo>> ListModelsAsync(CancellationToken cancellationToken = default);
    Task<bool> LoadModelAsync(string modelName, CancellationToken cancellationToken = default);
    Task<string> GenerateAsync(string prompt, GenerationConfig? config = null, CancellationToken cancellationToken = default);
    IAsyncEnumerable<string> GenerateStreamAsync(string prompt, GenerationConfig? config = null, CancellationToken cancellationToken = default);
    Task<string> ChatAsync(IEnumerable<ChatMessage> messages, GenerationConfig? config = null, CancellationToken cancellationToken = default);
    IAsyncEnumerable<string> ChatStreamAsync(IEnumerable<ChatMessage> messages, GenerationConfig? config = null, CancellationToken cancellationToken = default);
}

public class WoollyHttpClient : IWoollyClient
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    private Process? _serverProcess;
    
    public WoollyHttpClient(string host = "localhost", int port = 11434)
    {
        _baseUrl = $"http://{host}:{port}";
        _httpClient = new HttpClient
        {
            BaseAddress = new Uri(_baseUrl),
            Timeout = TimeSpan.FromMinutes(5)
        };
    }
    
    public bool StartServer(string? modelsDirectory = null)
    {
        if (_serverProcess?.HasExited == false)
            return true;
            
        var startInfo = new ProcessStartInfo
        {
            FileName = "woolly-server",
            Arguments = $"--host 127.0.0.1 --port {new Uri(_baseUrl).Port}",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        
        if (!string.IsNullOrEmpty(modelsDirectory))
        {
            startInfo.Arguments += $" --models-dir \"{modelsDirectory}\"";
        }
        
        try
        {
            _serverProcess = Process.Start(startInfo);
            
            // Wait for server to be ready
            for (int i = 0; i < 30; i++)
            {
                Thread.Sleep(1000);
                if (CheckHealthAsync().Result)
                    return true;
            }
            
            return false;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Failed to start Woolly server: {ex.Message}");
            return false;
        }
    }
    
    public void StopServer()
    {
        if (_serverProcess?.HasExited == false)
        {
            _serverProcess.Kill();
            _serverProcess.WaitForExit();
            _serverProcess.Dispose();
            _serverProcess = null;
        }
    }
    
    public async Task<bool> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var response = await _httpClient.GetAsync("/health", cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }
    
    public async Task<IEnumerable<ModelInfo>> ListModelsAsync(CancellationToken cancellationToken = default)
    {
        var response = await _httpClient.GetFromJsonAsync<List<ModelInfo>>("/api/models", cancellationToken);
        return response ?? Enumerable.Empty<ModelInfo>();
    }
    
    public async Task<bool> LoadModelAsync(string modelName, CancellationToken cancellationToken = default)
    {
        var response = await _httpClient.PostAsJsonAsync("/api/models/load", 
            new { name = modelName }, cancellationToken);
        return response.IsSuccessStatusCode;
    }
    
    public async Task<string> GenerateAsync(string prompt, GenerationConfig? config = null, CancellationToken cancellationToken = default)
    {
        config ??= new GenerationConfig();
        
        var response = await _httpClient.PostAsJsonAsync("/api/generate", new
        {
            prompt,
            max_tokens = config.MaxTokens,
            temperature = config.Temperature,
            top_p = config.TopP,
            top_k = config.TopK,
            repeat_penalty = config.RepeatPenalty,
            stream = false
        }, cancellationToken);
        
        response.EnsureSuccessStatusCode();
        var result = await response.Content.ReadFromJsonAsync<JsonDocument>(cancellationToken);
        return result?.RootElement.GetProperty("text").GetString() ?? string.Empty;
    }
    
    public async IAsyncEnumerable<string> GenerateStreamAsync(
        string prompt, 
        GenerationConfig? config = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        config ??= new GenerationConfig();
        
        var response = await _httpClient.PostAsJsonAsync("/api/generate", new
        {
            prompt,
            max_tokens = config.MaxTokens,
            temperature = config.Temperature,
            top_p = config.TopP,
            top_k = config.TopK,
            repeat_penalty = config.RepeatPenalty,
            stream = true
        }, cancellationToken);
        
        response.EnsureSuccessStatusCode();
        
        using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);
        
        while (!reader.EndOfStream && !cancellationToken.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync();
            if (string.IsNullOrEmpty(line)) continue;
            
            if (line.StartsWith("data: "))
            {
                var data = line.Substring(6);
                if (data == "[DONE]") yield break;
                
                try
                {
                    var json = JsonDocument.Parse(data);
                    if (json.RootElement.TryGetProperty("token", out var token))
                    {
                        yield return token.GetString() ?? string.Empty;
                    }
                }
                catch (JsonException)
                {
                    // Skip invalid JSON
                }
            }
        }
    }
    
    public async Task<string> ChatAsync(
        IEnumerable<ChatMessage> messages, 
        GenerationConfig? config = null, 
        CancellationToken cancellationToken = default)
    {
        config ??= new GenerationConfig();
        
        var response = await _httpClient.PostAsJsonAsync("/api/chat", new
        {
            messages = messages.Select(m => new { role = m.Role, content = m.Content }),
            max_tokens = config.MaxTokens,
            temperature = config.Temperature,
            top_p = config.TopP,
            stream = false
        }, cancellationToken);
        
        response.EnsureSuccessStatusCode();
        var result = await response.Content.ReadFromJsonAsync<JsonDocument>(cancellationToken);
        return result?.RootElement.GetProperty("content").GetString() ?? string.Empty;
    }
    
    public async IAsyncEnumerable<string> ChatStreamAsync(
        IEnumerable<ChatMessage> messages,
        GenerationConfig? config = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        config ??= new GenerationConfig();
        
        var response = await _httpClient.PostAsJsonAsync("/api/chat", new
        {
            messages = messages.Select(m => new { role = m.Role, content = m.Content }),
            max_tokens = config.MaxTokens,
            temperature = config.Temperature,
            top_p = config.TopP,
            stream = true
        }, cancellationToken);
        
        response.EnsureSuccessStatusCode();
        
        using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);
        
        while (!reader.EndOfStream && !cancellationToken.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync();
            if (string.IsNullOrEmpty(line)) continue;
            
            if (line.StartsWith("data: "))
            {
                var data = line.Substring(6);
                if (data == "[DONE]") yield break;
                
                try
                {
                    var json = JsonDocument.Parse(data);
                    if (json.RootElement.TryGetProperty("content", out var content))
                    {
                        yield return content.GetString() ?? string.Empty;
                    }
                }
                catch (JsonException)
                {
                    // Skip invalid JSON
                }
            }
        }
    }
    
    public void Dispose()
    {
        StopServer();
        _httpClient?.Dispose();
    }
}
```

## WPF Implementation

### MainWindow.xaml

```xml
<Window x:Class="WoollyWpf.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        xmlns:d="http://schemas.microsoft.com/expression/blend/2008"
        xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
        mc:Ignorable="d"
        Title="Woolly AI Assistant" Height="800" Width="1200"
        WindowStartupLocation="CenterScreen">
    
    <Window.Resources>
        <!-- Modern Dark Theme -->
        <Style TargetType="Window">
            <Setter Property="Background" Value="#1e1e1e"/>
            <Setter Property="Foreground" Value="White"/>
        </Style>
        
        <Style TargetType="Button">
            <Setter Property="Background" Value="#0084ff"/>
            <Setter Property="Foreground" Value="White"/>
            <Setter Property="BorderThickness" Value="0"/>
            <Setter Property="Padding" Value="12,8"/>
            <Setter Property="Cursor" Value="Hand"/>
            <Setter Property="FontWeight" Value="Medium"/>
            <Setter Property="Template">
                <Setter.Value>
                    <ControlTemplate TargetType="Button">
                        <Border Background="{TemplateBinding Background}"
                                CornerRadius="4"
                                Padding="{TemplateBinding Padding}">
                            <ContentPresenter HorizontalAlignment="Center"
                                            VerticalAlignment="Center"/>
                        </Border>
                        <ControlTemplate.Triggers>
                            <Trigger Property="IsMouseOver" Value="True">
                                <Setter Property="Background" Value="#0073e6"/>
                            </Trigger>
                            <Trigger Property="IsPressed" Value="True">
                                <Setter Property="Background" Value="#005bb5"/>
                            </Trigger>
                            <Trigger Property="IsEnabled" Value="False">
                                <Setter Property="Background" Value="#555"/>
                                <Setter Property="Foreground" Value="#888"/>
                            </Trigger>
                        </ControlTemplate.Triggers>
                    </ControlTemplate>
                </Setter.Value>
            </Setter>
        </Style>
        
        <Style TargetType="ComboBox">
            <Setter Property="Background" Value="#2d2d2d"/>
            <Setter Property="Foreground" Value="White"/>
            <Setter Property="BorderBrush" Value="#555"/>
            <Setter Property="Padding" Value="8,4"/>
        </Style>
        
        <Style TargetType="TextBox">
            <Setter Property="Background" Value="#2d2d2d"/>
            <Setter Property="Foreground" Value="White"/>
            <Setter Property="BorderBrush" Value="#555"/>
            <Setter Property="CaretBrush" Value="White"/>
            <Setter Property="Padding" Value="8"/>
        </Style>
    </Window.Resources>
    
    <Grid>
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="*"/>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="Auto"/>
        </Grid.RowDefinitions>
        
        <!-- Header -->
        <Border Grid.Row="0" Background="#2d2d2d" Padding="16">
            <Grid>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="Auto"/>
                    <ColumnDefinition Width="*"/>
                    <ColumnDefinition Width="Auto"/>
                </Grid.ColumnDefinitions>
                
                <TextBlock Grid.Column="0" 
                          Text="Woolly AI Assistant" 
                          FontSize="24" 
                          FontWeight="Bold"
                          VerticalAlignment="Center"/>
                
                <StackPanel Grid.Column="2" 
                           Orientation="Horizontal" 
                           HorizontalAlignment="Right">
                    <ComboBox x:Name="ModelComboBox"
                             Width="200"
                             Margin="0,0,8,0"
                             SelectionChanged="ModelComboBox_SelectionChanged"/>
                    
                    <Button x:Name="LoadModelButton"
                           Content="Load Model"
                           Click="LoadModelButton_Click"
                           Margin="0,0,8,0"/>
                    
                    <Button x:Name="UploadModelButton"
                           Content="Upload Model"
                           Click="UploadModelButton_Click"/>
                </StackPanel>
            </Grid>
        </Border>
        
        <!-- Chat Area -->
        <ScrollViewer Grid.Row="1" 
                     x:Name="ChatScrollViewer"
                     Background="#1e1e1e"
                     Padding="16"
                     VerticalScrollBarVisibility="Auto">
            <ItemsControl x:Name="ChatItemsControl">
                <ItemsControl.ItemTemplate>
                    <DataTemplate>
                        <Border Margin="0,8"
                               Padding="12"
                               CornerRadius="8"
                               HorizontalAlignment="{Binding Alignment}">
                            <Border.Background>
                                <SolidColorBrush Color="{Binding BackgroundColor}"/>
                            </Border.Background>
                            <StackPanel>
                                <TextBlock Text="{Binding Header}"
                                          FontSize="12"
                                          Foreground="#888"
                                          Margin="0,0,0,4"/>
                                <TextBlock Text="{Binding Content}"
                                          TextWrapping="Wrap"
                                          MaxWidth="600"/>
                            </StackPanel>
                        </Border>
                    </DataTemplate>
                </ItemsControl.ItemTemplate>
            </ItemsControl>
        </ScrollViewer>
        
        <!-- Input Area -->
        <Border Grid.Row="2" 
               Background="#2d2d2d" 
               Padding="16">
            <Grid>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="*"/>
                    <ColumnDefinition Width="Auto"/>
                </Grid.ColumnDefinitions>
                
                <TextBox x:Name="MessageInput"
                        Grid.Column="0"
                        TextWrapping="Wrap"
                        AcceptsReturn="True"
                        Height="80"
                        VerticalScrollBarVisibility="Auto"
                        KeyDown="MessageInput_KeyDown"/>
                
                <Button x:Name="SendButton"
                       Grid.Column="1"
                       Content="Send"
                       Width="80"
                       Height="80"
                       Margin="8,0,0,0"
                       Click="SendButton_Click"/>
            </Grid>
        </Border>
        
        <!-- Status Bar -->
        <StatusBar Grid.Row="3" Background="#2d2d2d">
            <StatusBarItem>
                <TextBlock x:Name="StatusText" Text="Initializing..."/>
            </StatusBarItem>
            <StatusBarItem HorizontalAlignment="Right">
                <TextBlock x:Name="TokensPerSecText"/>
            </StatusBarItem>
        </StatusBar>
    </Grid>
</Window>
```

### MainWindow.xaml.cs

```csharp
using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using Microsoft.Win32;
using WoollyDesktop.Services;

namespace WoollyWpf;

public partial class MainWindow : Window
{
    private readonly WoollyHttpClient _woollyClient;
    private readonly ObservableCollection<ChatMessageViewModel> _messages;
    private readonly DispatcherTimer _scrollTimer;
    private CancellationTokenSource? _generationCts;
    private string? _currentModel;
    private bool _isGenerating;

    public MainWindow()
    {
        InitializeComponent();
        
        _woollyClient = new WoollyHttpClient();
        _messages = new ObservableCollection<ChatMessageViewModel>();
        ChatItemsControl.ItemsSource = _messages;
        
        _scrollTimer = new DispatcherTimer
        {
            Interval = TimeSpan.FromMilliseconds(100)
        };
        _scrollTimer.Tick += (s, e) =>
        {
            _scrollTimer.Stop();
            ChatScrollViewer.ScrollToEnd();
        };
        
        Loaded += MainWindow_Loaded;
        Closing += MainWindow_Closing;
    }
    
    private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        UpdateStatus("Starting Woolly server...");
        
        var modelsPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Woolly",
            "Models"
        );
        Directory.CreateDirectory(modelsPath);
        
        await Task.Run(() => _woollyClient.StartServer(modelsPath));
        
        if (await _woollyClient.CheckHealthAsync())
        {
            UpdateStatus("Woolly server running");
            await RefreshModels();
        }
        else
        {
            UpdateStatus("Failed to start Woolly server");
            MessageBox.Show("Failed to start Woolly server", "Error", 
                MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }
    
    private void MainWindow_Closing(object sender, CancelEventArgs e)
    {
        _generationCts?.Cancel();
        _woollyClient.Dispose();
    }
    
    private async Task RefreshModels()
    {
        try
        {
            var models = await _woollyClient.ListModelsAsync();
            
            await Dispatcher.InvokeAsync(() =>
            {
                ModelComboBox.Items.Clear();
                
                foreach (var model in models)
                {
                    var item = new ComboBoxItem
                    {
                        Content = $"{model.Name} ({FormatBytes(model.Size)})",
                        Tag = model
                    };
                    ModelComboBox.Items.Add(item);
                    
                    if (model.Loaded)
                    {
                        ModelComboBox.SelectedItem = item;
                        _currentModel = model.Name;
                    }
                }
                
                if (ModelComboBox.Items.Count == 0)
                {
                    ModelComboBox.Items.Add(new ComboBoxItem { Content = "No models found" });
                }
            });
        }
        catch (Exception ex)
        {
            UpdateStatus($"Failed to load models: {ex.Message}");
        }
    }
    
    private async void LoadModelButton_Click(object sender, RoutedEventArgs e)
    {
        if (ModelComboBox.SelectedItem is not ComboBoxItem { Tag: ModelInfo model })
            return;
            
        LoadModelButton.IsEnabled = false;
        UpdateStatus($"Loading model: {model.Name}");
        
        try
        {
            if (await _woollyClient.LoadModelAsync(model.Name))
            {
                _currentModel = model.Name;
                UpdateStatus($"Model loaded: {model.Name}");
                AddSystemMessage($"Model '{model.Name}' loaded successfully");
            }
            else
            {
                UpdateStatus("Failed to load model");
                MessageBox.Show("Failed to load model", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
        catch (Exception ex)
        {
            UpdateStatus($"Error: {ex.Message}");
        }
        finally
        {
            LoadModelButton.IsEnabled = true;
        }
    }
    
    private async void UploadModelButton_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Filter = "GGUF Files (*.gguf)|*.gguf|All Files (*.*)|*.*",
            Title = "Select GGUF Model"
        };
        
        if (dialog.ShowDialog() == true)
        {
            var destPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Woolly",
                "Models",
                Path.GetFileName(dialog.FileName)
            );
            
            try
            {
                await Task.Run(() => File.Copy(dialog.FileName, destPath, true));
                AddSystemMessage($"Model uploaded: {Path.GetFileName(dialog.FileName)}");
                await RefreshModels();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to upload model: {ex.Message}", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }
    
    private void MessageInput_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && !Keyboard.IsKeyDown(Key.LeftShift))
        {
            e.Handled = true;
            SendButton_Click(sender, e);
        }
    }
    
    private async void SendButton_Click(object sender, RoutedEventArgs e)
    {
        if (_isGenerating || string.IsNullOrEmpty(_currentModel))
            return;
            
        var message = MessageInput.Text.Trim();
        if (string.IsNullOrEmpty(message))
            return;
            
        MessageInput.Clear();
        AddUserMessage(message);
        
        _isGenerating = true;
        SendButton.IsEnabled = false;
        SendButton.Content = "Generating...";
        _generationCts = new CancellationTokenSource();
        
        var assistantMessage = AddAssistantMessage("");
        var fullResponse = "";
        var tokenCount = 0;
        var startTime = DateTime.Now;
        
        try
        {
            var messages = _messages
                .Where(m => m.Role != "system")
                .TakeLast(10)
                .Select(m => new ChatMessage(m.Role, m.RawContent))
                .Append(new ChatMessage("user", message));
                
            await foreach (var token in _woollyClient.ChatStreamAsync(
                messages, null, _generationCts.Token))
            {
                fullResponse += token;
                tokenCount++;
                
                await Dispatcher.InvokeAsync(() =>
                {
                    assistantMessage.Content = fullResponse + "▊";
                    
                    if (tokenCount % 5 == 0)
                    {
                        var elapsed = (DateTime.Now - startTime).TotalSeconds;
                        var tokensPerSec = tokenCount / Math.Max(elapsed, 0.001);
                        TokensPerSecText.Text = $"{tokensPerSec:F1} tokens/s";
                    }
                    
                    _scrollTimer.Stop();
                    _scrollTimer.Start();
                });
            }
            
            assistantMessage.Content = fullResponse;
        }
        catch (OperationCanceledException)
        {
            assistantMessage.Content = fullResponse + "\n\n[Generation cancelled]";
        }
        catch (Exception ex)
        {
            assistantMessage.Content = $"Error: {ex.Message}";
        }
        finally
        {
            _isGenerating = false;
            SendButton.IsEnabled = true;
            SendButton.Content = "Send";
            TokensPerSecText.Text = "";
            _generationCts?.Dispose();
            _generationCts = null;
        }
    }
    
    private void ModelComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        LoadModelButton.IsEnabled = ModelComboBox.SelectedItem is ComboBoxItem { Tag: ModelInfo };
    }
    
    private void AddUserMessage(string content)
    {
        _messages.Add(new ChatMessageViewModel
        {
            Role = "user",
            Content = content,
            RawContent = content,
            Header = $"You ({DateTime.Now:HH:mm})",
            Alignment = HorizontalAlignment.Right,
            BackgroundColor = "#0084ff"
        });
    }
    
    private ChatMessageViewModel AddAssistantMessage(string content)
    {
        var message = new ChatMessageViewModel
        {
            Role = "assistant",
            Content = content,
            RawContent = content,
            Header = $"Assistant ({DateTime.Now:HH:mm})",
            Alignment = HorizontalAlignment.Left,
            BackgroundColor = "#333333"
        };
        _messages.Add(message);
        return message;
    }
    
    private void AddSystemMessage(string content)
    {
        _messages.Add(new ChatMessageViewModel
        {
            Role = "system",
            Content = content,
            RawContent = content,
            Header = "System",
            Alignment = HorizontalAlignment.Center,
            BackgroundColor = "#555555"
        });
    }
    
    private void UpdateStatus(string status)
    {
        Dispatcher.Invoke(() => StatusText.Text = status);
    }
    
    private static string FormatBytes(long bytes)
    {
        string[] sizes = { "B", "KB", "MB", "GB" };
        double size = bytes;
        int order = 0;
        
        while (size >= 1024 && order < sizes.Length - 1)
        {
            order++;
            size /= 1024;
        }
        
        return $"{size:0.#} {sizes[order]}";
    }
}

public class ChatMessageViewModel : INotifyPropertyChanged
{
    private string _content = "";
    
    public string Role { get; set; } = "";
    public string Header { get; set; } = "";
    public string RawContent { get; set; } = "";
    public HorizontalAlignment Alignment { get; set; }
    public string BackgroundColor { get; set; } = "#333333";
    
    public string Content
    {
        get => _content;
        set
        {
            _content = value;
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Content)));
        }
    }
    
    public event PropertyChangedEventHandler? PropertyChanged;
}
```

## .NET MAUI Implementation

### MauiProgram.cs

```csharp
using Microsoft.Extensions.Logging;
using WoollyMaui.Services;
using WoollyMaui.ViewModels;
using WoollyMaui.Views;

namespace WoollyMaui;

public static class MauiProgram
{
    public static MauiApp CreateMauiApp()
    {
        var builder = MauiApp.CreateBuilder();
        builder
            .UseMauiApp<App>()
            .ConfigureFonts(fonts =>
            {
                fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
            });

#if DEBUG
        builder.Logging.AddDebug();
#endif

        // Register services
        builder.Services.AddSingleton<IWoollyClient, WoollyHttpClient>();
        builder.Services.AddSingleton<ChatViewModel>();
        builder.Services.AddTransient<ChatPage>();

        return builder.Build();
    }
}
```

### ViewModels/ChatViewModel.cs

```csharp
using System.Collections.ObjectModel;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using WoollyMaui.Models;
using WoollyMaui.Services;

namespace WoollyMaui.ViewModels;

public partial class ChatViewModel : ObservableObject
{
    private readonly IWoollyClient _woollyClient;
    private CancellationTokenSource? _generationCts;
    
    [ObservableProperty]
    private ObservableCollection<ChatMessageModel> messages = new();
    
    [ObservableProperty]
    private ObservableCollection<ModelInfo> models = new();
    
    [ObservableProperty]
    private ModelInfo? selectedModel;
    
    [ObservableProperty]
    private string messageInput = string.Empty;
    
    [ObservableProperty]
    private bool isGenerating;
    
    [ObservableProperty]
    private string status = "Initializing...";
    
    [ObservableProperty]
    private double tokensPerSecond;
    
    public ChatViewModel(IWoollyClient woollyClient)
    {
        _woollyClient = woollyClient;
        _ = InitializeAsync();
    }
    
    private async Task InitializeAsync()
    {
        Status = "Starting Woolly server...";
        
        if (_woollyClient is WoollyHttpClient httpClient)
        {
            var modelsPath = Path.Combine(FileSystem.AppDataDirectory, "models");
            Directory.CreateDirectory(modelsPath);
            
            await Task.Run(() => httpClient.StartServer(modelsPath));
        }
        
        if (await _woollyClient.CheckHealthAsync())
        {
            Status = "Woolly server running";
            await RefreshModelsAsync();
        }
        else
        {
            Status = "Failed to start Woolly server";
            await Application.Current!.MainPage!.DisplayAlert(
                "Error", "Failed to start Woolly server", "OK");
        }
    }
    
    [RelayCommand]
    private async Task RefreshModelsAsync()
    {
        try
        {
            var modelList = await _woollyClient.ListModelsAsync();
            
            await MainThread.InvokeOnMainThreadAsync(() =>
            {
                Models.Clear();
                foreach (var model in modelList)
                {
                    Models.Add(model);
                    if (model.Loaded)
                    {
                        SelectedModel = model;
                    }
                }
            });
        }
        catch (Exception ex)
        {
            Status = $"Failed to load models: {ex.Message}";
        }
    }
    
    [RelayCommand]
    private async Task LoadModelAsync()
    {
        if (SelectedModel == null) return;
        
        Status = $"Loading model: {SelectedModel.Name}";
        
        try
        {
            if (await _woollyClient.LoadModelAsync(SelectedModel.Name))
            {
                Status = $"Model loaded: {SelectedModel.Name}";
                AddSystemMessage($"Model '{SelectedModel.Name}' loaded successfully");
                await RefreshModelsAsync();
            }
            else
            {
                Status = "Failed to load model";
                await Application.Current!.MainPage!.DisplayAlert(
                    "Error", "Failed to load model", "OK");
            }
        }
        catch (Exception ex)
        {
            Status = $"Error: {ex.Message}";
        }
    }
    
    [RelayCommand]
    private async Task SendMessageAsync()
    {
        if (IsGenerating || string.IsNullOrWhiteSpace(MessageInput))
            return;
            
        var message = MessageInput.Trim();
        MessageInput = string.Empty;
        
        AddUserMessage(message);
        
        IsGenerating = true;
        _generationCts = new CancellationTokenSource();
        
        var assistantMessage = AddAssistantMessage("");
        var fullResponse = "";
        var tokenCount = 0;
        var startTime = DateTime.Now;
        
        try
        {
            var chatMessages = Messages
                .Where(m => m.Role != "system")
                .TakeLast(10)
                .Select(m => new ChatMessage(m.Role, m.Content))
                .Append(new ChatMessage("user", message));
                
            await foreach (var token in _woollyClient.ChatStreamAsync(
                chatMessages, null, _generationCts.Token))
            {
                fullResponse += token;
                tokenCount++;
                
                await MainThread.InvokeOnMainThreadAsync(() =>
                {
                    assistantMessage.Content = fullResponse + "▊";
                    
                    if (tokenCount % 5 == 0)
                    {
                        var elapsed = (DateTime.Now - startTime).TotalSeconds;
                        TokensPerSecond = tokenCount / Math.Max(elapsed, 0.001);
                    }
                });
            }
            
            assistantMessage.Content = fullResponse;
        }
        catch (OperationCanceledException)
        {
            assistantMessage.Content = fullResponse + "\n\n[Generation cancelled]";
        }
        catch (Exception ex)
        {
            assistantMessage.Content = $"Error: {ex.Message}";
        }
        finally
        {
            IsGenerating = false;
            TokensPerSecond = 0;
            _generationCts?.Dispose();
            _generationCts = null;
        }
    }
    
    [RelayCommand]
    private void CancelGeneration()
    {
        _generationCts?.Cancel();
    }
    
    private void AddUserMessage(string content)
    {
        Messages.Add(new ChatMessageModel
        {
            Role = "user",
            Content = content,
            Timestamp = DateTime.Now,
            IsUser = true
        });
    }
    
    private ChatMessageModel AddAssistantMessage(string content)
    {
        var message = new ChatMessageModel
        {
            Role = "assistant",
            Content = content,
            Timestamp = DateTime.Now,
            IsUser = false
        };
        Messages.Add(message);
        return message;
    }
    
    private void AddSystemMessage(string content)
    {
        Messages.Add(new ChatMessageModel
        {
            Role = "system",
            Content = content,
            Timestamp = DateTime.Now,
            IsSystem = true
        });
    }
}
```

### Views/ChatPage.xaml

```xml
<?xml version="1.0" encoding="utf-8" ?>
<ContentPage xmlns="http://schemas.microsoft.com/dotnet/2021/maui"
             xmlns:x="http://schemas.microsoft.com/winfx/2009/xaml"
             x:Class="WoollyMaui.Views.ChatPage"
             Title="Woolly AI Assistant"
             BackgroundColor="#1e1e1e">
    
    <ContentPage.Resources>
        <ResourceDictionary>
            <Style x:Key="UserMessageStyle" TargetType="Frame">
                <Setter Property="BackgroundColor" Value="#0084ff"/>
                <Setter Property="CornerRadius" Value="18"/>
                <Setter Property="Padding" Value="12,8"/>
                <Setter Property="HorizontalOptions" Value="End"/>
                <Setter Property="MaximumWidthRequest" Value="300"/>
            </Style>
            
            <Style x:Key="AssistantMessageStyle" TargetType="Frame">
                <Setter Property="BackgroundColor" Value="#333333"/>
                <Setter Property="CornerRadius" Value="18"/>
                <Setter Property="Padding" Value="12,8"/>
                <Setter Property="HorizontalOptions" Value="Start"/>
                <Setter Property="MaximumWidthRequest" Value="300"/>
            </Style>
            
            <Style x:Key="SystemMessageStyle" TargetType="Frame">
                <Setter Property="BackgroundColor" Value="#555555"/>
                <Setter Property="CornerRadius" Value="8"/>
                <Setter Property="Padding" Value="8,4"/>
                <Setter Property="HorizontalOptions" Value="Center"/>
            </Style>
        </ResourceDictionary>
    </ContentPage.Resources>
    
    <Grid RowDefinitions="Auto,*,Auto,Auto">
        <!-- Header -->
        <Grid Grid.Row="0" 
              BackgroundColor="#2d2d2d" 
              Padding="16"
              ColumnDefinitions="*,Auto">
            
            <Label Grid.Column="0"
                   Text="Woolly AI Assistant"
                   FontSize="24"
                   FontAttributes="Bold"
                   VerticalOptions="Center"/>
            
            <HorizontalStackLayout Grid.Column="1" Spacing="8">
                <Picker ItemsSource="{Binding Models}"
                        SelectedItem="{Binding SelectedModel}"
                        ItemDisplayBinding="{Binding Name}"
                        WidthRequest="200"
                        BackgroundColor="#333"
                        TextColor="White"/>
                
                <Button Text="Load Model"
                        Command="{Binding LoadModelCommand}"
                        BackgroundColor="#0084ff"/>
                
                <Button Text="Refresh"
                        Command="{Binding RefreshModelsCommand}"
                        BackgroundColor="#666"/>
            </HorizontalStackLayout>
        </Grid>
        
        <!-- Chat Messages -->
        <ScrollView Grid.Row="1" x:Name="ChatScrollView">
            <CollectionView ItemsSource="{Binding Messages}"
                           Margin="16">
                <CollectionView.ItemTemplate>
                    <DataTemplate>
                        <Grid Padding="0,4">
                            <!-- User Message -->
                            <Frame IsVisible="{Binding IsUser}"
                                   Style="{StaticResource UserMessageStyle}">
                                <VerticalStackLayout>
                                    <Label Text="{Binding Timestamp, StringFormat='{0:HH:mm}'}"
                                           FontSize="12"
                                           TextColor="#ddd"
                                           HorizontalTextAlignment="End"/>
                                    <Label Text="{Binding Content}"
                                           TextColor="White"/>
                                </VerticalStackLayout>
                            </Frame>
                            
                            <!-- Assistant Message -->
                            <Frame IsVisible="{Binding IsAssistant}"
                                   Style="{StaticResource AssistantMessageStyle}">
                                <VerticalStackLayout>
                                    <Label Text="Assistant"
                                           FontSize="12"
                                           TextColor="#aaa"/>
                                    <Label Text="{Binding Content}"
                                           TextColor="White"/>
                                </VerticalStackLayout>
                            </Frame>
                            
                            <!-- System Message -->
                            <Frame IsVisible="{Binding IsSystem}"
                                   Style="{StaticResource SystemMessageStyle}">
                                <Label Text="{Binding Content}"
                                       TextColor="#ccc"
                                       FontSize="12"
                                       HorizontalTextAlignment="Center"/>
                            </Frame>
                        </Grid>
                    </DataTemplate>
                </CollectionView.ItemTemplate>
            </CollectionView>
        </ScrollView>
        
        <!-- Input Area -->
        <Grid Grid.Row="2" 
              BackgroundColor="#2d2d2d"
              Padding="16"
              ColumnDefinitions="*,Auto">
            
            <Editor Grid.Column="0"
                    Text="{Binding MessageInput}"
                    Placeholder="Type your message..."
                    PlaceholderColor="#888"
                    TextColor="White"
                    BackgroundColor="#333"
                    HeightRequest="80"
                    AutoSize="TextChanges"/>
            
            <Button Grid.Column="1"
                    Text="{Binding IsGenerating, Converter={StaticResource BoolToSendButtonTextConverter}}"
                    Command="{Binding SendMessageCommand}"
                    IsEnabled="{Binding IsGenerating, Converter={StaticResource InverseBoolConverter}}"
                    BackgroundColor="#0084ff"
                    WidthRequest="80"
                    HeightRequest="80"
                    Margin="8,0,0,0"/>
        </Grid>
        
        <!-- Status Bar -->
        <Grid Grid.Row="3" 
              BackgroundColor="#2d2d2d"
              Padding="16,8"
              ColumnDefinitions="*,Auto">
            
            <Label Grid.Column="0"
                   Text="{Binding Status}"
                   TextColor="#888"
                   VerticalOptions="Center"/>
            
            <Label Grid.Column="1"
                   IsVisible="{Binding IsGenerating}"
                   TextColor="#888"
                   VerticalOptions="Center">
                <Label.FormattedText>
                    <FormattedString>
                        <Span Text="{Binding TokensPerSecond, StringFormat='{0:F1}'}" />
                        <Span Text=" tokens/s" />
                    </FormattedString>
                </Label.FormattedText>
            </Label>
        </Grid>
    </Grid>
</ContentPage>
```

## Advanced Features

### 1. P/Invoke for Direct Rust Integration

```csharp
// WoollyNative.cs
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace WoollyDesktop.Native;

public static class WoollyNative
{
    private const string LibraryName = "woolly_core";
    
    [StructLayout(LayoutKind.Sequential)]
    public struct WoollyConfig
    {
        public int MaxBatchSize;
        public int MaxSequenceLength;
        public bool EnableGpu;
        public IntPtr ModelPath;
    }
    
    [StructLayout(LayoutKind.Sequential)]
    public struct WoollyGenerateRequest
    {
        public IntPtr Prompt;
        public int MaxTokens;
        public float Temperature;
        public float TopP;
        public int TopK;
        public float RepeatPenalty;
    }
    
    // Engine lifecycle
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr woolly_engine_new(ref WoollyConfig config);
    
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void woolly_engine_free(IntPtr engine);
    
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int woolly_engine_load_model(IntPtr engine, IntPtr modelPath);
    
    // Generation
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr woolly_generate(
        IntPtr engine,
        ref WoollyGenerateRequest request,
        IntPtr callback,
        IntPtr userData
    );
    
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void woolly_free_string(IntPtr str);
    
    // Callback for streaming
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void TokenCallback(IntPtr token, IntPtr userData);
}

// Safe wrapper
public class WoollyNativeEngine : IDisposable
{
    private IntPtr _engine;
    private bool _disposed;
    
    public WoollyNativeEngine(WoollyNative.WoollyConfig config)
    {
        _engine = WoollyNative.woolly_engine_new(ref config);
        if (_engine == IntPtr.Zero)
        {
            throw new InvalidOperationException("Failed to create Woolly engine");
        }
    }
    
    public void LoadModel(string modelPath)
    {
        var pathPtr = Marshal.StringToHGlobalAnsi(modelPath);
        try
        {
            var result = WoollyNative.woolly_engine_load_model(_engine, pathPtr);
            if (result != 0)
            {
                throw new InvalidOperationException($"Failed to load model: {result}");
            }
        }
        finally
        {
            Marshal.FreeHGlobal(pathPtr);
        }
    }
    
    public async Task<string> GenerateAsync(
        string prompt,
        GenerationConfig config,
        Action<string>? onToken = null,
        CancellationToken cancellationToken = default)
    {
        var tcs = new TaskCompletionSource<string>();
        var result = new StringBuilder();
        
        // Pin callback
        WoollyNative.TokenCallback callback = (tokenPtr, userData) =>
        {
            if (!cancellationToken.IsCancellationRequested)
            {
                var token = Marshal.PtrToStringAnsi(tokenPtr) ?? "";
                result.Append(token);
                onToken?.Invoke(token);
            }
        };
        
        var callbackPtr = Marshal.GetFunctionPointerForDelegate(callback);
        var promptPtr = Marshal.StringToHGlobalAnsi(prompt);
        
        try
        {
            var request = new WoollyNative.WoollyGenerateRequest
            {
                Prompt = promptPtr,
                MaxTokens = config.MaxTokens,
                Temperature = config.Temperature,
                TopP = config.TopP,
                TopK = config.TopK,
                RepeatPenalty = config.RepeatPenalty
            };
            
            await Task.Run(() =>
            {
                var resultPtr = WoollyNative.woolly_generate(
                    _engine, ref request, callbackPtr, IntPtr.Zero);
                    
                if (resultPtr != IntPtr.Zero)
                {
                    WoollyNative.woolly_free_string(resultPtr);
                }
            }, cancellationToken);
            
            return result.ToString();
        }
        finally
        {
            Marshal.FreeHGlobal(promptPtr);
            GC.KeepAlive(callback); // Keep callback alive
        }
    }
    
    public void Dispose()
    {
        if (!_disposed)
        {
            if (_engine != IntPtr.Zero)
            {
                WoollyNative.woolly_engine_free(_engine);
                _engine = IntPtr.Zero;
            }
            _disposed = true;
        }
    }
}
```

### 2. Model Download with Progress

```csharp
public class ModelDownloadService
{
    public event EventHandler<DownloadProgressEventArgs>? ProgressChanged;
    
    public async Task<string> DownloadModelAsync(
        string url,
        string destinationPath,
        CancellationToken cancellationToken = default)
    {
        using var client = new HttpClient();
        using var response = await client.GetAsync(url, 
            HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        
        response.EnsureSuccessStatusCode();
        
        var totalBytes = response.Content.Headers.ContentLength ?? -1;
        var buffer = new byte[8192];
        var totalBytesRead = 0L;
        
        using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var fileStream = new FileStream(destinationPath, FileMode.Create);
        
        while (true)
        {
            var bytesRead = await stream.ReadAsync(buffer, cancellationToken);
            if (bytesRead == 0) break;
            
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
            totalBytesRead += bytesRead;
            
            if (totalBytes > 0)
            {
                var progress = (double)totalBytesRead / totalBytes * 100;
                ProgressChanged?.Invoke(this, new DownloadProgressEventArgs
                {
                    BytesReceived = totalBytesRead,
                    TotalBytesToReceive = totalBytes,
                    ProgressPercentage = progress
                });
            }
        }
        
        return destinationPath;
    }
}

public class DownloadProgressEventArgs : EventArgs
{
    public long BytesReceived { get; init; }
    public long TotalBytesToReceive { get; init; }
    public double ProgressPercentage { get; init; }
}
```

### 3. System Tray Integration (WPF)

```csharp
public partial class App : Application
{
    private NotifyIcon? _notifyIcon;
    private IWoollyClient? _woollyClient;
    
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        
        // Create system tray icon
        _notifyIcon = new NotifyIcon
        {
            Icon = new Icon("woolly.ico"),
            Visible = true,
            Text = "Woolly AI Assistant"
        };
        
        // Context menu
        var contextMenu = new ContextMenuStrip();
        contextMenu.Items.Add("Open", null, (s, e) => ShowMainWindow());
        contextMenu.Items.Add("Settings", null, (s, e) => ShowSettings());
        contextMenu.Items.Add("-");
        contextMenu.Items.Add("Exit", null, (s, e) => Shutdown());
        
        _notifyIcon.ContextMenuStrip = contextMenu;
        _notifyIcon.DoubleClick += (s, e) => ShowMainWindow();
        
        // Start Woolly in background
        _woollyClient = new WoollyHttpClient();
        if (_woollyClient is WoollyHttpClient httpClient)
        {
            httpClient.StartServer();
        }
    }
    
    private void ShowMainWindow()
    {
        if (MainWindow == null)
        {
            MainWindow = new MainWindow();
        }
        
        MainWindow.Show();
        MainWindow.WindowState = WindowState.Normal;
        MainWindow.Activate();
    }
    
    protected override void OnExit(ExitEventArgs e)
    {
        _notifyIcon?.Dispose();
        _woollyClient?.Dispose();
        base.OnExit(e);
    }
}
```

## Performance Optimization

### 1. Background Service Pattern

```csharp
public class WoollyBackgroundService : BackgroundService
{
    private readonly IWoollyClient _client;
    private readonly ILogger<WoollyBackgroundService> _logger;
    
    public WoollyBackgroundService(
        IWoollyClient client,
        ILogger<WoollyBackgroundService> logger)
    {
        _client = client;
        _logger = logger;
    }
    
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // Perform background tasks
                await _client.CheckHealthAsync(stoppingToken);
                await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in background service");
            }
        }
    }
}
```

### 2. Memory-Mapped Model Loading

```csharp
public class MemoryMappedModelLoader
{
    public unsafe byte[] LoadModel(string path)
    {
        using var mmf = MemoryMappedFile.CreateFromFile(
            path,
            FileMode.Open,
            null,
            0,
            MemoryMappedFileAccess.Read);
            
        using var accessor = mmf.CreateViewAccessor(0, 0, 
            MemoryMappedFileAccess.Read);
            
        var length = new FileInfo(path).Length;
        var buffer = new byte[length];
        
        accessor.ReadArray(0, buffer, 0, (int)length);
        return buffer;
    }
}
```

## Deployment

### Windows Store (MSIX)

```xml
<!-- Package.appxmanifest -->
<?xml version="1.0" encoding="utf-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
         xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
         xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities">
  
  <Identity Name="YourCompany.WoollyAI"
            Publisher="CN=YourCompany"
            Version="1.0.0.0" />
            
  <Properties>
    <DisplayName>Woolly AI Assistant</DisplayName>
    <PublisherDisplayName>Your Company</PublisherDisplayName>
    <Logo>Images\StoreLogo.png</Logo>
  </Properties>
  
  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" 
                        MinVersion="10.0.17763.0" 
                        MaxVersionTested="10.0.19041.0" />
  </Dependencies>
  
  <Resources>
    <Resource Language="x-generate"/>
  </Resources>
  
  <Applications>
    <Application Id="App"
                 Executable="WoollyAI.exe"
                 EntryPoint="WoollyAI.App">
      <uap:VisualElements
        DisplayName="Woolly AI Assistant"
        Description="Local AI assistant powered by Woolly"
        BackgroundColor="transparent"
        Square150x150Logo="Images\Square150x150Logo.png"
        Square44x44Logo="Images\Square44x44Logo.png">
      </uap:VisualElements>
    </Application>
  </Applications>
  
  <Capabilities>
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>
</Package>
```

### ClickOnce Deployment

```xml
<!-- In .csproj -->
<PropertyGroup>
  <PublishProtocol>ClickOnce</PublishProtocol>
  <PublishUrl>https://yourserver/woolly/</PublishUrl>
  <InstallUrl>https://yourserver/woolly/</InstallUrl>
  <UpdateEnabled>true</UpdateEnabled>
  <UpdateMode>Background</UpdateMode>
  <UpdateInterval>7</UpdateInterval>
  <UpdateIntervalUnits>Days</UpdateIntervalUnits>
  <ProductName>Woolly AI Assistant</ProductName>
  <PublisherName>Your Company</PublisherName>
  <ApplicationRevision>1</ApplicationRevision>
  <ApplicationVersion>1.0.0.%2a</ApplicationVersion>
  <UseApplicationTrust>false</UseApplicationTrust>
  <CreateDesktopShortcut>true</CreateDesktopShortcut>
  <PublishWizardCompleted>true</PublishWizardCompleted>
  <BootstrapperEnabled>true</BootstrapperEnabled>
</PropertyGroup>
```

## Best Practices

1. **Async All The Way**: Use async/await for all I/O operations
2. **MVVM Pattern**: Separate UI from business logic
3. **Dependency Injection**: Use DI for testability
4. **Cancellation**: Support cancellation tokens everywhere
5. **Progress Reporting**: Use IProgress<T> for progress updates
6. **Exception Handling**: Global exception handlers
7. **Logging**: Structured logging with Serilog
8. **Settings**: Use user/app settings appropriately
9. **Accessibility**: Support screen readers and keyboard navigation
10. **Localization**: Prepare for multiple languages