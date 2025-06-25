//! Plugin registry system for MCP

use dashmap::DashMap;
use std::sync::Arc;
use std::fmt;
use uuid::Uuid;

use crate::protocol::{ToolHandler, ResourceHandler, PromptHandler, McpHandler};
use crate::types::{ToolInfo, ResourceInfo, PromptInfo};

/// Plugin identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PluginId(pub Uuid);

impl Default for PluginId {
    fn default() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for PluginId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Plugin metadata
#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub id: PluginId,
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: Option<String>,
    pub homepage: Option<String>,
    pub capabilities: PluginCapabilities,
}

/// Plugin capabilities
#[derive(Debug, Clone, Default)]
pub struct PluginCapabilities {
    pub tools: Vec<String>,
    pub resources: Vec<String>,
    pub prompts: Vec<String>,
    pub handlers: Vec<String>,
}

/// Plugin trait that all MCP plugins must implement
pub trait Plugin: Send + Sync {
    /// Get plugin information
    fn info(&self) -> PluginInfo;
    
    /// Initialize the plugin
    fn initialize(&self) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Shutdown the plugin
    fn shutdown(&self) -> Result<(), Box<dyn std::error::Error>>;
}

/// Plugin registry for managing MCP plugins
pub struct PluginRegistry {
    /// Registered plugins
    plugins: DashMap<PluginId, Arc<dyn Plugin>>,
    
    /// Tool handlers by name
    tools: DashMap<String, (PluginId, Arc<dyn ToolHandler>)>,
    
    /// Resource handlers by URI pattern
    resources: DashMap<String, (PluginId, Arc<dyn ResourceHandler>)>,
    
    /// Prompt handlers by name
    prompts: DashMap<String, (PluginId, Arc<dyn PromptHandler>)>,
    
    /// Method handlers by pattern
    handlers: DashMap<String, (PluginId, Arc<dyn McpHandler>)>,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            plugins: DashMap::new(),
            tools: DashMap::new(),
            resources: DashMap::new(),
            prompts: DashMap::new(),
            handlers: DashMap::new(),
        }
    }
    
    /// Register a plugin
    pub fn register_plugin(&self, plugin: Arc<dyn Plugin>) -> Result<PluginId, RegistryError> {
        let info = plugin.info();
        let plugin_id = info.id;
        
        // Check if plugin is already registered
        if self.plugins.contains_key(&plugin_id) {
            return Err(RegistryError::PluginAlreadyRegistered(plugin_id));
        }
        
        // Initialize the plugin
        plugin.initialize()
            .map_err(|e| RegistryError::PluginInitializationFailed(plugin_id, e.to_string()))?;
        
        // Register the plugin
        self.plugins.insert(plugin_id, plugin);
        
        Ok(plugin_id)
    }
    
    /// Unregister a plugin
    pub fn unregister_plugin(&self, plugin_id: PluginId) -> Result<(), RegistryError> {
        // Remove the plugin
        let plugin = self.plugins.remove(&plugin_id)
            .ok_or(RegistryError::PluginNotFound(plugin_id))?;
        
        // Shutdown the plugin
        plugin.1.shutdown()
            .map_err(|e| RegistryError::PluginShutdownFailed(plugin_id, e.to_string()))?;
        
        // Remove all associated handlers
        self.tools.retain(|_, (id, _)| *id != plugin_id);
        self.resources.retain(|_, (id, _)| *id != plugin_id);
        self.prompts.retain(|_, (id, _)| *id != plugin_id);
        self.handlers.retain(|_, (id, _)| *id != plugin_id);
        
        Ok(())
    }
    
    /// Register a tool handler
    pub fn register_tool(
        &self,
        plugin_id: PluginId,
        handler: Arc<dyn ToolHandler>
    ) -> Result<(), RegistryError> {
        // Verify plugin exists
        if !self.plugins.contains_key(&plugin_id) {
            return Err(RegistryError::PluginNotFound(plugin_id));
        }
        
        let info = handler.tool_info();
        
        // Check if tool name is already taken
        if self.tools.contains_key(&info.name) {
            return Err(RegistryError::ToolAlreadyRegistered(info.name.clone()));
        }
        
        self.tools.insert(info.name.clone(), (plugin_id, handler));
        Ok(())
    }
    
    /// Register a resource handler
    pub fn register_resource(
        &self,
        plugin_id: PluginId,
        handler: Arc<dyn ResourceHandler>
    ) -> Result<(), RegistryError> {
        // Verify plugin exists
        if !self.plugins.contains_key(&plugin_id) {
            return Err(RegistryError::PluginNotFound(plugin_id));
        }
        
        let info = handler.resource_info();
        
        // Check if resource URI is already taken
        if self.resources.contains_key(&info.uri) {
            return Err(RegistryError::ResourceAlreadyRegistered(info.uri.clone()));
        }
        
        self.resources.insert(info.uri.clone(), (plugin_id, handler));
        Ok(())
    }
    
    /// Register a prompt handler
    pub fn register_prompt(
        &self,
        plugin_id: PluginId,
        handler: Arc<dyn PromptHandler>
    ) -> Result<(), RegistryError> {
        // Verify plugin exists
        if !self.plugins.contains_key(&plugin_id) {
            return Err(RegistryError::PluginNotFound(plugin_id));
        }
        
        let info = handler.prompt_info();
        
        // Check if prompt name is already taken
        if self.prompts.contains_key(&info.name) {
            return Err(RegistryError::PromptAlreadyRegistered(info.name.clone()));
        }
        
        self.prompts.insert(info.name.clone(), (plugin_id, handler));
        Ok(())
    }
    
    /// Register a method handler
    pub fn register_handler(
        &self,
        plugin_id: PluginId,
        handler: Arc<dyn McpHandler>
    ) -> Result<(), RegistryError> {
        // Verify plugin exists
        if !self.plugins.contains_key(&plugin_id) {
            return Err(RegistryError::PluginNotFound(plugin_id));
        }
        
        let pattern = handler.method_pattern().to_string();
        
        // Check if pattern is already taken
        if self.handlers.contains_key(&pattern) {
            return Err(RegistryError::HandlerAlreadyRegistered(pattern));
        }
        
        self.handlers.insert(pattern, (plugin_id, handler));
        Ok(())
    }
    
    /// Get a tool handler by name
    pub fn get_tool(&self, name: &str) -> Option<Arc<dyn ToolHandler>> {
        self.tools.get(name).map(|entry| entry.1.clone())
    }
    
    /// Get a resource handler by URI
    pub fn get_resource(&self, uri: &str) -> Option<Arc<dyn ResourceHandler>> {
        // First try exact match
        if let Some(entry) = self.resources.get(uri) {
            return Some(entry.1.clone());
        }
        
        // Then try pattern matching
        for entry in self.resources.iter() {
            if Self::matches_uri_pattern(entry.key(), uri) {
                return Some(entry.value().1.clone());
            }
        }
        
        None
    }
    
    /// Get a prompt handler by name
    pub fn get_prompt(&self, name: &str) -> Option<Arc<dyn PromptHandler>> {
        self.prompts.get(name).map(|entry| entry.1.clone())
    }
    
    /// Get a method handler by method name
    pub fn get_handler(&self, method: &str) -> Option<Arc<dyn McpHandler>> {
        // First try exact match
        if let Some(entry) = self.handlers.get(method) {
            return Some(entry.1.clone());
        }
        
        // Then try pattern matching
        for entry in self.handlers.iter() {
            if Self::matches_method_pattern(entry.key(), method) {
                return Some(entry.value().1.clone());
            }
        }
        
        None
    }
    
    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<PluginInfo> {
        self.plugins.iter()
            .map(|entry| entry.value().info())
            .collect()
    }
    
    /// List all registered tools
    pub fn list_tools(&self) -> Vec<ToolInfo> {
        self.tools.iter()
            .map(|entry| entry.value().1.tool_info())
            .collect()
    }
    
    /// List all registered resources
    pub fn list_resources(&self) -> Vec<ResourceInfo> {
        self.resources.iter()
            .map(|entry| entry.value().1.resource_info())
            .collect()
    }
    
    /// List all registered prompts
    pub fn list_prompts(&self) -> Vec<PromptInfo> {
        self.prompts.iter()
            .map(|entry| entry.value().1.prompt_info())
            .collect()
    }
    
    /// Check if a URI matches a pattern
    fn matches_uri_pattern(pattern: &str, uri: &str) -> bool {
        // Simple pattern matching for now
        // Could be extended with glob patterns or regex
        if pattern.ends_with("/*") {
            let prefix = &pattern[..pattern.len() - 2];
            uri.starts_with(prefix)
        } else {
            pattern == uri
        }
    }
    
    /// Check if a method matches a pattern
    fn matches_method_pattern(pattern: &str, method: &str) -> bool {
        // Simple pattern matching for now
        if pattern.ends_with("/*") {
            let prefix = &pattern[..pattern.len() - 2];
            method.starts_with(prefix) && method.contains('/')
        } else {
            pattern == method
        }
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry errors
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("Plugin already registered: {0}")]
    PluginAlreadyRegistered(PluginId),
    
    #[error("Plugin not found: {0}")]
    PluginNotFound(PluginId),
    
    #[error("Plugin initialization failed for {0}: {1}")]
    PluginInitializationFailed(PluginId, String),
    
    #[error("Plugin shutdown failed for {0}: {1}")]
    PluginShutdownFailed(PluginId, String),
    
    #[error("Tool already registered: {0}")]
    ToolAlreadyRegistered(String),
    
    #[error("Resource already registered: {0}")]
    ResourceAlreadyRegistered(String),
    
    #[error("Prompt already registered: {0}")]
    PromptAlreadyRegistered(String),
    
    #[error("Handler already registered for pattern: {0}")]
    HandlerAlreadyRegistered(String),
}

/// Builder for creating plugins
pub struct PluginBuilder {
    id: PluginId,
    name: String,
    version: String,
    description: String,
    author: Option<String>,
    homepage: Option<String>,
    capabilities: PluginCapabilities,
}

impl PluginBuilder {
    /// Create a new plugin builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: PluginId::default(),
            name: name.into(),
            version: "0.1.0".to_string(),
            description: String::new(),
            author: None,
            homepage: None,
            capabilities: PluginCapabilities::default(),
        }
    }
    
    /// Set the plugin ID
    pub fn id(mut self, id: PluginId) -> Self {
        self.id = id;
        self
    }
    
    /// Set the plugin version
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }
    
    /// Set the plugin description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
    
    /// Set the plugin author
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }
    
    /// Set the plugin homepage
    pub fn homepage(mut self, homepage: impl Into<String>) -> Self {
        self.homepage = Some(homepage.into());
        self
    }
    
    /// Add a tool capability
    pub fn tool(mut self, tool: impl Into<String>) -> Self {
        self.capabilities.tools.push(tool.into());
        self
    }
    
    /// Add a resource capability
    pub fn resource(mut self, resource: impl Into<String>) -> Self {
        self.capabilities.resources.push(resource.into());
        self
    }
    
    /// Add a prompt capability
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.capabilities.prompts.push(prompt.into());
        self
    }
    
    /// Add a handler capability
    pub fn handler(mut self, handler: impl Into<String>) -> Self {
        self.capabilities.handlers.push(handler.into());
        self
    }
    
    /// Build the plugin info
    pub fn build(self) -> PluginInfo {
        PluginInfo {
            id: self.id,
            name: self.name,
            version: self.version,
            description: self.description,
            author: self.author,
            homepage: self.homepage,
            capabilities: self.capabilities,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_plugin_builder() {
        let info = PluginBuilder::new("test-plugin")
            .version("1.0.0")
            .description("A test plugin")
            .author("Test Author")
            .homepage("https://example.com")
            .tool("test-tool")
            .resource("test://resource")
            .prompt("test-prompt")
            .handler("test/*")
            .build();
        
        assert_eq!(info.name, "test-plugin");
        assert_eq!(info.version, "1.0.0");
        assert_eq!(info.description, "A test plugin");
        assert_eq!(info.author, Some("Test Author".to_string()));
        assert_eq!(info.homepage, Some("https://example.com".to_string()));
        assert_eq!(info.capabilities.tools.len(), 1);
        assert_eq!(info.capabilities.resources.len(), 1);
        assert_eq!(info.capabilities.prompts.len(), 1);
        assert_eq!(info.capabilities.handlers.len(), 1);
    }
    
    #[test]
    fn test_registry_creation() {
        let registry = PluginRegistry::new();
        assert_eq!(registry.list_plugins().len(), 0);
        assert_eq!(registry.list_tools().len(), 0);
        assert_eq!(registry.list_resources().len(), 0);
        assert_eq!(registry.list_prompts().len(), 0);
    }
}