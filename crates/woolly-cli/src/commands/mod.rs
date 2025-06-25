//! Command implementations for Woolly CLI

pub mod run;
pub mod info;
pub mod benchmark;

use anyhow::Result;
use async_trait::async_trait;

/// Trait for CLI command execution
#[async_trait]
pub trait Command {
    /// Execute the command
    async fn execute(&self, config: &crate::config::Config, json_output: bool) -> Result<()>;
}