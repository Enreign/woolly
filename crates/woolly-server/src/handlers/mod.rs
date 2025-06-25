//! HTTP request handlers

pub mod auth;
pub mod health;
pub mod inference;
pub mod models;
pub mod sessions;

#[cfg(feature = "mcp")]
pub mod mcp;

// Re-export handler modules for convenience
pub use auth::*;
pub use health::*;
pub use inference::*;
pub use models::*;
pub use sessions::*;

#[cfg(feature = "mcp")]
pub use mcp::*;