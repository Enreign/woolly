//! Woolly Server - HTTP/WebSocket server with MCP integration
//!
//! This binary provides a web server that exposes Woolly's LLM capabilities
//! over HTTP and WebSocket connections with full Model Context Protocol (MCP) support.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tokio;
use tracing::{error, info, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use woolly_server::{
    config::ServerConfig,
    error::ServerResult,
    server::{run_server, WoollyServer},
};

/// Command line arguments
#[derive(Parser)]
#[command(
    name = "woolly-server",
    about = "Woolly LLM HTTP/WebSocket server with MCP integration",
    long_about = "A high-performance web server that exposes Woolly's LLM capabilities over HTTP and WebSocket connections with full Model Context Protocol (MCP) support for tool execution and resource serving."
)]
struct Args {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Server bind address
    #[arg(short, long, default_value = "127.0.0.1:8080")]
    bind: String,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Enable JSON logging
    #[arg(long)]
    json_logs: bool,

    /// Subcommands
    #[command(subcommand)]
    command: Option<Commands>,
}

/// Available subcommands
#[derive(Subcommand)]
enum Commands {
    /// Start the server
    Start {
        /// Run in daemon mode
        #[arg(short, long)]
        daemon: bool,
    },
    /// Generate a default configuration file
    Config {
        /// Output file path
        #[arg(short, long, default_value = "woolly-server.toml")]
        output: PathBuf,
        /// Overwrite existing file
        #[arg(long)]
        force: bool,
    },
    /// Validate configuration file
    Validate {
        /// Configuration file to validate
        config: PathBuf,
    },
    /// Show server information
    Info,
}

#[tokio::main]
async fn main() -> ServerResult<()> {
    let args = Args::parse();

    // Initialize logging
    init_logging(&args)?;

    // Handle subcommands
    match args.command {
        Some(Commands::Start { daemon }) => {
            if daemon {
                todo!("Daemon mode not yet implemented");
            }
            start_server(args).await
        }
        Some(Commands::Config { output, force }) => {
            generate_config(output, force).await
        }
        Some(Commands::Validate { config }) => {
            validate_config(config).await
        }
        Some(Commands::Info) => {
            show_info().await
        }
        None => {
            // Default action is to start the server
            start_server(args).await
        }
    }
}

/// Initialize logging based on command line arguments
fn init_logging(args: &Args) -> ServerResult<()> {
    let log_level = args.log_level.parse::<Level>()
        .map_err(|_| woolly_server::error::ServerError::Config(
            format!("Invalid log level: {}", args.log_level)
        ))?;

    let env_filter = EnvFilter::from_default_env()
        .add_directive(format!("woolly_server={}", log_level).parse().unwrap())
        .add_directive(format!("woolly_core={}", log_level).parse().unwrap())
        .add_directive(format!("woolly_mcp={}", log_level).parse().unwrap());

    if args.json_logs {
        tracing_subscriber::registry()
            .with(fmt::layer().compact())
            .with(env_filter)
            .init();
    } else {
        tracing_subscriber::registry()
            .with(fmt::layer().pretty())
            .with(env_filter)
            .init();
    }

    Ok(())
}

/// Start the server
async fn start_server(args: Args) -> ServerResult<()> {
    info!("Starting Woolly Server v{}", woolly_server::VERSION);

    // Load or create configuration
    let mut config = if let Some(config_path) = args.config {
        info!("Loading configuration from: {:?}", config_path);
        ServerConfig::from_file(&config_path)
            .map_err(|e| woolly_server::error::ServerError::Config(
                format!("Failed to load config: {}", e)
            ))?
    } else {
        info!("Using default configuration");
        ServerConfig::default()
    };

    // Override bind address from command line if provided
    if args.bind != "127.0.0.1:8080" {
        config.bind = args.bind.parse()
            .map_err(|e| woolly_server::error::ServerError::Config(
                format!("Invalid bind address: {}", e)
            ))?;
    }

    // Create and start server
    let server = WoollyServer::new(config)?;
    
    info!("Server configuration:");
    info!("  Bind address: {}", server.config().bind);
    info!("  MCP enabled: {}", server.config().mcp.enabled);
    info!("  Max sessions: {}", server.config().models.max_sessions);
    info!("  Models directory: {:?}", server.config().models.models_dir);

    run_server(server).await
}

/// Generate a default configuration file
async fn generate_config(output: PathBuf, force: bool) -> ServerResult<()> {
    if output.exists() && !force {
        error!("Configuration file already exists: {:?}", output);
        error!("Use --force to overwrite");
        return Err(woolly_server::error::ServerError::Config(
            "Configuration file already exists".to_string()
        ));
    }

    let config = ServerConfig::default();
    config.to_file(&output)
        .map_err(|e| woolly_server::error::ServerError::Config(
            format!("Failed to write config: {}", e)
        ))?;

    info!("Generated default configuration file: {:?}", output);
    Ok(())
}

/// Validate a configuration file
async fn validate_config(config_path: PathBuf) -> ServerResult<()> {
    info!("Validating configuration file: {:?}", config_path);

    match ServerConfig::from_file(&config_path) {
        Ok(config) => {
            info!("Configuration is valid");
            info!("  Bind address: {}", config.bind);
            info!("  MCP enabled: {}", config.mcp.enabled);
            info!("  Max sessions: {}", config.models.max_sessions);
            info!("  Models directory: {:?}", config.models.models_dir);
            Ok(())
        }
        Err(e) => {
            error!("Configuration validation failed: {}", e);
            Err(woolly_server::error::ServerError::Config(
                format!("Invalid configuration: {}", e)
            ))
        }
    }
}

/// Show server information
async fn show_info() -> ServerResult<()> {
    println!("Woolly Server v{}", woolly_server::VERSION);
    println!("A high-performance HTTP/WebSocket server for Woolly LLM");
    println!();
    println!("Features:");
    println!("  • HTTP REST API for model inference");
    println!("  • WebSocket support for real-time communication");
    println!("  • Model Context Protocol (MCP) integration");
    println!("  • JWT and API key authentication");
    println!("  • Rate limiting and request throttling");
    println!("  • Session management");
    println!("  • Tool execution and resource serving");
    println!();
    println!("Configuration:");
    println!("  • Default bind address: 127.0.0.1:8080");
    println!("  • Default models directory: ./models");
    println!("  • Configuration file: woolly-server.toml");
    println!();
    println!("Usage:");
    println!("  woolly-server                    # Start with default config");
    println!("  woolly-server -c config.toml     # Start with custom config");
    println!("  woolly-server config             # Generate default config");
    println!("  woolly-server validate config.toml  # Validate config file");
    
    Ok(())
}