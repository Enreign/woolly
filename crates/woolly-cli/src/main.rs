//! Woolly CLI - Command Line Interface for Woolly LLM
//!
//! A powerful CLI tool for LLM inference, model inspection, and benchmarking.

use anyhow::Result;
use clap::{Parser, Subcommand};
use console::style;
use std::path::PathBuf;
use tracing::{debug, info, Level};
use tracing_subscriber::FmtSubscriber;

mod commands;
mod config;
mod utils;

use commands::{Command, run::RunCommand, info::InfoCommand, benchmark::BenchmarkCommand};

#[derive(Parser)]
#[command(
    name = "woolly",
    version = env!("CARGO_PKG_VERSION"),
    about = "Woolly LLM inference engine CLI",
    long_about = "A fast, efficient command-line interface for running LLM inference, inspecting models, and benchmarking performance."
)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Enable debug logging
    #[arg(short, long, global = true)]
    debug: bool,

    /// Quiet output (errors only)
    #[arg(short, long, global = true)]
    quiet: bool,

    /// Configuration file path
    #[arg(short, long, global = true, env = "WOOLLY_CONFIG")]
    config: Option<PathBuf>,

    /// JSON output format
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a model
    #[command(name = "run", alias = "r")]
    Run(RunCommand),
    
    /// Display model information
    #[command(name = "info", alias = "i")]
    Info(InfoCommand),
    
    /// Run benchmarks
    #[command(name = "benchmark", alias = "bench", alias = "b")]
    Benchmark(BenchmarkCommand),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    init_logging(&cli)?;
    
    debug!("Woolly CLI v{} starting", env!("CARGO_PKG_VERSION"));
    
    // Load configuration
    let config = config::Config::load(cli.config.as_deref())?;
    debug!("Configuration loaded: {:?}", config);
    
    // Execute command
    let result = match cli.command {
        Commands::Run(cmd) => cmd.execute(&config, cli.json).await,
        Commands::Info(cmd) => cmd.execute(&config, cli.json).await,
        Commands::Benchmark(cmd) => cmd.execute(&config, cli.json).await,
    };
    
    match result {
        Ok(_) => {
            if !cli.quiet {
                info!("Command completed successfully");
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("{} {}", style("Error:").red().bold(), e);
            std::process::exit(1);
        }
    }
}

fn init_logging(cli: &Cli) -> Result<()> {
    let level = if cli.debug {
        Level::DEBUG
    } else if cli.verbose {
        Level::INFO
    } else if cli.quiet {
        Level::ERROR
    } else {
        Level::WARN
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .compact()
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}