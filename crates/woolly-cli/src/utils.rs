//! Utility functions for Woolly CLI

use anyhow::Result;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;
use std::time::Duration;

/// Create a progress bar with standard styling
pub fn create_progress_bar(len: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(message.to_string());
    pb
}

/// Create a spinner progress bar
pub fn create_spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"])
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(message.to_string());
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

/// Format bytes in human-readable format
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

/// Format duration in human-readable format
pub fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    let millis = duration.subsec_millis();

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else if seconds > 0 {
        format!("{}.{:03}s", seconds, millis)
    } else {
        format!("{}ms", duration.as_millis())
    }
}

/// Print formatted output (JSON or human-readable)
pub fn print_output(data: &Value, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(data)?);
    } else {
        print_human_readable(data);
    }
    Ok(())
}

/// Print human-readable output
fn print_human_readable(data: &Value) {
    match data {
        Value::Object(map) => {
            for (key, value) in map {
                match value {
                    Value::String(s) => println!("{}: {}", style(key).bold(), s),
                    Value::Number(n) => println!("{}: {}", style(key).bold(), n),
                    Value::Bool(b) => println!("{}: {}", style(key).bold(), b),
                    Value::Array(arr) => {
                        println!("{}:", style(key).bold());
                        for (i, item) in arr.iter().enumerate() {
                            println!("  {}: {}", i + 1, format_value(item));
                        }
                    }
                    Value::Object(_) => {
                        println!("{}:", style(key).bold());
                        print_nested_object(value, 1);
                    }
                    Value::Null => println!("{}: null", style(key).bold()),
                }
            }
        }
        _ => println!("{}", format_value(data)),
    }
}

fn print_nested_object(data: &Value, indent: usize) {
    let prefix = "  ".repeat(indent);
    
    if let Value::Object(map) = data {
        for (key, value) in map {
            match value {
                Value::Object(_) => {
                    println!("{}{}:", prefix, style(key).bold());
                    print_nested_object(value, indent + 1);
                }
                _ => println!("{}{}: {}", prefix, style(key).bold(), format_value(value)),
            }
        }
    }
}

fn format_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        _ => value.to_string(),
    }
}

/// Print error with styling
#[allow(dead_code)]
pub fn print_error(message: &str) {
    eprintln!("{} {}", style("Error:").red().bold(), message);
}

/// Print warning with styling
#[allow(dead_code)]
pub fn print_warning(message: &str) {
    eprintln!("{} {}", style("Warning:").yellow().bold(), message);
}

/// Print success message with styling
pub fn print_success(message: &str) {
    println!("{} {}", style("Success:").green().bold(), message);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0.00 B");
        assert_eq!(format_bytes(1023), "1023.00 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_millis(500)), "500ms");
        assert_eq!(format_duration(Duration::from_secs(30)), "30.000s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h 1m 1s");
    }
}