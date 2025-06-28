#!/usr/bin/env python3
"""
Example script to run Woolly validation suite
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# Try to import rich for better output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better output formatting")


def check_woolly_server():
    """Check if Woolly server is running"""
    import aiohttp
    import asyncio
    
    async def check():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8080/api/v1/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except:
            return False
    
    return asyncio.run(check())


def print_header():
    """Print validation suite header"""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold blue]Woolly True Performance Validator[/bold blue]\n"
            "[dim]Comprehensive validation suite for Woolly inference server[/dim]",
            border_style="blue"
        ))
    else:
        print("=" * 60)
        print("Woolly True Performance Validator")
        print("Comprehensive validation suite for Woolly inference server")
        print("=" * 60)


def print_config_summary(config):
    """Print configuration summary"""
    if RICH_AVAILABLE:
        table = Table(title="Test Configuration", show_header=True, header_style="bold magenta")
        table.add_column("Test Category", style="cyan", width=20)
        table.add_column("Enabled", style="green", width=10)
        table.add_column("Details", style="yellow")
        
        for test_name, test_config in config["tests"].items():
            if isinstance(test_config, dict) and "enabled" in test_config:
                enabled = "✓" if test_config["enabled"] else "✗"
                
                # Create details based on test type
                details = []
                if test_name == "load_testing":
                    details.append(f"Users: {test_config['concurrent_users']}")
                    details.append(f"Duration: {test_config['duration']}s")
                elif test_name == "reliability":
                    details.append(f"Duration: {test_config['duration']/3600:.1f}h")
                    details.append(f"Interval: {test_config['check_interval']}s")
                elif test_name == "inference_speed":
                    details.append(f"Prompts: {len(test_config['prompts'])}")
                    details.append(f"Max tokens: {test_config['max_tokens']}")
                
                table.add_row(
                    test_name.replace("_", " ").title(),
                    enabled,
                    ", ".join(details[:2])  # Show first 2 details
                )
        
        console.print(table)
    else:
        print("\nTest Configuration:")
        for test_name, test_config in config["tests"].items():
            if isinstance(test_config, dict) and "enabled" in test_config:
                status = "Enabled" if test_config["enabled"] else "Disabled"
                print(f"  - {test_name}: {status}")


async def run_quick_tests(config_path: str):
    """Run a quick subset of tests"""
    # Load config and modify for quick testing
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Modify config for quick tests
    config["tests"]["inference_speed"]["prompts"] = config["tests"]["inference_speed"]["prompts"][:2]
    config["tests"]["inference_speed"]["max_tokens"] = [10, 50]
    config["tests"]["load_testing"]["concurrent_users"] = [1, 5]
    config["tests"]["load_testing"]["duration"] = 30
    config["tests"]["reliability"]["enabled"] = False  # Skip long test
    config["tests"]["comparative"]["enabled"] = False  # Skip if other tools not installed
    
    # Save temporary config
    temp_config = Path("validation_config_quick.json")
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Import and run validator
    from woolly_true_validator import WoollyTrueValidator
    
    validator = WoollyTrueValidator(str(temp_config))
    results = await validator.run_all_tests()
    
    # Clean up temp config
    temp_config.unlink()
    
    return results


async def run_specific_test(test_name: str, config_path: str):
    """Run a specific test category"""
    from woolly_true_validator import WoollyTrueValidator
    
    # Load config and enable only specified test
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Disable all tests except the specified one
    for test in config["tests"]:
        config["tests"][test]["enabled"] = (test == test_name)
    
    # Save temporary config
    temp_config = Path(f"validation_config_{test_name}.json")
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    validator = WoollyTrueValidator(str(temp_config))
    results = await validator.run_all_tests()
    
    # Clean up temp config
    temp_config.unlink()
    
    return results


def print_results_summary(results):
    """Print summary of results"""
    if RICH_AVAILABLE:
        console.print("\n[bold green]Validation Complete![/bold green]\n")
        
        # Create summary table
        table = Table(title="Results Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="yellow", width=30)
        table.add_column("Value", style="green")
        
        # Extract key metrics
        tests = results.get("tests", {})
        
        # Quality score
        if "quality_validation" in tests and "quality_score" in tests["quality_validation"]:
            table.add_row("Overall Quality Score", f"{tests['quality_validation']['quality_score']:.1f}%")
        
        # Inference speed
        if "inference_speed" in tests:
            if "single_token_latency" in tests["inference_speed"]:
                latencies = list(tests["inference_speed"]["single_token_latency"].values())
                if latencies:
                    avg_latency = sum(latencies) / len(latencies)
                    table.add_row("Avg Single Token Latency", f"{avg_latency:.2f}ms")
        
        # Load testing
        if "load_testing" in tests:
            max_users = max([int(k.split('_')[0]) for k in tests["load_testing"].keys()])
            if f"{max_users}_users" in tests["load_testing"]:
                data = tests["load_testing"][f"{max_users}_users"]
                if "throughput" in data:
                    table.add_row(f"Throughput ({max_users} users)", f"{data['throughput']:.2f} req/s")
        
        # Reliability
        if "reliability" in tests and "summary" in tests["reliability"]:
            uptime = tests["reliability"]["summary"].get("uptime_percentage", 0)
            table.add_row("Uptime", f"{uptime:.1f}%")
        
        console.print(table)
        
        # Report location
        output_dir = results.get("metadata", {}).get("config", {}).get("reporting", {}).get("output_dir", "validation_results")
        console.print(f"\n[bold]Reports saved to:[/bold] [blue]{output_dir}[/blue]")
    else:
        print("\nValidation Complete!")
        print(f"Reports saved to: {results.get('metadata', {}).get('config', {}).get('reporting', {}).get('output_dir', 'validation_results')}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Woolly Performance Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with default config
  python run_validation.py
  
  # Run quick validation tests
  python run_validation.py --quick
  
  # Run specific test category
  python run_validation.py --test inference_speed
  
  # Use custom configuration
  python run_validation.py --config my_config.json
  
  # Skip server check (if running remotely)
  python run_validation.py --skip-server-check
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="validation_config.json",
        help="Path to validation configuration file"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation tests (subset with shorter duration)"
    )
    
    parser.add_argument(
        "--test",
        type=str,
        choices=[
            "inference_speed", "resource_utilization", "model_loading",
            "quality_validation", "load_testing", "comparative", "reliability"
        ],
        help="Run specific test category"
    )
    
    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Skip checking if Woolly server is running"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install required dependencies"
    )
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed!")
        return
    
    print_header()
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"\nError: Configuration file '{args.config}' not found!")
        print("Using default configuration...")
    
    # Load config for summary
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print_config_summary(config)
    except Exception as e:
        print(f"Warning: Could not load config for preview: {e}")
    
    # Check if Woolly server is running
    if not args.skip_server_check:
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Checking Woolly server...", total=None)
                server_running = check_woolly_server()
        else:
            print("\nChecking Woolly server...")
            server_running = check_woolly_server()
        
        if not server_running:
            print("\n[ERROR] Woolly server is not running!")
            print("Please start the Woolly server before running validation tests.")
            print("\nExample: woolly serve --model models/llama-3.2-1b-instruct.gguf")
            sys.exit(1)
        else:
            if RICH_AVAILABLE:
                console.print("[green]✓[/green] Woolly server is running")
            else:
                print("✓ Woolly server is running")
    
    # Run tests
    try:
        if args.quick:
            if RICH_AVAILABLE:
                console.print("\n[yellow]Running quick validation tests...[/yellow]")
            else:
                print("\nRunning quick validation tests...")
            results = asyncio.run(run_quick_tests(args.config))
        elif args.test:
            if RICH_AVAILABLE:
                console.print(f"\n[yellow]Running {args.test} test...[/yellow]")
            else:
                print(f"\nRunning {args.test} test...")
            results = asyncio.run(run_specific_test(args.test, args.config))
        else:
            if RICH_AVAILABLE:
                console.print("\n[yellow]Running full validation suite...[/yellow]")
                console.print("[dim]This may take a while depending on your configuration.[/dim]")
            else:
                print("\nRunning full validation suite...")
                print("This may take a while depending on your configuration.")
            
            # Import and run
            from woolly_true_validator import main as validator_main
            asyncio.run(validator_main())
            return
        
        # Print results summary
        print_results_summary(results)
        
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()