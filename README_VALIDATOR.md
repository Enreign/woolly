# Woolly True Performance Validator

A comprehensive, production-quality validation suite for measuring the TRUE performance of Woolly inference server without any shortcuts or simplified versions.

## Overview

This validation suite provides extensive performance testing, quality validation, and reliability measurements for Woolly. It measures:

- **Inference Speed**: Single token latency, sustained throughput, batch processing, context scaling
- **Resource Utilization**: Memory (RAM/GPU), CPU utilization, disk I/O, cache performance
- **Model Loading**: Cold start time, warm restart, memory footprint, success rate
- **Quality & Correctness**: Output consistency, numerical stability, context preservation
- **Scalability**: Concurrent users, load testing, model size scaling
- **Comparative Performance**: vs llama.cpp, vs Ollama benchmarks
- **Reliability**: Error rates, recovery time, stability, memory leak detection
- **User Experience**: Time to first token, inter-token latency, API response times

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. For GPU monitoring (optional):
```bash
pip install gputil
```

3. For PDF report generation (optional):
```bash
pip install weasyprint
```

## Quick Start

1. Start your Woolly server:
```bash
woolly serve --model models/llama-3.2-1b-instruct.gguf
```

2. Run the validation suite:
```bash
python run_validation.py
```

3. For quick tests (shorter duration):
```bash
python run_validation.py --quick
```

4. Run specific test category:
```bash
python run_validation.py --test inference_speed
```

## Configuration

The validation suite uses `validation_config.json` for configuration. Key sections:

### Woolly Server Configuration
```json
{
  "woolly": {
    "host": "localhost",
    "port": 8080,
    "api_key": null,
    "timeout": 300
  }
}
```

### Test Configuration

Each test category can be individually configured and enabled/disabled:

#### Inference Speed Tests
```json
"inference_speed": {
  "enabled": true,
  "prompts": ["Hello", "Explain quantum computing", ...],
  "max_tokens": [10, 50, 100, 200, 500],
  "batch_sizes": [1, 2, 4, 8, 16]
}
```

#### Load Testing
```json
"load_testing": {
  "enabled": true,
  "concurrent_users": [1, 5, 10, 20, 50],
  "duration": 300,
  "ramp_up_time": 30
}
```

#### Reliability Testing
```json
"reliability": {
  "enabled": true,
  "duration": 3600,
  "check_interval": 60,
  "memory_leak_detection": true
}
```

## Test Categories

### 1. Inference Speed (`inference_speed`)
- Measures single token generation latency
- Tests sustained throughput with various token counts
- Evaluates batch processing performance
- Analyzes performance scaling with context size

### 2. Resource Utilization (`resource_utilization`)
- Monitors CPU, memory, disk I/O during inference
- Tracks Woolly process-specific metrics
- GPU utilization (if available)
- Generates time-series data for analysis

### 3. Model Loading (`model_loading`)
- Cold start performance (first load)
- Warm restart times
- Memory footprint measurement
- Load success rate tracking

### 4. Quality Validation (`quality_validation`)
- Output consistency across multiple runs
- Correctness testing with known answers
- Numerical stability for calculations
- Context preservation over long prompts
- Determinism verification (temperature=0)
- Output format compliance

### 5. Load Testing (`load_testing`)
- Simulates concurrent users
- Measures throughput under load
- Tracks latency percentiles (P50, P95, P99)
- Time to first token (TTFT) metrics
- Inter-token latency analysis

### 6. Comparative Benchmarks (`comparative`)
- Performance comparison with llama.cpp
- Performance comparison with Ollama
- Side-by-side throughput analysis
- Relative performance metrics

### 7. Reliability Testing (`reliability`)
- Extended duration testing (default: 1 hour)
- Health check monitoring
- Memory leak detection
- Uptime percentage calculation
- Error pattern analysis

## Output and Reports

The validation suite generates comprehensive reports in multiple formats:

### HTML Report
- Interactive visualizations
- Performance charts and graphs
- Detailed test results
- System information

### PDF Report (optional)
- Printable version of HTML report
- Executive summary
- All charts and tables

### Raw Data
- JSON format with all test results
- Time-series data for further analysis
- Detailed metrics and measurements

Reports are saved to the `validation_results` directory by default.

## Architecture

### Core Modules

1. **woolly_true_validator.py**
   - Main orchestrator
   - Coordinates all test execution
   - Manages configuration
   - Generates final results

2. **performance_monitor.py**
   - System resource monitoring
   - Process-specific tracking
   - Memory leak detection
   - Real-time metrics collection

3. **load_tester.py**
   - Concurrent user simulation
   - Request timing and analysis
   - Throughput measurement
   - Reliability testing

4. **model_benchmarks.py**
   - Model loading tests
   - Inference performance
   - Comparative benchmarks
   - Context scaling tests

5. **quality_validator.py**
   - Output consistency checks
   - Correctness validation
   - Format compliance
   - Determinism testing

6. **report_generator.py**
   - HTML/PDF generation
   - Chart creation
   - Results visualization
   - Summary statistics

## Advanced Usage

### Custom Configuration
```bash
python run_validation.py --config custom_config.json
```

### Running Without Server Check
```bash
python run_validation.py --skip-server-check
```

### Programmatic Usage
```python
import asyncio
from woolly_true_validator import WoollyTrueValidator

async def run_custom_validation():
    validator = WoollyTrueValidator("custom_config.json")
    results = await validator.run_all_tests()
    return results

results = asyncio.run(run_custom_validation())
```

### Specific Test Execution
```python
validator = WoollyTrueValidator()
await validator._run_inference_speed_tests()
await validator._run_load_tests()
```

## Interpreting Results

### Key Metrics

1. **Single Token Latency**: Time to generate first token (lower is better)
   - Excellent: < 50ms
   - Good: 50-100ms
   - Acceptable: 100-200ms

2. **Throughput**: Tokens per second (higher is better)
   - Varies by model size and hardware
   - Compare with llama.cpp/Ollama baselines

3. **Quality Score**: Overall output quality (0-100%)
   - Based on correctness, consistency, determinism
   - > 90%: Excellent
   - > 80%: Good
   - < 70%: Needs investigation

4. **Uptime**: Percentage of successful health checks
   - Production target: > 99.9%
   - Development acceptable: > 95%

## Troubleshooting

### Common Issues

1. **Woolly server not found**
   - Ensure Woolly is running on configured host/port
   - Check firewall settings
   - Verify API endpoints

2. **High latency results**
   - Check model quantization level
   - Verify hardware acceleration
   - Monitor system resources

3. **Memory leaks detected**
   - Review memory growth rate
   - Check for increasing baseline
   - Monitor over extended periods

4. **Quality score low**
   - Review specific failing tests
   - Check model configuration
   - Verify temperature settings

## Contributing

To add new tests or metrics:

1. Add test method to appropriate module
2. Update configuration schema
3. Add visualization in report generator
4. Update documentation

## License

This validation suite is part of the Woolly project and follows the same license terms.