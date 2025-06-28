"""
Report Generator Module
Generates comprehensive HTML and PDF reports with visualizations
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
import io

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from jinja2 import Template

# Optional PDF generation
try:
    import weasyprint
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    

class ReportGenerator:
    """Generates comprehensive validation reports"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Set style for plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def generate_html_report(self, results: Dict[str, Any], output_path: Path):
        """Generate HTML report with embedded visualizations"""
        self.logger.info(f"Generating HTML report: {output_path}")
        
        # Generate visualizations
        charts = self._generate_all_charts(results)
        
        # Create HTML content
        html_content = self._create_html_content(results, charts)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {output_path}")
    
    def generate_pdf_report(self, results: Dict[str, Any], output_path: Path):
        """Generate PDF report from HTML"""
        if not PDF_AVAILABLE:
            self.logger.warning("PDF generation not available (install weasyprint)")
            return
        
        self.logger.info(f"Generating PDF report: {output_path}")
        
        # First generate HTML
        html_path = output_path.with_suffix('.html')
        self.generate_html_report(results, html_path)
        
        # Convert to PDF
        try:
            html = weasyprint.HTML(filename=str(html_path))
            html.write_pdf(str(output_path))
            self.logger.info(f"PDF report generated: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate PDF: {e}")
    
    def _generate_all_charts(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate all visualization charts"""
        charts = {}
        
        # Inference speed charts
        if "inference_speed" in results.get("tests", {}):
            charts["latency_comparison"] = self._create_latency_chart(
                results["tests"]["inference_speed"]
            )
            charts["throughput_chart"] = self._create_throughput_chart(
                results["tests"]["inference_speed"]
            )
            charts["context_scaling"] = self._create_context_scaling_chart(
                results["tests"]["inference_speed"]
            )
        
        # Resource utilization charts
        if "resource_utilization" in results.get("tests", {}):
            charts["resource_timeline"] = self._create_resource_timeline(
                results["tests"]["resource_utilization"]
            )
            charts["resource_summary"] = self._create_resource_summary(
                results["tests"]["resource_utilization"]
            )
        
        # Load testing charts
        if "load_testing" in results.get("tests", {}):
            charts["load_test_latency"] = self._create_load_test_latency_chart(
                results["tests"]["load_testing"]
            )
            charts["load_test_throughput"] = self._create_load_test_throughput_chart(
                results["tests"]["load_testing"]
            )
        
        # Quality validation charts
        if "quality_validation" in results.get("tests", {}):
            charts["quality_scores"] = self._create_quality_scores_chart(
                results["tests"]["quality_validation"]
            )
        
        # Comparative benchmarks
        if "comparative" in results.get("tests", {}):
            charts["comparative_performance"] = self._create_comparative_chart(
                results["tests"]["comparative"]
            )
        
        # Reliability charts
        if "reliability" in results.get("tests", {}):
            charts["reliability_timeline"] = self._create_reliability_timeline(
                results["tests"]["reliability"]
            )
        
        return charts
    
    def _create_latency_chart(self, data: Dict[str, Any]) -> str:
        """Create latency comparison chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract latency data
        if "single_token_latency" in data:
            prompts = list(data["single_token_latency"].keys())
            latencies = list(data["single_token_latency"].values())
            
            # Truncate long prompts for display
            prompts = [p[:30] + "..." if len(p) > 30 else p for p in prompts]
            
            bars = ax.bar(prompts, latencies)
            
            # Color bars based on performance
            colors = ['green' if l < 50 else 'orange' if l < 100 else 'red' for l in latencies]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_ylabel('Latency (ms)')
            ax.set_xlabel('Prompt')
            ax.set_title('Single Token Latency by Prompt')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}ms',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_throughput_chart(self, data: Dict[str, Any]) -> str:
        """Create throughput comparison chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if "sustained_throughput" in data:
            token_counts = []
            throughputs = []
            
            for key, value in data["sustained_throughput"].items():
                if isinstance(value, dict) and "tokens_per_second" in value:
                    token_counts.append(key)
                    throughputs.append(value["tokens_per_second"])
            
            if token_counts and throughputs:
                ax.plot(token_counts, throughputs, 'o-', linewidth=2, markersize=8)
                ax.set_xlabel('Token Count Configuration')
                ax.set_ylabel('Tokens per Second')
                ax.set_title('Sustained Throughput Performance')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for x, y in zip(token_counts, throughputs):
                    ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                              xytext=(0,10), ha='center')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_context_scaling_chart(self, data: Dict[str, Any]) -> str:
        """Create context scaling performance chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if "context_scaling" in data:
            context_sizes = []
            latencies = []
            
            for key, value in sorted(data["context_scaling"].items()):
                if isinstance(value, dict) and "latency" in value and value.get("success", False):
                    size = int(key.split('_')[0])
                    context_sizes.append(size)
                    latencies.append(value["latency"])
            
            if context_sizes and latencies:
                ax.plot(context_sizes, latencies, 'o-', linewidth=2, markersize=8)
                ax.set_xlabel('Context Size (tokens)')
                ax.set_ylabel('Latency (seconds)')
                ax.set_title('Latency vs Context Size')
                ax.set_xscale('log')
                ax.grid(True, alpha=0.3)
                
                # Add reference line for linear scaling
                if len(context_sizes) > 1:
                    x_ref = np.array(context_sizes)
                    y_ref = latencies[0] * (x_ref / context_sizes[0])
                    ax.plot(x_ref, y_ref, '--', alpha=0.5, label='Linear scaling')
                    ax.legend()
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_resource_timeline(self, data: Dict[str, Any]) -> str:
        """Create resource utilization timeline"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        if "time_series" in data:
            ts_data = data["time_series"]
            timestamps = ts_data.get("timestamps", [])
            
            # CPU usage
            if "cpu_percent" in ts_data:
                ax1.plot(timestamps, ts_data["cpu_percent"], linewidth=1)
                ax1.set_title('CPU Usage Over Time')
                ax1.set_ylabel('CPU %')
                ax1.set_ylim(0, 100)
                ax1.grid(True, alpha=0.3)
            
            # Memory usage
            if "memory_percent" in ts_data:
                ax2.plot(timestamps, ts_data["memory_percent"], linewidth=1, color='orange')
                ax2.set_title('Memory Usage Over Time')
                ax2.set_ylabel('Memory %')
                ax2.set_ylim(0, 100)
                ax2.grid(True, alpha=0.3)
            
            # Memory absolute
            if "memory_used" in ts_data:
                memory_gb = [m / (1024**3) for m in ts_data["memory_used"]]
                ax3.plot(timestamps, memory_gb, linewidth=1, color='green')
                ax3.set_title('Memory Usage (Absolute)')
                ax3.set_ylabel('Memory (GB)')
                ax3.set_xlabel('Time (seconds)')
                ax3.grid(True, alpha=0.3)
            
            # Combined view
            if "cpu_percent" in ts_data and "memory_percent" in ts_data:
                ax4.plot(timestamps, ts_data["cpu_percent"], label='CPU %', linewidth=1)
                ax4.plot(timestamps, ts_data["memory_percent"], label='Memory %', linewidth=1)
                ax4.set_title('Resource Usage Summary')
                ax4.set_ylabel('Usage %')
                ax4.set_xlabel('Time (seconds)')
                ax4.set_ylim(0, 100)
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_resource_summary(self, data: Dict[str, Any]) -> str:
        """Create resource utilization summary chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # CPU statistics
        if "cpu" in data:
            cpu_stats = data["cpu"]
            metrics = ['Mean', 'Max', 'P95', 'P99']
            values = [
                cpu_stats.get("mean", 0),
                cpu_stats.get("max", 0),
                cpu_stats.get("percentiles", {}).get("p95", 0),
                cpu_stats.get("percentiles", {}).get("p99", 0)
            ]
            
            bars = ax1.bar(metrics, values)
            ax1.set_ylabel('CPU Usage %')
            ax1.set_title('CPU Usage Statistics')
            ax1.set_ylim(0, 100)
            
            # Color code bars
            for bar, val in zip(bars, values):
                if val > 80:
                    bar.set_color('red')
                elif val > 60:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
                
                # Add value label
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.1f}%', ha='center', va='bottom')
        
        # Memory statistics
        if "memory" in data and "percent" in data["memory"]:
            mem_stats = data["memory"]["percent"]
            metrics = ['Mean', 'Max', 'Min']
            values = [
                mem_stats.get("mean", 0),
                mem_stats.get("max", 0),
                mem_stats.get("min", 0)
            ]
            
            bars = ax2.bar(metrics, values)
            ax2.set_ylabel('Memory Usage %')
            ax2.set_title('Memory Usage Statistics')
            ax2.set_ylim(0, 100)
            
            # Color code bars
            for bar, val in zip(bars, values):
                if val > 80:
                    bar.set_color('red')
                elif val > 60:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
                
                # Add value label
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_load_test_latency_chart(self, data: Dict[str, Any]) -> str:
        """Create load test latency comparison chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        user_counts = []
        p50_latencies = []
        p95_latencies = []
        p99_latencies = []
        
        for user_config, results in sorted(data.items()):
            if isinstance(results, dict) and "latency" in results:
                users = int(user_config.split('_')[0])
                user_counts.append(users)
                
                percentiles = results["latency"].get("percentiles", {})
                p50_latencies.append(percentiles.get("p50", 0))
                p95_latencies.append(percentiles.get("p95", 0))
                p99_latencies.append(percentiles.get("p99", 0))
        
        if user_counts:
            ax.plot(user_counts, p50_latencies, 'o-', label='P50', linewidth=2, markersize=8)
            ax.plot(user_counts, p95_latencies, 's-', label='P95', linewidth=2, markersize=8)
            ax.plot(user_counts, p99_latencies, '^-', label='P99', linewidth=2, markersize=8)
            
            ax.set_xlabel('Concurrent Users')
            ax.set_ylabel('Latency (seconds)')
            ax.set_title('Request Latency vs Concurrent Users')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set x-axis to show all user counts
            ax.set_xticks(user_counts)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_load_test_throughput_chart(self, data: Dict[str, Any]) -> str:
        """Create load test throughput chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        user_counts = []
        throughputs = []
        success_rates = []
        
        for user_config, results in sorted(data.items()):
            if isinstance(results, dict) and "throughput" in results:
                users = int(user_config.split('_')[0])
                user_counts.append(users)
                throughputs.append(results["throughput"])
                success_rates.append(results.get("success_rate", 100))
        
        if user_counts:
            # Create bar chart for throughput
            bars = ax.bar(user_counts, throughputs, alpha=0.7)
            
            # Color bars based on success rate
            for bar, success_rate in zip(bars, success_rates):
                if success_rate < 90:
                    bar.set_color('red')
                elif success_rate < 95:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
            
            # Add success rate as line plot
            ax2 = ax.twinx()
            ax2.plot(user_counts, success_rates, 'ro-', linewidth=2, markersize=8)
            ax2.set_ylabel('Success Rate %', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, 105)
            
            ax.set_xlabel('Concurrent Users')
            ax.set_ylabel('Throughput (req/s)')
            ax.set_title('Throughput and Success Rate vs Concurrent Users')
            ax.set_xticks(user_counts)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_quality_scores_chart(self, data: Dict[str, Any]) -> str:
        """Create quality validation scores chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = []
        scores = []
        
        # Extract scores from different quality tests
        if "correctness" in data:
            for category, result in data["correctness"].items():
                if isinstance(result, dict) and "correctness_score" in result:
                    categories.append(f"Correctness: {category}")
                    scores.append(result["correctness_score"] * 100)
        
        if "numerical_stability" in data:
            categories.append("Numerical Stability")
            scores.append(data["numerical_stability"].get("overall_accuracy", 0) * 100)
        
        if "determinism" in data:
            categories.append("Determinism")
            scores.append(100 if data["determinism"].get("is_deterministic", False) else 0)
        
        if categories and scores:
            bars = ax.barh(categories, scores)
            
            # Color code based on score
            for bar, score in zip(bars, scores):
                if score >= 90:
                    bar.set_color('green')
                elif score >= 70:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            ax.set_xlabel('Score (%)')
            ax.set_title('Quality Validation Scores')
            ax.set_xlim(0, 100)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                       f'{width:.1f}%', ha='left', va='center')
        
        # Add overall quality score if available
        if "quality_score" in data:
            ax.axvline(x=data["quality_score"], color='blue', linestyle='--', 
                      linewidth=2, label=f'Overall: {data["quality_score"]:.1f}%')
            ax.legend()
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_comparative_chart(self, data: Dict[str, Any]) -> str:
        """Create comparative performance chart"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        implementations = ['Woolly']
        test_results = {}
        
        # Collect results for each implementation
        for impl_name, impl_data in data.items():
            if isinstance(impl_data, dict) and "tests" in impl_data:
                implementations.append(impl_name.capitalize())
                
                for test_name, test_data in impl_data["tests"].items():
                    if test_name not in test_results:
                        test_results[test_name] = {'Woolly': 0}
                    
                    if isinstance(test_data, dict) and "tokens_per_second" in test_data:
                        test_results[test_name][impl_name.capitalize()] = test_data["tokens_per_second"]
                
                # Get Woolly results from comparison
                if "woolly_comparison" in impl_data:
                    for test_name, test_data in impl_data["woolly_comparison"].items():
                        if test_name in test_results and isinstance(test_data, dict):
                            test_results[test_name]['Woolly'] = test_data.get("tokens_per_second", 0)
        
        if test_results:
            # Create grouped bar chart
            x = np.arange(len(test_results))
            width = 0.25
            
            for i, impl in enumerate(implementations):
                values = [test_results[test].get(impl, 0) for test in test_results]
                offset = (i - len(implementations)/2) * width + width/2
                bars = ax.bar(x + offset, values, width, label=impl)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Test Type')
            ax.set_ylabel('Tokens per Second')
            ax.set_title('Performance Comparison: Woolly vs Other Implementations')
            ax.set_xticks(x)
            ax.set_xticklabels(list(test_results.keys()))
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_reliability_timeline(self, data: Dict[str, Any]) -> str:
        """Create reliability test timeline"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        if "checks" in data:
            timestamps = []
            statuses = []
            response_times = []
            
            for check in data["checks"]:
                timestamps.append(check["timestamp"] / 60)  # Convert to minutes
                statuses.append(1 if check["status"] == "healthy" else 0)
                
                if check.get("response_time") is not None:
                    response_times.append(check["response_time"] * 1000)  # Convert to ms
                else:
                    response_times.append(None)
            
            # Health status plot
            ax1.fill_between(timestamps, 0, statuses, step='post', alpha=0.7,
                           color='green', label='Healthy')
            ax1.fill_between(timestamps, statuses, 1, step='post', alpha=0.7,
                           color='red', label='Unhealthy')
            ax1.set_ylabel('Health Status')
            ax1.set_ylim(-0.1, 1.1)
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(['Unhealthy', 'Healthy'])
            ax1.legend()
            ax1.set_title('Service Health Over Time')
            
            # Response time plot
            valid_times = [(t, rt) for t, rt in zip(timestamps, response_times) if rt is not None]
            if valid_times:
                times, resp_times = zip(*valid_times)
                ax2.plot(times, resp_times, 'o-', markersize=4)
                ax2.set_ylabel('Response Time (ms)')
                ax2.set_xlabel('Time (minutes)')
                ax2.set_title('Health Check Response Times')
                ax2.grid(True, alpha=0.3)
        
        # Add summary statistics
        if "summary" in data:
            summary = data["summary"]
            uptime = summary.get("uptime_percentage", 0)
            
            # Add text box with summary
            textstr = f'Uptime: {uptime:.1f}%\nMax continuous uptime: {summary.get("max_continuous_uptime_minutes", 0):.0f} min'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def _create_html_content(self, results: Dict[str, Any], charts: Dict[str, str]) -> str:
        """Create HTML report content"""
        template_str = '''
<!DOCTYPE html>
<html>
<head>
    <title>Woolly Performance Validation Report</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        h3 {
            color: #666;
            margin-top: 20px;
        }
        .summary-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background: white;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #495057;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .status-healthy { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-error { color: #dc3545; }
        .timestamp {
            color: #6c757d;
            font-size: 14px;
        }
        .section {
            margin-bottom: 40px;
        }
        .test-result {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 10px 15px;
            margin: 10px 0;
        }
        .error {
            color: #dc3545;
            font-weight: bold;
        }
        .success {
            color: #28a745;
            font-weight: bold;
        }
        @media print {
            body { background: white; }
            .container { box-shadow: none; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Woolly Performance Validation Report</h1>
        
        <div class="summary-box">
            <p class="timestamp">Generated: {{ metadata.timestamp }}</p>
            <p>Version: {{ metadata.version }}</p>
        </div>
        
        <!-- Executive Summary -->
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary-box">
                {% if tests.quality_validation and tests.quality_validation.quality_score %}
                <div class="metric">
                    <div class="metric-value">{{ "%.1f"|format(tests.quality_validation.quality_score) }}%</div>
                    <div class="metric-label">Overall Quality Score</div>
                </div>
                {% endif %}
                
                {% if tests.reliability and tests.reliability.summary %}
                <div class="metric">
                    <div class="metric-value">{{ "%.1f"|format(tests.reliability.summary.uptime_percentage) }}%</div>
                    <div class="metric-label">Uptime</div>
                </div>
                {% endif %}
                
                {% if tests.load_testing %}
                <div class="metric">
                    <div class="metric-value">{{ tests.load_testing|length }}</div>
                    <div class="metric-label">Load Test Scenarios</div>
                </div>
                {% endif %}
            </div>
        </div>
        
        <!-- Inference Speed Results -->
        {% if tests.inference_speed %}
        <div class="section">
            <h2>Inference Speed Performance</h2>
            
            {% if charts.latency_comparison %}
            <div class="chart-container">
                <h3>Single Token Latency</h3>
                <img src="data:image/png;base64,{{ charts.latency_comparison }}" alt="Latency Comparison">
            </div>
            {% endif %}
            
            {% if charts.throughput_chart %}
            <div class="chart-container">
                <h3>Sustained Throughput</h3>
                <img src="data:image/png;base64,{{ charts.throughput_chart }}" alt="Throughput Chart">
            </div>
            {% endif %}
            
            {% if charts.context_scaling %}
            <div class="chart-container">
                <h3>Context Scaling Performance</h3>
                <img src="data:image/png;base64,{{ charts.context_scaling }}" alt="Context Scaling">
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <!-- Resource Utilization -->
        {% if tests.resource_utilization %}
        <div class="section">
            <h2>Resource Utilization</h2>
            
            {% if charts.resource_timeline %}
            <div class="chart-container">
                <h3>Resource Usage Timeline</h3>
                <img src="data:image/png;base64,{{ charts.resource_timeline }}" alt="Resource Timeline">
            </div>
            {% endif %}
            
            {% if charts.resource_summary %}
            <div class="chart-container">
                <h3>Resource Usage Summary</h3>
                <img src="data:image/png;base64,{{ charts.resource_summary }}" alt="Resource Summary">
            </div>
            {% endif %}
            
            <!-- System Info Table -->
            {% if tests.resource_utilization.system_info %}
            <h3>System Information</h3>
            <table>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Platform</td>
                    <td>{{ tests.resource_utilization.system_info.platform }}</td>
                </tr>
                <tr>
                    <td>CPU Count</td>
                    <td>{{ tests.resource_utilization.system_info.cpu_count }}</td>
                </tr>
                <tr>
                    <td>Total Memory</td>
                    <td>{{ "%.2f"|format(tests.resource_utilization.system_info.memory_total / (1024**3)) }} GB</td>
                </tr>
                {% if tests.resource_utilization.system_info.gpu_available %}
                <tr>
                    <td>GPU Available</td>
                    <td class="success">Yes</td>
                </tr>
                {% endif %}
            </table>
            {% endif %}
        </div>
        {% endif %}
        
        <!-- Load Testing Results -->
        {% if tests.load_testing %}
        <div class="section">
            <h2>Load Testing Results</h2>
            
            {% if charts.load_test_latency %}
            <div class="chart-container">
                <h3>Latency vs Concurrent Users</h3>
                <img src="data:image/png;base64,{{ charts.load_test_latency }}" alt="Load Test Latency">
            </div>
            {% endif %}
            
            {% if charts.load_test_throughput %}
            <div class="chart-container">
                <h3>Throughput vs Concurrent Users</h3>
                <img src="data:image/png;base64,{{ charts.load_test_throughput }}" alt="Load Test Throughput">
            </div>
            {% endif %}
            
            <!-- Load Test Summary Table -->
            <h3>Load Test Summary</h3>
            <table>
                <tr>
                    <th>Concurrent Users</th>
                    <th>Success Rate</th>
                    <th>Throughput (req/s)</th>
                    <th>P95 Latency (s)</th>
                    <th>P99 Latency (s)</th>
                </tr>
                {% for config, data in tests.load_testing.items()|sort %}
                {% if data.latency %}
                <tr>
                    <td>{{ config.split('_')[0] }}</td>
                    <td class="{% if data.success_rate >= 95 %}success{% elif data.success_rate >= 90 %}status-warning{% else %}error{% endif %}">
                        {{ "%.1f"|format(data.success_rate) }}%
                    </td>
                    <td>{{ "%.2f"|format(data.throughput) }}</td>
                    <td>{{ "%.2f"|format(data.latency.percentiles.p95) }}</td>
                    <td>{{ "%.2f"|format(data.latency.percentiles.p99) }}</td>
                </tr>
                {% endif %}
                {% endfor %}
            </table>
        </div>
        {% endif %}
        
        <!-- Quality Validation -->
        {% if tests.quality_validation %}
        <div class="section">
            <h2>Quality Validation</h2>
            
            {% if charts.quality_scores %}
            <div class="chart-container">
                <h3>Quality Scores by Category</h3>
                <img src="data:image/png;base64,{{ charts.quality_scores }}" alt="Quality Scores">
            </div>
            {% endif %}
            
            <!-- Determinism Test -->
            {% if tests.quality_validation.determinism %}
            <div class="test-result">
                <h3>Determinism Test</h3>
                <p>Temperature 0.0 determinism: 
                    <span class="{% if tests.quality_validation.determinism.is_deterministic %}success{% else %}error{% endif %}">
                        {{ "PASS" if tests.quality_validation.determinism.is_deterministic else "FAIL" }}
                    </span>
                </p>
                {% if not tests.quality_validation.determinism.is_deterministic %}
                <p>Found {{ tests.quality_validation.determinism.unique_outputs|length }} unique outputs in {{ tests.quality_validation.determinism.num_runs }} runs</p>
                {% endif %}
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <!-- Comparative Benchmarks -->
        {% if tests.comparative %}
        <div class="section">
            <h2>Comparative Benchmarks</h2>
            
            {% if charts.comparative_performance %}
            <div class="chart-container">
                <h3>Performance Comparison</h3>
                <img src="data:image/png;base64,{{ charts.comparative_performance }}" alt="Comparative Performance">
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <!-- Reliability Testing -->
        {% if tests.reliability %}
        <div class="section">
            <h2>Reliability Testing</h2>
            
            {% if charts.reliability_timeline %}
            <div class="chart-container">
                <h3>Service Health Timeline</h3>
                <img src="data:image/png;base64,{{ charts.reliability_timeline }}" alt="Reliability Timeline">
            </div>
            {% endif %}
            
            {% if tests.reliability.summary %}
            <div class="summary-box">
                <h3>Reliability Summary</h3>
                <p>Total Checks: {{ tests.reliability.summary.total_checks }}</p>
                <p>Uptime: <span class="{% if tests.reliability.summary.uptime_percentage >= 99 %}success{% elif tests.reliability.summary.uptime_percentage >= 95 %}status-warning{% else %}error{% endif %}">
                    {{ "%.2f"|format(tests.reliability.summary.uptime_percentage) }}%
                </span></p>
                <p>Max Continuous Uptime: {{ tests.reliability.summary.max_continuous_uptime_minutes }} minutes</p>
                <p>Memory Leaks Detected: 
                    <span class="{% if tests.reliability.summary.memory_leaks_detected %}error{% else %}success{% endif %}">
                        {{ "Yes" if tests.reliability.summary.memory_leaks_detected else "No" }}
                    </span>
                </p>
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <!-- Configuration Details -->
        <div class="section">
            <h2>Test Configuration</h2>
            <details>
                <summary>Click to view full configuration</summary>
                <pre>{{ metadata.config|tojson(indent=2) }}</pre>
            </details>
        </div>
        
    </div>
</body>
</html>
        '''
        
        template = Template(template_str)
        return template.render(
            metadata=results.get("metadata", {}),
            tests=results.get("tests", {}),
            charts=charts
        )