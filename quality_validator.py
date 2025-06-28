"""
Quality Validation Module
Validates output quality, consistency, and correctness
"""

import asyncio
import aiohttp
import hashlib
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import difflib
import statistics


class QualityValidator:
    """Validates quality and consistency of model outputs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_url = f"http://{config['woolly']['host']}:{config['woolly']['port']}"
        self.api_key = config['woolly'].get('api_key')
        self.timeout = config['woolly'].get('inference_timeout', config['woolly'].get('timeout', 600))
        
        # Test cases for quality validation
        self.test_cases = [
            {
                "category": "factual",
                "prompt": "What is the capital of France?",
                "expected_keywords": ["Paris"],
                "max_tokens": 50
            },
            {
                "category": "math",
                "prompt": "What is 25 + 17?",
                "expected_keywords": ["42"],
                "max_tokens": 50
            },
            {
                "category": "code",
                "prompt": "Write a Python function to calculate factorial of n:",
                "expected_keywords": ["def", "factorial", "return"],
                "max_tokens": 200
            },
            {
                "category": "reasoning",
                "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
                "expected_keywords": ["no", "cannot conclude", "logical"],
                "max_tokens": 200
            },
            {
                "category": "context_preservation",
                "prompt": "Remember this number: 7823. Now tell me a short story about a dog. At the end, remind me what number I asked you to remember.",
                "expected_keywords": ["7823"],
                "max_tokens": 300
            }
        ]
    
    async def run_all_validations(self, consistency_runs: int = 10, 
                                 temperature: float = 0.0) -> Dict[str, Any]:
        """Run all quality validation tests"""
        self.logger.info("Starting quality validation tests...")
        
        results = {
            "consistency": await self._test_output_consistency(consistency_runs, temperature),
            "correctness": await self._test_output_correctness(),
            "numerical_stability": await self._test_numerical_stability(),
            "context_preservation": await self._test_context_preservation(),
            "determinism": await self._test_determinism(),
            "output_format": await self._test_output_format_compliance()
        }
        
        # Calculate overall quality score
        results["quality_score"] = self._calculate_quality_score(results)
        
        return results
    
    async def _test_output_consistency(self, runs: int, temperature: float) -> Dict[str, Any]:
        """Test consistency of outputs across multiple runs"""
        self.logger.info(f"Testing output consistency with {runs} runs...")
        
        consistency_results = {}
        
        for test_case in self.test_cases[:3]:  # Use subset for consistency testing
            prompt = test_case["prompt"]
            outputs = []
            
            # Generate multiple outputs
            for i in range(runs):
                output = await self._generate_completion(
                    prompt, 
                    max_tokens=test_case["max_tokens"],
                    temperature=temperature
                )
                if output:
                    outputs.append(output)
            
            if len(outputs) >= 2:
                # Calculate consistency metrics
                consistency_metrics = self._calculate_consistency_metrics(outputs)
                consistency_results[test_case["category"]] = {
                    "prompt": prompt,
                    "num_outputs": len(outputs),
                    "metrics": consistency_metrics,
                    "sample_outputs": outputs[:3]  # Include first 3 for inspection
                }
            else:
                consistency_results[test_case["category"]] = {
                    "error": "Insufficient outputs generated"
                }
        
        return consistency_results
    
    async def _test_output_correctness(self) -> Dict[str, Any]:
        """Test correctness of outputs against expected results"""
        self.logger.info("Testing output correctness...")
        
        correctness_results = {}
        
        for test_case in self.test_cases:
            output = await self._generate_completion(
                test_case["prompt"],
                max_tokens=test_case["max_tokens"],
                temperature=0.0
            )
            
            if output:
                # Check for expected keywords
                keywords_found = []
                keywords_missing = []
                
                output_lower = output.lower()
                for keyword in test_case["expected_keywords"]:
                    if keyword.lower() in output_lower:
                        keywords_found.append(keyword)
                    else:
                        keywords_missing.append(keyword)
                
                correctness_results[test_case["category"]] = {
                    "prompt": test_case["prompt"],
                    "output": output,
                    "expected_keywords": test_case["expected_keywords"],
                    "keywords_found": keywords_found,
                    "keywords_missing": keywords_missing,
                    "correctness_score": len(keywords_found) / len(test_case["expected_keywords"])
                }
            else:
                correctness_results[test_case["category"]] = {
                    "error": "Failed to generate output"
                }
        
        return correctness_results
    
    async def _test_numerical_stability(self) -> Dict[str, Any]:
        """Test numerical stability and calculation accuracy"""
        self.logger.info("Testing numerical stability...")
        
        numerical_tests = [
            {
                "prompt": "Calculate: 123 + 456 = ",
                "expected": 579,
                "type": "addition"
            },
            {
                "prompt": "Calculate: 1000 - 237 = ",
                "expected": 763,
                "type": "subtraction"
            },
            {
                "prompt": "Calculate: 12 * 15 = ",
                "expected": 180,
                "type": "multiplication"
            },
            {
                "prompt": "Calculate: 144 / 12 = ",
                "expected": 12,
                "type": "division"
            },
            {
                "prompt": "What is 15% of 200?",
                "expected": 30,
                "type": "percentage"
            }
        ]
        
        results = {}
        correct_count = 0
        
        for test in numerical_tests:
            output = await self._generate_completion(
                test["prompt"],
                max_tokens=50,
                temperature=0.0
            )
            
            if output:
                # Extract number from output
                extracted_number = self._extract_number(output)
                
                if extracted_number is not None:
                    error = abs(extracted_number - test["expected"])
                    relative_error = error / test["expected"] if test["expected"] != 0 else float('inf')
                    is_correct = error < 0.01  # Allow small floating point errors
                    
                    if is_correct:
                        correct_count += 1
                    
                    results[test["type"]] = {
                        "prompt": test["prompt"],
                        "expected": test["expected"],
                        "output": output,
                        "extracted": extracted_number,
                        "error": error,
                        "relative_error": relative_error,
                        "correct": is_correct
                    }
                else:
                    results[test["type"]] = {
                        "prompt": test["prompt"],
                        "expected": test["expected"],
                        "output": output,
                        "error": "Could not extract number"
                    }
            else:
                results[test["type"]] = {
                    "error": "Failed to generate output"
                }
        
        accuracy = correct_count / len(numerical_tests) if numerical_tests else 0
        results["overall_accuracy"] = accuracy
        
        return results
    
    async def _test_context_preservation(self) -> Dict[str, Any]:
        """Test ability to preserve context across long prompts"""
        self.logger.info("Testing context preservation...")
        
        context_tests = [
            {
                "name": "number_memory",
                "setup": "Remember these numbers: 42, 17, 93.",
                "distractor": "Now let me tell you about the weather. It's sunny today with a temperature of 75 degrees. The forecast shows rain tomorrow.",
                "query": "What were the three numbers I asked you to remember?",
                "expected": ["42", "17", "93"]
            },
            {
                "name": "instruction_following",
                "setup": "I want you to always end your responses with the word 'DONE'.",
                "distractor": "Tell me about Python programming.",
                "query": "What is a list comprehension?",
                "expected": ["DONE"]
            },
            {
                "name": "context_switching",
                "setup": "We're discussing cars. My favorite is a blue Tesla Model 3.",
                "distractor": "Python is a great programming language for data science.",
                "query": "What car did I mention earlier?",
                "expected": ["Tesla", "Model 3", "blue"]
            }
        ]
        
        results = {}
        
        for test in context_tests:
            # Build full prompt with context
            full_prompt = f"{test['setup']}\n\n{test['distractor']}\n\n{test['query']}"
            
            output = await self._generate_completion(
                full_prompt,
                max_tokens=100,
                temperature=0.0
            )
            
            if output:
                # Check for expected elements
                found_elements = []
                missing_elements = []
                
                output_lower = output.lower()
                for element in test["expected"]:
                    if element.lower() in output_lower:
                        found_elements.append(element)
                    else:
                        missing_elements.append(element)
                
                preservation_score = len(found_elements) / len(test["expected"])
                
                results[test["name"]] = {
                    "full_prompt": full_prompt,
                    "output": output,
                    "expected_elements": test["expected"],
                    "found_elements": found_elements,
                    "missing_elements": missing_elements,
                    "preservation_score": preservation_score
                }
            else:
                results[test["name"]] = {
                    "error": "Failed to generate output"
                }
        
        return results
    
    async def _test_determinism(self) -> Dict[str, Any]:
        """Test deterministic behavior with temperature=0"""
        self.logger.info("Testing determinism...")
        
        test_prompt = "Complete this sentence: The capital of France is"
        num_runs = 5
        
        outputs = []
        for i in range(num_runs):
            output = await self._generate_completion(
                test_prompt,
                max_tokens=10,
                temperature=0.0,
                seed=42  # Use fixed seed if supported
            )
            if output:
                outputs.append(output.strip())
        
        # Check if all outputs are identical
        unique_outputs = list(set(outputs))
        is_deterministic = len(unique_outputs) == 1
        
        return {
            "prompt": test_prompt,
            "num_runs": num_runs,
            "is_deterministic": is_deterministic,
            "unique_outputs": unique_outputs,
            "all_outputs": outputs
        }
    
    async def _test_output_format_compliance(self) -> Dict[str, Any]:
        """Test compliance with requested output formats"""
        self.logger.info("Testing output format compliance...")
        
        format_tests = [
            {
                "name": "json_format",
                "prompt": 'Output a JSON object with keys "name" and "age" for a person:',
                "validator": self._validate_json_format
            },
            {
                "name": "list_format",
                "prompt": "List three programming languages, one per line:",
                "validator": self._validate_list_format
            },
            {
                "name": "markdown_format",
                "prompt": "Write a markdown heading and a bullet list with 2 items:",
                "validator": self._validate_markdown_format
            }
        ]
        
        results = {}
        
        for test in format_tests:
            output = await self._generate_completion(
                test["prompt"],
                max_tokens=100,
                temperature=0.0
            )
            
            if output:
                is_valid, details = test["validator"](output)
                
                results[test["name"]] = {
                    "prompt": test["prompt"],
                    "output": output,
                    "format_valid": is_valid,
                    "validation_details": details
                }
            else:
                results[test["name"]] = {
                    "error": "Failed to generate output"
                }
        
        return results
    
    async def _generate_completion(self, prompt: str, max_tokens: int, 
                                 temperature: float, seed: Optional[int] = None) -> Optional[str]:
        """Generate a completion from the model"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        session_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/v1/inference/complete",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("choices", [{}])[0].get("text", "")
                    else:
                        self.logger.error(f"Request failed: {response.status}")
                        return None
                        
            except Exception as e:
                self.logger.error(f"Error generating completion: {e}")
                return None
    
    def _calculate_consistency_metrics(self, outputs: List[str]) -> Dict[str, float]:
        """Calculate consistency metrics for a set of outputs"""
        if len(outputs) < 2:
            return {}
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                # Use sequence matcher for similarity
                ratio = difflib.SequenceMatcher(None, outputs[i], outputs[j]).ratio()
                similarities.append(ratio)
        
        # Calculate semantic hashes (simple approach)
        semantic_hashes = []
        for output in outputs:
            # Normalize and hash
            normalized = ' '.join(output.lower().split())
            hash_val = hashlib.md5(normalized.encode()).hexdigest()
            semantic_hashes.append(hash_val)
        
        unique_hashes = len(set(semantic_hashes))
        
        # Length statistics
        lengths = [len(output) for output in outputs]
        
        return {
            "mean_similarity": statistics.mean(similarities) if similarities else 0,
            "min_similarity": min(similarities) if similarities else 0,
            "max_similarity": max(similarities) if similarities else 0,
            "unique_outputs": unique_hashes,
            "length_mean": statistics.mean(lengths),
            "length_std": statistics.stdev(lengths) if len(lengths) > 1 else 0,
            "consistency_score": statistics.mean(similarities) if similarities else 0
        }
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract a number from text"""
        import re
        
        # Look for numbers in the text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        if numbers:
            try:
                # Return the first valid number
                return float(numbers[0])
            except ValueError:
                return None
        return None
    
    def _validate_json_format(self, output: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate JSON format"""
        try:
            # Try to find JSON in the output
            start_idx = output.find('{')
            end_idx = output.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = output[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Check for expected keys
                has_name = "name" in parsed
                has_age = "age" in parsed
                
                return True, {
                    "valid_json": True,
                    "has_name_key": has_name,
                    "has_age_key": has_age,
                    "parsed_content": parsed
                }
        except:
            pass
        
        return False, {"valid_json": False, "error": "Could not parse JSON"}
    
    def _validate_list_format(self, output: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate list format"""
        lines = output.strip().split('\n')
        
        # Filter out empty lines
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Check if we have multiple lines
        is_valid = len(non_empty_lines) >= 2
        
        return is_valid, {
            "num_lines": len(non_empty_lines),
            "lines": non_empty_lines[:5],  # First 5 lines
            "appears_to_be_list": is_valid
        }
    
    def _validate_markdown_format(self, output: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate markdown format"""
        has_heading = any(line.strip().startswith('#') for line in output.split('\n'))
        has_bullet = any(line.strip().startswith(('*', '-', '+')) for line in output.split('\n'))
        
        bullet_count = sum(1 for line in output.split('\n') 
                          if line.strip().startswith(('*', '-', '+')))
        
        is_valid = has_heading and has_bullet
        
        return is_valid, {
            "has_heading": has_heading,
            "has_bullets": has_bullet,
            "bullet_count": bullet_count,
            "valid_markdown": is_valid
        }
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall quality score from all test results"""
        scores = []
        
        # Consistency score
        if "consistency" in results:
            consistency_scores = []
            for category, data in results["consistency"].items():
                if isinstance(data, dict) and "metrics" in data:
                    consistency_scores.append(data["metrics"].get("consistency_score", 0))
            if consistency_scores:
                scores.append(statistics.mean(consistency_scores))
        
        # Correctness score
        if "correctness" in results:
            correctness_scores = []
            for category, data in results["correctness"].items():
                if isinstance(data, dict) and "correctness_score" in data:
                    correctness_scores.append(data["correctness_score"])
            if correctness_scores:
                scores.append(statistics.mean(correctness_scores))
        
        # Numerical stability score
        if "numerical_stability" in results:
            scores.append(results["numerical_stability"].get("overall_accuracy", 0))
        
        # Context preservation score
        if "context_preservation" in results:
            preservation_scores = []
            for test_name, data in results["context_preservation"].items():
                if isinstance(data, dict) and "preservation_score" in data:
                    preservation_scores.append(data["preservation_score"])
            if preservation_scores:
                scores.append(statistics.mean(preservation_scores))
        
        # Determinism score
        if "determinism" in results:
            scores.append(1.0 if results["determinism"].get("is_deterministic", False) else 0.0)
        
        # Format compliance score
        if "output_format" in results:
            format_scores = []
            for format_name, data in results["output_format"].items():
                if isinstance(data, dict) and "format_valid" in data:
                    format_scores.append(1.0 if data["format_valid"] else 0.0)
            if format_scores:
                scores.append(statistics.mean(format_scores))
        
        # Calculate overall score
        return statistics.mean(scores) * 100 if scores else 0.0