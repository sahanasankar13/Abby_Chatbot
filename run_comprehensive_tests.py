"""
Comprehensive Test Runner for Reproductive Health Chatbot

This script runs all the test suites for the reproductive health chatbot,
including metrics tracking, citation system, topic detection, and ROUGE metrics,
and compiles the results into a comprehensive report.
"""

import os
import sys
import json
import time
import importlib
import subprocess
from datetime import datetime
from typing import Dict, Any, List

# Test modules to run
TEST_MODULES = [
    "scripts.test_metrics_tracking",
    "scripts.test_rouge_metrics",
    "scripts.test_citation_system",
    "scripts.test_topic_detection"
]

# Result files to collect
RESULT_FILES = [
    "metrics_test_results.json",
    "rouge_metrics_test_results.json",
    "citation_system_test_results.json",
    "topic_detection_test_results.json"
]

def run_test_module(module_name: str) -> bool:
    """
    Run a test module and return whether it succeeded
    
    Args:
        module_name: Name of the module to run
    
    Returns:
        bool: True if test succeeded, False otherwise
    """
    print(f"\n{'=' * 80}")
    print(f"Running test module: {module_name}")
    print(f"{'=' * 80}")
    
    try:
        # Convert module name to script path
        script_name = module_name.replace(".", "/") + ".py"
        
        # Run the script as a subprocess to isolate it
        result = subprocess.run([sys.executable, script_name], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Test failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
        
        # Print the output
        print(result.stdout)
        
        return True
    except Exception as e:
        print(f"Error running test module {module_name}: {str(e)}")
        return False


def collect_result_files() -> Dict[str, Any]:
    """
    Collect and combine all test result files
    
    Returns:
        dict: Combined results from all test files
    """
    combined_results = {}
    
    for file_name in RESULT_FILES:
        if os.path.exists(file_name):
            try:
                with open(file_name, 'r') as f:
                    results = json.load(f)
                
                # Add to combined results
                test_name = file_name.replace("_results.json", "")
                combined_results[test_name] = results
                
                print(f"Collected results from {file_name}")
            except Exception as e:
                print(f"Error reading results from {file_name}: {str(e)}")
        else:
            print(f"Warning: Result file {file_name} not found")
    
    return combined_results


def generate_summary_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary report from all test results
    
    Args:
        results: Combined results from all tests
    
    Returns:
        dict: Summary report
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "modules_tested": len(results),
        "overall_metrics": {},
        "topic_detection": {},
        "citation_system": {},
        "response_quality": {}
    }
    
    # Extract topic detection accuracy
    if "topic_detection" in results:
        topic_data = results["topic_detection"].get("topic_detection", {})
        
        # Calculate overall accuracy
        total_correct = 0
        total_tests = 0
        
        for topic, data in topic_data.items():
            if isinstance(data, dict) and "correct" in data and "total" in data:
                total_correct += data["correct"]
                total_tests += data["total"]
        
        if total_tests > 0:
            summary["topic_detection"]["accuracy"] = (total_correct / total_tests) * 100
        else:
            summary["topic_detection"]["accuracy"] = 0
    
    # Extract citation metrics
    if "citation_system" in results:
        citation_metrics = results["citation_system"].get("metrics", {})
        citation_counters = citation_metrics.get("counters", {})
        
        # Count citation-related metrics
        citation_counters_filtered = {k: v for k, v in citation_counters.items() 
                                    if "citation" in k.lower()}
        
        if citation_counters_filtered:
            summary["citation_system"]["counters"] = citation_counters_filtered
    
    # Extract ROUGE metrics
    if "rouge_metrics" in results:
        rouge_data = results["rouge_metrics"].get("rouge_metrics", {})
        if rouge_data:
            summary["response_quality"]["rouge"] = rouge_data
    
    # Extract overall performance metrics
    for result_key, result_data in results.items():
        if "metrics" in result_data:
            metrics = result_data["metrics"]
            
            # Extract timings
            if "timings" in metrics:
                timings = metrics["timings"]
                
                for timing_key, timing_value in timings.items():
                    if isinstance(timing_value, (int, float)):
                        # Add directly if it's a simple value
                        if "timings" not in summary["overall_metrics"]:
                            summary["overall_metrics"]["timings"] = {}
                        summary["overall_metrics"]["timings"][timing_key] = timing_value
                    elif isinstance(timing_value, dict):
                        # For category or topic specific timings, compute averages
                        if timing_key not in summary["overall_metrics"]:
                            summary["overall_metrics"][timing_key] = {}
                        
                        for category, value in timing_value.items():
                            summary["overall_metrics"][timing_key][category] = value
    
    return summary


def run_all_tests() -> Dict[str, Any]:
    """
    Run all test modules and compile results
    
    Returns:
        dict: Comprehensive test results
    """
    print(f"\n{'=' * 80}")
    print("Starting Comprehensive Test Suite")
    print(f"{'=' * 80}")
    
    start_time = time.time()
    
    # Run each test module
    results = {}
    for module_name in TEST_MODULES:
        success = run_test_module(module_name)
        results[module_name] = {"success": success}
    
    # Collect and compile results
    compiled_results = collect_result_files()
    
    # Generate summary report
    summary = generate_summary_report(compiled_results)
    
    # Calculate total test time
    elapsed_time = time.time() - start_time
    summary["test_duration_seconds"] = elapsed_time
    
    # Combine everything into a final report
    final_report = {
        "summary": summary,
        "module_status": results,
        "detailed_results": compiled_results
    }
    
    return final_report


if __name__ == "__main__":
    # Run all tests
    report = run_all_tests()
    
    # Save comprehensive report
    report_filename = f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"Comprehensive test report saved to: {report_filename}")
    print(f"{'=' * 80}")
    
    # Print summary
    summary = report["summary"]
    print("\nTest Summary:")
    print(f"Total test modules: {len(report['module_status'])}")
    print(f"Successful modules: {sum(1 for status in report['module_status'].values() if status['success'])}")
    print(f"Total test duration: {summary['test_duration_seconds']:.2f} seconds")
    
    if "topic_detection" in summary and "accuracy" in summary["topic_detection"]:
        print(f"Topic detection accuracy: {summary['topic_detection']['accuracy']:.2f}%")
    
    if "response_quality" in summary and "rouge" in summary["response_quality"]:
        rouge = summary["response_quality"]["rouge"]
        print("\nResponse quality metrics:")
        for metric, value in rouge.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nSee the comprehensive report for detailed results.")