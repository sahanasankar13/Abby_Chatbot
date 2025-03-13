"""
Test metrics tracking functionality

This script tests the metrics tracking functionality by simulating questions across
different categories and topics, then verifying that the metrics are properly recorded.
"""

import sys
import os
import time
import json
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot.baseline_model import BaselineModel
from utils.metrics import get_metrics, flush_metrics

def simulate_question(model: BaselineModel, question: str, category: str = None) -> None:
    """
    Simulate asking a question to the chatbot model
    
    Args:
        model: The chatbot model
        question: The question to ask
        category: If provided, force this category
    """
    print(f"\nProcessing question: '{question}'")
    if category:
        print(f"Forced category: {category}")
        response = model.process_question(question, force_category=category)
    else:
        response = model.process_question(question)
    
    # Truncate long responses for display
    if len(response) > 100:
        response = response[:100] + "..."
    print(f"Response: {response}")


def test_by_category() -> None:
    """Test metrics tracking across different question categories"""
    print("\n=== Testing Metrics Tracking by Category ===")
    
    # Initialize the model
    model = BaselineModel()
    
    # Ask a policy question
    simulate_question(model, "What are the abortion laws in Texas?", "policy")
    
    # Ask a knowledge question
    simulate_question(model, "What are the symptoms of pregnancy?", "knowledge")
    
    # Ask a conversational question
    simulate_question(model, "How are you today?", "conversational")
    
    # Ask an out-of-scope question
    simulate_question(model, "What is the weather like today?", "out_of_scope")


def test_by_topic() -> None:
    """Test metrics tracking across different reproductive health topics"""
    print("\n=== Testing Metrics Tracking by Topic ===")
    
    # Initialize the model
    model = BaselineModel()
    
    # Test different reproductive health topics
    topics = {
        "pregnancy": "What are early signs of pregnancy?",
        "birth_control": "How effective is the pill?",
        "menstruation": "Why is my period irregular?",
        "abortion": "What is a medication abortion?",
        "pregnancy_planning": "What can I do to increase fertility?"
    }
    
    for topic, question in topics.items():
        print(f"\nTesting topic: {topic}")
        simulate_question(model, question, "knowledge")


def test_multi_query() -> None:
    """Test metrics tracking for multi-query questions"""
    print("\n=== Testing Metrics Tracking for Multi-Query Questions ===")
    
    # Initialize the model
    model = BaselineModel()
    
    # Test a multi-query question
    multi_query = "How effective is the pill and what are early signs of pregnancy?"
    simulate_question(model, multi_query)


def display_metrics(metrics: Dict[str, Any]) -> None:
    """Display the metrics in a readable format"""
    print("\n=== Metrics ===")
    
    # Display counters
    if "counters" in metrics:
        print("\nCounters:")
        for counter, value in metrics["counters"].items():
            print(f"  {counter}: {value}")
    
    # Display timing metrics
    if "timings" in metrics:
        print("\nTiming Metrics:")
        for metric, values in metrics["timings"].items():
            if isinstance(values, dict):
                print(f"  {metric}:")
                for submetric, subvalues in values.items():
                    if isinstance(subvalues, dict):
                        print(f"    {submetric}:")
                        for k, v in subvalues.items():
                            print(f"      {k}: {v:.2f}ms")
                    else:
                        print(f"    {submetric}: {subvalues:.2f}ms")
            else:
                print(f"  {metric}: {values:.2f}ms")
    
    # Display ROUGE metrics
    if "rouge_metrics" in metrics:
        print("\nROUGE Metrics:")
        for metric, value in metrics["rouge_metrics"].items():
            print(f"  {metric}: {value:.4f}")


def run_tests() -> None:
    """Run all tests and display the results"""
    # Reset metrics for clean test
    flush_metrics(reset=True)
    
    # Run the tests
    test_by_category()
    test_by_topic()
    test_multi_query()
    
    # Get and display the metrics
    metrics = get_metrics()
    display_metrics(metrics)
    
    # Save metrics to file for analysis
    with open("metrics_test_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to metrics_test_results.json")


if __name__ == "__main__":
    run_tests()