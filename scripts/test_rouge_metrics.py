"""
Test ROUGE metrics tracking

This script tests the ROUGE metrics tracking functionality by comparing
sample reference texts with model-generated predictions.
"""

import sys
import os
import json
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.advanced_metrics import AdvancedMetricsCalculator
from utils.metrics import get_metrics, flush_metrics, record_rouge_metrics

def test_rouge_metrics():
    """Test ROUGE metrics calculation and tracking"""
    print("\n=== Testing ROUGE Metrics Calculation and Tracking ===")
    
    # Initialize the metrics calculator
    metrics_calculator = AdvancedMetricsCalculator()
    
    # Reset metrics for clean test
    flush_metrics(reset=True)
    
    # Sample reference-prediction pairs
    samples = [
        {
            "reference": "Pregnancy symptoms include missed periods, nausea, fatigue, and breast tenderness. You should consult a healthcare provider if you suspect you are pregnant.",
            "prediction": "Common symptoms of pregnancy include missed menstrual periods, morning sickness, fatigue, and tender breasts. If you think you might be pregnant, it's important to speak with a healthcare professional.",
            "quality": "high"
        },
        {
            "reference": "Birth control pills work by preventing ovulation. They are about 99% effective when used perfectly.",
            "prediction": "Birth control pills prevent pregnancy by stopping ovulation. When taken correctly, they are approximately 99% effective.",
            "quality": "high"
        },
        {
            "reference": "Abortion laws vary by state. Some states allow abortion until viability, while others have more restrictions.",
            "prediction": "Abortion regulations differ across states in the US. While some states permit abortions until fetal viability, others have implemented more restrictive laws.",
            "quality": "high"
        },
        {
            "reference": "Menstrual cycles typically last 28 days, but can range from 21 to 35 days. Menstruation usually lasts 3-7 days.",
            "prediction": "The menstrual cycle is usually around 28 days, but can vary between women.",
            "quality": "medium"
        },
        {
            "reference": "STI testing is recommended for sexually active individuals, especially when changing partners or if symptoms appear.",
            "prediction": "Getting tested for sexually transmitted infections is important.",
            "quality": "low"
        }
    ]
    
    # Test individual ROUGE score calculation
    print("\nTesting individual ROUGE score calculation:")
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1} ({sample['quality']} quality match):")
        print(f"Reference: {sample['reference']}")
        print(f"Prediction: {sample['prediction']}")
        
        # Calculate ROUGE scores
        rouge_scores = metrics_calculator.calculate_single_rouge_score(
            reference=sample['reference'],
            prediction=sample['prediction'],
            record_realtime=True
        )
        
        print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    # Test batch ROUGE calculation
    references = [sample["reference"] for sample in samples]
    predictions = [sample["prediction"] for sample in samples]
    
    print("\nTesting batch ROUGE calculation:")
    batch_scores = metrics_calculator.calculate_text_similarity_metrics(
        references=references,
        predictions=predictions,
        record_realtime_metrics=True
    )
    
    print(f"Average ROUGE-1: {batch_scores['rouge']['rouge1']:.4f}")
    print(f"Average ROUGE-2: {batch_scores['rouge']['rouge2']:.4f}")
    print(f"Average ROUGE-L: {batch_scores['rouge']['rougeL']:.4f}")
    
    # Get and display the tracked metrics
    metrics = get_metrics()
    
    print("\n=== Tracked ROUGE Metrics ===")
    if "rouge_metrics" in metrics:
        for metric, value in metrics["rouge_metrics"].items():
            print(f"{metric}: {value:.4f}")
    else:
        print("No ROUGE metrics were tracked.")
    
    # Save metrics to file
    with open("rouge_metrics_test_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to rouge_metrics_test_results.json")

if __name__ == "__main__":
    test_rouge_metrics()