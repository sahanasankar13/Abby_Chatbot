"""
Test Topic Detection System

This script tests the reproductive health topic detection system of the chatbot,
verifying that it correctly identifies topics across various reproductive health categories.
"""

import sys
import os
import json
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot.baseline_model import BaselineModel
from chatbot.question_classifier import QuestionClassifier
from utils.metrics import get_metrics, flush_metrics

def test_topic_detection():
    """Test the reproductive health topic detection functionality"""
    print("\n=== Testing Reproductive Health Topic Detection ===")
    
    # Initialize the model
    model = BaselineModel()
    
    # Test questions by expected topic
    topic_questions = {
        "pregnancy_planning": [
            "What should I do before trying to get pregnant?",
            "How can I increase my chances of conception?",
            "What vitamins should I take when TTC?",
            "Are there fertility tracking methods that work?",
            "How long should I try before seeing a fertility specialist?"
        ],
        "pregnancy": [
            "What are early signs of pregnancy?",
            "Is spotting normal in the first trimester?",
            "How does the baby develop each week?",
            "What foods should I avoid while pregnant?",
            "When will I feel the baby kick?"
        ],
        "birth_control": [
            "What birth control method is most effective?",
            "How does an IUD work?",
            "What are the side effects of the pill?",
            "Can antibiotics make birth control less effective?",
            "How long after stopping the pill can I get pregnant?"
        ],
        "menstruation": [
            "Why is my period irregular?",
            "What causes severe menstrual cramps?",
            "Is it normal to have very heavy periods?",
            "Can stress affect my menstrual cycle?",
            "Why did my period stop unexpectedly?"
        ],
        "reproductive_health": [
            "What are symptoms of PCOS?",
            "How is endometriosis diagnosed?",
            "What STIs can cause infertility?",
            "How often should I get a pap smear?",
            "What causes recurring yeast infections?"
        ],
        "abortion": [
            "What is the difference between medical and surgical abortion?",
            "How long does recovery take after an abortion?",
            "What should I expect during a medication abortion?",
            "Are there emotional side effects after abortion?",
            "What are my options if I'm pregnant and don't want to be?"
        ]
    }
    
    results = {}
    total_correct = 0
    total_tests = 0
    
    for expected_topic, questions in topic_questions.items():
        print(f"\nTesting topic: {expected_topic}")
        results[expected_topic] = {"correct": 0, "total": len(questions), "questions": []}
        
        for question in questions:
            # Get the detected topic
            detected_topic = model._get_reproductive_health_topic(question)
            
            # Check if correct
            is_correct = detected_topic == expected_topic
            if is_correct:
                total_correct += 1
                results[expected_topic]["correct"] += 1
            
            total_tests += 1
            
            # Store result
            results[expected_topic]["questions"].append({
                "question": question,
                "detected": detected_topic,
                "correct": is_correct
            })
            
            # Print result
            status = "✓" if is_correct else "✗"
            print(f"{status} Question: '{question}'")
            if not is_correct:
                print(f"   Expected: {expected_topic}, Detected: {detected_topic}")
    
    # Calculate accuracy
    accuracy = (total_correct / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nOverall accuracy: {accuracy:.2f}% ({total_correct}/{total_tests})")
    
    # Print topic-wise accuracy
    print("\nTopic-wise accuracy:")
    for topic, data in results.items():
        topic_accuracy = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
        print(f"{topic}: {topic_accuracy:.2f}% ({data['correct']}/{data['total']})")
    
    return results


def test_question_categorization():
    """Test the question categorization functionality"""
    print("\n=== Testing Question Categorization ===")
    
    # Initialize the classifier
    classifier = QuestionClassifier()
    
    # Test questions by expected category
    category_questions = {
        "policy": [
            "Is abortion legal in Texas?",
            "What are the abortion laws in California?",
            "Do I need parental consent for abortion in Florida?",
            "What are the gestational limits for abortion in New York?",
            "Are there waiting periods for abortion in Missouri?"
        ],
        "knowledge": [
            "What are the signs of pregnancy?",
            "How does an IUD work?",
            "What are the stages of fetal development?",
            "How effective is the morning after pill?",
            "What happens during a pap smear?"
        ],
        "conversational": [
            "Hello, how are you?",
            "Thank you for the information.",
            "Can you help me understand this?",
            "That's good to know.",
            "I'm worried about my health."
        ],
        "out_of_scope": [
            "What's the weather like today?",
            "Can you recommend a good movie?",
            "How do I file my taxes?",
            "What's the capital of France?",
            "How do I change my password?"
        ]
    }
    
    results = {}
    total_correct = 0
    total_tests = 0
    
    for expected_category, questions in category_questions.items():
        print(f"\nTesting category: {expected_category}")
        results[expected_category] = {"correct": 0, "total": len(questions), "questions": []}
        
        for question in questions:
            # Get the classified category
            classification = classifier.classify_question(question)
            detected_category = classification.get("primary_category", "unknown")
            
            # Check if correct
            is_correct = detected_category == expected_category
            if is_correct:
                total_correct += 1
                results[expected_category]["correct"] += 1
            
            total_tests += 1
            
            # Store result
            results[expected_category]["questions"].append({
                "question": question,
                "detected": detected_category,
                "classification": classification,
                "correct": is_correct
            })
            
            # Print result
            status = "✓" if is_correct else "✗"
            print(f"{status} Question: '{question}'")
            if not is_correct:
                print(f"   Expected: {expected_category}, Detected: {detected_category}")
    
    # Calculate accuracy
    accuracy = (total_correct / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nOverall accuracy: {accuracy:.2f}% ({total_correct}/{total_tests})")
    
    # Print category-wise accuracy
    print("\nCategory-wise accuracy:")
    for category, data in results.items():
        category_accuracy = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
        print(f"{category}: {category_accuracy:.2f}% ({data['correct']}/{data['total']})")
    
    return results


def test_topic_detection_with_inference_time():
    """Test topic detection with inference time tracking"""
    print("\n=== Testing Topic Detection with Inference Time Tracking ===")
    
    # Initialize the model
    model = BaselineModel()
    
    # Reset metrics
    flush_metrics(reset=True)
    
    # Sample questions from different topics
    mixed_questions = [
        {"question": "What vitamins should I take before getting pregnant?", "expected_topic": "pregnancy_planning"},
        {"question": "What are early signs of pregnancy?", "expected_topic": "pregnancy"},
        {"question": "How effective is the pill?", "expected_topic": "birth_control"},
        {"question": "Why is my period irregular?", "expected_topic": "menstruation"},
        {"question": "What are common STI symptoms?", "expected_topic": "reproductive_health"},
        {"question": "What happens during a medication abortion?", "expected_topic": "abortion"}
    ]
    
    for test_case in mixed_questions:
        question = test_case["question"]
        expected_topic = test_case["expected_topic"]
        
        print(f"\nQuestion: '{question}'")
        print(f"Expected topic: {expected_topic}")
        
        # Process the question (this will trigger metrics recording)
        response = model.process_question(question, force_category="knowledge")
        
        # Get the detected topic
        detected_topic = model._get_reproductive_health_topic(question)
        
        # Check if correct
        is_correct = detected_topic == expected_topic
        status = "✓" if is_correct else "✗"
        
        print(f"Detected topic: {detected_topic} {status}")
        print(f"Response length: {len(response)} chars")
    
    # Get the metrics
    metrics = get_metrics()
    
    # Print inference time metrics
    print("\nInference Time Metrics:")
    if "timings" in metrics:
        timing_metrics = metrics["timings"]
        
        # Overall inference time
        if "inference_time" in timing_metrics:
            print(f"Average inference time: {timing_metrics['inference_time']:.2f}ms")
        
        # Category-specific inference times
        if "inference_time_by_category" in timing_metrics:
            category_times = timing_metrics["inference_time_by_category"]
            print("\nCategory-specific inference times:")
            for category, time in category_times.items():
                print(f"{category}: {time:.2f}ms")
        
        # Topic-specific inference times
        if "inference_time_by_topic" in timing_metrics:
            topic_times = timing_metrics["inference_time_by_topic"]
            print("\nTopic-specific inference times:")
            for topic, time in topic_times.items():
                print(f"{topic}: {time:.2f}ms")
    
    return metrics


def run_tests():
    """Run all topic detection tests"""
    # Run the tests
    topic_results = test_topic_detection()
    category_results = test_question_categorization()
    inference_metrics = test_topic_detection_with_inference_time()
    
    # Combine results
    all_results = {
        "topic_detection": topic_results,
        "question_categorization": category_results,
        "inference_metrics": inference_metrics
    }
    
    # Save the results
    with open("topic_detection_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nTest results saved to topic_detection_test_results.json")


if __name__ == "__main__":
    run_tests()