"""
Sample Metrics Generator for Development and Testing

This script generates sample evaluation logs for development and testing
of the metrics dashboard. It should only be used in development environments.
"""

import os
import json
import random
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_logs(num_entries=50, output_file="evaluation_logs.json"):
    """
    Generate sample evaluation logs
    
    Args:
        num_entries (int): Number of log entries to generate
        output_file (str): Path to output file
    """
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Sample question templates
    question_templates = [
        "What are the symptoms of {}?",
        "How does {} work?",
        "Can you tell me about {}?",
        "What is the process for {}?",
        "Is {} safe?",
        "How effective is {} for preventing pregnancy?",
        "What are the risks of {}?",
        "When should I consider {}?",
        "How long does {} take?",
        "What are my options for {} in {state}?",
    ]
    
    # Sample topics
    topics = [
        "birth control pills", "IUDs", "condoms", "emergency contraception",
        "pregnancy", "menstruation", "ovulation", "fertility tracking",
        "STI testing", "pap smears", "gynecological exams",
        "abortion", "medication abortion", "menopause", "PMS",
        "reproductive health", "family planning", "contraception"
    ]
    
    # Sample states
    states = [
        "California", "Texas", "New York", "Florida", "Illinois",
        "Ohio", "Georgia", "Mississippi", "Colorado", "Washington"
    ]
    
    # Generate sample log entries
    logs = []
    
    # Start date (30 days ago)
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(num_entries):
        # Generate random date between start date and now
        random_days = random.randint(0, 30)
        entry_date = start_date + timedelta(days=random_days)
        
        # Generate random question
        topic = random.choice(topics)
        template = random.choice(question_templates)
        question = template.format(topic, state=random.choice(states))
        
        # Generate simulated response characteristics
        is_improved = random.random() < 0.3  # 30% chance of improvement
        has_safety_issue = random.random() < 0.1  # 10% chance of safety issue
        
        # Generate a score between 1 and 10, weighted toward better scores
        score = max(1, min(10, random.normalvariate(7, 2)))
        
        # Safety issues
        safety_issues = []
        if has_safety_issue:
            possible_issues = [
                "Contains potentially harmful content: 'self-harm'",
                "Contains potentially misleading medical advice",
                "Suggests non-evidence-based treatments",
                "Missing important safety warnings"
            ]
            safety_issues = [random.choice(possible_issues)]
        
        # Create evaluation object
        evaluation = {
            "score": score,
            "improved": is_improved,
            "relevance_score": random.uniform(0.5, 1.0),
            "quality_score": random.uniform(0.5, 1.0),
            "safety": {
                "is_safe": not has_safety_issue,
                "score": random.uniform(0.7, 1.0) if not has_safety_issue else random.uniform(0.3, 0.7),
                "issues": safety_issues
            },
            "issues": []
        }
        
        if score < 7:
            possible_issues = [
                "Response does not fully address the question",
                "Response could be more positive and supportive",
                "Response lacks sufficient detail",
                "Sources not properly cited"
            ]
            evaluation["issues"] = random.sample(possible_issues, k=min(2, random.randint(1, 3)))
        
        # Create full log entry
        log_entry = {
            "timestamp": entry_date.isoformat(),
            "question": question,
            "response": "Sample response for: " + question,
            "evaluation": evaluation
        }
        
        logs.append(log_entry)
    
    # Write logs to file
    try:
        with open(output_file, 'w') as f:
            for log in logs:
                f.write(json.dumps(log) + "\n")
                
        logger.info(f"Successfully generated {num_entries} sample log entries in {output_file}")
    except Exception as e:
        logger.error(f"Error writing to log file: {str(e)}")
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample evaluation logs')
    parser.add_argument('--entries', type=int, default=50, help='Number of log entries to generate')
    parser.add_argument('--output', default="evaluation_logs.json", help='Output file path')
    
    args = parser.parse_args()
    generate_sample_logs(args.entries, args.output)