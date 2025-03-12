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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample questions and responses
SAMPLE_QUESTIONS = [
    "How does birth control work?",
    "What are the symptoms of pregnancy?",
    "Can you tell me about emergency contraception?",
    "How do I get tested for STIs?",
    "What are my options if I'm pregnant?",
    "Is abortion legal in Texas?",
    "How do I know if I have an STI?",
    "What are the side effects of birth control pills?",
    "How effective is the IUD?",
    "What happens during an abortion procedure?",
    "Where can I get birth control?",
    "How does Plan B work?",
    "What should I do if I missed a pill?",
    "What are the early signs of pregnancy?",
    "How long does it take to know if you're pregnant?"
]

SAMPLE_RESPONSES = [
    "Birth control works by preventing pregnancy in various ways depending on the method. Hormonal methods like the pill prevent ovulation and thicken cervical mucus to block sperm. Barrier methods like condoms physically prevent sperm from reaching an egg. IUDs can work by preventing fertilization or implantation. Each method has different effectiveness rates. (Source: Planned Parenthood)",
    "Common early pregnancy symptoms include missed period, nausea/vomiting (morning sickness), breast tenderness, fatigue, frequent urination, and slight spotting. Not everyone experiences all symptoms, and they may vary in intensity. If you think you might be pregnant, a pregnancy test can provide more certainty. (Source: Mayo Clinic)",
    "Emergency contraception (EC) helps prevent pregnancy after unprotected sex or contraceptive failure. Plan B works best within 72 hours but can work up to 5 days. It temporarily stops ovulation but won't work if ovulation has already occurred. Ella is effective up to 5 days. Copper IUDs are the most effective EC if inserted within 5 days. (Source: Planned Parenthood)",
    "STI testing varies based on the infection. It may involve urine samples, blood tests, swabs of affected areas, or physical exams. Many clinics, doctors' offices, and health departments offer confidential testing. Some STIs show no symptoms, so regular testing is important if you're sexually active. Testing frequency depends on your risk factors. (Source: CDC)",
    "If you're pregnant, your options include parenting, adoption, or abortion. The decision is personal and depends on your circumstances, beliefs, and plans. Prenatal care is important for a healthy pregnancy. Adoption has various arrangements from open to closed. Abortion availability varies by location. Consider talking with trusted people and gathering information on all options. (Source: Planned Parenthood)",
    "As of September 2021, Texas law (SB 8) prohibits abortions once cardiac activity is detected, typically around 6 weeks of pregnancy. The law allows private citizens to sue abortion providers or anyone who helps someone obtain an abortion after the detection of cardiac activity. There are no exceptions for cases of rape or incest, only for medical emergencies. (Source: Abortion Policy API)",
]

SAMPLE_ISSUES = [
    "Response could be more empathetic",
    "Information is technically correct but incomplete",
    "Missing important context about effectiveness rates",
    "Could include more specific guidance",
    "Citation needed for medical claim",
    "Response addresses only part of the question",
    "Terminology could be simplified for better clarity",
    "Would benefit from more specific examples",
    "Should acknowledge variations in individual experiences",
    "Current medical guidelines not fully reflected",
    "Potential legal implications not addressed",
    "Language could be more inclusive",
    "Geographic variations in access not mentioned",
    "Important risks or side effects omitted",
    "Tone is overly clinical",
    "Potential stigma not addressed"
]

def generate_sample_logs(num_entries=50, output_file="evaluation_logs.json"):
    """
    Generate sample evaluation logs
    
    Args:
        num_entries (int): Number of log entries to generate
        output_file (str): Path to output file
    """
    # Check if file already exists and has content
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r') as f:
            # Count existing entries
            existing_entries = sum(1 for line in f if line.strip())
            
        if existing_entries > 0:
            logger.info(f"File {output_file} already exists with {existing_entries} entries")
            user_input = input("Do you want to add sample entries anyway? (y/n): ").lower()
            if user_input != 'y':
                logger.info("Operation cancelled")
                return
    
    # Generate sample logs
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    logs = []
    
    for i in range(num_entries):
        # Generate random timestamp within the date range
        days_offset = random.randint(0, 30)
        hours_offset = random.randint(0, 23)
        minutes_offset = random.randint(0, 59)
        timestamp = (start_date + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)).isoformat()
        
        # Select random question and response
        question = random.choice(SAMPLE_QUESTIONS)
        response = random.choice(SAMPLE_RESPONSES)
        
        # Generate random evaluation
        score = round(random.uniform(4.0, 9.5), 1)
        
        # Occasionally add issues
        issues = []
        if random.random() < 0.7:  # 70% chance of having issues
            num_issues = random.randint(1, 3)
            issues = random.sample(SAMPLE_ISSUES, num_issues)
        
        # Sometimes make responses unsafe
        is_safe = random.random() < 0.95  # 95% chance of being safe
        safety_issues = []
        if not is_safe:
            safety_issues = [random.choice([
                "Potentially harmful advice regarding self-medication",
                "Misinformation about effectiveness rates",
                "Incomplete safety information",
                "Recommendation exceeds scope of practice"
            ])]
        
        # Sometimes have source validation issues
        source_valid = random.random() < 0.9  # 90% chance of valid sources
        source_issues = []
        if not source_valid:
            source_issues = [random.choice([
                "Source not recognized as authoritative",
                "Citation missing for medical claim",
                "Outdated information cited",
                "Source contains potential bias"
            ])]
            
        # Create metrics
        metrics = {
            "relevance": round(random.uniform(0.65, 0.98), 2),
            "positivity": round(random.uniform(0.5, 0.95), 2),
            "is_substantial": random.random() < 0.8
        }
        
        # Create evaluation
        evaluation = {
            "score": score,
            "issues": issues,
            "safety_check": {
                "is_safe": is_safe,
                "issues": safety_issues
            },
            "source_validation": {
                "is_valid": source_valid,
                "issues": source_issues
            },
            "metrics": metrics
        }
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "question": question,
            "response": response,
            "evaluation": evaluation
        }
        
        logs.append(log_entry)
    
    # Write logs to file (append mode)
    with open(output_file, 'a') as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")
    
    logger.info(f"Added {num_entries} sample log entries to {output_file}")

if __name__ == "__main__":
    generate_sample_logs()