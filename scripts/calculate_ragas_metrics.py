"""
Calculate Ragas metrics for chatbot performance using Planned Parenthood data as ground truth.

This script evaluates the chatbot's performance using Ragas metrics by:
1. Loading the Planned Parenthood QA pairs as ground truth
2. Generating model responses for the questions
3. Calculating and storing Ragas metrics in the evaluation logs
"""

import os
import sys
import json
import logging
import datetime
from typing import List, Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from utils.data_loader import load_reproductive_health_data
from utils.advanced_metrics import AdvancedMetricsCalculator
from chatbot.baseline_model import BaselineModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_qa_data() -> List[Dict[str, str]]:
    """
    Load question-answer pairs from Planned Parenthood data
    
    Returns:
        List[Dict[str, str]]: List of QA pairs
    """
    logger.info("Loading Planned Parenthood QA data")
    qa_pairs = load_reproductive_health_data()
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")
    return qa_pairs

def prepare_evaluation_data(qa_pairs: List[Dict[str, str]], sample_size: int = 50) -> Tuple[List[str], List[str]]:
    """
    Prepare evaluation data from QA pairs
    
    Args:
        qa_pairs (List[Dict[str, str]]): List of QA pairs
        sample_size (int, optional): Number of samples to use. Defaults to 50.
    
    Returns:
        Tuple[List[str], List[str]]: Questions and ground truth answers
    """
    # Select a sample of questions to evaluate (to avoid long processing times)
    import random
    if len(qa_pairs) > sample_size:
        qa_sample = random.sample(qa_pairs, sample_size)
    else:
        qa_sample = qa_pairs
    
    # Extract questions and ground truth answers
    questions = [qa['Question'] for qa in qa_sample]
    ground_truth = [qa['Answer'] for qa in qa_sample]
    
    return questions, ground_truth

def generate_model_responses(questions: List[str], model: BaselineModel) -> Tuple[List[str], List[List[str]]]:
    """
    Generate model responses for the given questions
    
    Args:
        questions (List[str]): List of questions
        model (BaselineModel): The chatbot model
    
    Returns:
        Tuple[List[str], List[List[str]]]: Generated answers and contexts
    """
    logger.info("Generating model responses")
    answers = []
    all_contexts = []
    
    # Clear the model's context tracking to start fresh
    model.recent_responses = []
    model.recent_contexts = []
    
    for i, question in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}")
        
        # Use the model to get a response - this will automatically store context in model.recent_contexts
        # Force category to be 'knowledge' to ensure we use the BERT RAG model which provides contexts
        response = model.process_question(question, force_category='knowledge')
        answers.append(response)
        
        # Extract context for this question if available
        if i < len(model.recent_contexts):
            # Convert the context dictionaries to strings for Ragas
            context_texts = []
            for ctx in model.recent_contexts[i]:
                # Format context as a string containing the question and answer
                context_str = f"Question: {ctx.get('question', '')}\nAnswer: {ctx.get('answer', '')}"
                context_texts.append(context_str)
            all_contexts.append(context_texts)
        else:
            # Fallback to empty context if not found
            logger.warning(f"No context found for question {i+1}")
            all_contexts.append([])
    
    return answers, all_contexts

def calculate_ragas_metrics(questions: List[str], contexts: List[List[str]], 
                           generated_answers: List[str], ground_truth: List[str]) -> Dict[str, Any]:
    """
    Calculate Ragas metrics for the generated answers
    
    Args:
        questions (List[str]): List of questions
        contexts (List[List[str]]): Retrieved contexts
        generated_answers (List[str]): Generated answers
        ground_truth (List[str]): Ground truth answers
    
    Returns:
        Dict[str, Any]: Ragas metrics
    """
    logger.info("Calculating Ragas metrics")
    
    metrics_calculator = AdvancedMetricsCalculator()
    
    # Calculate Ragas metrics
    ragas_metrics = metrics_calculator.calculate_ragas_metrics(
        questions=questions,
        retrieved_contexts=contexts,
        generated_answers=generated_answers
    )
    
    # Calculate text similarity metrics
    similarity_metrics = metrics_calculator.calculate_text_similarity_metrics(
        references=ground_truth,
        predictions=generated_answers
    )
    
    # Combine metrics
    all_metrics = {
        "ragas": ragas_metrics,
        "text_similarity": similarity_metrics
    }
    
    return all_metrics

def save_metrics_to_log(metrics: Dict[str, Any], log_file: str = "evaluation_logs.json") -> None:
    """
    Save the calculated metrics to the evaluation log file
    
    Args:
        metrics (Dict[str, Any]): The metrics to save
        log_file (str, optional): Path to the log file. Defaults to "evaluation_logs.json".
    """
    logger.info(f"Saving metrics to {log_file}")
    
    # Create timestamp for this evaluation
    timestamp = datetime.datetime.now().isoformat()
    
    # Create log entry
    log_entry = {
        "timestamp": timestamp,
        "evaluation_type": "ragas",
        "metrics": metrics
    }
    
    # Load existing logs
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
    except json.JSONDecodeError:
        # If the file exists but is not valid JSON, start with an empty list
        logs = []
    
    # Append new log entry
    logs.append(log_entry)
    
    # Save updated logs
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    logger.info("Metrics saved successfully")

def main():
    """Main function to run the Ragas metrics calculation"""
    logger.info("Starting Ragas metrics calculation")
    
    # Load QA data
    qa_pairs = load_qa_data()
    
    # Prepare evaluation data
    questions, ground_truth = prepare_evaluation_data(qa_pairs, sample_size=20)
    
    # Initialize model
    model = BaselineModel()
    
    # Generate model responses
    generated_answers, contexts = generate_model_responses(questions, model)
    
    # Calculate metrics
    metrics = calculate_ragas_metrics(questions, contexts, generated_answers, ground_truth)
    
    # Save metrics to log
    save_metrics_to_log(metrics)
    
    logger.info("Ragas metrics calculation completed")
    
    # Print summary
    if "ragas" in metrics and "faithfulness" in metrics["ragas"]:
        logger.info(f"Faithfulness score: {metrics['ragas']['faithfulness']}")
    if "ragas" in metrics and "context_precision" in metrics["ragas"]:
        logger.info(f"Context precision score: {metrics['ragas']['context_precision']}")
    if "ragas" in metrics and "context_recall" in metrics["ragas"]:
        logger.info(f"Context recall score: {metrics['ragas']['context_recall']}")
    
    if "text_similarity" in metrics and "bleu" in metrics["text_similarity"]:
        logger.info(f"BLEU score: {metrics['text_similarity']['bleu']['score']}")
    if "text_similarity" in metrics and "rouge" in metrics["text_similarity"]:
        logger.info(f"ROUGE-L score: {metrics['text_similarity']['rouge']['rougeL']}")
    if "text_similarity" in metrics and "bert_score" in metrics["text_similarity"]:
        logger.info(f"BERTScore F1: {metrics['text_similarity']['bert_score']['f1']}")

if __name__ == "__main__":
    main()