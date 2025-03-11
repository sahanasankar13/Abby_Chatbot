import os
import csv
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_reproductive_health_data():
    """
    Load reproductive health Q&A data from CSV
    
    Returns:
        list: List of dictionaries containing question-answer pairs
    """
    try:
        logger.info("Loading reproductive health data")
        
        # Try to find the data file
        data_paths = [
            'attached_assets/Planned Parenthood Data - Sahana (1).csv',
            'Planned Parenthood Data - Sahana (1).csv',
            './data/Planned Parenthood Data.csv'
        ]
        
        data_file = None
        for path in data_paths:
            if os.path.exists(path):
                data_file = path
                break
        
        if not data_file:
            logger.warning("Data file not found. Using fallback data.")
            return _get_fallback_data()
        
        # Load data from CSV
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
        
        # Convert to list of dictionaries
        qa_pairs = []
        for _, row in df.iterrows():
            qa_pairs.append({
                'Question': row['Question'],
                'Answer': row['Answer'],
                'Link': row.get('Link', ''),
                'Category': row.get('LLM_Category', 'General')
            })
        
        logger.info(f"Loaded {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    
    except Exception as e:
        logger.error(f"Error loading reproductive health data: {str(e)}", exc_info=True)
        return _get_fallback_data()

def _get_fallback_data():
    """
    Provide fallback data if the data file is not found
    
    Returns:
        list: List of basic Q&A pairs
    """
    logger.info("Using fallback data")
    
    return [
        {
            'Question': 'What kinds of emergency contraception are there?',
            'Answer': 'There are 2 ways to prevent pregnancy after unprotected sex: You can get certain IUDs within 120 hours (five days) after having unprotected sex (most effective), or take an emergency contraception pill (morning-after pill) within 120 hours. There are two types of morning-after pills: one with ulipristal acetate (ella, requires prescription) and one with levonorgestrel (Plan B, available over-the-counter).',
            'Link': 'https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception/which-kind-emergency-contraception-should-i-use',
            'Category': 'Birth Control'
        },
        {
            'Question': 'How does emergency contraception work?',
            'Answer': 'Emergency contraception works by preventing or delaying ovulation. Sperm can live in your body up to 6 days after sex, waiting for an egg. Morning-after pills temporarily stop ovulation, preventing the sperm from meeting an egg. They do not cause abortion and won\'t work if you\'re already pregnant.',
            'Link': 'https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception/which-kind-emergency-contraception-should-i-use',
            'Category': 'Birth Control'
        },
        {
            'Question': 'How long do I have to get emergency contraception?',
            'Answer': 'You can use emergency contraception up to 5 days (120 hours) after unprotected sex. It\'s important to act quickly. IUDs and ella are effective throughout all 5 days, while Plan B and other levonorgestrel pills work best within the first 3 days (72 hours).',
            'Link': 'https://www.plannedparenthood.org/learn/morning-after-pill-emergency-contraception/which-kind-emergency-contraception-should-i-use',
            'Category': 'Birth Control'
        }
    ]

def preprocess_question_answer_pairs(qa_pairs):
    """
    Preprocess question-answer pairs for better model performance
    
    Args:
        qa_pairs (list): List of QA pair dictionaries
    
    Returns:
        list: Processed QA pairs
    """
    processed_pairs = []
    
    for pair in qa_pairs:
        # Clean and process the text
        question = pair['Question'].strip()
        answer = pair['Answer'].strip()
        
        # Skip empty or very short Q&As
        if len(question) < 5 or len(answer) < 10:
            continue
        
        processed_pairs.append({
            'Question': question,
            'Answer': answer,
            'Link': pair.get('Link', ''),
            'Category': pair.get('Category', 'General')
        })
    
    return processed_pairs
