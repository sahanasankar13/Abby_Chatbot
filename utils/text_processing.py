import re
import logging
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Clean text by removing extra whitespace, normalizing, etc.
    
    Args:
        text (str): Text to clean
    
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def detect_language(text):
    """
    Detect the language of the given text
    
    Args:
        text (str): Text to analyze
    
    Returns:
        str: Language code (e.g., 'en' for English)
    """
    try:
        if not text or len(text.strip()) < 10:
            return 'en'  # Default to English for short texts
        
        return detect(text)
    except LangDetectException as e:
        logger.warning(f"Language detection error: {str(e)}")
        return 'en'  # Default to English on error

def extract_keywords(text, max_keywords=5):
    """
    Extract important keywords from text
    
    Args:
        text (str): Text to analyze
        max_keywords (int): Maximum number of keywords to extract
    
    Returns:
        list: Extracted keywords
    """
    # This is a simple keyword extraction based on word frequency
    # In a real implementation, you might use NLTK, spaCy, or another NLP library
    
    if not text:
        return []
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stopwords
    stopwords = {'the', 'and', 'or', 'but', 'because', 'as', 'what', 'when', 
                'where', 'how', 'why', 'who', 'which', 'this', 'that', 'these', 
                'those', 'with', 'for', 'from', 'to', 'at', 'by', 'about', 'like'}
    filtered_words = [word for word in words if word not in stopwords]
    
    # Count word frequency
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in keywords[:max_keywords]]

def is_question(text):
    """
    Check if text is likely a question
    
    Args:
        text (str): Text to analyze
    
    Returns:
        bool: True if text is likely a question, False otherwise
    """
    # Simple heuristics to detect questions
    # More sophisticated analysis could be used in a real implementation
    
    text = text.strip()
    
    # Check for question marks
    if '?' in text:
        return True
    
    # Check for question words at the beginning
    question_starters = ['what', 'who', 'when', 'where', 'why', 'how', 'is', 'are', 
                         'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should']
    first_word = text.lower().split()[0] if text else ''
    
    return first_word in question_starters
