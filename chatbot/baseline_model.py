import os
import logging
from chatbot.bert_rag import BertRAGModel
from chatbot.gpt_integration import GPTModel
from chatbot.policy_api import PolicyAPI

logger = logging.getLogger(__name__)

class BaselineModel:
    """
    Baseline model that combines BERT-based RAG, GPT-4 integration, and policy API calls
    """
    def __init__(self):
        """
        Initialize the baseline model components
        """
        logger.info("Initializing Baseline Model")
        self.bert_rag = BertRAGModel()
        self.gpt_model = GPTModel()
        self.policy_api = PolicyAPI()
    
    def categorize_question(self, question):
        """
        Categorize the question to determine which model to use
        
        Args:
            question (str): The user's question
        
        Returns:
            str: Category of the question ('policy', 'knowledge', or 'conversational')
        """
        # Simple keyword-based categorization
        policy_keywords = ['law', 'legal', 'state', 'policy', 'ban', 'illegal', 'allowed', 'permit', 'legislation']
        
        question_lower = question.lower()
        
        # Check for policy-related keywords
        if any(keyword in question_lower for keyword in policy_keywords):
            return 'policy'
        
        # For questions that seem to be seeking specific information
        information_indicators = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'can i']
        if any(indicator in question_lower for indicator in information_indicators):
            return 'knowledge'
        
        # Default to conversational
        return 'conversational'
    
    def process_question(self, question):
        """
        Process a question using the appropriate model based on its category
        
        Args:
            question (str): The user's question
        
        Returns:
            str: The model's response
        """
        try:
            # Categorize the question
            category = self.categorize_question(question)
            logger.debug(f"Question category: {category}")
            
            # Process according to category
            if category == 'policy':
                logger.debug("Using Policy API for response")
                return self.policy_api.get_policy_response(question)
            
            elif category == 'knowledge':
                logger.debug("Using BERT RAG for response")
                # First try to get a response from the RAG model
                rag_response = self.bert_rag.get_response(question)
                
                # If RAG response is confident, return it
                if self.bert_rag.is_confident(question, rag_response):
                    return rag_response
                
                # If not confident, enhance with GPT
                logger.debug("RAG not confident, enhancing with GPT")
                return self.gpt_model.enhance_response(question, rag_response)
            
            else:  # conversational
                logger.debug("Using GPT for conversational response")
                return self.gpt_model.get_response(question)
        
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}", exc_info=True)
            return "I'm sorry, I encountered an error processing your question. Please try again or rephrase your question."
