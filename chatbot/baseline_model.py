import os
import logging
import random
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
        policy_keywords = ['law', 'legal', 'state', 'policy', 'ban', 'illegal', 'allowed', 'permit', 'legislation', 
                          'restrict', 'abortion policy', 'abortion law', 'abortion access', 'gestational', 'limit',
                          'parental consent', 'waiting period', 'insurance', 'medicaid', 'coverage']
        
        question_lower = question.lower()
        
        # Check for explicit state mentions combined with abortion/policy keywords
        states = ["alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", 
                 "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", 
                 "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan", 
                 "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire", 
                 "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio", 
                 "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota", 
                 "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia", 
                 "wisconsin", "wyoming"]
        
        # If question mentions abortion and a state, categorize as policy
        if ('abortion' in question_lower and any(state in question_lower for state in states)):
            return 'policy'
        
        # Check for policy-related keywords
        if any(keyword in question_lower for keyword in policy_keywords):
            return 'policy'
        
        # For questions that seem to be seeking specific information
        information_indicators = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'can i']
        if any(indicator in question_lower for indicator in information_indicators):
            return 'knowledge'
        
        # Default to conversational
        return 'conversational'
    
    def _is_greeting(self, question):
        """
        Check if the question is a simple greeting
        
        Args:
            question (str): The user's question
            
        Returns:
            bool: True if the message is a greeting, False otherwise
        """
        # Convert to lowercase and strip punctuation
        clean_question = question.lower().strip().rstrip('?!.,')
        
        # List of common greetings
        greetings = [
            'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
            'good evening', 'how are you', 'how are you doing', 'how is it going',
            'what\'s up', 'whats up', 'sup', 'yo', 'hi there', 'hello there',
            'how do you do', 'nice to meet you', 'hi how are you'
        ]
        
        # Check if the entire message is a greeting
        for greeting in greetings:
            if clean_question == greeting or clean_question.startswith(greeting + ' '):
                return True
                
        return False
        
    def _get_greeting_response(self):
        """
        Generate a friendly greeting response
        
        Returns:
            str: A friendly greeting
        """
        greeting_responses = [
            "Hi there! I'm your reproductive health assistant. How can I help you today?",
            "Hello! I'm here to provide information about reproductive health. What would you like to know?",
            "Hi! I'm doing well, thanks for asking. I'm ready to answer your reproductive health questions.",
            "Hello! I'm here and ready to assist with any reproductive health questions you might have.",
            "Hi there! I'm your AI assistant for reproductive health information. How can I assist you today?"
        ]
        return random.choice(greeting_responses)
    
    def process_question(self, question):
        """
        Process a question using the appropriate model based on its category
        Handles multi-query questions by combining responses
        
        Args:
            question (str): The user's question
        
        Returns:
            str: The model's response
        """
        try:
            # First check if this is a simple greeting
            if self._is_greeting(question):
                logger.debug("Detected greeting, providing standard response")
                return self._get_greeting_response()
                
            # Check if this is a multi-query question
            if " and " in question.lower() or ";" in question:
                return self._handle_multi_query(question)
            
            # Single query flow
            # Categorize the question
            category = self.categorize_question(question)
            logger.debug(f"Question category: {category}")
            
            # Process according to category
            return self._process_single_query(question, category)
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}", exc_info=True)
            return "I'm sorry, I encountered an error processing your question. Please try again or rephrase your question."
                
    def _handle_multi_query(self, compound_question):
        """
        Handle multi-part questions by splitting them and processing each part
        
        Args:
            compound_question (str): The compound question with multiple parts
            
        Returns:
            str: Combined response to all parts of the question
        """
        try:
            logger.debug(f"Handling multi-query question: {compound_question}")
            
            # Split the question into parts
            if ";" in compound_question:
                parts = compound_question.split(";")
            else:
                parts = compound_question.split(" and ")
                
            # Clean up parts
            parts = [part.strip() for part in parts if part.strip()]
            
            if len(parts) <= 1:
                # Not actually a multi-query, process normally
                category = self.categorize_question(compound_question)
                return self._process_single_query(compound_question, category)
                
            # Process each part separately
            responses = []
            for part in parts:
                category = self.categorize_question(part)
                logger.debug(f"Part: '{part}', Category: {category}")
                response = self._process_single_query(part, category)
                responses.append(response)
                
            # Combine responses with GPT for a coherent answer
            combined_prompt = f"""
            The user asked the following multi-part question: "{compound_question}"
            
            Here are the separate responses to each part:
            
            {' '.join(responses)}
            
            Please combine these responses into a single, coherent answer that addresses all parts of the user's question.
            Organize the information clearly with appropriate headings for each part.
            """
            
            logger.debug("Combining multi-query responses with GPT")
            return self.gpt_model.get_response(combined_prompt)
            
        except Exception as e:
            logger.error(f"Error handling multi-query: {str(e)}", exc_info=True)
            return "I'm sorry, I had trouble processing your multi-part question. Could you try asking one question at a time?"
        
    def _process_single_query(self, question, category):
        """Process a single query based on its category"""
        try:
            if category == 'policy':
                logger.debug(f"Using Policy API for response to: {question}")
                return self.policy_api.get_policy_response(question)
            
            elif category == 'knowledge':
                logger.debug(f"Using BERT RAG for response to: {question}")
                rag_response = self.bert_rag.get_response(question)
                
                if self.bert_rag.is_confident(question, rag_response):
                    return rag_response
                
                logger.debug("RAG not confident, enhancing with GPT")
                return self.gpt_model.enhance_response(question, rag_response)
            
            else:  # conversational
                logger.debug(f"Using GPT for conversational response to: {question}")
                return self.gpt_model.get_response(question)
                
        except Exception as e:
            logger.error(f"Error processing single query: {str(e)}", exc_info=True)
            return "I'm sorry, I encountered an error with that question. Could you try rephrasing it?"