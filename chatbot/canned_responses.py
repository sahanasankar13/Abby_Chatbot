"""
Canned responses for the Abby chatbot.

This module provides pre-defined responses for common scenarios,
eliminating the need to use the more complex RAG system for simple responses.
"""

import logging

logger = logging.getLogger(__name__)

class ResponseManager:
    """Manages canned responses for the chatbot"""
    
    def __init__(self):
        """Initialize the response manager"""
        logger.info("Initializing Canned Response Manager")
        
        # Define response categories
        self.responses = {
            # Greetings and conversation starters
            "greeting": "Hello! I'm Abby, a chatbot designed to provide information about reproductive health. How can I help you today?",
            "goodbye": "Goodbye! Take care and stay healthy. If you have more questions in the future, I'll be here to help.",
            
            # Topic-specific redirections
            "out_of_scope": {
                "weather": "I'm designed to provide information about reproductive health, not weather forecasts. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
                "politics": "I'm here to provide information about reproductive health, not political matters. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
                "technology": "I'm programmed to assist with reproductive health questions, not technology matters. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
                "food": "I'm trained to provide information about reproductive health, not food or nutrition in general. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
                "sports": "I'm programmed to assist with reproductive health questions, not sports information. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
                "entertainment": "I'm designed to provide information about reproductive health, not entertainment. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
                "emotional_expression": "I understand you're expressing a personal feeling. While I'm here to listen, I'm specifically designed to provide information about reproductive health topics. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those.",
                "general": "I'm Abby, a chatbot specifically designed to provide information about reproductive health, including abortion access. If you're asking about abortion policies, please specify which state you're interested in. For questions about contraception, pregnancy, or sexual health, I'm here to help."
            },
            
            # State policy responses
            "state_policy_request": "I can provide information about abortion policies in {state}. Please let me know if you'd like specific details.",
            "no_state_specified": "I can help with abortion policy information, but I need to know which state you're asking about. Could you please specify a state?",
            
            # Error responses
            "general_error": "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question.",
            "unclear_input": "I'm not sure I understood your question. Could you please rephrase it or provide more details?",
            
            # ZIP code responses
            "zip_code_received": "Thank you for providing your ZIP code. I'll search for reproductive health clinics in your area.",
            "zip_code_error": "I couldn't process the ZIP code you provided. Please make sure it's a valid 5-digit US ZIP code."
        }
    
    def get_response(self, category, subtype=None, **kwargs):
        """
        Get a response for a specific category
        
        Args:
            category (str): Response category
            subtype (str, optional): Subtype of response
            **kwargs: Formatting arguments for the response
            
        Returns:
            str: The formatted response
        """
        try:
            if subtype and category in self.responses and isinstance(self.responses[category], dict):
                response_template = self.responses[category].get(subtype, self.responses[category].get("general", ""))
            else:
                response_template = self.responses.get(category, "")
                
            # Format the response with any provided keyword arguments
            if response_template and kwargs:
                return response_template.format(**kwargs)
            
            return response_template
        except Exception as e:
            logger.error(f"Error getting canned response: {str(e)}")
            return self.responses.get("general_error", "I'm sorry, something went wrong. Please try again.") 