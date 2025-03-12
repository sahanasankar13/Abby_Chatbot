import logging
import time
from chatbot.baseline_model import BaselineModel
from chatbot.friendly_bot import FriendlyBot
from chatbot.citation_manager import CitationManager

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages the conversation flow, integrating the baseline model with friendly elements
    """
    def __init__(self):
        """Initialize the conversation manager"""
        logger.info("Initializing Conversation Manager")
        try:
            self.baseline_model = BaselineModel()
            self.friendly_bot = FriendlyBot()
            self.citation_manager = CitationManager()
            self.conversation_history = []
            
            logger.info("Conversation Manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Conversation Manager: {str(e)}", exc_info=True)
            raise
    
    def process_message(self, message):
        """
        Process a user message and generate a response with citations
        
        Args:
            message (str): User's message
        
        Returns:
            dict: Response with text and citations
        """
        try:
            # Store message in history
            self.add_to_history('user', message)
            
            # Detect question type for adding appropriate friendly elements
            question_type = self.friendly_bot.detect_question_type(message)
            logger.debug(f"Detected question type: {question_type}")
            
            # Get response from baseline model
            start_time = time.time()
            response = self.baseline_model.process_question(message)
            processing_time = time.time() - start_time
            logger.debug(f"Baseline model processing time: {processing_time:.2f} seconds")
            
            # Add friendly elements to the response
            friendly_response = self.friendly_bot.add_friendly_elements(response, question_type)
            
            # Add citation markers based on question type and content
            if question_type == 'policy':
                friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'abortion_policy_api')
                friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'guttmacher')
            elif 'pregnancy' in message.lower() or 'contraception' in message.lower():
                friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'planned_parenthood')
                friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'acog')
            elif 'health' in message.lower() or 'infection' in message.lower() or 'disease' in message.lower():
                friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'cdc')
                friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'who')
            else:
                friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'planned_parenthood')
                friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'ai_generated')
            
            # Format response with citations
            formatted_response = self.citation_manager.format_response_with_citations(friendly_response)
            
            # Store response in history
            self.add_to_history('bot', formatted_response['text'])
            
            return formatted_response
        
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            error_response = "I'm sorry, I encountered a problem processing your message. Please try again or ask a different question."
            return {
                "text": error_response,
                "citations": [],
                "citation_objects": []
            }
    
    def add_to_history(self, sender, message):
        """
        Add a message to the conversation history
        
        Args:
            sender (str): 'user' or 'bot'
            message (str): Message content
        """
        timestamp = time.time()
        self.conversation_history.append({
            'sender': sender,
            'message': message,
            'timestamp': timestamp
        })
        
        # Keep only the last 10 messages to avoid memory issues
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_history(self):
        """
        Get the conversation history
        
        Returns:
            list: List of conversation messages
        """
        return self.conversation_history
