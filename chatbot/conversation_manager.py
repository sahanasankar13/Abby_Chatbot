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
        Process a user message and generate a response
        
        Args:
            message (str): User's message
        
        Returns:
            str: Bot's response
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
            
            # Store response in history
            self.add_to_history('bot', friendly_response)
            
            return friendly_response
        
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return "I'm sorry, I encountered a problem processing your message. Please try again or ask a different question."
    
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
