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
            
            # Special handling for abortion requests
            message_lower = message.lower()
            if "need an abortion" in message_lower or "want an abortion" in message_lower or "get an abortion" in message_lower:
                logger.info("Detected abortion request, providing empathetic response with guidance")
                empathetic_response = (
                    "I understand this can be a difficult and personal situation, and I'm here to support you. "
                    "I can provide information about abortion policies in your state, general information about "
                    "abortion procedures, or help connect you with resources. Would you like to know about "
                    "abortion policies in your area, general information about abortion options, or something else?"
                )
                
                # Add citation since we're discussing reproductive health
                cited_response = self.citation_manager.add_citation_to_text(empathetic_response, 'planned_parenthood')
                formatted_response = self.citation_manager.format_response_with_citations(cited_response)
                self.add_to_history('bot', formatted_response['text'])
                return formatted_response
            
            # Detect question type for adding appropriate friendly elements
            question_type = self.friendly_bot.detect_question_type(message)
            logger.debug(f"Detected question type: {question_type}")
            
            # Extract location context if present
            location_context = self._extract_location(message, self.conversation_history)
            if location_context:
                logger.info(f"Detected location context: {location_context}")
            
            # Get response from baseline model, passing conversation history for context
            start_time = time.time()
            # Pass the conversation history for context awareness
            response = self.baseline_model.process_question(message, self.conversation_history)
            processing_time = time.time() - start_time
            logger.debug(f"Baseline model processing time: {processing_time:.2f} seconds")
            
            # Get the category of the question from baseline model
            category = self.baseline_model.categorize_question(message, self.conversation_history)
            
            # Check confidence for knowledge questions
            if category == 'knowledge' and not self.baseline_model.bert_rag.is_confident(message, response):
                uncertain_response = (
                    "I'm not completely sure about the answer to your question. It's important that you receive accurate "
                    "information on reproductive health topics. Consider reaching out to a healthcare provider or "
                    "Planned Parenthood for more specific and reliable information about this topic."
                )
                response = uncertain_response
            
            # Add friendly elements to the response
            friendly_response = self.friendly_bot.add_friendly_elements(response, question_type)
            
            # Add appropriate citation based on the source of information
            if category == 'knowledge' and self.baseline_model.bert_rag.is_confident(message, response):
                friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'planned_parenthood')
            elif category == 'policy':
                friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'abortion_policy_api')
            else:
                # Default citation for general responses that aren't policy-specific
                friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'planned_parenthood')
            
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
    
    def _extract_location(self, message, history):
        """
        Extract location information from the current message or conversation history
        
        Args:
            message (str): Current message
            history (list): Conversation history
            
        Returns:
            str: Location information if found, None otherwise
        """
        # Common US state names and abbreviations
        states = {
            "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
            "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
            "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
            "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
            "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
            "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
            "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
            "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
            "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
            "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY"
        }
        
        # Check current message for location
        message_lower = message.lower()
        location_phrases = ["i live in", "i'm in", "i am in", "i'm from", "i am from"]
        
        for phrase in location_phrases:
            if phrase in message_lower:
                # Extract the part after the phrase
                location_part = message_lower.split(phrase)[1].strip()
                # Get the first word which is likely the location
                location_words = location_part.split()
                if location_words:
                    potential_location = location_words[0].strip('.,!?')
                    # Check if it's a valid state
                    if potential_location in states:
                        return potential_location
                    
        # Check history for location mentions
        for entry in reversed(history):
            if entry['sender'] == 'user':
                entry_lower = entry['message'].lower()
                for phrase in location_phrases:
                    if phrase in entry_lower:
                        location_part = entry_lower.split(phrase)[1].strip()
                        location_words = location_part.split()
                        if location_words:
                            potential_location = location_words[0].strip('.,!?')
                            if potential_location in states:
                                return potential_location
        
        return None
    
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
