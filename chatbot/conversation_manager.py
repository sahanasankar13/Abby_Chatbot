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

            # Extract location context with enhanced search through history
            location_context = self._extract_location(message, self.conversation_history)
            if location_context:
                logger.info(f"Detected location context: {location_context}")

            # Special handling for referential questions about previous states
            message_lower = message.lower()
            referential_terms = ["there", "that state", "this state", "that place", "here"]
            if any(term in message_lower for term in referential_terms) and not location_context:
                # Look through conversation history for most recent state mention
                for entry in reversed(self.conversation_history):
                    if entry['sender'] == 'user':
                        potential_location = self._extract_location(entry['message'], [])
                        if potential_location:
                            location_context = potential_location
                            logger.info(f"Found location '{location_context}' in history for referential question")
                            break

            # Special handling for abortion requests
            abortion_indicators = [
                "need an abortion", "want an abortion", "get an abortion", 
                "abortion there", "abort", "terminate", "terminate pregnancy", 
                "pregnancy termination", "can i get an abortion", "abortion in",
                "where can i get an abortion", "where to get an abortion"
            ]

            is_abortion_question = any(indicator in message_lower for indicator in abortion_indicators)

            if is_abortion_question:
                logger.info("Detected abortion request, treating as policy question with state context")

                # If we have a location context, provide direct policy information
                if location_context:
                    logger.info(f"Using location context '{location_context}' for direct policy response")

                    # Force policy categorization and pass location context
                    response = self.baseline_model.process_question(
                        message, 
                        self.conversation_history, 
                        location_context=location_context, 
                        force_category='policy'
                    )

                    # Format with citations based on whether we got real policy data
                    if "I'm sorry, I'm having trouble providing policy information" in response:
                        # Only use Planned Parenthood as source if we didn't get policy data
                        response = self.citation_manager.add_citation_to_text(response, 'planned_parenthood')
                    else:
                        response = self.citation_manager.add_citation_to_text(response, 'abortion_policy_api')

                    formatted = self.citation_manager.format_response_with_citations(response)
                    self.add_to_history('bot', formatted['text'])
                    return formatted
                else:
                    # No location context, provide concise response asking for location
                    logger.info("No location context found, providing empathetic response asking for location")
                    empathetic_response = (
                        "I'd like to provide accurate information about abortion access for you. Could you let me know which state you're asking about? Different states have different regulations."
                    )

                    # Add citation since we're discussing reproductive health
                    # But don't add citations to short conversational responses
                    if len(empathetic_response.split()) > 20:
                        cited_response = self.citation_manager.add_citation_to_text(empathetic_response, 'planned_parenthood')
                    else:
                        cited_response = empathetic_response

                    formatted_response = self.citation_manager.format_response_with_citations(cited_response)
                    message_id = self.add_to_history('bot', formatted_response['text'])
                    formatted_response['message_id'] = message_id
                    return formatted_response

            # Detect question type for adding appropriate friendly elements
            question_type = self.friendly_bot.detect_question_type(message)
            logger.debug(f"Detected question type: {question_type}")

            # Get response from baseline model, passing conversation history and location context
            start_time = time.time()
            # Pass the conversation history and location context for complete context awareness
            response = self.baseline_model.process_question(
                message, 
                self.conversation_history, 
                location_context=location_context
            )
            processing_time = time.time() - start_time
            logger.debug(f"Baseline model processing time: {processing_time:.2f} seconds")

            # Get the category of the question from baseline model
            category = self.baseline_model.categorize_question(message, self.conversation_history)

            # Check confidence for knowledge questions
            if category == 'knowledge' and not self.baseline_model.bert_rag.is_confident(message, response):
                uncertain_response = (
                    "I'm not completely sure about that. For accurate information on this topic, I'd recommend contacting a healthcare provider or Planned Parenthood directly."
                )
                response = uncertain_response

            # Add friendly elements to the response
            friendly_response = self.friendly_bot.add_friendly_elements(response, question_type)

            # Only add citations to substantial responses (not simple conversational exchanges)
            is_conversational = len(friendly_response.split()) < 20

            # Add appropriate citation based on the source of information, only for non-conversational responses
            if not is_conversational:
                if category == 'knowledge' and self.baseline_model.bert_rag.is_confident(message, response):
                    friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'planned_parenthood')
                elif category == 'policy':
                    friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'abortion_policy_api')
                else:
                    # Default citation for general responses that aren't policy-specific
                    friendly_response = self.citation_manager.add_citation_to_text(friendly_response, 'planned_parenthood')

            # Format response with citations, modified to not show citations for conversational responses
            formatted_response = self.citation_manager.format_response_with_citations(friendly_response)

            # Store response in history
            self.add_to_history('bot', formatted_response['text'])

            return formatted_response

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            error_response = "I'm sorry, I encountered a problem. Please try again or ask a different question."
            return {
                "text": error_response,
                "citations": [],
                "citation_objects": []
            }

    def _check_for_state_names(self, message):
        """
        Check for state names in a message.

        Args:
            message (str): Message to check

        Returns:
            str: State name if found, None otherwise
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
            "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
            "district of columbia": "DC", "dc": "DC", "washington dc": "DC"
        }

        message_lower = message.lower().strip()

        # First check if message is a direct state name
        if message_lower in states:
            logger.info(f"Message is a direct state name: {message_lower}")
            return message_lower

        # Check for direct state mentions in message
        for state in states:
            if f" {state} " in f" {message_lower} " or message_lower.startswith(f"{state} ") or message_lower.endswith(f" {state}"):
                logger.info(f"Found direct state mention in message: {state}")
                return state

        return None

    def _extract_location(self, message, history):
        """
        Extract location information from the current message or conversation history

        Args:
            message (str): Current message
            history (list): Conversation history

        Returns:
            str: Location information if found, None otherwise
        """
        # First check for direct state mentions using the helper method
        direct_state = self._check_for_state_names(message)
        if direct_state:
            return direct_state

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
        message_lower = message.lower().strip()
        location_phrases = ["i live in", "i'm in", "i am in", "i'm from", "i am from"]

        # Check for location phrases in current message
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
                        logger.info(f"Found location in message using phrase '{phrase}': {potential_location}")
                        return potential_location
                    # Try more advanced matching for partial matches
                    for state in states:
                        if potential_location in state or state in potential_location:
                            logger.info(f"Found partial state match: '{potential_location}' matches '{state}'")
                            return state

        # Check if the message contains "there" or similar referential terms and look for states in history
        referential_terms = ["there", "that state", "this state", "that place", "it", "here"]
        if any(term in message_lower for term in referential_terms):
            logger.info("Message contains referential location term, checking history for most recent state mention")

            # Look through conversation history for most recent state mention
            for entry in reversed(history):
                if entry['sender'] == 'user':
                    entry_lower = entry['message'].lower()

                    # Check if a direct state name or mention
                    direct_state = self._check_for_state_names(entry['message'])
                    if direct_state:
                        logger.info(f"Found state '{direct_state}' in history for referential context")
                        return direct_state

                    # Location phrases in history
                    for phrase in location_phrases:
                        if phrase in entry_lower:
                            location_part = entry_lower.split(phrase)[1].strip()
                            location_words = location_part.split()
                            if location_words:
                                potential_location = location_words[0].strip('.,!?')
                                if potential_location in states:
                                    logger.info(f"Found location '{potential_location}' in history for referential context")
                                    return potential_location

        # Check history for any location mentions if not found in current message
        for entry in reversed(history):
            if entry['sender'] == 'user':
                entry_lower = entry['message'].lower()

                # Direct state detection from history
                direct_state = self._check_for_state_names(entry['message'])
                if direct_state:
                    logger.info(f"Found state mention in history: {direct_state}")
                    return direct_state

                # Location phrases
                for phrase in location_phrases:
                    if phrase in entry_lower:
                        location_part = entry_lower.split(phrase)[1].strip()
                        location_words = location_part.split()
                        if location_words:
                            potential_location = location_words[0].strip('.,!?')
                            if potential_location in states:
                                logger.info(f"Found location in history using phrase '{phrase}': {potential_location}")
                                return potential_location

        logger.info("No location context found in message or history")
        return None

    def add_to_history(self, sender, message, message_id=None):
        """
        Add a message to the conversation history

        Args:
            sender (str): 'user' or 'bot'
            message (str): Message content
            message_id (str, optional): Unique ID for the message

        Returns:
            str: The message ID used
        """
        import uuid
        from utils.text_processing import PIIDetector

        # Generate message ID if not provided
        if not message_id:
            message_id = str(uuid.uuid4())

        # Check for and redact PII in user messages
        if sender == 'user':
            pii_detector = PIIDetector()
            if pii_detector.has_pii(message):
                logger.warning("PII detected in user message, redacting before storing in history")
                redacted_message, _ = pii_detector.redact_pii(message)
                message = redacted_message
                logger.info("User message redacted for PII in conversation history")

        timestamp = time.time()
        self.conversation_history.append({
            'sender': sender,
            'message': message,
            'message_id': message_id,
            'timestamp': timestamp
        })

        # Keep only the last 10 messages to avoid memory issues
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        return message_id

    def get_history(self):
        """
        Get the conversation history

        Returns:
            list: List of conversation messages
        """
        return self.conversation_history