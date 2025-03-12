import logging
import time
import random
import uuid
from typing import Optional, List, Dict, Any
from chatbot.baseline_model import BaselineModel
from chatbot.friendly_bot import FriendlyBot
from chatbot.citation_manager import CitationManager
from chatbot.policy_api import PolicyAPI

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages the conversation flow, integrating the baseline model with friendly elements
    """

    def __init__(self, evaluation_model="both"):
        """
        Initialize the conversation manager

        Args:
            evaluation_model (str): Model to use for response evaluation
                "openai": Use OpenAI's models only
                "local": Use local transformer models only
                "both": Use both (default)
        """
        logger.info(f"Initializing Conversation Manager with evaluation_model={evaluation_model}")
        try:
            # Initialize the baseline model with response evaluation capabilities
            self.baseline_model = BaselineModel(evaluation_model=evaluation_model)
            self.friendly_bot = FriendlyBot()
            self.citation_manager = CitationManager()
            self.policy_api = PolicyAPI()
            self.conversation_history = []

            logger.info("Conversation Manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Conversation Manager: {str(e)}",
                         exc_info=True)
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
            location_context = self._extract_location(
                message, self.conversation_history)
            if location_context:
                logger.info(f"Detected location context: {location_context}")

            # Special handling for referential questions about previous states
            message_lower = message.lower()
            referential_terms = [
                "there", "that state", "this state", "that place", "here"
            ]
            if any(term in message_lower
                   for term in referential_terms) and not location_context:
                # Look through conversation history for most recent state mention
                for entry in reversed(self.conversation_history):
                    if entry['sender'] == 'user':
                        potential_location = self._extract_location(
                            entry['message'], [])
                        if potential_location:
                            location_context = potential_location
                            logger.info(
                                f"Found location '{location_context}' in history for referential question"
                            )
                            break

            # Special handling for abortion requests
            abortion_indicators = [
                "need an abortion", "want an abortion", "get an abortion",
                "abortion there", "abort", "terminate", "terminate pregnancy",
                "pregnancy termination", "can i get an abortion",
                "abortion in", "where can i get an abortion",
                "where to get an abortion"
            ]

            # Emotional indicators for personal support rather than just policy information
            emotional_indicators = [
                "stressed", "worried", "scared", "afraid", "nervous", "anxious",
                "confused", "uncertain", "unsure", "help me", "don't know what to do",
                "not sure what to do", "difficult", "hard decision", "tough decision",
                "feeling", "feel", "scared of", "worried about", "regret", "guilt"
            ]

            is_abortion_question = any(indicator in message_lower
                                       for indicator in abortion_indicators)

            has_emotional_content = any(indicator in message_lower
                                     for indicator in emotional_indicators)

            # Check for state information in the question for combined emotional+policy handling
            state_context = self._check_for_state_names(message)
            if state_context:
                location_context = state_context
                logger.info(f"Found state context directly in message: {location_context}")

            # Check if we need to include general knowledge information along with policy/emotional content
            has_knowledge_question = any(indicator in message_lower
                               for indicator in ["how does", "what is", "how do", "what are", "tell me about", 
                                                "how will", "will i", "explain", "works", "procedure", "pill",
                                                "medication", "recovery", "timeline", "process"])

            # Handle combined emotional abortion-related questions with location information AND knowledge request
            if is_abortion_question and has_emotional_content and location_context and has_knowledge_question:
                logger.info(f"Detected complex query: emotional + policy + knowledge for {location_context}")

                # 1. Get policy information
                policy_response = self.baseline_model.process_question(
                    message,
                    self.conversation_history,
                    location_context=location_context,
                    force_category='policy')

                # 2. Get knowledge information about the procedure
                # Create a knowledge-specific query by removing emotional content
                knowledge_query = "How does an abortion procedure work? What methods are available?"
                knowledge_response = self.baseline_model.process_question(
                    knowledge_query,
                    [],  # Empty history to avoid confusion
                    force_category='knowledge')

                # 3. Combine all three components (emotional support, policy, knowledge)
                comprehensive_response = (
                    f"I understand feeling anxious about this situation. It's completely normal to have these emotions "
                    f"when making important decisions about your reproductive health. I'm here to provide you with accurate "
                    f"information without judgment.\n\n"
                    f"First, regarding abortion access in {location_context}:\n{policy_response}\n\n"
                    f"You also asked about how the procedure works. Here's some general information:\n{knowledge_response}\n\n"
                    f"Remember that many people experience similar feelings when facing these decisions. "
                    f"If you're feeling overwhelmed, consider speaking with a mental health professional who specializes "
                    f"in reproductive health concerns."
                )

                # Add citations to both sources
                cited_response = comprehensive_response
                if "I'm sorry, I'm having trouble providing policy information" not in policy_response:
                    cited_response = self.citation_manager.add_citation_to_text(
                        cited_response, 'abortion_policy_api')

                cited_response = self.citation_manager.add_citation_to_text(
                    cited_response, 'planned_parenthood')

                formatted_response = self.citation_manager.format_response_with_citations(
                    cited_response)
                message_id = self.add_to_history('bot', formatted_response['text'])
                formatted_response['message_id'] = message_id
                return formatted_response

            # Handle combined emotional abortion-related questions with location information
            elif is_abortion_question and has_emotional_content and location_context:
                logger.info(f"Detected emotional abortion question with location {location_context}, providing supportive policy response")

                # Get the policy information
                policy_response = self.baseline_model.process_question(
                    message,
                    self.conversation_history,
                    location_context=location_context,
                    force_category='policy')

                # Add emotional support wrapper
                supportive_policy_response = (
                    f"I understand feeling anxious about this situation. It's completely normal to have these emotions "
                    f"when making important decisions about your reproductive health. I'm here to provide you with accurate "
                    f"information about options in {location_context} without judgment.\n\n"
                    f"{policy_response}\n\n"
                    f"Remember that many people experience similar feelings when facing these decisions. "
                    f"If you're feeling overwhelmed, consider speaking with a mental health professional who specializes "
                    f"in reproductive health concerns."
                )

                # Add citation based on policy data
                cited_response = self.citation_manager.add_citation_to_text(
                    supportive_policy_response, 'abortion_policy_api')

                formatted_response = self.citation_manager.format_response_with_citations(
                    cited_response)
                message_id = self.add_to_history('bot', formatted_response['text'])
                formatted_response['message_id'] = message_id
                return formatted_response

            # Handle emotional abortion questions with knowledge request but without location
            elif is_abortion_question and has_emotional_content and has_knowledge_question:
                logger.info("Detected emotional abortion question with knowledge request but no location")

                # Get knowledge information about the procedure
                knowledge_query = "How does an abortion procedure work? What methods are available?"
                knowledge_response = self.baseline_model.process_question(
                    knowledge_query,
                    [],  # Empty history to avoid confusion
                    force_category='knowledge')

                supportive_knowledge_response = (
                    "I understand this is a stressful and emotional decision. It's completely normal to feel uncertain "
                    "or worried when making choices about your reproductive health. I'm here to provide support and "
                    "accurate information without judgment.\n\n"
                    f"Regarding how the procedure works:\n{knowledge_response}\n\n"
                    "Many people experience similar feelings when considering their options. To give you specific "
                    "information about abortion access in your area, could you let me know which state you're in? "
                    "Different states have different regulations, and I want to make sure I provide you with the most accurate information."
                )

                # Add citation for this supportive response
                cited_response = self.citation_manager.add_citation_to_text(
                    supportive_knowledge_response, 'planned_parenthood')

                formatted_response = self.citation_manager.format_response_with_citations(
                    cited_response)
                message_id = self.add_to_history('bot', formatted_response['text'])
                formatted_response['message_id'] = message_id
                return formatted_response

            # Handle emotional abortion-related questions without location or knowledge request
            elif is_abortion_question and has_emotional_content:
                logger.info("Detected emotional abortion question, providing supportive response")

                supportive_response = (
                    "I understand this is a stressful and emotional decision. It's completely normal to feel uncertain "
                    "or worried when making choices about your reproductive health. I'm here to provide support and "
                    "accurate information without judgment.\n\n"
                    "Many people experience similar feelings when considering their options. To give you specific "
                    "information about abortion access in your area, could you let me know which state you're in? "
                    "Different states have different regulations, and I want to make sure I provide you with the most accurate information."
                )

                # Add citation for this supportive response
                cited_response = self.citation_manager.add_citation_to_text(
                    supportive_response, 'planned_parenthood')

                formatted_response = self.citation_manager.format_response_with_citations(
                    cited_response)
                message_id = self.add_to_history('bot', formatted_response['text'])
                formatted_response['message_id'] = message_id
                return formatted_response

            # Handle standard abortion policy questions
            elif is_abortion_question:
                logger.info(
                    "Detected abortion request, treating as policy question with state context"
                )

                # If we have a location context, provide direct policy information
                if location_context:
                    logger.info(
                        f"Using location context '{location_context}' for direct policy response"
                    )

                    # Force policy categorization and pass location context
                    response = self.baseline_model.process_question(
                        message,
                        self.conversation_history,
                        location_context=location_context,
                        force_category='policy')

                    # Format with citations based on whether we got real policy data
                    # First check if this is a non-US country to avoid incorrect citations
                    is_non_us_country = location_context.lower() in ['india', 'canada', 'uk', 'australia', 'mexico', 'france', 'germany', 
                                         'china', 'japan', 'brazil', 'spain', 'italy', 'russia', 'north korea']
                    
                    if is_non_us_country or "I'm sorry, I'm having trouble providing policy information" in response:
                        # For non-US countries or error responses, cite Planned Parenthood
                        response = self.citation_manager.add_citation_to_text(
                            response, 'planned_parenthood')
                    else:
                        # Only cite Abortion Policy API for actual US state data
                        response = self.citation_manager.add_citation_to_text(
                            response, 'abortion_policy_api')

                    formatted = self.citation_manager.format_response_with_citations(
                        response)
                    self.add_to_history('bot', formatted['text'])
                    return formatted
                else:
                    # No location context, provide concise response asking for location
                    logger.info(
                        "No location context found, providing empathetic response asking for location"
                    )
                    empathetic_response = (
                        "I'd like to provide accurate information about abortion access for you. Could you let me know which state you're asking about? Different states have different regulations."
                    )

                    # Add citation since we're discussing reproductive health
                    # But don't add citations to short conversational responses
                    if len(empathetic_response.split()) > 20:
                        cited_response = self.citation_manager.add_citation_to_text(
                            empathetic_response, 'planned_parenthood')
                    else:
                        cited_response = empathetic_response

                    formatted_response = self.citation_manager.format_response_with_citations(
                        cited_response)
                    message_id = self.add_to_history(
                        'bot', formatted_response['text'])
                    formatted_response['message_id'] = message_id
                    return formatted_response

            # Check if this is a simple greeting that should have a brief response
            simple_greeting_indicators = ["hi", "hello", "hey", "good morning", "good afternoon", 
                                        "good evening", "how are you", "what's up", "greetings"]
            is_simple_greeting = any(message.lower().strip() == greeting or 
                                  message.lower().strip().startswith(f"{greeting} ") or
                                  message.lower().strip().endswith(f" {greeting}")
                                  for greeting in simple_greeting_indicators)

            # If it's a very simple greeting, respond directly without using the full model pipeline
            if is_simple_greeting and len(message.split()) <= 4:
                logger.info("Detected simple greeting, bypassing full model pipeline")
                greeting_responses = [
                    "Hello! How can I help you today?",
                    "Hi there! How can I assist you?",
                    "Hello! I'm here if you have any questions about reproductive health.",
                    "Hi! What can I help you with today?"
                ]
                simple_response = random.choice(greeting_responses)
                formatted_response = {
                    "text": simple_response,
                    "citations": [],
                    "citation_objects": []
                }
                message_id = self.add_to_history('bot', simple_response)
                formatted_response['message_id'] = message_id
                return formatted_response

            # For non-greeting messages, continue with normal processing
            # Detect question type for adding appropriate friendly elements
            question_type = self.friendly_bot.detect_question_type(message)
            logger.debug(f"Detected question type: {question_type}")

            # Get the category of the question from baseline model first
            category = self.baseline_model.categorize_question(
                message, self.conversation_history)

            logger.info(f"Message categorized as: {category}")

            # Get response from baseline model, passing conversation history and location context
            start_time = time.time()
            # Pass the conversation history and location context for complete context awareness
            response = self.baseline_model.process_question(
                message,
                self.conversation_history,
                location_context=location_context)
            processing_time = time.time() - start_time
            logger.debug(
                f"Baseline model processing time: {processing_time:.2f} seconds"
            )

            # Check confidence for knowledge questions
            if category == 'knowledge' and not self.baseline_model.bert_rag.is_confident(
                    message, response):
                uncertain_response = (
                    "I'm not completely sure about that. For accurate information on this topic, I'd recommend contacting a healthcare provider or Planned Parenthood directly."
                )
                response = uncertain_response

            # Add friendly elements to the response
            friendly_response = self.friendly_bot.add_friendly_elements(
                response, question_type)

            # Only add citations to substantial responses (not simple conversational exchanges)
            # Increased threshold to consider more responses as substantial enough for citations
            is_conversational = len(friendly_response.split()) < 10

            # Add appropriate citation based on the source of information, only for non-conversational responses
            if not is_conversational:
                if category == 'knowledge' and self.baseline_model.bert_rag.is_confident(
                        message, response):
                    friendly_response = self.citation_manager.add_citation_to_text(
                        friendly_response, 'planned_parenthood')
                elif category == 'policy':
                    friendly_response = self.citation_manager.add_citation_to_text(
                        friendly_response, 'abortion_policy_api')
                else:
                    # Default citation for general responses that aren't policy-specific
                    friendly_response = self.citation_manager.add_citation_to_text(
                        friendly_response, 'planned_parenthood')

            # Format response with citations, modified to not show citations for conversational responses
            formatted_response = self.citation_manager.format_response_with_citations(
                friendly_response)

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
            "alabama": "AL",
            "alaska": "AK",
            "arizona": "AZ",
            "arkansas": "AR",
            "california": "CA",
            "colorado": "CO",
            "connecticut": "CT",
            "delaware": "DE",
            "florida": "FL",
            "georgia": "GA",
            "hawaii": "HI",
            "idaho": "ID",
            "illinois": "IL",
            "indiana": "IN",
            "iowa": "IA",
            "kansas": "KS",
            "kentucky": "KY",
            "louisiana": "LA",
            "maine": "ME",
            "maryland": "MD",
            "massachusetts": "MA",
            "michigan": "MI",
            "minnesota": "MN",
            "mississippi": "MS",
            "missouri": "MO",
            "montana": "MT",
            "nebraska": "NE",
            "nevada": "NV",
            "new hampshire": "NH",
            "new jersey": "NJ",
            "new mexico": "NM",
            "new york": "NY",
            "north carolina": "NC",
            "north dakota": "ND",
            "ohio": "OH",
            "oklahoma": "OK",
            "oregon": "OR",
            "pennsylvania": "PA",
            "rhode island": "RI",
            "south carolina": "SC",
            "south dakota": "SD",
            "tennessee": "TN",
            "texas": "TX",
            "utah": "UT",
            "vermont": "VT",
            "virginia": "VA",
            "washington": "WA",
            "west virginia": "WV",
            "wisconsin": "WI",
            "wyoming": "WY",
            "district of columbia": "DC",
            "dc": "DC",
            "washington dc": "DC"
        }

        message_lower = message.lower().strip()

        # First check if message is a direct state name
        if message_lower in states:
            logger.info(f"Message is a direct state name: {message_lower}")
            return message_lower

        # Check for direct state mentions in message
        for state in states:
            if f" {state} " in f" {message_lower} " or message_lower.startswith(
                    f"{state} ") or message_lower.endswith(f" {state}"):
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
            "alabama": "AL",
            "alaska": "AK",
            "arizona": "AZ",
            "arkansas": "AR",
            "california": "CA",
            "colorado": "CO",
            "connecticut": "CT",
            "delaware": "DE",
            "florida": "FL",
            "georgia": "GA",
            "hawaii": "HI",
            "idaho": "ID",
            "illinois": "IL",
            "indiana": "IN",
            "iowa": "IA",
            "kansas": "KS",
            "kentucky": "KY",
            "louisiana": "LA",
            "maine": "ME",
            "maryland": "MD",
            "massachusetts": "MA",
            "michigan": "MI",
            "minnesota": "MN",
            "mississippi": "MS",
            "missouri": "MO",
            "montana": "MT",
            "nebraska": "NE",
            "nevada": "NV",
            "new hampshire": "NH",
            "new jersey": "NJ",
            "new mexico": "NM",
            "new york": "NY",
            "north carolina": "NC",
            "north dakota": "ND",
            "ohio": "OH",
            "oklahoma": "OK",
            "oregon": "OR",
            "pennsylvania": "PA",
            "rhode island": "RI",
            "south carolina": "SC",
            "south dakota": "SD",
            "tennessee": "TN",
            "texas": "TX",
            "utah": "UT",
            "vermont": "VT",
            "virginia": "VA",
            "washington": "WA",
            "west virginia": "WV",
            "wisconsin": "WI",
            "wyoming": "WY"
        }

        # Check current message for location
        message_lower = message.lower().strip()
        location_phrases = [
            "i live in", "i'm in", "i am in", "i'm from", "i am from"
        ]

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
                        logger.info(
                            f"Found location in message using phrase '{phrase}': {potential_location}"
                        )
                        return potential_location
                    # Try more advanced matching for partial matches
                    for state in states:
                        if potential_location in state or state in potential_location:
                            logger.info(
                                f"Found partial state match: '{potential_location}' matches '{state}'"
                            )
                            return state

        # Check if the message contains "there" or similar referential terms and look for states in history
        referential_terms = [
            "there", "that state", "this state", "that place", "it", "here"
        ]
        if any(term in message_lower for term in referential_terms):
            logger.info(
                "Message contains referential location term, checking history for most recent state mention"
            )

            # Look through conversation history for most recent state mention
            for entry in reversed(history):
                if entry['sender'] == 'user':
                    entry_lower = entry['message'].lower()

                    # Check if a direct state name or mention
                    direct_state = self._check_for_state_names(
                        entry['message'])
                    if direct_state:
                        logger.info(
                            f"Found state '{direct_state}' in history for referential context"
                        )
                        return direct_state

                    # Location phrases in history
                    for phrase in location_phrases:
                        if phrase in entry_lower:
                            location_part = entry_lower.split(
                                phrase)[1].strip()
                            location_words = location_part.split()
                            if location_words:
                                potential_location = location_words[0].strip(
                                    '.,!?')
                                if potential_location in states:
                                    logger.info(
                                        f"Found location '{potential_location}' in history for referential context"
                                    )
                                    return potential_location

        # Check history for any location mentions if not found in current message
        for entry in reversed(history):
            if entry['sender'] == 'user':
                entry_lower = entry['message'].lower()

                # Direct state detection from history
                direct_state = self._check_for_state_names(entry['message'])
                if direct_state:
                    logger.info(
                        f"Found state mention in history: {direct_state}")
                    return direct_state

                # Location phrases
                for phrase in location_phrases:
                    if phrase in entry_lower:
                        location_part = entry_lower.split(phrase)[1].strip()
                        location_words = location_part.split()
                        if location_words:
                            potential_location = location_words[0].strip(
                                '.,!?')
                            if potential_location in states:
                                logger.info(
                                    f"Found location in history using phrase '{phrase}': {potential_location}"
                                )
                                return potential_location

        logger.info("No location context found in message or history")
        return None

    def _is_us_location(self, location):
        """Check if a location string refers to the US or a US state."""
        # Now we use the STATE_NAMES from PolicyAPI for consistency
        us_states = {state.lower() for state in self.policy_api.STATE_NAMES.values()}
        state_abbrevs = {code.lower() for code in self.policy_api.STATE_NAMES.keys()}

        us_general_terms = {
            "united states", "usa", "us", "america", "the us", "the states",
            "the united states"
        }
        
        # Non-US countries to specifically identify international requests
        non_us_countries = {
            'india', 'canada', 'uk', 'australia', 'mexico', 'france', 'germany', 
            'china', 'japan', 'brazil', 'spain', 'italy', 'russia'
        }

        location_lower = location.lower().strip()
        
        # First check if this is a known non-US country
        if location_lower in non_us_countries:
            logger.debug(f"Identified non-US country in location check: {location_lower}")
            return False
            
        # Otherwise check if it's a US state or general US reference
        return (location_lower in us_states or location_lower in state_abbrevs
                or location_lower in us_general_terms)

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
                logger.warning(
                    "PII detected in user message, redacting before storing in history"
                )
                redacted_message, _ = pii_detector.redact_pii(message)
                message = redacted_message
                logger.info(
                    "User message redacted for PII in conversation history")

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

    def detect_location_context(self, message: str) -> Optional[str]:
        """
        Detect if a US location is mentioned in the message.
        Also identifies international locations for proper handling.

        Args:
            message (str): The user message to analyze

        Returns:
            Optional[str]: The detected location (US state or international) 
                          or None if no location is found
        """
        message_lower = message.lower()
        
        # First check for international countries
        non_us_countries = ['india', 'canada', 'uk', 'australia', 'mexico', 'france', 'germany', 
                           'china', 'japan', 'brazil', 'spain', 'italy', 'russia']
                           
        for country in non_us_countries:
            if country in message_lower:
                logger.info(f"Found international location mention in message: {country}")
                return country

        # Check for direct mentions of US states - using the state names from policy_api
        for code, state in self.policy_api.STATE_NAMES.items():
            if state.lower() in message_lower or code.lower() in message_lower:
                logger.info(f"Found direct state mention in message: {state.lower()}")
                return state.lower()