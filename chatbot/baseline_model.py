import os
import logging
from chatbot.bert_rag import BertRAGModel
from chatbot.gpt_integration import GPTModel
from chatbot.policy_api import PolicyAPI
from chatbot.response_evaluator import ResponseEvaluator

logger = logging.getLogger(__name__)

class BaselineModel:
    """
    Baseline model that combines BERT-based RAG, GPT-4 integration, policy API calls,
    and response evaluation for safety and quality
    """
    def __init__(self, evaluation_model="both"):
        """
        Initialize the baseline model components
        
        Args:
            evaluation_model (str): Model to use for response evaluation
                "openai": Use OpenAI's models only
                "local": Use local transformer models only
                "both": Use both (default)
        """
        logger.info(f"Initializing Baseline Model with evaluation_model={evaluation_model}")
        self.bert_rag = BertRAGModel()
        self.gpt_model = GPTModel()
        self.policy_api = PolicyAPI()
        self.response_evaluator = ResponseEvaluator(evaluation_model=evaluation_model)

    def categorize_question(self, question, conversation_history=None):
        """
        Categorize the question to determine which model to use

        Args:
            question (str): The user's question
            conversation_history (list, optional): Previous conversation messages

        Returns:
            str: Category of the question ('policy', 'knowledge', or 'conversational')
        """
        # Simple keyword-based categorization
        policy_keywords = ['law', 'legal', 'state', 'policy', 'ban', 'illegal', 'allowed', 'permit', 'legislation', 
                          'restrict', 'abortion policy', 'abortion law', 'abortion access', 'gestational', 'limit',
                          'parental consent', 'waiting period', 'insurance', 'medicaid', 'coverage', 'laws']

        question_lower = question.lower()
        
        # Direct policy pattern detection (what is the abortion policy in [state])
        if "abortion policy in" in question_lower or "abortion policies in" in question_lower:
            logger.debug("Direct abortion policy question with state detected")
            return 'policy'

        # Check for explicit state mentions combined with abortion/policy keywords
        states = ["alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", 
                 "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", 
                 "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan", 
                 "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire", 
                 "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio", 
                 "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota", 
                 "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia", 
                 "wisconsin", "wyoming", "dc", "district of columbia", "washington dc"]

        # Check for state abbreviations
        state_abbrevs = ["al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", 
                         "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", 
                         "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", 
                         "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc"]

        # Direct abortion policy questions - always categorize as policy
        if question_lower.startswith("can i get an abortion in") or "can i get an abortion" in question_lower:
            logger.debug("Direct abortion policy question detected")
            return 'policy'

        # Questions about abortion laws/legality in a state
        if "abortion laws" in question_lower or "abortion law" in question_lower:
            logger.debug("Abortion law question detected")
            return 'policy'

        # If question mentions abortion and a state, categorize as policy
        state_mentioned = any(f" {state} " in f" {question_lower} " or
                             question_lower.endswith(f" {state}") or
                             question_lower.startswith(f"{state} ") 
                             for state in states)
                             
        abbr_mentioned = any(f" {abbr} " in f" {question_lower} " or
                            question_lower.endswith(f" {abbr}") or
                            question_lower.startswith(f"{abbr} ") 
                            for abbr in state_abbrevs)
                         
        if 'abortion' in question_lower and (state_mentioned or abbr_mentioned):
            logger.debug(f"Question mentions abortion and a state: {question}")
            return 'policy'

        # Check for policy-related keywords
        if any(keyword in question_lower for keyword in policy_keywords):
            logger.debug(f"Question contains policy keyword: {[kw for kw in policy_keywords if kw in question_lower]}")
            return 'policy'

        # Special case for abortion-related questions with action verbs
        if 'abortion' in question_lower:
            action_indicators = ['can i', 'get an', 'have an', 'available', 'access']
            if any(indicator in question_lower for indicator in action_indicators):
                logger.info(f"Abortion question with action indicator detected: {question}")
                return 'policy'

        # Special case: Check if current question is about abortion availability or legality
        # and we have state information in conversation history
        if ('abortion' in question_lower or 'pregnant' in question_lower) and conversation_history:
            logger.info(f"Special case check: Question about abortion with history: {question_lower}")
            access_indicators = ['get', 'have', 'legal', 'available', 'there', 'allowed', 'can i']

            # Special case for referential questions about a location mentioned before
            if ('there' in question_lower or 'that state' in question_lower) and any(indicator in question_lower for indicator in access_indicators):
                logger.info("Detected referential location question - treating as policy question")
                return 'policy'

            # If question contains "abortion" and any access indicators, check for state info in history
            if any(indicator in question_lower for indicator in access_indicators):
                logger.info(f"Found access indicator in question: {[ind for ind in access_indicators if ind in question_lower]}")

                # Check if there's state information in the conversation history
                for message in reversed(conversation_history):
                    if message['sender'] == 'user':
                        msg_lower = message['message'].lower()
                        logger.info(f"Checking history message: {msg_lower}")

                        # Check for direct state mentions
                        state_found = False
                        for state in states:
                            if state in msg_lower:
                                logger.info(f"Found state in history: {state}")
                                state_found = True
                                return 'policy'

                        # Check for "I live in" patterns with any state
                        location_phrases = ["i live in", "i'm in", "i am in", "i'm from", "i am from"]
                        for phrase in location_phrases:
                            if phrase in msg_lower:
                                location_part = msg_lower.split(phrase)[1].strip()
                                # Look for a state name in this part
                                for state in states:
                                    if state in location_part:
                                        logger.info(f"Found location context with phrase '{phrase}': {state}")
                                        return 'policy'
                                        
                        # More general check for state mentions
                        for state in states:
                            if state in msg_lower:
                                logger.info(f"Found state name {state} in message: {msg_lower}")
                                return 'policy'

            logger.info("No state information found in history for abortion question")

        # For questions that seem to be seeking specific information
        information_indicators = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
        if any(indicator in question_lower for indicator in information_indicators):
            return 'knowledge'

        # Use GPT to detect more complex policy questions that our rules might miss
        try:
            if ('abortion' in question_lower or 'pregnancy' in question_lower):
                if self.gpt_model.detect_policy_question(question, conversation_history):
                    logger.info("GPT detected policy question")
                    return 'policy'
        except Exception as e:
            logger.error(f"Error using GPT for policy detection: {str(e)}")
            
        # Default to conversational
        return 'conversational'

    def process_question(self, question, conversation_history=[], location_context=None, force_category=None):
        """
        Process a question using the appropriate model based on its category
        Handles multi-query questions by combining responses

        Args:
            question (str): The user's question
            conversation_history (list, optional): List of previous messages in the conversation
            location_context (str): User's location if detected
            force_category (str, optional): Force a specific category ('policy', 'knowledge', 'conversational')

        Returns:
            str: The model's response
        """
        try:
            # Check if this is a multi-query question
            if " and " in question.lower() or ";" in question:
                return self._handle_multi_query(question, conversation_history)

            # Single query flow
            # Use forced category if provided, otherwise categorize normally
            if force_category:
                category = force_category
                logger.debug(f"Using forced category: {category}")
            else:
                # Categorize the question (passing conversation history for context awareness)
                category = self.categorize_question(question, conversation_history)
                logger.debug(f"Question category: {category}")

            # Process according to category
            return self._process_single_query(question, category, conversation_history, location_context)

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}", exc_info=True)
            return "I'm sorry, I encountered an error processing your question. Please try again or rephrase your question."

    def _handle_multi_query(self, compound_question, conversation_history=None):
        """
        Handle multi-part questions by splitting them and processing each part

        Args:
            compound_question (str): The compound question with multiple parts
            conversation_history (list, optional): List of previous messages in the conversation

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
                category = self.categorize_question(compound_question, conversation_history)
                return self._process_single_query(compound_question, category, conversation_history)

            # Process each part separately
            responses = []
            for part in parts:
                category = self.categorize_question(part, conversation_history)
                logger.debug(f"Part: '{part}', Category: {category}")
                response = self._process_single_query(part, category, conversation_history)
                responses.append(response)

            # Combine responses with GPT for a coherent answer
            combined_prompt = f"""
            The user asked the following multi-part question: "{compound_question}"

            Here are the separate responses to each part:

            {' '.join(responses)}

            Please combine these responses into a single, coherent answer that addresses all parts of the user's question.
            Organize the information clearly with appropriate headings for each part.
            Make sure to preserve all source attributions and citations from the original responses.
            """

            logger.debug("Combining multi-query responses with GPT")
            combined_response = self.gpt_model.get_response(combined_prompt)
            
            # Apply safety checks and quality evaluation to the combined response
            logger.info("Evaluating combined multi-query response")
            
            # Create combined sources info
            combined_sources = {
                "source": "mixed",
                "citations": [
                    {"source": "Planned Parenthood", "url": "https://www.plannedparenthood.org/"},
                    {"source": "Abortion Policy API", "url": "https://www.abortionpolicyapi.com/"}
                ]
            }
            
            # Evaluate and potentially improve the response
            final_response = self.response_evaluator.get_improved_response(
                compound_question, 
                combined_response, 
                combined_sources
            )
            
            return final_response

        except Exception as e:
            logger.error(f"Error handling multi-query: {str(e)}", exc_info=True)
            return "I'm sorry, I had trouble processing your multi-part question. Could you try asking one question at a time?"

    def _process_single_query(self, question, category, conversation_history=None, location_context=None):
        """
        Process a single query based on its category, applying response evaluation
        for safety checks and quality improvements

        Args:
            question (str): The user's question
            category (str): The question category ('policy', 'knowledge', or 'conversational')
            conversation_history (list, optional): List of previous messages in the conversation
            location_context (str): User's location if detected

        Returns:
            str: The model's response after safety and quality evaluation
        """
        try:
            # Get initial response based on category
            initial_response = ""
            source_info = {}
            
            if category == 'policy':
                logger.debug(f"Using Policy API for response to: {question}")
                # Pass conversation history to the policy API for context
                initial_response = self.policy_api.get_policy_response(question, conversation_history, location_context)
                # Add source information for the evaluator
                source_info = {
                    "source": "abortion_policy_api",
                    "citations": [{"source": "Abortion Policy API", "url": "https://www.abortionpolicyapi.com/"}]
                }
                
            elif category == 'knowledge':
                logger.debug(f"Using BERT RAG for response to: {question}")
                rag_response = self.bert_rag.get_response(question)
                
                # Add source information for the evaluator
                source_info = {
                    "source": "planned_parenthood",
                    "citations": [{"source": "Planned Parenthood", "url": "https://www.plannedparenthood.org/"}]
                }

                if self.bert_rag.is_confident(question, rag_response):
                    initial_response = rag_response
                else:
                    logger.debug("RAG not confident, enhancing with GPT")
                    initial_response = self.gpt_model.enhance_response(question, rag_response)

            else:  # conversational
                logger.debug(f"Using GPT for conversational response to: {question}")
                initial_response = self.gpt_model.get_response(question)
                # No specific sources for conversational responses
                source_info = {"source": "conversational"}
            
            # Apply safety checks and quality evaluation to the response
            logger.info(f"Evaluating response quality and safety for question: {question}")
            
            # Skip evaluation for very short responses (likely greetings)
            if len(initial_response.split()) < 20:
                logger.debug("Response too short, skipping evaluation")
                return initial_response
                
            # Evaluate and potentially improve the response
            final_response = self.response_evaluator.get_improved_response(
                question, 
                initial_response, 
                source_info
            )
            
            # Log if the response was improved
            if final_response != initial_response:
                logger.info("Response was improved by the evaluator")
            
            return final_response

        except Exception as e:
            logger.error(f"Error processing single query: {str(e)}", exc_info=True)
            return "I'm sorry, I encountered an error with that question. Could you try rephrasing it?"