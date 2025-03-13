import os
import time
import logging
from typing import List, Dict, Any, Optional

from chatbot.bert_rag import BertRAGModel
from chatbot.gpt_integration import GPTModel
from chatbot.policy_api import PolicyAPI
from chatbot.response_evaluator import ResponseEvaluator
from chatbot.question_classifier import QuestionClassifier
from utils.metrics import increment_counter, record_time, record_api_call

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
        self.question_classifier = QuestionClassifier()
        
        # Track recent responses and contexts for evaluation
        self.recent_responses = []
        self.recent_contexts = []
        self.max_tracked_responses = 100

    def categorize_question(self, question, history=None):
        """
        Determine the category of the question using GPT-3.5 Turbo

        Args:
            question (str): User's question
            history (list, optional): List of conversation history

        Returns:
            str: Category ('policy', 'knowledge', 'conversational', 'out_of_scope')
        """
        try:
            # First, use the GPT-3.5 classifier for smart categorization
            logger.info(f"Using GPT-3.5 classifier for question: {question}")
            record_api_call("gpt-3.5-turbo-classification")
            
            # Time the classification for metrics
            start_time = time.time()
            classification = self.question_classifier.get_full_classification(question, history)
            elapsed_time = time.time() - start_time
            record_time("question_classification", elapsed_time)
            
            logger.info(f"GPT-3.5 classification result: {classification}")
            
            # Handle special case for location-only input (just a state name)
            if "location_only" in classification.get("categories", []):
                logger.info(f"Location-only input detected: {classification.get('location_context')}")
                return "policy"
                
            # Map the GPT-3.5 classification to our categories
            if classification.get("is_policy_question", False) and classification.get("is_abortion_related", False):
                logger.info("Classified as abortion policy question")
                return "policy"
                
            # Check if this is primarily an informational question
            if "information" in classification.get("categories", []):
                logger.info("Classified as knowledge/information question")
                return "knowledge"
                
            # Check if this is an emotional support/personal question
            if "emotional_support" in classification.get("categories", []) or classification.get("is_emotional", False):
                logger.info("Classified as emotional/personal question")
                return "conversational"
                
            # Check if this is a greeting or simple conversational exchange
            if "greeting" in classification.get("categories", []):
                logger.info("Classified as greeting/conversational")
                return "conversational"
                
            # If we're still not sure, do some additional checks for policy questions
            # This helps catch policy questions that might have been missed
            question_lower = question.lower()
            
            # Special case for single-word state inputs after abortion context
            if len(question.split()) <= 1 and history:
                # Look for abortion-related messages in history
                abortion_terms = ["abortion", "terminate", "pregnancy", "termination"]
                if any(any(term in entry['message'].lower() for term in abortion_terms) 
                       for entry in history if entry['sender'] == 'user'):
                    # Check if the single word looks like a state name/abbreviation
                    state_input = self.policy_api.get_state_code(question)
                    if state_input:
                        logger.info(f"Single-word state input detected after abortion context: {question}")
                        return "policy"
            
            # Direct pattern detection for policy questions that might be missed
            if "abortion policy" in question_lower or "abortion laws" in question_lower:
                logger.info("Direct abortion policy reference detected")
                return "policy"
                
            # If the question has a location context, it's likely about policy
            if classification.get("location_context") and "abortion" in question_lower:
                logger.info(f"Location context with abortion question: {classification.get('location_context')}")
                return "policy"
                
            # For informational questions that don't fit the above categories
            if any(indicator in question_lower for indicator in 
                  ['what', 'how', 'when', 'where', 'why', 'who', 'which']):
                return 'knowledge'
                
            # Default based on GPT classification confidence
            if classification.get("confidence", 0) > 0.7:
                # Trust the primary category from GPT classification
                primary_category = classification.get("categories", ["information"])[0]
                if primary_category == "policy":
                    return "policy"
                elif primary_category == "information":
                    return "knowledge"
                elif primary_category in ["emotional_support", "greeting"]:
                    return "conversational"
                else:
                    return "conversational"
            
            # If still uncertain, default to conversational
            return "conversational"

        except Exception as e:
            logger.error(f"Error categorizing question with GPT-3.5: {str(e)}", exc_info=True)
            
            # Fallback to simpler categorization if GPT classification fails
            try:
                question_lower = question.lower()
                
                # Simple checks for policy questions
                if "abortion" in question_lower and any(state in question_lower for state in self.policy_api.STATE_NAMES_LOWER.keys()):
                    return "policy"
                    
                # Simple checks for knowledge questions
                if any(indicator in question_lower for indicator in 
                      ['what', 'how', 'when', 'where', 'why', 'who', 'which']):
                    return "knowledge"
                    
                # Default to conversational
                return "conversational"
                
            except Exception as fallback_error:
                logger.error(f"Error in fallback categorization: {str(fallback_error)}", exc_info=True)
                return "out_of_scope"

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
        # Start timing the inference
        import time
        from utils.metrics import record_time, record_category_time
        
        start_time = time.time()
        
        try:
            # Check if this is a multi-query question
            is_multi_query = " and " in question.lower() or ";" in question
            if is_multi_query:
                category = "multi_query"
                logger.debug("Detected multi-query question")
                response = self._handle_multi_query(question, conversation_history)
            else:
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
            response = self._process_single_query(question, category, conversation_history, location_context)

            # Record inference time with category
            elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            try:
                # Record overall inference time
                record_time("inference_time", elapsed_time)
                
                # Record category-specific inference time
                record_category_time("inference_time", category, elapsed_time)
                
                # Also identify reproductive health topic for more detailed tracking
                topic = self._get_reproductive_health_topic(question)
                if topic:
                    record_category_time("inference_time_by_topic", topic, elapsed_time)
            except Exception as e:
                logger.warning(f"Failed to record inference time metrics: {str(e)}")
                
            return response

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

            # Record multi-query-specific metrics
            try:
                from utils.metrics import record_category_time, increment_counter
                increment_counter("multi_query_responses")
                
                # Record the number of parts in the multi-query
                increment_counter(f"multi_query_parts_{len(parts)}")
                
                # Record the reproductive health topics covered in this multi-query
                topics = set()
                for part in parts:
                    topic = self._get_reproductive_health_topic(part)
                    if topic:
                        topics.add(topic)
                        
                for topic in topics:
                    increment_counter(f"multi_query_topic_{topic}")
            except Exception as e:
                logger.warning(f"Failed to record multi-query metrics: {str(e)}")

            return final_response

        except Exception as e:
            logger.error(f"Error handling multi-query: {str(e)}", exc_info=True)
            return "I'm sorry, I had trouble processing your multi-part question. Could you try asking one question at a time?"

    def _get_reproductive_health_topic(self, question):
        """
        Identify the reproductive health topic from a question
        
        Args:
            question (str): The user's question
            
        Returns:
            str: The identified topic or None if no specific topic is found
        """
        question_lower = question.lower()
        
        # Define topic categories and their keywords
        topic_keywords = {
            "pregnancy_planning": [
                "trying to conceive", "trying to get pregnant", "ttc", 
                "fertility", "ovulation", "conception", "conceive",
                "getting pregnant", "preconception", "pre-conception",
                "prepare for pregnancy", "planning for pregnancy", 
                "before getting pregnant", "before pregnancy"
            ],
            "pregnancy": [
                "pregnant", "pregnancy", "prenatal", "trimester", 
                "fetal", "fetus", "morning sickness", "baby bump",
                "ultrasound", "expecting", "gestational"
            ],
            "birth_control": [
                "birth control", "contraception", "contraceptive", "iud", 
                "pill", "condom", "implant", "nexplanon", "depo",
                "morning after", "plan b", "spermicide"
            ],
            "menstruation": [
                "period", "menstrual", "menstruation", "cycle", "pms",
                "cramps", "bleeding", "spotting", "tampon", "pad", "flow"
            ],
            "reproductive_health": [
                "reproductive health", "sexual health", "gynecology", 
                "obgyn", "pelvic", "vaginal", "cervical", "uterine", 
                "ovarian", "testicular", "std", "sti", "infection"
            ],
            "abortion": [
                "abortion", "terminate", "termination", "miscarriage", 
                "pregnancy loss", "roe", "wade", "pro-choice", "pro-life"
            ]
        }
        
        # Check for topic matches
        for topic, keywords in topic_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return topic
                
        # Return a general reproductive health topic if no specific match
        return "general_reproductive_health"
        
    def _add_trusted_sources(self, response, question):
        """
        Add links to trusted external sources for reproductive health questions
        based on the specific topic detected
        
        Args:
            response (str): The original response
            question (str): The user's question
            
        Returns:
            tuple: (Enhanced response with source links, Updated source info)
        """
        # Only add sources if the response doesn't already have citations
        if "(Source:" in response:
            return response, {
                "source": "planned_parenthood",
                "citations": [{"source": "Planned Parenthood", "url": "https://www.plannedparenthood.org/"}]
            }
        
        # Initialize source info
        source_info = {
            "source": "planned_parenthood",
            "citations": [{"source": "Planned Parenthood", "url": "https://www.plannedparenthood.org/"}]
        }
        
        # Identify the topic
        topic = self._get_reproductive_health_topic(question)
        
        # Define trusted sources for different topics
        topic_sources = {
            "pregnancy_planning": [
                ("Mayo Clinic", "Preconception planning", "https://www.mayoclinic.org/healthy-lifestyle/getting-pregnant/in-depth/preconception-planning/art-20047296"),
                ("National Institutes of Health", "Fertility and Infertility", "https://www.nichd.nih.gov/health/topics/fertility"),
                ("Centers for Disease Control and Prevention", "Planning for Pregnancy", "https://www.cdc.gov/preconception/planning.html")
            ],
            "pregnancy": [
                ("American College of Obstetricians and Gynecologists", "Pregnancy Resources", "https://www.acog.org/womens-health/pregnancy"),
                ("Mayo Clinic", "Pregnancy week by week", "https://www.mayoclinic.org/healthy-lifestyle/pregnancy-week-by-week/basics/healthy-pregnancy/hlv-20049471"),
                ("Centers for Disease Control and Prevention", "Pregnancy", "https://www.cdc.gov/pregnancy/index.html")
            ],
            "birth_control": [
                ("Centers for Disease Control and Prevention", "Contraception", "https://www.cdc.gov/reproductivehealth/contraception/index.htm"),
                ("American College of Obstetricians and Gynecologists", "Birth Control", "https://www.acog.org/womens-health/birth-control"),
                ("Mayo Clinic", "Birth control options", "https://www.mayoclinic.org/tests-procedures/birth-control/about/pac-20384621")
            ],
            "menstruation": [
                ("Mayo Clinic", "Menstrual cycle", "https://www.mayoclinic.org/healthy-lifestyle/womens-health/in-depth/menstrual-cycle/art-20047186"),
                ("National Institutes of Health", "Menstruation and Menstrual Problems", "https://www.nichd.nih.gov/health/topics/menstruation"),
                ("American College of Obstetricians and Gynecologists", "Abnormal Uterine Bleeding", "https://www.acog.org/womens-health/faqs/abnormal-uterine-bleeding")
            ],
            "reproductive_health": [
                ("Centers for Disease Control and Prevention", "Reproductive Health", "https://www.cdc.gov/reproductivehealth/index.html"),
                ("World Health Organization", "Sexual and reproductive health", "https://www.who.int/health-topics/sexual-and-reproductive-health"),
                ("National Institutes of Health", "Reproductive Health", "https://www.nichd.nih.gov/health/topics/reproductive")
            ],
            "abortion": [
                ("American College of Obstetricians and Gynecologists", "Abortion Policy", "https://www.acog.org/clinical-information/policy-and-position-statements/statements-of-policy/2022/abortion-policy"),
                ("National Institutes of Health", "Medical Care After Pregnancy Loss", "https://www.nichd.nih.gov/health/topics/pregnancyloss/conditioninfo/treatment"),
                ("Abortion Policy API", "State Policies", "https://www.abortionpolicyapi.com/")
            ],
            "general_reproductive_health": [
                ("Centers for Disease Control and Prevention", "Reproductive Health", "https://www.cdc.gov/reproductivehealth/index.html"),
                ("American College of Obstetricians and Gynecologists", "Women's Health", "https://www.acog.org/womens-health"),
                ("National Institutes of Health", "Reproductive Health", "https://www.nichd.nih.gov/health/topics/reproductive")
            ]
        }
        
        # Add introduction for additional sources
        additional_sources = f"\n\nFor more detailed information on {topic.replace('_', ' ')}, you may want to check these trusted resources:\n"
        
        # Add the relevant sources for the topic
        selected_sources = topic_sources.get(topic, topic_sources["general_reproductive_health"])
        
        for source, title, url in selected_sources:
            additional_sources += f"- {source}: {title} (Source: {source})\n"
            source_info["citations"].append({"source": source, "url": url})
        
        # If there's a response, enhance it with the sources
        enhanced_response = response + additional_sources.rstrip()
        return enhanced_response, source_info
        
    def _process_single_query(self, question, category, conversation_history=None, location_context=None):
        """
        Process a single query based on its category, applying response evaluation
        for safety checks and quality improvements

        Args:
            question (str): The user's question
            category (str): The question category ('policy', 'knowledge', 'conversational', or 'out_of_scope')
            conversation_history (list, optional): List of previous messages in the conversation
            location_context (str): User's location if detected

        Returns:
            str: The model's response after safety and quality evaluation
        """
        try:
            # Start timing for performance metrics
            start_time = time.time()

            # Track question category for analytics
            increment_counter(f"questions_{category}")

            # Get initial response based on category
            initial_response = ""
            source_info = {}

            if category == 'out_of_scope':
                logger.debug(f"Handling out-of-scope question: {question}")
                # Get out-of-scope response from BERT RAG
                out_of_scope = self.bert_rag._is_out_of_scope(question)
                initial_response = self.bert_rag._get_out_of_scope_response(out_of_scope if out_of_scope else ["general"])
                # No citations for out-of-scope responses
                source_info = {"source": "out_of_scope"}

                # Return immediately without further evaluation
                return initial_response

            elif category == 'policy':
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
                rag_response, contexts = self.bert_rag.get_response_with_context(question)
                
                # Store the contexts for Ragas evaluation
                self.recent_contexts.append(contexts)
                while len(self.recent_contexts) > self.max_tracked_responses:
                    self.recent_contexts.pop(0)
                
                # Add source information for the evaluator
                source_info = {
                    "source": "planned_parenthood",
                    "citations": [{"source": "Planned Parenthood", "url": "https://www.plannedparenthood.org/"}]
                }
                
                # Check for special question types that need additional handling
                question_lower = question.lower()
                
                # Check if this is an abortion types or methods question that should offer policy info too
                is_abortion_types_question = "types of abortion" in question_lower or "different types of abortion" in question_lower or "abortion methods" in question_lower
                
                if is_abortion_types_question:
                    logger.info("Detected abortion types/methods question - providing RAG response with policy offer")
                    base_response = rag_response if self.bert_rag.is_confident(question, rag_response) else self.gpt_model.enhance_response(question, rag_response)
                    
                    # Append the offer for state-specific policy information
                    policy_offer = "\n\nWould you like information about abortion policies in a specific state? If so, please let me know which state you're interested in."
                    initial_response = base_response + policy_offer
                    
                elif not self.bert_rag.is_confident(question, rag_response):
                    # The RAG system doesn't have a confident answer
                    logger.info("RAG not confident, adding trusted sources based on topic")
                    
                    # First enhance the response with GPT to get a better quality answer
                    base_response = self.gpt_model.enhance_response(question, rag_response)
                    
                    # Add trusted sources based on the detected topic
                    initial_response, source_info = self._add_trusted_sources(base_response, question)
                    
                else:
                    # RAG is confident, use its response directly
                    initial_response = rag_response

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

            # Record timing metrics for this request
            elapsed_time = time.time() - start_time
            record_time(f"response_time_{category}", elapsed_time)

            # Track total tokens for OpenAI-based responses (approximation)
            if category in ['conversational', 'policy']:
                approx_tokens = len(question.split()) + len(final_response.split())
                record_api_call("openai", approx_tokens)
                
            # Store response for evaluation
            self.recent_responses.append({
                'question': question,
                'response': final_response,
                'category': category,
                'timestamp': time.time()
            })
            
            # Limit the number of tracked responses
            while len(self.recent_responses) > self.max_tracked_responses:
                self.recent_responses.pop(0)

            return final_response

        except Exception as e:
            logger.error(f"Error processing single query: {str(e)}", exc_info=True)
            return "I'm sorry, I encountered an error with that question. Could you try rephrasing it?"