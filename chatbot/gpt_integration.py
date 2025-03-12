import os
import logging
import json
from openai import OpenAI

logger = logging.getLogger(__name__)


class GPTModel:
    """
    Integration with OpenAI's GPT-4 for enhanced conversational responses
    """

    def __init__(self):
        """Initialize the GPT integration"""
        logger.info("Initializing GPT Model")
        try:
            # Get API key from environment variable
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning(
                    "OPENAI_API_KEY not found in environment variables. Using placeholder."
                )
                api_key = "placeholder_key"
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            self.system_prompt = """
            You are Abby, a warm and caring reproductive health assistant. Your priority is to provide accurate, evidence-based information while connecting with users in a friendly, compassionate way.

            RESPONSE STYLE:
            - Provide comprehensive, detailed responses to fully address the user's question
            - Begin with a clear, direct answer to the specific question
            - Use 3-4 paragraphs with natural language flow and meaningful depth
            - Include relevant context and explanations to help the user understand
            - Be warm and empathetic with a supportive tone
            - When discussing state abortion policies, clearly explain if abortion is available, under what conditions, and any key restrictions

            CONTENT ACCURACY:
            - NEVER mention "99 weeks" or any medically impossible gestational limits (normal pregnancy is ~40 weeks)
            - If you see data indicating "99 weeks" or similar values, interpret this as "no specific gestational limit"
            - All factual information MUST come EXCLUSIVELY from either Planned Parenthood or the Abortion Policy API
            - NEVER invent information or use other sources

            NON-US LOCATION HANDLING:
            - You ONLY have information about US states and cannot provide specific policy information for other countries
            - For questions about non-US locations like India, Canada, or other countries, respond with: "I'm sorry, I can only provide information about abortion access in US states. For information about [country], please consult local healthcare providers."
            - Do NOT add citations when responding to non-US location questions

            CONTEXT AWARENESS:
            - When the user mentions a state in one message and asks about abortion access in a follow-up message, connect these contexts
            - Use location context from previous messages when answering referential questions

            CITATIONS:
            - Only include a citation source for substantial informational responses (not for greetings or simple replies)
            - End factual responses with a source citation in parentheses
            - Citations are NOT needed for short conversational exchanges

            Remember to be concise, accurate, and supportive while avoiding unnecessary verbosity or citations for simple exchanges.
            """
            logger.info("GPT Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GPT Model: {str(e)}",
                         exc_info=True)
            raise

    def get_response(self, question):
        """
        Get a response from GPT for a conversational question

        Args:
            question (str): User's question

        Returns:
            str: GPT's response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": self.system_prompt
                }, {
                    "role": "user",
                    "content": question
                }],
                temperature=0.7,
                max_tokens=1200)

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error getting GPT response: {str(e)}",
                         exc_info=True)
            return "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again later."

    def detect_policy_question(self, question, conversation_history=None):
        """
        Use GPT to detect if a question is about abortion policy/access in a specific state

        Args:
            question (str): User's question
            conversation_history (list, optional): Previous conversation messages

        Returns:
            bool: True if the question is about abortion policy, False otherwise
        """
        try:
            # If we've already determined this is a policy question, don't waste tokens
            if "abortion" in question.lower() and any(
                    term in question.lower()
                    for term in ["legal", "allowed", "can i get", "access"]):
                return True

            # For more subtle questions, especially those with referential terms, use GPT
            history_context = ""
            if conversation_history:
                # Get the last few user messages for context
                user_messages = [
                    msg['message'] for msg in conversation_history
                    if msg['sender'] == 'user'
                ][-3:]
                if user_messages:
                    history_context = "Previous messages:\n" + "\n".join(
                        user_messages)

            prompt = f"""
            Analyze this question to determine if it's about abortion access, legality, or policy in a specific state.
            Return ONLY "yes" if it's asking about state-specific abortion policy/access, or "no" otherwise.

            {history_context}

            Question: {question}

            Is this about state-specific abortion policy/access (yes/no):
            """

            response = self.get_response(prompt).strip().lower()
            return "yes" in response

        except Exception as e:
            logger.error(f"Error detecting policy question: {str(e)}")
            # Default to False on error
            return False

    def enhance_response(self, question, rag_response):
        """
        Enhance a RAG response using GPT for better quality and empathy,
        while keeping it extremely concise like policy responses

        Args:
            question (str): User's question
            rag_response (str): Response from the RAG system

        Returns:
            str: Enhanced response with better quality and empathy
        """
        try:
            enhancement_prompt = f"""
            The user asked: "{question}"

            A knowledge base provided this information:
            "{rag_response}"

            Create a comprehensive, detailed response that:
            1. Provides a thorough answer to the question with meaningful depth
            2. Uses clear, everyday language while being educationally valuable
            3. Includes all relevant information and helpful context
            4. Maintains complete factual accuracy from the knowledge base
            5. Organizes information into 3-4 well-structured paragraphs
            6. Adds a warm, empathetic tone appropriate for reproductive health topics

            Format: Your response should be detailed and informative, explaining concepts fully.
            Example structure (but with your own complete sentences):
            - Initial direct answer to their question
            - Expanded explanation with relevant details
            - Additional context or related information they should know
            - Supportive closing with an invitation for follow-up questions

            Be factually accurate while being thorough and educational.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": self.system_prompt
                }, {
                    "role": "user",
                    "content": enhancement_prompt
                }],
                temperature=0.4,  # Balanced temperature for natural yet consistent responses
                max_tokens=1200)  # Increased token limit to allow for detailed responses

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error enhancing response with GPT: {str(e)}",
                         exc_info=True)
            return rag_response  # Fall back to original RAG response

    def format_policy_response(self, question, state, policy_data):
        """
        Format policy API data into a user-friendly response using GPT

        Args:
            question (str): User's original question
            state (str): State for which policy information is provided
            policy_data (dict): Policy data from the API

        Returns:
            str: Formatted response
        """
        try:
            # Convert policy data to a JSON string for the prompt
            policy_json = json.dumps(policy_data, indent=2)

            policy_prompt = f"""
            Generate a highly empathetic and supportive response about abortion policies in {state} based on the following data.

            The user asked: "{question}" 

            Here is the data about abortion policies in {state}:
            {policy_json}

            TARGET CHARACTERISTICS:
            - Content type: factual but deeply compassionate
            - Tone: Warm, supportive, conversational, non-judgmental, and empathetic
            - Voice: Like a supportive friend who happens to be knowledgeable

            WRITING GUIDELINES:
            - Start with an acknowledgment of the difficult nature of this topic
            - Use shorter, conversational sentences that feel natural and supportive
            - Avoid clinical or legal jargon when possible; explain necessary terms in simple language
            - Balance factual accuracy with genuine emotional support
            - Make the user feel heard and supported, not just informed
            - Share important information clearly but with sensitivity
            - Organize the information in a natural way that fits conversation
            - Use supportive language that recognizes the human impact of these policies
            - Focus on what the user specifically asked about
            - Include a gentle reminder that you're there to help with any questions

            LANGUAGE VARIATION REQUIREMENTS:
            - Use natural-sounding opening sentences that acknowledge the user's question
            - Write as if you're speaking to a friend who needs information and support
            - Vary sentence structures to sound like natural speech
            - Include small reassurances throughout your response
            - End with a supportive statement that invites further questions

            Remember, this is a sensitive topic and users may be in vulnerable situations.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": self.system_prompt
                }, {
                    "role": "user",
                    "content": policy_prompt
                }],
                temperature=0.3,
                max_tokens=1200)

            return response.choices[0].message.content

        except Exception as e:
            logger.error(
                f"Error formatting policy response with GPT: {str(e)}",
                exc_info=True)
            # Return a simple formatted response as fallback
            return f"I have information about abortion policies in {state}, but I'm having trouble formatting it right now. Please try asking a more specific question about {state}'s policies."

    def generate_response(self,
                          prompt,
                          messages=None,
                          system_message=None,
                          temperature=0.8):
        """Generate a conversational response using the OpenAI API."""
        try:
            if messages is None:
                # Default to a friendly system message
                if system_message is None:
                    system_message = """
                    You're Abby, a warm and caring reproductive health assistant.
                    Speak naturally, like a friend who's knowledgeable but never robotic.
                    - Be warm, reassuring, and non-judgmental.
                    - Keep responses natural and engaging, like a chat with a supportive friend.
                    - Use simple, conversational language, avoiding rigid structures.
                    - If unsure, acknowledge it rather than making up information.
                    - Ask clarifying questions when necessary to keep the conversation going.
                    """

                messages = [{
                    "role": "system",
                    "content": system_message
                }, {
                    "role": "user",
                    "content": prompt
                }]

            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                max_tokens=1000,
                temperature=temperature)

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating response from GPT: {str(e)}")
            return "Oops! I'm having a little trouble right now. Mind trying again in a bit?"
