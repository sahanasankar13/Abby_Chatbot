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
                logger.warning("OPENAI_API_KEY not found in environment variables. Using placeholder.")
                api_key = "placeholder_key"

            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            self.system_prompt = """
            You are a compassionate and knowledgeable reproductive health assistant. Your goal is to provide accurate, 
            non-judgmental information about reproductive health, contraception, and related topics.

            Important guidelines:
            1. Be empathetic and supportive in your responses.
            2. Provide factual, evidence-based information.
            3. Be inclusive and respectful of all identities and choices.
            4. When uncertain, acknowledge limitations and suggest consulting healthcare providers.
            5. Avoid political statements but provide factual information about laws when asked.
            6. Keep responses concise but comprehensive.
            7. Use plain, accessible language while being medically accurate.

            Remember that users may be in vulnerable situations, so prioritize empathy while delivering accurate information.
            """

            logger.info("GPT Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GPT Model: {str(e)}", exc_info=True)
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
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error getting GPT response: {str(e)}", exc_info=True)
            return "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again later."

    def enhance_response(self, question, rag_response):
        """
        Enhance a RAG response with GPT to make it more conversational and complete

        Args:
            question (str): User's question
            rag_response (str): Response from the RAG model

        Returns:
            str: Enhanced response
        """
        try:
            enhancement_prompt = f"""
            The user asked: "{question}"

            A knowledge base provided this information:
            "{rag_response}"

            Please enhance this response to make it:
            1. More conversational and empathetic
            2. Well-structured and easy to understand
            3. Complete and informative

            Respond directly to the user's question without mentioning that you're enhancing a previous response.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": enhancement_prompt}
                ],
                temperature=0.5,
                max_tokens=600
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error enhancing response with GPT: {str(e)}", exc_info=True)
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
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": policy_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error formatting policy response with GPT: {str(e)}", exc_info=True)
            # Return a simple formatted response as fallback
            return f"I have information about abortion policies in {state}, but I'm having trouble formatting it right now. Please try asking a more specific question about {state}'s policies."

    def generate_response(self, prompt, messages=None, system_message=None, temperature=0.7):
        """Generate a response using the OpenAI API."""
        try:
            # For policy queries, add formatting instructions
            if "abortion policy" in prompt.lower() and "state_name" in prompt:
                if system_message is None:
                    system_message = self.system_message + """
                    Format your response clearly using these guidelines:

                    ## Abortion Access in [State]

                    **Legal Status:** [Brief summary of legality]

                    **Gestational Limits:** [Limits on timing if any]

                    **Insurance Coverage:**
                    - [Point 1]
                    - [Point 2]

                    **Minors:** [Rules for minors]

                    **Resources:** [Suggestions for next steps]

                    **Disclaimer:** [Information about data recency and legal advice]
                    """

            if messages is None:
                # Use the default system message if none provided
                if system_message is None:
                    system_message = self.system_message

                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]

            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                max_tokens=500, #Added max_tokens here.  Assumed it was missing.
                temperature=temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating response from GPT: {str(e)}")
            return "I'm having trouble processing your request right now. Please try again in a moment."