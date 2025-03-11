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
            The user asked: "{question}" about abortion policies in {state}.
            
            Here is the raw API data about abortion policies in {state}:
            {policy_json}
            
            Please format this information into a clear, well-structured response that:
            1. Organizes the information by category (gestational limits, waiting periods, insurance, etc.)
            2. Uses bullet points and clear formatting to make the information easy to read
            3. Is empathetic and conversational while remaining factual
            4. Addresses the specific aspects the user asked about, if mentioned
            5. Reminds the user that laws can change and they should consult healthcare providers for the most current information
            6. Can handle multiple queries about different aspects of abortion policy
            
            Keep your response comprehensive but concise and friendly. Don't mention that you're formatting API data.
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
