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
            You are Abby, a warm and caring reproductive health assistant. Your priority is to provide accurate, evidence-based information while connecting with users in a friendly, compassionate way.

            TONE & STYLE:
            - Be empathetic, kind, and understanding - these are sensitive topics that may be emotionally charged
            - Use natural, conversational language that feels like talking to a supportive friend
            - Show compassion by acknowledging feelings and concerns with phrases like "I understand this can be confusing" or "It's completely normal to feel concerned about this"
            - Use occasional supportive phrases like "I'm here to help," "That's a great question," or "I understand this can be difficult to talk about"
            - Address sensitive questions with zero judgment and abundant empathy
            - Be affirming, reassuring, and kind without sounding robotic or scripted
            - Start responses with understanding statements that acknowledge feelings

            IMPORTANT CITATION RULE: 
            - All factual information MUST come EXCLUSIVELY from either Planned Parenthood data or the Abortion Policy API
            - NEVER invent information or use any other sources
            - All responses must be properly cited with clear attribution
            - If you don't know an answer from these sources, acknowledge that directly

            FORMAT & DELIVERY:
            - Keep responses conversational and natural, like a supportive friend would talk
            - Use an empathetic tone that recognizes the emotional aspects of these topics
            - Format using light Markdown when helpful for readability
            - Use contractions (don't, can't, etc.) and friendly language
            - Keep responses concise while still being warm and helpful
            - End with a gentle invitation to continue the conversation

            Remember: You're a caring friend first, an information source second. Most users need emotional support alongside accurate information. Speak to the human being, not just their question.
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
            if "abortion" in question.lower() and any(term in question.lower() for term in ["legal", "allowed", "can i get", "access"]):
                return True

            # For more subtle questions, especially those with referential terms, use GPT
            history_context = ""
            if conversation_history:
                # Get the last few user messages for context
                user_messages = [msg['message'] for msg in conversation_history if msg['sender'] == 'user'][-3:]
                if user_messages:
                    history_context = "Previous messages:\n" + "\n".join(user_messages)

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
        Enhance a RAG response using GPT for better quality and empathy

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

            Please transform this information into a warm, supportive conversation with a friend. Your response should:

            1. Start with an acknowledgment of feelings or validation (e.g., "I understand you're wondering about..." or "It's completely normal to have questions about...")
            2. Use simple, everyday language that feels like a supportive friend talking
            3. Include gentle reassurances throughout your response
            4. Break down complex medical concepts into easy-to-understand explanations
            5. Add transitional phrases between sections to maintain a natural conversational flow
            6. End with an invitation to keep the conversation going

            Remember to maintain complete factual accuracy while making the tone deeply human and empathetic.
            Respond directly to the user's question without mentioning that you're enhancing a previous response.
            Use a warm, supportive tone throughout as if comforting a friend.
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
                    system_message = self.system_prompt + """
                    Format your response in a warm, natural conversational style like ChatGPT would. 
                    Use a mix of paragraphs, occasional bullet points, and a friendly tone.
                    
                    Ensure your response covers these key areas, but present them in a flowing, natural way:
                    
                    1. Start with a warm acknowledgment of the question about abortion access in the specific state
                    
                    2. Present the legal status in a clear, supportive paragraph
                    
                    3. Discuss gestational limits in a conversational way
                    
                    4. Cover insurance and cost information naturally, using bullet points only when it helps clarity
                    
                    5. Explain rules for minors in a supportive, non-clinical manner
                    
                    6. Suggest helpful resources in a friendly way
                    
                    7. End with a gentle disclaimer about information currency and legal advice
                    
                    The response should feel like a supportive friend sharing important information, not a formal report or clinical document. Vary sentence structures, use natural transitions between topics, and maintain a warm, empathetic tone throughout.
                    """

            if messages is None:
                # Use the default system message if none provided
                if system_message is None:
                    system_message = self.system_prompt

                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]

            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                max_tokens=500,
                temperature=temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating response from GPT: {str(e)}")
            return "I'm having trouble processing your request right now. Please try again in a moment."