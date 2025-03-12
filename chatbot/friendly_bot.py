import logging
import random

logger = logging.getLogger(__name__)

class FriendlyBot:
    """
    Adds friendly, empathetic elements to the chatbot responses
    """
    def __init__(self):
        """Initialize the friendly bot component"""
        logger.info("Initializing Friendly Bot")
        self.greetings = [
            "Hi there! ", 
            "Hello! ", 
            "Hey! ", 
            "Hi! "
        ]
        
        self.empathetic_phrases = [
            "I understand this can be a sensitive topic, and I'm here to support you. ",
            "It's completely okay to have questions about this. I'm here to provide information. ",
            "I appreciate you reaching out with this question. ",
            "This is an important topic, and I'm here to help with accurate information. ",
            "Thank you for trusting me with your question. I'll do my best to provide helpful information. "
        ]
        
        self.reassurance_phrases = [
            "I hope that helps! Let me know if you have any other questions. ",
            "Please feel free to ask if you need any clarification. ",
            "Remember, it's always best to consult with a healthcare provider for personalized advice. ",
            "I'm here if you need more information. ",
            "Feel free to ask more questions if something isn't clear. "
        ]
        
        self.caring_phrases = [
            "Taking care of your reproductive health is important. ",
            "Your health and wellbeing matter. ",
            "It's great that you're seeking information about your health. ",
            "Knowledge is an important part of healthcare. ",
            "Taking the time to learn about these topics is a great step. "
        ]
    
    def add_friendly_elements(self, message, question_type):
        """
        Add friendly, empathetic elements to the response
        
        Args:
            message (str): The original response message
            question_type (str): Type of question ('personal', 'informational', 'policy')
        
        Returns:
            str: Enhanced friendly response
        """
        try:
            # Determine if this is a new conversation (more comprehensive check for greeting words or phrases)
            greeting_words = ["hello", "hi ", "hey", "welcome", "how are you", "doing well", "thanks for asking"]
            is_greeting = any(word in message.lower() for word in greeting_words)
            
            # Don't modify conversational exchanges or greetings
            if is_greeting or len(message.split()) < 15:  # Simple response or greeting
                return message
            
            # Start with empathy for personal questions
            if question_type == 'personal':
                prefix = random.choice(self.empathetic_phrases)
            # Use caring phrases for informational questions
            elif question_type == 'informational':
                prefix = random.choice(self.caring_phrases) if random.random() < 0.3 else ""
            # No special prefix for policy questions
            else:
                prefix = ""
            
            # Add reassurance at the end sometimes
            suffix = random.choice(self.reassurance_phrases) if random.random() < 0.4 else ""
            
            # Combine elements
            friendly_message = f"{prefix}{message}{suffix}"
            
            return friendly_message.strip()
        
        except Exception as e:
            logger.error(f"Error adding friendly elements: {str(e)}", exc_info=True)
            return message  # Return original message if enhancement fails
    
    def detect_question_type(self, question):
        """
        Detect the type of question being asked
        
        Args:
            question (str): User's question
        
        Returns:
            str: Question type ('personal', 'informational', 'policy')
        """
        question_lower = question.lower()
        
        # Check for personal question indicators
        personal_indicators = [
            "i am", "i'm", "i have", "i've", "my", "me", "myself", 
            "worried", "scared", "afraid", "nervous", "help me"
        ]
        if any(indicator in question_lower for indicator in personal_indicators):
            return 'personal'
        
        # Check for policy question indicators
        policy_indicators = [
            "law", "legal", "state", "policy", "ban", "illegal", 
            "allowed", "permit", "legislation"
        ]
        if any(indicator in question_lower for indicator in policy_indicators):
            return 'policy'
        
        # Default to informational
        return 'informational'
