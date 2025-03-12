import logging
import random
import re

logger = logging.getLogger(__name__)

class FriendlyBot:
    """
    Adds friendly, empathetic elements to the chatbot responses
    and improves the structure and quality of responses
    """
    def __init__(self):
        """Initialize the friendly bot component"""
        logger.info("Initializing Friendly Bot")
        self.greetings = [
            "Hi there! ", 
            "Hello! ", 
            "Hey! ", 
            "Hi! I'm Abby. "
        ]
        
        self.empathetic_phrases = [
            "I understand this can be a sensitive topic, and I'm here to support you. ",
            "It's completely okay to have questions about this. I'm here to provide information without judgment. ",
            "I appreciate you reaching out with this question. These are important topics to discuss. ",
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
        
        # Introduction for complex topic explanations
        self.educational_intros = [
            "Based on your question, here's what I know: ",
            "Here's some helpful information about that: ",
            "Let me explain how this works: ",
            "I'd like to provide some information that might help: "
        ]
    
    def add_friendly_elements(self, message, question_type):
        """
        Add friendly, empathetic elements to the response and improve structure
        
        Args:
            message (str): The original response message
            question_type (str): Type of question ('personal', 'informational', 'policy')
        
        Returns:
            str: Enhanced friendly response with improved structure
        """
        try:
            # Comprehensive check for greetings and simple conversational responses
            greeting_words = ["hello", "hi ", "hey", "welcome", "how are you", "doing well", "thanks for asking", "nice to meet", "pleasure to meet"]
            conversational_markers = ["can help", "assist you", "what can i", "how can i", "i'm here", "here to help"]
            
            # Check if message is a greeting or simple conversational response
            is_greeting = any(word in message.lower() for word in greeting_words)
            is_simple_conversation = any(marker in message.lower() for marker in conversational_markers)
            
            # Don't modify greetings or short conversational exchanges
            if is_greeting or is_simple_conversation or len(message.split()) < 20:  # Simple response or greeting
                return message
            
            # Format the original message with better paragraph breaks
            formatted_message = self._format_paragraphs(message)
            
            # Add educational intro for informational content
            if len(formatted_message.split()) > 40 and question_type in ['informational', 'policy']:
                intro = random.choice(self.educational_intros)
                formatted_message = f"{intro}{formatted_message}"
            
            # Start with empathy for personal questions
            if question_type == 'personal':
                prefix = random.choice(self.empathetic_phrases)
            # Use caring phrases for informational questions
            elif question_type == 'informational':
                prefix = random.choice(self.caring_phrases) if random.random() < 0.5 else ""
            # Lighter touch for policy questions
            else:
                prefix = ""
            
            # Add reassurance at the end sometimes
            suffix = random.choice(self.reassurance_phrases) if random.random() < 0.6 else ""
            
            # Combine elements
            friendly_message = f"{prefix}{formatted_message}{suffix}"
            
            return friendly_message.strip()
        
        except Exception as e:
            logger.error(f"Error adding friendly elements: {str(e)}", exc_info=True)
            return message  # Return original message if enhancement fails
    
    def _format_paragraphs(self, message):
        """
        Improve paragraph formatting for better readability
        
        Args:
            message (str): Original message
            
        Returns:
            str: Message with improved paragraph formatting
        """
        # Split message into sentences
        sentences = re.split(r'(?<=[.!?])\s+', message)
        
        if len(sentences) <= 2:
            return message
            
        # Group sentences into paragraphs (3-4 sentences per paragraph)
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            
            # Create a new paragraph every 2-3 sentences or when topic seems to change
            if len(current_paragraph) >= 3 or self._is_topic_change(sentence):
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # Add any remaining sentences
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            
        # Join paragraphs with double line breaks
        return '\n\n'.join(paragraphs)
    
    def _is_topic_change(self, sentence):
        """
        Detect if a sentence likely indicates a topic change
        
        Args:
            sentence (str): The sentence to analyze
            
        Returns:
            bool: True if it seems to be a topic change
        """
        topic_change_indicators = [
            "another", "additionally", "moreover", "furthermore", 
            "however", "on the other hand", "in contrast",
            "first", "second", "third", "finally", "lastly"
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in topic_change_indicators)
    
    def detect_question_type(self, question):
        """
        Detect the type of question being asked
        
        Args:
            question (str): User's question
        
        Returns:
            str: Question type ('personal', 'informational', 'policy')
        """
        question_lower = question.lower()
        
        # Check for personal question indicators - expanded list
        personal_indicators = [
            "i am", "i'm", "i have", "i've", "my", "me", "myself", 
            "worried", "scared", "afraid", "nervous", "help me",
            "i feel", "feeling", "stressed", "anxious", "concerned",
            "i need help", "i don't know what to do", "unsure"
        ]
        if any(indicator in question_lower for indicator in personal_indicators):
            return 'personal'
        
        # Check for policy question indicators
        policy_indicators = [
            "law", "legal", "state", "policy", "ban", "illegal", 
            "allowed", "permit", "legislation", "restriction",
            "regulation", "rule", "government", "rights"
        ]
        if any(indicator in question_lower for indicator in policy_indicators):
            return 'policy'
        
        # Default to informational
        return 'informational'
