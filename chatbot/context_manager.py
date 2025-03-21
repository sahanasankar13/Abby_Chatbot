from typing import List, Dict, Any, Optional
import logging
from collections import deque

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages conversation context and history with efficient memory usage"""
    
    def __init__(self, max_context_length: int = 3):
        """
        Initialize the context manager
        
        Args:
            max_context_length (int): Maximum number of previous messages to keep in context
        """
        self.max_context_length = max_context_length
        self.conversation_history = deque(maxlen=max_context_length)
        self.current_state = None
        self.location_context = None
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """
        Add a message to the conversation history
        
        Args:
            message (Dict[str, Any]): Message to add with keys 'sender', 'message', 'timestamp'
        """
        self.conversation_history.append(message)
        self._update_state(message)
    
    def get_context(self) -> List[Dict[str, Any]]:
        """
        Get the current conversation context
        
        Returns:
            List[Dict[str, Any]]: List of recent messages
        """
        return list(self.conversation_history)
    
    def get_location_context(self) -> Optional[str]:
        """
        Get the current location context
        
        Returns:
            Optional[str]: Current location context if available
        """
        return self.location_context
    
    def set_location_context(self, location: str) -> None:
        """
        Set the location context
        
        Args:
            location (str): Location to set as context
        """
        self.location_context = location
        logger.info(f"Location context set to: {location}")
    
    def clear_context(self) -> None:
        """Clear all context and history"""
        self.conversation_history.clear()
        self.current_state = None
        self.location_context = None
        logger.info("Context cleared")
    
    def _update_state(self, message: Dict[str, Any]) -> None:
        """
        Update the conversation state based on new message
        
        Args:
            message (Dict[str, Any]): New message
        """
        # Extract key information from message
        if message['sender'] == 'user':
            msg_lower = message['message'].lower()
            
            # Update location context if message contains location information
            if any(state_indicator in msg_lower for state_indicator in 
                  ['in', 'at', 'state of', 'policy in', 'laws in']):
                # Note: Actual state detection would be handled by PolicyAPI
                logger.debug("Message may contain location context")
            
            # Track conversation topic
            self.current_state = {
                'last_user_message': message['message'],
                'timestamp': message.get('timestamp')
            }
    
    def get_relevant_history(self, query: str) -> List[Dict[str, Any]]:
        """
        Get history relevant to the current query
        
        Args:
            query (str): Current user query
            
        Returns:
            List[Dict[str, Any]]: Relevant conversation history
        """
        # For now, simply return recent history
        # This could be enhanced with semantic similarity matching
        return list(self.conversation_history) 