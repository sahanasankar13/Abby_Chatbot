# This file contains database models for the application
# Currently implementing message and feedback models

import datetime
import uuid

class ChatMessage:
    """Represents a message in the chat conversation"""
    def __init__(self, content, sender, message_id=None, timestamp=None):
        self.content = content
        self.sender = sender  # 'user' or 'bot'
        self.message_id = message_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.datetime.now()
        
    def to_dict(self):
        """Convert message to dictionary format"""
        return {
            "message_id": self.message_id,
            "content": self.content,
            "sender": self.sender,
            "timestamp": self.timestamp.isoformat()
        }

class UserFeedback:
    """Represents user feedback on a chatbot response"""
    def __init__(self, message_id, rating, comment=None, timestamp=None):
        self.message_id = message_id  # ID of the message being rated
        self.rating = rating  # Positive (1) or negative (-1) rating
        self.comment = comment  # Optional comment from user
        self.timestamp = timestamp or datetime.datetime.now()
        
    def to_dict(self):
        """Convert feedback to dictionary for storage/retrieval"""
        return {
            "message_id": self.message_id,
            "rating": self.rating,
            "comment": self.comment,
            "timestamp": self.timestamp.isoformat()
        }
