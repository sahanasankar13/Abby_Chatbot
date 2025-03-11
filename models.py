# This file would contain database models if needed
# Currently we're not implementing database persistence for the chat history
# but this file is included for potential future expansion

class ChatMessage:
    """Represents a message in the chat conversation"""
    def __init__(self, content, sender, timestamp=None):
        self.content = content
        self.sender = sender  # 'user' or 'bot'
        self.timestamp = timestamp
