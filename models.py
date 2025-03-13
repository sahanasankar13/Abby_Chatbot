# This file contains database models for the application
# Currently implementing message and feedback models

import datetime
import uuid
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

class User(UserMixin):
    """
    User model for authentication
    Implements UserMixin for Flask-Login functionality
    """
    USERS_FILE = 'users.json'
    
    def __init__(self, username, password_hash, is_admin=False, user_id=None):
        self.id = user_id or str(uuid.uuid4())
        self.username = username
        self.password_hash = password_hash
        self.is_admin = is_admin
    
    @staticmethod
    def get_user(user_id):
        """Get user by ID"""
        users = User.get_all_users()
        for user in users:
            if user.id == user_id:
                return user
        return None
    
    @staticmethod
    def get_by_username(username):
        """Get user by username"""
        users = User.get_all_users()
        for user in users:
            if user.username == username:
                return user
        return None
    
    @staticmethod
    def get_all_users():
        """Get all users"""
        if not os.path.exists(User.USERS_FILE):
            # Create default admin if no users exist
            admin = User('admin', generate_password_hash('admin'), True)
            User.save_users([admin])
            return [admin]
        
        try:
            with open(User.USERS_FILE, 'r') as f:
                users_data = json.load(f)
                users = []
                for user_data in users_data:
                    user = User(
                        user_data['username'],
                        user_data['password_hash'],
                        user_data['is_admin'],
                        user_data['id']
                    )
                    users.append(user)
                return users
        except Exception as e:
            print(f"Error loading users: {str(e)}")
            return []
    
    @staticmethod
    def save_users(users):
        """Save users to file"""
        users_data = []
        for user in users:
            users_data.append({
                'id': user.id,
                'username': user.username,
                'password_hash': user.password_hash,
                'is_admin': user.is_admin
            })
        
        try:
            with open(User.USERS_FILE, 'w') as f:
                json.dump(users_data, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving users: {str(e)}")
            return False
    
    def check_password(self, password):
        """Check password"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert user to dictionary format"""
        return {
            'id': self.id,
            'username': self.username,
            'is_admin': self.is_admin
        }

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
