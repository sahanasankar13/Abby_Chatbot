import os
import logging
from flask import Flask, render_template, request, jsonify
from chatbot.conversation_manager import ConversationManager

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Initialize conversation manager
conversation_manager = ConversationManager()

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Process chat messages and return responses"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        logger.debug(f"Received message: {user_message}")
        
        # Process the message through the conversation manager
        response = conversation_manager.process_message(user_message)
        logger.debug(f"Generated response: {response}")
        
        return jsonify({
            "response": response
        })
    
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Sorry, I encountered an error processing your message. Please try again."
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({"status": "ok"})

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}", exc_info=True)
    return jsonify({"error": "An internal server error occurred. Please try again later."}), 500
