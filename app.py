import os
import logging
from flask import Flask, render_template, request, jsonify
from chatbot.conversation_manager import ConversationManager
from utils.text_processing import PIIDetector # Added import for PII detection

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.config['JSON_AS_ASCII'] = False  # Ensure proper UTF-8 response

# Initialize conversation manager
conversation_manager = ConversationManager()

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST']) # Added /api/chat endpoint
def chat():
    """
    Chat API endpoint with PII detection
    """
    try:
        # Get message from request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing message parameter'}), 400

        message = data['message']
        
        # Log the incoming question for debugging
        logger.debug(f"Received chat message: {message}")

        # Check for PII in the message
        pii_detector = PIIDetector()

        pii_warning = pii_detector.warn_about_pii(message)
        if pii_warning:
            logger.warning("PII detected in user message")

            # If PII is detected, redact it before processing
            redacted_message, _ = pii_detector.redact_pii(message)
            response_data = conversation_manager.process_message(redacted_message)

            # Add PII warning to the response text
            response_data['text'] = f"{pii_warning}\n\n{response_data['text']}"
        else:
            # Get response from conversation manager
            response_data = conversation_manager.process_message(message)
            
        # Import visual info if needed
        from chatbot.visual_info import VisualInfoGraphics
        visual_info = VisualInfoGraphics()
        
        # Add visual graphics if relevant
        response_data = visual_info.add_graphics_to_response(response_data, message)

        return jsonify({
            'response': response_data['text'],
            'citations': response_data['citations'],
            'citation_objects': response_data['citation_objects'],
            'graphics': response_data.get('graphics', [])
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred processing your request'}), 500

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