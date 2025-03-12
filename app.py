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

# Initialize conversation manager with response evaluation
# Options for evaluation_model: "openai", "local", "both"
evaluation_model = os.environ.get("EVALUATION_MODEL", "both")
logger.info(f"Initializing conversation manager with evaluation_model={evaluation_model}")
conversation_manager = ConversationManager(evaluation_model=evaluation_model)

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
            
        # Graphics disabled per user request
        # from chatbot.visual_info import VisualInfoGraphics
        # visual_info = VisualInfoGraphics()
        # response_data = visual_info.add_graphics_to_response(response_data, message)

        return jsonify({
            'response': response_data['text'],
            'citations': response_data['citations'],
            'citation_objects': response_data['citation_objects'],
            'graphics': []  # Empty array to avoid frontend errors
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred processing your request'}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    API endpoint for submitting user feedback on chatbot responses
    """
    try:
        # Get feedback data from request
        data = request.get_json()
        if not data or 'message_id' not in data or 'rating' not in data:
            return jsonify({'error': 'Missing required feedback parameters'}), 400

        message_id = data['message_id']
        rating = data['rating']  # 1 for thumbs up, -1 for thumbs down
        comment = data.get('comment', None)  # Optional comment
        
        # Check for PII in the comment if provided
        if comment:
            pii_detector = PIIDetector()
            if pii_detector.has_pii(comment):
                # Redact any PII from the comment
                logger.warning("PII detected in feedback comment, redacting")
                comment, redacted_items = pii_detector.redact_pii(comment)
                logger.info(f"Redacted {len(redacted_items)} PII items from feedback comment")
        
        # Log the feedback with redacted comment for debugging
        logger.info(f"Received feedback for message {message_id}: rating={rating}")
        
        # Store feedback using the feedback manager
        from utils.feedback_manager import FeedbackManager
        from utils.metrics import record_feedback
        
        # Track feedback in metrics
        record_feedback(positive=(rating > 0))
        
        feedback_manager = FeedbackManager()
        success = feedback_manager.add_feedback(message_id, rating, comment)
        
        if success:
            return jsonify({'success': True, 'message': 'Feedback recorded successfully'})
        else:
            return jsonify({'error': 'Failed to record feedback'}), 500
            
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred processing your feedback'}), 500

@app.route('/admin/feedback', methods=['GET'])
def view_feedback():
    """
    Admin dashboard for viewing feedback statistics
    """
    try:
        from utils.feedback_manager import FeedbackManager
        feedback_manager = FeedbackManager()
        
        # Get all feedback data
        all_feedback = feedback_manager.get_all_feedback()
        stats = feedback_manager.get_feedback_stats()
        
        return render_template('admin/feedback.html', feedback=all_feedback, stats=stats)
    except Exception as e:
        logger.error(f"Error retrieving feedback data: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred retrieving feedback data'}), 500
        
@app.route('/admin/metrics', methods=['GET'])
def view_metrics():
    """
    Admin dashboard for viewing chatbot performance metrics
    """
    try:
        from utils.metrics_analyzer import MetricsAnalyzer
        import datetime
        
        # Get date range parameters
        end_date = request.args.get('end_date')
        start_date = request.args.get('start_date')
        
        # Parse dates
        if end_date:
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            # Set to end of day
            end_date = end_date.replace(hour=23, minute=59, second=59)
        else:
            end_date = datetime.datetime.now()
            
        if start_date:
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        else:
            # Default to 30 days ago
            start_date = end_date - datetime.timedelta(days=30)
        
        # Format dates for form display
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Get metrics
        metrics_analyzer = MetricsAnalyzer()
        metrics = metrics_analyzer.get_metrics(start_date, end_date)
        
        return render_template('admin/metrics.html', 
                              metrics=metrics, 
                              start_date=start_date_str,
                              end_date=end_date_str)
    except Exception as e:
        logger.error(f"Error retrieving metrics data: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred retrieving metrics data'}), 500

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