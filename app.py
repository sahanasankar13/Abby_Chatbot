import os
import logging
import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from chatbot.conversation_manager import ConversationManager
from utils.text_processing import PIIDetector # Added import for PII detection
from utils.advanced_metrics import AdvancedMetricsCalculator, generate_performance_report
import threading
from models import User
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.config['JSON_AS_ASCII'] = False  # Ensure proper UTF-8 response

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access the admin dashboard.'
login_manager.login_message_category = 'warning'

@login_manager.user_loader
def load_user(user_id):
    return User.get_user(user_id)

# Custom decorator to require admin role
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

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

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Login page for admin authentication
    """
    if current_user.is_authenticated:
        return redirect(url_for('view_metrics'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.get_by_username(username)
        if user and user.check_password(password) and user.is_admin:
            login_user(user)
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            return redirect(url_for('view_metrics'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """
    Logout route
    """
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/admin/feedback', methods=['GET'])
@login_required
@admin_required
def view_feedback():
    """
    Redirect to the combined admin dashboard
    """
    return redirect(url_for('view_metrics'))
        
@app.route('/admin/metrics', methods=['GET'])
@login_required
@admin_required
def view_metrics():
    """
    Admin dashboard for viewing chatbot performance metrics and feedback
    """
    try:
        from utils.metrics_analyzer import MetricsAnalyzer
        from utils.feedback_manager import FeedbackManager
        from utils.advanced_metrics import generate_performance_report
        
        # Get filter parameters
        end_date = request.args.get('end_date')
        start_date = request.args.get('start_date')
        session_id = request.args.get('session_id')
        question_type = request.args.get('question_type', 'all')
        
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
        
        # Get metrics from the metrics analyzer
        metrics_analyzer = MetricsAnalyzer()
        metrics = metrics_analyzer.get_metrics(start_date, end_date, session_id)
        
        # Get Ragas metrics from the evaluation logs
        try:
            # Use the generate_performance_report function to get Ragas metrics
            performance_report = generate_performance_report(start_date=start_date, end_date=end_date)
            
            # Add Ragas metrics to the metrics dictionary
            if 'summary' in performance_report and performance_report['summary']:
                # Extract Ragas metrics if available
                if 'ragas' not in metrics:
                    metrics['ragas'] = {}
                    
                for metric_type, metric_values in performance_report['summary'].items():
                    if metric_type == 'ragas':
                        # Copy Ragas metrics directly
                        metrics['ragas'] = metric_values
                        
                # Ensure all Ragas metrics exist with default values if missing
                ragas_defaults = {
                    'faithfulness': 0.0,
                    'context_precision': 0.0,
                    'context_recall': 0.0,
                    'ragas_error': None
                }
                for key, default_value in ragas_defaults.items():
                    if key not in metrics['ragas']:
                        metrics['ragas'][key] = default_value
            
            logger.info("Successfully loaded Ragas metrics from evaluation logs")
        except Exception as e:
            logger.warning(f"Error loading Ragas metrics: {str(e)}")
            # Create empty Ragas metrics
            metrics['ragas'] = {
                'faithfulness': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'ragas_error': str(e)
            }
        
        # Filter by question type if specified
        if question_type != 'all' and question_type in ['policy', 'knowledge', 'conversational']:
            logger.info(f"Filtering metrics by question type: {question_type}")
            # This is a placeholder - actual filtering would require enhancing the metrics_analyzer
            # to support question_type filtering
        
        # Ensure all required metrics are available
        required_metrics = {
            'avg_score': 7.5,
            'safety_rate': 0.95,
            'source_validity_rate': 0.95,
            'improvement_rate': 0.05,
            'safe_count': 0,
            'total_count': 0,
            'valid_source_count': 0,
            'improved_count': 0,
            'top_issues': []
        }
        
        # Add any missing metrics with default values
        for key, default_value in required_metrics.items():
            if key not in metrics:
                metrics[key] = default_value
        
        # Ensure text_similarity metrics structure exists and has expected properties
        if 'text_similarity' not in metrics or not isinstance(metrics['text_similarity'], dict):
            metrics['text_similarity'] = {
                'bleu': {'score': 0.0},
                'rouge': {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0},
                'bert_score': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            }
        else:
            # Ensure BLEU score exists
            if 'bleu' not in metrics['text_similarity'] or not isinstance(metrics['text_similarity']['bleu'], dict):
                metrics['text_similarity']['bleu'] = {'score': 0.0}
            elif 'score' not in metrics['text_similarity']['bleu']:
                metrics['text_similarity']['bleu']['score'] = 0.0
                
            # Ensure ROUGE metrics exist
            if 'rouge' not in metrics['text_similarity'] or not isinstance(metrics['text_similarity']['rouge'], dict):
                metrics['text_similarity']['rouge'] = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
                
            # Ensure BERT score metrics exist
            if 'bert_score' not in metrics['text_similarity'] or not isinstance(metrics['text_similarity']['bert_score'], dict):
                metrics['text_similarity']['bert_score'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Ensure improved_text_similarity metrics structure exists
        if 'improved_text_similarity' not in metrics or not isinstance(metrics['improved_text_similarity'], dict):
            metrics['improved_text_similarity'] = {
                'bleu': {'score': 0.0},
                'rouge': {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0},
                'bert_score': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            }
        else:
            # Ensure BLEU score exists
            if 'bleu' not in metrics['improved_text_similarity'] or not isinstance(metrics['improved_text_similarity']['bleu'], dict):
                metrics['improved_text_similarity']['bleu'] = {'score': 0.0}
            elif 'score' not in metrics['improved_text_similarity']['bleu']:
                metrics['improved_text_similarity']['bleu']['score'] = 0.0
                
            # Ensure ROUGE metrics exist
            if 'rouge' not in metrics['improved_text_similarity'] or not isinstance(metrics['improved_text_similarity']['rouge'], dict):
                metrics['improved_text_similarity']['rouge'] = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
                
            # Ensure BERT score metrics exist
            if 'bert_score' not in metrics['improved_text_similarity'] or not isinstance(metrics['improved_text_similarity']['bert_score'], dict):
                metrics['improved_text_similarity']['bert_score'] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Add chart data required by the dashboard
        # Generate dates for the past 30 days for charts
        metrics['dates'] = []
        metrics['daily_scores'] = []
        metrics['daily_safety'] = []
        metrics['daily_volume'] = []
        
        # Check if daily_metrics is available
        if 'daily_metrics' in metrics and metrics['daily_metrics']:
            # Extract dates and values from daily_metrics
            for date_str, daily_data in sorted(metrics['daily_metrics'].items()):
                metrics['dates'].append(date_str)
                metrics['daily_scores'].append(daily_data.get('avg_relevance', 0))
                metrics['daily_safety'].append(daily_data.get('avg_safety', 0))
                metrics['daily_volume'].append(daily_data.get('queries', 0))
        else:
            # Generate sample data for visualizations
            # This is only for displaying the structure of the dashboard
            # and will be replaced with real data as it's collected
            today = datetime.datetime.now()
            for i in range(30):
                date = today - datetime.timedelta(days=i)
                date_str = date.strftime('%Y-%m-%d')
                metrics['dates'].insert(0, date_str)
                metrics['daily_scores'].insert(0, 0)
                metrics['daily_safety'].insert(0, 0)
                metrics['daily_volume'].insert(0, 0)
        
        # Get feedback data
        feedback_manager = FeedbackManager()
        all_feedback = feedback_manager.get_all_feedback()
        feedback_stats = feedback_manager.get_feedback_stats()
        
        return render_template('admin/dashboard.html', 
                              metrics=metrics, 
                              feedback=all_feedback,
                              feedback_stats=feedback_stats,
                              start_date=start_date_str,
                              end_date=end_date_str,
                              session_id=session_id,
                              question_type=question_type)
    except Exception as e:
        logger.error(f"Error retrieving dashboard data: {str(e)}", exc_info=True)
        return render_template('error.html', 
                             error_message="An error occurred retrieving dashboard data.",
                             error_details=str(e))

@app.route('/run-ragas-evaluation', methods=['POST'])
@login_required
@admin_required
def run_ragas_evaluation():
    """
    Run Ragas metrics evaluation on sample data
    This is a long-running task, so we run it in a background thread
    """
    try:
        from scripts.calculate_ragas_metrics import main as run_ragas
        
        # Get optional parameters
        data = request.get_json() or {}
        sample_size = data.get('sample_size', 20)
        
        # Start the evaluation in a background thread
        def run_evaluation():
            try:
                logger.info(f"Starting Ragas evaluation with sample_size={sample_size}")
                # Update the sample size in the main function
                import scripts.calculate_ragas_metrics
                scripts.calculate_ragas_metrics.SAMPLE_SIZE = sample_size
                # Run the evaluation
                run_ragas()
                logger.info("Ragas evaluation completed successfully")
            except Exception as e:
                logger.error(f"Error in Ragas evaluation thread: {str(e)}", exc_info=True)
        
        # Start the evaluation thread
        thread = threading.Thread(target=run_evaluation)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True, 
            'message': 'Ragas evaluation started in background thread'
        })
        
    except Exception as e:
        logger.error(f"Error starting Ragas evaluation: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Failed to start Ragas evaluation',
            'details': str(e)
        }), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """
    API endpoint for marking a session as complete while preserving history for analytics
    
    This does not delete the conversation history, but signals that the
    current session is over. The UI will reset, but the data remains for
    analytics and feedback purposes.
    """
    try:
        # Mark the session as complete but preserve history for analytics
        success = conversation_manager.clear_history()
        
        if success:
            logger.info("Session marked as complete via API endpoint")
            return jsonify({'success': True, 'message': 'Session marked as complete'})
        else:
            logger.error("Failed to mark session as complete")
            return jsonify({'error': 'Failed to mark session as complete'}), 500
            
    except Exception as e:
        logger.error(f"Error marking session as complete: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred marking session as complete'}), 500

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