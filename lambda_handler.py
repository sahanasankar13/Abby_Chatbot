import json
import os
import logging
import boto3
import uuid
from datetime import datetime
import base64
import traceback
from urllib.parse import parse_qs

# Initialize DynamoDB resources
dynamodb = boto3.resource('dynamodb')
conversations_table = dynamodb.Table(os.environ.get('conversationsTable', 'abby-chatbot-conversations-dev'))
users_table = dynamodb.Table(os.environ.get('usersTable', 'abby-chatbot-users-dev'))
feedback_table = dynamodb.Table(os.environ.get('feedbackTable', 'abby-chatbot-feedback-dev'))

# Initialize logging
logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# Import app components
from chatbot.conversation_manager import ConversationManager

# Initialize the conversation manager (do it outside the handler for warm Lambda benefits)
conversation_manager = ConversationManager(evaluation_model=os.environ.get('EVALUATION_MODEL', 'local'))

def get_conversation_from_dynamodb(conversation_id):
    """Retrieve conversation history from DynamoDB"""
    try:
        response = conversations_table.get_item(Key={'conversation_id': conversation_id})
        if 'Item' in response:
            return response['Item'].get('history', [])
        return []
    except Exception as e:
        logger.error(f"Error retrieving conversation from DynamoDB: {str(e)}")
        return []

def save_conversation_to_dynamodb(conversation_id, history):
    """Save conversation history to DynamoDB"""
    try:
        conversations_table.put_item(
            Item={
                'conversation_id': conversation_id,
                'history': history,
                'last_updated': datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Error saving conversation to DynamoDB: {str(e)}")

def save_feedback_to_dynamodb(feedback_data):
    """Save user feedback to DynamoDB"""
    try:
        feedback_id = str(uuid.uuid4())
        feedback_table.put_item(
            Item={
                'feedback_id': feedback_id,
                'conversation_id': feedback_data.get('conversation_id', 'unknown'),
                'rating': feedback_data.get('rating'),
                'comments': feedback_data.get('comments', ''),
                'message_id': feedback_data.get('message_id', ''),
                'timestamp': datetime.now().isoformat()
            }
        )
        return {"success": True, "feedback_id": feedback_id}
    except Exception as e:
        logger.error(f"Error saving feedback to DynamoDB: {str(e)}")
        return {"success": False, "error": str(e)}

def handle_chat_request(event):
    """Handle chat API requests"""
    try:
        body = json.loads(event.get('body', '{}'))
        message = body.get('message', '')
        conversation_id = body.get('conversation_id')
        
        # Create a new conversation ID if one doesn't exist
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Get conversation history from DynamoDB
        history = get_conversation_from_dynamodb(conversation_id)
        
        # Process the message
        response_data = conversation_manager.process_message(message, conversation_id)
        
        # If needed, update context with the new history
        if history:
            # Add the current exchange to history
            history.append({
                'sender': 'user',
                'message': message,
                'timestamp': datetime.now().isoformat()
            })
            history.append({
                'sender': 'bot',
                'message': response_data.get('response', ''),
                'timestamp': datetime.now().isoformat(),
                'message_id': response_data.get('message_id', '')
            })
        else:
            # Initialize history with the current exchange
            history = [
                {
                    'sender': 'user',
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'sender': 'bot',
                    'message': response_data.get('response', ''),
                    'timestamp': datetime.now().isoformat(),
                    'message_id': response_data.get('message_id', '')
                }
            ]
        
        # Save updated history to DynamoDB
        save_conversation_to_dynamodb(conversation_id, history)
        
        # Include the conversation ID in the response
        response_data['conversation_id'] = conversation_id
        
        return {
            'statusCode': 200,
            'body': json.dumps(response_data),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            }
        }
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'An error occurred while processing your request.'
            }),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            }
        }

def handle_feedback_request(event):
    """Handle feedback API requests"""
    try:
        body = json.loads(event.get('body', '{}'))
        result = save_feedback_to_dynamodb(body)
        
        return {
            'statusCode': 200 if result.get('success', False) else 500,
            'body': json.dumps(result),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            }
        }
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e)
            }),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            }
        }

def handle_clear_history(event):
    """Handle clear history requests"""
    try:
        body = json.loads(event.get('body', '{}'))
        conversation_id = body.get('conversation_id')
        
        if conversation_id:
            # Clear conversation in DynamoDB by overwriting with empty history
            save_conversation_to_dynamodb(conversation_id, [])
            
            return {
                'statusCode': 200,
                'body': json.dumps({'success': True, 'message': 'Conversation history cleared'}),
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Credentials': True
                }
            }
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'success': False, 'message': 'Conversation ID is required'}),
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Credentials': True
                }
            }
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'success': False, 'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            }
        }

def handle_health_check(event):
    """Handle health check requests"""
    return {
        'statusCode': 200,
        'body': json.dumps({'status': 'healthy', 'timestamp': datetime.now().isoformat()}),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': True
        }
    }

def handle_index(event):
    """Handle root path requests - serves the frontend"""
    # Return a simple HTML page or redirect to a static site hosted elsewhere
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Abby Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
            .container { max-width: 800px; margin: 0 auto; }
            .chat-container { border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-top: 20px; }
            .message { margin-bottom: 10px; }
            .user-message { text-align: right; }
            .bot-message { text-align: left; }
            .message-bubble { display: inline-block; padding: 10px 15px; border-radius: 18px; max-width: 70%; }
            .user-bubble { background-color: #0084ff; color: white; }
            .bot-bubble { background-color: #e9e9eb; color: black; }
            #message-input { width: 75%; padding: 10px; border: 1px solid #ddd; border-radius: 20px; }
            #send-button { width: 20%; padding: 10px; background-color: #0084ff; color: white; border: none; border-radius: 20px; cursor: pointer; }
            .input-container { display: flex; margin-top: 20px; justify-content: space-between; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to Abby Chatbot</h1>
            <p>Ask me anything about reproductive health policies.</p>
            
            <div class="chat-container" id="chat-container"></div>
            
            <div class="input-container">
                <input type="text" id="message-input" placeholder="Type your message...">
                <button id="send-button">Send</button>
            </div>
        </div>

        <script>
            let conversationId = null;
            
            document.getElementById('send-button').addEventListener('click', sendMessage);
            document.getElementById('message-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            function sendMessage() {
                const messageInput = document.getElementById('message-input');
                const message = messageInput.value.trim();
                
                if (message) {
                    // Add user message to chat
                    addMessageToChat('user', message);
                    messageInput.value = '';
                    
                    // Add "thinking" indicator
                    const thinkingElement = document.createElement('div');
                    thinkingElement.className = 'message bot-message';
                    thinkingElement.innerHTML = '<div class="message-bubble bot-bubble">Thinking...</div>';
                    thinkingElement.id = 'thinking-indicator';
                    document.getElementById('chat-container').appendChild(thinkingElement);
                    
                    // Send request to API
                    fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            message: message,
                            conversation_id: conversationId
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        // Remove thinking indicator
                        const thinkingIndicator = document.getElementById('thinking-indicator');
                        if (thinkingIndicator) {
                            thinkingIndicator.remove();
                        }
                        
                        // Save conversation ID
                        conversationId = data.conversation_id;
                        
                        // Add bot response to chat
                        addMessageToChat('bot', data.response);
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        
                        // Remove thinking indicator
                        const thinkingIndicator = document.getElementById('thinking-indicator');
                        if (thinkingIndicator) {
                            thinkingIndicator.remove();
                        }
                        
                        // Add error message
                        addMessageToChat('bot', 'Sorry, there was an error processing your request.');
                    });
                }
            }
            
            function addMessageToChat(sender, message) {
                const chatContainer = document.getElementById('chat-container');
                const messageElement = document.createElement('div');
                
                if (sender === 'user') {
                    messageElement.className = 'message user-message';
                    messageElement.innerHTML = `<div class="message-bubble user-bubble">${message}</div>`;
                } else {
                    messageElement.className = 'message bot-message';
                    messageElement.innerHTML = `<div class="message-bubble bot-bubble">${message}</div>`;
                }
                
                chatContainer.appendChild(messageElement);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    
    return {
        'statusCode': 200,
        'body': html,
        'headers': {
            'Content-Type': 'text/html',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': True
        }
    }

def handler(event, context):
    """Main Lambda handler function"""
    logger.info(f"Received event: {json.dumps(event)}")
    
    # Handle CORS preflight requests
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
                'Access-Control-Allow-Credentials': True
            },
            'body': ''
        }
    
    # Route to appropriate handler based on path and method
    path = event.get('path', '')
    http_method = event.get('httpMethod', '')
    
    if path == '/' and http_method == 'GET':
        return handle_index(event)
    elif path == '/api/chat' and http_method == 'POST':
        return handle_chat_request(event)
    elif path == '/api/feedback' and http_method == 'POST':
        return handle_feedback_request(event)
    elif path == '/api/clear-history' and http_method == 'POST':
        return handle_clear_history(event)
    elif path == '/health' and http_method == 'GET':
        return handle_health_check(event)
    else:
        return {
            'statusCode': 404,
            'body': json.dumps({'error': 'Not Found'}),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            }
        } 