import os
from flask import Flask, jsonify

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        "status": "ok",
        "message": "Abby Chatbot simple test app is running!"
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "environment": os.environ.get("FLASK_ENV", "development")
    })

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 