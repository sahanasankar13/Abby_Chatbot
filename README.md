# Reproductive Health Chatbot

An advanced reproductive health conversational AI platform providing personalized, privacy-conscious support through intelligent interaction technologies.

## Overview

This system provides accurate, context-aware information for reproductive health queries with a focus on US-based policy information, while gracefully handling international queries. It features a simplified user interface, comprehensive performance tracking, and advanced metrics for quality assessment.

## Key Technologies

- **GPT-4 Conversational AI**: Powers natural, empathetic responses
- **BERT-based Retrieval-Augmented Generation (RAG)**: Ensures accurate information retrieval
- **Dynamic Policy Information Integration**: Provides up-to-date US state policy data
- **AWS Cloud Deployment**: Enables scalable, reliable hosting
- **Responsive Web Interface**: Works across devices with privacy-focused design
- **Enhanced Markdown Text Processing**: Improves readability of responses
- **Advanced Metrics Dashboard**: Tracks performance, Ragas evaluation metrics, and user feedback

## Project Structure

### Core Components

- **app.py**: Flask application entry point and route definitions
- **main.py**: Server startup script
- **models.py**: Data models for chat messages and user feedback

### Chatbot Components

- **chatbot/baseline_model.py**: Core model that combines BERT-RAG, GPT-4, and policy integration
- **chatbot/bert_rag.py**: BERT-based Retrieval-Augmented Generation implementation
- **chatbot/citation_manager.py**: Handles citations for information sources
- **chatbot/conversation_manager.py**: Manages conversation flow and context
- **chatbot/friendly_bot.py**: Adds empathetic, friendly elements to responses
- **chatbot/gpt_integration.py**: Integration with OpenAI's GPT models
- **chatbot/policy_api.py**: Integration with abortion policy API
- **chatbot/response_evaluator.py**: Evaluates response quality, safety, and accuracy
- **chatbot/visual_info.py**: Generates visual information graphics

### Utility Components

- **utils/advanced_metrics.py**: Advanced metrics for evaluating chatbot responses
- **utils/data_loader.py**: Loads and processes reproductive health data
- **utils/feedback_manager.py**: Manages user feedback storage and retrieval
- **utils/metrics.py**: Tracks and reports performance metrics
- **utils/metrics_analyzer.py**: Analyzes evaluation logs for dashboard
- **utils/text_processing.py**: Text processing utilities including PII detection

### Frontend Components

- **templates/**: HTML templates for the web interface
  - **templates/index.html**: Main chat interface
  - **templates/admin/dashboard.html**: Admin metrics dashboard
  - **templates/layout.html**: Base layout template
- **static/**: Static assets (CSS, JS, images)
  - **static/css/style.css**: Custom CSS styles
  - **static/js/chat.js**: Client-side chat functionality

## Features

1. **Natural Language Understanding**: Interprets user questions naturally
2. **Personalized Responses**: Tailors information to user context and location
3. **Privacy Protection**: PII detection and redaction
4. **Policy Information**: Up-to-date state policy data via API integration
5. **Response Evaluation**: Quality, accuracy, and safety checks
6. **Performance Metrics**: Comprehensive tracking and reporting
7. **User Feedback**: Thumbs up/down feedback with optional comments
8. **Admin Dashboard**: Metrics visualization and analysis
9. **Visual Information**: SVG-based graphics for reproductive health topics
10. **Session Management**: Type 'end' to end a session and clear history

## Environment Variables

The following environment variables are required:

- `SESSION_SECRET`: Secret key for Flask session management
- `OPENAI_API_KEY`: API key for OpenAI GPT models
- `ABORTION_POLICY_API_KEY`: API key for abortion policy data
- `DATABASE_URL`: (Optional) Database connection string for persistent storage
- `AWS_REGION`: (Optional) AWS region for CloudWatch metrics when deployed

## AWS Deployment

### Prerequisites

1. An AWS account with appropriate permissions
2. AWS CLI installed and configured
3. Docker installed (for container deployment)

### Deployment Options

#### Option 1: AWS Elastic Beanstalk (Recommended)

1. Package the application:
   ```bash
   zip -r app.zip . -x "*.git*" "*.env" "*.venv*" "__pycache__/*"
   ```

2. Create an Elastic Beanstalk environment:
   - Platform: Python
   - Application code: Upload the zip file
   - Configure environment properties with the required environment variables

3. Deploy:
   ```bash
   eb deploy
   ```

#### Option 2: AWS ECS (Container-based)

1. Build the Docker image:
   ```bash
   docker build -t reproductive-health-chatbot .
   ```

2. Tag and push to Amazon ECR:
   ```bash
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
   docker tag reproductive-health-chatbot:latest <account-id>.dkr.ecr.<region>.amazonaws.com/reproductive-health-chatbot:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/reproductive-health-chatbot:latest
   ```

3. Create an ECS task definition and service:
   - Use the CloudFormation template provided in `cloudformation-template.yaml`
   - Configure environment variables in the task definition

#### Option 3: AWS EC2

1. Launch an EC2 instance with Amazon Linux 2
2. Install dependencies:
   ```bash
   sudo yum update -y
   sudo yum install -y python3 python3-pip git
   ```

3. Clone the repository:
   ```bash
   git clone <repository-url>
   cd reproductive-health-chatbot
   ```

4. Install Python dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

5. Set up environment variables:
   ```bash
   echo "export OPENAI_API_KEY='your-api-key'" >> ~/.bashrc
   echo "export ABORTION_POLICY_API_KEY='your-api-key'" >> ~/.bashrc
   echo "export SESSION_SECRET='your-secret-key'" >> ~/.bashrc
   source ~/.bashrc
   ```

6. Run with Gunicorn:
   ```bash
   gunicorn -c gunicorn.conf.py main:app
   ```

7. Set up Nginx as a reverse proxy (optional but recommended)

### CloudWatch Integration

This application supports sending metrics to AWS CloudWatch when deployed. To enable:

1. Ensure the `AWS_REGION` environment variable is set
2. Attach an IAM role with CloudWatch permissions to your deployment
3. Metrics will automatically be sent to the namespace `ReproductiveHealthChatbot`

## User Interface

### Chat Interface

- Simple, clean interface for asking questions
- Type 'end' to end a session and clear history
- Feedback options on each bot response (except welcome message)
- Direct link to admin dashboard

### Admin Dashboard

- Comprehensive metrics visualization with multiple tabs
- Ragas metrics evaluation with configurable sample size
- Filtering by date range, session ID, and question type
- User feedback analysis and performance metrics tracking
- Advanced metrics visualization (BLEU, ROUGE, BERTScore, faithfulness)

## Performance Metrics

The system tracks the following metrics:

- **Response Quality**: Relevance, accuracy, and completeness
- **Safety**: Content safety and guideline adherence
- **User Feedback**: Thumbs up/down rates and comments
- **System Performance**: Inference time, token counts, memory usage
- **Advanced NLP Metrics**: BLEU, ROUGE, BERTScore, and Ragas (faithfulness, context precision, context recall)

## License

This project is confidential. All rights reserved.

## Contributors

This project was developed by a dedicated team of professionals committed to advancing reproductive health education and support.