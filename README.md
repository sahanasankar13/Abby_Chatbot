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
3. Docker installed (for container deployment options)

### School Project Deployment (Budget-Friendly)

For school projects with a limited budget (e.g. $250 or less), we recommend using a simplified AWS Elastic Beanstalk setup:

1. See the **[aws-simplified-deployment-guide.md](aws-simplified-deployment-guide.md)** for complete step-by-step instructions designed for AWS beginners.

2. Key cost-saving tips:
   - Use t2.micro instances (free tier eligible)
   - Deploy a single instance without a load balancer
   - Turn off the environment when not in use
   - Set up billing alerts to avoid unexpected charges
   - Delete all resources when the project is complete

3. Quick deployment commands:
   ```bash
   # Install EB CLI
   pip install awsebcli
   
   # Initialize application
   eb init -p python-3.11 reproductive-health-chatbot
   
   # Create a low-cost environment
   eb create reproductive-health-chatbot-env --instance-type t2.micro --single
   
   # Set environment variables
   eb setenv OPENAI_API_KEY=your-key ABORTION_POLICY_API_KEY=your-key SESSION_SECRET=your-secret
   
   # Open your application
   eb open
   ```

### Production Deployment Options

For production deployments with higher requirements for scalability and reliability:

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

# Abby Chatbot Metrics Documentation

This document provides detailed information about the various metrics used to evaluate the performance of the Abby chatbot.

## Cost Metrics

### OpenAI API Costs

The system tracks costs for both GPT-4 and GPT-4 mini models:

#### GPT-4 Pricing (per 1M tokens)
- **Input**: $2.50
- **Output**: $10.00
- **Cached Input**: $1.25

#### GPT-4 Mini Pricing (per 1M tokens)
- **Input**: $0.150
- **Output**: $0.600
- **Cached Input**: $0.075

### Cost Tracking
For each model, the following metrics are tracked:
- **Input Tokens**: Total number of input tokens used
- **Output Tokens**: Total number of output tokens generated
- **Cached Input Tokens**: Total number of cached input tokens
- **Input Cost**: Cost of input tokens
- **Output Cost**: Cost of output tokens
- **Cached Input Cost**: Cost of cached input tokens
- **Total Cost**: Total cost for the model

### Cost Calculation
Costs are automatically calculated based on:
1. Token usage per request
2. Model type (GPT-4 or GPT-4 mini)
3. Token type (input, output, or cached input)
4. Current pricing rates

### Cost Optimization
The system provides insights for cost optimization:
- Token usage patterns
- Cost breakdown by model
- Daily/weekly/monthly cost trends
- Cost per conversation
- Cost per response type

## Core Metrics

### Basic Evaluation Metrics
- **Total Evaluations**: Total number of evaluations performed
- **Average Scores**: Mean scores across different dimensions
  - Relevance: How well the response matches the query
  - Quality: Overall response quality
  - Safety: Safety compliance score
  - Empathy: Emotional intelligence and supportiveness
  - Clarity: Response clarity and understandability
- **Improvement Rate**: Percentage of responses that showed improvement over time

## Advanced Text Similarity Metrics

### ROUGE Metrics
- **ROUGE-1**: Measures overlap of unigrams between reference and generated text
- **ROUGE-2**: Measures overlap of bigrams between reference and generated text
- **ROUGE-L**: Measures longest common subsequence between reference and generated text

### BLEU Score
- **Score**: Bilingual Evaluation Understudy score (0-100)
- **Details**:
  - Precisions: N-gram precisions (1-4)
  - BP: Brevity penalty
  - Ratio: Length ratio between system and reference
  - System/Reference Length: Token counts

### BERTScore
- **Precision**: How well the generated text matches the reference
- **Recall**: How much of the reference is covered by the generated text
- **F1**: Harmonic mean of precision and recall

## RAG (Retrieval-Augmented Generation) Metrics

### RAGAS Metrics
- **Faithfulness**: How well the response is grounded in the retrieved context
- **Context Precision**: How relevant the retrieved context is
- **Context Recall**: How much of the relevant context was retrieved

### Retrieval Metrics
- **Precision@K**: Precision at different retrieval depths (1, 3, 5, 10)
- **Recall@K**: Recall at different retrieval depths (1, 3, 5, 10)
- **MRR**: Mean Reciprocal Rank of relevant documents

## Performance Metrics

### Response Time
- **Average (ms)**: Mean response generation time
- **Min (ms)**: Fastest response time
- **Max (ms)**: Slowest response time

### Resource Usage
- **Tokens**:
  - Average: Mean token count per response
  - Min: Minimum tokens used
  - Max: Maximum tokens used
- **Memory**:
  - Average (MB): Mean memory usage
  - Min (MB): Minimum memory usage
  - Max (MB): Maximum memory usage

## Conversation Metrics

### Session Statistics
- **Total Conversations**: Number of unique chat sessions
- **Average Messages per Conversation**: Mean messages per session
- **Total Messages**: Total number of messages across all sessions

## Daily Metrics

### Aggregated Daily Scores
- **Relevance**: Daily average relevance score
- **Quality**: Daily average quality score
- **Safety**: Daily average safety score
- **Total Evaluations**: Number of evaluations per day

## Chart Metrics

### Performance Trends
- **Dates**: Chronological list of evaluation dates
- **Daily Scores**: Quality scores over time
- **Daily Safety**: Safety scores over time

### Average Scores
- **Relevance**: Overall relevance score
- **Accuracy**: Overall accuracy score
- **Completeness**: Overall completeness score
- **Clarity**: Overall clarity score
- **Empathy**: Overall empathy score

## Calculation Methods

### Text Similarity Metrics
1. **ROUGE**: Uses the `rouge` library to calculate n-gram overlaps
2. **BLEU**: Uses NLTK's BLEU implementation with standard parameters
3. **BERTScore**: Uses the `bert-score` library with the default BERT model

### RAG Metrics
1. **RAGAS**: Uses the `ragas` library to evaluate:
   - Faithfulness through entailment checking
   - Context precision through relevance scoring
   - Context recall through coverage analysis
2. **Retrieval Metrics**: Calculated using standard IR metrics on retrieved documents

### Performance Metrics
1. **Response Time**: Measured using Python's `time` module
2. **Token Usage**: Counted using the tokenizer from the language model
3. **Memory Usage**: Measured using `psutil` for process memory tracking

### Core and Conversation Metrics
1. **Basic Stats**: Calculated using standard statistical methods (mean, min, max)
2. **Session Analysis**: Grouped by session ID and analyzed for patterns
3. **Daily Aggregation**: Grouped by date and averaged for trend analysis

### Cost Calculation
1. **Token Counting**: Tracks input, output, and cached input tokens per request
2. **Model Identification**: Identifies the model used (GPT-4 or GPT-4 mini)
3. **Rate Application**: Applies appropriate rates based on token type and model
4. **Cost Aggregation**: Sums costs across all requests and models

## Usage

The metrics are automatically calculated when:
1. The chatbot processes a new conversation
2. A user submits feedback
3. The dashboard is loaded
4. A date range filter is applied

Metrics can be accessed through the dashboard API endpoints:
- `/api/metrics`: Get all metrics
- `/api/metrics/filtered`: Get metrics filtered by date range or session
- `/api/metrics/daily`: Get daily aggregated metrics
- `/api/metrics/performance`: Get performance-specific metrics
- `/api/metrics/costs`: Get cost breakdown and analysis

## Dependencies

Required Python packages:
- `rouge`: For ROUGE metrics
- `nltk`: For BLEU score calculation
- `bert-score`: For BERTScore calculation
- `ragas`: For RAG evaluation metrics
- `psutil`: For system resource monitoring
- `numpy`: For statistical calculations
- `pandas`: For data manipulation and analysis