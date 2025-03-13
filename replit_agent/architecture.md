# Architecture Overview

## 1. Overview

The Reproductive Health Chatbot is an AI-powered conversational platform designed to provide accurate, context-aware information on reproductive health topics with a focus on US-based policy information. The system combines several AI technologies to deliver personalized, privacy-conscious support through intelligent interaction.

The architecture employs a hybrid approach combining:
- BERT-based Retrieval-Augmented Generation (RAG) for accurate information retrieval
- OpenAI's GPT models for natural, empathetic conversational responses
- Integration with external policy data APIs for up-to-date information
- Advanced evaluation metrics for quality assessment
- Privacy-focused design with PII detection

The application is built as a web service using Flask, with deployment options for AWS cloud infrastructure.

## 2. System Architecture

The application follows a modular architecture with the following high-level components:

### 2.1 Frontend Layer

- **Web Interface**: A responsive web interface built with HTML, CSS, and JavaScript
- **Admin Dashboard**: Secure admin interface for monitoring metrics and feedback

### 2.2 Application Layer

- **Flask Web Server**: Handles HTTP requests, routes, and serves the web interface
- **Conversation Management**: Orchestrates the conversation flow and context preservation
- **Authentication System**: Login-based access control for admin features

### 2.3 Domain Layer

- **Baseline Model**: Core intelligence that combines RAG, GPT, and policy integration
- **BERT-RAG System**: Knowledge retrieval from curated reproductive health information
- **Policy Integration**: Real-time policy information from external API
- **Friendly Response Generation**: Adds empathetic elements to technical responses

### 2.4 Evaluation Layer

- **Response Evaluator**: Assesses response quality, safety, and accuracy
- **Metrics Tracking**: Records performance and usage statistics
- **Feedback Management**: Collects and analyzes user feedback

### 2.5 Data Layer

- **Reproductive Health Knowledge Base**: Curated Q&A pairs on reproductive health topics
- **User Feedback Storage**: JSON-based storage for user ratings and comments
- **Metrics and Logs**: Performance data and evaluation metrics

## 3. Key Components

### 3.1 Chatbot Components

#### 3.1.1 Conversation Manager (`chatbot/conversation_manager.py`)
- **Purpose**: Manages the overall conversation flow and context
- **Key Functions**:
  - Tracks conversation history for context-aware responses
  - Detects location context for policy-related questions
  - Combines emotional support with factual information
  - Manages PII detection and privacy protection

#### 3.1.2 Baseline Model (`chatbot/baseline_model.py`)
- **Purpose**: Core intelligence orchestrating different response modes
- **Key Functions**:
  - Categorizes questions into policy, knowledge, and conversational types
  - Coordinates between RAG-based knowledge, policy API data, and empathetic responses
  - Handles multi-query questions by splitting and processing each part

#### 3.1.3 BERT-RAG Model (`chatbot/bert_rag.py`)
- **Purpose**: Implements Retrieval-Augmented Generation using BERT embeddings
- **Key Functions**:
  - Performs semantic search on reproductive health Q&A database
  - Features advanced relevance scoring and confidence assessment
  - Provides natural language response formatting

#### 3.1.4 GPT Integration (`chatbot/gpt_integration.py`)
- **Purpose**: Integrates with OpenAI's GPT models
- **Key Functions**:
  - Enhances response quality for complex or ambiguous questions
  - Provides fallback for questions outside the RAG model's knowledge
  - Formats policy data into natural, user-friendly language

#### 3.1.5 Policy API (`chatbot/policy_api.py`)
- **Purpose**: Integrates with external abortion policy API
- **Key Functions**:
  - Fetches up-to-date state-specific policy information
  - Extracts state context from user queries
  - Formats policy data for user-friendly presentation

#### 3.1.6 Response Evaluator (`chatbot/response_evaluator.py`)
- **Purpose**: Evaluates chatbot responses for quality, accuracy, and safety
- **Key Functions**:
  - Uses both OpenAI's GPT and local transformer models for evaluation
  - Detects and flags sensitive topics
  - Ensures information comes from approved sources

#### 3.1.7 Friendly Bot (`chatbot/friendly_bot.py`)
- **Purpose**: Adds empathetic elements to technical responses
- **Key Functions**:
  - Improves response structure and readability
  - Adds appropriate emotional tone based on question type
  - Ensures responses are both accurate and supportive

### 3.2 Utility Components

#### 3.2.1 Advanced Metrics (`utils/advanced_metrics.py`)
- **Purpose**: Calculates advanced evaluation metrics
- **Key Functions**:
  - Computes BLEU, ROUGE, BERTScore for text similarity
  - Calculates RAG-specific metrics (faithfulness, context precision)
  - Generates performance reports

#### 3.2.2 Data Loader (`utils/data_loader.py`)
- **Purpose**: Loads and processes reproductive health data
- **Key Functions**:
  - Loads Q&A pairs from CSV sources
  - Provides fallback data when primary sources unavailable
  - Preprocesses data for RAG system

#### 3.2.3 Feedback Manager (`utils/feedback_manager.py`)
- **Purpose**: Manages user feedback storage and retrieval
- **Key Functions**:
  - Stores user ratings and comments in JSON format
  - Provides aggregated feedback statistics
  - Supports feedback analysis for quality improvement

#### 3.2.4 Metrics Tracker (`utils/metrics.py`)
- **Purpose**: Tracks and reports performance metrics
- **Key Functions**:
  - Records API calls, response times, and token usage
  - Stores metrics locally with optional AWS CloudWatch integration
  - Supports both development and production monitoring

#### 3.2.5 Text Processing (`utils/text_processing.py`)
- **Purpose**: Provides text processing utilities
- **Key Functions**:
  - Detects and handles PII (Personally Identifiable Information)
  - Cleans and normalizes text
  - Detects language and extracts keywords

### 3.3 Web Application Components

#### 3.3.1 Flask Application (`app.py`)
- **Purpose**: Main web application entry point
- **Key Functions**:
  - Defines routes and request handlers
  - Manages user authentication for admin features
  - Initializes conversation manager and other components

#### 3.3.2 Admin Interface
- **Purpose**: Secure dashboard for monitoring and administration
- **Key Functions**:
  - Displays performance metrics and user feedback
  - Requires authentication with admin credentials
  - Provides data filtering and analysis tools

## 4. Data Flow

### 4.1 User Interaction Flow

1. **User Inquiry**:
   - User submits a question through the web interface
   - Frontend sends the question to the server via AJAX

2. **Request Processing**:
   - Flask application receives the request
   - Conversation Manager processes the inquiry
   - PII detection removes any sensitive personal information

3. **Question Analysis**:
   - Baseline Model categorizes the question (policy, knowledge, conversational)
   - Location extraction identifies any state-specific context

4. **Answer Generation**:
   - For knowledge questions: BERT-RAG retrieves relevant information
   - For policy questions: Policy API fetches state-specific data
   - For conversational questions: GPT Model generates a response
   - Friendly Bot enhances response with empathetic elements

5. **Response Evaluation**:
   - Response Evaluator checks quality, accuracy, and safety
   - Citations are added for information sources
   - Metrics are recorded for performance analysis

6. **User Response**:
   - Processed answer is returned to the frontend
   - User can provide feedback on the response quality

### 4.2 Admin Dashboard Flow

1. **Authentication**:
   - Admin users login with credentials
   - Flask-Login verifies authentication
   - Admin-specific decorators enforce authorization

2. **Metrics Display**:
   - Performance metrics are loaded from storage
   - Advanced metrics calculator processes evaluation logs
   - Data is presented in dashboard format

3. **Feedback Analysis**:
   - User feedback is loaded and aggregated
   - Statistics are calculated for different time periods
   - Detailed feedback is available for review

## 5. External Dependencies

### 5.1 Third-Party APIs

- **OpenAI API**: Used for GPT-4 integration and response evaluation
  - Environment Variable: `OPENAI_API_KEY`

- **Abortion Policy API**: Provides state-specific policy information
  - Environment Variable: `POLICY_API_KEY` or `ABORTION_POLICY_API_KEY`

### 5.2 Key Libraries

- **Flask**: Web framework for the application
- **Flask-Login**: Authentication management
- **BERT/Transformers**: NLP models for RAG implementation
- **FAISS**: Vector similarity search for BERT-RAG
- **NLTK**: Natural language processing utilities
- **Torch**: Deep learning framework for models
- **Requests**: HTTP client for API calls
- **Gunicorn**: WSGI HTTP server for production deployment

### 5.3 Evaluation Metrics Libraries

- **RAGAS**: For RAG-specific evaluation metrics
- **BERT-Score**: For semantic similarity evaluation
- **ROUGE-Score**: For text summary evaluation
- **SacreBLEU**: For BLEU score calculation

## 6. Deployment Strategy

The application supports multiple deployment strategies, with a focus on AWS deployment:

### 6.1 AWS Deployment Options

#### 6.1.1 Docker Deployment
- Uses Docker container with Python 3.11 base image
- Configurable through environment variables
- Supports deployment to ECS, EKS, or standalone EC2

#### 6.1.2 Elastic Beanstalk Deployment
- Simplified deployment and management
- Uses Procfile and .ebignore configuration
- Automatic scaling and load balancing

#### 6.1.3 Manual EC2 Deployment
- More control over infrastructure
- Systemd service for application management
- CloudFormation template available for infrastructure as code

### 6.2 Environment Configurations

- **Development**: Local development with debug mode
- **Staging**: Testing environment with live API connections
- **Production**: Optimized for performance and reliability

### 6.3 Deployment Considerations

- **Security**: Proper handling of API keys via environment variables
- **Scaling**: Configured for auto-scaling based on load
- **Monitoring**: CloudWatch integration for metrics in AWS
- **Reliability**: Health checks and auto-recovery configuration
- **Cost Optimization**: Student-friendly deployment options available

### 6.4 CI/CD Strategy

- AWS CodeDeploy scripts for automated deployment
- Application lifecycle hooks in `scripts/` directory
- Support for blue-green deployment strategy