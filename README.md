# Abby Health Chatbot

An advanced reproductive health conversational AI platform providing personalized, privacy-conscious support through intelligent interaction technologies.

## Overview

This system provides accurate, context-aware information for reproductive health queries with a focus on US-based policy information, while gracefully handling international queries. It features a clean user interface, comprehensive citation handling, and multi-aspect query processing.

## Key Features

- **Multi-Aspect Query Processing**: Handles knowledge, emotional support, and policy aspects simultaneously
- **Citation Management**: Provides properly formatted citations with specific URLs
- **Clean, Accessible UI**: Responsive design with accessibility features
- **Suggestion Prompts**: Helps users get started with common health questions
- **Preprocessing Layer**: Input validation, language detection, PII redaction, and typo correction

## Quick Setup and Running Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for version control)

### Setup Steps

1. **Extract the ZIP file or clone the repository**
   ```bash
   unzip Abby_Chatbot_20250321.zip
   cd Abby_Chatbot
   ```
   or
   ```bash
   git clone <repository-url>
   cd Abby_Chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**

   **On MacOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

   **On Windows:**
   ```bash
   .venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**
   Create a `.env` file in the root directory (or copy from `.env.example`):
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   POLICY_API_BASE_URL=https://api.abortionpolicyapi.com/v1/
   ```

### Running the Chatbot

1. **Run the web interface**
   ```bash
   uvicorn app:app --reload
   ```
   Then open your browser to http://127.0.0.1:8000

2. **Run the demo preprocessor script**
   ```bash
   python demo_preprocessor.py
   ```

3. **Run tests**
   ```bash
   python test_preprocessor.py
   python test_integration.py
   ```

### Troubleshooting

- If you encounter missing models, run: `python -m spacy download en_core_web_sm`
- For ZIP code functionality issues, ensure `zipcodes` library is installed
- For typo correction, verify `data/frequency_dictionary_en_82_765.txt` exists
- If you see a "port already in use" error, try a different port: `uvicorn app:app --reload --port 8081`

## Detailed Installation and Setup

### Prerequisites
- Python 3.9+ installed
- Git
- OpenAI API key

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Abby_Chatbot
   ```

2. **Create and activate a virtual environment**
   ```bash
   # For macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   
   # For Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and add your OpenAI API key and other required credentials.

5. **Run the application**
   ```bash
   # If port 8080 is already in use, try a different port:
   uvicorn app:app --reload --port 8081
   ```

6. **Access the application**
   Open your browser and navigate to: `http://localhost:8081`

### Troubleshooting

- **Port already in use error**
  If you see an error like `[Errno 48] Address already in use`, try:
  ```bash
  # Kill any running uvicorn processes
  pkill -f "uvicorn"
  # Try starting the app with a different port
  uvicorn app:app --reload --port 8082
  ```

- **Python package errors**
  If you encounter any package errors, try:
  ```bash
  pip install --upgrade -r requirements.txt
  ```

## Project Structure and File Interactions

### Core Application Files

- **app.py**: FastAPI application entry point that:
  - Sets up routes and API endpoints
  - Initializes the multi-aspect query processor
  - Handles user requests and serves the web interface
  - Coordinates conversation flow and memory management
  - Provides test endpoints for citation handling and responses

- **setup.py**: Package installation configuration for deploying the application

- **Procfile**: Configuration for web server deployment 

- **requirements.txt**: Core dependencies needed for the application
  
- **requirements-minimal.txt**: Minimal set of dependencies for lightweight deployment

### Chatbot Architecture (in `chatbot/` directory)

- **__init__.py**: Exports the main chatbot components and defines the package version

- **multi_aspect_processor.py**: The main orchestrator that:
  - Manages the workflow from query to response
  - Coordinates all specialized handlers
  - Combines multiple aspects into cohesive responses
  - Entry point for all user queries

- **memory_manager.py**: Manages conversation history and context by:
  - Storing conversation history
  - Retrieving relevant context for new queries
  - Maintaining user session information

- **unified_classifier.py**: Determines the nature of user queries by:
  - Classifying whether a query needs knowledge, emotional support, or policy information
  - Routing queries to appropriate specialized handlers

- **aspect_decomposer.py**: Breaks down complex queries into multiple aspects by:
  - Identifying different dimensions of a query
  - Creating specialized sub-queries for each aspect

- **knowledge_handler.py**: Provides factual health information by:
  - Retrieving relevant medical and health information
  - Ensuring accuracy through citation management
  - Using BertRAG for information retrieval

- **emotional_support_handler.py**: Provides empathetic support by:
  - Generating compassionate and understanding responses
  - Addressing emotional aspects of reproductive health concerns

- **policy_handler.py**: Provides legal and policy information by:
  - Retrieving up-to-date policy information by US state
  - Presenting policy in an understandable format
  - Adding appropriate disclaimers

- **response_composer.py**: Combines specialized responses by:
  - Integrating multiple aspect outputs
  - Ensuring coherent, well-structured final responses
  - Formatting responses with appropriate markdown

- **citation_manager.py**: Manages citation handling by:
  - Tracking sources of information
  - Formatting citations consistently
  - Connecting citations to specific parts of responses

- **bert_rag.py**: Implements retrieval-augmented generation by:
  - Retrieving relevant documents based on user queries
  - Enhancing responses with factual information
  - Supporting citation tracking for retrieved information

- **config.py**: Central configuration management for the chatbot

### Frontend Components

- **templates/index.html**: Main chat interface that:
  - Provides the user input area
  - Displays chat messages
  - Shows suggestion prompts
  - Renders feedback buttons

- **templates/layout.html**: Base HTML template that:
  - Sets up the page structure
  - Includes required CSS and JavaScript
  - Configures responsive design elements

- **static/css/style.css**: Styling for the chat interface, including:
  - Chat message formatting
  - Color schemes and visual design
  - Responsive layout adjustments
  - Accessibility features

- **static/js/chat.js**: Client-side JavaScript that:
  - Handles user input submission
  - Processes and displays bot responses
  - Renders citation formatting and linking
  - Manages UI interactions and animations
  - Implements suggestion prompts functionality
  - Handles feedback submission

- **static/js/quick-exit.js**: Safety feature allowing users to quickly exit the page

- **static/images/**: Contains images used in the interface

### Data Components

- **data/**: Contains conversation logs and other data files
  - Conversation histories are stored in JSON format
  - Knowledge documents used for retrieval are organized here

## System Interaction Flow

1. User sends a query through the web interface (`static/js/chat.js` → `app.py`)
2. The query is processed by the multi-aspect processor (`multi_aspect_processor.py`)
3. The unified classifier determines query type (`unified_classifier.py`)
4. The query is decomposed into aspects if needed (`aspect_decomposer.py`)
5. Specialized handlers process relevant aspects:
   - Knowledge aspect (`knowledge_handler.py` → `bert_rag.py`)
   - Emotional aspect (`emotional_support_handler.py`)
   - Policy aspect (`policy_handler.py`)
6. Citations are collected and formatted (`citation_manager.py`)
7. All aspects are combined into a coherent response (`response_composer.py`)
8. The response is sent back to the user interface (`app.py` → `static/js/chat.js`)
9. Conversation history is updated (`memory_manager.py`)
10. The UI displays the response with proper formatting (`static/js/chat.js`)

## Running the Application

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables or create a `.env` file
3. Run the application: `uvicorn app:app --reload`
4. Access the chatbot at `http://localhost:8000`

## Setting Up the Environment

To run this application, you'll need:

1. Python 3.9+ installed
2. Required environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key for GPT access
   - `POLICY_API_KEY`: API key for abortion policy information (optional)

Create a `.env` file in the project root with these variables:

```
OPENAI_API_KEY=your_openai_api_key_here
POLICY_API_BASE_URL=https://api.abortionpolicyapi.com/v1/
POLICY_API_KEY=your_policy_api_key_here
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Abby Chatbot Deployment

This repository contains the deployment configuration for the Abby Chatbot, a FastAPI application served by Gunicorn.

## Prerequisites

1. AWS CLI installed and configured with appropriate credentials
2. Terraform installed
3. Python 3.8 or higher
4. An SSH key pair in AWS (for EC2 instance access)

## Directory Structure

```
.
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── appspec.yml           # CodeDeploy configuration
├── scripts/              # Deployment scripts
│   ├── before_install.sh
│   ├── after_install.sh
│   ├── start_application.sh
│   ├── stop_application.sh
│   └── validate_service.sh
└── terraform/            # Terraform configuration
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    └── deploy.sh
```

## Deployment Steps

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd abby-chatbot
   ```

2. Create an SSH key pair in AWS if you haven't already:
   ```bash
   aws ec2 create-key-pair --key-name abby-key --query 'KeyMaterial' --output text > abby-key.pem
   chmod 400 abby-key.pem
   ```

3. Initialize and apply Terraform configuration:
   ```bash
   cd terraform
   terraform init
   terraform apply
   ```

4. Deploy the application:
   ```bash
   ./deploy.sh
   ```

## Infrastructure Components

- VPC with public subnet
- Internet Gateway
- Application Load Balancer
- EC2 instance (t2.micro) running Amazon Linux 2
- CodeDeploy application and deployment group
- S3 bucket for deployment artifacts
- IAM roles and policies
- Security groups

## Application Components

- FastAPI application running on port 5006
- Gunicorn WSGI server
- Systemd service for process management
- Health check endpoint at `/health`

## Monitoring

- Application logs: `/var/log/abby-chatbot/`
- CodeDeploy logs: `/var/log/aws/codedeploy-agent/`
- System logs: `journalctl -u abby-chatbot`

## Cleanup

To remove all AWS resources:

1. Run the cleanup script:
   ```bash
   ./cleanup.sh
   ```

2. Destroy Terraform resources:
   ```bash
   cd terraform
   terraform destroy
   ```

## Security Notes

- The EC2 instance accepts SSH connections from any IP (0.0.0.0/0). In production, restrict this to your IP range.
- The application uses dummy values for sensitive environment variables. In production, use AWS Secrets Manager or SSM Parameter Store.
- SSL/TLS termination is not configured. In production, add HTTPS support to the ALB.

## Troubleshooting

1. Check if the service is running:
   ```bash
   systemctl status abby-chatbot
   ```

2. View application logs:
   ```bash
   journalctl -u abby-chatbot -f
   ```

3. Test the health endpoint:
   ```bash
   curl http://localhost:5006/health
   ```

4. Check CodeDeploy deployment status:
   ```bash
   aws deploy get-deployment --deployment-id <deployment-id>
   ```