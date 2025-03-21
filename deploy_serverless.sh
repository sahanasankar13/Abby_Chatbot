#!/bin/bash

# Abort on error
set -e

# Default values
STAGE="dev"
REGION="us-east-1"
SKIP_MODELS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --stage STAGE      Deployment stage (default: dev)"
            echo "  --region REGION    AWS region (default: us-east-1)"
            echo "  --skip-models      Skip uploading models to S3"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Deploying Abby Chatbot to AWS Lambda (Stage: $STAGE, Region: $REGION)"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed. Please install Node.js and npm."
    exit 1
fi

# Check if serverless is installed
if ! command -v serverless &> /dev/null; then
    echo "Serverless Framework not found. Installing..."
    npm install -g serverless
fi

# Install dependencies
echo "Installing dependencies..."
npm install

# Create deployment package
echo "Creating serverless deployment package..."

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements-serverless.txt

# Upload ML models to S3 if needed
if [ "$SKIP_MODELS" = false ]; then
    BUCKET_NAME="abby-chatbot-data-$STAGE"
    
    # Check if bucket exists, create if it doesn't
    if ! aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
        echo "Creating S3 bucket: $BUCKET_NAME"
        aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION"
    fi
    
    echo "Uploading ML models and NLTK data to S3..."
    python upload_models_to_s3.py --bucket "$BUCKET_NAME" --stage "$STAGE" --download-nltk
else
    echo "Skipping model upload (--skip-models flag is set)"
fi

# Deploy with Serverless Framework
echo "Deploying with Serverless Framework..."
serverless deploy --stage "$STAGE" --region "$REGION" --verbose

# Deactivate virtual environment
deactivate

echo "Deployment completed!"
echo "Note: The first few requests might be slow due to cold starts." 