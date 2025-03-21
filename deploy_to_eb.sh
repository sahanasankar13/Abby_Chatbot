#!/bin/bash

# Abort on error
set -e

# Default values
REGION="us-east-1"
APP_NAME="abby-chatbot"
ENV_NAME="abby-chatbot-env"
PLATFORM="Python 3.9"
SKIP_CREATE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --region)
            REGION="$2"
            shift 2
            ;;
        --app-name)
            APP_NAME="$2"
            shift 2
            ;;
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --skip-create)
            SKIP_CREATE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --region REGION     AWS region (default: us-east-1)"
            echo "  --app-name APP_NAME Application name (default: abby-chatbot)"
            echo "  --env-name ENV_NAME Environment name (default: abby-chatbot-env)"
            echo "  --skip-create       Skip environment creation (use for updating existing environments)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Deploying Abby Chatbot to Elastic Beanstalk (Region: $REGION)"

# Check if aws CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed. Please install the AWS CLI."
    exit 1
fi

# Install NLTK data
echo "Downloading NLTK data..."
python download_nltk_data.py

# Ensure Elastic Beanstalk .ebextensions directory exists
if [ ! -d ".ebextensions" ]; then
    echo "Creating .ebextensions directory..."
    mkdir -p .ebextensions
fi

# Create configuration file for Python settings
cat > .ebextensions/01_python.config << EOF
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: application.py
  aws:elasticbeanstalk:application:environment:
    PYTHONPATH: "/var/app/current"
  aws:elasticbeanstalk:environment:proxy:staticfiles:
    /static: static
EOF

# Create configuration for nginx
cat > .ebextensions/02_nginx.config << EOF
files:
  "/etc/nginx/conf.d/proxy.conf":
    mode: "000644"
    owner: root
    group: root
    content: |
      client_max_body_size 20M;
      proxy_connect_timeout 300;
      proxy_read_timeout 300;
EOF

# Create an 'application.py' file to serve as the WSGI entry point
if [ ! -f "application.py" ]; then
    echo "Creating application.py entry point..."
    cat > application.py << EOF
from app import app as application

if __name__ == "__main__":
    application.run()
EOF
fi

# Create deployment package
echo "Creating deployment package..."
ZIP_FILE="${APP_NAME}.zip"
ZIP_FILES=("app.py" "models.py" "application.py" ".ebextensions" "templates" "static" "chatbot" "utils" "requirements.txt" ".env" "nltk_data" "data" "config")

# Add models directory if it exists
if [ -d "models" ]; then
    ZIP_FILES+=("models")
fi

zip -r "$ZIP_FILE" "${ZIP_FILES[@]}" -x "**/__pycache__/*" "**/.DS_Store" "**/.git/*"

# Check if EB application exists, create if it doesn't
if ! aws elasticbeanstalk describe-applications --region "$REGION" --application-names "$APP_NAME" &> /dev/null; then
    echo "Creating Elastic Beanstalk application: $APP_NAME"
    aws elasticbeanstalk create-application --region "$REGION" --application-name "$APP_NAME"
fi

if [ "$SKIP_CREATE" = true ]; then
    echo "Skipping environment creation (--skip-create flag is set)"
    echo "Using EB CLI to deploy the application..."
    eb deploy
else
    # Check if environment exists
    ENV_EXISTS=$(aws elasticbeanstalk describe-environments --region "$REGION" --application-name "$APP_NAME" --environment-names "$ENV_NAME" --query "Environments[0].Status" --output text 2>/dev/null || echo "")

    if [ "$ENV_EXISTS" = "Ready" ] || [ "$ENV_EXISTS" = "Updating" ]; then
        # Environment exists, update it
        echo "Updating existing environment: $ENV_NAME"
        aws elasticbeanstalk update-environment --region "$REGION" --application-name "$APP_NAME" --environment-name "$ENV_NAME" --version-label "${APP_NAME}-$(date +%Y%m%d%H%M%S)"
        aws elasticbeanstalk update-application-version --region "$REGION" --application-name "$APP_NAME" --version-label "${APP_NAME}-$(date +%Y%m%d%H%M%S)" --source-bundle S3Bucket="${APP_NAME}-deploy",S3Key="$ZIP_FILE"
    else
        # Environment doesn't exist, create it
        echo "Creating new environment: $ENV_NAME"
        # First upload the zip file to S3
        S3_BUCKET="${APP_NAME}-deploy"
        
        # Create S3 bucket if it doesn't exist
        if ! aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
            echo "Creating S3 bucket: $S3_BUCKET"
            aws s3api create-bucket --bucket "$S3_BUCKET" --region "$REGION" --create-bucket-configuration LocationConstraint="$REGION"
        fi
        
        # Upload application package to S3
        aws s3 cp "$ZIP_FILE" "s3://$S3_BUCKET/"
        
        # Create application version
        VERSION_LABEL="${APP_NAME}-$(date +%Y%m%d%H%M%S)"
        aws elasticbeanstalk create-application-version --region "$REGION" --application-name "$APP_NAME" --version-label "$VERSION_LABEL" --source-bundle S3Bucket="$S3_BUCKET",S3Key="$ZIP_FILE"
        
        # Create environment
        aws elasticbeanstalk create-environment --region "$REGION" --application-name "$APP_NAME" --environment-name "$ENV_NAME" --solution-stack-name "64bit Amazon Linux 2 v3.5.0 running $PLATFORM" --version-label "$VERSION_LABEL" --option-settings file://.ebextensions/01_python.config
    fi
fi

echo "Deployment initiated!"
echo "You can check the status with: aws elasticbeanstalk describe-environments --environment-names $ENV_NAME"
echo "Once deployed, your application will be available at: http://$ENV_NAME.${REGION}.elasticbeanstalk.com" 