#!/bin/bash

# AWS Deployment Script for Reproductive Health Chatbot
# This script helps package and deploy the application to AWS Elastic Beanstalk

# Exit on any error
set -e

# Configuration
APP_NAME="reproductive-health-chatbot"
EB_ENVIRONMENT="production"
S3_BUCKET="$APP_NAME-deployments"
REGION="us-east-1"
VERSION=$(date +%Y%m%d%H%M%S)
ZIP_FILE="$APP_NAME-$VERSION.zip"

echo "=== Starting deployment process for $APP_NAME ==="

# Check for AWS CLI
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check for AWS credentials
echo "Checking AWS credentials..."
aws sts get-caller-identity > /dev/null || { echo "AWS credentials not found or invalid"; exit 1; }

# Create deployment package
echo "Creating deployment package..."
# Make sure script is executable
chmod +x aws-deploy.sh

# Create zip excluding unnecessary files
echo "Creating zip file: $ZIP_FILE"
zip -r "$ZIP_FILE" . \
    -x "*.git*" \
    -x "*.pyc" \
    -x "__pycache__/*" \
    -x ".env" \
    -x "venv/*" \
    -x ".DS_Store" \
    -x "$ZIP_FILE" \
    -x "aws-deploy.sh"

echo "Deployment package created: $ZIP_FILE"

# Create S3 bucket if it doesn't exist
echo "Checking if S3 bucket exists..."
if ! aws s3api head-bucket --bucket "$S3_BUCKET" --region "$REGION" 2>/dev/null; then
    echo "Creating S3 bucket: $S3_BUCKET"
    aws s3api create-bucket --bucket "$S3_BUCKET" --region "$REGION"
fi

# Upload to S3
echo "Uploading deployment package to S3..."
aws s3 cp "$ZIP_FILE" "s3://$S3_BUCKET/$ZIP_FILE" --region "$REGION"

# Check if the Elastic Beanstalk application exists
echo "Checking if Elastic Beanstalk application exists..."
if ! aws elasticbeanstalk describe-applications --application-names "$APP_NAME" --region "$REGION" &>/dev/null; then
    echo "Creating Elastic Beanstalk application: $APP_NAME"
    aws elasticbeanstalk create-application --application-name "$APP_NAME" --region "$REGION"
fi

# Check if the environment exists
echo "Checking if Elastic Beanstalk environment exists..."
if ! aws elasticbeanstalk describe-environments --application-name "$APP_NAME" --environment-names "$EB_ENVIRONMENT" --region "$REGION" | grep -q "\"Status\": \"Ready\""; then
    echo "Creating Elastic Beanstalk environment: $EB_ENVIRONMENT"
    aws elasticbeanstalk create-environment \
        --application-name "$APP_NAME" \
        --environment-name "$EB_ENVIRONMENT" \
        --solution-stack-name "64bit Amazon Linux 2023 v4.0.7 running Python 3.11" \
        --option-settings file://.ebextensions/01_flask.config \
        --region "$REGION"
    
    echo "Waiting for environment to be ready..."
    aws elasticbeanstalk wait environment-exists \
        --application-name "$APP_NAME" \
        --environment-names "$EB_ENVIRONMENT" \
        --region "$REGION"
fi

# Create a new application version
echo "Creating new application version: $VERSION"
aws elasticbeanstalk create-application-version \
    --application-name "$APP_NAME" \
    --version-label "$VERSION" \
    --source-bundle S3Bucket="$S3_BUCKET",S3Key="$ZIP_FILE" \
    --region "$REGION"

# Deploy the new version
echo "Deploying version $VERSION to environment $EB_ENVIRONMENT"
aws elasticbeanstalk update-environment \
    --application-name "$APP_NAME" \
    --environment-name "$EB_ENVIRONMENT" \
    --version-label "$VERSION" \
    --region "$REGION"

echo "Deployment initiated. Check the AWS Elastic Beanstalk console for status."
echo "Deployment complete! Access your application at:"
aws elasticbeanstalk describe-environments \
    --application-name "$APP_NAME" \
    --environment-names "$EB_ENVIRONMENT" \
    --region "$REGION" \
    --query "Environments[0].CNAME" \
    --output text