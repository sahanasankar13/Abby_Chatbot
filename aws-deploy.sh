#!/bin/bash

# AWS Deployment Script for Reproductive Health Chatbot
# This script automates the deployment process to AWS

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
REGION=${AWS_REGION:-"us-east-1"}
APP_NAME="reproductive-health-chatbot"
ENVIRONMENT_NAME=${ENVIRONMENT:-"production"}
S3_BUCKET="${APP_NAME}-deployment-${REGION}"
EB_PLATFORM="Python 3.11"
ZIP_FILE="${APP_NAME}-$(date +%Y%m%d%H%M%S).zip"

# Functions
function print_header {
    echo -e "\n${YELLOW}===== $1 =====${NC}\n"
}

function check_prerequisites {
    print_header "Checking Prerequisites"
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
        exit 1
    fi
    
    # Check EB CLI for Elastic Beanstalk deployment
    if ! command -v eb &> /dev/null; then
        echo -e "${YELLOW}Elastic Beanstalk CLI not found. Installing...${NC}"
        pip install awsebcli
    fi
    
    # Check if AWS credentials are configured
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED}AWS credentials not configured. Run 'aws configure' first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Prerequisites check passed!${NC}"
}

function create_deployment_artifacts {
    print_header "Creating Deployment Artifacts"
    
    # Create zip file for deployment, excluding unnecessary files
    echo "Creating deployment package: ${ZIP_FILE}"
    zip -r "${ZIP_FILE}" . -x "*.git*" "*.zip" "*.pyc" "__pycache__/*" "venv/*" ".env*" "node_modules/*" ".DS_Store" "*.log"
    
    echo -e "${GREEN}Deployment artifact created!${NC}"
}

function deploy_to_elastic_beanstalk {
    print_header "Deploying to Elastic Beanstalk"
    
    # Check if the application exists
    if ! aws elasticbeanstalk describe-applications --application-names "${APP_NAME}" &> /dev/null; then
        echo "Creating Elastic Beanstalk application: ${APP_NAME}"
        aws elasticbeanstalk create-application --application-name "${APP_NAME}" --description "Reproductive Health Chatbot Application"
    fi
    
    # Check if the environment exists
    if aws elasticbeanstalk describe-environments --application-name "${APP_NAME}" --environment-names "${ENVIRONMENT_NAME}" | grep -q "\"Status\": \"Ready\""; then
        echo "Updating existing environment: ${ENVIRONMENT_NAME}"
        aws elasticbeanstalk update-environment --application-name "${APP_NAME}" --environment-name "${ENVIRONMENT_NAME}" --version-label "${APP_NAME}-${VERSION_LABEL}"
    else
        echo "Creating new environment: ${ENVIRONMENT_NAME}"
        # Initialize EB application if not already done
        if [ ! -f ".elasticbeanstalk/config.yml" ]; then
            eb init "${APP_NAME}" --region "${REGION}" --platform "${EB_PLATFORM}"
        fi
        
        # Create the environment
        eb create "${ENVIRONMENT_NAME}" --region "${REGION}" --platform "${EB_PLATFORM}"
        
        # Wait for the environment to be ready
        echo "Waiting for environment to be ready..."
        aws elasticbeanstalk wait environment-exists --application-name "${APP_NAME}" --environment-names "${ENVIRONMENT_NAME}"
    fi
    
    echo -e "${GREEN}Deployment to Elastic Beanstalk completed!${NC}"
}

function deploy_to_ec2_with_codedeploy {
    print_header "Deploying to EC2 with CodeDeploy"
    
    # Create S3 bucket if it doesn't exist
    if ! aws s3api head-bucket --bucket "${S3_BUCKET}" 2>/dev/null; then
        echo "Creating S3 bucket: ${S3_BUCKET}"
        aws s3 mb s3://"${S3_BUCKET}" --region "${REGION}"
    fi
    
    # Upload deployment artifact to S3
    echo "Uploading deployment package to S3"
    aws s3 cp "${ZIP_FILE}" s3://"${S3_BUCKET}"/
    
    # Create CodeDeploy application if it doesn't exist
    if ! aws deploy get-application --application-name "${APP_NAME}" &>/dev/null; then
        echo "Creating CodeDeploy application: ${APP_NAME}"
        aws deploy create-application --application-name "${APP_NAME}"
    fi
    
    # Create deployment group if it doesn't exist
    DEPLOYMENT_GROUP="${ENVIRONMENT_NAME}-deployment-group"
    if ! aws deploy get-deployment-group --application-name "${APP_NAME}" --deployment-group-name "${DEPLOYMENT_GROUP}" &>/dev/null; then
        echo "Creating deployment group: ${DEPLOYMENT_GROUP}"
        # Note: This requires an existing service role and EC2 tag group
        # You'll need to adjust this part to match your infrastructure
        aws deploy create-deployment-group \
            --application-name "${APP_NAME}" \
            --deployment-group-name "${DEPLOYMENT_GROUP}" \
            --deployment-config-name CodeDeployDefault.OneAtATime \
            --ec2-tag-filters Key=Name,Value=reproductive-health-chatbot,Type=KEY_AND_VALUE \
            --service-role-arn "arn:aws:iam::$(aws sts get-caller-identity --query 'Account' --output text):role/CodeDeployServiceRole"
    fi
    
    # Create deployment
    echo "Creating deployment"
    aws deploy create-deployment \
        --application-name "${APP_NAME}" \
        --deployment-group-name "${DEPLOYMENT_GROUP}" \
        --revision revisionType=S3,s3Location="{bucket=${S3_BUCKET},key=${ZIP_FILE},bundleType=zip}"
    
    echo -e "${GREEN}Deployment to EC2 with CodeDeploy initiated!${NC}"
}

function deploy_using_cloudformation {
    print_header "Deploying using CloudFormation"
    
    # Create S3 bucket for CloudFormation templates if it doesn't exist
    CF_BUCKET="${APP_NAME}-cloudformation-${REGION}"
    if ! aws s3api head-bucket --bucket "${CF_BUCKET}" 2>/dev/null; then
        echo "Creating S3 bucket for CloudFormation: ${CF_BUCKET}"
        aws s3 mb s3://"${CF_BUCKET}" --region "${REGION}"
    fi
    
    # Upload CloudFormation template to S3
    echo "Uploading CloudFormation template to S3"
    aws s3 cp cloudformation-template.yaml s3://"${CF_BUCKET}"/
    
    # Create or update CloudFormation stack
    STACK_NAME="${APP_NAME}-${ENVIRONMENT_NAME}"
    if aws cloudformation describe-stacks --stack-name "${STACK_NAME}" &>/dev/null; then
        echo "Updating existing CloudFormation stack: ${STACK_NAME}"
        aws cloudformation update-stack \
            --stack-name "${STACK_NAME}" \
            --template-url "https://${CF_BUCKET}.s3.amazonaws.com/cloudformation-template.yaml" \
            --parameters \
                ParameterKey=EnvironmentName,ParameterValue="${ENVIRONMENT_NAME}" \
                ParameterKey=InstanceType,ParameterValue=t3.medium \
            --capabilities CAPABILITY_IAM
    else
        echo "Creating new CloudFormation stack: ${STACK_NAME}"
        aws cloudformation create-stack \
            --stack-name "${STACK_NAME}" \
            --template-url "https://${CF_BUCKET}.s3.amazonaws.com/cloudformation-template.yaml" \
            --parameters \
                ParameterKey=EnvironmentName,ParameterValue="${ENVIRONMENT_NAME}" \
                ParameterKey=InstanceType,ParameterValue=t3.medium \
            --capabilities CAPABILITY_IAM
    fi
    
    echo "Waiting for stack operation to complete..."
    aws cloudformation wait stack-create-complete --stack-name "${STACK_NAME}" || aws cloudformation wait stack-update-complete --stack-name "${STACK_NAME}"
    
    # Output stack resources
    echo "Stack resources:"
    aws cloudformation describe-stack-resources --stack-name "${STACK_NAME}" --query "StackResources[].{LogicalResourceId:LogicalResourceId,PhysicalResourceId:PhysicalResourceId,Type:ResourceType}" --output table
    
    echo -e "${GREEN}CloudFormation deployment completed!${NC}"
}

function setup_monitoring {
    print_header "Setting up Monitoring and Logging"
    
    # Create CloudWatch log group if it doesn't exist
    LOG_GROUP="/aws/ec2/${APP_NAME}"
    if ! aws logs describe-log-groups --log-group-name-prefix "${LOG_GROUP}" | grep -q "${LOG_GROUP}"; then
        echo "Creating CloudWatch log group: ${LOG_GROUP}"
        aws logs create-log-group --log-group-name "${LOG_GROUP}"
    fi
    
    # Create basic CloudWatch alarm for high CPU
    ALARM_NAME="${APP_NAME}-high-cpu"
    if ! aws cloudwatch describe-alarms --alarm-names "${ALARM_NAME}" | grep -q "${ALARM_NAME}"; then
        echo "Creating CloudWatch alarm for high CPU usage"
        aws cloudwatch put-metric-alarm \
            --alarm-name "${ALARM_NAME}" \
            --alarm-description "Alarm when CPU exceeds 80% for 5 minutes" \
            --metric-name CPUUtilization \
            --namespace AWS/EC2 \
            --statistic Average \
            --period 300 \
            --threshold 80 \
            --comparison-operator GreaterThanThreshold \
            --dimensions "Name=AutoScalingGroupName,Value=${APP_NAME}-${ENVIRONMENT_NAME}-asg" \
            --evaluation-periods 2 \
            --alarm-actions arn:aws:sns:${REGION}:$(aws sts get-caller-identity --query 'Account' --output text):${APP_NAME}-alerts
    fi
    
    echo -e "${GREEN}Monitoring and logging setup completed!${NC}"
}

function cleanup {
    print_header "Cleanup"
    
    # Remove temporary files
    echo "Removing temporary files"
    rm -f "${ZIP_FILE}"
    
    echo -e "${GREEN}Cleanup completed!${NC}"
}

function display_deployment_info {
    print_header "Deployment Information"
    
    # Get deployment URL
    if [[ "$DEPLOYMENT_TYPE" == "elastic-beanstalk" ]]; then
        DEPLOY_URL=$(aws elasticbeanstalk describe-environments --application-name "${APP_NAME}" --environment-names "${ENVIRONMENT_NAME}" --query "Environments[0].CNAME" --output text)
        echo -e "Elastic Beanstalk URL: ${GREEN}http://${DEPLOY_URL}${NC}"
    elif [[ "$DEPLOYMENT_TYPE" == "cloudformation" ]]; then
        LB_URL=$(aws cloudformation describe-stacks --stack-name "${STACK_NAME}" --query "Stacks[0].Outputs[?OutputKey=='ApplicationURL'].OutputValue" --output text)
        echo -e "Application URL: ${GREEN}${LB_URL}${NC}"
    fi
    
    echo -e "\n${GREEN}Deployment completed successfully!${NC}"
    echo -e "Application: ${APP_NAME}"
    echo -e "Environment: ${ENVIRONMENT_NAME}"
    echo -e "Region: ${REGION}"
    echo -e "Deployment Type: ${DEPLOYMENT_TYPE}"
    echo -e "\nRemember to set environment variables using the 'aws-environment-variables.md' guide."
}

# Main
print_header "Reproductive Health Chatbot AWS Deployment"

echo "Select deployment type:"
echo "1) Elastic Beanstalk"
echo "2) EC2 with CodeDeploy"
echo "3) CloudFormation"
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        DEPLOYMENT_TYPE="elastic-beanstalk"
        check_prerequisites
        create_deployment_artifacts
        deploy_to_elastic_beanstalk
        ;;
    2)
        DEPLOYMENT_TYPE="codedeploy"
        check_prerequisites
        create_deployment_artifacts
        deploy_to_ec2_with_codedeploy
        ;;
    3)
        DEPLOYMENT_TYPE="cloudformation"
        check_prerequisites
        deploy_using_cloudformation
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

setup_monitoring
cleanup
display_deployment_info