# Serverless Deployment for Abby Chatbot

This guide explains how to deploy the Abby Chatbot as a serverless application using AWS Lambda and API Gateway through the Serverless Framework.

## Cost-Effective Architecture

This serverless architecture is optimized for infrequent usage patterns and will help make your AWS credits last longer:

- Lambda functions only run when invoked, so there's no cost when idle
- AWS Free Tier includes 1 million Lambda requests and 400,000 GB-seconds of compute time per month
- API Gateway's free tier includes 1 million API calls per month
- DynamoDB's free tier includes 25GB of storage and 25 read/write capacity units
- Expected cost for low usage patterns: ~$1-5/month

## Prerequisites

1. **AWS Account and Credentials**
   - AWS account with access to Lambda, API Gateway, and DynamoDB
   - AWS CLI installed and configured with appropriate credentials

2. **Node.js and npm**
   - Node.js 14+ and npm for Serverless Framework

3. **Python Environment**
   - Python 3.9 or higher
   - virtualenv or similar for dependency management

## Setup and Deployment

### 1. Prepare Environment Variables

Create a `.env` file with your configuration:

```bash
# Required environment variables
OPENAI_API_KEY=your-openai-api-key
ABORTION_POLICY_API_KEY=your-policy-api-key
SESSION_SECRET=your-session-secret
GOOGLE_MAPS_API_KEY=your-google-maps-api-key

# Cost optimization settings
USE_CHEAP_CLASSIFIER=true
EVALUATION_FREQUENCY=20
MONTHLY_TOKEN_BUDGET=1000000
ENABLE_CACHING=true
EVALUATION_MODEL=local
```

### 2. Install Dependencies

```bash
# Install Serverless Framework and plugins
npm install

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Deploy to AWS

```bash
# Deploy the application
npm run deploy

# Or manually using Serverless CLI
serverless deploy
```

### 4. Use the Application

After deployment, you'll receive URLs for your API endpoints. The main entry points are:

- **Frontend**: The base URL provided in the deployment outputs
- **API**: /api/chat endpoint for sending messages to the chatbot

## Uploading Model Files

For large ML models that exceed Lambda's package size limits:

```bash
# Upload model files to S3
aws s3 cp ./models/ s3://abby-chatbot-data-dev/models/ --recursive
```

## Monitoring and Maintenance

```bash
# View Lambda logs
serverless logs -f api

# Update the deployment
serverless deploy

# Remove the deployment when no longer needed
npm run remove
```

## Cost Management Tips

1. **Monitor Usage**: 
   - Set up CloudWatch Alarms to monitor Lambda invocations and durations
   - Use AWS Budgets to get alerts when costs exceed thresholds

2. **Optimize Lambda Size**:
   - Only include necessary dependencies
   - Use Lambda Layers for large dependencies

3. **Cold Start Optimization**:
   - Keep the handler code minimal
   - Use Provisioned Concurrency only if cold starts become problematic

4. **Database Optimization**:
   - Use DynamoDB On-demand pricing for unpredictable workloads
   - Implement efficient data access patterns

## Troubleshooting

- **Deployment Errors**: Check that your AWS credentials are valid and have appropriate permissions
- **Lambda Timeouts**: Increase the Lambda timeout in serverless.yml (max 15 minutes)
- **Package Size Limits**: Use Lambda Layers and S3 for large dependencies or models
- **API Gateway Errors**: Check Lambda logs for detailed error information

## Advanced Configuration

See serverless.yml for additional configuration options including:
- Custom domain names
- Advanced security settings
- Resource scaling
- VPC configuration 