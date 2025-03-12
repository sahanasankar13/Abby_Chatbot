# AWS Environment Variables Guide

This document outlines the environment variables required for deploying the Reproductive Health Chatbot on AWS.

## Core Environment Variables

These variables are essential for the application to function correctly:

| Variable Name | Description | Required | Default |
|---------------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 integration | Yes | None |
| `ABORTION_POLICY_API_KEY` | API key for accessing abortion policy data | Yes | None |
| `FLASK_ENV` | Flask environment setting | Yes | `development` |
| `SESSION_SECRET` | Secret key for Flask sessions | Yes | None |

## Database Configuration (If Applicable)

If using a database:

| Variable Name | Description | Required | Default |
|---------------|-------------|----------|---------|
| `DATABASE_URL` | Connection string for the database | No | SQLite (local) |

## Application Configuration

Optional application configuration variables:

| Variable Name | Description | Required | Default |
|---------------|-------------|----------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No | `INFO` |
| `MAX_TOKENS` | Maximum tokens for GPT responses | No | `300` |
| `MODEL_NAME` | GPT model to use | No | `gpt-4o` |

## AWS-Specific Configuration

Variables specific to AWS deployments:

| Variable Name | Description | Required | Default |
|---------------|-------------|----------|---------|
| `AWS_REGION` | AWS region for services | No | Deployment region |
| `AWS_LOG_GROUP` | CloudWatch log group | No | `/reproductive-health-chatbot` |

## Security Recommendations

For secure configuration in AWS:

1. **Use Parameter Store or Secrets Manager**:
   ```
   # Retrieve secrets in your application
   import boto3
   
   ssm = boto3.client('ssm', region_name='your-region')
   response = ssm.get_parameter(
       Name='/reproductive-health-chatbot/OPENAI_API_KEY',
       WithDecryption=True
   )
   openai_api_key = response['Parameter']['Value']
   ```

2. **Environment Variable Encryption**:
   - When using Elastic Beanstalk, enable environment variable encryption.
   - For ECS/EKS, use secrets as environment variables.

3. **IAM Role Configuration**:
   - Create a role with minimal permissions needed for accessing secrets.
   - Example policy:
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Effect": "Allow",
           "Action": [
             "ssm:GetParameter"
           ],
           "Resource": "arn:aws:ssm:region:account-id:parameter/reproductive-health-chatbot/*"
         }
       ]
     }
     ```

## Setting Environment Variables in Different AWS Services

### Elastic Beanstalk

```bash
eb setenv OPENAI_API_KEY=your-api-key ABORTION_POLICY_API_KEY=your-api-key FLASK_ENV=production SESSION_SECRET=your-secret
```

### ECS

In task definition JSON:
```json
"environment": [
  {
    "name": "FLASK_ENV",
    "value": "production"
  }
],
"secrets": [
  {
    "name": "OPENAI_API_KEY",
    "valueFrom": "arn:aws:ssm:region:account-id:parameter/reproductive-health-chatbot/OPENAI_API_KEY"
  },
  {
    "name": "ABORTION_POLICY_API_KEY",
    "valueFrom": "arn:aws:ssm:region:account-id:parameter/reproductive-health-chatbot/ABORTION_POLICY_API_KEY"
  }
]
```

### EC2 (Using systemd)

In your systemd service file:
```
[Service]
Environment="FLASK_ENV=production"
Environment="LOG_LEVEL=INFO"
# Use EnvironmentFile for secrets
EnvironmentFile=/etc/reproductive-health-chatbot/env
```

Contents of `/etc/reproductive-health-chatbot/env`:
```
OPENAI_API_KEY=your-api-key
ABORTION_POLICY_API_KEY=your-api-key
SESSION_SECRET=your-secret
```

## Testing Environment Variables

To verify environment variables are correctly set up:

```python
# Add this to a route in app.py for debugging
@app.route('/api/env-check')
def env_check():
    env_vars = {
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY') is not None,
        'ABORTION_POLICY_API_KEY': os.environ.get('ABORTION_POLICY_API_KEY') is not None,
        'FLASK_ENV': os.environ.get('FLASK_ENV'),
    }
    return jsonify(env_vars)
```

**SECURITY WARNING**: Remove this route before production deployment. This is only for validation during setup.