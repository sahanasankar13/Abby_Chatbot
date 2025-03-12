# AWS Environment Variables Configuration

When deploying the Reproductive Health Chatbot to AWS Elastic Beanstalk, you'll need to configure the following environment variables in the AWS Elastic Beanstalk Console.

## Required Environment Variables

| Variable Name | Description | Example Value |
|---------------|-------------|---------------|
| `OPENAI_API_KEY` | Your OpenAI API key for GPT-4 integration | sk-... |
| `ABORTION_POLICY_API_KEY` | API key for abortion policy information | abc123... |
| `FLASK_ENV` | Flask environment mode | production |
| `FLASK_APP` | Flask application entry point | app.py |

## Optional Environment Variables

| Variable Name | Description | Default Value |
|---------------|-------------|---------------|
| `LOG_LEVEL` | Logging level | INFO |
| `DATABASE_URL` | Database connection URL (if using a database) | None |
| `SESSION_SECRET` | Secret key for Flask sessions | Auto-generated if not provided |

## How to Set Environment Variables in AWS Elastic Beanstalk

1. Navigate to the AWS Elastic Beanstalk Console
2. Select your application and environment
3. Go to "Configuration" in the left navigation panel
4. Under "Software", click "Modify"
5. Scroll down to the "Environment properties" section
6. Add each key-value pair
7. Click "Apply" to update the configuration

## Important Security Notes

- Never commit API keys or secrets to your repository
- Use AWS Parameter Store or Secrets Manager for highly sensitive values
- Regularly rotate your API keys and secrets
- Ensure proper IAM roles and permissions are configured

## Verification

After deployment, you can verify that environment variables are correctly set by checking your application logs in the AWS Elastic Beanstalk Console.