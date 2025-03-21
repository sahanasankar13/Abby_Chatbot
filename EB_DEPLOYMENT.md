# Deploying Abby Chatbot to AWS Elastic Beanstalk

This document outlines the steps for deploying the Abby Chatbot application to AWS Elastic Beanstalk.

## Prerequisites

1. AWS CLI installed and configured with appropriate credentials
2. Elastic Beanstalk CLI (`awsebcli`) installed
3. Python 3.9+ installed
4. All required API keys in your `.env` file

## Deployment Options

### Option 1: Automated Deployment

The easiest way to deploy is to use the provided deployment script:

```
./deploy_to_eb.sh
```

This script will:
1. Download NLTK data
2. Create the necessary EB configuration files
3. Package your application
4. Deploy it to Elastic Beanstalk

You can customize the deployment with these options:
- `--region` - AWS region (default: us-east-1)
- `--app-name` - Application name (default: abby-chatbot)
- `--env-name` - Environment name (default: abby-chatbot-env)

Example with custom settings:
```
./deploy_to_eb.sh --region us-west-2 --app-name my-chatbot --env-name production
```

### Option 2: Manual Deployment with EB CLI

If you prefer to use the EB CLI directly, follow these steps:

1. Initialize EB application (first time only):
   ```
   eb init -p python-3.9 abby-chatbot --region us-east-1
   ```

2. Create environment (first time only):
   ```
   eb create abby-chatbot-env
   ```

3. Deploy updates:
   ```
   eb deploy
   ```

4. Open the deployed application:
   ```
   eb open
   ```

## Monitoring and Management

- View application logs:
  ```
  eb logs
  ```

- Check application status:
  ```
  eb status
  ```

- SSH into the EC2 instance:
  ```
  eb ssh
  ```

- Terminate the environment when not in use:
  ```
  eb terminate abby-chatbot-env
  ```

## Cost Considerations

Elastic Beanstalk itself is free, but you pay for the underlying resources:
- EC2 instance (t3.micro is eligible for the free tier)
- Other AWS resources like EBS volumes

To minimize costs:
1. Use t3.micro instances for low-traffic applications
2. Terminate environments when not in use
3. Monitor your AWS billing dashboard

## Troubleshooting

1. Application not responding:
   - Check EB logs: `eb logs`
   - SSH into instance: `eb ssh` and check application logs

2. Deployment failures:
   - Ensure your application works locally before deploying
   - Check that all requirements are properly listed in requirements.txt
   - Verify your .ebextensions configurations

3. Environment creation timeout:
   - This sometimes happens during the first deployment
   - Try terminating and recreating the environment

4. Permissions issues:
   - Ensure your AWS user has the necessary IAM permissions for Elastic Beanstalk
   - Check if your application needs access to other AWS services 