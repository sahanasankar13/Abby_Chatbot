# AWS Deployment Guide for Reproductive Health Chatbot

This guide provides step-by-step instructions for deploying the Reproductive Health Chatbot on AWS.

## Prerequisites

1. **AWS Account**: You need an AWS account with appropriate permissions.
2. **AWS CLI**: Install and configure the AWS CLI on your local machine.
3. **Git**: Install Git to clone the repository.
4. **API Keys**: Have your OpenAI API key and Abortion Policy API key ready.

## Deployment Options

You have three deployment options:

1. **Docker Deployment**: Using Elastic Container Service (ECS) or Elastic Kubernetes Service (EKS)
2. **Elastic Beanstalk Deployment**: Simplified deployment and management
3. **Manual EC2 Deployment**: More control over the infrastructure

## Option 1: Docker Deployment

### Build and Push Docker Image

1. Build the Docker image:
   ```bash
   docker build -t reproductive-health-chatbot .
   ```

2. Tag and push to Amazon ECR:
   ```bash
   aws ecr create-repository --repository-name reproductive-health-chatbot
   aws ecr get-login-password | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com
   docker tag reproductive-health-chatbot:latest <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/reproductive-health-chatbot:latest
   docker push <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/reproductive-health-chatbot:latest
   ```

3. Create an ECS cluster and service using the AWS Console or CLI.

## Option 2: Elastic Beanstalk Deployment

1. Install the Elastic Beanstalk CLI:
   ```bash
   pip install awsebcli
   ```

2. Initialize an Elastic Beanstalk application:
   ```bash
   eb init -p python-3.11 reproductive-health-chatbot --region <your-region>
   ```

3. Create an environment and deploy:
   ```bash
   eb create production-environment
   ```

4. Set environment variables for API keys:
   ```bash
   eb setenv OPENAI_API_KEY=<your-openai-api-key> ABORTION_POLICY_API_KEY=<your-abortion-policy-api-key> FLASK_ENV=production
   ```

5. Open the application:
   ```bash
   eb open
   ```

## Option 3: CloudFormation Deployment

1. Use the provided CloudFormation template:
   ```bash
   aws cloudformation create-stack \
     --stack-name reproductive-health-chatbot \
     --template-body file://cloudformation-template.yaml \
     --parameters \
       ParameterKey=EnvironmentName,ParameterValue=prod \
       ParameterKey=InstanceType,ParameterValue=t3.medium \
       ParameterKey=KeyName,ParameterValue=<your-key-pair-name> \
       ParameterKey=OpenAIApiKey,ParameterValue=<your-openai-api-key> \
       ParameterKey=AbortionPolicyApiKey,ParameterValue=<your-abortion-policy-api-key> \
     --capabilities CAPABILITY_IAM
   ```

2. Monitor the stack creation:
   ```bash
   aws cloudformation describe-stacks --stack-name reproductive-health-chatbot
   ```

## Option 4: Manual EC2 Deployment

1. Launch an EC2 instance with Amazon Linux 2023.

2. SSH into your instance:
   ```bash
   ssh -i your-key.pem ec2-user@your-instance-public-ip
   ```

3. Clone the repository:
   ```bash
   git clone https://github.com/your-username/reproductive-health-chatbot.git
   cd reproductive-health-chatbot
   ```

4. Follow the deployment script:
   ```bash
   chmod +x scripts/before_install.sh
   chmod +x scripts/after_install.sh
   chmod +x scripts/application_start.sh
   sudo ./scripts/before_install.sh
   sudo ./scripts/after_install.sh
   sudo ./scripts/application_start.sh
   ```

## Environment Variables

Set these environment variables for your deployment:

- `OPENAI_API_KEY`: Your OpenAI API key
- `ABORTION_POLICY_API_KEY`: Your Abortion Policy API key
- `FLASK_ENV`: Set to `production` for production deployments
- `SESSION_SECRET`: Secret key for Flask sessions (optional)

## Monitoring and Scaling

- Set up CloudWatch alarms for monitoring CPU and memory usage.
- Configure auto-scaling if using ECS, EKS, or EC2 Auto Scaling Groups.
- Set up Application Load Balancer for high availability.
- Enable CloudWatch Logs for tracking Ragas metrics and evaluation results.
- Consider setting up a CloudWatch Dashboard for visualizing Ragas metrics over time.

## Security Considerations

- Store API keys in AWS Secrets Manager or Parameter Store.
- Use HTTPS for all production deployments.
- Configure security groups to restrict access.
- Implement AWS WAF for additional security.

## Troubleshooting

- Check instance logs: `/var/log/cloud-init-output.log`
- Check application logs: `journalctl -u reproductive-health-chatbot`
- Verify environment variables: `systemctl show reproductive-health-chatbot`
- Test connectivity: `curl -v http://localhost:5000/api/health`

## Updating the Application

For updates:

1. Push changes to your repository.
2. For Elastic Beanstalk: `eb deploy`
3. For EC2/CloudFormation: Use CodeDeploy or rerun deployment scripts
4. For Docker: Rebuild and push the image, then update the service

## Cost Optimization

- Use Reserved Instances for predictable workloads.
- Scale down during low-traffic periods.
- Monitor and optimize resource usage.
- Consider AWS Lambda for serverless deployment of specific components.

## Conclusion

Follow this guide to successfully deploy the Reproductive Health Chatbot on AWS. Choose the deployment option that best fits your requirements for control, scalability, and management complexity.