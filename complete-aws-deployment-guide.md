# Complete AWS Deployment Guide for Reproductive Health Chatbot

This comprehensive guide provides detailed, step-by-step instructions for deploying the Reproductive Health Chatbot application on AWS infrastructure. This guide is designed to be beginner-friendly while covering all technical aspects of deployment.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Preparing Your Environment](#preparing-your-environment)
3. [Setting Up AWS Resources](#setting-up-aws-resources)
4. [Database Setup](#database-setup)
5. [Application Deployment](#application-deployment)
6. [Setting Up Continuous Integration/Deployment](#setting-up-continuous-integrationdeployment)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Security Best Practices](#security-best-practices)
9. [Scaling Strategies](#scaling-strategies)
10. [Cost Optimization](#cost-optimization)
11. [Troubleshooting](#troubleshooting)
12. [Maintenance and Updates](#maintenance-and-updates)

## Prerequisites

Before starting the deployment process, ensure you have:

1. **AWS Account**: Create an AWS account if you don't have one already
2. **API Keys**:
   - OpenAI API Key (for GPT integration)
   - Abortion Policy API Key
3. **Required Tools**:
   - AWS CLI installed and configured
   - Git installed
   - Python 3.11 or later
   - Pip package manager

## Preparing Your Environment

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/reproductive-health-chatbot.git
cd reproductive-health-chatbot
```

### 2. Configure Environment Variables

Create a `.env` file for local testing:

```
OPENAI_API_KEY=your-openai-api-key
ABORTION_POLICY_API_KEY=your-abortion-policy-api-key
SESSION_SECRET=your-session-secret
FLASK_ENV=development
```

### 3. Create Deployment Configuration Files

#### Create `.ebignore` file:

```
.git
.gitignore
.env
__pycache__/
*.pyc
*.pyo
*.pyd
venv/
.DS_Store
```

#### Verify `requirements.txt` is up to date:

```bash
pip freeze > requirements.txt
```

#### Ensure Proper Procfile (create if missing):

```
web: gunicorn --bind 0.0.0.0:5000 --workers=2 --threads=4 main:app
```

## Setting Up AWS Resources

### 1. Create IAM User for Deployment

1. Navigate to AWS IAM in the console
2. Click "Users" → "Add user"
3. Username: `reproductive-health-deployment`
4. Access type: "Programmatic access"
5. Attach existing policies directly:
   - AWSElasticBeanstalkFullAccess
   - AmazonRDSFullAccess (if using RDS)
   - CloudWatchLogsFullAccess
   - IAMFullAccess (if configuring additional roles)
6. Save the Access Key ID and Secret Access Key securely

### 2. Configure AWS CLI

```bash
aws configure
# Enter the access key, secret key, region (e.g., us-east-1), output format (json)
```

### 3. Create Security Groups

#### Database Security Group

```bash
aws ec2 create-security-group \
  --group-name reproductive-health-rds-sg \
  --description "Security group for RDS database" \
  --vpc-id your-vpc-id

# Allow access only from the application servers
aws ec2 authorize-security-group-ingress \
  --group-id sg-rds-id \
  --protocol tcp \
  --port 5432 \
  --source-group app-security-group-id
```

#### Application Security Group

```bash
aws ec2 create-security-group \
  --group-name reproductive-health-app-sg \
  --description "Security group for application servers" \
  --vpc-id your-vpc-id

# Allow HTTP/HTTPS traffic
aws ec2 authorize-security-group-ingress \
  --group-id sg-app-id \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-app-id \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0
```

## Database Setup

### 1. Create PostgreSQL RDS Instance

Using the AWS Console:

1. Navigate to RDS in the console
2. Click "Create database"
3. Select "Standard create" and "PostgreSQL"
4. For Templates, choose "Free tier" for development or "Production" for production
5. Settings:
   - DB instance identifier: `reproductive-health-db`
   - Master username: Create a secure username
   - Master password: Create a secure password (store this safely!)
6. DB instance size: 
   - Development: "db.t3.micro"
   - Production: "db.t3.small" or larger
7. Storage: Start with 20GB, enable storage autoscaling
8. Connectivity:
   - VPC: Select your VPC
   - Subnet group: Create or select existing
   - Public access: No (for security)
   - VPC security group: Select the DB security group created earlier
9. Additional configuration:
   - Initial database name: `reproductive_health_db`
   - Backup retention: 7 days (production) or 1 day (development)
   - Enable encryption
10. Click "Create database"

Or using the AWS CLI:

```bash
aws rds create-db-instance \
  --db-instance-identifier reproductive-health-db \
  --engine postgres \
  --engine-version 13.7 \
  --db-instance-class db.t3.micro \
  --allocated-storage 20 \
  --master-username dbadmin \
  --master-user-password your-secure-password \
  --db-name reproductive_health_db \
  --vpc-security-group-ids sg-rds-id \
  --db-subnet-group-name your-db-subnet-group \
  --backup-retention-period 7 \
  --storage-encrypted \
  --no-publicly-accessible
```

### 2. Get Database Connection Information

```bash
aws rds describe-db-instances --db-instance-identifier reproductive-health-db
```

Note the endpoint, port, database name, username, and password for the next steps.

## Application Deployment

### 1. Initialize Elastic Beanstalk Application

```bash
eb init -p python-3.11 reproductive-health-chatbot --region your-region
```

Answer the prompts:
- When asked to use CodeCommit, select "No"
- When asked to set up SSH, select "Yes" if you want SSH access to instances

### 2. Create the Elastic Beanstalk Environment

#### For development (single instance):

```bash
eb create reproductive-health-dev \
  --instance-type t2.micro \
  --single \
  --envvars "OPENAI_API_KEY=your-openai-key,ABORTION_POLICY_API_KEY=your-policy-api-key,SESSION_SECRET=your-session-secret,FLASK_ENV=production,DATABASE_URL=postgresql://username:password@db-endpoint:5432/reproductive_health_db"
```

#### For production (load-balanced):

```bash
eb create reproductive-health-prod \
  --instance-type t3.small \
  --min-instances 2 \
  --max-instances 4 \
  --envvars "OPENAI_API_KEY=your-openai-key,ABORTION_POLICY_API_KEY=your-policy-api-key,SESSION_SECRET=your-session-secret,FLASK_ENV=production,DATABASE_URL=postgresql://username:password@db-endpoint:5432/reproductive_health_db" \
  --elb-type application \
  --ssl-certificate your-certificate-arn
```

### 3. Configure Health Checks and Monitoring

1. In the AWS Console, navigate to the Elastic Beanstalk environment
2. Go to "Configuration" → "Monitoring"
3. Set up enhanced health reporting
4. Add environment-specific metrics
5. Enable CloudWatch logs

### 4. Set Up Custom Domain and HTTPS (Production)

1. Purchase/configure your domain in Route 53 or your preferred DNS provider
2. Request an SSL certificate using AWS Certificate Manager (ACM):
   - In the AWS Console, navigate to ACM
   - Click "Request a certificate"
   - Enter your domain name (and optionally www.yourdomain.com)
   - Select DNS validation
   - Add the validation records to your DNS configuration
3. In your Elastic Beanstalk environment:
   - Go to "Configuration" → "Load balancer"
   - Add a listener on port 443 with your SSL certificate
4. Create a CNAME record in your DNS that points your domain to your Elastic Beanstalk environment URL

## Setting Up Continuous Integration/Deployment

### 1. Set Up AWS CodePipeline (Optional)

1. In the AWS Console, navigate to CodePipeline
2. Click "Create pipeline"
3. Pipeline name: `reproductive-health-pipeline`
4. Select "New service role"
5. Source provider: Select your source (GitHub, CodeCommit, etc.)
6. Connect to your repository and select the branch
7. Build provider: Skip build stage (or add CodeBuild if needed)
8. Deploy provider: Elastic Beanstalk
   - Application name: `reproductive-health-chatbot`
   - Environment name: Your environment name
9. Create the pipeline

### 2. Configure GitHub Actions (Alternative)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to AWS Elastic Beanstalk

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install awsebcli
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ secrets.AWS_REGION }}
    
    - name: Deploy to Elastic Beanstalk
      run: |
        eb deploy reproductive-health-prod
```

Then add your secrets in GitHub repository settings.

## Monitoring and Logging

### 1. Set Up CloudWatch Logs

1. In the AWS Console, navigate to Elastic Beanstalk environment
2. Go to "Configuration" → "Software"
3. Under "Log options", enable log streaming to CloudWatch Logs

### 2. Create CloudWatch Dashboard

1. In the AWS Console, navigate to CloudWatch
2. Click "Dashboards" → "Create dashboard"
3. Dashboard name: `reproductive-health-dashboard`
4. Add widgets:
   - EC2 instance metrics
   - RDS metrics
   - Application logs
   - Custom metrics for Ragas evaluation

### 3. Set Up CloudWatch Alarms

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name reproductive-health-cpu-alarm \
  --alarm-description "Alarm when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/EC2 \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=AutoScalingGroupName,Value=your-asg-name \
  --evaluation-periods 2 \
  --alarm-actions your-sns-topic-arn
```

## Security Best Practices

### 1. Store Secrets Properly

Move API keys to AWS Secrets Manager:

```bash
aws secretsmanager create-secret \
  --name "reproductive-health/api-keys" \
  --description "API keys for the reproductive health chatbot" \
  --secret-string "{\"OPENAI_API_KEY\":\"your-key\",\"ABORTION_POLICY_API_KEY\":\"your-key\"}"
```

Then update your application to fetch secrets:

```python
import boto3
import json

def get_secrets():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='reproductive-health/api-keys')
    secrets = json.loads(response['SecretString'])
    return secrets
```

### 2. Enable Network Protection

1. Use WAF for application protection:
   - In the AWS Console, navigate to WAF & Shield
   - Create a web ACL with rules for SQL injection, XSS, etc.
   - Associate with your load balancer

2. Enable DDoS protection with AWS Shield

### 3. Configure Security Policies

1. In Elastic Beanstalk, update security settings:
   - Go to "Configuration" → "Security"
   - Configure service role and instance profile
   - Update security groups as needed

## Scaling Strategies

### 1. Configure Auto Scaling

1. In Elastic Beanstalk, go to "Configuration" → "Capacity"
2. Set up scaling triggers:
   - Metric: CPUUtilization
   - Statistic: Average
   - Unit: Percent
   - Period: 5 minutes
   - Upper threshold: 70%
   - Lower threshold: 30%
   - Scale-up increment: 1
   - Scale-down increment: -1

### 2. Implement Caching (Optional)

For improved performance, add ElastiCache:

```bash
aws elasticache create-cache-cluster \
  --cache-cluster-id reproductive-health-cache \
  --engine redis \
  --cache-node-type cache.t2.micro \
  --num-cache-nodes 1 \
  --security-group-ids your-cache-sg-id
```

Then update your application to use Redis caching.

## Cost Optimization

### 1. Use Reserved Instances for Production

For consistent workloads, purchase reserved instances:

```bash
aws ec2 purchase-reserved-instances-offering \
  --reserved-instances-offering-id offering-id \
  --instance-count 1
```

### 2. Implement Auto-Shutdown for Development

For dev/test environments, create Lambda functions to stop resources during off-hours.

### 3. Monitor and Analyze Costs

1. Enable AWS Cost Explorer
2. Set up budget alerts
3. Regularly review and optimize resource usage

## Troubleshooting

### 1. Common Issues and Solutions

#### Application Fails to Deploy

Check the logs:

```bash
eb logs
```

Look for Python dependency issues, environment variables, or database connection problems.

#### Database Connection Issues

Verify security groups and test connection:

```bash
psql -h your-db-endpoint -U username -d reproductive_health_db
```

#### High CPU or Memory Usage

Check metrics and logs to identify the cause:

```bash
eb health
```

### 2. Getting Support

1. AWS documentation: https://docs.aws.amazon.com/elasticbeanstalk/
2. AWS Support: For production environments, consider AWS Business Support
3. Community forums: AWS forums, Stack Overflow

## Maintenance and Updates

### 1. Regular Updates

1. Regularly update dependencies:
   ```bash
   pip install -r requirements.txt --upgrade
   pip freeze > requirements.txt
   ```

2. Apply OS and security patches:
   - In Elastic Beanstalk, use "Configuration" → "Updates and deployments" to set up maintenance windows

### 2. Backup Strategy

1. Database:
   - Automatic backups via RDS
   - Consider supplemental manual snapshots before major changes
   
2. Application code:
   - Use version control (Git)
   - Consider AWS Backup for EC2 instances

### 3. Deploying Updates

For each new version:

```bash
git pull  # Get latest code
eb deploy  # Deploy to environment
```

For major updates with downtime, consider blue-green deployment:

```bash
eb clone reproductive-health-prod -n reproductive-health-prod-v2
# Update DNS after the new environment is ready
```

## Conclusion

This comprehensive guide covers all aspects of deploying the Reproductive Health Chatbot on AWS. By following these steps, you'll have a secure, scalable, and cost-effective deployment that meets both development and production requirements.

Remember to monitor your application regularly, keep dependencies updated, and optimize resources to maintain performance and security while controlling costs.

For more advanced configurations or tailored architectures, consider consulting with AWS Solutions Architects or DevOps professionals.