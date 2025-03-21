# AWS Elastic Beanstalk Deployment Guide

This guide will help you deploy the Abby Chatbot to AWS Elastic Beanstalk.

## Prerequisites

1. **AWS Account Setup**
   - Create an AWS account if you don't have one
   - Install AWS CLI
   - Configure AWS credentials:
     ```bash
     aws configure
     ```

2. **Install EB CLI**
   ```bash
   # Mac
   brew install awsebcli

   # Windows (using pip)
   pip install awsebcli
   ```

3. **Required Files**
   Ensure these files are in your project root:
   - `requirements.txt`
   - `.env.example`
   - `Procfile`
   - `gunicorn.conf.py`

## Deployment Steps

### 1. Prepare Your Application

1. **Create Elastic Beanstalk Configuration**
   ```bash
   mkdir .ebextensions
   ```

   Create `.ebextensions/01_flask.config`:
   ```yaml
   option_settings:
     aws:elasticbeanstalk:container:python:
       WSGIPath: app:app
     aws:elasticbeanstalk:application:environment:
       FLASK_APP: app.py
       FLASK_ENV: production
       PYTHONPATH: "/var/app/current:$PYTHONPATH"
     aws:elasticbeanstalk:environment:proxy:staticfiles:
       /static: static
   ```

2. **Update Environment Variables**
   Create `.ebextensions/02_environment.config`:
   ```yaml
   option_settings:
     aws:elasticbeanstalk:application:environment:
       SECRET_KEY: "your-secret-key-here"
       LOG_LEVEL: "INFO"
       ENABLE_CLOUDWATCH: "true"
   ```

3. **Configure HTTPS (Optional)**
   Create `.ebextensions/03_https.config`:
   ```yaml
   Resources:
     sslSecurityGroupIngress:
       Type: AWS::EC2::SecurityGroupIngress
       Properties:
         GroupId: {"Fn::GetAtt" : ["AWSEBSecurityGroup", "GroupId"]}
         IpProtocol: tcp
         ToPort: 443
         FromPort: 443
         CidrIp: 0.0.0.0/0
   ```

### 2. Initialize Elastic Beanstalk

1. **Create Application**
   ```bash
   eb init -p python-3.11 abby-chatbot
   ```

2. **Create Environment**
   ```bash
   # Development/Testing (Single instance)
   eb create abby-chatbot-dev --instance-type t2.micro --single

   # Production (Load balanced)
   eb create abby-chatbot-prod --instance-type t2.small -i 2 --elb-type application
   ```

### 3. Configure Environment Variables

1. **Set Environment Variables**
   ```bash
   eb setenv \
     OPENAI_API_KEY=your-openai-api-key \
     SECRET_KEY=your-secret-key \
     FLASK_ENV=production
   ```

2. **Verify Configuration**
   ```bash
   eb printenv
   ```

### 4. Deploy Your Application

1. **Deploy**
   ```bash
   eb deploy
   ```

2. **Open Application**
   ```bash
   eb open
   ```

## Monitoring and Maintenance

### 1. View Logs
```bash
# View recent logs
eb logs

# View specific log file
eb logs --log-group /aws/elasticbeanstalk/abby-chatbot-dev/var/log/app.log
```

### 2. Monitor Health
```bash
# Check environment health
eb health

# Check status
eb status
```

### 3. Scale Your Application
```bash
# Modify environment
eb config

# Scale instances
eb scale 3
```

## Cost Management

1. **Monitor Costs**
   - Set up AWS Budget alerts
   - Use AWS Cost Explorer to track spending
   - Consider using AWS Free Tier eligible resources

2. **Optimize Resources**
   ```bash
   # Scale down when not in use
   eb scale 1

   # Terminate environment when not needed
   eb terminate abby-chatbot-dev
   ```

## Security Best Practices

1. **SSL/TLS Configuration**
   - Configure SSL certificate in AWS Certificate Manager
   - Update load balancer listener to use HTTPS

2. **Security Groups**
   - Restrict inbound traffic to necessary ports
   - Use VPC for enhanced network security

3. **IAM Roles**
   - Use least privilege principle
   - Create separate roles for development and production

## Troubleshooting

### Common Issues

1. **Deployment Fails**
   ```bash
   # Check deployment logs
   eb logs -g /aws/elasticbeanstalk/abby-chatbot-dev/var/log/eb-activity.log
   ```

2. **Health Check Fails**
   - Verify application is running on port 5000
   - Check gunicorn configuration
   - Review security group settings

3. **Environment Variables Missing**
   ```bash
   # Verify environment variables
   eb printenv
   ```

### Quick Fixes

1. **Application Not Responding**
   ```bash
   # Rebuild environment
   eb rebuild
   ```

2. **Instance Issues**
   ```bash
   # Restart application server
   eb restart
   ```

3. **Configuration Problems**
   ```bash
   # Update configuration
   eb config update
   ```

## Cleanup

When you're done with your environment:

1. **Terminate Environment**
   ```bash
   eb terminate abby-chatbot-dev
   ```

2. **Delete Application**
   ```bash
   eb delete
   ```

## Additional Resources

- [AWS Elastic Beanstalk Documentation](https://docs.aws.amazon.com/elasticbeanstalk/)
- [Python on Elastic Beanstalk](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-apps.html)
- [AWS Free Tier](https://aws.amazon.com/free/)
- [AWS Pricing Calculator](https://calculator.aws/) 