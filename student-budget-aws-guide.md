# Student's AWS Deployment Guide: Budget-Friendly Edition

This guide is specifically designed for students who need to deploy their Reproductive Health Chatbot for a class project or demonstration with minimal cost. We'll focus on the most cost-effective approach while ensuring the application works properly.

## Cost-Conscious Deployment Strategy

**Expected monthly cost: $0-$10** (If you're in AWS Free Tier and follow shutdown procedures)

## Step 1: Prerequisites

1. Create an AWS account (if you don't have one already)
   - Go to [AWS Free Tier](https://aws.amazon.com/free/)
   - Sign up for the 12-month free tier
   - You'll need a credit card, but we'll ensure minimal to no charges

2. **IMPORTANT**: Set up billing alerts immediately
   - In AWS Console → Billing Dashboard → Budgets
   - Create a budget with a $10 threshold
   - Add email alerts at 50%, 80%, and 100%

3. Install required tools
   - AWS CLI: [Install Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
   - Elastic Beanstalk CLI: `pip install awsebcli`
   - Git: [Install Guide](https://github.com/git-guides/install-git)

4. Gather your API keys:
   - OpenAI API Key
   - Abortion Policy API Key

## Step 2: Prepare Your Application

1. Make sure your application runs locally with the correct settings in `main.py`:
   ```python
   if __name__ == "__main__":
       app.run(host="0.0.0.0", port=5000)
   ```

2. Ensure your `requirements.txt` includes all necessary packages:
   ```bash
   pip freeze > requirements.txt
   ```

3. Create a `Procfile` in the project root (if it doesn't exist):
   ```
   web: gunicorn --bind 0.0.0.0:5000 main:app
   ```

4. Create an `.ebignore` file to control what gets uploaded:
   ```
   .git/
   .gitignore
   .env
   __pycache__/
   *.pyc
   *.pyo
   *.pyd
   .Python
   venv/
   ```

## Step 3: Deploy Using Elastic Beanstalk (Cheapest Option)

1. Configure AWS credentials
   - Sign into AWS Console
   - Go to IAM → Users → Add User
     - Name: `eb-deployer`
     - Access type: Programmatic access
     - Permissions: AWSElasticBeanstalkFullAccess
   - Save the Access Key ID and Secret Access Key

2. Run AWS configure:
   ```bash
   aws configure
   ```
   Enter your access key, secret key, region (choose us-east-1 for best free tier options), and output format (json)

3. Initialize your EB application:
   ```bash
   eb init -p python-3.11 reproductive-health-chatbot
   ```
   - Choose your region (same as configured above)
   - When asked to use CodeCommit, choose "No"
   - When asked to set up SSH, choose "Yes" if you want to be able to connect directly

4. Create a SINGLE INSTANCE environment (critical for cost savings):
   ```bash
   eb create reproductive-health-env --instance-type t2.micro --single
   ```
   The `--single` flag creates an environment without a load balancer, saving ~$20/month.

5. Set environment variables for your API keys:
   ```bash
   eb setenv OPENAI_API_KEY=your-openai-key ABORTION_POLICY_API_KEY=your-policy-api-key SESSION_SECRET=your-session-secret FLASK_ENV=production
   ```

6. Open your application:
   ```bash
   eb open
   ```

## Step 4: Cost Management (MOST IMPORTANT STEP!)

### Save Money When Not Using Your Application

When you're not actively demoing or working on your application:

1. STOP your environment to avoid charges:
   ```bash
   eb stop
   ```
   This preserves your configuration but shuts down the EC2 instance to avoid hourly charges.

2. When you need your application again:
   ```bash
   eb start
   ```
   Wait 5-10 minutes for it to be available again.

### Check Your Costs Daily

1. In AWS Console, go to Billing Dashboard → Bills
2. Set a calendar reminder to check this at least weekly
3. If costs exceed $5, investigate immediately

## Step 5: Common Issues and Solutions

### Deployment Fails

Check the logs:
```bash
eb logs
```

Common issues:
- Missing packages in requirements.txt
- Errors in application code
- Python version mismatch

### Application Loads But Doesn't Work

1. Check environment variables:
   ```bash
   eb printenv
   ```

2. Check application logs:
   ```bash
   eb logs
   ```

3. SSH into the instance for direct troubleshooting:
   ```bash
   eb ssh
   ```

## Step 6: Updating Your Application

When you make changes:

1. Test locally first
2. Deploy the updated code:
   ```bash
   eb deploy
   ```

## Step 7: Clean Up When Your Project Is Done

To avoid ANY future charges when your class is over:

1. Terminate your environment:
   ```bash
   eb terminate reproductive-health-env
   ```

2. Delete the application (optional):
   ```bash
   aws elasticbeanstalk delete-application --application-name reproductive-health-chatbot
   ```

3. Check your billing dashboard one final time to ensure no unexpected resources remain

## Cost Breakdown

With this approach and the AWS Free Tier:
- EC2 t2.micro: FREE for first 12 months (750 hours/month)
- EBS Storage: FREE for first 12 months (30GB)
- Data Transfer: FREE for first 12 months (15GB out)

If beyond free tier or heavy usage:
- EC2 t2.micro (if running 24/7): ~$8-10/month
- Data transfer: Minimal for demo usage
- **BUT if you stop the environment when not in use**: < $2/month

## Final Tips for Students

1. **NEVER leave your environment running when not in use**
2. Set a calendar reminder to check your AWS billing dashboard weekly
3. Try to deploy close to your demo date, not weeks in advance
4. If you accidentally incur charges, contact AWS support - they often waive first-time charges for students
5. For your final project presentation, consider starting the environment a few hours before to ensure everything is working

Remember: The most cost-effective AWS environment is one that's turned off when not being used!

## For Further Assistance

- AWS Elastic Beanstalk documentation: [AWS EB Documentation](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/Welcome.html)
- Your instructor or TA may have specific guidance for student projects
- AWS Educate: If your school participates, this provides additional free credits for students

Good luck with your project!