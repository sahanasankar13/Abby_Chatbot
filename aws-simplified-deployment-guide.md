# Simplified AWS Deployment Guide for School Projects

This guide provides easy-to-follow instructions for deploying the Reproductive Health Chatbot on AWS with a limited budget (under $250). It's designed for students who are new to AWS.

## Step 1: Create an AWS Account

1. Go to [AWS Free Tier](https://aws.amazon.com/free/)
2. Click "Create a Free Account"
3. Follow the signup process
4. **Important**: Set up billing alerts to avoid unexpected charges
   - In AWS Console, go to Billing Dashboard
   - Click "Budgets" → "Create budget"
   - Create a budget of $200 (leaves $50 buffer) with email alerts at 50%, 80%, and 100%

## Step 2: Install Required Tools (on your computer)

1. Install AWS CLI
   - [Windows](https://awscli.amazonaws.com/AWSCLIV2.msi): Download and run the installer
   - Mac: Run `brew install awscli` or download from AWS
   - Linux: Run `sudo apt-get install awscli` or similar for your distribution

2. Configure AWS CLI
   ```bash
   aws configure
   ```
   You'll need to enter:
   - AWS Access Key ID (from your AWS account security credentials)
   - AWS Secret Access Key
   - Default region (choose us-east-1 or nearest region)
   - Default output format (json)

## Step 3: Prepare Your Application

1. Make sure your application code is ready and working locally.

2. Set up environment variables in a `.env` file (for local testing only):
   ```
   OPENAI_API_KEY=your-openai-key
   ABORTION_POLICY_API_KEY=your-policy-api-key
   SESSION_SECRET=your-secret-key
   FLASK_ENV=development
   ```

3. Create a new file called `.ebignore` with:
   ```
   .git
   .gitignore
   .env
   __pycache__/
   *.pyc
   *.pyo
   *.pyd
   venv/
   ```

## Step 4: Deploy Using AWS Elastic Beanstalk (Cheapest Option)

Elastic Beanstalk is perfect for school projects because:
- It's easy to use (no complex setup)
- Minimal AWS knowledge required
- Free tier eligible (t2.micro instances are free for 12 months)
- Auto-scaling can be turned off to control costs

### Installation

1. Install the Elastic Beanstalk CLI:
   ```bash
   pip install awsebcli
   ```

### Deployment

1. Create an application:
   ```bash
   eb init -p python-3.11 reproductive-health-chatbot
   ```
   - When asked to use CodeCommit, choose "No"
   - When asked to set up SSH, choose "Yes" if you want to be able to connect directly

2. Create the cheapest possible environment:
   ```bash
   eb create reproductive-health-chatbot-env --instance-type t2.micro --single
   ```
   
   The `--single` flag creates a single-instance environment without a load balancer, which significantly reduces costs.

3. Set your environment variables:
   ```bash
   eb setenv OPENAI_API_KEY=your-openai-key ABORTION_POLICY_API_KEY=your-policy-api-key SESSION_SECRET=your-secret-key FLASK_ENV=production
   ```

4. Wait for deployment to complete, then open your application:
   ```bash
   eb open
   ```

## Step 5: Monitoring Costs and Usage

1. Set up daily cost monitoring:
   - In AWS Console, go to "Billing Dashboard"
   - Check "Cost Explorer" daily

2. Check your instance health:
   ```bash
   eb health
   ```

3. Check logs when needed:
   ```bash
   eb logs
   ```

## Step 6: Making Updates

Whenever you update your code:

1. Commit your changes locally
2. Deploy updated code:
   ```bash
   eb deploy
   ```

## Step 7: Shutting Down When Not in Use (Save Money!)

When you're not actively working on or demonstrating your project:

1. Stop your environment (will save most costs but keep configuration):
   ```bash
   eb stop
   ```

2. To restart it later:
   ```bash
   eb start
   ```

## Cost-Saving Tips for School Projects

1. **Use t2.micro or t3.micro instances** (free tier eligible)
2. **Turn off your environment** when not using it (biggest money saver!)
3. **Use single instance mode** without a load balancer
4. **Check your bill regularly** to avoid surprises
5. **Delete your environment** completely when the project is over:
   ```bash
   eb terminate reproductive-health-chatbot-env
   ```

## Estimated Costs (Using Free Tier)

If you're a new AWS customer (within 12 months of signing up):
- EC2 t2.micro instance: **Free** for 750 hours/month (more than enough)
- Data transfer: **Free** for 15GB/month outbound
- Elastic Beanstalk service: **Free**

If you're beyond the free tier or using larger instances:
- t2.micro 24/7 operation: ~$8-15/month depending on region
- t3.micro 24/7 operation: ~$8-15/month depending on region
- **Only running during development/demo**: < $5/month

**Money-saving reminder**: Shut down your environment when not in use!

## Troubleshooting Common Issues

1. **Deployment Timeout**:
   - Try again with `eb deploy --timeout 20`

2. **Environment Creation Fails**:
   - Check `eb logs` for details
   - Ensure your application runs locally

3. **Connection Issues**:
   - Check security group settings in AWS Console

4. **Application Errors After Deployment**:
   - Run `eb logs` to check for errors
   - Make sure all requirements are in your requirements.txt file

5. **API Keys Not Working**:
   - Verify they were set correctly: `eb printenv`
   - Update if needed: `eb setenv KEY=value`

## Getting Help

- Elastic Beanstalk issues: Check [AWS Elastic Beanstalk documentation](https://docs.aws.amazon.com/elasticbeanstalk/)
- Project-specific issues: Check the project README
- AWS Free Tier info: [AWS Free Tier](https://aws.amazon.com/free/)

Remember to terminate all resources when your school project is complete to avoid any ongoing charges!