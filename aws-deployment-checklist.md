# AWS Deployment Checklist for Reproductive Health Chatbot

Use this checklist to ensure a smooth deployment process for your reproductive health chatbot application. Check off each item as you complete it.

## Pre-Deployment Preparation

### Local Development
- [ ] Application runs correctly on local machine
- [ ] All required packages are in `requirements.txt`
- [ ] Environment variables are properly configured
- [ ] Database migrations work correctly
- [ ] All tests pass

### Code Preparation
- [ ] Create `Procfile` with correct configuration:
  ```
  web: gunicorn --bind 0.0.0.0:5000 --workers=2 --threads=4 main:app
  ```
- [ ] Create `.ebignore` file to exclude unnecessary files
- [ ] Ensure `main.py` has correct host/port configuration
- [ ] Remove any hardcoded localhost references
- [ ] Clean up debug print statements
- [ ] Set `debug=False` for production

### AWS Account Setup
- [ ] Create AWS account (or use existing)
- [ ] Set up billing alerts and budgets
- [ ] Create IAM user with appropriate permissions
- [ ] Configure AWS CLI with credentials
- [ ] Choose deployment region (nearest to target users)

## Database Setup

### If Using RDS for PostgreSQL
- [ ] Create DB subnet group
- [ ] Set up DB security group with appropriate rules
- [ ] Create RDS instance with PostgreSQL
- [ ] Note the endpoint, port, username, and password
- [ ] Test database connection from your local machine

### If Using SQLite (for development/testing only)
- [ ] Ensure SQLite file permissions are set correctly
- [ ] Include database path in your application configuration

## Elastic Beanstalk Deployment

### Initialize Application
- [ ] Run `eb init -p python-3.11 reproductive-health-chatbot`
- [ ] Configure SSH access if needed
- [ ] Select correct region

### Environment Creation
- [ ] For development/testing:
  ```
  eb create reproductive-health-dev --instance-type t2.micro --single
  ```
- [ ] For production:
  ```
  eb create reproductive-health-prod --instance-type t3.small --min-instances 2 --max-instances 4
  ```

### Environment Configuration
- [ ] Set environment variables:
  ```
  eb setenv OPENAI_API_KEY=your-key ABORTION_POLICY_API_KEY=your-key SESSION_SECRET=your-secret FLASK_ENV=production
  ```
- [ ] If using RDS, add the DATABASE_URL environment variable
- [ ] Configure health check path to `/api/health`
- [ ] Set up log streaming to CloudWatch
- [ ] Configure monitoring thresholds

## Security Configuration

- [ ] Review security groups to ensure minimum necessary access
- [ ] Set up HTTPS if using a custom domain:
  - [ ] Request SSL certificate in AWS Certificate Manager
  - [ ] Configure load balancer with HTTPS listener
  - [ ] Set up redirect from HTTP to HTTPS
- [ ] Move API keys to AWS Secrets Manager (for production)
- [ ] Enable AWS WAF for additional protection (production)
- [ ] Update security headers in application
- [ ] Enable HSTS if using HTTPS

## Monitoring and Logging

- [ ] Enable enhanced health monitoring
- [ ] Set up CloudWatch Logs for application logs
- [ ] Create CloudWatch Dashboard for key metrics
- [ ] Set up alarms for critical thresholds
- [ ] Configure notification methods (email, SMS)
- [ ] Enable X-Ray tracing (optional)

## Performance Tuning

- [ ] Configure appropriate instance type and count
- [ ] Adjust Gunicorn workers and threads based on instance size
- [ ] Set up auto-scaling triggers (if using load balancer)
- [ ] Configure proper timeouts
- [ ] Enable gzip compression
- [ ] Implement caching where appropriate

## Post-Deployment Verification

- [ ] Open application URL (`eb open`)
- [ ] Test basic functionality
- [ ] Verify all API integrations are working
- [ ] Check database connectivity
- [ ] Test user authentication
- [ ] Validate SSL certificate if using HTTPS
- [ ] Run stress test (optional)
- [ ] Check application logs for errors
- [ ] Verify metrics are being collected
- [ ] Test backup and restore procedures

## Cost Optimization (Critical for Students)

- [ ] Use t2.micro or t3.micro instances (free tier eligible)
- [ ] Implement environment stop/start procedures for development
- [ ] Minimize the number of resources
- [ ] Monitor usage regularly
- [ ] Use single-instance environments when possible
- [ ] Set up billing alerts at 50%, 80%, and 100% of budget

## Routine Maintenance Plan

- [ ] Schedule regular updates for dependencies
- [ ] Plan for periodic security patches
- [ ] Implement database backup strategy
- [ ] Document update procedures
- [ ] Create rollback plan for failed deployments
- [ ] Set up system health checks

## Final Documentation

- [ ] Record all configuration settings
- [ ] Document environment variables
- [ ] Create troubleshooting guide
- [ ] Document deployment process
- [ ] Update README with deployment instructions
- [ ] Document scaling procedures
- [ ] Create incident response plan

## Special Considerations for Student Projects

- [ ] Know how to stop environment when not in use
- [ ] Understand how to terminate all resources when project is complete
- [ ] Have process for regular cost monitoring
- [ ] Know how to contact AWS support if unexpected charges occur
- [ ] Understanding of estimated monthly costs

## Emergency Procedures

- [ ] Know how to roll back to previous version
- [ ] Procedure for database restore
- [ ] Steps to restart application/servers
- [ ] Understand how to terminate environment in case of security breach
- [ ] Have contact information for AWS support

Use this checklist to ensure a successful deployment. For each phase of deployment, mark items as "Completed", "In Progress", or "N/A" as appropriate for your specific project requirements.