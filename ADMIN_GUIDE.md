
# Reproductive Health Chatbot: Administrator Guide

This comprehensive guide provides detailed instructions for administrators of the Reproductive Health Chatbot system. It covers system architecture, admin dashboard access, monitoring tools, metrics analysis, and maintenance procedures.

## Table of Contents

1. [System Overview](#system-overview)
2. [Admin Access](#admin-access)
3. [Dashboard Navigation](#dashboard-navigation)
4. [Performance Monitoring](#performance-monitoring)
5. [User Feedback Analysis](#user-feedback-analysis)
6. [Content Management](#content-management)
7. [System Maintenance](#system-maintenance)
8. [Troubleshooting](#troubleshooting)
9. [Security Considerations](#security-considerations)

## System Overview

The Reproductive Health Chatbot is an AI-powered conversational platform with these key components:

### Architecture Components

- **BERT-RAG System**: Retrieves information from 550+ curated Q&A pairs
- **GPT Integration**: Enhances responses with natural language processing
- **Policy API**: Provides up-to-date abortion policy information by state
- **Evaluation System**: Monitors response quality, accuracy, and safety
- **Admin Dashboard**: Visualizes performance metrics and user feedback

### Key Features

- Natural language understanding of reproductive health questions
- State-specific policy information delivery
- Privacy protection with PII detection and redaction
- Response quality evaluation and improvement
- Comprehensive metrics tracking and visualization

## Admin Access

### Login Credentials

Admin access is secured with username/password authentication.

**Access URL**: `/login`

**Default Admin Credentials**:

| Username | Initial Password | Access Level |
|----------|-----------------|--------------|
| admin    | admin           | Admin        |
| nadya    | nadya           | Admin        |
| sahana   | sahana          | Admin        |
| nicole   | nicole          | Admin        |
| chloe    | chloe           | Admin        |

**Note**: For security reasons, change your password after initial login.

### User Management

Currently, user management must be done manually by editing the `users.json` file. 

**Adding a new admin**:
1. Access the server file system
2. Navigate to the root directory
3. Open `users.json`
4. Add a new user entry with password hash
5. Save the file and restart the application

## Dashboard Navigation

The admin dashboard is your central hub for monitoring system performance, user feedback, and system metrics.

### Main Dashboard URL

Access the dashboard at: `/admin/metrics`

### Dashboard Sections

1. **Overview**: Summary of key metrics and system status
2. **Performance**: System response times and resource usage
3. **Quality Metrics**: BLEU, ROUGE, BERTScore measurements
4. **RAG Metrics**: Faithfulness, context precision, context recall
5. **User Feedback**: Analysis of thumbs up/down ratings and comments
6. **Category Distribution**: Question types (policy, knowledge, conversational)
7. **Daily Metrics**: Performance trends over time

### Dashboard Controls

- **Date Range Selector**: Filter metrics by custom time periods
- **Session ID Filter**: View metrics for specific user sessions
- **Export Options**: Download metrics in CSV or JSON formats
- **Refresh Button**: Update metrics with latest data

## Performance Monitoring

### Key Performance Metrics

The system tracks these critical performance indicators:

#### Response Quality
- **Average Relevance Score**: How well responses address questions
- **Average Quality Score**: Overall response coherence and completeness
- **Safety Score**: Freedom from harmful or misleading content

#### System Performance
- **Average Inference Time**: Response generation speed (milliseconds)
- **Token Usage**: Average tokens per response
- **Memory Usage**: RAM consumption during response generation

#### Advanced NLP Metrics
- **BLEU Score**: n-gram precision between references and responses
- **ROUGE Score**: Recall-oriented metrics for response evaluation
- **BERTScore**: Semantic similarity using BERT embeddings

#### RAG-Specific Metrics
- **Faithfulness**: How accurately responses reflect source information
- **Context Precision**: Relevance of retrieved context to queries
- **Context Recall**: Coverage of relevant information for queries

### Monitoring Best Practices

1. **Daily Monitoring**: Check the dashboard at least once daily
2. **Metric Thresholds**: Watch for scores falling below these thresholds:
   - Relevance Score: < 7.0
   - Quality Score: < 7.5
   - Safety Score: < 9.0
   - Faithfulness: < 0.85
3. **Performance Alerts**: Investigate if inference times exceed 5000ms
4. **Trend Analysis**: Monitor metrics over time to identify degradation

## User Feedback Analysis

### Feedback Collection

The system collects user feedback in these forms:
- **Thumbs Up/Down**: Binary rating for each response
- **Optional Comments**: Free-text feedback about responses

### Feedback Analysis Tools

1. **Feedback Summary**: Overall positive/negative ratio
2. **Comment Analysis**: Categorized user comments
3. **Feedback Trends**: Changes in feedback patterns over time
4. **Session Analysis**: Correlation between feedback and session length

### Using Feedback Effectively

1. **Identify Problem Areas**: Look for patterns in negative feedback
2. **Correlation Analysis**: Connect negative feedback to specific topics
3. **Quality Improvement**: Use insights to enhance knowledge base
4. **Content Gaps**: Identify missing information based on feedback

## Content Management

### Knowledge Base Management

The knowledge base consists of 550+ curated Q&A pairs about reproductive health topics.

**Updating Knowledge Base**:
1. Edit the CSV file in `attached_assets/Planned Parenthood Data - Sahana (1).csv`
2. Maintain the existing format: question, answer, source
3. Restart the application to rebuild the BERT-RAG index

### Policy Information

Policy information comes from the external Abortion Policy API.

**API Configuration**:
- API key is stored as an environment variable
- No direct editing of policy data is required
- Policy information updates automatically when the API is updated

### Response Templates

The system uses various prompt templates for different response types.

**Customizing Templates**:
1. Review existing templates in `chatbot/friendly_bot.py`
2. Modify templates to adjust tone, style, or formatting
3. Test changes thoroughly before deployment

## System Maintenance

### Regular Maintenance Tasks

1. **Log Rotation**: Setup log rotation to prevent disk space issues
2. **Feedback Database**: Backup `user_feedback.json` periodically
3. **Metrics Database**: Backup `evaluation_logs.json` periodically
4. **API Key Rotation**: Update API keys every 90 days for security

### Updating the System

When deploying updates:

1. **Test Environment**: Always test changes in a staging environment first
2. **Backup Data**: Create backups of all data and configuration files
3. **Update Process**:
   - Stop the application
   - Deploy new files
   - Verify environment variables
   - Start the application
4. **Verification**: Check logs for startup errors and test basic functionality

### Monitoring Scripts

The system includes several monitoring scripts in the `scripts/` directory:

- `monitor_feedback.py`: Analyzes user feedback
- `calculate_ragas_metrics.py`: Calculates RAG quality metrics
- `generate_sample_metrics.py`: Creates sample metrics for testing

To run a monitoring script:
```bash
python scripts/monitor_feedback.py --hours 24
```

## Troubleshooting

### Common Issues and Solutions

#### Application Won't Start
- Check environment variables are set correctly
- Verify OpenAI API key is valid
- Check for errors in the application logs

#### Slow Response Times
- Monitor system resources (CPU, memory)
- Check OpenAI API latency
- Verify FAISS index is loading correctly

#### Poor Response Quality
- Review RAG metrics for degradation
- Check knowledge base for gaps
- Verify policy API is responding correctly

#### Authentication Failures
- Check `users.json` file integrity
- Verify login route is accessible
- Check session management configuration

### Log Files

Important log files to check during troubleshooting:

- `server.log`: Main application log
- `evaluation_logs.json`: Detailed response evaluation data
- Flask development server output

## Security Considerations

### Data Protection

The system includes several privacy and security features:

1. **PII Detection**: Automatically detects and redacts personal identifiable information
2. **Secure Authentication**: Password-protected admin access
3. **API Key Security**: API keys stored as environment variables
4. **User Data Privacy**: No personal data stored with user questions

### Security Best Practices

1. **Change Default Passwords**: Update all default admin passwords
2. **Regular Updates**: Keep dependencies updated with security patches
3. **Access Control**: Limit admin access to necessary personnel
4. **Monitoring**: Watch for unusual activity patterns in logs
5. **Data Backups**: Regularly backup system data with encryption

## AWS Deployment Specifics

If deployed on AWS, additional monitoring tools are available:

1. **CloudWatch**: Metrics are sent to AWS CloudWatch if configured
2. **CloudWatch Logs**: Application logs can be streamed to CloudWatch
3. **SNS Alerts**: Set up alerts for critical metrics or errors
4. **Elastic Beanstalk Console**: Monitor application health and status

To enable CloudWatch integration:
1. Ensure the AWS_REGION environment variable is set
2. Attach an IAM role with CloudWatch permissions to your deployment
3. Metrics will be sent to the namespace "ReproductiveHealthChatbot"

## Conclusion

This administrator guide covers the essential aspects of managing and maintaining the Reproductive Health Chatbot. By following these guidelines, you can ensure optimal performance, high-quality responses, and a secure environment for users seeking reproductive health information.

For additional technical details, refer to the main README.md file and source code documentation. If you encounter issues not covered in this guide, please contact the development team for support.
