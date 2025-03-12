#!/usr/bin/env python3
"""
Feedback Monitoring Script for AWS Deployment

This script monitors feedback statistics and sends them to CloudWatch Metrics.
It can be run as a cron job on EC2 instances to track user feedback over time.
"""

import os
import json
import logging
import boto3
from datetime import datetime, timedelta
import sys
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feedback-monitor')

# Default path to feedback file
FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'user_feedback.json')

def load_feedback_data(file_path=FEEDBACK_FILE):
    """Load feedback data from the JSON file"""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Feedback file not found at {file_path}")
            return []
            
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading feedback data: {str(e)}")
        return []

def get_feedback_stats(feedback_data, hours_ago=24):
    """
    Calculate feedback statistics for the specified time period
    
    Args:
        feedback_data (list): List of feedback items
        hours_ago (int): How many hours back to analyze
        
    Returns:
        dict: Statistics about feedback
    """
    if not feedback_data:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "positive_percentage": 0,
            "negative_percentage": 0
        }
    
    # Calculate the timestamp for hours_ago
    cutoff_time = time.time() - (hours_ago * 3600)
    
    # Filter feedback data for the time period
    recent_feedback = [f for f in feedback_data if f.get('timestamp', 0) >= cutoff_time]
    
    total = len(recent_feedback)
    if total == 0:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "positive_percentage": 0,
            "negative_percentage": 0
        }
        
    positive = sum(1 for f in recent_feedback if f.get('rating', 0) > 0)
    negative = sum(1 for f in recent_feedback if f.get('rating', 0) < 0)
    
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_percentage": round((positive / total) * 100, 2) if total > 0 else 0,
        "negative_percentage": round((negative / total) * 100, 2) if total > 0 else 0
    }

def send_metrics_to_cloudwatch(stats, namespace="ReproductiveHealthChatbot"):
    """
    Send feedback metrics to CloudWatch
    
    Args:
        stats (dict): Feedback statistics
        namespace (str): CloudWatch namespace
    """
    try:
        # Check if we're running on AWS
        region = os.environ.get('AWS_REGION')
        if not region:
            logger.info("Not running on AWS or AWS_REGION not set - skipping CloudWatch metrics")
            return
            
        # Create CloudWatch client
        cloudwatch = boto3.client('cloudwatch', region_name=region)
        
        # Prepare metrics
        metrics = [
            {
                'MetricName': 'TotalFeedback',
                'Value': stats['total'],
                'Unit': 'Count'
            },
            {
                'MetricName': 'PositiveFeedback',
                'Value': stats['positive'],
                'Unit': 'Count'
            },
            {
                'MetricName': 'NegativeFeedback',
                'Value': stats['negative'],
                'Unit': 'Count'
            },
            {
                'MetricName': 'PositiveFeedbackPercentage',
                'Value': stats['positive_percentage'],
                'Unit': 'Percent'
            }
        ]
        
        # Send metrics to CloudWatch
        cloudwatch.put_metric_data(
            Namespace=namespace,
            MetricData=metrics
        )
        
        logger.info(f"Successfully sent feedback metrics to CloudWatch: {stats}")
        
    except Exception as e:
        logger.error(f"Error sending metrics to CloudWatch: {str(e)}")

def main():
    """Main function to run the monitoring script"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Monitor feedback statistics')
    parser.add_argument('--file', help='Path to feedback JSON file', default=FEEDBACK_FILE)
    parser.add_argument('--hours', type=int, help='Hours to look back', default=24)
    parser.add_argument('--no-cloudwatch', action='store_true', help='Skip sending to CloudWatch')
    args = parser.parse_args()
    
    # Load feedback data
    feedback_data = load_feedback_data(args.file)
    
    # Calculate statistics
    stats = get_feedback_stats(feedback_data, args.hours)
    
    # Print statistics
    print(f"Feedback Statistics (last {args.hours} hours):")
    print(f"Total feedback: {stats['total']}")
    print(f"Positive feedback: {stats['positive']} ({stats['positive_percentage']}%)")
    print(f"Negative feedback: {stats['negative']} ({stats['negative_percentage']}%)")
    
    # Send to CloudWatch if requested
    if not args.no_cloudwatch:
        send_metrics_to_cloudwatch(stats)

if __name__ == "__main__":
    main()