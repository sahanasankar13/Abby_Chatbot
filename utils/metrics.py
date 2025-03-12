"""
Metrics tracking system for the reproductive health chatbot.

This module provides functionality to track and report various metrics 
about the chatbot's performance, usage patterns, and feedback.
It is designed to work locally for development and send metrics to 
AWS CloudWatch when deployed.
"""

import os
import time
import json
import logging
import threading
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union

# Import boto3 conditionally to work in environments without AWS
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetricsTracker:
    """
    Tracks various metrics about the chatbot's usage and performance.
    
    This class provides methods to record metrics and periodically 
    flush them to different destinations (local file, CloudWatch).
    """
    
    def __init__(self, 
                 app_name: str = "ReproductiveHealthChatbot",
                 metrics_file: str = "metrics.json",
                 auto_flush_interval: int = 300, 
                 enable_cloudwatch: bool = None):
        """
        Initialize the metrics tracker
        
        Args:
            app_name (str): Application name for CloudWatch namespace
            metrics_file (str): Path to metrics file for local storage
            auto_flush_interval (int): Seconds between auto-flush to storage
            enable_cloudwatch (bool): Whether to send metrics to CloudWatch
                                     (defaults to True if AWS_REGION is set)
        """
        self.app_name = app_name
        self.metrics_file = metrics_file
        self.auto_flush_interval = auto_flush_interval
        
        # Determine if CloudWatch should be enabled based on environment
        if enable_cloudwatch is None:
            # Auto-detect if we're running on AWS
            self.enable_cloudwatch = bool(os.environ.get('AWS_REGION')) and BOTO3_AVAILABLE
        else:
            self.enable_cloudwatch = enable_cloudwatch and BOTO3_AVAILABLE
            
        # Initialize CloudWatch client if needed
        self.cloudwatch = None
        if self.enable_cloudwatch and BOTO3_AVAILABLE:
            region = os.environ.get('AWS_REGION', 'us-east-1')
            try:
                self.cloudwatch = boto3.client('cloudwatch', region_name=region)
                logger.info(f"CloudWatch metrics enabled (region: {region})")
            except Exception as e:
                logger.warning(f"Failed to initialize CloudWatch client: {str(e)}")
                self.enable_cloudwatch = False
                
        # Metrics storage 
        self._metrics_lock = threading.Lock()
        self.reset_metrics()
        
        # Set up auto-flushing if interval > 0
        if auto_flush_interval > 0:
            self._setup_auto_flush()
            
        logger.info(f"Metrics tracker initialized: cloudwatch={self.enable_cloudwatch}")
        
    def reset_metrics(self):
        """Reset all metrics to initial state"""
        with self._metrics_lock:
            # Counters for different events
            self._counters = defaultdict(int)
            
            # Timers for performance measurements
            self._timers = defaultdict(list)
            
            # API usage tracking
            self._api_calls = defaultdict(int)
            self._api_tokens = defaultdict(int)
            
            # Feedback tracking
            self._feedback = {
                "positive": 0,
                "negative": 0
            }
            
            # Store last reset time
            self._last_reset = time.time()
    
    def _setup_auto_flush(self):
        """Set up a timer to periodically flush metrics"""
        def auto_flush():
            self.flush_metrics()
            # Schedule the next flush
            timer = threading.Timer(self.auto_flush_interval, auto_flush)
            timer.daemon = True
            timer.start()
            
        # Start the first timer
        timer = threading.Timer(self.auto_flush_interval, auto_flush)
        timer.daemon = True
        timer.start()
        logger.debug(f"Auto-flush enabled (interval: {self.auto_flush_interval}s)")
    
    def increment_counter(self, metric_name: str, value: int = 1):
        """
        Increment a counter metric
        
        Args:
            metric_name (str): Name of the counter to increment
            value (int): Amount to increment by (default: 1)
        """
        with self._metrics_lock:
            self._counters[metric_name] += value
    
    def record_time(self, metric_name: str, elapsed_time: float):
        """
        Record a timing metric
        
        Args:
            metric_name (str): Name of the timing metric
            elapsed_time (float): Time in seconds
        """
        with self._metrics_lock:
            self._timers[metric_name].append(elapsed_time)
    
    def time_function(self, metric_name: str):
        """
        Decorator to time a function and record the elapsed time
        
        Args:
            metric_name (str): Name of the timing metric
            
        Returns:
            Decorated function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                self.record_time(metric_name, elapsed_time)
                return result
            return wrapper
        return decorator
    
    def record_api_call(self, api_name: str, tokens_used: int = 0):
        """
        Record an API call with optional token usage
        
        Args:
            api_name (str): Name of the API called
            tokens_used (int): Number of tokens used in this call
        """
        with self._metrics_lock:
            self._api_calls[api_name] += 1
            if tokens_used > 0:
                self._api_tokens[api_name] += tokens_used
    
    def record_feedback(self, positive: bool = True):
        """
        Record user feedback
        
        Args:
            positive (bool): Whether the feedback was positive
        """
        with self._metrics_lock:
            if positive:
                self._feedback["positive"] += 1
            else:
                self._feedback["negative"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current metrics
        
        Returns:
            Dict[str, Any]: Dictionary of all metrics
        """
        with self._metrics_lock:
            # Process timer data to get statistics
            timer_stats = {}
            for name, values in self._timers.items():
                if values:
                    timer_stats[name] = {
                        "count": len(values),
                        "total": sum(values),
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }
                else:
                    timer_stats[name] = {
                        "count": 0,
                        "total": 0,
                        "average": 0,
                        "min": 0,
                        "max": 0
                    }
            
            # Calculate feedback percentages
            total_feedback = self._feedback["positive"] + self._feedback["negative"]
            pos_pct = (self._feedback["positive"] / total_feedback * 100) if total_feedback > 0 else 0
            neg_pct = (self._feedback["negative"] / total_feedback * 100) if total_feedback > 0 else 0
            
            return {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": time.time() - self._last_reset,
                "counters": dict(self._counters),
                "timers": timer_stats,
                "api_calls": dict(self._api_calls),
                "api_tokens": dict(self._api_tokens),
                "feedback": {
                    "positive": self._feedback["positive"],
                    "negative": self._feedback["negative"],
                    "total": total_feedback,
                    "positive_percentage": round(pos_pct, 2),
                    "negative_percentage": round(neg_pct, 2)
                }
            }
    
    def flush_metrics(self, reset: bool = True) -> Dict[str, Any]:
        """
        Flush metrics to storage and optionally reset
        
        Args:
            reset (bool): Whether to reset metrics after flushing
            
        Returns:
            Dict[str, Any]: The metrics that were flushed
        """
        metrics = self.get_metrics()
        
        # Save to local file
        try:
            # Create file with empty list if it doesn't exist
            if not os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'w') as f:
                    f.write("[]")
            
            # Read existing metrics
            with open(self.metrics_file, 'r') as f:
                try:
                    existing_metrics = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in metrics file, starting new file")
                    existing_metrics = []
            
            # Append new metrics and write back
            existing_metrics.append(metrics)
            with open(self.metrics_file, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
                
            logger.debug(f"Flushed metrics to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics to file: {str(e)}")
        
        # Send to CloudWatch if enabled
        if self.enable_cloudwatch and self.cloudwatch:
            try:
                self._send_to_cloudwatch(metrics)
                logger.debug("Flushed metrics to CloudWatch")
            except Exception as e:
                logger.error(f"Error sending metrics to CloudWatch: {str(e)}")
        
        # Reset metrics if requested
        if reset:
            self.reset_metrics()
            
        return metrics
    
    def _send_to_cloudwatch(self, metrics: Dict[str, Any]):
        """
        Send metrics to CloudWatch
        
        Args:
            metrics (Dict[str, Any]): Metrics to send
        """
        if not self.cloudwatch:
            return
            
        # Prepare CloudWatch metrics
        cw_metrics = []
        
        # Add counter metrics
        for name, value in metrics["counters"].items():
            cw_metrics.append({
                'MetricName': f"Counter_{name}",
                'Value': value,
                'Unit': 'Count'
            })
        
        # Add timer metrics (average and count)
        for name, stats in metrics["timers"].items():
            if stats["count"] > 0:
                cw_metrics.append({
                    'MetricName': f"Timer_{name}_Average",
                    'Value': stats["average"],
                    'Unit': 'Seconds'
                })
                cw_metrics.append({
                    'MetricName': f"Timer_{name}_Count",
                    'Value': stats["count"],
                    'Unit': 'Count'
                })
        
        # Add API call metrics
        for name, count in metrics["api_calls"].items():
            cw_metrics.append({
                'MetricName': f"API_{name}_Calls",
                'Value': count,
                'Unit': 'Count'
            })
        
        # Add token usage metrics
        for name, tokens in metrics["api_tokens"].items():
            cw_metrics.append({
                'MetricName': f"API_{name}_Tokens",
                'Value': tokens,
                'Unit': 'Count'
            })
        
        # Add feedback metrics
        feedback = metrics["feedback"]
        cw_metrics.extend([
            {
                'MetricName': 'Feedback_Positive',
                'Value': feedback["positive"],
                'Unit': 'Count'
            },
            {
                'MetricName': 'Feedback_Negative',
                'Value': feedback["negative"],
                'Unit': 'Count'
            },
            {
                'MetricName': 'Feedback_PositivePercentage',
                'Value': feedback["positive_percentage"],
                'Unit': 'Percent'
            }
        ])
        
        # Send metrics in batches (CloudWatch limit is 20 per call)
        batch_size = 20
        for i in range(0, len(cw_metrics), batch_size):
            batch = cw_metrics[i:i+batch_size]
            self.cloudwatch.put_metric_data(
                Namespace=self.app_name,
                MetricData=batch
            )


# Global instance for easy access
metrics = MetricsTracker()


# Utility functions for easy access to the global metrics instance

def increment_counter(metric_name: str, value: int = 1):
    """Increment a counter metric"""
    metrics.increment_counter(metric_name, value)

def record_time(metric_name: str, elapsed_time: float):
    """Record a timing metric"""
    metrics.record_time(metric_name, elapsed_time)

def time_function(metric_name: str):
    """Decorator to time a function"""
    return metrics.time_function(metric_name)

def record_api_call(api_name: str, tokens_used: int = 0):
    """Record an API call with optional token usage"""
    metrics.record_api_call(api_name, tokens_used)

def record_feedback(positive: bool = True):
    """Record user feedback"""
    metrics.record_feedback(positive)

def get_metrics() -> Dict[str, Any]:
    """Get the current metrics"""
    return metrics.get_metrics()

def flush_metrics(reset: bool = True) -> Dict[str, Any]:
    """Flush metrics to storage and optionally reset"""
    return metrics.flush_metrics(reset)