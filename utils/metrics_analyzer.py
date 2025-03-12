"""
Metrics Analyzer for the reproductive health chatbot

This module analyzes evaluation logs to generate performance metrics
for the chatbot dashboard.
"""

import json
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class MetricsAnalyzer:
    """Analyzes chatbot evaluation logs for performance metrics"""
    
    def __init__(self, log_file="evaluation_logs.json"):
        """
        Initialize the metrics analyzer
        
        Args:
            log_file (str): Path to the evaluation logs file
        """
        self.log_file = log_file
        self.logs = []
        self._load_logs()
    
    def _load_logs(self):
        """Load evaluation logs from the log file"""
        try:
            with open(self.log_file, 'r') as f:
                self.logs = []
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        self.logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Loaded {len(self.logs)} log entries from {self.log_file}")
        except Exception as e:
            logger.error(f"Error loading logs: {str(e)}")
            self.logs = []
    
    def filter_logs_by_date(self, start_date=None, end_date=None):
        """
        Filter logs by date range
        
        Args:
            start_date (datetime, optional): Start date for filtering
            end_date (datetime, optional): End date for filtering
            
        Returns:
            list: Filtered log entries
        """
        if not start_date and not end_date:
            return self.logs
        
        filtered_logs = []
        for log in self.logs:
            try:
                log_date = datetime.fromisoformat(log.get('timestamp', ''))
                
                if start_date and log_date < start_date:
                    continue
                if end_date and log_date > end_date:
                    continue
                    
                filtered_logs.append(log)
            except (ValueError, TypeError):
                continue
                
        return filtered_logs
    
    def get_metrics(self, start_date=None, end_date=None) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics from the evaluation logs
        
        Args:
            start_date (datetime, optional): Start date for filtering
            end_date (datetime, optional): End date for filtering
            
        Returns:
            dict: Dictionary with calculated metrics
        """
        filtered_logs = self.filter_logs_by_date(start_date, end_date)
        
        if not filtered_logs:
            return self._get_empty_metrics()
        
        # Basic metrics
        total_count = len(filtered_logs)
        
        # Safety metrics
        safe_count = sum(1 for log in filtered_logs 
                        if log.get('evaluation', {}).get('safety_check', {}).get('is_safe', True))
        safety_rate = safe_count / total_count if total_count > 0 else 0
        
        # Source validation metrics
        valid_source_count = sum(1 for log in filtered_logs 
                               if log.get('evaluation', {}).get('source_validation', {}).get('is_valid', True))
        source_validity_rate = valid_source_count / total_count if total_count > 0 else 0
        
        # Score metrics
        scores = [log.get('evaluation', {}).get('score', 0) for log in filtered_logs]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Improvement metrics
        improved_count = 0
        for log in filtered_logs:
            evaluation = log.get('evaluation', {})
            # Check if improved response differs from original response
            improved_response = evaluation.get('improved_response', '')
            original_response = log.get('response', '')
            if improved_response and improved_response != original_response:
                improved_count += 1
        
        improvement_rate = improved_count / total_count if total_count > 0 else 0
        
        # Issue analysis
        all_issues = []
        for log in filtered_logs:
            issues = log.get('evaluation', {}).get('issues', [])
            # Issues might be a list of strings or a list of dicts with a 'concern' field
            for issue in issues:
                if isinstance(issue, dict) and 'concern' in issue:
                    all_issues.append(issue['concern'])
                elif isinstance(issue, str):
                    all_issues.append(issue)
        
        issue_counter = Counter(all_issues)
        top_issues = [{"issue": issue, "count": count} 
                      for issue, count in issue_counter.most_common(5)]
        
        # Quality component metrics (may not be present in all logs)
        relevance_scores = []
        accuracy_scores = []
        completeness_scores = []
        clarity_scores = []
        empathy_scores = []
        
        for log in filtered_logs:
            metrics = log.get('evaluation', {}).get('metrics', {})
            if metrics:
                if 'relevance' in metrics:
                    relevance_scores.append(metrics['relevance'] * 10)  # Scale to 0-10
                if 'accuracy' in metrics:
                    accuracy_scores.append(metrics.get('accuracy', 0) * 10)
                if 'completeness' in metrics:
                    completeness_scores.append(metrics.get('completeness', 0) * 10)
                if 'clarity' in metrics:
                    clarity_scores.append(metrics.get('clarity', 0) * 10)
                if 'positivity' in metrics:
                    empathy_scores.append(metrics.get('positivity', 0) * 10)
        
        # Calculate daily metrics for the time series chart
        daily_metrics = self._calculate_daily_metrics(filtered_logs)
        
        return {
            'total_count': total_count,
            'safe_count': safe_count,
            'safety_rate': safety_rate,
            'valid_source_count': valid_source_count,
            'source_validity_rate': source_validity_rate,
            'avg_score': avg_score,
            'improved_count': improved_count,
            'improvement_rate': improvement_rate,
            'top_issues': top_issues,
            'avg_relevance': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            'avg_accuracy': sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 7.5,
            'avg_completeness': sum(completeness_scores) / len(completeness_scores) if completeness_scores else 7.5,
            'avg_clarity': sum(clarity_scores) / len(clarity_scores) if clarity_scores else 7.5,
            'avg_empathy': sum(empathy_scores) / len(empathy_scores) if empathy_scores else 7.5,
            'dates': daily_metrics['dates'],
            'daily_scores': daily_metrics['scores'],
            'daily_safety': daily_metrics['safety_rates']
        }
    
    def _calculate_daily_metrics(self, logs):
        """
        Calculate metrics aggregated by day
        
        Args:
            logs (list): Log entries to analyze
            
        Returns:
            dict: Dictionary with dates and daily metrics
        """
        daily_data = defaultdict(lambda: {'scores': [], 'safety': []})
        
        for log in logs:
            try:
                log_date = datetime.fromisoformat(log.get('timestamp', '')).date().isoformat()
                score = log.get('evaluation', {}).get('score', 0)
                is_safe = log.get('evaluation', {}).get('safety_check', {}).get('is_safe', True)
                
                daily_data[log_date]['scores'].append(score)
                daily_data[log_date]['safety'].append(1 if is_safe else 0)
            except (ValueError, TypeError):
                continue
        
        # Calculate daily averages
        dates = sorted(daily_data.keys())
        daily_scores = []
        daily_safety_rates = []
        
        for date in dates:
            scores = daily_data[date]['scores']
            safety_values = daily_data[date]['safety']
            
            avg_score = sum(scores) / len(scores) if scores else 0
            safety_rate = sum(safety_values) / len(safety_values) if safety_values else 0
            
            daily_scores.append(avg_score)
            daily_safety_rates.append(safety_rate * 10)  # Scale to 0-10 for chart
        
        return {
            'dates': dates,
            'scores': daily_scores,
            'safety_rates': daily_safety_rates
        }
    
    def _get_empty_metrics(self):
        """Return empty metrics structure when no data is available"""
        return {
            'total_count': 0,
            'safe_count': 0,
            'safety_rate': 0,
            'valid_source_count': 0,
            'source_validity_rate': 0,
            'avg_score': 0,
            'improved_count': 0,
            'improvement_rate': 0,
            'top_issues': [],
            'avg_relevance': 0,
            'avg_accuracy': 0,
            'avg_completeness': 0,
            'avg_clarity': 0,
            'avg_empathy': 0,
            'dates': [],
            'daily_scores': [],
            'daily_safety': []
        }