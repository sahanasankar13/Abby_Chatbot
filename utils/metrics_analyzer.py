"""
Metrics Analyzer for the reproductive health chatbot

This module analyzes evaluation logs to generate performance metrics
for the chatbot dashboard.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

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
        self.logs = self._load_logs()
    
    def _load_logs(self):
        """Load evaluation logs from the log file"""
        try:
            if not os.path.exists(self.log_file):
                logger.warning(f"Log file not found: {self.log_file}")
                return []
                
            logs = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        # Parse timestamp string to datetime
                        if 'timestamp' in log_entry:
                            log_entry['timestamp'] = datetime.fromisoformat(log_entry['timestamp'])
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in log file: {line}")
                    except Exception as e:
                        logger.warning(f"Error processing log entry: {str(e)}")
            
            return logs
        except Exception as e:
            logger.error(f"Error loading logs: {str(e)}")
            return []
    
    def filter_logs_by_date(self, start_date=None, end_date=None):
        """
        Filter logs by date range
        
        Args:
            start_date (datetime, optional): Start date for filtering
            end_date (datetime, optional): End date for filtering
            
        Returns:
            list: Filtered log entries
        """
        if not self.logs:
            return []
            
        # Default to all logs if no dates provided
        if not start_date and not end_date:
            return self.logs
            
        # Set default end_date to now if not provided
        if not end_date:
            end_date = datetime.now()
            
        # Set default start_date to 30 days ago if not provided
        if not start_date:
            start_date = end_date - timedelta(days=30)
            
        # Filter logs by date range
        filtered_logs = [
            log for log in self.logs 
            if 'timestamp' in log and start_date <= log['timestamp'] <= end_date
        ]
        
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
            logger.warning("No log entries found for the specified date range")
            return self._get_empty_metrics()
        
        # Calculate overall metrics
        total_queries = len(filtered_logs)
        total_improved = sum(1 for log in filtered_logs 
                            if log.get('evaluation', {}).get('improved', False))
        
        # Calculate average scores
        relevance_scores = [log.get('evaluation', {}).get('relevance_score', 0) 
                           for log in filtered_logs]
        quality_scores = [log.get('evaluation', {}).get('quality_score', 0) 
                         for log in filtered_logs]
        safety_scores = [log.get('evaluation', {}).get('safety', {}).get('score', 0) 
                        for log in filtered_logs]
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        avg_safety = sum(safety_scores) / len(safety_scores) if safety_scores else 0
        
        # Calculate category distribution
        categories = {}
        for log in filtered_logs:
            question = log.get('question', '').lower()
            
            # Basic categorization based on keywords
            if any(kw in question for kw in ['policy', 'legal', 'law', 'state']):
                category = 'policy'
            elif any(kw in question for kw in ['what', 'how', 'when', 'why']):
                category = 'knowledge'
            else:
                category = 'conversational'
                
            categories[category] = categories.get(category, 0) + 1
        
        # Calculate daily metrics
        daily_metrics = self._calculate_daily_metrics(filtered_logs)
        
        # Calculate safety issues
        safety_issues = {}
        for log in filtered_logs:
            issues = log.get('evaluation', {}).get('safety', {}).get('issues', [])
            for issue in issues:
                if isinstance(issue, dict) and 'type' in issue:
                    issue_type = issue['type']
                    safety_issues[issue_type] = safety_issues.get(issue_type, 0) + 1
                elif isinstance(issue, str):
                    safety_issues[issue] = safety_issues.get(issue, 0) + 1
        
        return {
            'date_range': {
                'start': start_date.isoformat() if start_date else None,
                'end': end_date.isoformat() if end_date else datetime.now().isoformat()
            },
            'overview': {
                'total_queries': total_queries,
                'total_improved': total_improved,
                'improvement_rate': (total_improved / total_queries) * 100 if total_queries > 0 else 0
            },
            'scores': {
                'average_relevance': avg_relevance,
                'average_quality': avg_quality,
                'average_safety': avg_safety
            },
            'categories': categories,
            'safety_issues': safety_issues,
            'daily_metrics': daily_metrics
        }
        
    def _calculate_daily_metrics(self, logs):
        """
        Calculate metrics aggregated by day
        
        Args:
            logs (list): Log entries to analyze
            
        Returns:
            dict: Dictionary with dates and daily metrics
        """
        daily_data = {}
        
        for log in logs:
            timestamp = log.get('timestamp')
            if not timestamp:
                continue
                
            date_key = timestamp.strftime('%Y-%m-%d')
            
            if date_key not in daily_data:
                daily_data[date_key] = {
                    'queries': 0,
                    'improved': 0,
                    'safety_issues': 0,
                    'relevance_scores': [],
                    'quality_scores': [],
                    'safety_scores': []
                }
                
            daily_data[date_key]['queries'] += 1
            
            evaluation = log.get('evaluation', {})
            if evaluation.get('improved', False):
                daily_data[date_key]['improved'] += 1
                
            safety_issues = evaluation.get('safety', {}).get('issues', [])
            if safety_issues:
                daily_data[date_key]['safety_issues'] += len(safety_issues)
                
            if 'relevance_score' in evaluation:
                daily_data[date_key]['relevance_scores'].append(evaluation['relevance_score'])
                
            if 'quality_score' in evaluation:
                daily_data[date_key]['quality_scores'].append(evaluation['quality_score'])
                
            if 'safety' in evaluation and 'score' in evaluation['safety']:
                daily_data[date_key]['safety_scores'].append(evaluation['safety']['score'])
        
        # Calculate averages for each day
        result = {}
        for date, data in daily_data.items():
            avg_relevance = sum(data['relevance_scores']) / len(data['relevance_scores']) if data['relevance_scores'] else 0
            avg_quality = sum(data['quality_scores']) / len(data['quality_scores']) if data['quality_scores'] else 0
            avg_safety = sum(data['safety_scores']) / len(data['safety_scores']) if data['safety_scores'] else 0
            
            result[date] = {
                'queries': data['queries'],
                'improved': data['improved'],
                'safety_issues': data['safety_issues'],
                'avg_relevance': avg_relevance,
                'avg_quality': avg_quality,
                'avg_safety': avg_safety
            }
            
        return result
    
    def _get_empty_metrics(self):
        """Return empty metrics structure when no data is available"""
        return {
            'date_range': {
                'start': None,
                'end': datetime.now().isoformat()
            },
            'overview': {
                'total_queries': 0,
                'total_improved': 0,
                'improvement_rate': 0
            },
            'scores': {
                'average_relevance': 0,
                'average_quality': 0,
                'average_safety': 0
            },
            'categories': {},
            'safety_issues': {},
            'daily_metrics': {}
        }