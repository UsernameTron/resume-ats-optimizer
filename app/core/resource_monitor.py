import psutil
import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
import torch


class ResourceMonitor:
    """Monitor system resources and application performance.
    
    This class provides comprehensive monitoring of system resources and application
    performance metrics, with specific focus on CX and CS role-specific tracking.
    
    Thresholds:
        Memory Usage:
            WARNING: >70% (33.6GB) - Alert on elevated memory usage
            CRITICAL: >85% (40.8GB) - Take preventive action
            ROLLBACK: >90% (43.2GB) - Trigger automatic rollback
            
        Error Rates:
            WARNING: >5% - Monitor closely
            CRITICAL: >10% - Investigate immediately
            HUMAN_INTERVENTION: >15% - Require manual intervention
    """
    
    # Memory thresholds (in percentage)
    WARNING_MEMORY_THRESHOLD = 70.0  # Alert at 70% memory usage
    CRITICAL_MEMORY_THRESHOLD = 85.0  # Critical at 85% memory usage
    ROLLBACK_MEMORY_THRESHOLD = 90.0  # Rollback at 90% memory usage
    
    # Error rate thresholds (in percentage)
    WARNING_ERROR_RATE = 5.0  # Alert at 5% error rate
    CRITICAL_ERROR_RATE = 10.0  # Critical at 10% error rate
    HUMAN_INTERVENTION_ERROR_RATE = 15.0  # Manual intervention at 15% error rate
    
    def __init__(self):
        """Initialize the resource monitor with role-specific tracking."""
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
        self.start_time = time.time()
        
        # General counters
        self.error_count = 0
        self.warning_count = 0
        self.request_count = 0
        
        # Role-specific metrics
        self.role_metrics = {
            'cx': {
                'requests': 0,
                'errors': 0,
                'warnings': 0,
                'processing_times': [],
                'memory_usage': [],
                'pattern_matches': 0,
                'skill_extractions': 0
            },
            'cs': {
                'requests': 0,
                'errors': 0,
                'warnings': 0,
                'processing_times': [],
                'memory_usage': [],
                'pattern_matches': 0,
                'skill_extractions': 0
            }
        }
        
        self.last_metrics = {}
        self.performance_metrics = {}
    
    def get_memory_usage(self) -> float:
        """Get current memory usage as a percentage."""
        try:
            return self.process.memory_percent()
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {str(e)}")
            return 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics with role-specific monitoring."""
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            error_rate = (self.error_count / max(1, self.request_count)) * 100
            
            # Calculate role-specific metrics
            role_specific = {}
            for role in ['cx', 'cs']:
                role_data = self.role_metrics[role]
                requests = role_data['requests']
                if requests > 0:
                    avg_processing_time = sum(role_data['processing_times']) / len(role_data['processing_times'])
                    avg_memory = sum(role_data['memory_usage']) / len(role_data['memory_usage'])
                    error_rate = (role_data['errors'] / requests) * 100
                    warning_rate = (role_data['warnings'] / requests) * 100
                else:
                    avg_processing_time = 0
                    avg_memory = 0
                    error_rate = 0
                    warning_rate = 0
                
                role_specific[role] = {
                    'requests': requests,
                    'errors': role_data['errors'],
                    'warnings': role_data['warnings'],
                    'avg_processing_time': avg_processing_time,
                    'avg_memory': avg_memory,
                    'error_rate': error_rate,
                    'warning_rate': warning_rate,
                    'pattern_matches': role_data['pattern_matches'],
                    'skill_extractions': role_data['skill_extractions']
                }
            
            return {
                'cpu_percent': cpu_percent,
                'memory_rss': memory_info.rss / (1024 * 1024),  # MB
                'memory_percent': memory_percent,
                'error_rate': error_rate,
                'request_count': self.request_count,
                'error_count': self.error_count,
                'warning_count': self.warning_count,
                'uptime': time.time() - self.start_time,
                'role_specific': role_specific
            }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}")
            return {}
    
    def log_request(self, role_type: Optional[str] = None) -> None:
        """Log a new request with optional role tracking.
        
        Args:
            role_type: Optional role type (cx/cs) for role-specific tracking
        """
        self.request_count += 1
        
        # Update role-specific metrics if role_type provided
        if role_type and role_type.lower() in ['cx', 'cs']:
            role = role_type.lower()
            self.role_metrics[role]['requests'] += 1
    
    def log_error(self, operation: str, error_msg: str, error_type: Optional[str] = None, role_type: Optional[str] = None) -> None:
        """Log a new error with context and role tracking.
        
        Args:
            operation: The operation that failed
            error_msg: The error message
            error_type: Optional error classification (P0-P3)
            role_type: Optional role type (cx/cs) for role-specific tracking
        """
        self.error_count += 1
        
        # Update role-specific metrics if role_type provided
        if role_type and role_type.lower() in ['cx', 'cs']:
            role = role_type.lower()
            self.role_metrics[role]['errors'] += 1
            
        # Add role context to log message
        role_context = f" [{role_type.upper()}]" if role_type else ""
        error_context = f" [{error_type}]" if error_type else ""
        self.logger.error(f"Error in {operation}{role_context}{error_context}: {error_msg}")
    
    @contextmanager
    def track_performance(self, context_name: str, role_type: Optional[str] = None):
        """Context manager for performance tracking with role-specific metrics.
        
        Args:
            context_name: Name of the operation being tracked
            role_type: Optional role type (cx/cs) for role-specific tracking
        """
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss
        start_metrics = self.get_performance_metrics()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss
            end_metrics = self.get_performance_metrics()
            
            duration = end_time - start_time
            memory_change = (end_memory - start_memory) / (1024 * 1024)  # MB
            
            # Update role-specific metrics if role_type provided
            if role_type and role_type.lower() in ['cx', 'cs']:
                role = role_type.lower()
                self.role_metrics[role]['processing_times'].append(duration * 1000)  # Convert to ms
                self.role_metrics[role]['memory_usage'].append(memory_change)
                
                # Keep only last 100 measurements to prevent memory growth
                if len(self.role_metrics[role]['processing_times']) > 100:
                    self.role_metrics[role]['processing_times'] = self.role_metrics[role]['processing_times'][-100:]
                    self.role_metrics[role]['memory_usage'] = self.role_metrics[role]['memory_usage'][-100:]
            
            # Add role context to log messages
            role_context = f" [{role_type.upper()}]" if role_type else ""
            
            self.performance_metrics[context_name] = {
                'duration': duration,
                'memory_change_mb': memory_change,
                'start_metrics': start_metrics,
                'end_metrics': end_metrics,
                'timestamp': time.time()
            }
            
            # Log performance data
            self.logger.info(
                f"Performance metrics for {context_name}{role_context}:\n"
                f"Duration: {duration:.2f}s\n"
                f"Memory change: {memory_change:.2f}MB\n"
                f"Memory usage: {end_metrics['memory_percent']:.1f}%\n"
                f"Error rate: {end_metrics['error_rate']:.1f}%")
    
    def log_pattern_match(self, role_type: str) -> None:
        """Log a pattern match for role-specific tracking.
        
        Args:
            role_type: Role type (cx/cs) for tracking pattern matches
        """
        if role_type and role_type.lower() in ['cx', 'cs']:
            role = role_type.lower()
            self.role_metrics[role]['pattern_matches'] += 1
    
    def log_skill_extraction(self, role_type: str) -> None:
        """Log a skill extraction for role-specific tracking.
        
        Args:
            role_type: Role type (cx/cs) for tracking skill extractions
        """
        if role_type and role_type.lower() in ['cx', 'cs']:
            role = role_type.lower()
            self.role_metrics[role]['skill_extractions'] += 1
    
    def log_warning(self, context: str, message: str, role_type: Optional[str] = None) -> None:
        """Log a new warning with context and role tracking.
        
        Args:
            context: The context where the warning occurred
            message: Warning message
            role_type: Optional role type (cx/cs) for role-specific tracking
        """
        self.warning_count += 1
        
        # Update role-specific metrics if role_type provided
        if role_type and role_type.lower() in ['cx', 'cs']:
            role = role_type.lower()
            self.role_metrics[role]['warnings'] += 1
            
        # Add role context to log message
        role_context = f" [{role_type.upper()}]" if role_type else ""
        self.logger.warning(f"Warning in {context}{role_context}: {message}")
    
    def clear_role_metrics(self, role_type: Optional[str] = None) -> None:
        """Clear role-specific metrics.
        
        Args:
            role_type: Optional role type (cx/cs). If None, clear all roles.
        """
        roles_to_clear = ['cx', 'cs'] if role_type is None else [role_type.lower()]
        
        for role in roles_to_clear:
            if role in self.role_metrics:
                self.role_metrics[role] = {
                    'requests': 0,
                    'errors': 0,
                    'warnings': 0,
                    'processing_times': [],
                    'memory_usage': [],
                    'pattern_matches': 0,
                    'skill_extractions': 0
                }
    
    def check_resource_limits(self) -> Dict[str, Any]:
        """Check if resource usage is within acceptable limits.
        
        Returns:
            Dict containing status and details of resource checks
        """
        metrics = self.get_performance_metrics()
        status = {
            'healthy': True,
            'warnings': [],
            'critical': []
        }
        
        # Memory checks
        if metrics['memory_percent'] > self.ROLLBACK_MEMORY_THRESHOLD:
            status['critical'].append(f"Memory usage critical: {metrics['memory_percent']}%")
            status['healthy'] = False
        elif metrics['memory_percent'] > self.CRITICAL_MEMORY_THRESHOLD:
            status['critical'].append(f"Memory usage high: {metrics['memory_percent']}%")
            status['healthy'] = False
        elif metrics['memory_percent'] > self.WARNING_MEMORY_THRESHOLD:
            status['warnings'].append(f"Memory usage elevated: {metrics['memory_percent']}%")
        
        # Error rate checks
        if metrics['error_rate'] > self.HUMAN_INTERVENTION_ERROR_RATE:
            status['critical'].append(f"Error rate critical: {metrics['error_rate']}%")
            status['healthy'] = False
        elif metrics['error_rate'] > self.CRITICAL_ERROR_RATE:
            status['critical'].append(f"Error rate high: {metrics['error_rate']}%")
            status['healthy'] = False
        elif metrics['error_rate'] > self.WARNING_ERROR_RATE:
            status['warnings'].append(f"Error rate elevated: {metrics['error_rate']}%")
        
        # CPU checks
        if metrics['cpu_percent'] > 90.0:
            status['warnings'].append(f"CPU usage high: {metrics['cpu_percent']}%")
        
        return status
