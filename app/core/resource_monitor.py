import psutil
import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
import torch
from contextlib import contextmanager

class ResourceMonitor:
    """Monitor system resources and application performance."""
    
    # Memory thresholds (in percentage)
    WARNING_MEMORY_THRESHOLD = 70.0
    CRITICAL_MEMORY_THRESHOLD = 85.0
    ROLLBACK_MEMORY_THRESHOLD = 90.0
    
    # Error rate thresholds (in percentage)
    WARNING_ERROR_RATE = 5.0
    CRITICAL_ERROR_RATE = 10.0
    HUMAN_INTERVENTION_ERROR_RATE = 15.0
    
    def __init__(self):
        """Initialize the resource monitor."""
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
        self.start_time = time.time()
        self.error_count = 0
        self.warning_count = 0
        self.request_count = 0
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
        """Get current performance metrics with threshold checks."""
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            error_rate = (self.error_count / max(1, self.request_count)) * 100
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_used_mb': memory_info.rss / (1024 * 1024),
                'memory_percent': memory_percent,
                'uptime_seconds': time.time() - self.start_time,
                'error_rate': error_rate,
                'warning_rate': (self.warning_count / max(1, self.request_count)) * 100,
                'total_requests': self.request_count,
                'total_errors': self.error_count,
                'total_warnings': self.warning_count
            }
            
            # Check thresholds and log warnings
            if memory_percent > self.ROLLBACK_MEMORY_THRESHOLD:
                self.logger.critical(f"Memory usage critical: {memory_percent}% > {self.ROLLBACK_MEMORY_THRESHOLD}%")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif memory_percent > self.CRITICAL_MEMORY_THRESHOLD:
                self.logger.error(f"Memory usage high: {memory_percent}% > {self.CRITICAL_MEMORY_THRESHOLD}%")
            elif memory_percent > self.WARNING_MEMORY_THRESHOLD:
                self.logger.warning(f"Memory usage elevated: {memory_percent}% > {self.WARNING_MEMORY_THRESHOLD}%")
                
            if error_rate > self.HUMAN_INTERVENTION_ERROR_RATE:
                self.logger.critical(f"Error rate critical: {error_rate}% > {self.HUMAN_INTERVENTION_ERROR_RATE}%")
            elif error_rate > self.CRITICAL_ERROR_RATE:
                self.logger.error(f"Error rate high: {error_rate}% > {self.CRITICAL_ERROR_RATE}%")
            elif error_rate > self.WARNING_ERROR_RATE:
                self.logger.warning(f"Error rate elevated: {error_rate}% > {self.WARNING_ERROR_RATE}%")
            
            self.last_metrics = metrics
            return metrics
            
        except Exception as e:
            error_msg = f"Error getting performance metrics: {str(e)}"
            self.logger.error(error_msg)
            self.log_error("get_performance_metrics", error_msg)
            return self.last_metrics or {
                'cpu_percent': 0,
                'memory_used_mb': 0,
                'memory_percent': 0,
                'uptime_seconds': 0,
                'error_rate': 0,
                'warning_rate': 0,
                'total_requests': 0,
                'total_errors': 0,
                'total_warnings': 0
            }
    
    def log_request(self):
        """Log a new request."""
        self.request_count += 1
    
    def log_error(self, operation: str, error_msg: str, error_type: Optional[str] = None):
        """Log a new error with context.
        
        Args:
            operation: The operation that failed
            error_msg: The error message
            error_type: Optional error classification (P0-P3)
        """
        self.error_count += 1
        error_data = {
            'context': operation,
            'message': error_msg,
            'type': error_type or 'P2',
            'timestamp': time.time()
        }
        
        # Log based on priority
        if error_type == 'P0':
            self.logger.critical(f"CRITICAL ERROR in {operation}: {error_msg}")
        elif error_type == 'P1':
            self.logger.error(f"MAJOR ERROR in {operation}: {error_msg}")
        else:
            self.logger.error(f"Error in {operation}: {error_msg}")
            
        return error_data
    
    @contextmanager
    def track_performance(self, context_name: str):
        """Context manager for performance tracking with detailed metrics.
        
        Args:
            context_name: Name of the context being monitored
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
            
            self.performance_metrics[context_name] = {
                'duration': duration,
                'memory_change_mb': memory_change,
                'start_metrics': start_metrics,
                'end_metrics': end_metrics,
                'timestamp': time.time()
            }
            
            # Log performance data
            self.logger.info(
                f"Performance metrics for {context_name}:\n"
                f"Duration: {duration:.2f}s\n"
                f"Memory change: {memory_change:.2f}MB\n"
                f"Memory usage: {end_metrics['memory_percent']:.1f}%\n"
                f"Error rate: {end_metrics['error_rate']:.1f}%")
    
    def log_warning(self, context: str, message: str):
        """Log a new warning with context.
        
        Args:
            context: The context where the warning occurred
            message: Warning message
        """
        self.warning_count += 1
        self.logger.warning(f"Warning in {context}: {message}")
    
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
