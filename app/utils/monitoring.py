import psutil
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    memory_usage: float  # Percentage
    cpu_usage: float    # Percentage
    execution_time: float  # Seconds
    error_rate: float   # Percentage

class ResourceMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
        self.total_requests = 0
        self.failed_requests = 0
        
        # Thresholds from global configuration
        self.MEMORY_WARNING = 70.0    # 70% (33.6GB)
        self.MEMORY_CRITICAL = 85.0   # 85% (40.8GB)
        self.MEMORY_ROLLBACK = 90.0   # 90% (43.2GB)
        
        self.ERROR_WARNING = 5.0      # 5% failure
        self.ERROR_CRITICAL = 10.0    # 10% failure
        self.ERROR_INTERVENTION = 15.0 # 15% failure

    @contextmanager
    def track_performance(self):
        """Context manager to track performance metrics during execution."""
        start_time = time.time()
        start_memory = self.process.memory_percent()
        
        try:
            yield
            self.total_requests += 1
        except Exception as e:
            self.failed_requests += 1
            self.total_requests += 1
            raise e
        finally:
            end_time = time.time()
            end_memory = self.process.memory_percent()
            
            metrics = PerformanceMetrics(
                memory_usage=end_memory,
                cpu_usage=self.process.cpu_percent(),
                execution_time=end_time - start_time,
                error_rate=self._calculate_error_rate()
            )
            
            self._check_thresholds(metrics)

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100

    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check if any metrics exceed defined thresholds."""
        # Memory checks
        if metrics.memory_usage >= self.MEMORY_ROLLBACK:
            self.logger.critical(f"CRITICAL: Memory usage ({metrics.memory_usage:.1f}%) exceeded rollback threshold")
            raise MemoryError("System memory critical - rollback required")
        elif metrics.memory_usage >= self.MEMORY_CRITICAL:
            self.logger.error(f"Memory usage critical: {metrics.memory_usage:.1f}%")
        elif metrics.memory_usage >= self.MEMORY_WARNING:
            self.logger.warning(f"Memory usage warning: {metrics.memory_usage:.1f}%")

        # Error rate checks
        if metrics.error_rate >= self.ERROR_INTERVENTION:
            self.logger.critical(f"CRITICAL: Error rate ({metrics.error_rate:.1f}%) requires human intervention")
        elif metrics.error_rate >= self.ERROR_CRITICAL:
            self.logger.error(f"Error rate critical: {metrics.error_rate:.1f}%")
        elif metrics.error_rate >= self.ERROR_WARNING:
            self.logger.warning(f"Error rate warning: {metrics.error_rate:.1f}%")

        # Performance check
        if metrics.execution_time > 2.5:  # 2.5 seconds threshold
            self.logger.warning(f"Execution time warning: {metrics.execution_time:.2f}s")
