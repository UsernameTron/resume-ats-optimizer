import unittest
import logging
from rich.logging import RichHandler
from datetime import datetime
import asyncio
import sys
import os
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.enhanced_analyzer import EnhancedAnalyzer

# Configure rich logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler('error_monitoring.log')
    ]
)

logger = logging.getLogger("ErrorMonitoring")

class TestErrorMonitoring(unittest.TestCase):
    async def simulate_resource_error(self):
        logger.info("Simulating resource exhaustion...")
        try:
            # Attempt to create a large list to trigger memory warning
            large_list = [i for i in range(10**7)]
            return large_list
        except Exception as e:
            logger.error(f"Resource error occurred: {str(e)}", exc_info=True)
            return None

    async def simulate_invalid_input(self):
        logger.info("Simulating invalid input processing...")
        try:
            analyzer = EnhancedAnalyzer()
            # Pass invalid input types
            result = await analyzer.analyze_resume(None)
            return result
        except Exception as e:
            logger.error(f"Invalid input error: {str(e)}", exc_info=True)
            return None

    async def simulate_async_timeout(self):
        logger.info("Simulating async operation timeout...")
        try:
            await asyncio.sleep(0.1)  # Simulate slow operation
            raise asyncio.TimeoutError("Operation timed out")
        except Exception as e:
            logger.error(f"Async timeout error: {str(e)}", exc_info=True)
            return None

    def monitor_system_health(self):
        logger.info("Monitoring system health...")
        try:
            # Check Python version
            logger.debug(f"Python version: {sys.version}")
            
            # Check memory usage
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.debug(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
            
            # Log current time
            logger.debug(f"Current time: {datetime.now().isoformat()}")
            
        except Exception as e:
            logger.error(f"Health monitoring error: {str(e)}", exc_info=True)

    async def test_error_scenarios(self):
        logger.info("=== Starting Error Monitoring Tests ===")
        
        # Monitor system health before tests
        self.monitor_system_health()
        
        # Test scenarios
        await self.simulate_resource_error()
        await self.simulate_invalid_input()
        await self.simulate_async_timeout()
        
        # Monitor system health after tests
        self.monitor_system_health()
        
        logger.info("=== Error Monitoring Tests Completed ===")

if __name__ == '__main__':
    logger.info("Starting error monitoring test suite")
    asyncio.run(TestErrorMonitoring().test_error_scenarios())
