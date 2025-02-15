import unittest
import logging
from rich.logging import RichHandler
import sys
import os
from pathlib import Path
import time
import re
from typing import Dict, Any, List

# Add parent directory to path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

# Configure rich logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("EnhancedAnalyzerTest")

class MockDataManager:
    def __init__(self):
        self.master_resume = ""
        self.job_descriptions = {}

class SimpleAnalyzer:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.master_resume = ""
        self.logger = logging.getLogger(__name__)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Split text into words and clean
        words = re.findall(r'\w+', text.lower())
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return list(set(keywords))

    def _calculate_keyword_density(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword density percentage."""
        if not text or not keywords:
            return 0.0
        
        text_lower = text.lower()
        total_words = len(re.findall(r'\w+', text_lower))
        if total_words == 0:
            return 0.0
        
        keyword_count = sum(text_lower.count(keyword.lower()) for keyword in keywords)
        return (keyword_count / total_words) * 100

    def _calculate_ats_score(self, text: str, keywords: List[str]) -> float:
        """Calculate ATS match score."""
        if not text or not keywords:
            return 0.0
        
        text_lower = text.lower()
        matched_keywords = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return matched_keywords / len(keywords) if keywords else 0.0

    def _optimize_resume(self, text: str, keywords: List[str]) -> str:
        """Basic resume optimization."""
        if not text or not keywords:
            return text
        
        optimized = text
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword.lower() not in text_lower:
                # Add missing keyword in a relevant section
                if 'SKILLS' in optimized:
                    skills_index = optimized.index('SKILLS')
                    optimized = optimized[:skills_index] + f"- {keyword}\n" + optimized[skills_index:]
        
        return optimized

    def analyze_resume(self, job_description: str) -> Dict[str, Any]:
        """Analyze resume against job description."""
        try:
            # Extract keywords
            keywords = self._extract_keywords(job_description)
            
            # Calculate original metrics
            original_ats = self._calculate_ats_score(self.master_resume, keywords)
            original_density = self._calculate_keyword_density(self.master_resume, keywords)
            
            # Optimize resume
            optimized = self._optimize_resume(self.master_resume, keywords)
            
            # Calculate new metrics
            new_ats = self._calculate_ats_score(optimized, keywords)
            new_density = self._calculate_keyword_density(optimized, keywords)
            
            return {
                'original_ats': original_ats,
                'original_density': original_density,
                'new_ats': new_ats,
                'new_density': new_density,
                'optimized_resume': optimized
            }
            
        except Exception as e:
            self.logger.error(f"Error in resume analysis: {str(e)}", exc_info=True)
            raise

class TestSimpleAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize analyzer and test data"""
        logger.info("Setting up test environment...")
        cls.data_manager = MockDataManager()
        cls.analyzer = SimpleAnalyzer(cls.data_manager)
        
        # Test data
        cls.test_resume = """
        PROFESSIONAL EXPERIENCE
        
        Senior Customer Success Manager | TechCorp Inc. | 2020-Present
        - Led strategic initiatives resulting in 95% customer retention
        - Managed portfolio of 50+ enterprise clients worth $10M ARR
        - Implemented AI-driven customer health scoring system
        - Developed automated onboarding process reducing time-to-value by 40%
        
        SKILLS
        - Customer Success: Gainsight, Totango, Salesforce
        - Analytics: Tableau, PowerBI, SQL
        - Technical: Python, API Integration, Data Analysis
        """
        
        cls.test_job = """
        Senior Customer Success Manager
        
        Requirements:
        - 5+ years experience in Customer Success
        - Experience with CS tools (Gainsight, Salesforce)
        - Strong analytical and problem-solving skills
        
        Location: San Francisco, CA
        """
        
        logger.info("Test environment setup complete")

    def setUp(self):
        """Reset state before each test"""
        logger.info("\n=== Starting new test ===")
        self.start_time = time.time()
        self.analyzer.master_resume = self.test_resume

    def tearDown(self):
        """Log test completion time"""
        duration = time.time() - self.start_time
        logger.info(f"Test completed in {duration:.2f} seconds")

    def test_analyze_resume(self):
        """Test the main resume analysis functionality"""
        logger.info("Testing analyze_resume method...")
        
        try:
            # Perform analysis
            logger.debug("Starting resume analysis...")
            result = self.analyzer.analyze_resume(self.test_job)
            
            # Validate result structure
            logger.debug("Validating result structure...")
            required_keys = {'original_ats', 'original_density', 'new_ats', 'new_density', 'optimized_resume'}
            self.assertEqual(set(result.keys()), required_keys)
            
            # Validate score ranges
            logger.debug("Validating score ranges...")
            self.assertGreaterEqual(result['original_ats'], 0)
            self.assertLessEqual(result['original_ats'], 1)
            self.assertGreaterEqual(result['new_ats'], 0)
            self.assertLessEqual(result['new_ats'], 1)
            
            # Validate density calculations
            logger.debug("Validating keyword density...")
            self.assertGreaterEqual(result['original_density'], 0)
            self.assertGreaterEqual(result['new_density'], 0)
            
            # Validate optimized resume
            logger.debug("Validating optimized resume...")
            self.assertIsInstance(result['optimized_resume'], str)
            self.assertGreater(len(result['optimized_resume']), 0)
            
            logger.info("Resume analysis test completed successfully")
            logger.debug(f"Analysis results: {result}")
            
        except Exception as e:
            logger.error(f"Error during resume analysis test: {str(e)}", exc_info=True)
            raise

    def test_keyword_extraction(self):
        """Test keyword extraction functionality"""
        logger.info("Testing keyword extraction...")
        
        try:
            keywords = self.analyzer._extract_keywords(self.test_job)
            
            # Validate keywords
            self.assertIsInstance(keywords, list)
            self.assertGreater(len(keywords), 0)
            
            # Check for expected keywords
            expected_keywords = {'gainsight', 'salesforce', 'customer', 'success', 'analytical'}
            found_keywords = set(keywords)
            self.assertTrue(any(keyword in found_keywords for keyword in expected_keywords))
            
            logger.info("Keyword extraction test completed successfully")
            logger.debug(f"Extracted keywords: {keywords}")
            
        except Exception as e:
            logger.error(f"Error during keyword extraction test: {str(e)}", exc_info=True)
            raise

if __name__ == '__main__':
    unittest.main()
