import unittest
import torch
from app.core.analyzer import ATSAnalyzer
from app.utils.monitoring import ResourceMonitor

class TestATSAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ATSAnalyzer()
        self.sample_resume = """
        PROFESSIONAL EXPERIENCE
        Software Engineer at Tech Corp
        - Developed scalable applications
        - Implemented ML models
        
        EDUCATION
        BS in Computer Science
        
        SKILLS
        Python, Machine Learning, Data Analysis
        """
        
        self.sample_job = """
        Looking for a Software Engineer with:
        - Python programming
        - Machine learning experience
        - Data analysis skills
        """

    def test_basic_analysis(self):
        result = self.analyzer.analyze_resume(self.sample_resume, self.sample_job)
        
        # Basic checks
        self.assertIsNotNone(result)
        self.assertTrue(0 <= result.ats_score <= 1)
        self.assertTrue(0 <= result.keyword_density <= 100)
        
        # Check if key skills are matched
        self.assertIn('python', result.matched_keywords)
        self.assertIn('machine learning', ' '.join(result.matched_keywords))
        
        # Check section scores
        self.assertIn('experience', result.section_scores)
        self.assertIn('education', result.section_scores)
        self.assertIn('skills', result.section_scores)

    def test_performance_requirements(self):
        monitor = ResourceMonitor()
        
        with monitor.track_performance():
            result = self.analyzer.analyze_resume(self.sample_resume, self.sample_job)
        
        # Check execution time
        self.assertLess(monitor._execution_time, 2.5, 
                       "Analysis took longer than 2.5 seconds")
        
        # Check memory usage
        memory_usage = monitor.process.memory_percent()
        self.assertLess(memory_usage, 70.0, 
                       "Memory usage exceeded warning threshold")

    def test_hardware_acceleration(self):
        if torch.backends.mps.is_available():
            self.assertEqual(self.analyzer.device.type, 'mps',
                           "MPS acceleration not enabled when available")
        else:
            self.assertEqual(self.analyzer.device.type, 'cpu',
                           "Fallback to CPU not working")
