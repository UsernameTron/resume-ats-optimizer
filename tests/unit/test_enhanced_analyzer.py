import unittest
from pathlib import Path
import pandas as pd
import torch
import psutil
import time
import logging
from app.core.data_manager import DataManager
from app.core.enhanced_analyzer import EnhancedAnalyzer
from app.core.resource_monitor import ResourceMonitor

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TestEnhancedAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure logging for tests
        cls.logger = logging.getLogger(__name__)
        
        # Create test data directory
        cls.test_data_dir = Path("tests/test_data")
        cls.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample test data
        cls.create_test_data()
        
        try:
            # Initialize components
            cls.data_manager = DataManager(str(cls.test_data_dir))
            cls.analyzer = EnhancedAnalyzer(cls.data_manager)
            cls.monitor = ResourceMonitor()
            
            # Log device information
            cls.logger.info(f"Test running with device: {cls.analyzer.device}")
            cls.logger.info(f"MPS available: {torch.backends.mps.is_available()}")
        except Exception as e:
            cls.logger.error(f"Error in test setup: {str(e)}")
            raise

    @classmethod
    def create_test_data(cls):
        # Create sample job descriptions
        job_data = {
            'Job Title': ['Software Engineer', 'Data Scientist', 'Product Manager'],
            'Industry': ['Tech', 'Tech', 'Tech'],
            'Responsibilities': [
                'Develop scalable applications; Write clean code; Lead technical projects',
                'Build ML models; Analyze data; Present insights',
                'Define product strategy; Work with stakeholders; Drive product development'
            ],
            'Qualifications': [
                '5+ years experience; Python expertise; Cloud platforms',
                '3+ years experience; ML expertise; Statistics background',
                '4+ years experience; Product management; Agile methodologies'
            ],
            'Location': ['Seattle, WA', 'San Francisco, CA', 'New York, NY'],
            'Job Type': ['Full-Time', 'Full-Time', 'Full-Time'],
            'Salary Range': ['$120k - $180k', '$130k - $190k', '$140k - $200k']
        }
        
        df = pd.DataFrame(job_data)
        df.to_csv(cls.test_data_dir / "test_job_descriptions.csv", index=False)

    def setUp(self):
        self.sample_resume = """
        PROFESSIONAL EXPERIENCE
        Senior Software Engineer at Tech Corp (2018-Present)
        - Developed scalable cloud applications using Python and AWS
        - Led team of 5 engineers in microservices architecture implementation
        - Improved system performance by 40%

        Software Engineer at StartupCo (2015-2018)
        - Built RESTful APIs using Python and Django
        - Implemented CI/CD pipelines
        
        EDUCATION
        BS in Computer Science, University of Washington
        Seattle, WA
        
        SKILLS
        Python, AWS, Docker, Kubernetes, CI/CD, Microservices
        """
        
        self.sample_job = """
        Senior Software Engineer
        
        We're looking for a senior software engineer with:
        - 5+ years of experience in software development
        - Strong Python programming skills
        - Experience with cloud platforms (AWS/Azure/GCP)
        - Knowledge of microservices architecture
        
        Location: Seattle, WA
        Salary: $140k - $180k
        """

    def test_transformer_initialization(self):
        """Test transformer model initialization"""
        self.assertIsNotNone(self.analyzer.model)
        self.assertIsNotNone(self.analyzer.tokenizer)
        self.assertTrue(hasattr(self.analyzer, 'device'))
        
    def test_text_embedding(self):
        """Test text embedding generation"""
        with self.monitor.track_performance('test_text_embedding'):
            text = "Python developer with cloud experience"
            embedding = self.analyzer._get_text_embedding(text)
            
            self.assertIsNotNone(embedding)
            self.assertTrue(torch.is_tensor(embedding))
            self.assertEqual(embedding.device.type, self.analyzer.device.type)
            
            # Log device information for debugging
            self.logger.info(f"Embedding device: {embedding.device}")
            self.logger.info(f"Analyzer device: {self.analyzer.device}")
        
    def test_text_chunking(self):
        """Test text chunking functionality"""
        long_text = " ".join(["word"] * 1000)  # Create a long text
        chunks = self.analyzer._split_text_into_chunks(long_text, max_length=512)
        
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            tokens = self.analyzer.tokenizer.encode(chunk)
            self.assertLessEqual(len(tokens), 512)
            
    def test_basic_analysis(self):
        """Test basic resume analysis functionality"""
        try:
            result = self.analyzer.analyze_resume(
                self.sample_resume,
                self.sample_job,
                target_job="Software Engineer"
            )
            
            # Basic checks
            self.assertIsNotNone(result)
            self.assertTrue(0 <= result.ats_score <= 1)
            self.assertTrue(0 <= result.industry_match_score <= 1)
            
            # Check skill matches
            self.assertIn('Python', result.skill_matches)
            self.assertGreater(result.skill_matches['Python'], 0.5)
            
            # Check experience score
            self.assertGreaterEqual(result.experience_match_score, 0.8)  # Should match 5+ years requirement
            
        except Exception as e:
            self.fail(f"Analysis failed with error: {str(e)}")
        self.assertTrue(0 <= result.experience_match_score <= 1)
        
        # Check if key skills are matched
        self.assertIn('python', result.skill_matches)
        self.assertIn('aws', result.skill_matches)
        
        # Check location and salary matching
        self.assertTrue(result.location_match)
        self.assertTrue(result.salary_match)

    def test_experience_matching(self):
        """Test experience year extraction and matching"""
        result = self.analyzer.analyze_resume(
            self.sample_resume,
            self.sample_job,
            target_job="Software Engineer"
        )
        
        # Should detect ~7 years of experience (2015-Present)
        self.assertGreaterEqual(result.experience_match_score, 0.8)

    def test_industry_matching(self):
        """Test industry-specific matching"""
        result = self.analyzer.analyze_resume(
            self.sample_resume,
            self.sample_job,
            target_industry="Tech"
        )
        
        # Should have high industry match for tech role
        self.assertGreaterEqual(result.industry_match_score, 0.7)

    def test_performance_requirements(self):
        """Test performance metrics"""
        with self.monitor.track_performance('test_performance'):
            result = self.analyzer.analyze_resume(
                self.sample_resume,
                self.sample_job
            )
        
        # Check execution time (should be under 2.5s)
        self.assertLess(self.monitor.last_execution_time, 2.5)
        
        # Check memory usage (should be under 70%)
        self.assertLess(self.monitor.last_memory_usage, 70.0)

    def test_suggestion_generation(self):
        """Test improvement suggestions"""
        # Resume missing some key skills
        limited_resume = """
        EXPERIENCE
        Junior Developer (2020-Present)
        - Basic Python programming
        
        EDUCATION
        BS in Computer Science
        """
        
        result = self.analyzer.analyze_resume(
            limited_resume,
            self.sample_job,
            target_job="Software Engineer"
        )
        
        # Should suggest missing skills and experience
        self.assertTrue(any('experience' in sugg.lower() 
                          for sugg in result.improvement_suggestions))
        self.assertTrue(any('skills' in sugg.lower() 
                          for sugg in result.improvement_suggestions))

    def test_location_matching(self):
        """Test location matching logic"""
        # Resume with different location
        remote_resume = self.sample_resume.replace(
            "Seattle, WA", "Remote"
        )
        
        result = self.analyzer.analyze_resume(
            remote_resume,
            self.sample_job,
            target_job="Software Engineer"
        )
        
        # Should detect location mismatch
        self.assertFalse(result.location_match)
        self.assertTrue(any('location' in sugg.lower() 
                          for sugg in result.improvement_suggestions))

    def test_salary_matching(self):
        """Test salary range matching"""
        # Resume with salary expectations
        salary_resume = self.sample_resume + "\nSalary Expectations: $150k"
        
        result = self.analyzer.analyze_resume(
            salary_resume,
            self.sample_job,
            target_job="Software Engineer"
        )
        
        # Should detect salary match within range
        self.assertTrue(result.salary_match)

if __name__ == '__main__':
    unittest.main()
