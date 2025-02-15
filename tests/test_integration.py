import logging
import sys
import os
from pathlib import Path
import pytest
from rich.logging import RichHandler

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from app.core.enhanced_analyzer import EnhancedAnalyzer
from app.core.data_manager import DataManager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

@pytest.fixture
def analyzer():
    """Create an EnhancedAnalyzer instance for testing."""
    data_manager = DataManager()
    return EnhancedAnalyzer(data_manager)

@pytest.fixture
def test_data():
    """Load test resume and job description."""
    test_data_dir = Path(project_root) / "test_data"
    
    with open(test_data_dir / "cx_resume.txt", "r") as f:
        resume_text = f.read()
    
    with open(test_data_dir / "cx_job_description.txt", "r") as f:
        job_description = f.read()
        
    return {
        "resume_text": resume_text,
        "job_description": job_description
    }

def test_full_analysis_flow(analyzer, test_data):
    """Test the complete analysis flow with CX test data."""
    logger.info("Starting full analysis flow test")
    
    try:
        # Run analysis
        result = analyzer.analyze_resume(
            resume_text=test_data["resume_text"],
            job_description=test_data["job_description"]
        )
        
        # Verify result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        
        # Check required keys
        required_keys = {
            'ats_score', 'keyword_density', 'skill_matches',
            'experience_score', 'suggestions', 'role_type',
            'role_metrics'
        }
        assert all(key in result for key in required_keys), \
            f"Missing required keys. Expected {required_keys}, got {result.keys()}"
        
        # Verify role detection
        assert result['role_type'] == 'cx', \
            f"Expected role_type 'cx', got {result['role_type']}"
        
        # Verify score ranges
        assert 0 <= result['ats_score'] <= 1, \
            f"ATS score {result['ats_score']} not in range [0,1]"
        assert 0 <= result['experience_score'] <= 1, \
            f"Experience score {result['experience_score']} not in range [0,1]"
        
        # Verify role metrics
        assert result['role_metrics'] is not None, "Role metrics should be present for CX role"
        assert isinstance(result['role_metrics'], dict), "Role metrics should be a dictionary"
        
        # Verify suggestions
        assert isinstance(result['suggestions'], list), "Suggestions should be a list"
        assert len(result['suggestions']) > 0, "Should have at least one suggestion"
        
        # Log detailed results for manual review
        logger.info("Analysis Results:")
        for key, value in result.items():
            logger.info(f"{key}: {value}")
            
        logger.info("Full analysis flow test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

def test_error_handling(analyzer):
    """Test error handling with invalid inputs."""
    logger.info("Starting error handling test")
    
    # Test empty inputs
    with pytest.raises(ValueError):
        analyzer.analyze_resume("", "job description")
    
    with pytest.raises(ValueError):
        analyzer.analyze_resume("resume text", "")
    
    # Test invalid input types
    with pytest.raises(ValueError):
        analyzer.analyze_resume(None, "job description")
    
    with pytest.raises(ValueError):
        analyzer.analyze_resume("resume text", None)
        
    logger.info("Error handling test completed successfully")
