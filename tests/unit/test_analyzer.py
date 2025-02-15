import pytest
import logging
from rich.logging import RichHandler
from app.core.enhanced_analyzer import EnhancedAnalyzer
from app.core.data_manager import DataManager

# Configure rich logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)

# Test data
SAMPLE_RESUME = """
Professional Experience:
AI/ML Customer Success Manager | TechCorp | 2020-Present
- Led implementation of AI-powered customer support automation, reducing response time by 45%
- Trained and deployed ML models for customer churn prediction with 85% accuracy
- Managed automated onboarding system using NLP for 500+ enterprise clients
- Developed predictive analytics dashboard for customer health monitoring

Skills:
- Machine Learning: Model deployment, Training pipeline management, Performance monitoring
- AI Tools: Chatbots, NLP systems, Predictive analytics
- Customer Success: Health scoring, Churn prediction, Revenue forecasting
- Technical: Python, SQL, Jupyter, Git
"""

SAMPLE_JOB = """
Customer Success Manager (AI/ML Focus)
We're seeking a Customer Success Manager with strong AI/ML experience to help our enterprise clients succeed with our ML platform.

Required Skills:
- Experience with AI/ML model deployment and monitoring
- Strong understanding of NLP and chatbot systems
- Ability to analyze customer health using predictive analytics
- Experience with automated onboarding systems
- Knowledge of ML pipeline management

Responsibilities:
- Manage AI-powered customer success initiatives
- Monitor and optimize ML model performance
- Develop automated customer health tracking systems
- Lead technical onboarding for enterprise clients
"""

@pytest.fixture
def analyzer():
    data_manager = DataManager()
    return EnhancedAnalyzer(data_manager)

def test_analyze_resume(analyzer):
    logger.info("Starting resume analysis test")
    
    try:
        # Analyze resume
        result = analyzer.analyze_resume(SAMPLE_RESUME, SAMPLE_JOB)
        
        # Log results
        logger.info(f"Analysis completed successfully")
        logger.info(f"Role type: {result['role_type']}")
        
        # Verify result structure
        assert result['ats_score'] >= 0 and result['ats_score'] <= 1
        assert result['experience_score'] >= 0 and result['experience_score'] <= 1
        assert isinstance(result['skill_matches'], dict)
        assert isinstance(result['suggestions'], list)
        assert isinstance(result['role_metrics'], dict)
        
        # Log role-specific metrics
        if result['role_metrics']:
            logger.info("Role-specific metrics:")
            for category, metrics in result['role_metrics'].items():
                logger.info(f"{category}: {metrics}")
        
        # Log skill matches
        logger.info("Top skill matches:")
        skill_matches = result['skill_matches']
        if isinstance(skill_matches, dict):
            for skill, score in sorted(skill_matches.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:5]:
                logger.info(f"{skill}: {score}")
        
        # Log suggestions
        if result['suggestions']:
            logger.info("Improvement suggestions:")
            for suggestion in result['suggestions']:
                logger.info(f"- {suggestion}")
        
        assert result is not None
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Run test directly
    analyzer = EnhancedAnalyzer(DataManager())
    test_analyze_resume(analyzer)
