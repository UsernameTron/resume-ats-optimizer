import streamlit as st
import re
from pathlib import Path
import json
from typing import Dict, List, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Must be the first Streamlit command
st.set_page_config(page_title="ATS Resume Optimizer", layout="wide")

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import core components
from app.core.enhanced_analyzer import EnhancedAnalyzer
from app.core.resume_manager import ResumeManager
from app.core.data_manager import DataManager

def handle_exception(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions."""
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

def ensure_data_directory():
    """Ensure the data directory exists."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    return data_dir

def init_components():
    """Initialize components with proper error handling."""
    try:
        logger.info("Initializing components...")
        data_manager = DataManager()
        analyzer = EnhancedAnalyzer(data_manager)
        resume_manager = ResumeManager()
        logger.info("Components initialized successfully")
        return data_manager, analyzer, resume_manager
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}", exc_info=True)
        raise

def analyze_resume(analyzer: EnhancedAnalyzer, resume_text: str, job_description: str):
    """Analyze resume with proper error handling."""
    try:
        logger.info("Starting resume analysis")
        logger.debug("Resume text length: %d, Job description length: %d", 
                  len(resume_text), len(job_description))
        
        result = analyzer.analyze_resume(resume_text, job_description)
        logger.info("Resume analysis completed successfully")
        logger.debug("Analysis result: %s", result)
        return result
    except Exception as e:
        logger.error(f"Error in resume analysis: {str(e)}", exc_info=True)
        raise

def optimize_resume(analyzer: EnhancedAnalyzer, resume_text: str, job_description: str):
    """Optimize resume with proper error handling."""
    try:
        logger.info("Starting resume optimization")
        result = analyzer.optimize_resume(resume_text, job_description)
        logger.info("Resume optimization completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in resume optimization: {str(e)}", exc_info=True)
        raise

def generate_report(result):
    """Generate a detailed report from analysis results."""
    report = []
    report.append("ATS Resume Analysis Report")
    report.append("-" * 30)
    report.append(f"\nATS Score: {result['ats_score']:.1f}%\n")
    
    report.append("\nSkill Matches:")
    for skill, match in result["skill_matches"].items():
        report.append(f"- {skill}: {'✓' if match else '✗'}")
    
    report.append("\nOptimization Suggestions:")
    for suggestion in result["suggestions"]:
        report.append(f"- {suggestion}")
    
    return "\n".join(report)

def extract_skills(text: str) -> List[str]:
    """Extract skills from text using regex patterns."""
    # Common skill patterns
    skill_patterns = [
        r'\b(?:python|java|javascript|react|node\.js|sql|aws|azure)\b',
        r'\b(?:machine learning|artificial intelligence|data science|nlp)\b',
        r'\b(?:project management|agile|scrum|leadership)\b',
        r'\b(?:customer experience|cx|customer success|cs)\b'
    ]
    
    skills = set()
    for pattern in skill_patterns:
        matches = re.finditer(pattern, text.lower())
        skills.update(match.group() for match in matches)
    
    return list(skills)

def calculate_match_score(resume: str, job_description: str) -> Dict[str, Any]:
    """Calculate the match score between resume and job description."""
    resume_skills = extract_skills(resume)
    job_skills = extract_skills(job_description)
    
    matched_skills = [skill for skill in resume_skills if skill in job_skills]
    missing_skills = [skill for skill in job_skills if skill not in resume_skills]
    
    match_score = len(matched_skills) / len(job_skills) if job_skills else 0
    
    return {
        'ats_score': match_score * 100,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'keyword_density': len(matched_skills) / (len(resume.split()) / 100) if resume else 0
    }

def main():
    """Main application function."""
    st.title("ATS Resume Optimizer")
    
    # Resume Input
    st.header("Your Resume")
    resume = st.text_area(
        "Paste your resume",
        height=300,
        placeholder="Paste your resume here...",
        key="resume_input"
    )
    
    # Job Description Input
    st.header("Job Description")
    job_description = st.text_area(
        "Paste the job description",
        height=300,
        placeholder="Paste the job description here...",
        key="job_desc_input"
    )
    
    # Analysis Button
    analyze_button = st.button("Analyze Resume", type="primary", key="analyze_button")
    
    if analyze_button:
        if not resume or not job_description:
            st.warning("Please provide both a resume and job description")
            return
            
        try:
            with st.spinner("Analyzing your resume..."):
- Certifications: PMP, Certified Customer Experience Professional (CCXP)

PROFESSIONAL EXPERIENCE

Senior CX Manager | TechCorp Inc. | 2020-Present
- Implemented AI chatbot solution resulting in 40% improvement in response time
- Increased CSAT scores from 75% to 92% through data-driven improvements
- Led cross-functional teams in customer journey mapping initiatives
- Managed stakeholder relationships across Product, Engineering, and Support
- Developed comprehensive dashboards using Tableau for executive reporting

CX Team Lead | AI Solutions Co. | 2018-2020
- Deployed Intercom-based customer feedback system
- Improved NPS score by 25 points through systematic improvements
- Managed team of 5 CX specialists and process improvement initiatives

EDUCATION
- MBA, Customer Experience Management
- BS, Business Administration

Salary Expectation: $140,000 - $150,000"""

                    test_job = """Senior Customer Experience Manager - AI/ML Solutions

We're seeking a Senior CX Manager to lead our AI-powered customer experience initiatives. 

Required Skills:
- 5+ years of customer experience/success management
- Experience with AI chatbots and conversational AI platforms
- Strong analytics and data visualization skills using Tableau/Power BI
- Proven track record in improving CSAT and NPS scores
- Experience with Zendesk, Intercom, and Salesforce
- Project management certification
- Stakeholder management experience

Key Responsibilities:
- Lead implementation of AI-powered customer support solutions
- Monitor and optimize customer satisfaction metrics
- Develop and track customer journey maps
- Manage cross-functional CX improvement projects
- Analyze customer feedback and sentiment data
- Drive customer-centric process improvements

Location: San Francisco, CA (Remote OK)
Salary Range: $120,000 - $160,000"""

                    # Log test data details
                    logger.debug("Using test resume (length: %d) and job description (length: %d)",
                                len(test_resume), len(test_job))
                    
                # Calculate match score
                results = calculate_match_score(resume, job_description)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("ATS Match Score")
                    st.metric(
                        "Score",
                        f"{results['ats_score']:.1f}%"
                    )
                
                with col2:
                    st.subheader("Keyword Density")
                    st.metric(
                        "Density",
                        f"{results['keyword_density']:.1f}%"
                    )
                
                # Display matched skills
                if results['matched_skills']:
                    st.subheader("Matched Skills")
                    st.write(", ".join(results['matched_skills']))
                
                # Display missing skills
                if results['missing_skills']:
                    st.subheader("Missing Skills")
                    st.write(", ".join(results['missing_skills']))
                
                # Suggestions
                st.subheader("Optimization Suggestions")
                if results['missing_skills']:
                    st.write("• Add these missing skills to your resume (if you have them):")
                    for skill in results['missing_skills']:
                        st.write(f"  - {skill}")
                if results['ats_score'] < 50:
                    st.write("• Your resume needs significant improvement to match this job")
                elif results['ats_score'] < 75:
                    st.write("• Consider adding more relevant keywords from the job description")
                else:
                    st.write("• Your resume is well-matched to this position!")
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")on error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)
