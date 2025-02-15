import sys
import os
from pathlib import Path
import logging
from rich.logging import RichHandler
from typing import Dict, Set, List
import json

# Add parent directory to path to import core modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.core.data_manager import DataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("cs_cx_validation")

def create_test_job_description() -> str:
    """Create a realistic CS/CX job description"""
    return """
    Senior Customer Success Manager - Enterprise SaaS
    
    About the Role:
    We're seeking an experienced Senior Customer Success Manager to join our Enterprise team. You'll be responsible for managing our largest enterprise accounts, ensuring customer satisfaction, and driving product adoption and revenue growth.
    
    Key Responsibilities:
    - Own and manage a portfolio of enterprise customers with ARR >$500K
    - Drive customer satisfaction, product adoption, and revenue retention
    - Conduct quarterly business reviews (QBRs) and strategic planning sessions
    - Monitor and analyze customer health metrics (NPS, CSAT, CES)
    - Collaborate with cross-functional teams to resolve customer issues
    - Identify and execute upsell/cross-sell opportunities
    - Develop and maintain customer success playbooks
    
    Required Skills & Experience:
    - 5+ years of enterprise customer success experience in SaaS
    - Proven track record of managing enterprise accounts and driving revenue growth
    - Expert in CS tools: Salesforce, Gainsight, or similar platforms
    - Strong data analysis and reporting skills using SQL, Excel, and BI tools
    - Experience with customer lifecycle management and churn prevention
    - Excellent project management and stakeholder management abilities
    - Strong presentation and communication skills
    - Experience with implementation and onboarding processes
    
    Technical Skills:
    - Proficient in Salesforce CRM and Gainsight
    - SQL and data analysis capabilities
    - Experience with BI tools (Tableau, Looker, or Power BI)
    - Understanding of API integrations and basic technical concepts
    
    Metrics & KPIs:
    - Track record of maintaining >95% retention rate
    - Experience managing NPS, CSAT, and CES metrics
    - History of driving upsell/cross-sell revenue growth
    """

def create_test_resume() -> str:
    """Create a matching candidate resume"""
    return """
    PROFESSIONAL SUMMARY
    Results-driven Senior Customer Success Manager with 7 years of experience in enterprise SaaS. Proven track record of driving customer satisfaction, retention, and revenue growth through strategic account management and data-driven decision making.
    
    SKILLS
    - Customer Success: Enterprise account management, customer lifecycle management, churn prevention
    - Tools: Salesforce, Gainsight, Tableau, SQL, Excel
    - Metrics: NPS, CSAT, CES, retention rate optimization, revenue growth
    - Technical: SQL, API integrations, BI tools, data analysis
    - Soft Skills: Strategic planning, stakeholder management, presentation skills
    
    PROFESSIONAL EXPERIENCE
    
    Senior Customer Success Manager | CloudTech Solutions
    2020 - Present
    - Managed portfolio of 15 enterprise accounts with total ARR of $8M
    - Achieved 98% retention rate and 120% net revenue retention
    - Implemented new QBR process resulting in 25% increase in upsell opportunities
    - Led customer health monitoring using Gainsight, maintaining average NPS of 65
    - Developed and executed account expansion strategies generating $2M in additional ARR
    
    Customer Success Manager | SaaS Innovations
    2017 - 2020
    - Managed 25 mid-market accounts through full customer lifecycle
    - Increased portfolio retention rate from 85% to 96%
    - Created customer success playbooks for onboarding and QBRs
    - Utilized Salesforce and SQL for data-driven account management
    - Maintained CSAT score of 9.2/10 across portfolio
    
    TECHNICAL SKILLS
    - Platforms: Salesforce, Gainsight, Zendesk, Jira
    - Analysis: SQL, Excel, Tableau, Power BI
    - Tools: Confluence, Slack, MS Office Suite
    
    CERTIFICATIONS
    - Certified Customer Success Manager (CCSM)
    - Salesforce Certified Administrator
    - Gainsight Fundamentals Certification
    """

def calculate_match_accuracy(job_skills: Set[str], resume_skills: Set[str]) -> float:
    """Calculate the match accuracy between job requirements and resume skills"""
    if not job_skills:
        return 0.0
    
    matched_skills = job_skills.intersection(resume_skills)
    return len(matched_skills) / len(job_skills) * 100

def validate_skill_extraction() -> Dict:
    """Validate skill extraction and matching accuracy"""
    data_manager = DataManager()
    
    # Create test content
    job_description = create_test_job_description()
    resume = create_test_resume()
    
    # Extract skills
    job_skills = data_manager._extract_skills(job_description)
    resume_skills = data_manager._extract_skills(resume)
    
    # Calculate accuracy
    accuracy = calculate_match_accuracy(job_skills, resume_skills)
    
    # Prepare detailed results
    results = {
        "accuracy": accuracy,
        "job_skills": sorted(list(job_skills)),
        "resume_skills": sorted(list(resume_skills)),
        "matched_skills": sorted(list(job_skills.intersection(resume_skills))),
        "missing_skills": sorted(list(job_skills - resume_skills)),
        "additional_skills": sorted(list(resume_skills - job_skills))
    }
    
    return results

def main():
    """Main validation function"""
    logger.info("Starting CS/CX skill extraction validation...")
    
    try:
        results = validate_skill_extraction()
        
        logger.info(f"\nValidation Results:")
        logger.info(f"Accuracy: {results['accuracy']:.2f}%")
        logger.info(f"\nJob Required Skills: {len(results['job_skills'])}")
        logger.info(f"Resume Skills: {len(results['resume_skills'])}")
        logger.info(f"Matched Skills: {len(results['matched_skills'])}")
        
        # Save detailed results
        output_file = Path(__file__).parent / "validation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nDetailed results saved to: {output_file}")
        
        # Validate against threshold
        if results['accuracy'] >= 90.0:
            logger.info("\n✅ Validation PASSED: Accuracy meets or exceeds 90% threshold")
        else:
            logger.error("\n❌ Validation FAILED: Accuracy below 90% threshold")
            logger.error("Critical: System requires improvements before deployment")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
