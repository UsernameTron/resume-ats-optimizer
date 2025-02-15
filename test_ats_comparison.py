from app.core.enhanced_analyzer import EnhancedAnalyzer
from app.core.data_manager import DataManager, JobRequirements
import torch
import torch.nn.functional as F

# Test data that previously gave 70.4%
test_resume = """
Senior Customer Success Manager
10+ years experience in enterprise SaaS
Skills: Customer relationship management, Salesforce, Gainsight
Led customer success initiatives for Fortune 500 clients
Reduced churn by 25% through proactive engagement
Managed portfolio of $5M+ ARR
Certifications: CSM, CCSM
"""

test_job = """
Senior Enterprise Customer Success Manager
Required Skills:
- 8+ years in enterprise SaaS customer success
- Proven track record managing large accounts
- Experience with Salesforce, Gainsight
- Strong relationship management skills
- Revenue growth and churn reduction
- Enterprise customer success methodology

Responsibilities:
- Manage portfolio of enterprise accounts
- Drive customer satisfaction and growth
- Develop success strategies
- Monitor customer health
- Lead QBRs and executive meetings

Required:
- Bachelor's degree
- Enterprise SaaS experience
- CSM certification
"""

def test_analyzer():
    print("\nRunning ATS Analysis...")
    print("-" * 50)
    
    data_manager = DataManager()
    analyzer = EnhancedAnalyzer(data_manager)
    
    # First calculate title match directly
    title_match = analyzer._calculate_title_match(test_resume, "Senior Enterprise Customer Success Manager")
    print(f"Direct Title Match Score: {title_match:.1%}")
    
    # Now run full analysis
    result = analyzer.analyze_resume(test_resume, test_job)
    
    print(f"\nOverall ATS Score: {result.ats_score:.1%}")
    print("\nBreakdown of Scores:")
    print("-" * 50)
    
    print("\nSkill Matches:")
    for skill, score in result.skill_matches.items():
        print(f"- {skill}: {score:.1%}")
        
    print(f"\nExperience Score: {result.experience_match_score:.1%}")
    print(f"Industry Score: {result.industry_match_score:.1%}")
    
    if result.role_specific_metrics:
        print("\nRole-Specific Metrics:")
        for metric, value in result.role_specific_metrics.items():
            print(f"- {metric}: {value}")
            
    print("\nBoolean Checks:")
    print(f"Location Match: {result.location_match}")
    print(f"Salary Match: {result.salary_match}")
    
    print("\nMissing Critical Skills:")
    for skill in result.missing_critical_skills:
        print(f"- {skill}")

if __name__ == "__main__":
    test_analyzer()
